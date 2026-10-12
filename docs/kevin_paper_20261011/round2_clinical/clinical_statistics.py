#!/usr/bin/env python3
"""Small-cohort patient-level statistics; no inference or model selection."""
from __future__ import annotations
import hashlib
import numpy as np
from scipy.stats import beta, binomtest
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score, f1_score


def exact_interval(successes, total):
    """Two-sided 95% Clopper-Pearson interval; absent denominator stays absent."""
    if not 0 <= successes <= total:
        raise ValueError('Invalid binomial counts')
    if total == 0:
        return None
    return [0.0 if successes == 0 else float(beta.ppf(.025, successes, total-successes+1)),
            1.0 if successes == total else float(beta.ppf(.975, successes+1, total-successes))]


def validate(y, probabilities):
    raw=np.asarray(y)
    if not np.isin(raw,[0,1]).all():
        raise ValueError('Only explicit binary labels may be scored')
    y=np.asarray(y, dtype=int);p=np.asarray(probabilities,dtype=float)
    if y.ndim != 1 or p.shape != (len(y),2) or not len(y):
        raise ValueError('Expected one probability vector per patient')
    if not set(y).issubset({0,1}) or not np.isfinite(p).all() or np.any(p<0) or np.any(p>1) or not np.allclose(p.sum(1),1,atol=1e-6,rtol=0):
        raise ValueError('Invalid labels/probabilities')
    return y,p


def metrics(y, probabilities):
    y,p=validate(y,probabilities);pred=(p[:,1]>=.5).astype(int)
    tp=int(((y==1)&(pred==1)).sum());tn=int(((y==0)&(pred==0)).sum())
    fn=int(((y==1)&(pred==0)).sum());fp=int(((y==0)&(pred==1)).sum())
    n=len(y);positive=int((y==1).sum());negative=n-positive
    bins=[];ece=0.
    # Fixed five-bin AD probability reliability; descriptive at tiny sample sizes.
    for lo,hi in zip(np.linspace(0,1,6)[:-1],np.linspace(0,1,6)[1:]):
        mask=(p[:,1]>=lo)&((p[:,1]<hi) if hi<1 else (p[:,1]<=hi));count=int(mask.sum())
        observed=float(y[mask].mean()) if count else None
        confidence=float(p[mask,1].mean()) if count else None
        if count:ece+=count/n*abs(observed-confidence)
        bins.append(dict(lower=float(lo),upper=float(hi),n=count,mean_p_ad=confidence,observed_ad_fraction=observed))
    both=positive>0 and negative>0
    return dict(n_patients=n,n_ad=positive,n_cn=negative,correct=tp+tn,
        confusion_cn_ad=[[tn,fp],[fn,tp]],accuracy=(tp+tn)/n,
        accuracy_exact95=exact_interval(tp+tn,n),
        sensitivity_ad=tp/positive if positive else None,sensitivity_exact95=exact_interval(tp,positive),
        specificity_cn=tn/negative if negative else None,specificity_exact95=exact_interval(tn,negative),
        precision_ad=tp/(tp+fp) if tp+fp else None,negative_predictive_value=tn/(tn+fn) if tn+fn else None,
        balanced_accuracy=float(balanced_accuracy_score(y,pred)) if both else None,
        macro_f1=float(f1_score(y,pred,labels=[0,1],average='macro',zero_division=0)),
        roc_auc=float(roc_auc_score(y,p[:,1])) if both else None,
        average_precision=float(average_precision_score(y,p[:,1])) if positive else None,
        nll=float(-np.log(np.clip(p[np.arange(n),y],1e-12,1)).mean()),
        binary_brier=float(np.mean((p[:,1]-y)**2)),
        ad_probability_ece_5_bins=float(ece),reliability_bins=bins,
        threshold=.5,calibration_fitted=False,unit='patient')


def paired_comparison(y, newer, older, identity, draws=10000):
    """Conditional paired patient bootstrap; exact McNemar; no seed pooling."""
    y,newer=validate(y,newer);_,older=validate(y,older)
    new_pred=(newer[:,1]>=.5).astype(int);old_pred=(older[:,1]>=.5).astype(int)
    new_correct=new_pred==y;old_correct=old_pred==y
    wins=int((new_correct&~old_correct).sum());losses=int((~new_correct&old_correct).sum())
    base_new=metrics(y,newer);base_old=metrics(y,older)
    fields=['accuracy','balanced_accuracy','sensitivity_ad','specificity_cn','nll','binary_brier','roc_auc']
    values={k:[] for k in fields}
    rng=np.random.default_rng(int.from_bytes(hashlib.sha256(identity.encode()).digest()[:8],'big'))
    # Stratification conditions on the observed AD/CN counts, which is explicit.
    strata=[np.flatnonzero(y==k) for k in [0,1]]
    def binary_auc(labels, scores):
        differences=scores[labels==1,None]-scores[labels==0][None,:]
        return float(np.mean((differences>0)+.5*(differences==0)))
    for _ in range(draws):
        idx=np.concatenate([rng.choice(s,len(s),replace=True) for s in strata if len(s)])
        yy=y[idx];nn=newer[idx];oo=older[idx]
        nc=(nn[:,1]>=.5)==yy;oc=(oo[:,1]>=.5)==yy
        sensitivity=float(nc[yy==1].mean()-oc[yy==1].mean()) if np.any(yy==1) else None
        specificity=float(nc[yy==0].mean()-oc[yy==0].mean()) if np.any(yy==0) else None
        sample=dict(accuracy=float(nc.mean()-oc.mean()),sensitivity_ad=sensitivity,specificity_cn=specificity,
          balanced_accuracy=(sensitivity+specificity)/2 if sensitivity is not None and specificity is not None else None,
          nll=float(np.mean(-np.log(np.clip(nn[np.arange(len(yy)),yy],1e-12,1))+np.log(np.clip(oo[np.arange(len(yy)),yy],1e-12,1)))),
          binary_brier=float(np.mean((nn[:,1]-yy)**2-(oo[:,1]-yy)**2)),
          roc_auc=binary_auc(yy,nn[:,1])-binary_auc(yy,oo[:,1]) if len(np.unique(yy))==2 else None)
        for k,v in sample.items():
            if v is not None:values[k].append(v)
    return dict(unit='patient',new_only_correct=wins,old_only_correct=losses,
        mcnemar_exact_two_sided_p=float(binomtest(wins,wins+losses,.5).pvalue) if wins+losses else 1.,
        bootstrap_draws=draws,bootstrap_scheme='Paired patient resampling stratified by observed diagnosis counts',
        differences={k:dict(point=base_new[k]-base_old[k] if base_new[k] is not None and base_old[k] is not None else None,
            conditional95=np.quantile(values[k],[.025,.975]).tolist() if values[k] else None) for k in fields},
        multiplicity_adjusted=False,interpretation='Small fixed-cohort exploratory comparison; no population-wide or clinical-superiority claim.')
