#!/usr/bin/env python3
"""Verified clinical patient report: no inference, fitting, or model selection."""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import numpy as np
import clinical_compare as c
import clinical_statistics as s
from sklearn.metrics import roc_curve,precision_recall_curve

TITLES={'harvard_fresh_external':'Fresh Harvard T1: 19 patients (9 AD, 10 CN)','besga_reused_test':'Reused Besga test: 11 patients (7 AD, 4 CN)','openneuro_reused_external':'Reused OpenNeuro: 15 patients (1 AD, 14 CN)'}
DISPLAY={**{n:f'Old clinical {n[-2:]}' for n in c.OLD},**{n:f'New seed {seed}' for n,seed in zip(c.NEW,c.SEEDS)},'round1_clinical_ensemble':'Old fixed ensemble','new_clinical_ensemble':'New fixed ensemble'}
COLORS=['#999999','#666666','#333333','#4c9bda','#36a26f','#a481c6','#c16421','#00758f']
NOTE='Patient-level research pilot; small convenience cohorts. No clinical superiority or population risk-calibration claim.'


def cohort_note(cohort):
    return 'Only one AD patient: exact sensitivity intervals remain broad; conditional bootstrap does not establish population sensitivity precision.' if cohort=='openneuro_reused_external' else NOTE


def table(path,rows):
    if not rows:return
    with path.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def assert_numeric_equal(actual,expected,context='metrics'):
    if isinstance(actual,dict):
        if not isinstance(expected,dict) or set(actual)!=set(expected):raise ValueError('Metric fields differ: '+context)
        for k,v in actual.items():assert_numeric_equal(v,expected[k],context+'.'+k)
    elif isinstance(actual,list):
        if not isinstance(expected,list) or len(actual)!=len(expected):raise ValueError('Metric array differs: '+context)
        for i,v in enumerate(actual):assert_numeric_equal(v,expected[i],context+f'[{i}]')
    elif isinstance(actual,(int,float)) and not isinstance(actual,bool):
        if not isinstance(expected,(int,float)) or not np.isfinite(expected) or not np.isclose(actual,expected,rtol=2e-6,atol=2e-7):raise ValueError('Metric value differs: '+context)
    elif actual!=expected:raise ValueError('Metric value differs: '+context)


def validate_predictions(predictions,patients,metrics):
    expected={r['group_key']:r for r in patients}
    if len(expected)!=60 or len(patients)!=60:raise ValueError('Expected exactly60 unique patient bags')
    from collections import Counter
    counts=Counter((r['evaluation_cohort'],r['label']) for r in patients)
    wanted={('harvard_fresh_external',1):9,('harvard_fresh_external',0):10,('harvard_fresh_external',-1):10,('besga_reused_test',1):7,('besga_reused_test',0):4,('openneuro_reused_external',1):1,('openneuro_reused_external',0):14,('openneuro_reused_external',-1):5}
    if counts!=wanted:raise ValueError('Patient cohort/diagnosis denominators changed')
    by={}
    for row in predictions:
        key=(row['model'],row['scenario'],row['group_key'])
        if key in by or row['model'] not in c.ORDER or row['scenario'] not in c.SCENARIOS or row['group_key'] not in expected:raise ValueError('Duplicate or unexpected prediction key')
        truth=expected[row['group_key']]
        if row['cohort']!=truth['evaluation_cohort'] or row['label']!=truth['label'] or row['diagnosis']!=truth['diagnosis'] or row['n_slices']!=9 or row['binary_evaluable']!=(truth['label']>=0):raise ValueError('Patient metadata/label mismatch')
        if {'AD':1,'CN':0,'MCI':-1}.get(row['diagnosis'])!=row['label']:raise ValueError('Diagnosis encoding mismatch')
        pp=np.asarray(row['probabilities'],dtype=float)
        if pp.shape!=(2,) or not np.isfinite(pp).all() or np.any(pp<0) or np.any(pp>1) or not np.isclose(pp.sum(),1,atol=1e-6,rtol=0) or row['predicted_ad']!=bool(pp[1]>=.5):raise ValueError('Invalid patient probability/decision')
        by[key]=pp
    keys={(model,scenario,key) for model in c.ORDER for scenario in c.SCENARIOS for key in expected}
    if set(by)!=keys or len(predictions)!=3360:raise ValueError('Incomplete3360-row probability grid')
    for scenario in c.SCENARIOS:
        for key in expected:
            for ensemble,members in [('round1_clinical_ensemble',list(c.OLD)),('new_clinical_ensemble',c.NEW)]:
                mean=np.mean([by[n,scenario,key] for n in members],axis=0)
                if not np.allclose(mean,by[ensemble,scenario,key],atol=1e-7,rtol=0):raise ValueError('Fixed uniform ensemble probabilities differ')
    metric_by={(r['model'],r['scenario'],r['cohort']):r for r in metrics}
    if len(metric_by)!=168 or len(metrics)!=168:raise ValueError('Incomplete or duplicated clinical metric grid')
    data={};recomputed=[]
    for cohort in TITLES:
        eligible=sorted([key for key,r in expected.items() if r['evaluation_cohort']==cohort and r['label']>=0]);y=np.array([expected[k]['label'] for k in eligible]);data[cohort]=dict(keys=eligible,y=y,models={})
        for model in c.ORDER:
            data[cohort]['models'][model]={}
            clean=np.array([by[model,'clean',k] for k in eligible])
            for scenario in c.SCENARIOS:
                prob=np.array([by[model,scenario,k] for k in eligible]);m=s.metrics(y,prob);m.update(mean_probability_total_variation=float(np.abs(prob-clean).sum(1).mean()/2),label_flip_fraction=float(np.mean((prob[:,1]>=.5)!=(clean[:,1]>=.5))))
                stored=metric_by.get((model,scenario,cohort))
                if stored is None:raise ValueError('Clinical metric missing')
                assert_numeric_equal(m,{k:v for k,v in stored.items() if k not in ['model','scenario','cohort']},model+'/'+scenario+'/'+cohort)
                data[cohort]['models'][model][scenario]=dict(prob=prob,metrics=m)
                recomputed.append(dict(model=model,scenario=scenario,cohort=cohort,**m))
    return data,recomputed


def read_verified(suite,manifest_path):
    source=suite/'clinical_evaluation';complete=c.read(source/'evaluation_complete.json');inventory=c.read(source/'SHA256SUMS.json');evaluation=c.read(source/'evaluation.json');freeze=c.read(suite/'clinical_evaluation_freeze.json')
    if set(inventory)!={'patient_predictions.json','metrics.json','model_state_verification.json','evaluation.json'}:raise ValueError('Clinical evaluation inventory must cover all four payloads')
    if complete.get('status')!='complete_verified' or complete['manifest_sha256']!=c.sha(source/'SHA256SUMS.json') or complete['payloads']!=len(inventory):raise ValueError('Incomplete clinical evaluation')
    for name,digest in inventory.items():
        if c.sha(c.safe(source,name))!=digest:raise ValueError('Clinical evaluation payload changed')
    if evaluation['status']!='complete' or evaluation['freeze']!=freeze or evaluation['freeze_sha256']!=c.sha(suite/'clinical_evaluation_freeze.json') or evaluation['model_order']!=c.ORDER or evaluation['scenarios']!=c.SCENARIOS:raise ValueError('Evaluation/freeze model or scenario mismatch')
    if freeze.get('status')!='frozen_before_clinical_heldout_inference' or freeze.get('heldout_outcomes_used_for_selection') is not False:raise ValueError('Clinical selection was not frozen before held-out inference')
    if freeze['source_code_sha256']!=c.code_hashes() or freeze['evaluation_manifest_sha256']!=c.sha(manifest_path) or freeze['protocol']!=c.read(c.HERE/'evaluation_protocol.json'):raise ValueError('Frozen code/protocol/manifest differs')
    if evaluation.get('fresh_binary_patients')!=19 or evaluation.get('mci_not_scored') is not True or evaluation.get('no_training_or_selection') is not True:raise ValueError('Clinical endpoint or evaluation role changed')
    states=c.read(source/'model_state_verification.json')
    if len(states)!=6 or {v['model'] for v in states}!=set(list(c.OLD)+c.NEW):raise ValueError('Incomplete model identity verification')
    for row in states:
        expected=freeze['selections'][row['model']]['checkpoint_sha256'] if row['model'] in c.NEW else freeze['old_checkpoints'][row['model']]['sha256']
        if row['checkpoint_sha256']!=expected or row['all_states_unchanged'] is not True or row['no_parameter_gradients'] is not True:raise ValueError('Checkpoint/state identity mismatch')
    manifest=c.read(manifest_path)
    if manifest.get('status')!='prepared_without_model_inference' or manifest['source_sha256']!=freeze['input_source_sha256']:raise ValueError('Manifest source identity differs')
    predictions=c.read(source/'patient_predictions.json');data,recomputed=validate_predictions(predictions,manifest['patients'],c.read(source/'metrics.json'))
    return freeze,predictions,data,recomputed


def plotting():
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'axes.spines.right':False,'axes.spines.top':False})
    return plt


def save(fig,out,stem,synthetic=False):
    if synthetic:fig.text(.5,.5,'SYNTHETIC QA',ha='center',fontsize=36,alpha=.15,rotation=20)
    for ext in ['png','pdf']:fig.savefig(out/f'{stem}.{ext}',dpi=220,facecolor='white')
    plotting().close(fig)


def cohort_figures(cohort,d,out,synthetic):
    plt=plotting();names=c.ORDER
    fig,axes=plt.subplots(1,3,figsize=(15,6.6))
    for ax,metric,ci in zip(axes,['accuracy','sensitivity_ad','specificity_cn'],['accuracy_exact95','sensitivity_exact95','specificity_exact95']):
        for i,(name,color) in enumerate(zip(names,COLORS)):
            m=d['models'][name]['clean']['metrics'];v=m[metric];lo,hi=m[ci];ax.errorbar(v*100,i,xerr=np.array([[v-lo],[hi-v]])*100,fmt='o',color=color,capsize=3,markersize=6 if 'ensemble' in name else 4)
        ax.set_yticks(range(8),[DISPLAY[n] for n in names] if ax is axes[0] else []);ax.invert_yaxis();ax.set(xlim=(-2,102),xlabel=metric.replace('_',' ')+' (%)');ax.grid(axis='x',alpha=.2)
    fig.suptitle(TITLES[cohort]+' | exact 95% binomial intervals',fontweight='bold');fig.text(.02,.025,cohort_note(cohort)+'\nSeparate sensitivity/specificity denominators; seed and patient uncertainty are distinct.',fontsize=9);fig.tight_layout(rect=(0,.10,1,.92));save(fig,out,cohort+'_exact_intervals',synthetic)
    fig,axes=plt.subplots(2,4,figsize=(13,8.3))
    for ax,name in zip(axes.flat,names):
        m=d['models'][name]['clean']['metrics'];cm=np.array(m['confusion_cn_ad']);den=cm.sum(1,keepdims=True);fraction=cm/den;ax.imshow(fraction,cmap='Blues',vmin=0,vmax=1)
        for i in range(2):
            for j in range(2):ax.text(j,i,str(cm[i,j]),ha='center',va='center',color='white' if fraction[i,j]>.55 else 'black',fontsize=15)
        ax.set_xticks([0,1],['CN','AD']);ax.set_yticks([0,1],['CN','AD']);ax.set(title=DISPLAY[name],xlabel='Predicted',ylabel='Provider diagnosis')
    fig.suptitle(TITLES[cohort]+' | patient confusion counts',fontweight='bold');fig.text(.02,.02,cohort_note(cohort),fontsize=9);fig.tight_layout(rect=(0,.07,1,.94),h_pad=3);save(fig,out,cohort+'_confusion',synthetic)
    fig,axes=plt.subplots(1,2,figsize=(12,5.6));y=d['y']
    for name,color in zip(names,COLORS):
        z=d['models'][name]['clean'];prob=z['prob'][:,1];m=z['metrics'];xx,yy,_=roc_curve(y,prob,drop_intermediate=False);axes[0].plot(xx,yy,color=color,lw=2.5 if 'ensemble' in name else 1,ls='-' if 'ensemble' in name else '--',label=DISPLAY[name]);yy,xx,_=precision_recall_curve(y,prob);axes[1].step(xx,yy,where='post',color=color,lw=2.5 if 'ensemble' in name else 1,ls='-' if 'ensemble' in name else '--')
    axes[0].plot([0,1],[0,1],ls=':',color='gray');axes[0].set(xlabel='False-positive rate',ylabel='AD sensitivity');axes[1].axhline(y.mean(),ls=':',color='gray');axes[1].set(xlabel='Recall',ylabel='AD precision')
    for ax in axes:ax.set(xlim=(0,1),ylim=(0,1.02))
    fig.suptitle(TITLES[cohort]+' | ROC and precision–recall',fontweight='bold');fig.legend(*axes[0].get_legend_handles_labels(),ncol=4,loc='lower center',bbox_to_anchor=(.5,.035),fontsize=8);fig.text(.02,.014,'Small fixed case-control samples; precision reflects sampled prevalence. All scores and uncertainty appear in analysis.json.',fontsize=8);fig.tight_layout(rect=(0,.16,1,.93));save(fig,out,cohort+'_roc_pr',synthetic)
    for kind in ['reliability','confidence']:
        fig,axes=plt.subplots(2,4,figsize=(14,7.8))
        for ax,name,color in zip(axes.flat,names,COLORS):
            z=d['models'][name]['clean'];m=z['metrics'];prob=z['prob'][:,1]
            if kind=='reliability':
                bins=[b for b in m['reliability_bins'] if b['n']];ax.plot([0,1],[0,1],ls=':',color='gray');ax.scatter([b['mean_p_ad'] for b in bins],[b['observed_ad_fraction'] for b in bins],s=[20+150*b['n']/len(y) for b in bins],color=color);ax.text(.04,.06,f"Brier {m['binary_brier']:.3f}\nNLL {m['nll']:.3f}",transform=ax.transAxes,fontsize=8);ax.set(xlim=(0,1),ylim=(0,1.03),xlabel='Mean P(AD) in fixed bin',ylabel='Observed AD fraction')
            else:
                ax.hist([prob[y==0],prob[y==1]],bins=np.linspace(0,1,11),stacked=True,color=['#7299bb','#c77959'],label=['Provider CN','Provider AD']);ax.axvline(.5,ls=':',color='black');ax.set(xlim=(0,1),xlabel='Patient AD score',ylabel='Patient count');ax.legend(fontsize=7)
            ax.set_title(DISPLAY[name],fontsize=10)
        caption='Five fixed probability bins are descriptive; sparse bins and case-control prevalence do not establish clinical-risk calibration.' if kind=='reliability' else 'Ten fixed histogram bins show patient scores by provider diagnosis; case-control score distributions do not establish clinical-risk calibration.'
        fig.suptitle(TITLES[cohort]+' | '+kind,fontweight='bold');fig.text(.02,.02,caption,fontsize=9);fig.tight_layout(rect=(0,.07,1,.94));save(fig,out,cohort+'_'+kind,synthetic)


def stress_figures(cohort,d,out,synthetic):
    plt=plotting();families=[('Intensity',['clean','intensity_0.9','intensity_1.1'],['Clean','0.9','1.1']),('Noise sigma',['clean','gaussian_noise_0.01','gaussian_noise_0.02'],['Clean','0.01','0.02']),('Blur radius',['clean','blur_0.5','blur_1.0'],['Clean','0.5','1.0'])]
    fig,axes=plt.subplots(3,3,figsize=(13.5,10.8))
    for j,(family,scenarios,ticks) in enumerate(families):
        for i,metric in enumerate(['balanced_accuracy','label_flip_fraction','mean_probability_total_variation']):
            ax=axes[i,j]
            for name,color in zip(c.ORDER,COLORS):ax.plot(range(3),[d['models'][name][scenario]['metrics'][metric] for scenario in scenarios],color=color,lw=2.4 if 'ensemble' in name else 1,ls='-' if 'ensemble' in name else '--',marker='o',ms=3,label=DISPLAY[name])
            ax.set_xticks(range(3),ticks);ax.set_xlabel(family);ax.set_ylabel(metric.replace('_',' '));ax.grid(alpha=.2)
    fig.suptitle(TITLES[cohort]+' | all 7 fixed pixel-stress conditions',fontweight='bold');fig.legend(*axes[0,0].get_legend_handles_labels(),ncol=4,loc='lower center',bbox_to_anchor=(.5,.04),fontsize=8);fig.text(.025,.015,'Same patient, perturbation and clean reference for every model. These synthetic 2D controls do not simulate MRI acquisition physics.',fontsize=8);fig.tight_layout(rect=(0,.12,1,.94));save(fig,out,cohort+'_all_stress',synthetic)


def training_figure(suite,freeze,out,synthetic):
    plt=plotting();fig,axes=plt.subplots(2,3,figsize=(14.2,8.2))
    for j,name in enumerate(c.NEW):
        selection=freeze['selections'][name];folder=c.safe(suite,selection['checkpoint_path']).parent
        history=c.read(folder/'history.json')
        if c.sha(folder/'history.json')!=selection['history_sha256'] or [r['epoch'] for r in history]!=list(range(len(history))):raise ValueError('Clinical training history changed')
        for prefix,label in [('train_aug','Augmented training'),('train_clean','Clean training'),('val','Clean validation')]:
            for i,metric in enumerate(['loss','accuracy']):
                axes[i,j].plot([r['epoch'] for r in history],[np.nan if r[prefix+'_'+metric] is None else r[prefix+'_'+metric] for r in history],label=label)
        for i,ax in enumerate(axes[:,j]):ax.axvline(selection['epoch'],ls=':',color='black');ax.set(xlabel='Epoch (0 = initialization)',ylabel='Patient NLL' if i==0 else 'Patient accuracy');ax.grid(alpha=.2);ax.legend(fontsize=8)
        axes[0,j].set_title(DISPLAY[name]+f" | selected epoch {selection['epoch']}")
    fig.suptitle('All three clinical runs | raw unsmoothed training and validation curves',fontweight='bold');fig.text(.02,.02,'32 training and 11 validation patients. Clean training is resubstitution; augmentation is applied only to optimization draws.\nStored trainer NLL uses its fixed 1e-15 clipping; held-out NLL uses the frozen comparison helper 1e-12 clipping.',fontsize=9);fig.tight_layout(rect=(0,.085,1,.94));save(fig,out,'all_training_curves',synthetic)


def analyze(data,draws=10000):
    analyses={}
    for cohort,d in data.items():
        models={n:d['models'][n]['clean']['metrics'] for n in c.ORDER};older=d['models']['round1_clinical_ensemble']['clean']['prob'];paired={}
        for name in ['new_clinical_ensemble']+c.NEW:paired[name+'_minus_round1_clinical_ensemble']=s.paired_comparison(d['y'],d['models'][name]['clean']['prob'],older,'clinical_fixed_v1/'+cohort+'/'+name,draws=draws)
        spread={}
        for condition,names in [('new',c.NEW),('old',list(c.OLD))]:spread[condition]={metric:dict(values={n:models[n][metric] for n in names},mean=float(np.mean([models[n][metric] for n in names])),sample_sd=float(np.std([models[n][metric] for n in names],ddof=1))) for metric in ['accuracy','balanced_accuracy','sensitivity_ad','specificity_cn','nll','binary_brier','roc_auc']}
        analyses[cohort]=dict(patient_ids=d['keys'],models=models,paired_comparisons=paired,optimization_seed_spread=spread)
    return analyses


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--suite',type=Path,required=True);ap.add_argument('--manifest',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args();suite=args.suite.resolve();output=args.output.resolve()
    if output.exists():raise ValueError('Clinical report output must be new')
    freeze,predictions,data,recomputed=read_verified(suite,args.manifest.resolve());analyses=analyze(data);output.mkdir(parents=True)
    training_figure(suite,freeze,output,False)
    for cohort,d in data.items():cohort_figures(cohort,d,output,False);stress_figures(cohort,d,output,False)
    mci=[dict(group_key=r['group_key'],cohort=r['cohort'],diagnosis='MCI',model=r['model'],scenario=r['scenario'],p_ad=r['probabilities'][1],threshold_label='AD' if r['predicted_ad'] else 'CN',prediction_only=True) for r in predictions if not r['binary_evaluable']]
    table(output/'mci_prediction_only.csv',mci)
    table(output/'all_model_scenario_metrics.csv',[{k:v for k,v in row.items() if not isinstance(v,(list,dict))} for row in recomputed])
    c.dump(output/'analysis.json',dict(status='complete_verified_predictions_only',freeze_sha256=c.sha(suite/'clinical_evaluation_freeze.json'),manifest_sha256=c.sha(args.manifest),all_models=c.ORDER,cohorts=analyses,recomputed_all168_metrics=True,recomputed_all_uniform_ensemble_probabilities=True,primary_contrast='new_clinical_ensemble minus round1_clinical_ensemble',fresh_external_binary_n=19,mci_patients=15,mci_scored=False,protocol=freeze['protocol'],no_inference_performed=True,real_figures_need_visual_review=True))
    lines=['# Separate clinical AD/control pilot','','Fresh Harvard results have 19 patients (9 AD, 10 CN); the 10 Harvard and 5 OpenNeuro MCI participants receive predictions only. Besga 11 and OpenNeuro 15 remain separate reused benchmarks. No clinical-superiority, population-risk-calibration or neuroradiologist claim follows.','','Accuracy, sensitivity and specificity below include exact 95% binomial intervals.','','| Cohort | Model | Accuracy (95% CI) | AD sensitivity (95% CI) | CN specificity (95% CI) | Balanced accuracy |','|---|---|---:|---:|---:|---:|']
    for cohort,a in analyses.items():
        for name,m in a['models'].items():
            interval_cells=[f'{100*m[k]:.1f}% ({100*m[ci][0]:.1f}, {100*m[ci][1]:.1f})' for k,ci in [('accuracy','accuracy_exact95'),('sensitivity_ad','sensitivity_exact95'),('specificity_cn','specificity_exact95')]]
            lines.append('| '+TITLES[cohort]+' | '+DISPLAY[name]+' | '+' | '.join(interval_cells+[f"{100*m['balanced_accuracy']:.1f}%"])+' |')
    lines+=['','Fixed new ensemble minus fixed old ensemble: paired differences in percentage points and conditional 95% intervals from 10,000 diagnosis-stratified patient bootstrap draws. These condition on the observed cohort and diagnosis counts.','','| Cohort | Accuracy difference | Balanced-accuracy difference | Sensitivity difference | Specificity difference |','|---|---:|---:|---:|---:|']
    for cohort,a in analyses.items():
        differences=a['paired_comparisons']['new_clinical_ensemble_minus_round1_clinical_ensemble']['differences']
        cells=[f"{100*differences[k]['point']:+.1f} ({100*differences[k]['conditional95'][0]:+.1f}, {100*differences[k]['conditional95'][1]:+.1f})" for k in ['accuracy','balanced_accuracy','sensitivity_ad','specificity_cn']]
        lines.append('| '+TITLES[cohort]+' | '+' | '.join(cells)+' |')
    lines+=['','Exact intervals, all patient-paired intervals, discordance counts and seed spread are in analysis.json. McNemar p-values are unadjusted exploratory diagnostics, not clinical-superiority evidence. All 168 scenario metrics are independently recomputed from the complete 3,360-record patient probability grid.','', 'OpenNeuro contains only one AD patient. Diagnosis-stratified bootstrap repeatedly resamples that same person; its conditional sensitivity interval is not population sensitivity precision. Use the exact binomial sensitivity interval and retain this limitation.', '', 'Old models retain mean slice probabilities; new models use softmax of mean slice logits. Both ensembles use uniform patient probability means. This compares complete pipelines.','', 'All 19 graph PDFs require visual review before publication. MCI scores are not diagnostic validation or progression predictions.']
    (output/'README.md').write_text('\n'.join(lines)+'\n');inventory={f.name:c.sha(f) for f in output.iterdir() if f.is_file()};c.dump(output/'SHA256SUMS.json',inventory);c.dump(output/'report_complete.json',dict(status='complete_numerically_verified_visual_review_required',figures=len(list(output.glob('*.pdf'))),payloads=len(inventory),manifest_sha256=c.sha(output/'SHA256SUMS.json'),unit='patient',no_inference=True));print(c.read(output/'report_complete.json'))

if __name__=='__main__':main()
