#!/usr/bin/env python3
"""Analyze frozen evaluator CSVs; never train, infer, select, or alter predictions.

Use --help for required inputs. The analysis protocol is fixed in PROTOCOL below.
All intervals are exploratory; their sampling assumptions do not remove leakage,
source confounding, unknown patient identities, or historical benchmark exposure.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import scipy
from scipy.stats import binomtest

STAGES = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']
METRICS = ['accuracy', 'balanced_accuracy', 'macro_f1']
PROTOCOL = {
    'version': 2, 'specified_date': '2026-10-11', 'bootstrap_resamples': 2000,
    'base_seed': 1729, 'confidence_level': .95,
    'bootstrap': 'paired, unstratified, percentile; sample clusters uniformly with replacement',
    'cluster_estimand': 'image-weighted metric; all images of each sampled hash group travel together',
    'internal_same_case_unit': 'prepared exact-pixel/dHash connected group; NOT patient identity',
    'internal_same_case_interpretation': 'Exact-row-held-out comparison only; prepared similarity groups can contain old strict-train/validation rows',
    'old_hash_group_unexposed_sensitivity': 'Prespecified secondary subset of unchanged primary: exclude every primary row whose prepared hash group contains any old strict train/val row; derive only from existing primary predictions; retain and report the full primary',
    'original100_unit': 'image; no supported patient/augmentation-family clusters',
    'clinical_unit': 'one source subject after the evaluator averages slice probabilities',
    'clinical_interval': 'two-sided Clopper-Pearson for accuracy, AD sensitivity and CN specificity',
    'paired_difference': 'candidate minus reference, using the same resampled cases/groups',
    'macro_f1': 'fixed class vocabulary, zero division contributes zero',
    'balanced_accuracy': 'mean class recall; undefined when a target class is absent',
    'bootstrap_missing_class': 'omit undefined replicate only for that metric; suppress CI if <95% valid',
    'decision_threshold': 'argmax in published class order; binary tie predicts CN',
    'multiplicity': 'no adjusted significance testing; all candidates/readers reported, none selected here',
    'reader_missingness': 'strict complete labels by default; explicit opt-in reports paired complete cases',
    'planned_modes': ['stage_only','multitask'], 'planned_seeds': [20261011,20261012,20261013],
    'primary_conditions': 'Report all three seeds per mode: arithmetic mean, sample SD and range of each metric; all six checkpoints remain visible',
    'validation_preference': 'Before any held-out evaluation, prefer mode with lowest mean stage validation NLL across the three seeds; lowest validation NLL seed within that mode is an exploratory representative, never a test-selected winner',
    'original100_old_exposure': {'train_images': 66, 'validation_images': 9, 'test_images': 25},
    'sources': [
        'https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html',
        'https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html',
        'https://docs.scipy.org/doc/scipy-1.16.1/reference/generated/scipy.stats._result_classes.BinomTestResult.proportion_ci.html',
    ],
}
LIMITATIONS = [
    'Original100 is a historically reused image-level descriptive benchmark. The old checkpoint had 66 train, 9 validation and 25 test image matches; it is not independent clinical validation.',
    'Legacy patient IDs and original augmentation families are unknown. Hash grouping is image similarity, not patient identity; cluster intervals do not establish clinical independence.',
    'The full primary intersection holds out exact rows for both models, but includes hash groups with old training/validation mates. A separately prespecified hash-unexposed sensitivity subset addresses only observed prepared-group overlap, changes sample composition, and does not prove patient independence or absence of other related old images.',
    'Bootstrap intervals condition on the observed sample and fixed fitted checkpoints. They omit training randomness and dataset/source uncertainty; boundary or identical predictions can yield degenerate intervals that do not prove certainty or equivalence.',
    'Unadjusted exploratory intervals across multiple candidates, endpoints and readers must not be treated as confirmatory significance or used to select a model on the test set.',
    'Reader labels concern these fixed images, not prospective clinical diagnosis; a fixed reader panel is not a random sample of neuroradiologists.',
    'Clinical AD/CN is a separate endpoint from the four legacy stages. MCI remains prediction-only and is excluded from binary accuracy denominators.',
    'Clinical exact binomial intervals assume independent, representative source subjects. Tiny, selected cohorts, source confounding and preprocessing inspection limit interpretation; one external AD subject cannot establish stable sensitivity.',
]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''): h.update(block)
    return h.hexdigest()


def read_csv(path, required):
    with Path(path).open(newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        if len(fields) != len(set(fields)) or not set(required) <= set(fields):
            raise ValueError(f'Invalid/missing CSV columns in {path}')
        rows = list(reader)
    if not rows or any(None in r or any(v is None for v in r.values()) for r in rows):
        raise ValueError(f'Empty or malformed CSV: {path}')
    return rows


def index_rows(rows, key, context):
    indexed = {}
    for row in rows:
        value = row[key]
        if not value or value in indexed: raise ValueError(f'Empty/duplicate {key}: {context}')
        indexed[value] = row
    return indexed


def require_same(actual, expected, context):
    if set(actual) != set(expected):
        raise ValueError(f'Case mismatch in {context}: missing={len(set(expected)-set(actual))}, extra={len(set(actual)-set(expected))}')


def class_metrics(y, pred, k):
    cm = np.bincount(k * y + pred, minlength=k*k).reshape(k, k)
    support, predicted = cm.sum(1), cm.sum(0)
    tp = np.diag(cm).astype(float)
    f1 = np.divide(2*tp, support+predicted, out=np.zeros(k), where=(support+predicted)>0)
    recall = np.divide(tp, support, out=np.full(k, np.nan), where=support>0)
    return {
        'accuracy': float(tp.sum()/len(y)),
        'balanced_accuracy': float(recall.mean()) if np.all(support>0) else None,
        'macro_f1': float(f1.mean()), 'confusion_matrix': cm.tolist(),
        'class_counts': support.tolist(), 'correct': int(tp.sum()), 'n': len(y),
    }


def exact_interval(successes, trials):
    if trials == 0: return {'successes': 0, 'n': 0, 'estimate': None, 'ci95': None}
    ci = binomtest(successes, trials).proportion_ci(confidence_level=.95, method='exact')
    return {'successes': int(successes), 'n': int(trials), 'estimate': successes/trials,
            'ci95': [float(ci.low), float(ci.high)], 'method': 'Clopper-Pearson exact; conditional on binomial assumptions'}


def named_seed(name):
    return int.from_bytes(hashlib.sha256(f"{PROTOCOL['base_seed']}:{name}".encode()).digest()[:8], 'big')


def interval(values):
    finite = np.asarray([v for v in values if v is not None], dtype=float)
    valid = len(finite)
    enough = valid >= .95 * PROTOCOL['bootstrap_resamples']
    return {'ci95': np.quantile(finite, [.025, .975]).tolist() if enough else None,
            'valid_resamples': valid, 'requested_resamples': PROTOCOL['bootstrap_resamples'],
            'degenerate': bool(enough and np.ptp(finite)==0),
            'suppressed_reason': None if enough else 'Fewer than 95% of resamples define this metric'}


def bootstrap_analysis(y, predictions, groups, context, pairs, labels=STAGES):
    """Same sampled cluster indices for every model and every paired contrast."""
    y = np.asarray(y, dtype=int)
    preds = {n: np.asarray(v, dtype=int) for n, v in predictions.items()}
    if not len(y) or len(groups)!=len(y) or any(len(v)!=len(y) for v in preds.values()):
        raise ValueError('Unaligned/empty bootstrap input')
    members = defaultdict(list)
    for i,g in enumerate(groups): members[g].append(i)
    members = [np.asarray(members[g]) for g in sorted(members)]
    rng = np.random.default_rng(named_seed(context))
    point = {n: class_metrics(y,v,len(labels)) for n,v in preds.items()}
    draws = {n: {m: [] for m in METRICS} for n in preds}
    differences = {f'{a}_minus_{b}': {m: [] for m in METRICS} for a,b in pairs}
    for _ in range(PROTOCOL['bootstrap_resamples']):
        idx = np.concatenate([members[j] for j in rng.integers(len(members), size=len(members))])
        scores = {n: class_metrics(y[idx],v[idx],len(labels)) for n,v in preds.items()}
        for n in preds:
            for m in METRICS: draws[n][m].append(scores[n][m])
        for a,b in pairs:
            for m in METRICS:
                x,z = scores[a][m],scores[b][m]
                differences[f'{a}_minus_{b}'][m].append(None if x is None or z is None else x-z)
    models = {n: {'point': point[n], 'bootstrap': {m: interval(draws[n][m]) for m in METRICS}} for n in preds}
    comparisons = {}
    for a,b in pairs:
        key = f'{a}_minus_{b}'
        ac,bc = preds[a]==y,preds[b]==y
        comparisons[key] = {
            'candidate': a, 'reference': b, 'n_paired': len(y),
            'candidate_correct_reference_wrong': int(np.sum(ac & ~bc)),
            'candidate_wrong_reference_correct': int(np.sum(~ac & bc)),
            'both_correct': int(np.sum(ac & bc)), 'both_wrong': int(np.sum(~ac & ~bc)),
            'difference': {m: {'estimate': None if point[a][m] is None or point[b][m] is None else point[a][m]-point[b][m],
                               **interval(differences[key][m])} for m in METRICS},
        }
    return {'n': len(y), 'n_resampling_units': len(members), 'seed': named_seed(context),
            'class_order': labels, 'models': models, 'paired_comparisons': comparisons}


def validate_probabilities(row, names, context):
    try: p = np.asarray([float(row[n]) for n in names])
    except (KeyError, ValueError) as exc: raise ValueError(f'Invalid probability in {context}') from exc
    if not np.all(np.isfinite(p)) or np.any(p<0) or np.any(p>1) or abs(p.sum()-1)>1e-5:
        raise ValueError(f'Probabilities must be finite, normalized and within [0,1]: {context}')
    return p


def load_stage_predictions(path, expected, key):
    rows = read_csv(path, [key,'path','TrueCategory','ModelPrediction'] + ['P_'+x for x in STAGES])
    actual = index_rows(rows,key,str(path)); require_same(actual,expected,path.name)
    result = []
    for case in sorted(expected):
        row,truth = actual[case],expected[case]
        if row['TrueCategory']!=truth or row['ModelPrediction'] not in STAGES:
            raise ValueError(f'Label mismatch in {path.name}: {case}')
        p = validate_probabilities(row,['P_'+x for x in STAGES],path.name)
        if row['ModelPrediction'] != STAGES[int(p.argmax())]: raise ValueError(f'Prediction/probability mismatch: {path.name}')
        if key=='FileName' and row['path']!='data/evaluation/images/'+case: raise ValueError('Original100 path mismatch')
        result.append(STAGES.index(row['ModelPrediction']))
    return result


def clinical_subjects(path):
    rows = read_csv(path,['diagnosis','binary_label','split'])
    subjects = {}
    slices = set()
    for r in rows:
        cohort = r.get('dataset') or r.get('cohort')
        subject = r.get('group_key') or r.get('subject_key') or (str(cohort)+':'+(r.get('subject_id') or r.get('source_subject_id','')))
        image = r.get('image_path') or r.get('path')
        if not cohort or not subject or not image or image in slices: raise ValueError('Invalid/duplicate clinical image identity')
        slices.add(image)
        wanted = {'AD':'1','CN':'0','MCI':''}.get(r['diagnosis'])
        if wanted is None or r['binary_label']!=wanted: raise ValueError('Invalid clinical diagnosis/label mapping')
        val = (cohort,subject,r['split'],r['binary_label'],r['diagnosis'])
        if subject in subjects and subjects[subject]['identity']!=val: raise ValueError('Clinical subject crosses cohorts/splits/labels')
        subjects.setdefault(subject,{'identity':val,'n_slices':0})['n_slices'] += 1
    result = defaultdict(dict)
    for s,record in subjects.items():
        co,_,split,label,diagnosis = record['identity']
        if split in {'test','external'}:
            result[split+'__'+co][s] = {'cohort':co,'label':label,'diagnosis':diagnosis,'n_slices':record['n_slices']}
    return dict(result)


def clinical_analysis(file, expected):
    rows = read_csv(file,['cohort','subject_id','binary_label','P_CN','P_AD','n_slices'])
    actual = index_rows(rows,'subject_id',file.name); require_same(actual,expected,file.name)
    y,pred,mci = [],[],[]
    for sid in sorted(expected):
        r,e = actual[sid],expected[sid]
        if r['cohort']!=e['cohort'] or r['binary_label']!=e['label'] or int(r['n_slices'])!=e['n_slices']:
            raise ValueError(f'Clinical identity/label/slice count mismatch: {file.name}')
        prob = validate_probabilities(r,['P_CN','P_AD'],file.name)
        if e['label']=='':
            mci.append({'subject_id':sid,'P_CN':float(prob[0]),'P_AD':float(prob[1]),'interpretation':'MCI: prediction-only; not an AD/CN target'})
        else: y.append(int(e['label'])); pred.append(int(prob.argmax()))
    y,pred = np.asarray(y,dtype=int),np.asarray(pred,dtype=int)
    point = class_metrics(y,pred,2) if len(y) else None
    ad,cn = y==1,y==0
    return {'n_subjects':len(expected),'n_binary_evaluable':len(y),'n_mci_prediction_only':len(mci),
            'class_order':['CN','AD'],'point':point,
            'accuracy_exact':exact_interval(int(np.sum(y==pred)),len(y)),
            'AD_sensitivity_exact':exact_interval(int(np.sum(pred[ad]==1)),int(ad.sum())),
            'CN_specificity_exact':exact_interval(int(np.sum(pred[cn]==0)),int(cn.sum())),
            'mci_prediction_only':mci,
            'warning':'Exploratory source-subject estimates. Exact intervals do not remove selection/source bias; small AD counts cannot establish stable sensitivity.'}


def condition_summaries(models, checkpoints):
    """Training-seed spread is separate from conditional case-resampling CIs."""
    result = {}
    for mode in PROTOCOL['planned_modes']:
        names = sorted(n for n,c in checkpoints.items() if c.get('selection_config',{}).get('mode')==mode and n in models)
        if not names: continue
        metrics = {}
        for metric in METRICS:
            values = [models[n]['point'][metric] for n in names]
            metrics[metric] = {'mean':float(np.mean(values)) if all(v is not None for v in values) else None,
                               'sample_sd':float(np.std(values,ddof=1)) if len(values)>1 and all(v is not None for v in values) else None,
                               'min':min(values) if all(v is not None for v in values) else None,
                               'max':max(values) if all(v is not None for v in values) else None,
                               'per_candidate':dict(zip(names,values))}
        result[mode] = {'n_seeds':len(names),'candidates':names,'metrics':metrics,
                        'interpretation':'Fixed, prespecified training-seed mean and spread; SD is not a confidence interval, and three seeds are not a robust estimate of training variability.'}
    return result


def baseline_group_exposure(stage, intersection):
    groups = defaultdict(Counter)
    for r in stage.values(): groups[r['group_id']][r['baseline_strict_split']] += 1
    primary_groups = {r['group_id'] for r in intersection.values()}
    train = {g for g in primary_groups if groups[g]['train']}
    train_val = {g for g in primary_groups if groups[g]['train'] or groups[g]['val']}
    unknown = {g for g in primary_groups if any(k not in {'train','val','test'} for k in groups[g])}
    return {'primary_images':len(intersection),'primary_hash_groups':len(primary_groups),
            'images_with_old_train_group_mate':sum(r['group_id'] in train for r in intersection.values()),
            'images_with_old_train_or_validation_group_mate':sum(r['group_id'] in train_val for r in intersection.values()),
            'groups_with_old_train_mate':len(train),'groups_with_old_train_or_validation_mate':len(train_val),
            'groups_with_unknown_old_split':len(unknown),
            'scope':'Within retained prepared stage rows; omitted legacy rows and unidentified relationships remain unresolved.'},groups,train_val


def validate_sensitivity_manifest(manifest_path, audit_path, stage_path, intersection_path, split_audit_path, stage, intersection):
    """Recompute membership from frozen prepared metadata, never predictions."""
    subset = index_rows(read_csv(manifest_path,list(next(iter(intersection.values())))),'path','old-hash-unexposed sensitivity')
    exposure,groups,excluded_groups = baseline_group_exposure(stage,intersection)
    expected = {p:r for p,r in intersection.items() if r['group_id'] not in excluded_groups}
    require_same(subset,expected,'old-hash-unexposed sensitivity exact subset')
    if any(r!=intersection[p] for p,r in subset.items()): raise ValueError('Sensitivity row metadata differs from frozen primary')
    audit = json.loads(Path(audit_path).read_text())
    if audit.get('status')!='prespecified_before_heldout_model_evaluation': raise ValueError('Sensitivity audit lacks prespecified status')
    for key,path in [('stage_manifest',stage_path),('same_case_test_intersection',intersection_path),('split_audit',split_audit_path)]:
        if audit.get('sources',{}).get(key,{}).get('sha256')!=digest(path): raise ValueError('Sensitivity audit source digest mismatch: '+key)
    if audit.get('subset',{}).get('sha256')!=digest(manifest_path): raise ValueError('Sensitivity manifest digest mismatch')
    excluded = {p:r for p,r in intersection.items() if p not in subset}
    for section,rows in [('primary',intersection),('subset',subset),('excluded',excluded)]:
        item = audit.get(section,{})
        counts = dict(Counter(STAGES[int(r['label'])] for r in rows.values()))
        if item.get('rows')!=len(rows) or item.get('unique_hash_groups')!=len({r['group_id'] for r in rows.values()}) or item.get('class_counts')!=counts:
            raise ValueError('Sensitivity audit counts mismatch: '+section)
    recorded_groups = index_rows(audit.get('excluded',{}).get('groups',[]),'group_id','sensitivity excluded groups')
    require_same(recorded_groups,excluded_groups,'sensitivity excluded group identities')
    for gid,item in recorded_groups.items():
        if item.get('intersection_rows')!=sum(r['group_id']==gid for r in intersection.values()) or item.get('old_strict_train_rows_in_prepared_group')!=groups[gid]['train'] or item.get('old_strict_validation_rows_in_prepared_group')!=groups[gid]['val'] or item.get('old_strict_splits_in_prepared_group')!=sorted(groups[gid]):
            raise ValueError('Sensitivity excluded-group metadata mismatch')
    return subset,{'status':'prespecified_secondary_sensitivity','primary_unchanged':True,'extra_model_inference':False,
                   'manifest_sha256':digest(manifest_path),'audit_sha256':digest(audit_path),
                   'n':len(subset),'n_hash_groups':len({r['group_id'] for r in subset.values()}),'excluded_primary_rows':len(excluded),
                   'baseline_group_exposure':exposure,
                   'interpretation':'Conservative image-similarity sensitivity subset with no observed old train/val mate in the retained prepared hash group; hash groups are not patients, omitted source relationships remain unresolved, and subset selection changes class composition/difficulty.'}


def analyze(args):
    ev,prepared = args.evaluation_dir.resolve(),args.prepared.resolve()
    used = []
    def use(path):
        path = Path(path).resolve(); used.append(path); return path
    meta = json.loads(use(ev/'evaluation.json').read_text())
    checkpoints = meta.get('checkpoints',{})
    if 'old_baseline' not in checkpoints or len(checkpoints)<2: raise ValueError('Require old baseline and all frozen candidates')
    names = sorted(checkpoints)
    if any(n!='old_baseline' and (not n.startswith('candidate_') or not n[10:].isdigit()) for n in names):
        raise ValueError('Unexpected checkpoint identifier')
    stage_path = use(prepared/'stage_manifest.csv')
    stage = index_rows(read_csv(stage_path,['path','label','group_id','split','baseline_strict_split']), 'path','stage manifest')
    group_splits = defaultdict(set)
    group_labels = defaultdict(set)
    for r in stage.values():
        if not r['group_id'] or r['label'] not in {'0','1','2','3'}: raise ValueError('Invalid stage group/label')
        group_splits[r['group_id']].add(r['split']); group_labels[r['group_id']].add(r['label'])
    if any(len(v)!=1 for v in group_splits.values()) or any(len(v)!=1 for v in group_labels.values()):
        raise ValueError('Stage hash group crosses splits or has mixed labels')
    intersection_path = use(prepared/'same_case_test_intersection.csv')
    intersection = index_rows(read_csv(intersection_path,['path','label','group_id','split','baseline_strict_split']), 'path','intersection')
    expected_intersection = {p:r for p,r in stage.items() if r['split']=='test' and r['baseline_strict_split']=='test'}
    require_same(intersection,expected_intersection,'prepared intersection')
    for p,r in intersection.items():
        if any(r[k]!=stage[p][k] for k in ('label','group_id','split','baseline_strict_split')): raise ValueError('Intersection metadata mismatch')
    audit = json.loads(use(prepared/'split_audit.json').read_text())
    if audit['manifest_sha256']!=digest(stage_path) or audit['same_case_test_intersection_sha256']!=digest(intersection_path):
        raise ValueError('Prepared manifest digest mismatch')
    key = index_rows(read_csv(use(args.original100_key),['FileName','TrueCategory']),'FileName','original100 key')
    if len(key)!=100 or any(r['TrueCategory'] not in STAGES for r in key.values()): raise ValueError('Require exact original100 key and class vocabulary')
    clinical = clinical_subjects(use(args.clinical_manifest)) if args.clinical_manifest else {}
    for n,info in checkpoints.items():
        if not info.get('sha256') or len(info['sha256'])!=64: raise ValueError('Missing checkpoint digest')
        cfg = info.get('selection_config',{})
        if n!='old_baseline':
            if cfg.get('smoke') or cfg.get('test_used_for_selection') is not False or cfg.get('mode') not in {'stage_only','multitask'}:
                raise ValueError('Candidate lacks research validation-only selection provenance')
            if cfg.get('stage_manifest_sha256')!=digest(stage_path): raise ValueError('Candidate stage manifest mismatch')
            clinical_digest = cfg.get('clinical_manifest_sha256')
            if clinical_digest and (not args.clinical_manifest or clinical_digest!=digest(args.clinical_manifest)):
                raise ValueError('Candidate clinical manifest mismatch')
            if cfg.get('mode')=='multitask' and not clinical: raise ValueError('Multitask candidate requires held-out clinical manifest')
    expected_files = {f'{n}_{co}_predictions.csv' for n in names for co in ['internal_same_case','original100']}
    for n,info in checkpoints.items():
        reported = set(info.get('clinical_aux_AD_CN',{}))
        wanted = set(clinical) if info.get('selection_config',{}).get('mode')=='multitask' else set()
        require_same(reported,wanted,f'{n} clinical cohort outputs')
        expected_files |= {f'{n}_{co}_subject_predictions.csv' for co in wanted}
    require_same({p.name for p in ev.glob('*_predictions.csv')},expected_files,'prediction files')
    seeds = {}
    for n,c in checkpoints.items():
        if n=='old_baseline': continue
        cfg = c['selection_config']; pair = (cfg['mode'],cfg.get('seed'))
        if pair in seeds: raise ValueError('Duplicated condition/seed')
        seeds[pair] = n
    require_same(seeds,{(m,s) for m in PROTOCOL['planned_modes'] for s in PROTOCOL['planned_seeds']},'prespecified six condition/seed runs')
    validation = json.loads(use(args.validation_selection).read_text())
    if validation.get('frozen_before_heldout_evaluation') is not True: raise ValueError('Missing pre-evaluation validation selection record')
    require_same(validation.get('candidates',{}),set(names)-{'old_baseline'},'validation selection candidates')
    for n,v in validation['candidates'].items():
        cfg = checkpoints[n]['selection_config']
        if v.get('checkpoint_sha256')!=checkpoints[n]['sha256'] or v.get('mode')!=cfg['mode'] or v.get('seed')!=cfg['seed']:
            raise ValueError('Validation selection record does not match evaluated checkpoints')
        if not np.isfinite(v.get('stage_validation_nll',float('nan'))): raise ValueError('Missing finite validation NLL')
    if validation.get('analysis_protocol')!=PROTOCOL or validation.get('analysis_script_sha256')!=digest(__file__):
        raise ValueError('Analysis code/protocol changed after validation selection was frozen')
    means = {mode:float(np.mean([r['stage_validation_nll'] for r in validation['candidates'].values() if r['mode']==mode])) for mode in PROTOCOL['planned_modes']}
    preferred_mode = min(means,key=lambda m:(means[m],m))
    preferred_candidate = min((n for n,r in validation['candidates'].items() if r['mode']==preferred_mode),key=lambda n:(validation['candidates'][n]['stage_validation_nll'],validation['candidates'][n]['seed']))
    if validation.get('preferred_mode')!=preferred_mode or validation.get('exploratory_representative_candidate')!=preferred_candidate or validation.get('mode_mean_stage_validation_nll')!=means:
        raise ValueError('Validation preference does not follow the prespecified rule')
    sensitivity_path = getattr(args,'old_hash_unexposed_manifest',None)
    sensitivity_audit = getattr(args,'baseline_exposure_audit',None)
    if bool(sensitivity_path)!=bool(sensitivity_audit): raise ValueError('Sensitivity manifest and audit must be supplied together')
    pinned_sensitivity = validation.get('old_hash_unexposed_sensitivity')
    sensitivity = None
    sensitivity_info = {'status':'not_requested','primary_unchanged':True}
    if sensitivity_path:
        if not pinned_sensitivity or pinned_sensitivity.get('manifest_sha256')!=digest(sensitivity_path) or pinned_sensitivity.get('audit_sha256')!=digest(sensitivity_audit):
            raise ValueError('Sensitivity files were not pinned before held-out evaluation or changed')
        sensitivity,sensitivity_info = validate_sensitivity_manifest(use(sensitivity_path),use(sensitivity_audit),stage_path,intersection_path,prepared/'split_audit.json',stage,intersection)
    elif pinned_sensitivity:
        raise ValueError('Prespecified sensitivity files are required by the frozen selection record')
    exposure,_,_ = baseline_group_exposure(stage,intersection)
    result = {'protocol':PROTOCOL,'limitations':LIMITATIONS,'checkpoint_provenance':checkpoints,
              'validation_only_preference':validation,'baseline_group_exposure':exposure,
              'sensitivity_provenance':sensitivity_info,'stage':{},'clinical':{}}
    pairs = [(n,'old_baseline') for n in names if n!='old_baseline']
    pairs += [(seeds[('multitask',s)],seeds[('stage_only',s)]) for s in PROTOCOL['planned_seeds']]
    expectations = {'internal_same_case':{p:STAGES[int(r['label'])] for p,r in intersection.items()},
                    'original100':{p:r['TrueCategory'] for p,r in key.items()}}
    all_preds = {}
    for co,expected in expectations.items():
        y = [STAGES.index(expected[p]) for p in sorted(expected)]
        predictions = {n:load_stage_predictions(use(ev/f'{n}_{co}_predictions.csv'),expected,'FileName' if co=='original100' else 'path') for n in names}
        all_preds[co] = predictions
        groups = [intersection[p]['group_id'] for p in sorted(expected)] if co=='internal_same_case' else sorted(expected)
        summary = bootstrap_analysis(y,predictions,groups,co,pairs)
        summary['primary_condition_summaries'] = condition_summaries(summary['models'],checkpoints)
        summary['resampling_unit'] = PROTOCOL[co+'_unit']
        summary['interpretation'] = 'Historically reused image-level descriptive benchmark; no independent clinical claim' if co=='original100' else 'Full primary exact-row-held-out comparison; prepared hash groups include old development exposure. Paired hash-group intervals are conditional and do not establish patient independence or equal absence of old image-family exposure.'
        if co=='internal_same_case': summary['baseline_group_exposure'] = exposure
        if co=='original100':
            for n,m in summary['models'].items():
                m['accuracy_binomial_sensitivity_analysis'] = exact_interval(m['point']['correct'],len(y))
            summary['binomial_warning'] = 'Image-level binomial intervals additionally assume independent images; this is not established. Shown only as a boundary-sensitive descriptive sensitivity analysis.'
        result['stage'][co] = summary
    if sensitivity is not None:
        primary_positions = {p:i for i,p in enumerate(sorted(intersection))}
        retained = sorted(sensitivity)
        y = [int(sensitivity[p]['label']) for p in retained]
        predictions = {n:[v[primary_positions[p]] for p in retained] for n,v in all_preds['internal_same_case'].items()}
        summary = bootstrap_analysis(y,predictions,[sensitivity[p]['group_id'] for p in retained],'old_hash_group_unexposed_sensitivity',pairs)
        summary['primary_condition_summaries'] = condition_summaries(summary['models'],checkpoints)
        summary['resampling_unit'] = PROTOCOL['internal_same_case_unit']
        summary['interpretation'] = sensitivity_info['interpretation']
        summary['role'] = 'Secondary prespecified sensitivity analysis; full primary remains unchanged'
        result['stage']['old_hash_group_unexposed_sensitivity'] = summary
    aligned = index_rows(read_csv(use(ev/'original100_case_aligned_old_candidates_readers.csv'),['FileName','TrueCategory']+names),'FileName','aligned original100')
    require_same(aligned,key,'aligned original100')
    reader_names = sorted(k for k in next(iter(aligned.values())) if k.startswith('Reader_'))
    require_same(reader_names,{'Reader_'+r for r in meta.get('reader_provenance',{})},'reader provenance')
    if not reader_names: raise ValueError('No readers in aligned benchmark')
    cases = sorted(key)
    for i,case in enumerate(cases):
        if aligned[case]['TrueCategory']!=key[case]['TrueCategory']: raise ValueError('Aligned truth mismatch')
        for n in names:
            if aligned[case][n]!=STAGES[all_preds['original100'][n][i]]: raise ValueError('Aligned model prediction mismatch')
    reader_result = {}
    for reader in reader_names:
        missing = [case for case in cases if aligned[case][reader] not in STAGES]
        if missing and not args.allow_incomplete_readers: raise ValueError(f'{reader} has {len(missing)} missing/unknown labels; explicit --allow-incomplete-readers required')
        if any(aligned[case][reader] not in {'','Missing','Unknown'} for case in missing): raise ValueError('Unrecognized reader label')
        keep = [case for case in cases if case not in missing]
        if not keep: raise ValueError(f'{reader} has no complete cases')
        y = [STAGES.index(key[case]['TrueCategory']) for case in keep]
        preds = {n:[STAGES.index(aligned[case][n]) for case in keep] for n in names+[reader]}
        summary = bootstrap_analysis(y,preds,keep,'reader:'+reader,[(n,reader) for n in names])
        summary.update({'excluded_cases':missing,'reader':'Fixed reader, paired complete cases only',
                        'complete_case_ids_sha256':hashlib.sha256('\n'.join(keep).encode()).hexdigest()})
        reader_result[reader] = summary
    common = [case for case in cases if all(aligned[case][r] in STAGES for r in reader_names)]
    panel = {'n_fixed_readers':len(reader_names),'common_complete_cases':len(common),'excluded_cases':[case for case in cases if case not in common],
             'interpretation':'Arithmetic mean of fixed-reader metrics on the same complete cases; does not pool reader-case observations as independent trials or generalize to a reader population.'}
    if common:
        y = np.asarray([STAGES.index(key[case]['TrueCategory']) for case in common])
        points = {r:class_metrics(y,np.asarray([STAGES.index(aligned[case][r]) for case in common]),4) for r in reader_names}
        panel['mean_reader_metrics'] = {m:float(np.mean([v[m] for v in points.values()])) if all(v[m] is not None for v in points.values()) else None for m in METRICS}
        panel['individual_reader_metrics'] = points
    result['readers'] = {'paired_comparisons':reader_result,'fixed_panel_summary':panel}
    for n,info in checkpoints.items():
        for co in sorted(info.get('clinical_aux_AD_CN',{})):
            result['clinical'].setdefault(co,{})[n] = clinical_analysis(use(ev/f'{n}_{co}_subject_predictions.csv'),clinical[co])
    result['clinical_condition_summaries'] = {co:condition_summaries({n:{'point':r['point']} for n,r in models.items() if r['point'] is not None},checkpoints) for co,models in result['clinical'].items()}
    result['input_manifest'] = [{'path':str(p),'sha256':digest(p)} for p in sorted(set(used))]
    result['software'] = {'numpy':np.__version__,'scipy':scipy.__version__,'analysis_script_sha256':digest(__file__)}
    return result


def write_outputs(result, out):
    out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()): raise ValueError('Output must be empty; preserve prior evidence')
    (out/'uncertainty_analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    rows = []
    for cohort,summary in result['stage'].items():
        for name,model in summary['models'].items():
            for metric in METRICS:
                interval_data = model['bootstrap'][metric]; ci = interval_data['ci95'] or [None,None]
                rows.append({'cohort':cohort,'model':name,'metric':metric,'estimate':model['point'][metric],
                             'ci95_low':ci[0],'ci95_high':ci[1],'n':summary['n'],'n_resampling_units':summary['n_resampling_units'],
                             'resampling_unit':summary['resampling_unit'],'bootstrap_valid':interval_data['valid_resamples'],'degenerate':interval_data['degenerate']})
    with (out/'stage_metrics.csv').open('w',newline='') as f:
        w = csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    pairs = []
    for cohort,summary in result['stage'].items():
        for name,contrast in summary['paired_comparisons'].items():
            for metric,value in contrast['difference'].items():
                ci = value['ci95'] or [None,None]
                pairs.append({'cohort':cohort,'comparison':name,'metric':metric,'difference_candidate_minus_reference':value['estimate'],
                              'ci95_low':ci[0],'ci95_high':ci[1],'n_paired':contrast['n_paired'],'degenerate':value['degenerate']})
    with (out/'paired_differences.csv').open('w',newline='') as f:
        w = csv.DictWriter(f,fieldnames=list(pairs[0])); w.writeheader(); w.writerows(pairs)
    readers = []
    for reader,summary in result['readers']['paired_comparisons'].items():
        for name,contrast in summary['paired_comparisons'].items():
            for metric,value in contrast['difference'].items():
                ci = value['ci95'] or [None,None]
                readers.append({'reader':reader,'comparison':name,'metric':metric,'difference_model_minus_reader':value['estimate'],
                                'ci95_low':ci[0],'ci95_high':ci[1],'n_paired':contrast['n_paired'],'n_missing':len(summary['excluded_cases']),
                                'interpretation':'Original100 reused image-level descriptive comparison; fixed reader'})
    with (out/'reader_paired_differences.csv').open('w',newline='') as f:
        w = csv.DictWriter(f,fieldnames=list(readers[0])); w.writeheader(); w.writerows(readers)
    clinical = []
    for cohort,models in result['clinical'].items():
        for name,model in models.items():
            for metric in ['accuracy_exact','AD_sensitivity_exact','CN_specificity_exact']:
                value = model[metric]; ci = value['ci95'] or [None,None]
                clinical.append({'cohort_and_split':cohort,'candidate':name,'metric':metric,'estimate':value['estimate'],
                                 'ci95_low':ci[0],'ci95_high':ci[1],'successes':value['successes'],'n':value['n'],
                                 'n_mci_prediction_only':model['n_mci_prediction_only'],'method':'Clopper-Pearson exact subject-level conditional binomial interval'})
    if clinical:
        with (out/'clinical_exact_intervals.csv').open('w',newline='') as f:
            w = csv.DictWriter(f,fieldnames=list(clinical[0])); w.writeheader(); w.writerows(clinical)
    (out/'interpretation_limits.txt').write_text('\n\n'.join(LIMITATIONS)+'\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evaluation-dir',required=True,type=Path)
    p.add_argument('--prepared',required=True,type=Path)
    p.add_argument('--original100-key',required=True,type=Path)
    p.add_argument('--clinical-manifest',type=Path)
    p.add_argument('--validation-selection',required=True,type=Path,help='Record created by freeze_validation_selection.py before held-out evaluator execution')
    p.add_argument('--old-hash-unexposed-manifest',type=Path,help='Optional prespecified subset pinned by the selection helper; paired with --baseline-exposure-audit')
    p.add_argument('--baseline-exposure-audit',type=Path,help='Prespecified audit for the sensitivity manifest; required when the frozen selection record pins it')
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--allow-incomplete-readers',action='store_true',help='Explicitly report paired complete cases and all exclusions; default refuses missing/unknown reader labels')
    args = p.parse_args()
    if args.output.exists() and any(args.output.iterdir()): p.error('Output must be empty')
    result = analyze(args)
    write_outputs(result,args.output)
    print(json.dumps({'status':'complete','output':str(args.output.resolve()),'stage_cohorts':list(result['stage']),'clinical_cohorts':list(result['clinical']),'bootstrap_resamples':2000}))


if __name__=='__main__': main()
