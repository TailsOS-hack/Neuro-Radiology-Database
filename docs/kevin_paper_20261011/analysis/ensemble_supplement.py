#!/usr/bin/env python3
"""Freeze and compute uniform three-seed supplements from existing predictions.

No model loading, inference, fitting, selection, promotion or primary-file edits.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

import summarize_frozen_evaluation as a

HERE = Path(__file__).resolve().parent
BASELINE_SHA256 = 'a8cc03aea74f5310c1fc7936c72f94a44e55af11e2ada2a8ba5f8edd6d924db5'
TRAINER_SHA256 = 'cbf67b7c6e842ce35eeef2e0c0a6a0752b865cec3399d7aa58bd5e8ff0e2a5e2'


def protocol():
    value = json.loads((HERE/'ensemble_protocol.json').read_text())
    if (value['version']!=1 or value['modes']!=a.PROTOCOL['planned_modes']
            or value['seeds']!=a.PROTOCOL['planned_seeds'] or value['members_per_condition']!=3
            or value['stage_class_order']!=a.STAGES or value['clinical_class_order']!=['CN','AD']):
        raise ValueError('Unsupported supplementary protocol or mismatch with primary plan')
    return value


def code_hashes():
    trainer = HERE.parent/'training/train_candidate.py'
    if a.digest(trainer)!=TRAINER_SHA256: raise ValueError('Frozen trainer reference changed')
    return {'supplement_script_sha256':a.digest(__file__),
            'supplement_protocol_sha256':a.digest(HERE/'ensemble_protocol.json'),
            'primary_analysis_script_sha256':a.digest(a.__file__),
            'primary_analysis_protocol_sha256':a.digest(HERE/'analysis_protocol.json'),
            'frozen_trainer_sha256':TRAINER_SHA256}


def source_manifest_paths():
    root = HERE.parents[2]
    prepared = root/'training_logs/experiments/kevin_20261011/prepared'
    return {'stage_manifest':prepared/'stage_manifest.csv',
            'primary_intersection':prepared/'same_case_test_intersection.csv',
            'split_audit':prepared/'split_audit.json',
            'original100_key':root/'data/evaluation/ground_truth/radiologist_test_key.csv',
            'clinical_manifest':root/'data/external/clinical_t1_preprocessed_v1/clinical_slices.csv',
            'sensitivity_manifest':HERE/'old_hash_group_unexposed_sensitivity.csv',
            'sensitivity_audit':HERE/'old_hash_group_unexposed_sensitivity_audit.json'}


def require_not_launched(evaluation_output):
    if evaluation_output.exists() and any(evaluation_output.iterdir()):
        raise ValueError('Freeze supplement before held-out outputs exist')
    if (evaluation_output.parent/'evaluation_launch.json').exists():
        raise ValueError('Primary evaluation already launched; supplement must be frozen first')


def freeze(evaluation_output, output):
    if output.exists() or output.is_symlink(): raise ValueError('Supplement freeze already exists')
    require_not_launched(evaluation_output)
    value = {'frozen_before_heldout_inference':True,'created_utc':datetime.now(timezone.utc).isoformat(),
             'checked_empty_evaluation_output':str(evaluation_output.resolve()),
             'protocol':protocol(),'code_hashes':code_hashes(),
             'source_manifests':{name:{'path':str(path.resolve()),'sha256':a.digest(path)} for name,path in source_manifest_paths().items()},
             'note':'Applies to all six checkpoints subsequently frozen by primary validation-only selection; never selects members from held-out outcomes.'}
    output.parent.mkdir(parents=True,exist_ok=True)
    require_not_launched(evaluation_output)
    with output.open('x') as file: json.dump(value,file,indent=2,allow_nan=False);file.write('\n')
    return value


def calibration(y, probabilities):
    """Same definitions as plot_evaluation_figures.calibration; descriptive only."""
    onehot = np.eye(probabilities.shape[1])[y]
    brier = float(np.mean(np.sum((probabilities-onehot)**2,axis=1)))
    pred,confidence = probabilities.argmax(axis=1),probabilities.max(axis=1)
    bins = np.minimum((confidence*10).astype(int),9)
    records,ece = [],0.
    for b in range(10):
        mask = bins==b;n = int(mask.sum())
        acc,conf = (float(np.mean(pred[mask]==y[mask])),float(confidence[mask].mean())) if n else (None,None)
        contribution = n/len(y)*abs(acc-conf) if n else 0.
        ece += contribution
        records.append({'lower':b/10,'upper':(b+1)/10,'upper_inclusive':b==9,'n':n,
                        'accuracy':acc,'mean_confidence':conf,'ece_contribution':contribution})
    return {'multiclass_brier_sum':brier,'top_label_ece_10_equal_width_bins':ece,'bins':records}


def member_summary(y, probabilities, classes):
    predictions = {n:p.argmax(axis=1) for n,p in probabilities.items()}
    points = {n:a.class_metrics(y,p,len(classes)) for n,p in predictions.items()}
    spread = {}
    for metric in a.METRICS:
        values = [p[metric] for p in points.values()]
        defined = all(v is not None for v in values)
        spread[metric] = {'mean':float(np.mean(values)) if defined else None,
                          'sample_sd':float(np.std(values,ddof=1)) if defined else None,
                          'min':min(values) if defined else None,'max':max(values) if defined else None}
    names = sorted(predictions)
    pairwise = {f'{names[i]}_versus_{names[j]}':float(np.mean(predictions[names[i]]!=predictions[names[j]]))
                for i in range(3) for j in range(i+1,3)}
    matrix = np.stack([predictions[n] for n in names])
    disagree = np.any(matrix!=matrix[0],axis=0)
    return {'member_points':points,'metric_spread':spread,
            'n_any_member_disagreement':int(disagree.sum()),'any_member_disagreement_rate':float(disagree.mean()),
            'pairwise_disagreement_rates':pairwise,
            'mean_pairwise_disagreement_rate':float(np.mean(list(pairwise.values()))),
            'interpretation':'Prespecified fixed-member spread and disagreement; not calibrated uncertainty or a rejection threshold.'}


def read_probabilities(file, ids, id_field, probability_fields):
    indexed = a.index_rows(a.read_csv(file,[id_field]+probability_fields),id_field,file.name)
    a.require_same(indexed,ids,file.name)
    return np.asarray([a.validate_probabilities(indexed[k],probability_fields,file.name) for k in ids])


def ensemble_specs(base, selected):
    specs,members = {},{}
    checkpoints = base['checkpoint_provenance']
    for mode in a.PROTOCOL['planned_modes']:
        names = sorted((n for n,c in checkpoints.items() if c.get('selection_config',{}).get('mode')==mode),
                       key=lambda n:checkpoints[n]['selection_config']['seed'])
        if len(names)!=3 or [checkpoints[n]['selection_config']['seed'] for n in names]!=a.PROTOCOL['planned_seeds']:
            raise ValueError('Ensemble must contain exactly the three prespecified seeds')
        configs = [checkpoints[n]['selection_config'] for n in names]
        sizes = {cfg.get('image_size') for cfg in configs}
        if sizes!={224}: raise ValueError('Shared frozen 224px input transform required')
        name = 'ensemble_'+mode;members[name] = names
        specs[name] = {
            'mode':mode,'head':'stage','class_order':a.STAGES,
            'members':[{'evaluation_name':n,'mode':mode,'seed':checkpoints[n]['selection_config']['seed'],
                        'checkpoint_sha256':checkpoints[n]['sha256'],
                        'checkpoint_path_in_primary_record':selected['candidates'][n]['checkpoint_path'],
                        'weight':1/3,'weight_fraction':'1/3'} for n in names],
            'aggregation':'Arithmetic mean of the three exported softmax probability vectors; fixed-order argmax',
            'shared_input_transform':{'frozen_trainer_sha256':TRAINER_SHA256,'function':'dataset(rows, root, train=False, image_size=224)',
                                      'steps':['PIL convert RGB','torchvision Resize((224,224), antialias=True), frozen default interpolation',
                                               'ToTensor','Normalize mean [0.485,0.456,0.406], std [0.229,0.224,0.225]'],
                                      'test_augmentation':False},
            'applicability':'Four legacy image-stage categories only; no binary clinical interpretation of this head',
            'stage_manifest_sha256':configs[0]['stage_manifest_sha256'],
            'no_production_wrapper_or_deployment':True,
        }
    clinical = dict(specs['ensemble_multitask'])
    clinical.update(head='aux',class_order=['CN','AD'],
                    aggregation='Average slice probabilities per source subject (already performed by evaluator), then arithmetic mean across all three multitask seeds',
                    applicability='Clinical AD/CN only; MCI has predictions but no binary target; stage-only auxiliary heads are inapplicable',
                    clinical_manifest_sha256=checkpoints[members['ensemble_multitask'][0]]['selection_config']['clinical_manifest_sha256'])
    specs['ensemble_multitask_AD_CN'] = clinical
    return specs,members


def verify_inputs(args):
    specification = protocol();pinned = json.loads(args.supplement_freeze.read_text())
    if pinned.get('frozen_before_heldout_inference') is not True or pinned.get('protocol')!=specification or pinned.get('code_hashes')!=code_hashes():
        raise ValueError('Supplement was not frozen or its code/protocol changed')
    selected = json.loads(args.validation_selection.read_text())
    if selected.get('checked_empty_evaluation_output')!=pinned.get('checked_empty_evaluation_output'):
        raise ValueError('Supplement and primary freeze refer to different evaluation outputs')
    if args.evaluation_dir.name!=Path(pinned['checked_empty_evaluation_output']).name:
        raise ValueError('Evaluation directory identity differs from pre-evaluation freeze')
    actual_sources = {'stage_manifest':args.prepared/'stage_manifest.csv',
                      'primary_intersection':args.prepared/'same_case_test_intersection.csv',
                      'split_audit':args.prepared/'split_audit.json','original100_key':args.original100_key,
                      'clinical_manifest':args.clinical_manifest,'sensitivity_manifest':args.old_hash_unexposed_manifest,
                      'sensitivity_audit':args.baseline_exposure_audit}
    a.require_same(pinned.get('source_manifests',{}),actual_sources,'supplement frozen source manifests')
    if any(pinned['source_manifests'][name]['sha256']!=a.digest(path) for name,path in actual_sources.items()):
        raise ValueError('Supplement source manifest changed after freeze')
    primary_saved = json.loads(args.primary_analysis.read_text())
    # Reuse every frozen case, source, class, probability, checkpoint and reader
    # validation. This recomputes deterministic CPU summaries, never inference.
    base = a.analyze(args)
    if (primary_saved.get('protocol')!=a.PROTOCOL
            or primary_saved.get('software',{}).get('analysis_script_sha256')!=a.digest(a.__file__)
            or primary_saved.get('input_manifest')!=base['input_manifest']
            or primary_saved.get('checkpoint_provenance')!=base['checkpoint_provenance']):
        raise ValueError('Saved primary report does not bind the same frozen inputs/code')
    if base['checkpoint_provenance']['old_baseline']['sha256']!=BASELINE_SHA256:
        raise ValueError('Old baseline checkpoint identity mismatch')
    if base['sensitivity_provenance']['status']!='prespecified_secondary_sensitivity':
        raise ValueError('This supplement requires the prespecified sensitivity subset')
    return base,selected,specification,pinned


def analyze(args):
    base,selected,specification,pinned = verify_inputs(args)
    specs,members = ensemble_specs(base,selected)
    result = {'role':'SUPPLEMENTARY; primary results remain unchanged', 'protocol':specification,
              'supplement_freeze':pinned,'ensemble_specifications':specs,
              'stage':{},'clinical':{},'limitations':specification['limitations']+a.LIMITATIONS,
              'new_model_inference':False,'winner_or_promotion':None}
    outputs = {}
    stage_rows = a.index_rows(a.read_csv(args.prepared/'same_case_test_intersection.csv',['path','label','group_id']),'path','primary')
    sensitivity = a.index_rows(a.read_csv(args.old_hash_unexposed_manifest,['path','label','group_id']),'path','sensitivity')
    original = a.index_rows(a.read_csv(args.original100_key,['FileName','TrueCategory']),'FileName','original100')
    all_names = ['old_baseline']+[n for names in members.values() for n in names]
    cached = {}
    for cohort in ('internal_same_case','original100'):
        ids = sorted(stage_rows if cohort=='internal_same_case' else original)
        cached[cohort] = (ids,{n:read_probabilities(args.evaluation_dir/f'{n}_{cohort}_predictions.csv',ids,
                                                 'path' if cohort=='internal_same_case' else 'FileName',['P_'+cl for cl in a.STAGES]) for n in all_names})
    for cohort in specification['stage_cohorts']:
        source = 'internal_same_case' if cohort=='old_hash_group_unexposed_sensitivity' else cohort
        source_ids,source_probs = cached[source]
        ids = sorted(sensitivity) if cohort=='old_hash_group_unexposed_sensitivity' else source_ids
        positions = {case:i for i,case in enumerate(source_ids)}
        probabilities = {n:p[[positions[case] for case in ids]] for n,p in source_probs.items()}
        y = np.asarray([a.STAGES.index(original[case]['TrueCategory']) if cohort=='original100' else int(stage_rows[case]['label']) for case in ids])
        groups = ids if cohort=='original100' else [stage_rows[case]['group_id'] for case in ids]
        averaged = {ens:np.mean([probabilities[n] for n in names],axis=0) for ens,names in members.items()}
        pred = {'old_baseline':probabilities['old_baseline'].argmax(axis=1),**{n:p.argmax(axis=1) for n,p in averaged.items()}}
        pairs = [(ens,'old_baseline') for ens in members]+[('ensemble_multitask','ensemble_stage_only')]
        summary = a.bootstrap_analysis(y,pred,groups,'ensemble_supplement:'+cohort,pairs)
        summary['interpretation'] = base['stage'][cohort]['interpretation']
        summary['resampling_unit'] = base['stage'][cohort]['resampling_unit']
        summary['member_analysis'] = {ens:member_summary(y,{n:probabilities[n] for n in names},a.STAGES) for ens,names in members.items()}
        summary['calibration_descriptive'] = {n:calibration(y,p) for n,p in {'old_baseline':probabilities['old_baseline'],**averaged}.items()}
        result['stage'][cohort] = summary
        for ens,p in averaged.items():
            outputs[f'{ens}_{cohort}_predictions.csv'] = [
                {'case_id':case,'TrueCategory':a.STAGES[int(y[i])],'ModelPrediction':a.STAGES[int(p[i].argmax())],
                 **{'P_'+cl:float(p[i,j]) for j,cl in enumerate(a.STAGES)},
                 'member_predictions':'|'.join(a.STAGES[int(probabilities[n][i].argmax())] for n in members[ens])}
                for i,case in enumerate(ids)]
    clinical = a.clinical_subjects(args.clinical_manifest)
    for cohort,expected in sorted(clinical.items()):
        ids = sorted(expected)
        probs = {n:read_probabilities(args.evaluation_dir/f'{n}_{cohort}_subject_predictions.csv',ids,'subject_id',['P_CN','P_AD']) for n in members['ensemble_multitask']}
        averaged = np.mean(list(probs.values()),axis=0)
        scored = np.asarray([i for i,sid in enumerate(ids) if expected[sid]['label']!=''],dtype=int)
        y = np.asarray([int(expected[ids[i]]['label']) for i in scored])
        if len(y)==0: raise ValueError('Prespecified clinical cohorts require binary-evaluable subjects')
        pred = averaged[scored].argmax(axis=1);ad,cn = y==1,y==0
        summary = {'ensemble':'ensemble_multitask_AD_CN','n_subjects':len(ids),'n_binary_evaluable':len(y),'n_mci_prediction_only':len(ids)-len(y),
                   'class_order':['CN','AD'],'point':a.class_metrics(y,pred,2),
                   'accuracy_exact':a.exact_interval(int(np.sum(y==pred)),len(y)),
                   'AD_sensitivity_exact':a.exact_interval(int(np.sum(pred[ad]==1)),int(ad.sum())),
                   'CN_specificity_exact':a.exact_interval(int(np.sum(pred[cn]==0)),int(cn.sum())),
                   'member_analysis':member_summary(y,{n:p[scored] for n,p in probs.items()},['CN','AD']),
                   'calibration_descriptive':calibration(y,averaged[scored]),
                   'interpretation':'Source-subject exploratory AD/CN ensemble; no old-baseline comparison. MCI excluded from all binary metrics. One external AD cannot establish stable sensitivity.'}
        result['clinical'][cohort] = summary
        outputs[f'ensemble_multitask_AD_CN_{cohort}_subject_predictions.csv'] = [
            {'cohort':expected[sid]['cohort'],'subject_id':sid,'binary_label':expected[sid]['label'],
             'P_CN':float(averaged[i,0]),'P_AD':float(averaged[i,1]),'n_slices_per_member':expected[sid]['n_slices'],
             'interpretation':'MCI prediction-only' if expected[sid]['label']=='' else 'Exploratory AD/CN subject prediction',
             'member_predictions':'|'.join(['CN','AD'][int(probs[n][i].argmax())] for n in members['ensemble_multitask'])}
            for i,sid in enumerate(ids)]
    result['provenance'] = {'primary_input_manifest':base['input_manifest'],'primary_analysis_sha256':a.digest(args.primary_analysis),
                            'primary_analysis_path':str(args.primary_analysis.resolve()),'supplement_freeze_sha256':a.digest(args.supplement_freeze),
                            'code_hashes':code_hashes(),'numpy':np.__version__,
                            'readiness_note':str(HERE/'real_world_readiness.txt')}
    if any(a.digest(record['path'])!=record['sha256'] for record in base['input_manifest']):
        raise ValueError('Frozen primary input changed while supplement was computed')
    return result,outputs


def write_outputs(result,outputs,destination):
    if destination.exists() and any(destination.iterdir()): raise ValueError('Supplement output must be empty')
    destination.mkdir(parents=True,exist_ok=True)
    with (destination/'ensemble_supplement.json').open('x') as file:json.dump(result,file,indent=2,allow_nan=False);file.write('\n')
    with (destination/'ensemble_specifications.json').open('x') as file:json.dump(result['ensemble_specifications'],file,indent=2,allow_nan=False);file.write('\n')
    for name,rows in outputs.items():
        with (destination/name).open('x',newline='') as file:
            writer = csv.DictWriter(file,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__);commands = p.add_subparsers(dest='command',required=True)
    f = commands.add_parser('freeze');f.add_argument('--evaluation-output',type=Path,required=True);f.add_argument('--output',type=Path,required=True)
    run = commands.add_parser('analyze')
    for flag in ['evaluation-dir','prepared','original100-key','clinical-manifest','validation-selection','old-hash-unexposed-manifest','baseline-exposure-audit','primary-analysis','supplement-freeze','output']:
        run.add_argument('--'+flag,type=Path,required=True)
    run.add_argument('--allow-incomplete-readers',action='store_true')
    args = p.parse_args()
    if args.command=='freeze':
        value = freeze(args.evaluation_output,args.output)
        print(json.dumps({'status':'supplement_frozen','output':str(args.output),'code_hashes':value['code_hashes']}))
    else:
        if args.output.exists() and any(args.output.iterdir()):p.error('Supplement output must be empty')
        result,outputs = analyze(args);write_outputs(result,outputs,args.output)
        print(json.dumps({'status':'supplement_complete','output':str(args.output),'stage_ensembles':2,'clinical_ensembles':1,'new_inference':False,'winner_or_promotion':None}))


if __name__=='__main__':main()
