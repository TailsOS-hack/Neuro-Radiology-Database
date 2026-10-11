#!/usr/bin/env python3
"""Synthetic-only verification; never opens project images or real evaluation outputs."""
import argparse
import csv
import json
from pathlib import Path
import tempfile
import unittest
from collections import Counter
from unittest.mock import patch

import numpy as np
import summarize_frozen_evaluation as a


def write_csv(path, rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def fixture(root):
    ev,prepared=root/'evaluation',root/'prepared';ev.mkdir();prepared.mkdir()
    stage=[dict(path=f'synthetic/{i}.png',label=str(i%4),group_id=f'group{i%4}',split='test',baseline_strict_split='test') for i in range(8)]
    write_csv(prepared/'stage_manifest.csv',stage);write_csv(prepared/'same_case_test_intersection.csv',stage)
    (prepared/'split_audit.json').write_text(json.dumps({'manifest_sha256':a.digest(prepared/'stage_manifest.csv'),'same_case_test_intersection_sha256':a.digest(prepared/'same_case_test_intersection.csv')}))
    key=[{'FileName':f'case{i:03}.png','TrueCategory':a.STAGES[i%4]} for i in range(100)]
    write_csv(root/'key.csv',key)
    clinical=[]
    for co,split in [('synthetic_internal','test'),('synthetic_external','external')]:
        for i,diag in enumerate(['CN','AD','MCI']):
            for s in range(2):
                clinical.append({'path':f'{co}/subject{i}/{s}.png','cohort':co,'subject_id':str(i),'group_key':f'{co}__{i}',
                                 'diagnosis':diag,'binary_label':{'CN':'0','AD':'1','MCI':''}[diag],'split':split})
    write_csv(root/'clinical.csv',clinical)
    checkpoints={'old_baseline':{'sha256':'0'*64,'selection_config':{}}}
    for i,(mode,seed) in enumerate(( (m,s) for m in a.PROTOCOL['planned_modes'] for s in a.PROTOCOL['planned_seeds']),1):
        cfg={'mode':mode,'seed':seed,'smoke':False,'test_used_for_selection':False,'stage_manifest_sha256':a.digest(prepared/'stage_manifest.csv'),'clinical_manifest_sha256':a.digest(root/'clinical.csv')}
        checkpoints[f'candidate_{i}']={'sha256':str(i)*64,'selection_config':cfg}
        if mode=='multitask':
            checkpoints[f'candidate_{i}']['clinical_aux_AD_CN']={s+'__'+co:{} for co,s in [('synthetic_internal','test'),('synthetic_external','external')]}
    for n,info in checkpoints.items():
        for cohort,expected in [('internal_same_case',stage),('original100',key)]:
            rows=[]
            for i,r in enumerate(expected):
                label=int(r['label']) if cohort=='internal_same_case' else a.STAGES.index(r['TrueCategory'])
                pred=(label+1)%4 if n=='candidate_1' and i%5==0 else label
                row={'path':r['path'] if cohort=='internal_same_case' else 'data/evaluation/images/'+r['FileName'],
                     'TrueCategory':a.STAGES[label],'ModelPrediction':a.STAGES[pred]}
                row.update({'P_'+cl:.97 if j==pred else .01 for j,cl in enumerate(a.STAGES)})
                if cohort=='original100': row['FileName']=r['FileName']
                rows.append(row)
            write_csv(ev/f'{n}_{cohort}_predictions.csv',rows)
        for co in info.get('clinical_aux_AD_CN',{}):
            source=co.split('__',1)[1]
            write_csv(ev/f'{n}_{co}_subject_predictions.csv',[
                {'cohort':source,'subject_id':f'{source}__{i}','binary_label':label,'P_CN':cn,'P_AD':1-cn,'n_slices':2}
                for i,label,cn in [(0,'0',.9),(1,'1',.1),(2,'',.5)]])
    aligned=[]
    for i,r in enumerate(key):
        row=dict(r)
        for n in checkpoints: row[n]=a.STAGES[(i%4+1)%4] if n=='candidate_1' and i%5==0 else r['TrueCategory']
        row['Reader_A']=r['TrueCategory'] if i%2==0 else a.STAGES[(i%4+1)%4]
        row['Reader_B']=r['TrueCategory']
        aligned.append(row)
    write_csv(ev/'original100_case_aligned_old_candidates_readers.csv',aligned)
    (ev/'evaluation.json').write_text(json.dumps({'checkpoints':checkpoints,'reader_provenance':{'A':[],'B':[]}}))
    records={n:{'checkpoint_sha256':info['sha256'],'mode':info['selection_config']['mode'],'seed':info['selection_config']['seed'],'stage_validation_nll':.2 if info['selection_config']['mode']=='multitask' else .3} for n,info in checkpoints.items() if n!='old_baseline'}
    selection={'frozen_before_heldout_evaluation':True,'candidates':records,'analysis_protocol':a.PROTOCOL,'analysis_script_sha256':a.digest(a.__file__),
               'mode_mean_stage_validation_nll':{'stage_only':float(np.mean([.3]*3)),'multitask':float(np.mean([.2]*3))},'preferred_mode':'multitask','exploratory_representative_candidate':'candidate_4'}
    (root/'selection.json').write_text(json.dumps(selection))
    return argparse.Namespace(evaluation_dir=ev,prepared=prepared,original100_key=root/'key.csv',clinical_manifest=root/'clinical.csv',
                              validation_selection=root/'selection.json',allow_incomplete_readers=False,output=root/'statistics')


def add_sensitivity(args):
    root=args.prepared.parent
    rows=a.read_csv(args.prepared/'same_case_test_intersection.csv',['path'])
    for i,r in enumerate(rows):r['group_id']=f'group{i}'
    mate=dict(rows[0],path='synthetic/old_train_mate.png',baseline_strict_split='train')
    write_csv(args.prepared/'stage_manifest.csv',rows+[mate]);write_csv(args.prepared/'same_case_test_intersection.csv',rows)
    split_audit={'manifest_sha256':a.digest(args.prepared/'stage_manifest.csv'),'same_case_test_intersection_sha256':a.digest(args.prepared/'same_case_test_intersection.csv')}
    (args.prepared/'split_audit.json').write_text(json.dumps(split_audit))
    meta=json.loads((args.evaluation_dir/'evaluation.json').read_text())
    for n,c in meta['checkpoints'].items():
        if n!='old_baseline':c['selection_config']['stage_manifest_sha256']=split_audit['manifest_sha256']
    (args.evaluation_dir/'evaluation.json').write_text(json.dumps(meta))
    args.old_hash_unexposed_manifest=root/'sensitivity.csv';write_csv(args.old_hash_unexposed_manifest,rows[1:])
    def counts(rs):
        return {'rows':len(rs),'unique_hash_groups':len({r['group_id'] for r in rs}),'class_counts':dict(Counter(a.STAGES[int(r['label'])] for r in rs))}
    audit={'status':'prespecified_before_heldout_model_evaluation','sources':{k:{'sha256':a.digest(args.prepared/f)} for k,f in [('stage_manifest','stage_manifest.csv'),('same_case_test_intersection','same_case_test_intersection.csv'),('split_audit','split_audit.json')]},
           'primary':counts(rows),'subset':dict(counts(rows[1:]),sha256=a.digest(args.old_hash_unexposed_manifest)),
           'excluded':dict(counts(rows[:1]),groups=[{'group_id':'group0','intersection_rows':1,'old_strict_splits_in_prepared_group':['test','train'],'old_strict_train_rows_in_prepared_group':1,'old_strict_validation_rows_in_prepared_group':0}])}
    args.baseline_exposure_audit=root/'sensitivity_audit.json';args.baseline_exposure_audit.write_text(json.dumps(audit))
    selection=json.loads(args.validation_selection.read_text())
    selection['old_hash_unexposed_sensitivity']={'manifest_sha256':a.digest(args.old_hash_unexposed_manifest),'audit_sha256':a.digest(args.baseline_exposure_audit)}
    args.validation_selection.write_text(json.dumps(selection))
    return args


class SyntheticAnalysisTests(unittest.TestCase):
    def test_freeze_selection_uses_only_validation_and_rejects_late_freeze(self):
        import torch
        from freeze_validation_selection import freeze
        with tempfile.TemporaryDirectory(prefix='kevin-selection-synthetic-') as tmp:
            root=Path(tmp);paths=[]
            for i,(mode,seed) in enumerate(( (m,s) for m in a.PROTOCOL['planned_modes'] for s in a.PROTOCOL['planned_seeds']),1):
                loss=.3 if mode=='stage_only' else .2
                ck={'arch':'mobilenet_v3_large','class_names':a.STAGES,'epoch':20,'history':[{'epoch':20}],
                    'stage_validation':{'loss':loss},'best_val_loss':loss,
                    'config':{'mode':mode,'seed':seed,'test_used_for_selection':False,'stage_manifest_sha256':'a'*64,'clinical_manifest_sha256':'b'*64}}
                path=root/f'synthetic_ck{i}.pt';torch.save(ck,path);paths.append(path)
            result=freeze(paths,root/'eval')
            self.assertTrue(result['frozen_before_heldout_evaluation'])
            self.assertEqual(result['preferred_mode'],'multitask')
            self.assertEqual(result['exploratory_representative_candidate'],'candidate_4')
            self.assertEqual(len(result['candidates']),6)
            subset=root/'subset.csv';subset.write_text('synthetic pinned manifest\n')
            audit=root/'subset_audit.json';audit.write_text(json.dumps({'status':'prespecified_before_heldout_model_evaluation','subset':{'sha256':a.digest(subset)}}))
            pinned=freeze(paths,root/'eval',subset,audit)
            self.assertEqual(pinned['old_hash_unexposed_sensitivity']['manifest_sha256'],a.digest(subset))
            with self.assertRaisesRegex(ValueError,'supplied together'):freeze(paths,root/'eval',subset,None)
            with self.assertRaisesRegex(ValueError,'six distinct'):freeze(paths[:-1],root/'eval')
            ck=torch.load(paths[0],weights_only=False);ck['smoke_only']=True;torch.save(ck,paths[0])
            with self.assertRaisesRegex(ValueError,'smoke'):freeze(paths,root/'eval')
            (root/'eval').mkdir();(root/'eval'/'anything.txt').write_text('synthetic marker')
            with self.assertRaisesRegex(ValueError,'before held-out'):freeze(paths,root/'eval')

    def test_exact_extreme_intervals_are_not_certain(self):
        self.assertAlmostEqual(a.exact_interval(1,1)['ci95'][0],.025)
        self.assertEqual(a.exact_interval(1,1)['ci95'][1],1)
        self.assertAlmostEqual(a.exact_interval(0,1)['ci95'][1],.975)
        self.assertIsNone(a.exact_interval(0,0)['ci95'])

    def test_metrics_hand_calculated_and_missing_class(self):
        m=a.class_metrics(np.array([0,0,1,1]),np.array([0,1,1,1]),2)
        self.assertEqual(m['accuracy'],.75);self.assertEqual(m['balanced_accuracy'],.75)
        self.assertAlmostEqual(m['macro_f1'],(.6666666666666666+.8)/2)
        self.assertIsNone(a.class_metrics(np.array([0]),np.array([0]),2)['balanced_accuracy'])

    def test_bootstrap_is_paired_deterministic_and_clustered(self):
        y=np.array([0,0,1,1]);p={'new':[0,1,1,1],'old':[0,1,1,1]}
        x=a.bootstrap_analysis(y,p,['a','a','b','b'],'SYNTHETIC',[('new','old')],['CN','AD'])
        z=a.bootstrap_analysis(y,p,['a','a','b','b'],'SYNTHETIC',[('new','old')],['CN','AD'])
        self.assertEqual(x,z);self.assertEqual(x['n_resampling_units'],2)
        self.assertEqual(x['paired_comparisons']['new_minus_old']['difference']['accuracy']['ci95'],[0.,0.])
        self.assertTrue(x['paired_comparisons']['new_minus_old']['difference']['accuracy']['degenerate'])
        self.assertIsNone(x['models']['new']['bootstrap']['balanced_accuracy']['ci95'])

    def test_full_six_candidate_schema_and_guards(self):
        with tempfile.TemporaryDirectory(prefix='kevin-statistics-synthetic-') as tmp:
            args=fixture(Path(tmp));result=a.analyze(args)
            self.assertEqual(len(result['stage']['internal_same_case']['paired_comparisons']),9)
            self.assertEqual(result['stage']['internal_same_case']['n_resampling_units'],4)
            self.assertEqual(result['stage']['original100']['n'],100)
            self.assertEqual(result['stage']['original100']['primary_condition_summaries']['multitask']['n_seeds'],3)
            self.assertEqual(result['stage']['original100']['paired_comparisons']['candidate_1_minus_old_baseline']['difference']['accuracy']['estimate'],-.19999999999999996)
            self.assertEqual(result['readers']['fixed_panel_summary']['common_complete_cases'],100)
            clinical=result['clinical']['external__synthetic_external']['candidate_4']
            self.assertEqual(clinical['n_mci_prediction_only'],1)
            self.assertAlmostEqual(clinical['AD_sensitivity_exact']['ci95'][0],.025)
            a.write_outputs(result,args.output)
            self.assertTrue((args.output/'uncertainty_analysis.json').exists())
            with self.assertRaisesRegex(ValueError,'Output must be empty'):a.write_outputs(result,args.output)
            # Fail on duplicate and incomplete case sets before any statistical computation.
            file=args.evaluation_dir/'candidate_1_original100_predictions.csv'
            rows=a.read_csv(file,['FileName'])
            write_csv(file,rows[:-1])
            with self.assertRaisesRegex(ValueError,'Case mismatch'):
                a.load_stage_predictions(file,{r['FileName']:r['TrueCategory'] for r in rows},'FileName')
            write_csv(file,rows+[rows[0]])
            with self.assertRaisesRegex(ValueError,'duplicate'):
                a.load_stage_predictions(file,{r['FileName']:r['TrueCategory'] for r in rows},'FileName')
            write_csv(file,rows)
            selection=json.loads(args.validation_selection.read_text());selection['preferred_mode']='stage_only';args.validation_selection.write_text(json.dumps(selection))
            with self.assertRaisesRegex(ValueError,'Validation preference'):a.analyze(args)

    def test_identity_label_probability_guards(self):
        with tempfile.TemporaryDirectory(prefix='kevin-statistics-synthetic-') as tmp:
            args=fixture(Path(tmp));file=args.evaluation_dir/'candidate_1_original100_predictions.csv'
            rows=a.read_csv(file,['FileName']);expected={r['FileName']:r['TrueCategory'] for r in rows}
            rows[0]['TrueCategory']=a.STAGES[1];write_csv(file,rows)
            with self.assertRaisesRegex(ValueError,'Label mismatch'):a.load_stage_predictions(file,expected,'FileName')
            rows[0]['TrueCategory']=expected[rows[0]['FileName']];rows[0]['P_'+a.STAGES[0]]='nan';write_csv(file,rows)
            with self.assertRaisesRegex(ValueError,'Probabilities'):a.load_stage_predictions(file,expected,'FileName')
            file=args.evaluation_dir/'candidate_4_external__synthetic_external_subject_predictions.csv'
            rows=a.read_csv(file,['subject_id']);rows[0]['binary_label']='1';write_csv(file,rows)
            with self.assertRaisesRegex(ValueError,'Clinical identity'):
                a.clinical_analysis(file,a.clinical_subjects(args.clinical_manifest)['external__synthetic_external'])

    def test_reader_missingness_requires_explicit_paired_complete_cases(self):
        with tempfile.TemporaryDirectory(prefix='kevin-statistics-synthetic-') as tmp:
            args=fixture(Path(tmp));file=args.evaluation_dir/'original100_case_aligned_old_candidates_readers.csv'
            rows=a.read_csv(file,['FileName']);rows[0]['Reader_A']='Missing';write_csv(file,rows)
            with self.assertRaisesRegex(ValueError,'allow-incomplete-readers'):a.analyze(args)
            args.allow_incomplete_readers=True;result=a.analyze(args)
            self.assertEqual(result['readers']['paired_comparisons']['Reader_A']['n'],99)
            self.assertEqual(result['readers']['paired_comparisons']['Reader_B']['n'],100)
            self.assertEqual(result['readers']['fixed_panel_summary']['common_complete_cases'],99)

    def test_prespecified_sensitivity_uses_exact_aligned_subset_and_preserves_primary(self):
        with tempfile.TemporaryDirectory(prefix='kevin-sensitivity-synthetic-') as tmp:
            args=add_sensitivity(fixture(Path(tmp)));result=a.analyze(args)
            self.assertEqual(result['stage']['internal_same_case']['n'],8)
            self.assertEqual(result['baseline_group_exposure']['images_with_old_train_group_mate'],1)
            subset=result['stage']['old_hash_group_unexposed_sensitivity']
            self.assertEqual(subset['n'],7);self.assertEqual(subset['n_resampling_units'],7)
            self.assertEqual(len(subset['paired_comparisons']),9)
            self.assertAlmostEqual(subset['models']['candidate_1']['point']['accuracy'],6/7)
            self.assertAlmostEqual(result['stage']['internal_same_case']['models']['candidate_1']['point']['accuracy'],6/8)
            self.assertFalse(result['sensitivity_provenance']['extra_model_inference'])
            rows=a.read_csv(args.old_hash_unexposed_manifest,['path']);write_csv(args.old_hash_unexposed_manifest,rows[:-1])
            with self.assertRaisesRegex(ValueError,'not pinned'):a.analyze(args)
            stage=a.index_rows(a.read_csv(args.prepared/'stage_manifest.csv',['path']),'path','stage')
            primary=a.index_rows(a.read_csv(args.prepared/'same_case_test_intersection.csv',['path']),'path','primary')
            with self.assertRaisesRegex(ValueError,'exact subset'):
                a.validate_sensitivity_manifest(args.old_hash_unexposed_manifest,args.baseline_exposure_audit,args.prepared/'stage_manifest.csv',args.prepared/'same_case_test_intersection.csv',args.prepared/'split_audit.json',stage,primary)


if __name__=='__main__':unittest.main(verbosity=2)
