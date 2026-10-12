"""Synthetic packaging tests only; no real predictions, models or MRI are read."""
import csv
import importlib.util
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

spec=importlib.util.spec_from_file_location('artifact_export',Path(__file__).with_name('export_artifacts.py'))
e=importlib.util.module_from_spec(spec);spec.loader.exec_module(e)


def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(value if isinstance(value,bytes) else e.encoded(value))


def table(path,rows):
    stream=io.StringIO();fields=list(dict.fromkeys(k for r in rows for k in r));w=csv.DictWriter(stream,fieldnames=fields);w.writeheader();w.writerows(rows);put(path,stream.getvalue().encode())


def seal(folder,marker,status):
    names={str(p.relative_to(folder)):e.sha(p) for p in sorted(folder.rglob('*')) if p.is_file() and p.name not in {marker,'SHA256SUMS.json'}}
    put(folder/'SHA256SUMS.json',names);put(folder/marker,dict(status=status,payloads=len(names),manifest_sha256=e.sha(folder/'SHA256SUMS.json')))


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);self.stage=self.root/'stage';self.clinical=self.root/'clinical';self.public=self.root/'publication';self.repo=self.root/'repo'
        self.public.mkdir();self.stage.mkdir();self.clinical.mkdir();self.historical=self.repo/e.HISTORICAL
        self.stack=[]
        for name,value in [('HERE',self.public),('REPO',self.repo)]:
            p=patch.object(e,name,value);p.start();self.stack.append(p)
        sf=dict(status='frozen_before_any_round2_evaluation',all_three_runs={})
        cf=dict(status='frozen_before_clinical_heldout_inference',heldout_outcomes_used_for_selection=False,selections={})
        for seed in e.STAGE_SEEDS:
            folder=f'stage_only_seed{seed}';history=[dict(epoch=0,clean_nll=.4,selected=True),dict(epoch=1,clean_nll=.3,selected=True,train_raw_nll=.5)]
            put(self.stage/folder/'history.json',history);table(self.stage/folder/'history.csv',history)
            sf['all_three_runs'][f'round2_seed{seed}']=dict(checkpoint=folder+'/best.pt',history_sha256=e.sha(self.stage/folder/'history.json'))
            for ext in ['png','pdf']:put(self.stage/folder/f'raw_curves.{ext}',f'synthetic {folder} {ext}'.encode())
        for seed in e.CLINICAL_SEEDS:
            folder=f'clinical_seed{seed}';history=[dict(epoch=0,train_aug_loss=None,val_loss=.6,selected=True),dict(epoch=1,train_aug_loss=.5,val_loss=.4,selected=True)]
            put(self.clinical/folder/'history.json',history);table(self.clinical/folder/'history.csv',history)
            cf['selections'][folder]=dict(checkpoint_path=folder+'/best.pt',history_sha256=e.sha(self.clinical/folder/'history.json'))
            for ext in ['png','pdf']:put(self.clinical/folder/f'raw_patient_curves.{ext}',f'synthetic {folder} {ext}'.encode())
        put(self.stage/'round2_evaluation_freeze.json',sf);put(self.clinical/'clinical_evaluation_freeze.json',cf)
        se=self.stage/'frozen_round2_evaluation';ce=self.clinical/'clinical_evaluation'
        put(se/'evaluation.json',dict(status='complete',freeze=sf,selection_freeze_sha256=e.sha(self.stage/'round2_evaluation_freeze.json')))
        for name in ['original100_case_aligned_models_readers.csv']+[n+'_original100_predictions.csv' for n in ['old_baseline','round1_c3']+[f'round2_seed{s}' for s in e.STAGE_SEEDS]]:table(se/name,[dict(FileName=f'synthetic_case{i}') for i in range(100)])
        seal(se,'evaluation_complete.json','complete_verified')
        put(ce/'evaluation.json',dict(status='complete',freeze=cf,freeze_sha256=e.sha(self.clinical/'clinical_evaluation_freeze.json'),fresh_binary_patients=19,mci_not_scored=True,total_patient_scenario_records=3360,metric_records=168))
        for name in ['patient_predictions.json','metrics.json','model_state_verification.json']:put(ce/name,[dict(private_fixture_id='PRIVATE_DO_NOT_EXPORT')])
        seal(ce,'evaluation_complete.json','complete_verified')
        self.sr=self.stage/'report';self.cr=self.clinical/'report';self.attr=self.stage/'attribution'
        for folder,count,prefix in [(self.sr,28,'stage'),(self.cr,19,'clinical'),(self.attr,2,'attribution')]:
            for i in range(count):
                for ext in ['png','pdf']:put(folder/f'{prefix}_{i:02d}.{ext}',f'synthetic {prefix} {i} {ext}'.encode())
        put(self.sr/'README.md',b'# Synthetic stage\n\n[Plot](stage_00.pdf)\n');put(self.sr/'analysis.json',dict(freeze_sha256=e.sha(self.stage/'round2_evaluation_freeze.json'),evaluation_sha256=e.sha(se/'evaluation.json')))
        table(self.sr/'probability_curve_metrics.csv',[dict(model='synthetic',accuracy=.5)]);put(self.sr/'figure_inputs.json',dict(stress=[],corner=[]))
        seal(self.sr,'report_complete.json',e.NUMERICAL)
        put(self.cr/'README.md',b'# Synthetic clinical\n\n[Plot](clinical_00.pdf)\n')
        self.private_analysis=dict(status='complete_verified_predictions_only',fresh_external_binary_n=19,mci_scored=False,freeze_sha256=e.sha(self.clinical/'clinical_evaluation_freeze.json'),cohorts={'synthetic':{'patient_ids':['PRIVATE_DO_NOT_EXPORT'],'models':{'synthetic':{'n':19}},'nested':[{'patient_ids':['PRIVATE_NESTED_ID'],'accuracy':.5}]}})
        put(self.cr/'analysis.json',self.private_analysis)
        table(self.cr/'all_model_scenario_metrics.csv',[dict(model=f'm{i}',scenario=f's{j}',cohort=f'c{k}',accuracy=.5) for i in range(8) for j in range(7) for k in range(3)])
        table(self.cr/'mci_prediction_only.csv',[dict(group_key='PRIVATE_MCI_ID',p_ad=.5)])
        seal(self.cr,'report_complete.json',e.NUMERICAL)
        cases=[dict(case_id=f'case{i:03d}',path=f'data/test/image{i}.jpg',label='class',role='reused_original100' if i<100 else 'historical_class_balanced_controls',image_sha256='a'*64,cam_valid=True) for i in range(115)]
        table(self.attr/'per_case.csv',cases)
        history=[]
        for row in cases:
            put(self.attr/'cases'/f'{row["case_id"]}.png',('synthetic panel '+row['case_id']).encode())
            put(self.attr/'maps'/f'round2_{row["case_id"]}.npy',b'synthetic new native array')
            for model in ['old_baseline','candidate_3']:
                history.append({**row,'model':model});put(self.historical/'maps'/f'{model}_{row["case_id"]}.npy',b'synthetic old native array')
        table(self.historical/'per_case.csv',history)
        put(self.historical/'SHA256SUMS.json',{str(p.relative_to(self.historical)):e.sha(p) for p in self.historical.rglob('*') if p.is_file()})
        mh=e.sha(self.historical/'SHA256SUMS.json');ch=e.sha(self.historical/'per_case.csv')
        for name,value in [('HISTORICAL_MANIFEST_SHA',mh),('HISTORICAL_CASES_SHA',ch)]:
            p=patch.object(e,name,value);p.start();self.stack.append(p)
        put(self.attr/'verification.json',dict(case_count=115,all_logits_modes_gradients_unchanged=True,original_checkpoint_state_restored=True,cloud_probability_or_argmax_failures=[],historical_sources=dict(case_inventory_sha256=ch,baseline_payload_manifest_sha256=mh)))
        for name in ['native_map_mass.csv','randomization.csv']:table(self.attr/name,[dict(case_id='synthetic')])
        put(self.attr/'gallery.html',b'<h1>Synthetic</h1><img src="cases/case000.png">')
        seal(self.attr,'attribution_complete.json',e.NUMERICAL);self.qa()

    def tearDown(self):
        for p in reversed(self.stack):p.stop()
        self.tmp.cleanup()

    def qa(self):
        for suite in [self.stage,self.clinical]:put(suite/'publication_visual_qa.json',dict(status=e.QA_STATUS,real_data=True,reviewed_files={str(p.relative_to(suite)):e.sha(p) for p in suite.rglob('*') if p.suffix in {'.png','.pdf'}}))

    def export(self,name='artifacts_test'):return e.export(self.stage,self.clinical,self.public/name)

    def test_complete_export_retains_sources_but_removes_private_data(self):
        before=e.sha(self.cr/'analysis.json');self.export();out=self.public/'artifacts_test';manifest=e.read(out/'EXPORT_MANIFEST.json')
        self.assertEqual(e.sha(self.cr/'analysis.json'),before)
        public=e.read(out/'clinical/report/analysis.json');self.assertNotIn('patient_ids',json.dumps(public));self.assertEqual(public['cohorts']['synthetic']['models']['synthetic']['n'],19)
        self.assertEqual(len(list((out/'stage/attribution/baseline_maps').glob('*.npy'))),230)
        self.assertEqual(len(list((out/'stage/attribution/cases').glob('*.png'))),115)
        for record in manifest['files']:
            self.assertEqual(e.sha(out/record['path']),record['output_sha256'])
            self.assertNotIn('PRIVATE_', (out/record['path']).read_bytes().decode(errors='ignore'))
        self.assertFalse((out/'clinical/report/mci_prediction_only.csv').exists())
        self.assertEqual(manifest['source_files_unchanged'],True)
        with self.assertRaises(ValueError):self.export()

    def test_absent_or_stale_visual_qa_refuses_without_output(self):
        q=e.read(self.stage/'publication_visual_qa.json');q['reviewed_files'].pop('attribution/cases/case000.png');put(self.stage/'publication_visual_qa.json',q)
        with self.assertRaisesRegex(ValueError,'lacks exact reviewed SHA'):self.export()
        self.assertFalse((self.public/'artifacts_test').exists())

    def test_research_weight_metadata_requires_matching_six_frozen_selections(self):
        folder=self.repo/'publication';manifest=self.repo/'models/kevin_round2_20261011/CHECKPOINTS.json'
        metadata=dict(frozen_selection_sha256={'stage':e.sha(self.stage/'round2_evaluation_freeze.json'),'clinical':e.sha(self.clinical/'clinical_evaluation_freeze.json')},checkpoints=[dict(kind=k,seed=s) for k,seeds in [('stage',e.STAGE_SEEDS),('clinical',e.CLINICAL_SEEDS)] for s in seeds])
        put(manifest,metadata)
        receipt=dict(status='six_fixed_research_exports_verified',checkpoint_manifest_sha256=e.sha(manifest),production_baseline_unchanged=True,real_image_inference_performed=False)
        put(folder/'research_weights_export_verification.json',receipt)
        with patch.object(e,'HERE',folder):
            plan=e.Plan();self.assertTrue(e.add_research_weight_metadata(plan,self.stage,self.clinical));self.assertEqual(len(plan.items),2)
            metadata['frozen_selection_sha256']['clinical']='0'*64;put(manifest,metadata);receipt['checkpoint_manifest_sha256']=e.sha(manifest);put(folder/'research_weights_export_verification.json',receipt)
            with self.assertRaisesRegex(ValueError,'different frozen selections'):e.add_research_weight_metadata(e.Plan(),self.stage,self.clinical)

    def test_numerical_byte_change_refuses_without_output(self):
        put(self.cr/'analysis.json',dict(changed=True))
        with self.assertRaisesRegex(ValueError,'Numerical payload changed'):self.export()
        self.assertFalse((self.public/'artifacts_test').exists())

    def test_omitted_private_input_inventory_is_not_treated_as_verified(self):
        ce=self.clinical/'clinical_evaluation';inventory=e.read(ce/'SHA256SUMS.json');inventory.pop('patient_predictions.json');put(ce/'SHA256SUMS.json',inventory)
        put(ce/'evaluation_complete.json',dict(status='complete_verified',payloads=3,manifest_sha256=e.sha(ce/'SHA256SUMS.json')))
        with self.assertRaisesRegex(ValueError,'inventory coverage'):self.export()

    def test_history_csv_mismatch_and_added_patient_fields_refuse(self):
        path=self.clinical/'clinical_seed20261021/history.csv';original=path.read_bytes();path.write_bytes(original.replace(b'0.6',b'0.7'))
        with self.assertRaisesRegex(ValueError,'JSON/CSV disagree'):self.export()
        path.write_bytes(original);a=e.read(self.cr/'analysis.json');a['subject_id']='PRIVATE_ID';put(self.cr/'analysis.json',a);seal(self.cr,'report_complete.json',e.NUMERICAL)
        with self.assertRaisesRegex(ValueError,'Private clinical record key'):self.export()

    def test_symlink_and_absolute_path_refused(self):
        path=self.sr/'README.md';path.unlink();outside=self.root/'outside.md';put(outside,b'test');path.symlink_to(outside)
        with self.assertRaisesRegex(ValueError,'symlinks'):self.export()
        with self.assertRaises(ValueError):e.portability('python /Users/example/private/path')

    def test_broken_link_refuses_without_output(self):
        put(self.cr/'README.md',b'[bad](mci_prediction_only.csv)');seal(self.cr,'report_complete.json',e.NUMERICAL)
        with self.assertRaisesRegex(ValueError,'Broken public artifact link'):self.export()
        self.assertFalse((self.public/'artifacts_test').exists())

    def test_atlas_cloud_paths_become_portable_and_all_sources_match(self):
        atlas=self.stage/'atlas';records=[]
        for folder in [self.sr,self.attr]:
            for path in sorted(folder.glob('*.pdf')):records.append(dict(page=len(records)+1,group='synthetic',title=path.stem,source='/workspace/private/'+path.name,source_sha256=e.sha(path)))
        table(atlas/'atlas_index.csv',records);put(atlas/'Round2_Graphs_Atlas.pdf',b'synthetic 30 page atlas')
        put(atlas/'atlas_verification.json',dict(pages=30,source_pdf_count=30,atlas_sha256=e.sha(atlas/'Round2_Graphs_Atlas.pdf')));self.qa();self.export()
        exported=self.public/'artifacts_test/stage/atlas/atlas_index.csv'
        self.assertNotIn('/workspace/',exported.read_text())
        for row in e.read_csv(exported):self.assertEqual(e.sha(exported.parent/row['source']),row['source_sha256'])

    def test_recovery_receipt_is_public_but_failure_log_is_private(self):
        put(self.attr/'recovery_receipt.json',dict(status='postprocessing_only_serialization_recovery',source_directory='failed_attribution',output_directory='attribution'))
        put(self.attr/'serialization_failure.log',b'PRIVATE_RUNTIME_DETAILS');seal(self.attr,'attribution_complete.json',e.NUMERICAL)
        self.export();out=self.public/'artifacts_test/stage/attribution'
        self.assertTrue((out/'recovery_receipt.json').is_file());self.assertFalse((out/'serialization_failure.log').exists())

    def test_blocked_attribution_cannot_be_published(self):
        marker=self.attr/'attribution_complete.json';v=e.read(marker);v['status']='blocked_cloud_probability_replay_review';put(marker,v)
        with self.assertRaisesRegex(ValueError,'Incomplete or unverified'):self.export()
        self.assertFalse((self.public/'artifacts_test').exists())

    def test_stage_only_release_needs_no_clinical_or_attribution_files(self):
        shutil.rmtree(self.clinical);shutil.rmtree(self.attr)
        e.export(self.stage,None,self.public/'artifacts_stage_only',stage_only=True)
        out=self.public/'artifacts_stage_only';manifest=e.read(out/'EXPORT_MANIFEST.json')
        self.assertEqual(manifest['export_scope'],'stage_quantitative_only')
        self.assertEqual(manifest['attribution_replay_status'],'not_part_of_export')
        self.assertEqual(len(list((out/'stage/report').glob('*.pdf'))),28)
        self.assertFalse((out/'clinical').exists());self.assertFalse((out/'stage/attribution').exists())

    def diagnostic_fixture(self):
        verification=e.read(self.attr/'verification.json');verification.update(cloud_probability_tolerance=1e-4,cloud_probability_or_argmax_failures=[dict(case_id='case005',max_probability_difference=.0002,argmax_agreement=True),dict(case_id='case010',max_probability_difference=.0003,argmax_agreement=True)])
        put(self.attr/'verification.json',verification);put(self.attr/'recovery_receipt.json',dict(status='postprocessing_only_serialization_recovery'))
        seal(self.attr,'attribution_complete.json',e.BLOCKED_STATUS)
        derived=self.stage/'diagnostic_appendix';shutil.copytree(self.attr,derived)
        (derived/'attribution_complete.json').rename(derived/'source_attribution_complete.json');(derived/'SHA256SUMS.json').rename(derived/'source_SHA256SUMS.json')
        visual={}
        for path in derived.rglob('*'):
            if path.suffix in {'.png','.pdf'}:path.write_bytes(b'DIAGNOSTIC REPLAY BLOCKED\n'+path.read_bytes());visual[str(path.relative_to(derived))]=e.sha(path)
        put(derived/'gallery.html',b'<h1>Diagnostic: probability replay blocked</h1><img src="cases/case000.png">')
        put(derived/'ATTRIBUTION_REPLAY_LIMITATION.md',b'# Synthetic diagnostic replay limitation\n')
        put(derived/'diagnostic_provenance.json',dict(original_marker_sha256=e.sha(derived/'source_attribution_complete.json'),warning_visuals_sha256=visual))
        seal(derived,'diagnostic_appendix_complete.json',e.DIAGNOSTIC_STATUS);self.qa()
        return derived

    def test_explicit_diagnostics_preserve_blocked_status_and_warning(self):
        derived=self.diagnostic_fixture()
        with self.assertRaises(ValueError):self.export()
        e.export(self.stage,self.clinical,self.public/'artifacts_diagnostics',attribution=derived,diagnostic=True)
        out=self.public/'artifacts_diagnostics';manifest=e.read(out/'EXPORT_MANIFEST.json')
        self.assertEqual(manifest['attribution_replay_status'],e.BLOCKED_STATUS)
        self.assertEqual(e.read(out/'stage/attribution_diagnostics/source_attribution_complete.json')['status'],e.BLOCKED_STATUS)
        self.assertIn('2/100', (out/'README.md').read_text());self.assertIn('remains BLOCKED',(out/'README.md').read_text())
        self.assertFalse((out/'stage/attribution').exists())

    def test_diagnostic_original_marker_tamper_is_rejected(self):
        derived=self.diagnostic_fixture();source_marker=derived/'source_attribution_complete.json';v=e.read(source_marker);v['status']=e.NUMERICAL;put(source_marker,v);seal(derived,'diagnostic_appendix_complete.json',e.DIAGNOSTIC_STATUS)
        with self.assertRaisesRegex(ValueError,'Original blocked'):e.export(self.stage,self.clinical,self.public/'artifacts_bad',attribution=derived,diagnostic=True)

    def test_diagnostic_missing_warning_coverage_is_rejected(self):
        derived=self.diagnostic_fixture();path=derived/'diagnostic_provenance.json';v=e.read(path);v['warning_visuals_sha256'].pop('cases/case000.png');put(path,v);seal(derived,'diagnostic_appendix_complete.json',e.DIAGNOSTIC_STATUS)
        with self.assertRaisesRegex(ValueError,'warning-bearing visual coverage'):e.export(self.stage,self.clinical,self.public/'artifacts_bad',attribution=derived,diagnostic=True)

    def test_diagnostic_atlas_schema_is_portable_and_stays_blocked(self):
        derived=self.diagnostic_fixture();atlas=self.stage/'diagnostic_atlas';rows=[]
        for folder in [self.sr,derived]:
            for path in sorted(folder.glob('*.pdf')):rows.append(dict(page=len(rows)+1,group='synthetic',title=path.stem,source_directory=folder.name,source_file=path.name,source_sha256=e.sha(path)))
        table(atlas/'atlas_index.csv',rows);put(atlas/'Round2_Diagnostic_Graphs_Atlas.pdf',b'synthetic flagged atlas')
        put(atlas/'diagnostic_atlas_verification.json',dict(status='complete_diagnostic_atlas_with_blocked_replay',attribution_replay_status=e.BLOCKED_STATUS,pages=30,quantitative_pages=28,diagnostic_pages=2,atlas_sha256=e.sha(atlas/'Round2_Diagnostic_Graphs_Atlas.pdf'),diagnostic_manifest_sha256=e.sha(derived/'SHA256SUMS.json'),report_manifest_sha256=e.sha(self.sr/'SHA256SUMS.json')))
        self.qa();e.export(self.stage,self.clinical,self.public/'artifacts_diagnostic_atlas',attribution=derived,diagnostic=True)
        index=self.public/'artifacts_diagnostic_atlas/stage/atlas/atlas_index.csv'
        for row in e.read_csv(index):self.assertEqual(e.sha(index.parent/row['source']),row['source_sha256'])
        self.assertIn('flagged diagnostic attribution',(self.public/'artifacts_diagnostic_atlas/README.md').read_text())


if __name__=='__main__':unittest.main()
