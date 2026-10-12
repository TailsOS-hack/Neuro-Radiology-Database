#!/usr/bin/env python3
"""Synthetic saved-artifact recovery tests; no real images, model or inference."""
import copy,json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from PIL import Image
import recover_attribution_serialization as r


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='attribution_recovery_synthetic_');self.suite=Path(self.tmp.name);self.source=self.suite/'failed';self.output=self.suite/'recovered';self.source.mkdir();self.evaluation=self.suite/'evaluation';self.evaluation.mkdir()
        self.chosen='round2_seed20261011';self.cp=self.suite/'synthetic_checkpoint.pt';self.cp.write_bytes(b'not a model; never loaded');self.cps={self.chosen:self.cp};self.freeze={'representative':self.chosen};r.p.dump(self.suite/'round2_evaluation_freeze.json',self.freeze)
        self.provenance={'case_inventory_sha256':'synthetic','baseline_payload_manifest_sha256':'synthetic'};self.cases=[];self.baseline={};self.rows=[];self.mass=[];self.random=[];cloud=[]
        for d in ['maps','cases','randomized_maps']:(self.source/d).mkdir()
        self.native=(np.arange(49,dtype=np.float32).reshape(7,7)/48)
        regions=r.p.corner.native_regions()
        for i in range(115):
            key=f'case{i:03d}';role='reused_original100' if i<100 else 'historical_class_balanced_controls' if i<108 else 'historical_gallery';label=r.p.stress.STAGES[i%4];case=dict(case_id=key,path=f'synthetic/{key}.png',label=label,role=role,image_sha256='synthetic_pixels');self.cases.append(case)
            for name in ['old_baseline','round1_c3']:self.baseline[key,name]=(dict(**case,prediction=label),self.native.copy())
            np.save(self.source/f'maps/{self.chosen}_{key}.npy',self.native);Image.new('RGB',(1680,328),'white').save(self.source/f'cases/{key}.png')
            prob=np.ones(4)*.05;prob[i%4]=.85;pp={'P_'+cl:float(v) for cl,v in zip(r.p.stress.STAGES,prob)}
            self.rows.append(dict(**case,model=self.chosen,prediction=label,confidence=.85,cam_valid=True,prediction_and_logits_unchanged=True,cloud_max_probability_difference=0. if i<100 else None,cloud_argmax_agreement=True if i<100 else None,**pp))
            if i<100:cloud.append(dict(path=case['path'],TrueCategory=label,ModelPrediction=label,**pp))
            for name in ['old_baseline','round1_c3','round2_representative']:
                norm=self.native/self.native.sum();self.mass.append(dict(case_id=key,role=role,model=name,valid=True,**{region+'_mass':float(norm[mask].sum()) for region,mask in regions.items()},peak_in_any_corner=True))
            if role=='historical_class_balanced_controls':
                for seed in [42,43,44]:
                    for stage in ['head','head_final','full']:
                        failed=i==100 and seed==42 and stage=='head';arr=self.native[::-1].copy();rho,iou=r.map_similarity(self.native,arr)
                        if not failed:np.save(self.source/f'randomized_maps/{key}_{seed}_{stage}.npy',arr)
                        self.random.append(dict(case_id=key,seed=seed,stage=stage,target=label,status='flat_or_failed' if failed else 'valid',spearman=None if failed else rho,top20_iou=None if failed else iou))
        self.protocol=dict(representative=self.chosen,checkpoint_sha256=r.p.sha(self.cp),freeze_sha256=r.p.sha(self.suite/'round2_evaluation_freeze.json'),historical_sources=self.provenance,class_target='Each model own unmodified argmax; new target fixed through parameter randomization',clean_probability_tolerance=1e-4,randomization_seeds=[42,43,44],randomization_stages=['head','head_final','full'],case_count=115,prior_benchmark_known=True,clinical_anatomical_validation=False,source_code_sha256=r.p.code_hashes())
        r.p.dump(self.source/'protocol_before_attribution.json',self.protocol);r.p.stress.csv_write(self.evaluation/f'{self.chosen}_original100_predictions.csv',cloud);self.write_tables()
        (self.source/'gallery.html').write_text(self.chosen+''.join(f'<img src="cases/{c["case_id"]}.png">' for c in self.cases))
        for stem in r.SUMMARY_STEMS:
            Image.new('RGB',(600,600),'white').save(self.source/f'{stem}.png');(self.source/f'{stem}.pdf').write_bytes(b'%PDF-1.4\n1 0 obj <</Type /Page /Parent 2 0 R>> endobj\n%%EOF\n')
        self.log=self.suite/'synthetic_failure.log';self.log.write_text('Traceback (most recent call last):\n  File \"synthetic/round2_attribution.py\", line 123, in main\n    p.dump(out/\'verification.json\', synthetic_integer_count)\nTypeError: Object of type int64 is not JSON serializable\n');self.trace_sha=r.p.sha(self.log);self.args=SimpleNamespace(data_root=r.p.ROOT,suite=self.suite,source=self.source,output=self.output,failure_log=self.log)

    def tearDown(self):self.tmp.cleanup()
    def write_tables(self):
        for name,rows in [('per_case.csv',self.rows),('native_map_mass.csv',self.mass),('randomization.csv',self.random)]:r.p.stress.csv_write(self.source/name,rows)
    def run_recovery(self):
        with patch.object(r,'TRACE_SHA',self.trace_sha),patch.object(r.report,'verified_outputs',return_value=(self.evaluation,{})),patch.object(r.p,'load_frozen',return_value=(self.freeze,self.cps)),patch.object(r.a,'verified_cases',return_value=(self.cases,self.baseline,self.provenance)):
            return r.recover(self.args)

    def test_complete_copy_is_byte_identical_and_failed_maps_retained(self):
        before=r.inventory(self.source);result=self.run_recovery();self.assertEqual(result['status'],'complete_numerically_verified_visual_review_required');self.assertEqual(before,r.inventory(self.source));self.assertTrue(all(r.p.sha(self.output/k)==v for k,v in before.items()))
        verify=r.p.read(self.output/'verification.json');self.assertIs(type(verify['valid_maps']),int);self.assertEqual(verify['valid_maps'],115)
        receipt=r.p.read(self.output/'recovery_receipt.json');self.assertEqual(receipt['independently_recomputed']['failed_randomization_records'],1);self.assertFalse(receipt['original_runtime_assertion_evidence']['original_state_assertions_reexecuted']);self.assertTrue(receipt['no_new_inference'])
        with self.assertRaises(ValueError):self.run_recovery()

    def test_missing_duplicate_and_state_flag_rejected(self):
        original=copy.deepcopy(self.rows)
        for rows in [original[:-1],original+[original[0]],[dict(v,prediction_and_logits_unchanged=False) if i==0 else v for i,v in enumerate(original)]]:
            self.rows=rows;self.write_tables()
            with self.assertRaises(ValueError):self.run_recovery()
            self.assertFalse(self.output.exists())

    def test_map_mass_or_randomization_tampering_rejected(self):
        self.mass[0]['center25_mass']=.99;self.write_tables()
        with self.assertRaises(ValueError):self.run_recovery()
        self.mass[0]['center25_mass']=float((self.native/self.native.sum())[r.p.corner.native_regions()['center25']].sum());self.random[-1]['spearman']=.888;self.write_tables()
        with self.assertRaises(ValueError):self.run_recovery()

    def test_probability_replay_violation_stays_blocked(self):
        self.rows[0]['P_MildDemented']=.849;self.rows[0]['P_ModerateDemented']=.051;self.rows[0]['confidence']=.849;self.rows[0]['cloud_max_probability_difference']=.001;self.write_tables()
        result=self.run_recovery();self.assertEqual(result['status'],'blocked_cloud_probability_replay_review');self.assertEqual(len(r.p.read(self.output/'verification.json')['cloud_probability_or_argmax_failures']),1)

    def test_failed_native_map_is_counted_without_fabrication(self):
        self.rows[0]['cam_valid']=False;(self.source/f'maps/{self.chosen}_case000.npy').unlink()
        for row in self.mass:
            if row['case_id']=='case000' and row['model']=='round2_representative':
                row.update(valid=False,corners4_mass=None,edge_ring24_mass=None,center25_mass=None,peak_in_any_corner=None)
        self.write_tables();self.run_recovery();verify=r.p.read(self.output/'verification.json')
        self.assertEqual(verify['valid_maps'],114);self.assertFalse((self.output/f'maps/{self.chosen}_case000.npy').exists())

    def test_changed_log_protocol_or_extra_payload_rejected(self):
        log=self.suite/'wrong.log';log.write_text(self.log.read_text()+'\nchanged');self.args.failure_log=log
        with self.assertRaises(ValueError):self.run_recovery()
        self.args.failure_log=self.log;self.protocol['clean_probability_tolerance']=.1;r.p.dump(self.source/'protocol_before_attribution.json',self.protocol)
        with self.assertRaises(ValueError):self.run_recovery()
        self.protocol['clean_probability_tolerance']=1e-4;r.p.dump(self.source/'protocol_before_attribution.json',self.protocol);(self.source/'unexpected.txt').write_text('extra')
        with self.assertRaises(ValueError):self.run_recovery()

    def test_symlink_and_source_mutation_cannot_seal(self):
        (self.source/'unexpected').symlink_to(self.cp)
        with self.assertRaises(ValueError):self.run_recovery()
        (self.source/'unexpected').unlink();original_copy=r.shutil.copyfileobj;changed=False
        def mutate(reader,writer):
            nonlocal changed
            original_copy(reader,writer)
            if not changed:changed=True;(self.source/'gallery.html').write_text('changed during copy')
        with patch.object(r.shutil,'copyfileobj',side_effect=mutate):
            with self.assertRaises(ValueError):self.run_recovery()
        self.assertFalse((self.output/'attribution_complete.json').exists())

if __name__=='__main__':unittest.main(verbosity=2)
