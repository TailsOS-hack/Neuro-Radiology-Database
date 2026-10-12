#!/usr/bin/env python3
"""Independent synthetic-only report receipt and alignment tests."""
import contextlib
import copy
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

CLINICAL=Path(__file__).resolve().parents[1]/'round2_clinical'
sys.path.insert(0,str(CLINICAL))
import clinical_compare as c
import clinical_report as report
from test_clinical_report import fixture


class IndependentClinicalReport(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='synthetic_clinical_report_')
        self.root=Path(self.tmp.name).resolve();self.out=self.root/'clinical_evaluation';self.out.mkdir()
        self.patients,self.predictions,self.metrics=fixture()
        self.manifest=self.root/'patients.json'
        c.dump(self.manifest,dict(status='prepared_without_model_inference',source_sha256={},patients=self.patients))
        self.freeze=dict(status='frozen_before_clinical_heldout_inference',heldout_outcomes_used_for_selection=False,
            source_code_sha256=c.code_hashes(),evaluation_manifest_sha256=c.sha(self.manifest),
            protocol=c.read(CLINICAL/'evaluation_protocol.json'),input_source_sha256={},
            selections={n:dict(checkpoint_sha256=str(i)*64) for i,n in enumerate(c.NEW,1)},
            old_checkpoints={n:dict(path=p,sha256=h) for n,(p,h) in c.OLD.items()})
        self.write_payload()

    def tearDown(self):self.tmp.cleanup()

    def write_payload(self,omit=None):
        c.dump(self.root/'clinical_evaluation_freeze.json',self.freeze)
        c.dump(self.out/'patient_predictions.json',self.predictions)
        c.dump(self.out/'metrics.json',self.metrics)
        states=[]
        for n in list(c.OLD)+c.NEW:
            h=self.freeze['selections'][n]['checkpoint_sha256'] if n in c.NEW else self.freeze['old_checkpoints'][n]['sha256']
            states.append(dict(model=n,checkpoint_sha256=h,all_states_unchanged=True,no_parameter_gradients=True))
        c.dump(self.out/'model_state_verification.json',states)
        c.dump(self.out/'evaluation.json',dict(status='complete',freeze=self.freeze,
            freeze_sha256=c.sha(self.root/'clinical_evaluation_freeze.json'),model_order=c.ORDER,scenarios=c.SCENARIOS,
            fresh_binary_patients=19,mci_not_scored=True,no_training_or_selection=True,
            total_patient_scenario_records=3360,metric_records=168,old_probability_ties=0))
        names=['patient_predictions.json','metrics.json','model_state_verification.json','evaluation.json']
        inv={n:c.sha(self.out/n) for n in names if n!=omit}
        c.dump(self.out/'SHA256SUMS.json',inv)
        c.dump(self.out/'evaluation_complete.json',dict(status='complete_verified',payloads=len(inv),manifest_sha256=c.sha(self.out/'SHA256SUMS.json')))

    def read(self):
        with mock.patch.object(c,'EVALUATION_MANIFEST_SHA',c.sha(self.manifest)):
            return report.read_verified(self.root,self.manifest)

    def test_complete_receipt_and_shuffled_patient_alignment(self):
        first=self.read()[2]
        self.predictions.reverse();self.metrics.reverse();self.patients.reverse();self.write_payload()
        second=self.read()[2]
        self.assertEqual({k:v['keys'] for k,v in first.items()},{k:v['keys'] for k,v in second.items()})
        for cohort in first:
            self.assertTrue((first[cohort]['y']==second[cohort]['y']).all())
            for model in c.ORDER:
                self.assertTrue((first[cohort]['models'][model]['clean']['prob']==second[cohort]['models'][model]['clean']['prob']).all())

    def test_unhashed_prediction_or_metric_payload_rejected(self):
        for name in ['patient_predictions.json','metrics.json','model_state_verification.json','evaluation.json']:
            self.write_payload(omit=name)
            with self.assertRaises(ValueError):self.read()

    def test_payload_byte_tamper_rejected(self):
        with (self.out/'patient_predictions.json').open('a') as f:f.write(' ')
        with self.assertRaisesRegex(ValueError,'payload changed'):self.read()

    def test_unfrozen_or_heldout_selected_receipt_rejected(self):
        self.freeze['status']='not_frozen';self.write_payload()
        with self.assertRaises(ValueError):self.read()
        self.freeze['status']='frozen_before_clinical_heldout_inference'
        self.freeze['heldout_outcomes_used_for_selection']=True;self.write_payload()
        with self.assertRaises(ValueError):self.read()

    def test_paired_analysis_is_patient_aligned_and_order_invariant(self):
        data=self.read()[2]
        first=report.analyze(data,draws=200)
        self.predictions.reverse();self.metrics.reverse();self.patients.reverse();self.write_payload()
        second=report.analyze(self.read()[2],draws=200)
        self.assertEqual(first,second)
        self.assertEqual(sum(len(v['paired_comparisons']) for v in first.values()),12)
        for cohort,d in first.items():
            self.assertEqual(len(d['patient_ids']),{'harvard_fresh_external':19,'besga_reused_test':11,'openneuro_reused_external':15}[cohort])
            for paired in d['paired_comparisons'].values():
                self.assertEqual(paired['unit'],'patient');self.assertEqual(paired['bootstrap_draws'],200)


if __name__=='__main__':unittest.main(verbosity=2)
