#!/usr/bin/env python3
"""Synthetic data only; does not open real predictions, images or checkpoints."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import ensemble_supplement as e
import summarize_frozen_evaluation as a
from test_synthetic_analysis import fixture, add_sensitivity, write_csv


def sources(args):
    return {'stage_manifest':args.prepared/'stage_manifest.csv',
            'primary_intersection':args.prepared/'same_case_test_intersection.csv',
            'split_audit':args.prepared/'split_audit.json','original100_key':args.original100_key,
            'clinical_manifest':args.clinical_manifest,'sensitivity_manifest':args.old_hash_unexposed_manifest,
            'sensitivity_audit':args.baseline_exposure_audit}


def ensemble_fixture(root):
    args = add_sensitivity(fixture(root))
    meta_path = args.evaluation_dir/'evaluation.json';meta = json.loads(meta_path.read_text())
    meta['checkpoints']['old_baseline']['sha256'] = e.BASELINE_SHA256
    for name,value in meta['checkpoints'].items():
        if name!='old_baseline':value['selection_config']['image_size'] = 224
    meta_path.write_text(json.dumps(meta))
    selected = json.loads(args.validation_selection.read_text())
    selected['checked_empty_evaluation_output'] = str(args.evaluation_dir.resolve())
    for name,record in selected['candidates'].items():record['checkpoint_path'] = str(root/name/'FAKE_NOT_LOADED.pt')
    args.validation_selection.write_text(json.dumps(selected))
    args.supplement_freeze = root/'supplement_freeze.json'
    stash = root/'synthetic_outputs_stashed';args.evaluation_dir.rename(stash)
    with patch.object(e,'source_manifest_paths',return_value=sources(args)):
        e.freeze(args.evaluation_dir,args.supplement_freeze)
    stash.rename(args.evaluation_dir)
    primary = a.analyze(args);a.write_outputs(primary,root/'primary_analysis')
    args.primary_analysis = root/'primary_analysis/uncertainty_analysis.json'
    args.output = root/'ensemble_analysis'
    return args


class EnsembleSupplementTests(unittest.TestCase):
    def test_full_supplement_and_reproducibility_spec_preserve_primary(self):
        with tempfile.TemporaryDirectory(prefix='kevin-ensemble-synthetic-') as tmp:
            args = ensemble_fixture(Path(tmp))
            before = {p:a.digest(p) for p in args.evaluation_dir.iterdir()}
            before[args.primary_analysis] = a.digest(args.primary_analysis)
            result,outputs = e.analyze(args)
            original = result['stage']['original100']
            self.assertEqual(original['models']['ensemble_stage_only']['point']['accuracy'],1.)
            self.assertEqual(original['member_analysis']['ensemble_stage_only']['n_any_member_disagreement'],20)
            self.assertAlmostEqual(original['member_analysis']['ensemble_stage_only']['mean_pairwise_disagreement_rate'],.4/3)
            self.assertEqual(original['paired_comparisons']['ensemble_stage_only_minus_old_baseline']['difference']['accuracy']['ci95'],[0.,0.])
            self.assertEqual(result['stage']['old_hash_group_unexposed_sensitivity']['n'],7)
            self.assertEqual(result['stage']['old_hash_group_unexposed_sensitivity']['n_resampling_units'],7)
            specs = result['ensemble_specifications']
            self.assertEqual(set(specs),{'ensemble_stage_only','ensemble_multitask','ensemble_multitask_AD_CN'})
            stage = specs['ensemble_stage_only'];self.assertEqual(stage['class_order'],a.STAGES)
            self.assertEqual([m['seed'] for m in stage['members']],[20261011,20261012,20261013])
            self.assertEqual([m['checkpoint_sha256'] for m in stage['members']],['1'*64,'2'*64,'3'*64])
            self.assertEqual(sum(m['weight'] for m in stage['members']),1.)
            self.assertEqual(stage['shared_input_transform']['frozen_trainer_sha256'],e.TRAINER_SHA256)
            self.assertFalse(result['new_model_inference']);self.assertIsNone(result['winner_or_promotion'])
            ext = result['clinical']['external__synthetic_external']
            self.assertEqual(ext['n_binary_evaluable'],2);self.assertEqual(ext['n_mci_prediction_only'],1)
            self.assertAlmostEqual(ext['AD_sensitivity_exact']['ci95'][0],.025)
            extrows = outputs['ensemble_multitask_AD_CN_external__synthetic_external_subject_predictions.csv']
            self.assertEqual(sum(row['interpretation']=='MCI prediction-only' for row in extrows),1)
            # The first member is wrong on this synthetic case: average all three,
            # rather than averaging hard labels, selecting members or refitting.
            row = outputs['ensemble_stage_only_original100_predictions.csv'][0]
            self.assertAlmostEqual(row['P_'+a.STAGES[0]],(.01+.97+.97)/3)
            e.write_outputs(result,outputs,args.output)
            self.assertEqual(len(list(args.output.glob('*.csv'))),8)
            self.assertEqual(json.loads((args.output/'ensemble_specifications.json').read_text()),specs)
            with self.assertRaisesRegex(ValueError,'must be empty'):e.write_outputs(result,outputs,args.output)
            self.assertEqual(before,{p:a.digest(p) for p in before})

    def test_calibration_and_argmax_tie_definitions(self):
        perfect = e.calibration(np.array([0,1]),np.array([[1.,0.],[0.,1.]]))
        self.assertEqual(perfect['multiclass_brier_sum'],0.)
        self.assertEqual(perfect['top_label_ece_10_equal_width_bins'],0.)
        self.assertEqual(perfect['bins'][-1]['n'],2)
        wrong = e.calibration(np.array([1,0]),np.array([[1.,0.],[0.,1.]]))
        self.assertEqual(wrong['multiclass_brier_sum'],2.)
        self.assertEqual(wrong['top_label_ece_10_equal_width_bins'],1.)
        tied = np.mean([np.array([[.5,.5]])]*3,axis=0)
        self.assertEqual(tied.argmax(axis=1).tolist(),[0])
        from plot_evaluation_figures import calibration
        y = np.array([0,1,1,0]);p = np.array([[.7,.3],[.3,.7],[.6,.4],[.5,.5]])
        self.assertEqual(e.calibration(y,p),calibration(y,p))

    def test_freeze_before_any_checkpoint_and_late_launch_guards(self):
        with tempfile.TemporaryDirectory(prefix='kevin-ensemble-freeze-synthetic-') as tmp:
            root = Path(tmp);source = root/'fake_manifest';source.write_text('SYNTHETIC\n')
            with patch.object(e,'source_manifest_paths',return_value={'synthetic_source':source}):
                pinned = e.freeze(root/'future_evaluation',root/'freeze.json')
                self.assertTrue(pinned['frozen_before_heldout_inference'])
                self.assertIn('primary_analysis_protocol_sha256',pinned['code_hashes'])
                self.assertFalse((root/'future_evaluation').exists())
                with self.assertRaisesRegex(ValueError,'already exists'):e.freeze(root/'future_evaluation',root/'freeze.json')
                (root/'evaluation_launch.json').write_text('{}')
                with self.assertRaisesRegex(ValueError,'already launched'):e.freeze(root/'future_evaluation',root/'late.json')
                (root/'evaluation_launch.json').unlink()
                (root/'future_evaluation').mkdir();(root/'future_evaluation/FAKE.csv').write_text('synthetic')
                with self.assertRaisesRegex(ValueError,'before held-out'):e.freeze(root/'future_evaluation',root/'late.json')
                (root/'future_evaluation/FAKE.csv').unlink()
                def mutate_during_source_hashing():
                    (root/'evaluation_launch.json').write_text('{}')
                    return {'synthetic_source':source}
                with patch.object(e,'source_manifest_paths',side_effect=mutate_during_source_hashing):
                    with self.assertRaisesRegex(ValueError,'already launched'):e.freeze(root/'future_evaluation',root/'race.json')
                self.assertFalse((root/'race.json').exists())

    def test_frozen_protocol_manifest_and_primary_report_guards(self):
        with tempfile.TemporaryDirectory(prefix='kevin-ensemble-guard-synthetic-') as tmp:
            args = ensemble_fixture(Path(tmp))
            pinned_text = args.supplement_freeze.read_text();pinned = json.loads(pinned_text)
            pinned['protocol']['weight_rule'] = 'fitted weights'
            args.supplement_freeze.write_text(json.dumps(pinned))
            with self.assertRaisesRegex(ValueError,'code/protocol changed'):e.analyze(args)
            args.supplement_freeze.write_text(pinned_text)
            key_bytes = args.original100_key.read_bytes();args.original100_key.write_bytes(key_bytes+b'\n')
            with self.assertRaisesRegex(ValueError,'source manifest changed'):e.analyze(args)
            args.original100_key.write_bytes(key_bytes)
            primary = json.loads(args.primary_analysis.read_text());primary['input_manifest'] = []
            args.primary_analysis.write_text(json.dumps(primary))
            with self.assertRaisesRegex(ValueError,'Saved primary report'):e.analyze(args)

    def test_missing_cases_and_checkpoint_mismatch_fail(self):
        with tempfile.TemporaryDirectory(prefix='kevin-ensemble-alignment-synthetic-') as tmp:
            args = ensemble_fixture(Path(tmp))
            path = args.evaluation_dir/'candidate_1_original100_predictions.csv';rows = a.read_csv(path,['FileName'])
            write_csv(path,rows[:-1])
            with self.assertRaisesRegex(ValueError,'Case mismatch'):e.analyze(args)
            write_csv(path,rows)
            path = args.evaluation_dir/'evaluation.json';meta = json.loads(path.read_text())
            meta['checkpoints']['candidate_1']['sha256'] = 'a'*64;path.write_text(json.dumps(meta))
            with self.assertRaisesRegex(ValueError,'does not match evaluated checkpoints'):e.analyze(args)

    def test_spec_rejects_wrong_transform_or_member_count(self):
        with tempfile.TemporaryDirectory(prefix='kevin-ensemble-spec-synthetic-') as tmp:
            args = ensemble_fixture(Path(tmp))
            base = json.loads(args.primary_analysis.read_text());selected = json.loads(args.validation_selection.read_text())
            base['checkpoint_provenance']['candidate_1']['selection_config']['image_size'] = 128
            with self.assertRaisesRegex(ValueError,'224px'):e.ensemble_specs(base,selected)
            base['checkpoint_provenance']['candidate_1']['selection_config']['image_size'] = 224
            del base['checkpoint_provenance']['candidate_1']
            with self.assertRaisesRegex(ValueError,'exactly the three'):e.ensemble_specs(base,selected)


if __name__=='__main__':unittest.main(verbosity=2)
