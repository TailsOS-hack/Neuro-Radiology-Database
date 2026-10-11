#!/usr/bin/env python3
"""Synthetic fixtures only; never opens project model or held-out outcomes."""
import argparse
import json
from pathlib import Path
import tempfile
import unittest

import build_results_report as report
import summarize_frozen_evaluation as analysis
from test_synthetic_analysis import add_sensitivity, fixture


class SyntheticReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix='kevin-report-synthetic-')
        cls.root = Path(cls.temp.name)
        args = add_sensitivity(fixture(cls.root))
        meta = json.loads((args.evaluation_dir/'evaluation.json').read_text())
        selection = json.loads(args.validation_selection.read_text())
        complete = []
        for name, record in selection['candidates'].items():
            cfg = meta['checkpoints'][name]['selection_config']
            cfg.update(epochs=100, min_epochs=20, patience=15, min_delta=.0001,
                       training_aux_subjects=2 if cfg['mode']=='multitask' else 0,
                       validation_aux_subjects=2 if cfg['mode']=='multitask' else 0)
            folder = cls.root/f"{cfg['mode']}_seed{cfg['seed']}"
            folder.mkdir()
            (folder/'best.pt').write_bytes(('Synthetic inert checkpoint bytes: '+name).encode())
            sha = report.digest(folder/'best.pt')
            meta['checkpoints'][name].update(sha256=sha, epoch=20)
            record.update(checkpoint_sha256=sha, epoch=20)
            (folder/'config.json').write_text(json.dumps(cfg))
            history = [{'epoch':i,'val_stage_loss':record['stage_validation_nll']+(20-i)*.001} for i in range(1,21)]
            (folder/'history.json').write_text(json.dumps(history))
            (folder/'history.csv').write_text('epoch,val_stage_loss\n20,0.2\n')
            for image in ['loss_curves.png','accuracy_curves.png']:
                (folder/image).write_bytes(b'SYNTHETIC artifact; no image decoding required')
            complete.append(folder.name)
        (args.evaluation_dir/'evaluation.json').write_text(json.dumps(meta))
        args.validation_selection.write_text(json.dumps(selection))
        (cls.root/'suite_status.json').write_text(json.dumps({'state':'training_complete_awaiting_frozen_evaluation','completed':complete}))
        result = analysis.analyze(args)
        analysis.write_outputs(result,args.output)
        cls.args = argparse.Namespace(results_root=cls.root,evaluation_dir=args.evaluation_dir,
            uncertainty_json=args.output/'uncertainty_analysis.json',validation_selection=args.validation_selection,
            cleanup_json=None,output=cls.root/'report.md')

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_complete_report_includes_all_conditions_and_exact_counts(self):
        text = report.render(self.args)
        self.assertIn('8 images; 8 resampling units',text)
        self.assertIn('7 images; 7 resampling units',text)
        self.assertIn('80/100',text)
        self.assertIn('100/100',text)
        self.assertIn('case099.png',text)
        self.assertIn('Reader_A',text)
        self.assertIn('Reader_B',text)
        self.assertIn('MCI descriptive probabilities',text)
        self.assertIn('No cleanup metadata was supplied',text)
        for i in range(1,7):
            self.assertIn(f'candidate_{i}',text)
        self.assertIn('training subjects',text)
        self.assertIn('Clinical AD/CN is distinct',text)
        self.assertIn('Sample SD',text)
        self.assertNotIn('TODO',text)

    def test_changed_checkpoint_is_refused(self):
        file = self.root/'stage_only_seed20261011/best.pt'
        original = file.read_bytes()
        try:
            file.write_bytes(b'altered synthetic checkpoint')
            with self.assertRaisesRegex(ValueError,'checkpoint digest mismatch'):
                report.render(self.args)
        finally:
            file.write_bytes(original)

    def test_changed_aligned_results_are_refused(self):
        file = self.args.evaluation_dir/'original100_case_aligned_old_candidates_readers.csv'
        original = file.read_bytes()
        try:
            file.write_bytes(original+b'\n')
            with self.assertRaisesRegex(ValueError,'Analysis input changed'):
                report.render(self.args)
        finally:
            file.write_bytes(original)

    def test_incomplete_suite_is_refused(self):
        file = self.root/'suite_status.json'
        original = file.read_bytes()
        try:
            file.write_text(json.dumps({'state':'failed','completed':[]}))
            with self.assertRaisesRegex(ValueError,'six completed'):
                report.render(self.args)
        finally:
            file.write_bytes(original)

    def test_cleanup_is_explicit_and_never_inferred(self):
        path = self.root/'synthetic_cleanup.json'
        path.write_text(json.dumps({'pod_deleted_verified':False,'storage_deleted_verified':False,'status':'synthetic record'}))
        args = argparse.Namespace(**vars(self.args))
        args.cleanup_json = path
        text = report.render(args)
        self.assertIn('"pod_deleted_verified": false',text)
        self.assertIn('does not contact Runpod',text)
        self.assertNotIn('No cleanup metadata was supplied',text)


if __name__=='__main__':
    unittest.main(verbosity=2)
