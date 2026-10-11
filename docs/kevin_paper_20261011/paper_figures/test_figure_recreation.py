#!/usr/bin/env python3
"""Read-only integrity tests plus synthetic guard checks; no model inference."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import recreate_quantitative_figures as f


class FigureRecreationTests(unittest.TestCase):
    def test_complete_real_input_alignment_and_saved_counts(self):
        base,primary,supplement,runs,validation,numeric=f.load_inputs(f.ROOT/f.RESULTS)
        self.assertEqual(len(numeric),27)
        self.assertEqual([len(base['datasets'][co]['y']) for co in f.COHORTS],[1058,756,100])
        self.assertEqual(len(base['names']),9);self.assertEqual(len(runs),6)
        self.assertEqual({v['n'] for v in validation.values()},{2705})
        self.assertEqual(len(primary['readers']['fixed_panel_summary']['individual_reader_metrics']),3)
        self.assertEqual(supplement['clinical']['external__openneuro_ds007561']['n_mci_prediction_only'],5)

    def test_changed_recomputed_points_are_rejected(self):
        original=f.stats.class_metrics
        def wrong(*args,**kwargs):
            value=original(*args,**kwargs);value['accuracy']-=.01;return value
        with patch.object(f.stats,'class_metrics',side_effect=wrong):
            with self.assertRaisesRegex(ValueError,'Plot values differ'):f.load_inputs(f.ROOT/f.RESULTS)

    def test_replay_manifest_mismatch_refused_before_reading_predictions(self):
        with tempfile.TemporaryDirectory(prefix='kevin-figure-guard-') as tmp:
            folder=Path(tmp);(folder/'replay_metadata.json').write_text(json.dumps({'stage_manifest_sha256':'0'*64,'class_order':f.stats.STAGES}))
            with self.assertRaisesRegex(ValueError,'Replay manifest/class identity mismatch'):f.load_clean_replay(folder,{})

    def test_existing_output_is_never_overwritten(self):
        with tempfile.TemporaryDirectory(prefix='kevin-figure-existing-') as tmp:
            folder=Path(tmp);marker=folder/'marker';marker.write_text('preserve')
            with patch('sys.argv',['recreate_quantitative_figures.py','--output',str(folder)]):
                with self.assertRaisesRegex(ValueError,'new directory'):f.main()
            self.assertEqual(marker.read_text(),'preserve')

    def test_relocation_uses_repo_anchors_and_rejects_unrelated_path(self):
        self.assertEqual(f.relocate('/different/computer/project/docs/x.json'),f.ROOT/'docs/x.json')
        self.assertEqual(f.relocate('training_logs/x.csv'),f.ROOT/'training_logs/x.csv')
        with self.assertRaisesRegex(ValueError,'supported repository artifact'):f.relocate('/different/private/x.json')


if __name__=='__main__':unittest.main(verbosity=2)
