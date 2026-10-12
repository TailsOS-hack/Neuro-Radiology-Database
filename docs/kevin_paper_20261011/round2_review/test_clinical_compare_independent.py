#!/usr/bin/env python3
"""Independent synthetic-only tests; never load research checkpoints or MRI."""
from __future__ import annotations
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np
from PIL import Image
import torch

CLINICAL = Path(__file__).resolve().parents[1] / 'round2_clinical'
sys.path.insert(0, str(CLINICAL))
import clinical_compare as cc
import clinical_bag as cb


def patients():
    counts = [('harvard_fresh_external', 9, 10, 10),
              ('besga_reused_test', 7, 4, 0),
              ('openneuro_reused_external', 1, 14, 5)]
    out = []
    for cohort, ad, cn, mci in counts:
        for diagnosis, label, n in [('AD', 1, ad), ('CN', 0, cn), ('MCI', -1, mci)]:
            for _ in range(n):
                i = len(out)
                out.append(dict(group_key=f'synthetic_{i:03}', subject_id=f'fixture_{i:03}',
                    evaluation_cohort=cohort, cohort=cohort, diagnosis=diagnosis, label=label,
                    split='test' if cohort=='besga_reused_test' else 'external',
                    paths=['image.png']*9, sha256=['0'*64]*9, pixel_sha256=['0'*64]*9))
    return out


def synthetic_bag(row, root=None, scenario=None, **kwargs):
    # Distinct slice logits make mean(logits)->softmax differ from mean(softmax).
    offset = (int(row['group_key'].split('_')[-1]) % 4) * .8
    x = torch.zeros(9, 3, 2, 2)
    x[:, 0, 0, 0] = torch.tensor([-4.]*8+[20.]) + offset
    return x


class TinyModel(torch.nn.Module):
    def __init__(self, scale=1., bad=None):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(float(scale)))
        self.register_buffer('calls', torch.tensor(0))
        self.bad = bad

    def forward(self, x):
        if self.bad == 'state': self.calls += 1
        value = x[:, 0, 0, 0] * self.scale
        if self.bad == 'tie': value = torch.zeros_like(value)
        if self.bad == 'nan': value = value * float('nan')
        return torch.stack([torch.zeros_like(value), value], 1)


class IndependentClinicalCompare(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='synthetic_clinical_review_')
        self.root = Path(self.tmp.name).resolve()
        self.suite = self.root / 'suite'; self.suite.mkdir()
        self.args = types.SimpleNamespace(data_root=self.root, suite=self.suite,
            manifest=self.root/'patients.json', prepared=self.root/'prepared', device='cpu')
        self.rows = patients()
        self.manifest = dict(status='prepared_without_model_inference', patients=self.rows,
            source_sha256={}, model_inference_performed=False)
        cc.dump(self.args.manifest, self.manifest)
        self.protocol = cc.read(CLINICAL/'evaluation_protocol.json')
        self.cps = {}
        for name in list(cc.OLD)+cc.NEW:
            self.cps[name] = self.root / (name+'.pt')
            self.cps[name].write_text('synthetic checkpoint identity: '+name)
        self.frozen = dict(protocol=self.protocol)
        cc.dump(self.suite/'clinical_evaluation_freeze.json', self.frozen)

    def tearDown(self): self.tmp.cleanup()

    def run_evaluate(self, bad=None, mismatch=False):
        old_scales = dict(zip(cc.OLD, [.9, 1.1, 1.3]))
        new_scales = dict(zip(cc.NEW, [.8, 1.2, 1.4]))
        fake_old = types.ModuleType('evaluate_candidates')
        fake_train = types.ModuleType('train_candidate')
        fake_train.imports = lambda: None
        fake_train.forward = lambda model, aux, x, head: model(x)
        fake_old.load = lambda path, device: (TinyModel(old_scales[path.stem], bad).to(device),
            torch.nn.Identity().to(device), {'config': {'mode': 'multitask'}})
        def fake_load(path, **kwargs):
            return {'model': TinyModel(new_scales[path.stem]).state_dict()}
        def clean(row, root, **kwargs):
            x = synthetic_bag(row)
            return x+1 if mismatch else x
        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.dict(sys.modules, {'train_candidate': fake_train,
                'evaluate_candidates': fake_old}))
            stack.enter_context(mock.patch.object(cc, 'load_frozen', return_value=(self.frozen, self.manifest, self.cps)))
            stack.enter_context(mock.patch.object(cc, 'bag_tensor', side_effect=synthetic_bag))
            stack.enter_context(mock.patch.object(cb, 'preprocess_bag', side_effect=clean))
            stack.enter_context(mock.patch.object(cb, 'build_model', side_effect=TinyModel))
            stack.enter_context(mock.patch.object(torch, 'load', side_effect=fake_load))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            cc.evaluate(self.args)

    def test_full_synthetic_grid_pooling_alignment_mci_and_repeat_guard(self):
        self.run_evaluate()
        out = self.suite/'clinical_evaluation'
        rows = cc.read(out/'patient_predictions.json')
        metrics = cc.read(out/'metrics.json')
        self.assertEqual(len(rows), 3360)
        self.assertEqual(len({(r['group_key'], r['model'], r['scenario']) for r in rows}), 3360)
        self.assertEqual(sum(not r['binary_evaluable'] for r in rows), 15*8*7)
        self.assertTrue(all(r['label']==-1 for r in rows if not r['binary_evaluable']))
        self.assertEqual(len(metrics), 168)
        for r in metrics:
            expected = {'harvard_fresh_external': (19,9,10), 'besga_reused_test':(11,7,4),
                        'openneuro_reused_external':(15,1,14)}[r['cohort']]
            self.assertEqual((r['n_patients'],r['n_ad'],r['n_cn']), expected)
        lookup = {(r['group_key'],r['model'],r['scenario']):np.array(r['probabilities']) for r in rows}
        x = synthetic_bag(self.rows[0]); key = self.rows[0]['group_key']
        for name, scale in zip(cc.OLD, [.9,1.1,1.3]):
            logits=TinyModel(scale)(x)
            np.testing.assert_allclose(lookup[key,name,'clean'], logits.softmax(1).mean(0).detach(), atol=1e-7)
            historical=np.mean(logits.softmax(1).detach().numpy().tolist(),axis=0)
            np.testing.assert_array_equal(lookup[key,name,'clean'],historical)
            self.assertFalse(np.allclose(lookup[key,name,'clean'], logits.mean(0).softmax(0).detach()))
        for name, scale in zip(cc.NEW, [.8,1.2,1.4]):
            logits=TinyModel(scale)(x)
            np.testing.assert_allclose(lookup[key,name,'clean'], logits.mean(0).softmax(0).detach(), atol=1e-7)
        for ensemble, members in [('round1_clinical_ensemble',list(cc.OLD)), ('new_clinical_ensemble',cc.NEW)]:
            np.testing.assert_allclose(lookup[key,ensemble,'clean'],
                np.mean([lookup[key,n,'clean'] for n in members], axis=0), atol=1e-7)
        states = cc.read(out/'model_state_verification.json')
        self.assertEqual(len(states), 6)
        self.assertTrue(all(s['all_states_unchanged'] and s['no_parameter_gradients'] for s in states))
        self.assertEqual(cc.read(out/'evaluation_complete.json')['status'], 'complete_verified')
        before = cc.sha(out/'patient_predictions.json')
        with self.assertRaisesRegex(ValueError, 'Refuse repeated'): self.run_evaluate()
        self.assertEqual(before, cc.sha(out/'patient_predictions.json'))

    def test_transform_disagreement_rejected_before_launch(self):
        with self.assertRaisesRegex(ValueError, 'transforms differ'): self.run_evaluate(mismatch=True)
        self.assertFalse((self.suite/'clinical_evaluation_launch.json').exists())

    def test_historical_old_exact_half_tie_is_fail_closed(self):
        with self.assertRaisesRegex(ValueError, 'historical argmax|Exact old comparator tie'): self.run_evaluate(bad='tie')
        self.assertTrue((self.suite/'clinical_evaluation_launch.json').exists())
        self.assertFalse((self.suite/'clinical_evaluation/evaluation_complete.json').exists())

    def test_nonfinite_predictions_are_fail_closed(self):
        with self.assertRaisesRegex(ValueError, 'Invalid patient probability'): self.run_evaluate(bad='nan')
        self.assertFalse((self.suite/'clinical_evaluation/evaluation_complete.json').exists())

    def test_state_mutating_model_is_fail_closed(self):
        with self.assertRaisesRegex(ValueError, 'changed model state'): self.run_evaluate(bad='state')
        self.assertFalse((self.suite/'clinical_evaluation/evaluation_complete.json').exists())

    def test_safe_paths_reject_absolute_parent_and_symlink(self):
        for bad in ['/tmp/file', '../outside', 'valid/../../outside']:
            with self.assertRaises(ValueError): cc.safe(self.root, bad)
        (self.root/'target').write_text('fixture')
        (self.root/'link').symlink_to(self.root/'target')
        with self.assertRaises(ValueError): cc.safe(self.root,'link')

    def test_manifest_pin_rejects_prefreeze_label_or_source_map_changes(self):
        with mock.patch.object(cc,'EVALUATION_MANIFEST_SHA',cc.sha(self.args.manifest)):
            cc.verify_manifest(self.root,self.args.manifest,False)
            changed=copy.deepcopy(self.manifest)
            changed['patients'][0]['label']=0
            cc.dump(self.args.manifest,changed)
            with self.assertRaisesRegex(ValueError,'membership/labels/QC changed'):
                cc.verify_manifest(self.root,self.args.manifest,False)
        cc.dump(self.args.manifest,self.manifest)
        with mock.patch.object(cc,'EVALUATION_MANIFEST_SHA',cc.sha(self.args.manifest)):
            changed=copy.deepcopy(self.manifest)
            changed['source_sha256']={'invented':'1'*64}
            cc.dump(self.args.manifest,changed)
            with self.assertRaisesRegex(ValueError,'membership/labels/QC changed'):
                cc.verify_manifest(self.root,self.args.manifest,False)

    def test_manifest_source_and_image_readbacks_reject_mutation(self):
        image_path=self.root/'image.png'
        Image.fromarray(np.indices((224,224)).sum(0).astype(np.uint8)).save(image_path)
        pixel=hashlib.sha256(np.asarray(Image.open(image_path).convert('RGB')).tobytes()).hexdigest()
        for row in self.manifest['patients']:
            row['sha256']=[cc.sha(image_path)]*9;row['pixel_sha256']=[pixel]*9
        source=self.root/'source.json';source.write_text('synthetic source')
        self.manifest['source_sha256']={'source.json':cc.sha(source)}
        cc.dump(self.args.manifest,self.manifest)
        with mock.patch.object(cc,'EVALUATION_MANIFEST_SHA',cc.sha(self.args.manifest)):
            cc.verify_manifest(self.root,self.args.manifest,True)
            source.write_text('changed source')
            with self.assertRaisesRegex(ValueError,'source/QC changed'):
                cc.verify_manifest(self.root,self.args.manifest,True)
            source.write_text('synthetic source')
            Image.fromarray(np.zeros((224,224),dtype=np.uint8)).save(image_path)
            with self.assertRaisesRegex(ValueError,'image bytes changed'):
                cc.verify_manifest(self.root,self.args.manifest,True)

    def test_actual_pixel_transform_identity_and_fixed_noise(self):
        row=copy.deepcopy(self.rows[0]); row['paths']=[]
        for j in range(9):
            a=(np.indices((224,224)).sum(0)+j*17).astype(np.uint8)
            path=f'slice{j}.png'; Image.fromarray(a).save(self.root/path);row['paths'].append(path)
        clean=cc.bag_tensor(row,self.root,'clean')
        self.assertTrue(torch.equal(clean,cb.preprocess_bag(row,self.root,train=False)))
        # Independently exercise the actual historical preprocessing function.
        sys.path.insert(0,str(CLINICAL.parent/'training'))
        import train_candidate as legacy
        legacy.imports()
        legacy_rows=[dict(path=str(self.root/p),label=0) for p in row['paths']]
        legacy_data=legacy.dataset(legacy_rows,self.root,False,224)
        legacy_clean=torch.stack([legacy_data[i][0] for i in range(9)])
        self.assertTrue(torch.equal(clean,legacy_clean))
        n1=cc.bag_tensor(row,self.root,'gaussian_noise_0.01')
        n2=cc.bag_tensor(row,self.root,'gaussian_noise_0.01')
        self.assertTrue(torch.equal(n1,n2));self.assertFalse(torch.equal(clean,n1))
        restored=n1*cb.STD+cb.MEAN
        self.assertLess(float((restored[:,0]-restored[:,1]).abs().max()),2e-7)
        self.assertLess(float((restored[:,0]-restored[:,2]).abs().max()),2e-7)
        row2={**row,'group_key':'different_synthetic_patient'}
        self.assertFalse(torch.equal(n1,cc.bag_tensor(row2,self.root,'gaussian_noise_0.01')))

    def test_freeze_and_load_reject_code_selection_and_checkpoint_changes(self):
        # Every path, checkpoint and verification record here is synthetic.
        (self.suite/'clinical_evaluation_freeze.json').unlink()
        selected=[]
        for seed,name in zip(cc.SEEDS,cc.NEW):
            cp=self.suite/f'{name}.pt';cp.write_text(name)
            selected.append(dict(seed=seed, checkpoint_path=str(cp),checkpoint_sha256=cc.sha(cp)))
        checked=dict(status='verified_complete_converged_no_inference',representative_seed=cc.SEEDS[0], selections=selected)
        fake_old={n:(str(p.relative_to(self.root)),cc.sha(p)) for n,p in self.cps.items() if n in cc.OLD}
        with mock.patch.object(cb,'verify_completed_suite',return_value=checked), \
             mock.patch.object(cc,'verify_manifest',return_value=self.manifest), \
             mock.patch.object(cc,'code_hashes',return_value={'synthetic.py':'A'}), \
             mock.patch.object(cc,'OLD',fake_old),contextlib.redirect_stdout(io.StringIO()):
            cc.freeze(self.args)
            cc.load_frozen(self.root,self.suite,self.args.prepared,self.args.manifest,True)
            with self.assertRaisesRegex(ValueError,'must be preserved'): cc.freeze(self.args)
            with mock.patch.object(cc,'code_hashes',return_value={'synthetic.py':'B'}):
                with self.assertRaisesRegex(ValueError,'frozen code/protocol/manifest changed'):
                    cc.load_frozen(self.root,self.suite,self.args.prepared,self.args.manifest,True)
            changed=copy.deepcopy(checked);changed['selections'][0]['epoch']=99
            with mock.patch.object(cb,'verify_completed_suite',return_value=changed):
                with self.assertRaisesRegex(ValueError,'checkpoint metadata changed'):
                    cc.load_frozen(self.root,self.suite,self.args.prepared,self.args.manifest,True)
            Path(selected[0]['checkpoint_path']).write_text('changed synthetic checkpoint')
            with self.assertRaisesRegex(ValueError,'comparator/checkpoint changed'):
                cc.load_frozen(self.root,self.suite,self.args.prepared,self.args.manifest,True)


if __name__=='__main__': unittest.main(verbosity=2)
