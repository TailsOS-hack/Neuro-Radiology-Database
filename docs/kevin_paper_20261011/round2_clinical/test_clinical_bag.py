"""Meaningful synthetic invariants; no patient MRI or model research outcomes."""
import importlib.util
import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np
import torch
from torch import nn
from PIL import Image

spec=importlib.util.spec_from_file_location('clinical_bag',Path(__file__).with_name('clinical_bag.py'))
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)


class ClinicalBagTests(unittest.TestCase):
    def test_patient_loss_is_not_slice_loss(self):
        logits=torch.tensor([[8.,0.]]*8+[[0.,20.]],requires_grad=True)
        patient=c.bag_logits(logits,1)
        self.assertEqual(tuple(patient.shape),(1,2))
        loss=nn.functional.cross_entropy(patient,torch.tensor([1]))
        slice_loss=nn.functional.cross_entropy(logits,torch.ones(9,dtype=torch.long))
        self.assertNotAlmostEqual(float(loss.detach()),float(slice_loss.detach()),places=3)
        loss.backward()
        self.assertTrue(torch.allclose(logits.grad,logits.grad[:1].expand_as(logits.grad)))

    def test_probability_pooling_is_deliberately_distinct(self):
        logits=torch.tensor([[8.,0.]]*8+[[0.,20.]])
        a=c.bag_logits(logits,1).softmax(1)
        b=logits.softmax(1).mean(0,keepdim=True)
        self.assertFalse(torch.allclose(a,b,atol=.01))

    def test_shape_guard(self):
        with self.assertRaises(ValueError):c.bag_logits(torch.zeros(8,2),1)

    def test_selection_fallback_tie_and_nonfinite(self):
        self.assertFalse(c.better_score(.5,.5))
        self.assertFalse(c.better_score(.49995,.5))
        self.assertTrue(c.better_score(.4998,.5))
        self.assertFalse(c.better_score(float('nan'),.5))

    def test_frozen_batchnorm_and_capacity(self):
        # Synthetic backbone mimics named block layout without ImageNet download.
        backbone=nn.Module();backbone.features=nn.Sequential(*[nn.Sequential(nn.Conv2d(3,3,1),nn.BatchNorm2d(3)) for _ in range(17)])
        model=c.SliceModel(backbone)
        c.training_mode(model,1)
        self.assertTrue(all(not p.requires_grad for p in model.features.parameters()))
        self.assertTrue(all(p.requires_grad for p in model.head.parameters()))
        before=[m.running_mean.clone() for m in model.modules() if isinstance(m,nn.BatchNorm2d)]
        c.training_mode(model,6)
        for i,block in enumerate(model.features):
            self.assertEqual(block[0].weight.requires_grad,i>=14)
            self.assertFalse(block[1].training)
            self.assertTrue(all(not p.requires_grad for p in block[1].parameters()))
        model.features(torch.rand(2,3,8,8))
        after=[m.running_mean for m in model.modules() if isinstance(m,nn.BatchNorm2d)]
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(before,after)))

    def synthetic_bag(self,root):
        paths=[]
        for i in range(9):
            a=np.zeros((224,224),dtype=np.uint8);a[30:190,40:185]=20+i*20
            p=root/f'image{i}.png';Image.fromarray(a).convert('RGB').save(p);paths.append(p.name)
        return dict(group_key='synthetic__01',paths=paths,label=1,cohort='synthetic',subject_id='01',diagnosis='AD',split='test')

    def test_eval_pixels_and_patient_rng(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);r=self.synthetic_bag(root)
            clean=c.preprocess_bag(r,root)
            raw=np.array(Image.open(root/r['paths'][0]),dtype=np.float32)/255
            expected=(torch.from_numpy(raw.transpose(2,0,1))-c.MEAN)/c.STD
            self.assertTrue(torch.equal(clean[0],expected))
            a=c.preprocess_bag(r,root,True,17,2);b=c.preprocess_bag(r,root,True,17,2)
            self.assertTrue(torch.equal(a,b))
            self.assertFalse(torch.equal(a,c.preprocess_bag(r,root,True,17,3)))
            self.assertEqual(tuple(a.shape),(9,3,224,224))
            self.assertTrue(torch.isfinite(a).all())

    def test_inference_membership_and_pooling(self):
        class Synthetic(nn.Module):
            def forward(self,x):
                z=x.mean((1,2,3));return torch.stack([-z,z],1)
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);r=self.synthetic_bag(root);model=Synthetic()
            metrics,records=c.infer_patients(model,[r],root,'cpu',1,0)
            direct=c.bag_logits(model(c.preprocess_bag(r,root)),1).softmax(1)[0]
            self.assertTrue(np.allclose(records[0]['probabilities'],direct.numpy()))
            self.assertEqual(records[0]['n_slices'],9)
            self.assertEqual(metrics['n'],1)

    def test_metrics_exclude_mci_and_guard_probs(self):
        rows=[dict(label=0,probabilities=[.8,.2]),dict(label=1,probabilities=[.1,.9]),dict(label=-1,probabilities=[.3,.7])]
        m=c.metrics_for_records(rows);self.assertEqual(m['n'],2);self.assertEqual(m['confusion_matrix'],[[1,0],[0,1]])
        rows[0]['probabilities']=[float('nan'),.2]
        with self.assertRaises(ValueError):c.metrics_for_records(rows)

    def test_image_hash_rejects_nonfrozen_size(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'image.png';Image.new('RGB',(223,224)).save(p)
            with self.assertRaises(ValueError):c.pixel_sha(p)

    def test_replay_rejects_fabricated_selection_and_missing_epochs(self):
        history=[dict(epoch=0,val_loss=.8,selected=True,stale_epochs=0),
                 dict(epoch=1,val_loss=.7,selected=True,stale_epochs=0),
                 dict(epoch=2,val_loss=.71,selected=False,stale_epochs=1)]
        self.assertEqual(c.replay_validation_selection(history)['epoch'],1)
        history[2]['selected']=True
        with self.assertRaises(ValueError):c.replay_validation_selection(history)
        with self.assertRaises(ValueError):c.replay_validation_selection([history[0],history[2]])

    def test_completed_suite_replay_and_corrupt_epoch_rejected(self):
        # A tiny synthetic artifact chain exercises every validation-epoch audit.
        # This does not fit a network or read a clinical MRI.
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);prep=root/'prepared';prep.mkdir();suite=root/'suite';suite.mkdir()
            rows=[]
            for split,labels in [('train',[0]*11+[1]*21),('val',[0]*4+[1]*7)]:
                for i,label in enumerate(labels):
                    rows.append(dict(group_key=f'{split}{i}',cohort='synthetic',subject_id=f'{split}{i}',
                                     diagnosis='AD' if label else 'CN',label=label,split=split,n_slices=9))
            protocol=dict(seeds=c.SEEDS,selection={'min_delta':1e-4},min_epochs=20,max_epochs=100,patience=15)
            c.atomic_json(prep/'patients.json',rows);c.atomic_json(prep/'protocol.json',protocol)
            pins=dict(protocol_sha256=c.sha(prep/'protocol.json'),manifest_sha256=c.sha(prep/'patients.json'),trainer_sha256=c.sha(c.__file__))
            selections=[]
            for seed in c.SEEDS:
                folder=suite/f'clinical_seed{seed}';folder.mkdir();history=[]
                selected=None
                for epoch in range(21):
                    confidence=.7 if epoch==0 else .8
                    records=[r|dict(probabilities=[1-confidence,confidence] if r['label'] else [confidence,1-confidence]) for r in rows if r['split']=='val']
                    metrics=c.metrics_for_records(records)
                    c.atomic_json(folder/f'epoch{epoch:03d}_validation.json',dict(metrics=metrics,patients=records))
                    history.append(dict(epoch=epoch,val_loss=metrics['loss'],selected=epoch<2,stale_epochs=max(0,epoch-1)))
                    if epoch==1:selected=dict(epoch=epoch,metrics=metrics,patients=records)
                c.atomic_json(folder/'history.json',history);c.atomic_json(folder/'selected_validation.json',selected)
                train=[r|dict(probabilities=[.2,.8] if r['label'] else [.8,.2]) for r in rows if r['split']=='train']
                c.atomic_json(folder/'selected_train_clean.json',dict(epoch=1,metrics=c.metrics_for_records(train),patients=train))
                meta=dict(seed=seed,epoch=1,architecture=c.ARCHITECTURE,protocol_sha256=pins['protocol_sha256'],manifest_sha256=pins['manifest_sha256'],research_mode='full_fixed_protocol',selection_metric='unweighted_patient_nll')
                c.atomic_checkpoint(folder/'best.pt',dict(model={'synthetic':torch.zeros(1)},metrics=selected['metrics'],**meta))
                selection=dict(**meta,checkpoint='best.pt',checkpoint_sha256=c.sha(folder/'best.pt'),
                               selected_validation_sha256=c.sha(folder/'selected_validation.json'),selected_train_clean_sha256=c.sha(folder/'selected_train_clean.json'),
                               history_sha256=c.sha(folder/'history.json'),validation_nll=selected['metrics']['loss'],last_epoch=20,
                               stop_reason='validation_plateau',held_out_data_used=False,needs_validation_convergence_review=False)
                c.atomic_json(folder/'selection.json',selection);c.atomic_json(folder/'run_status.json',selection|dict(status='complete'))
                selections.append(selection|dict(run_dir=folder.name))
            selection_path=suite/'validation_selection_before_any_clinical_test.json'
            c.atomic_json(selection_path,dict(selections=selections,representative_seed=c.SEEDS[0],held_out_data_used=False,needs_validation_convergence_review=False,threshold=.5,**pins))
            c.atomic_json(suite/'suite_status.json',dict(status='complete',completed_runs=3,needs_validation_convergence_review=False,selection_sha256=c.sha(selection_path),**pins))
            with patch.object(c,'load_prepared',return_value=(protocol,rows)):
                checked=c.verify_completed_suite(suite,prep,root)
                self.assertEqual(len(checked['selections']),3)
                broken=suite/f'clinical_seed{c.SEEDS[0]}'/'epoch017_validation.json'
                payload=json.loads(broken.read_text());payload['patients'][0]['label']=1;c.atomic_json(broken,payload)
                with self.assertRaisesRegex(ValueError,'identity/label'):c.verify_completed_suite(suite,prep,root)


if __name__=='__main__':
    torch.set_num_threads(2)
    unittest.main()
