#!/usr/bin/env python3
"""Separate exploratory AD/CN patient-bag experiment; no held-out selection.

Frozen nine-slice Besga inputs, ImageNet-only initialization, patient-level loss.
This module trains and selects on Besga TRAIN/VAL only. It never loads a clinical
test cohort in the training command. infer_patients is the shared evaluation API.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import os
import random
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from PIL import Image, ImageFilter
from torchvision import models

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SEEDS = [20261021, 20261022, 20261023]
ARCHITECTURE = 'mobilenet_v3_large_features_imagenet_v1_dropout02_linear960x2'
MANIFEST = 'data/external/clinical_t1_preprocessed_v1/clinical_slices.csv'
MANIFEST_SHA = '9dc70cad44f40f06dd028fe5e90790d656cf142ce4fb1d6f2f828646719951c1'
PREPROCESSOR = 'docs/kevin_paper_20261011/preprocess_clinical_t1.py'
PREPROCESSOR_SHA = 'bc3e1c7012cc54e9453af38d554073b3666cbc115fbb2a2072ed10a822f49475'
WEIGHTS = 'models/kevin_20261011/imagenet_mobilenet_v3_large_8738ca79.pth'
WEIGHTS_SHA = '8738ca797c879b547d18bbd15da5736ff2557b2036a9af72225393ca61759a04'
MEAN = torch.tensor([.485, .456, .406]).view(3, 1, 1)
STD = torch.tensor([.229, .224, .225]).view(3, 1, 1)
FRACTIONS = [.45, .50, .55, .60, .65, .70, .75, .80, .85]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()


def atomic_json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
    os.replace(tmp, path)


def atomic_checkpoint(path, obj):
    path = Path(path); tmp = path.with_suffix('.tmp')
    torch.save(obj, tmp); os.replace(tmp, path)


def utc():
    return datetime.now(timezone.utc).isoformat()


def pixel_sha(path):
    with Image.open(path) as im:
        a = np.asarray(im.convert('RGB'))
    if a.shape != (224, 224, 3) or not np.array_equal(a[..., 0], a[..., 1]) or not np.array_equal(a[..., 0], a[..., 2]):
        raise ValueError('Expected frozen 224x224 grayscale repeated RGB image')
    return hashlib.sha256(a.tobytes()).hexdigest()


def patient_rows_from_slices(path, root=ROOT, cohorts=None, splits=None, verify_pixels=True):
    """Join slice rows by supplied subject key, never by filename similarity.

    Filters are applied before opening any image. Test inputs are not needed for
    preparing training. Returned path strings are relative to repository root.
    Unknown diagnoses retain label -1 and cannot enter training/selection.
    """
    with Path(path).open(newline='') as f:
        slices = list(csv.DictReader(f))
    groups = defaultdict(list)
    for row in slices:
        if cohorts is not None and row['cohort'] not in cohorts:
            continue
        if splits is not None and row['split'] not in splits:
            continue
        groups[row['group_key']].append(row)
    patients = []
    for key, rows in sorted(groups.items()):
        rows.sort(key=lambda r: float(r['slice_fraction']))
        for field in ['cohort', 'subject_id', 'diagnosis', 'binary_label', 'split']:
            if len({r[field] for r in rows}) != 1:
                raise ValueError(f'Inconsistent patient {field}: {key}')
        if len(rows) != 9 or [float(r['slice_fraction']) for r in rows] != FRACTIONS:
            raise ValueError(f'Expected exactly nine frozen slices: {key}')
        r = rows[0]
        label = {'AD': 1, 'CN': 0}.get(r['diagnosis'], -1)
        expected_label = str(label) if label >= 0 else ''
        if r['binary_label'] != expected_label:
            raise ValueError('Clinical diagnosis/label mismatch')
        paths = [str(Path('data') / x['path']) for x in rows]
        if len(set(paths)) != 9 or len({x['slice_index'] for x in rows}) != 9:
            raise ValueError('Repeated path or slice index within patient')
        if any(Path(p).is_absolute() or '..' in Path(p).parts for p in paths):
            raise ValueError('Unsafe image path')
        hashes = [pixel_sha(Path(root) / p) for p in paths] if verify_pixels else []
        patients.append(dict(group_key=key, cohort=r['cohort'], subject_id=r['subject_id'],
                             diagnosis=r['diagnosis'], label=label, split=r['split'],
                             paths=paths, pixel_sha256=hashes, n_slices=9,
                             slice_fractions=FRACTIONS, age=r.get('age', ''), sex=r.get('sex', '')))
    return patients


class SliceModel(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.features = backbone.features
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(.2)
        self.head = nn.Linear(960, 2)

    def forward(self, x):
        z = self.avgpool(self.features(x)).flatten(1)
        return self.head(self.dropout(z))


def build_model(weights_file=None, initialize_imagenet=False):
    """Return a standard slice model [N,3,224,224] -> [N,2], without downloads.

    Evaluation calls build_model() and strictly loads checkpoint['model'].
    Training explicitly supplies the official, SHA-pinned ImageNet weights.
    """
    backbone = models.mobilenet_v3_large(weights=None)
    if initialize_imagenet:
        if weights_file is None or sha(weights_file) != WEIGHTS_SHA:
            raise ValueError('Official MobileNetV3-Large ImageNet V1 weights required')
        backbone.load_state_dict(torch.load(weights_file, map_location='cpu', weights_only=True), strict=True)
    return SliceModel(backbone)


def bag_logits(slice_logits, batch_size, n_slices=9):
    if slice_logits.shape != (batch_size * n_slices, 2):
        raise ValueError('Wrong patient-bag logit shape')
    return slice_logits.reshape(batch_size, n_slices, 2).mean(1)


def stable_seed(seed, epoch, group_key):
    return int.from_bytes(hashlib.sha256(f'{seed}|{epoch}|{group_key}'.encode()).digest()[:8], 'little')


def preprocess_bag(row, root, train=False, seed=0, epoch=0):
    """Fixed eval input; deterministic patient-consistent augmentation in train.

    Eval only decodes the existing frozen PNG pixels then ImageNet-normalizes.
    Rotation expands the square before resizing, retaining full input FOV.
    Geometric/intensity/blur parameters are shared within a patient; independent
    grayscale noise per slice is generated by a patient/epoch/seed keyed RNG.
    """
    if len(row['paths']) != 9:
        raise ValueError('A patient must have exactly nine slice paths')
    rng = np.random.default_rng(stable_seed(seed, epoch, row['group_key']))
    angle = float(rng.uniform(-5, 5)) if train else 0
    brightness = float(rng.uniform(.92, 1.08)) if train else 1
    contrast = float(rng.uniform(.9, 1.1)) if train else 1
    sigma = float(rng.uniform(0, .01)) if train and rng.random() < .5 else 0
    blur = float(rng.uniform(0, .5)) if train and rng.random() < .25 else 0
    tensors = []
    for path in row['paths']:
        with Image.open(Path(root) / path) as im:
            im = im.convert('RGB')
            if im.size != (224, 224):
                raise ValueError('Frozen clinical image size changed')
            if train:
                im = im.rotate(angle, resample=Image.Resampling.BILINEAR, expand=True, fillcolor=(0,0,0))
                im = im.resize((224,224), resample=Image.Resampling.BILINEAR)
                if blur:
                    im = im.filter(ImageFilter.GaussianBlur(blur))
            a = np.array(im, dtype=np.float32) / 255
        if train:
            a = ((a - a.mean()) * contrast + a.mean()) * brightness
            if sigma:
                a += rng.normal(0, sigma, a.shape[:2]).astype(np.float32)[..., None]
            a = np.clip(a, 0, 1)
        t = torch.from_numpy(np.ascontiguousarray(a.transpose(2,0,1)))
        tensors.append((t - MEAN) / STD)
    return torch.stack(tensors)


class PatientBags(Dataset):
    def __init__(self, rows, root, train=False, seed=0, epoch=0):
        self.rows, self.root = rows, Path(root)
        self.train, self.seed, self.epoch = train, seed, epoch
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, i):
        return preprocess_bag(self.rows[i], self.root, self.train, self.seed, self.epoch), int(self.rows[i]['label']), i


def metrics_for_records(records):
    from sklearn.metrics import balanced_accuracy_score, f1_score, confusion_matrix, roc_auc_score
    eligible = [r for r in records if int(r['label']) in (0,1)]
    if not eligible:
        return {'n':0, 'unit':'patient'}
    y = np.array([r['label'] for r in eligible], dtype=int)
    p = np.asarray([r['probabilities'] for r in eligible], dtype=np.float64)
    if p.shape != (len(y),2) or not np.isfinite(p).all() or np.any(p < 0) or np.any(p > 1) or not np.allclose(p.sum(1),1,atol=1e-6,rtol=0):
        raise ValueError('Invalid patient probabilities')
    pred = (p[:,1] >= .5).astype(int)
    nll = float(-np.log(np.clip(p[np.arange(len(y)),y],1e-15,1)).mean())
    return dict(n=len(y), unit='patient', loss=nll, nll=nll,
                accuracy=float(np.mean(pred == y)),
                balanced_accuracy=float(balanced_accuracy_score(y,pred)),
                macro_f1=float(f1_score(y,pred,labels=[0,1],average='macro',zero_division=0)),
                brier=float(np.mean((p[:,1]-y)**2)),
                auroc=float(roc_auc_score(y,p[:,1])) if len(set(y))==2 else None,
                confusion_matrix=confusion_matrix(y,pred,labels=[0,1]).tolist(),
                class_counts={str(v):int(np.sum(y==v)) for v in [0,1]}, threshold=.5)


def infer_patients(model, rows, root, device, batch_size=4, workers=0):
    loader = DataLoader(PatientBags(rows,root),batch_size=batch_size,shuffle=False,num_workers=workers)
    model.eval(); records = []
    with torch.inference_mode():
        for x, labels, indices in loader:
            b, s, c, h, w = x.shape
            logits = bag_logits(model(x.to(device).reshape(b*s,c,h,w)), b, s)
            p = logits.softmax(1).detach().cpu().numpy()
            if not np.isfinite(p).all():
                raise ValueError('Nonfinite inference probabilities')
            for i, label, prob in zip(indices.tolist(), labels.tolist(), p):
                row = rows[i]
                records.append({k:row[k] for k in ['group_key','cohort','subject_id','diagnosis','split']} |
                               dict(label=int(label), probabilities=[float(v) for v in prob], n_slices=9))
    if len(records) != len(rows) or len({r['group_key'] for r in records}) != len(rows):
        raise ValueError('Patient inference membership mismatch')
    return metrics_for_records(records), records


def configure_trainable(model, epoch, warmup_epochs=5):
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.head.parameters():
        p.requires_grad_(True)
    if epoch > warmup_epochs:
        for block in list(model.features.children())[14:]:
            for p in block.parameters():
                p.requires_grad_(True)
    # Small correlated slice batches cannot estimate stable BatchNorm statistics.
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.eval()
            for p in m.parameters():
                p.requires_grad_(False)


def training_mode(model, epoch, warmup_epochs=5):
    model.train()
    configure_trainable(model,epoch,warmup_epochs)


def better_score(score, best, min_delta=1e-4):
    return math.isfinite(score) and score < best - min_delta


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def prepare(args):
    root, out = args.root.resolve(), args.output.resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError('Refusing to overwrite a prepared clinical protocol')
    out.mkdir(parents=True,exist_ok=True)
    assert sha(root/MANIFEST) == MANIFEST_SHA and sha(root/PREPROCESSOR) == PREPROCESSOR_SHA
    weights = Path(args.weights_file)
    if not weights.is_absolute():
        weights = root/weights
    weights_hash = sha(weights)
    if weights_hash != WEIGHTS_SHA:
        raise ValueError('Wrong ImageNet initialization')
    rows = patient_rows_from_slices(root/MANIFEST,root,['besga_zenodo_3935636'],['train','val'])
    counts = Counter((r['split'],r['label']) for r in rows)
    assert counts == {('train',1):21,('train',0):11,('val',1):7,('val',0):4}, counts
    if len({r['group_key'] for r in rows}) != 43:
        raise ValueError('Patient identity collision')
    pixels = defaultdict(set)
    for r in rows:
        for h in r['pixel_sha256']:
            pixels[h].add(r['group_key'])
    if any(len(keys)>1 for keys in pixels.values()):
        raise ValueError('Exact decoded pixels occur across patients')
    # Demographics stay in the private source manifest and are not model inputs.
    rows = [{k:v for k,v in r.items() if k not in ['age','sex']} for r in rows]
    atomic_json(out/'patients.json', rows)
    protocol = dict(version='clinical_patient_bag_v1',status='prepared_before_fitting_and_fresh_external_predictions',
        endpoint='Source clinical AD versus CN; neither severity staging nor diagnosis of MCI',
        architecture=ARCHITECTURE,trainer_sha256=sha(Path(__file__)),
        source_manifest=MANIFEST,source_manifest_sha256=MANIFEST_SHA,
        preprocessor=PREPROCESSOR,preprocessor_sha256=PREPROCESSOR_SHA,
        patients_manifest_sha256=sha(out/'patients.json'),
        initialization=dict(path=WEIGHTS,sha256=weights_hash,source='Official torchvision MobileNetV3-Large ImageNet1K V1 only'),
        seeds=SEEDS,train_patients=32,val_patients=11,
        counts={f'{s}/{y}':n for (s,y),n in sorted(counts.items())},
        bag_size=9,bag_aggregation='softmax(mean of nine slice logits)',
        train_loss='unweighted patient cross entropy on mean slice logits',class_weights=[1.0,1.0],
        label_smoothing=0.0,selection_metric='unweighted_patient_nll',
        selection=dict(epoch0_eligible=True,min_delta=1e-4,tie='earliest eligible epoch retained',
                       rule='Save checkpoint only if full clean Besga validation patient NLL improves by more than min_delta'),
        max_epochs=100,min_epochs=20,patience=15,warmup_head_only_epochs=5,
        batch_size_patients=4,all_train_patients_once_per_epoch=True,
        optimizer=dict(name='AdamW',head_lr=1e-4,backbone_lr=1e-5,weight_decay=.001,gradient_clip_norm=5.0),
        learning_rate='Cosine decay over fixed100epochs to5percent of initial rates',
        trainable_features='Only feature blocks14,15,16 after5head-only epochs; all other feature parameters frozen',
        batchnorm='All running statistics and affine parameters frozen throughout',dropout=.2,
        augmentation=dict(seed='SHA256(seed|epoch|group_key), independent of worker order',
                          rotation_degrees=[-5,5],rotation_expand=True,resize_after_expansion=[224,224],
                          brightness=[.92,1.08],contrast=[.9,1.1],
                          gaussian_noise_probability=.5,gaussian_noise_sigma=[0,.01],
                          gaussian_blur_probability=.25,gaussian_blur_radius=[0,.5],
                          shared_within_patient=['rotation','brightness','contrast','noise_sigma','blur_radius'],
                          independent_grayscale_noise_per_slice=True,
                          flips=False,crops=False,translations=False,cutout=False),
        eval_transform='Existing frozen224x224RGB pixels /255 then ImageNet mean/std; no geometry or augmentation',
        evaluation=dict(binary_threshold=.5,calibration_fitting=False,threshold_tuning=False,
                        primary_model='Fixed equal-probability ensemble of all3 independently validation-selected seeds',
                        report_all_seeds=True,representative='lowest validation NLL, then ascending seed; attribution only',
                        fresh_primary_source='Harvard Dataverse10.7910/DVN/29352: verified9AD+10CN=19T1subjects',
                        mci='10Harvard MCI remain outsidebinary metrics; report separately without relabeling',
                        reused_sources='Besga test11 and OpenNeuro prior external are descriptive reused cohorts',
                        old_comparators='All3fixedmultitaskclinicalheads with historical mean-probability pooling, plus equal-probability ensemble; whole-pipelinecomparison'),
        no_selection_sources=['All legacy-stage data and outcomes','Besga test','OpenNeuro','Harvard Dataverse','AlzaSet'],
        convergence=dict(safety_wall_hours=12,wall_limit_is_not_scientific_convergence=True,
                         max_epochs_without_patience_requires_validation_only_review=True),
        spending_authority='User machine cap10USD/hour; no total cap; resource control belongs to root',
        limitations=['Exploratory small single-source development:32train11validation patients',
                     'Only19freshAD/CNT1patients, with broad uncertainty and case-control spectrum bias',
                     'No validated brain segmentation, atlas registration, prospective workflow or calibrated clinical-risk claim',
                     'Besga/OpenNeuro outcomes previously inspected; fresh external cohort must remain unused until model/selection freeze',
                     'Different scanner/site/population and preprocessing may impair transportability; no superiority promise'])
    atomic_json(out/'protocol.json',protocol)
    print(json.dumps(dict(prepared=str(out),protocol_sha256=sha(out/'protocol.json'),patients_sha256=sha(out/'patients.json'),n=len(rows)),indent=2))


def load_prepared(prepared, root, verify_images=True):
    protocol = json.loads((prepared/'protocol.json').read_text())
    rows = json.loads((prepared/'patients.json').read_text())
    if protocol['trainer_sha256'] != sha(Path(__file__)) or protocol['patients_manifest_sha256'] != sha(prepared/'patients.json'):
        raise ValueError('Protocol/source/manifest integrity mismatch')
    if sha(root/protocol['initialization']['path']) != protocol['initialization']['sha256']:
        raise ValueError('ImageNet weights changed')
    if len(rows)!=43 or len({r['group_key'] for r in rows})!=43:
        raise ValueError('Wrong prepared patient count')
    counts = Counter((r['split'],r['label']) for r in rows)
    if counts!={('train',1):21,('train',0):11,('val',1):7,('val',0):4}:
        raise ValueError('Wrong patient class/split membership')
    for r in rows:
        if r['cohort']!='besga_zenodo_3935636' or len(r['paths'])!=9:
            raise ValueError('Only frozen Besga TRAIN/VAL may enter fitting')
        if verify_images and [pixel_sha(root/p) for p in r['paths']] != r['pixel_sha256']:
            raise ValueError('Input pixels changed')
    return protocol, rows


def save_history(out, history):
    atomic_json(out/'history.json',history)
    fields = list(history[0])
    tmp = out/'history.csv.tmp'
    with tmp.open('w',newline='') as f:
        w = csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(history)
    os.replace(tmp,out/'history.csv')


def curves(out, history):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1,2,figsize=(12,4.5))
    x=[r['epoch'] for r in history]
    for name,label in [('train_aug','Augmented train (optimization)'),('train_clean','Clean train (resubstitution)'),('val','Clean validation')]:
        for ax,metric in zip(axes,['loss','accuracy']):
            y=[r.get(name+'_'+metric) for r in history]
            ax.plot(x,[np.nan if v is None else v for v in y],label=label,lw=1.4)
    axes[0].set_ylabel('Unweighted patient cross-entropy');axes[1].set_ylabel('Patient accuracy');axes[1].set_ylim(0,1.01)
    for ax in axes:
        ax.set_xlabel('Epoch');ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Exploratory clinical AD/CN:32train/11validation patients; raw curves, no smoothing')
    fig.tight_layout();fig.savefig(out/'raw_patient_curves.png',dpi=170);fig.savefig(out/'raw_patient_curves.pdf');plt.close(fig)


def train_one(seed, root, prepared, out, protocol, rows, device, workers, deadline):
    out.mkdir(parents=True,exist_ok=False)
    seed_all(seed)
    train_rows=[r for r in rows if r['split']=='train'];val_rows=[r for r in rows if r['split']=='val']
    model=build_model(root/protocol['initialization']['path'],initialize_imagenet=True).to(device)
    configure_trainable(model,0)
    # Parameter membership is fixed; frozen parameters receive no gradients until
    # their prescribed unfreezing epoch. Frozen BatchNorm is excluded entirely.
    bn_ids={id(p) for m in model.modules() if isinstance(m,nn.modules.batchnorm._BatchNorm) for p in m.parameters()}
    backbone_params=[p for block in list(model.features.children())[14:] for p in block.parameters() if id(p) not in bn_ids]
    optimizer=torch.optim.AdamW([dict(params=list(model.head.parameters()),lr=1e-4),dict(params=backbone_params,lr=1e-5)],weight_decay=.001)
    criterion=nn.CrossEntropyLoss()
    best_score=float('inf');best_epoch=0;stale=0;history=[];started=time.monotonic()
    pins=dict(protocol_sha256=sha(prepared/'protocol.json'),manifest_sha256=sha(prepared/'patients.json'),
              seed=seed,architecture=ARCHITECTURE,selection_metric='unweighted_patient_nll',research_mode='full_fixed_protocol')
    val_metrics,val_records=infer_patients(model,val_rows,root,device,4,workers)
    train_metrics,train_records=infer_patients(model,train_rows,root,device,4,workers)
    history.append(dict(epoch=0,train_aug_loss=None,train_aug_accuracy=None,train_clean_loss=train_metrics['loss'],train_clean_accuracy=train_metrics['accuracy'],
                        val_loss=val_metrics['loss'],val_accuracy=val_metrics['accuracy'],val_balanced_accuracy=val_metrics['balanced_accuracy'],
                        head_lr=0.0,backbone_lr=0.0,selected=True,stale_epochs=0,seconds=time.monotonic()-started))
    best_score=val_metrics['loss']
    atomic_checkpoint(out/'best.pt',dict(model=model.state_dict(),epoch=0,metrics=val_metrics,**pins))
    atomic_json(out/'epoch000_validation.json',dict(metrics=val_metrics,patients=val_records))
    atomic_json(out/'selected_validation.json',dict(epoch=0,metrics=val_metrics,patients=val_records))
    atomic_json(out/'selected_train_clean.json',dict(epoch=0,metrics=train_metrics,patients=train_records))
    save_history(out,history);curves(out,history)
    reason='max_epochs';last_epoch=0
    for epoch in range(1,protocol['max_epochs']+1):
        if time.monotonic() >= deadline:
            reason='safety_wall_limit';break
        factor=.05+.95*(1+math.cos(math.pi*(epoch-1)/protocol['max_epochs']))/2
        for g,lr in zip(optimizer.param_groups,[1e-4,1e-5]):
            g['lr']=lr*factor
        training_mode(model,epoch,protocol['warmup_head_only_epochs'])
        generator=torch.Generator().manual_seed(seed+epoch)
        loader=DataLoader(PatientBags(train_rows,root,True,seed,epoch),batch_size=4,shuffle=True,num_workers=workers,generator=generator)
        total_loss=0.0;correct=0;seen=[]
        for x,y,idx in loader:
            b,s,c,h,w=x.shape;x=x.to(device);y=y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits=bag_logits(model(x.reshape(b*s,c,h,w)),b,s)
            loss=criterion(logits,y)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite patient training loss')
            loss.backward();norm=nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],5)
            if not torch.isfinite(norm):
                raise ValueError('Nonfinite gradient norm')
            optimizer.step();total_loss+=float(loss.detach())*b
            correct+=int(((logits.detach().softmax(1)[:,1]>=.5).long()==y).sum())
            seen.extend(idx.tolist())
        if sorted(seen)!=list(range(32)):
            raise ValueError('Not every training patient was visited exactly once')
        val_metrics,val_records=infer_patients(model,val_rows,root,device,4,workers)
        train_metrics,train_records=infer_patients(model,train_rows,root,device,4,workers)
        selected=better_score(val_metrics['loss'],best_score,protocol['selection']['min_delta'])
        if selected:
            best_score=val_metrics['loss'];best_epoch=epoch;stale=0
            atomic_checkpoint(out/'best.pt',dict(model=model.state_dict(),epoch=epoch,metrics=val_metrics,**pins))
            atomic_json(out/'selected_validation.json',dict(epoch=epoch,metrics=val_metrics,patients=val_records))
            atomic_json(out/'selected_train_clean.json',dict(epoch=epoch,metrics=train_metrics,patients=train_records))
        else:
            stale+=1
        history.append(dict(epoch=epoch,train_aug_loss=total_loss/32,train_aug_accuracy=correct/32,
                            train_clean_loss=train_metrics['loss'],train_clean_accuracy=train_metrics['accuracy'],
                            val_loss=val_metrics['loss'],val_accuracy=val_metrics['accuracy'],val_balanced_accuracy=val_metrics['balanced_accuracy'],
                            head_lr=optimizer.param_groups[0]['lr'],backbone_lr=optimizer.param_groups[1]['lr'] if epoch>5 else 0.0,
                            selected=selected,stale_epochs=stale,seconds=time.monotonic()-started))
        atomic_json(out/f'epoch{epoch:03d}_validation.json',dict(metrics=val_metrics,patients=val_records))
        atomic_checkpoint(out/'last.pt',dict(model=model.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch,metrics=val_metrics,**pins))
        save_history(out,history);curves(out,history);last_epoch=epoch
        atomic_json(out/'run_status.json',dict(status='running',last_epoch=epoch,selected_epoch=best_epoch,best_validation_nll=best_score,updated_utc=utc(),**pins))
        print(json.dumps(dict(seed=seed,epoch=epoch,train_loss=total_loss/32,val_nll=val_metrics['loss'],selected_epoch=best_epoch,stale=stale)),flush=True)
        if epoch>=protocol['min_epochs'] and stale>=protocol['patience']:
            reason='validation_plateau';break
    needs_review=reason!='validation_plateau'
    selected=json.loads((out/'selected_validation.json').read_text())
    selection=dict(**pins,epoch=best_epoch,checkpoint='best.pt',checkpoint_sha256=sha(out/'best.pt'),
                   selected_validation_sha256=sha(out/'selected_validation.json'),
                   selected_train_clean_sha256=sha(out/'selected_train_clean.json'),
                   history_sha256=sha(out/'history.json'),
                   validation_nll=best_score,validation_metrics=selected['metrics'],last_epoch=last_epoch,
                   stop_reason=reason,needs_validation_convergence_review=needs_review,
                   held_out_data_used=False,completed_utc=utc())
    atomic_json(out/'selection.json',selection)
    atomic_json(out/'run_status.json',dict(status='complete' if not needs_review else 'complete_needs_validation_review',**selection))
    return selection


def replay_validation_selection(history, min_delta=1e-4):
    """Recompute the selected epoch without trusting saved selected/stale flags."""
    if not history or [r['epoch'] for r in history] != list(range(len(history))):
        raise ValueError('Missing or reordered raw history epochs')
    best=float('inf');selected_epoch=None;stale=0
    for row in history:
        value=float(row['val_loss'])
        if not math.isfinite(value):
            raise ValueError('Nonfinite validation history')
        selected=better_score(value,best,min_delta)
        if selected:
            best=value;selected_epoch=row['epoch'];stale=0
        else:
            stale+=1
        if bool(row['selected'])!=selected or row['stale_epochs']!=stale:
            raise ValueError('Stored selection/staleness disagrees with raw validation history')
    return dict(epoch=selected_epoch,validation_nll=best,stale_epochs=stale,last_epoch=history[-1]['epoch'])


def verify_completed_suite(suite, prepared, root):
    """Fail-closed, non-inference audit before a separate held-out evaluator.

    Independently checks each full validation prediction epoch, replays the
    selection rule, and verifies protocol/manifest/checkpoint identities.
    Returns a dict containing the three absolute checkpoint paths and pins.
    """
    suite,prepared,root=Path(suite),Path(prepared),Path(root)
    protocol,rows=load_prepared(prepared,root)
    status=json.loads((suite/'suite_status.json').read_text())
    selection_path=suite/'validation_selection_before_any_clinical_test.json'
    freeze=json.loads(selection_path.read_text())
    if status.get('status')!='complete' or status.get('completed_runs')!=3 or status.get('needs_validation_convergence_review'):
        raise ValueError('Incomplete or unconverged suite cannot unlock held-out evaluation')
    if status['selection_sha256']!=sha(selection_path):
        raise ValueError('Selection document changed')
    for obj in [status,freeze]:
        if obj['protocol_sha256']!=sha(prepared/'protocol.json') or obj['manifest_sha256']!=sha(prepared/'patients.json') or obj['trainer_sha256']!=sha(Path(__file__)):
            raise ValueError('Suite source/protocol pins do not match prepared protocol')
    if freeze['held_out_data_used'] or freeze['needs_validation_convergence_review'] or freeze['threshold']!=.5:
        raise ValueError('Selection boundary violation')
    if [r['seed'] for r in freeze['selections']]!=protocol['seeds']:
        raise ValueError('Wrong or reordered seeds')
    expected={r['group_key']:r for r in rows if r['split']=='val'}
    checked=[]
    for selected in freeze['selections']:
        seed=selected['seed'];folder=suite/f'clinical_seed{seed}'
        if selected['run_dir']!=folder.name or selected['checkpoint']!='best.pt':
            raise ValueError('Unexpected run/checkpoint path')
        saved=json.loads((folder/'selection.json').read_text())
        if saved!={k:v for k,v in selected.items() if k!='run_dir'}:
            raise ValueError('Run and suite selections disagree')
        run_status=json.loads((folder/'run_status.json').read_text())
        if run_status.get('status')!='complete' or any(run_status.get(k)!=v for k,v in saved.items()):
            raise ValueError('Run status mismatch')
        if saved.get('research_mode')!='full_fixed_protocol' or saved['stop_reason']!='validation_plateau' or saved['held_out_data_used'] or saved['needs_validation_convergence_review']:
            raise ValueError('Incomplete/smoke/nonconverged run')
        for name,key in [('best.pt','checkpoint_sha256'),('history.json','history_sha256'),
                         ('selected_validation.json','selected_validation_sha256'),('selected_train_clean.json','selected_train_clean_sha256')]:
            if sha(folder/name)!=saved[key]:
                raise ValueError(f'Run output checksum mismatch: {name}')
        history=json.loads((folder/'history.json').read_text())
        replay=replay_validation_selection(history,protocol['selection']['min_delta'])
        if replay['epoch']!=saved['epoch'] or abs(replay['validation_nll']-saved['validation_nll'])>1e-12:
            raise ValueError('Checkpoint selection cannot be reproduced')
        if replay['last_epoch']<protocol['min_epochs'] or replay['last_epoch']>protocol['max_epochs'] or replay['stale_epochs']<protocol['patience']:
            raise ValueError('Convergence rules not satisfied')
        for row in history:
            payload=json.loads((folder/f"epoch{row['epoch']:03d}_validation.json").read_text())
            records=payload['patients']
            if len(records)!=11 or len({r['group_key'] for r in records})!=11 or {r['group_key'] for r in records}!=set(expected):
                raise ValueError('Validation patient membership changed')
            for record in records:
                original=expected[record['group_key']]
                if any(record[k]!=original[k] for k in ['label','cohort','subject_id','diagnosis','split']) or record['n_slices']!=9:
                    raise ValueError('Validation identity/label mismatch')
            actual=metrics_for_records(records)
            if abs(actual['loss']-row['val_loss'])>1e-12 or abs(actual['loss']-payload['metrics']['loss'])>1e-12:
                raise ValueError('Validation NLL does not match saved predictions/history')
        best_payload=json.loads((folder/'selected_validation.json').read_text())
        if best_payload!=dict(epoch=saved['epoch'],**json.loads((folder/f"epoch{saved['epoch']:03d}_validation.json").read_text())):
            raise ValueError('Selected predictions are not selected epoch predictions')
        train_payload=json.loads((folder/'selected_train_clean.json').read_text())
        train_expected={r['group_key']:r for r in rows if r['split']=='train'}
        train_records=train_payload['patients']
        if train_payload['epoch']!=saved['epoch'] or len(train_records)!=32 or len({r['group_key'] for r in train_records})!=32 or {r['group_key'] for r in train_records}!=set(train_expected):
            raise ValueError('Selected clean-training membership changed')
        for record in train_records:
            original=train_expected[record['group_key']]
            if any(record[k]!=original[k] for k in ['label','cohort','subject_id','diagnosis','split']) or record['n_slices']!=9:
                raise ValueError('Training identity/label mismatch')
        if abs(metrics_for_records(train_records)['loss']-train_payload['metrics']['loss'])>1e-12:
            raise ValueError('Clean-training metrics do not match patient predictions')
        checkpoint=torch.load(folder/'best.pt',map_location='cpu',weights_only=True)
        for key in ['seed','epoch','architecture','protocol_sha256','manifest_sha256','research_mode','selection_metric']:
            if checkpoint.get(key)!=saved[key]:
                raise ValueError(f'Checkpoint metadata mismatch: {key}')
        if checkpoint['metrics']!=best_payload['metrics'] or not all(torch.isfinite(t).all() for t in checkpoint['model'].values()):
            raise ValueError('Invalid checkpoint metrics or tensors')
        checked.append(saved|dict(checkpoint_path=str((folder/'best.pt').resolve())))
    representative=min(checked,key=lambda r:(r['validation_nll'],r['seed']))['seed']
    if freeze['representative_seed']!=representative:
        raise ValueError('Representative seed selection mismatch')
    return dict(status='verified_complete_converged_no_inference',protocol_sha256=sha(prepared/'protocol.json'),
                manifest_sha256=sha(prepared/'patients.json'),selection_sha256=sha(selection_path),
                representative_seed=representative,selections=checked)


def run(args):
    root=args.root.resolve();prepared=args.prepared.resolve();out=args.output.resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError('Refusing to overwrite or silently resume an existing clinical suite')
    out.mkdir(parents=True,exist_ok=True)
    protocol,rows=load_prepared(prepared,root)
    device=torch.device(args.device)
    deadline=time.monotonic()+12*3600
    pins=dict(protocol_sha256=sha(prepared/'protocol.json'),manifest_sha256=sha(prepared/'patients.json'),trainer_sha256=sha(Path(__file__)))
    atomic_json(out/'suite_status.json',dict(status='running',started_utc=utc(),device=str(device),**pins))
    atomic_json(out/'runtime_environment.json',dict(torch=torch.__version__,numpy=np.__version__,device=str(device),
                                                 gpu=torch.cuda.get_device_name() if device.type=='cuda' else None,
                                                 workers=args.workers,**pins))
    selections=[]
    try:
        for seed in protocol['seeds']:
            if time.monotonic()>=deadline:
                raise TimeoutError('Safety wall limit before all seeds; retain partial diagnostics')
            selection=train_one(seed,root,prepared,out/f'clinical_seed{seed}',protocol,rows,device,args.workers,deadline)
            selection['run_dir']=f'clinical_seed{seed}';selections.append(selection)
        representative=min(selections,key=lambda r:(r['validation_nll'],r['seed']))['seed']
        freeze=dict(status='validation_selected_before_any_clinical_test',selections=selections,
                    representative_seed=representative,representative_usage='Attribution only, not the primary external performance claim',
                    primary_model='Equal-probability ensemble of all3selectedseeds',threshold=.5,
                    needs_validation_convergence_review=any(r['needs_validation_convergence_review'] for r in selections),
                    held_out_data_used=False,completed_utc=utc(),**pins)
        atomic_json(out/'validation_selection_before_any_clinical_test.json',freeze)
        atomic_json(out/'suite_status.json',dict(status='complete' if not freeze['needs_validation_convergence_review'] else 'complete_needs_validation_review',
                                               completed_runs=3,selection_sha256=sha(out/'validation_selection_before_any_clinical_test.json'),
                                               needs_validation_convergence_review=freeze['needs_validation_convergence_review'],completed_utc=utc(),**pins))
    except Exception as exc:
        atomic_json(out/'suite_status.json',dict(status='failed',error=f'{type(exc).__name__}: {exc}',completed_runs=len(selections),
                                               completed_selections=selections,updated_utc=utc(),**pins))
        raise


def main():
    ap=argparse.ArgumentParser(description=__doc__);sub=ap.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--weights-file',type=Path,default=Path(WEIGHTS));p.set_defaults(func=prepare)
    p=sub.add_parser('run');p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--prepared',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda',choices=['cuda','mps','cpu']);p.add_argument('--workers',type=int,default=2);p.set_defaults(func=run)
    args=ap.parse_args();args.func(args)


if __name__=='__main__':
    main()
