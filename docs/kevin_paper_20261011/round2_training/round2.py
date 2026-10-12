#!/usr/bin/env python3
"""Round 2: fixed, validation-only legacy-stage robustness development.

No clinical data or held-out images are opened. No claim of patient independence.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, math, os, platform, random, signal, subprocess, sys, time, traceback
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE.parent / 'training'))
import train_candidate as c

STAGE_SHA = '66a0cd9409d61edd32ee137154f769c749664f255ddebe461cf7683c1f4ba7b9'
TRAINER_SHA = 'cbf67b7c6e842ce35eeef2e0c0a6a0752b865cec3399d7aa58bd5e8ff0e2a5e2'
SEEDS = [20261011, 20261012, 20261013]
INIT_SHAS = {
 20261011: 'a3a811c02a43d5f0ba4b126955a50b2a9d5e98e1a3f333aaba7b201e6dc2d771',
 20261012: '642c39da1adecd7925cb4057718772af2603ef6077df676c4c4abd1907c3356e',
 20261013: 'b0cfa50d30cc6880a52ea0e3aaf7249de3b52ba91163bb1e6472a3d59d12d06b',
}
INIT_BASE = 'models/kevin_20261011'
SCENARIOS = [
 ('clean', 'clean', 0.),
 ('background_flat', 'background', .10), ('background_edge_pattern', 'background', .12),
 ('intensity_090', 'intensity', .9), ('intensity_110', 'intensity', 1.1),
 ('noise_001', 'noise', .01), ('noise_002', 'noise', .02),
 ('blur_050', 'blur', .5), ('blur_100', 'blur', 1.),
]
DEFAULTS = dict(max_epochs=100, min_epochs=20, patience=15, min_delta=1e-4,
 learning_rate=3e-5, weight_decay=.001, label_smoothing=.05, gradient_clip=5.,
 batch_size=64, image_size=224, clean_accuracy_drop=.002,
 clean_balanced_accuracy_drop=.002, clean_nll_multiplier=1.2, clean_nll_slack=.002,
 stress_per_class=64, stress_seed=2026101202, suite_safety_seconds=43200,
 support_threshold_uint8=8, support_dilation_pixels=5, border_width=8, corner_width=16)


def dump(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n'); os.replace(tmp, path)


def seed_for(*parts):
    return int.from_bytes(hashlib.sha256('|'.join(map(str, parts)).encode()).digest()[:8], 'big')


def dependencies():
    global np, torch, Image, ImageFilter, ndimage, TF
    c.imports(); np, torch = c.np, c.torch
    from PIL import Image, ImageFilter
    from scipy import ndimage
    from torchvision.transforms import functional as TF


def resize_rgb(path):
    with Image.open(path) as im:
        return np.array(TF.resize(im.convert('RGB'), [224, 224], antialias=True), copy=True)


def support_mask(rgb):
    gray = np.asarray(Image.fromarray(rgb).convert('L'))
    mask = ndimage.binary_fill_holes(gray > DEFAULTS['support_threshold_uint8'])
    mask = ndimage.binary_dilation(mask, structure=np.ones((3, 3), bool), iterations=5)
    # Near-empty images cannot support a meaningful foreground heuristic.
    # Conservatively skip background edits rather than replace most/all pixels.
    if mask.mean() < .01: mask[:] = True
    return mask


def edge_region():
    z = np.zeros((224, 224), bool); w=8; k=16
    z[:w] = True; z[-w:] = True; z[:, :w] = True; z[:, -w:] = True
    for yy in [0, 224-k]:
        for xx in [0, 224-k]: z[yy:yy+k, xx:xx+k] = True
    return z


def background_edit(rgb, support, rng, edge=False, amount=.12, pattern=False):
    """Only safe-background pixels change; every protected RGB pixel is exact."""
    out = rgb.copy(); change = ~support
    if edge: change = change & edge_region()
    if pattern:
        # Low-frequency grayscale texture, sampled independently of labels.
        low = rng.uniform(0., amount, (8, 8))
        texture = np.asarray(Image.fromarray(np.uint8(low*255)).resize((224,224), Image.Resampling.BILINEAR))
    else: texture = np.full((224,224), int(round(amount*255)), np.uint8)
    out[change] = texture[change, None]
    if not np.array_equal(out[support], rgb[support]): raise AssertionError('Protected support changed')
    return out


def transform(rgb, support, scenario, path, seed=0, epoch=0):
    """Same fixed validation perturbation per path across every model/epoch."""
    if scenario == 'clean': return rgb.astype(np.float32) / 255.
    if scenario == 'train':
        rng = np.random.default_rng(seed_for('train', seed, epoch, path))
        x = rgb.copy(); mask = support
        if rng.random() < .5: x = x[:, ::-1].copy(); mask = mask[:, ::-1]
        if rng.random() < .5:
            x = background_edit(x, mask, rng, edge=bool(rng.integers(2)),
                                amount=float(rng.uniform(0., .12)), pattern=bool(rng.integers(2)))
        if rng.random() < .25:
            x = np.asarray(Image.fromarray(x).filter(ImageFilter.GaussianBlur(float(rng.uniform(0., 1.)))))
        x = x.astype(np.float32) / 255.
        x = np.clip(x * rng.uniform(.9, 1.1), 0., 1.)
        if rng.random() < .5:
            x = np.clip(x + rng.normal(0., rng.uniform(0., .02), x.shape[:2])[...,None], 0., 1.)
        return x.astype(np.float32)
    spec = next(s for s in SCENARIOS if s[0] == scenario)
    _, family, amount = spec
    rng = np.random.default_rng(seed_for(DEFAULTS['stress_seed'], path, family))
    if family == 'background':
        return background_edit(rgb, support, rng, edge='edge' in scenario, amount=amount,
                               pattern='pattern' in scenario).astype(np.float32) / 255.
    if family == 'blur':
        return np.asarray(Image.fromarray(rgb).filter(ImageFilter.GaussianBlur(amount))).astype(np.float32) / 255.
    x = rgb.astype(np.float32) / 255.
    if family == 'intensity': x = x * amount
    if family == 'noise': x = x + amount * rng.normal(size=x.shape[:2])[...,None]
    return np.clip(x, 0., 1.).astype(np.float32)


def select_stress(rows, n=64):
    picked=[]
    for label in range(4):
        ordered=sorted((r for r in rows if int(r['label'])==label),
                       key=lambda r: seed_for(DEFAULTS['stress_seed'], r['path']))
        seen=set()
        for r in ordered:
            if r['group_id'] not in seen:
                picked.append(r); seen.add(r['group_id'])
                if len(seen)==n: break
        if len(seen)!=n: raise ValueError('Insufficient distinct validation hash groups')
    return picked


def selection(clean, stress, baseline, cfg=DEFAULTS):
    families={}
    for name, family, _ in SCENARIOS[1:]: families.setdefault(family, []).append(stress[name]['loss'])
    family_nll={k: sum(v)/len(v) for k,v in families.items()}
    stress_nll=sum(family_nll.values())/len(family_nll)
    score=.5*clean['loss']+.5*stress_nll
    reasons=[]
    if clean['accuracy'] < baseline['accuracy']-cfg['clean_accuracy_drop']-1e-12: reasons.append('clean_accuracy_guard')
    if clean['balanced_accuracy'] < baseline['balanced_accuracy']-cfg['clean_balanced_accuracy_drop']-1e-12: reasons.append('clean_balanced_accuracy_guard')
    if clean['loss'] > cfg['clean_nll_multiplier']*baseline['loss']+cfg['clean_nll_slack']: reasons.append('clean_NLL_guard')
    if not math.isfinite(score): raise ValueError('Nonfinite selection score')
    return dict(score=score, eligible=not reasons, rejection_reasons=reasons,
                stress_family_nll=family_nll, mean_stress_family_nll=stress_nll)


def source_hashes():
    files=[HERE/'round2.py', HERE/'test_round2.py', HERE/'README.md', HERE.parent/'training'/'train_candidate.py']
    return {str(p.relative_to(ROOT)):c.sha(p) for p in files}


def prepare(args):
    dependencies(); root=args.data_root.resolve(); out=args.prepared.resolve()
    src=root/'training_logs/experiments/kevin_20261011/prepared/stage_manifest.csv'
    if c.sha(src)!=STAGE_SHA: raise ValueError('Frozen stage manifest changed')
    if c.sha(HERE.parent/'training/train_candidate.py')!=TRAINER_SHA: raise ValueError('Frozen trainer changed')
    rows=[r for r in c.csv_read(src) if r['split'] in ['train','val']]
    if Counter(r['split'] for r in rows)!=dict(train=19023,val=2705): raise ValueError('Wrong train/val counts')
    if len({r['path'] for r in rows})!=len(rows): raise ValueError('Duplicate path')
    if {r['group_id'] for r in rows if r['split']=='train'} & {r['group_id'] for r in rows if r['split']=='val'}: raise ValueError('Group leakage')
    out.mkdir(parents=True,exist_ok=True); (out/'masks').mkdir(exist_ok=True)
    coverage=[]
    for i,r in enumerate(rows):
        path=root/r['path']; pixel,_=c.identities(path)
        if pixel!=r['pixel_sha256']: raise ValueError(f'Decoded image identity changed: {r["path"]}')
        x=resize_rgb(path); m=support_mask(x)
        mask_name=hashlib.sha256(r['path'].encode()).hexdigest()+'.png'
        mask_path=out/'masks'/mask_name
        Image.fromarray(m.astype(np.uint8)*255).save(mask_path)
        r.update(source_sha256=c.sha(path), mask_file='masks/'+mask_name, mask_sha256=c.sha(mask_path))
        coverage.append(dict(path=r['path'],split=r['split'],label=int(r['label']),
                             protected_fraction=float(m.mean()),safe_background_fraction=float((~m).mean()),
                             safe_edge_fraction=float((~m & edge_region()).mean())))
        if (i+1)%2000==0: print(f'Prepared {i+1}/{len(rows)} TRAIN/VAL images',flush=True)
    c.csv_write(out/'train_val_manifest.csv',rows)
    c.csv_write(out/'stress_validation_manifest.csv',select_stress([r for r in rows if r['split']=='val']))
    c.csv_write(out/'mask_coverage.csv',coverage)
    aggregate={s:{k:float(np.mean([r[k] for r in coverage if r['split']==s])) for k in ['protected_fraction','safe_background_fraction','safe_edge_fraction']} for s in ['train','val']}
    aggregate['background_edit_empty_images']=sum(r['safe_background_fraction']==0 for r in coverage)
    aggregate['background_edit_below_one_percent_images']=sum(r['safe_background_fraction']<.01 for r in coverage)
    aggregate['edge_edit_empty_images']=sum(r['safe_edge_fraction']==0 for r in coverage)
    dump(out/'preparation.json',dict(status='prepared',stage_manifest_sha256=STAGE_SHA,
         train_val_manifest_sha256=c.sha(out/'train_val_manifest.csv'),
         stress_manifest_sha256=c.sha(out/'stress_validation_manifest.csv'),
         mask_coverage_sha256=c.sha(out/'mask_coverage.csv'),counts=dict(Counter(r['split'] for r in rows)),
         stress_counts=dict(Counter(r['label'] for r in c.csv_read(out/'stress_validation_manifest.csv'))),mask_coverage=aggregate))
    print(json.dumps(aggregate,indent=2)); freeze(args)


def freeze(args):
    prepared=args.prepared.resolve(); prep=json.loads((prepared/'preparation.json').read_text())
    sources=source_hashes()
    cfg=dict(schema_version=2,experiment='round2_legacy_stage_robustness',defaults=DEFAULTS,
      seeds=SEEDS,initializations={str(s):dict(path=f'{INIT_BASE}/stage_only_seed{s}.pt',sha256=INIT_SHAS[s]) for s in SEEDS},
      class_order=c.STAGES,scenarios=[dict(name=n,family=f,amount=a) for n,f,a in SCENARIOS],
      selection='Equal clean NLL / equally-weighted four stress-family mean NLL; full clean validation, fixed 64/hash-distinct images per class for stress. Eligible epoch0 fallback. Min score improvement 1e-4; exact ties retain earlier epoch.',
      clean_guards='Accuracy and balanced accuracy no more 0.002 below same-device epoch0; clean NLL <=1.2*epoch0+0.002.',
      minimum_epoch_behavior='No early stop before20; afterward stop after15 consecutive epochs without eligible score improvement. Checkpoint selection considers every completed epoch, including epoch0.',
      training='All19023 train images once/epoch; inverse hash-group-size and class-group loss weighting; full-network AdamW; smoothing0.05; gradient clip5; cosine LR3e-5 to1.5e-6 over100epochs.',
      augmentation='Full224FOV; deterministic per seed/epoch/path flip0.5; background0.5: global or edge equally likely, constant or8x8bilinear texture equally likely, amplitudeU[0,.12]. Blur0.25 radiusU[0,1]. IntensityU[.9,1.1]. SharedRGBnoise0.5 SDU[0,.02]. Background edits only outside protected support; intensity/noise/blur may affect foreground.',
      mask='PIL L>8; fill holes;5 iterations3x3 dilation. Near-empty support<1% protects all. Not anatomical segmentation. Bright corner text remains protected. Edge region border8+corner16, always intersect safe background.',
      source_hashes=sources,preparation=prep,clinical_data_used=False,held_out_images_used_for_selection=False,
      prior_test_results_known=True,assessment_label='Retrospective reused-benchmark development, not fresh independent test validation.',
      attribution_used_for_selection=False,rep_seed_selection='No test-based or CAM-based seed selection. Report all3; optional representative is smallest eligible selected validation score, tie seed order.',
      budget_authority='User superseded $10 total cap; machine <=$10/hour; epochs stop by validation plateau or100max.12h suite timeout is operational failsafe only. Root monitors billing/retrieval/cleanup.',
      resume_policy='No automatic restart or resume; preserve diagnostics and completed checkpoints on failure. Any restart requires explicit reviewed continuation, not new hyperparameter search.',
      interpretation='Research-only 2D legacy-stage experiment. Does not establish real-world clinical readiness, patient independence, MRI domain robustness or improvement.')
    dump(prepared/'protocol.json',cfg)
    print(f'Protocol SHA256 {c.sha(prepared/"protocol.json")}',flush=True)


class Rows:
    def __init__(self, rows, root, prepared, scenarios, seed=0, train=False):
        self.rows,self.root,self.prepared,self.scenarios,self.seed,self.train=rows,root,prepared,scenarios,seed,train
        self.entries=[(i,s) for s in scenarios for i in range(len(rows))]
    def __len__(self): return len(self.entries)
    def __getitem__(self,key):
        if isinstance(key,tuple): key,epoch=key
        else: epoch=0
        i,scenario=self.entries[key]; r=self.rows[i]
        x=resize_rgb(self.root/r['path'])
        with Image.open(self.prepared/r['mask_file']) as im: m=np.asarray(im)>0
        x=transform(x,m,'train' if self.train else scenario,r['path'],self.seed,epoch)
        t=torch.from_numpy(x.transpose(2,0,1).copy())
        t=TF.normalize(t,[.485,.456,.406],[.229,.224,.225])
        return t,int(r['label']),i,scenario


class EpochSampler:
    def __init__(self,n,seed): self.n,self.seed,self.epoch=n,seed,0
    def __len__(self): return self.n
    def __iter__(self):
        order=np.random.default_rng(seed_for('order',self.seed,self.epoch)).permutation(self.n)
        return iter((int(i),self.epoch) for i in order)


class CombinedValidation:
    def __init__(self, clean, stress): self.clean,self.stress=clean,stress
    def __len__(self): return len(self.clean)+len(self.stress)
    def __getitem__(self,i): return self.clean[i] if i<len(self.clean) else self.stress[i-len(self.clean)]


def worker_init(worker_id):
    dependencies()


class Deadline(Exception): pass

def check_deadline(deadline):
    if time.monotonic()>=deadline: raise Deadline('Operational safety deadline reached')


def validate(model,aux,loader,device,deadline):
    model.eval(); aux.eval(); ys={n:[] for n,_,_ in SCENARIOS}; ps={n:[] for n,_,_ in SCENARIOS}
    with torch.inference_mode():
        for x,y,_,scenarios in loader:
            check_deadline(deadline)
            p=c.forward(model,aux,x.to(device,non_blocking=True),'stage').softmax(1).cpu().numpy()
            if not np.isfinite(p).all() or not np.allclose(p.sum(1),1,atol=1e-5): raise ValueError('Invalid probabilities')
            for yi,pi,s in zip(y.tolist(),p.tolist(),scenarios): ys[s].append(yi);ps[s].append(pi)
    ms={s:c.metrics(ys[s],ps[s]) for s in ys}
    return ms.pop('clean'),ms


def verify_prepared(root,prepared,protocol):
    for path,digest in protocol['source_hashes'].items():
        if c.sha(root/path)!=digest: raise ValueError(f'Source hash changed: {path}')
    prep=protocol['preparation']
    for name,key in [('train_val_manifest.csv','train_val_manifest_sha256'),('stress_validation_manifest.csv','stress_manifest_sha256'),('mask_coverage.csv','mask_coverage_sha256')]:
        if c.sha(prepared/name)!=prep[key]: raise ValueError(f'Prepared file changed: {name}')
    rows=c.csv_read(prepared/'train_val_manifest.csv'); stress=c.csv_read(prepared/'stress_validation_manifest.csv')
    if any(r['split'] not in ['train','val'] for r in rows) or any(r['split']!='val' for r in stress): raise ValueError('Non-development split present')
    if Counter(r['split'] for r in rows)!=dict(train=19023,val=2705): raise ValueError('Wrong counts')
    if len({r['path'] for r in rows})!=len(rows): raise ValueError('Duplicate paths')
    bypath={r['path']:r for r in rows}
    if any(r!=bypath.get(r['path']) for r in stress): raise ValueError('Stress sample membership changed')
    expected=select_stress([r for r in rows if r['split']=='val'])
    if stress!=expected: raise ValueError('Stress sample selection changed')
    for r in rows:
        if c.sha(root/r['path'])!=r['source_sha256'] or c.sha(prepared/r['mask_file'])!=r['mask_sha256']: raise ValueError(f'Input changed: {r["path"]}')
    return rows,stress


def plot_history(out,history,smoke=False):
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,3,figsize=(15,4.3))
    x=[r['epoch'] for r in history]
    for key,label in [('train_raw_nll','train raw NLL (augmented)'),('clean_nll','clean validation NLL'),('stress_nll','fixed-stress family mean NLL')]:
        axs[0].plot(x,[r.get(key,float('nan')) for r in history],label=label)
    axs[1].plot(x,[r['score'] for r in history],label='selection score')
    rejected=[r for r in history if not r['eligible']]
    axs[1].scatter([r['epoch'] for r in rejected],[r['score'] for r in rejected],marker='x',color='red',label='clean guard failed')
    for key,label in [('train_accuracy','train (augmented)'),('clean_accuracy','clean validation'),('stress_accuracy','stress mean')]:
        axs[2].plot(x,[r.get(key,float('nan')) for r in history],label=label)
    for ax,title in zip(axs,['Raw unsmoothed loss','Validation-only selection','Accuracy']):
        ax.set_title(title);ax.set_xlabel('Fine-tuning epoch (0 = initialization)');ax.grid(alpha=.2);ax.legend(fontsize=7)
    axs[2].set_ylim(0,1.01)
    fig.suptitle('SMOKE ONLY' if smoke else 'Round2 legacy-stage development; raw curves, no smoothing; not clinical validation')
    fig.tight_layout();fig.savefig(out/'raw_curves.png',dpi=160);fig.savefig(out/'raw_curves.pdf');plt.close(fig)


def train(args):
    dependencies(); root=args.data_root.resolve(); prepared=args.prepared.resolve(); out=args.output.resolve()
    if out.exists() and any(out.iterdir()): raise ValueError('Output must be new and empty')
    out.mkdir(parents=True,exist_ok=True); started=time.monotonic(); deadline=started+args.safety_seconds
    protocol=json.loads((prepared/'protocol.json').read_text()); cfg=protocol['defaults']; protocol_sha=c.sha(prepared/'protocol.json')
    if cfg!=DEFAULTS: raise ValueError('Protocol/default mismatch')
    rows,stress=verify_prepared(root,prepared,protocol)
    tr=[r for r in rows if r['split']=='train']; va=[r for r in rows if r['split']=='val']
    if args.smoke:
        tr=[next(r for r in tr if int(r['label'])==l) for l in range(4)]
        va=[next(r for r in va if int(r['label'])==l) for l in range(4)]
        stress=[next(r for r in stress if int(r['label'])==l) for l in range(4)]
    init=root/protocol['initializations'][str(args.seed)]['path']
    if c.sha(init)!=INIT_SHAS[args.seed]: raise ValueError('Initialization checkpoint changed')
    device=c.device_for(args.device); c.seed_all(args.seed)
    torch.set_float32_matmul_precision('highest')
    if device.type=='cuda': torch.backends.cudnn.benchmark=False
    model,aux=c.network(pretrained=False)
    ck=torch.load(init,map_location='cpu',weights_only=False)
    if ck['class_names']!=c.STAGES or ck['arch']!='mobilenet_v3_large' or ck['config']['seed']!=args.seed or ck['config']['mode']!='stage_only' or ck.get('smoke_only'): raise ValueError('Invalid initialization identity')
    model.load_state_dict(ck['model_state']);aux.load_state_dict(ck['aux_state']);del ck
    model.to(device);aux.to(device)
    batch=min(cfg['batch_size'],4) if args.smoke else cfg['batch_size']
    train_ds=Rows(tr,root,prepared,['clean'],args.seed,True);sampler=EpochSampler(len(tr),args.seed)
    loader_kw=dict(batch_size=batch,num_workers=args.workers,pin_memory=device.type=='cuda',persistent_workers=args.workers>0,worker_init_fn=worker_init)
    loader=torch.utils.data.DataLoader(train_ds,sampler=sampler,**loader_kw)
    val_ds=CombinedValidation(Rows(va,root,prepared,['clean']),Rows(stress,root,prepared,[n for n,_,_ in SCENARIOS[1:]]))
    vloader=torch.utils.data.DataLoader(val_ds,shuffle=False,**loader_kw)
    opt=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=cfg['max_epochs'],eta_min=cfg['learning_rate']*.05)
    gc=Counter(r['group_id'] for r in tr);classgc=Counter({r['group_id']:int(r['label']) for r in tr}.values())
    weights=np.array([1/(gc[r['group_id']]*classgc[int(r['label'])]) for r in tr]);weights*=len(weights)/weights.sum()
    config=dict(mode='stage_only',round=2,seed=args.seed,smoke=args.smoke,device=str(device),
                protocol_sha256=protocol_sha,initial_checkpoint_sha256=INIT_SHAS[args.seed],
                stage_manifest_sha256=STAGE_SHA,test_used_for_selection=False,prior_test_results_known=True,
                batch_size=batch,workers=args.workers,train_n=len(tr),validation_n=len(va),stress_n=len(stress))
    dump(out/'config.json',config);dump(out/'protocol.json',protocol)
    dump(out/'environment.json',dict(python=sys.version,platform=platform.platform(),torch=torch.__version__,numpy=np.__version__,device=str(device),cuda=torch.version.cuda))
    history=[]; best_epoch=0; bad=0; selected=None; status='failed'; reason=None
    def save(epoch,clean,stress,sel,best_score,bad):
        return dict(task='dementia',arch='mobilenet_v3_large',class_names=c.STAGES,image_size=224,
             model_state=model.state_dict(),aux_state=aux.state_dict(),epoch=epoch,round=2,
             optimizer_state=opt.state_dict(),scheduler_state=sched.state_dict(),
             best_selection_score=best_score,selection=sel,stage_validation=clean,stress_validation=stress,
             history=history,config=config,smoke_only=args.smoke,bad_epochs=bad)
    try:
        clean,st=validate(model,aux,vloader,device,deadline);baseline=clean;sel=selection(clean,st,baseline)
        best=sel['score']; selected=dict(epoch=0,clean=clean,stress=st,selection=sel)
        history.append(dict(epoch=0,clean_nll=clean['loss'],clean_accuracy=clean['accuracy'],stress_nll=sel['mean_stress_family_nll'],stress_accuracy=float(np.mean([m['accuracy'] for m in st.values()])),score=best,eligible=True,selected=True,seconds=time.monotonic()-started))
        c.atomic_save(save(0,clean,st,sel,best,0),out/'best.pt');c.atomic_save(save(0,clean,st,sel,best,0),out/'last.pt')
        dump(out/'epoch_000_validation.json',dict(clean=clean,stress=st,selection=sel))
        dump(out/'history.json',history);c.csv_write(out/'history.csv',history)
        print(json.dumps(dict(event='epoch0',seed=args.seed,**history[-1])),flush=True)
        for epoch in range(1,(1 if args.smoke else cfg['max_epochs'])+1):
            check_deadline(deadline);sampler.epoch=epoch;c.seed_all(args.seed+epoch);t=time.monotonic();model.train()
            n=correct=0; raw_sum=objective_sum=0.
            for x,y,indices,_ in loader:
                check_deadline(deadline);x=x.to(device,non_blocking=True);y=y.to(device,non_blocking=True)
                opt.zero_grad(set_to_none=True);logits=c.forward(model,aux,x,'stage')
                ww=torch.tensor(weights[indices.numpy()],device=device,dtype=logits.dtype)
                loss=(torch.nn.functional.cross_entropy(logits,y,label_smoothing=cfg['label_smoothing'],reduction='none')*ww).mean()
                if not torch.isfinite(loss): raise ValueError('Nonfinite training objective')
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip'],error_if_nonfinite=True);opt.step()
                raw_sum+=float(torch.nn.functional.cross_entropy(logits.detach(),y))*len(y);objective_sum+=float(loss.detach())*len(y)
                correct+=int((logits.detach().argmax(1)==y).sum());n+=len(y)
            clean,st=validate(model,aux,vloader,device,deadline);sel=selection(clean,st,baseline)
            improved=sel['eligible'] and sel['score'] < best-cfg['min_delta']
            if improved: best=sel['score'];best_epoch=epoch;bad=0;selected=dict(epoch=epoch,clean=clean,stress=st,selection=sel)
            else: bad+=1
            row=dict(epoch=epoch,train_raw_nll=raw_sum/n,train_weighted_smoothed_objective=objective_sum/n,
                     train_accuracy=correct/n,train_draws=n,clean_nll=clean['loss'],clean_accuracy=clean['accuracy'],
                     clean_balanced_accuracy=clean['balanced_accuracy'],stress_nll=sel['mean_stress_family_nll'],
                     stress_accuracy=float(np.mean([m['accuracy'] for m in st.values()])),score=sel['score'],eligible=sel['eligible'],
                     rejection_reasons='|'.join(sel['rejection_reasons']),selected=improved,best_epoch=best_epoch,bad_epochs=bad,
                     learning_rate=opt.param_groups[0]['lr'],seconds=time.monotonic()-t)
            history.append(row);sched.step();record=save(epoch,clean,st,sel,best,bad)
            c.atomic_save(record,out/'last.pt')
            if improved: c.atomic_save(record,out/'best.pt')
            dump(out/f'epoch_{epoch:03d}_validation.json',dict(clean=clean,stress=st,selection=sel))
            dump(out/'history.json',history);c.csv_write(out/'history.csv',history)
            dump(out/'progress.json',dict(status='running',seed=args.seed,epoch=epoch,best_epoch=best_epoch,seconds=time.monotonic()-started,protocol_sha256=protocol_sha))
            print(json.dumps(dict(event='epoch_complete',seed=args.seed,**row)),flush=True)
            if epoch>=cfg['min_epochs'] and bad>=cfg['patience']: reason='validation_plateau';break
        status='smoke_complete' if args.smoke else 'complete';reason=reason or ('smoke_only' if args.smoke else 'maximum_epochs')
    except Deadline as exc:
        status='safety_stopped';reason=str(exc)
    except BaseException as exc:
        status='failed';reason=f'{type(exc).__name__}: {exc}';(out/'failure_traceback.txt').write_text(traceback.format_exc())
        raise
    finally:
        if history: plot_history(out,history,args.smoke)
        if selected: dump(out/'selection.json',dict(seed=args.seed,selected=selected,epoch0_fallback=best_epoch==0,initial_checkpoint_sha256=INIT_SHAS[args.seed],protocol_sha256=protocol_sha,checkpoint_sha256=c.sha(out/'best.pt'),smoke_only=args.smoke))
        info=dict(status=status,reason=reason,seed=args.seed,completed_epochs=max([r['epoch'] for r in history],default=0),selected_epoch=best_epoch if selected else None,
                  seconds=time.monotonic()-started,protocol_sha256=protocol_sha,selection_sha256=c.sha(out/'selection.json') if selected else None,
                  best_checkpoint_sha256=c.sha(out/'best.pt') if selected else None,last_checkpoint_sha256=c.sha(out/'last.pt') if (out/'last.pt').exists() else None,
                  partial_epoch_discarded=status in ['safety_stopped','failed'],smoke_only=args.smoke,
                  validation_plateau_reached=reason=='validation_plateau',
                  needs_validation_convergence_review=reason=='maximum_epochs')
        dump(out/'run_status.json',info);print(json.dumps(info),flush=True)
    return status


def suite(args):
    out=args.output.resolve()
    if out.exists() and any(out.iterdir()): raise ValueError('Suite output must be new and empty')
    out.mkdir(parents=True); prepared=args.prepared.resolve();protocol=json.loads((prepared/'protocol.json').read_text())
    started=time.monotonic(); deadline=started+DEFAULTS['suite_safety_seconds']; runs={}
    dump(out/'protocol.json',protocol)
    try:
        for seed in SEEDS:
            remaining=deadline-time.monotonic()
            if remaining<=60: break
            run=out/f'stage_only_seed{seed}';log=out/f'stage_only_seed{seed}.log'
            cmd=[sys.executable,str(HERE/'round2.py'),'train','--data-root',str(args.data_root.resolve()),'--prepared',str(prepared),'--output',str(run),'--seed',str(seed),'--device',args.device,'--workers',str(args.workers),'--safety-seconds',str(max(1,remaining-30))]
            dump(out/'suite_progress.json',dict(status='running',current_seed=seed,runs=runs,protocol_sha256=c.sha(prepared/'protocol.json')))
            with log.open('w') as f:
                p=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT)
                try: code=p.wait(timeout=remaining)
                except subprocess.TimeoutExpired:
                    p.terminate()
                    try:p.wait(timeout=20)
                    except subprocess.TimeoutExpired:p.kill();p.wait()
                    code=p.returncode
            runs[str(seed)]=json.loads((run/'run_status.json').read_text()) if (run/'run_status.json').exists() else dict(status='failed_without_status',returncode=code)
            dump(out/'suite_progress.json',dict(status='running',runs=runs,protocol_sha256=c.sha(prepared/'protocol.json')))
            if code!=0 or runs[str(seed)]['status']!='complete': break
    finally:
        complete=len(runs)==3 and all(r['status']=='complete' for r in runs.values())
        selections={str(seed):json.loads((out/f'stage_only_seed{seed}'/'selection.json').read_text()) for seed in SEEDS if (out/f'stage_only_seed{seed}'/'selection.json').exists()}
        needs_review=any(r.get('needs_validation_convergence_review',False) for r in runs.values())
        selection_record=dict(complete=complete,needs_validation_convergence_review=needs_review,protocol_sha256=c.sha(prepared/'protocol.json'),selections=selections,all_seeds_reported=True,selection_used_only_fixed_validation=True,prior_tests_known=True)
        dump(out/'validation_selection_before_any_round2_test.json',selection_record)
        status=dict(status='complete' if complete else 'incomplete_preserve_diagnostics',runs=runs,seconds=time.monotonic()-started,
                    protocol_sha256=c.sha(prepared/'protocol.json'),selection_sha256=c.sha(out/'validation_selection_before_any_round2_test.json'),
                    needs_validation_convergence_review=needs_review,expected_seeds=SEEDS,completed_seeds=[int(s) for s,r in runs.items() if r['status']=='complete'],research_only=True)
        dump(out/'suite_status.json',status);print(json.dumps(status,indent=2),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);sp=p.add_subparsers(dest='command',required=True)
    for name in ['prepare','freeze','train','suite']:
        a=sp.add_parser(name);a.add_argument('--data-root',type=Path,default=ROOT);a.add_argument('--prepared',type=Path,required=True)
        if name in ['train','suite']:
            a.add_argument('--output',type=Path,required=True);a.add_argument('--device',choices=['cpu','mps','cuda'],default='cuda');a.add_argument('--workers',type=int,default=8)
        if name=='train':
            a.add_argument('--seed',type=int,choices=SEEDS,required=True);a.add_argument('--smoke',action='store_true');a.add_argument('--safety-seconds',type=float,default=43200)
    args=p.parse_args();globals()[args.command](args)
if __name__=='__main__':main()
