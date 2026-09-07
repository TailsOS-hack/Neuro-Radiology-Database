#!/usr/bin/env python3
"""Deterministic specialist Grad-CAM review: no training and no clinical localization claims.

Run from any directory. Requires the accepted pipeline checkpoints and local data.
Outputs raw CAM arrays, per-image CSV, descriptive JSON, PNG figures and HTML gallery.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import html
import json
import platform
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import experiment_pipeline as ep


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--per-class', type=int, default=32)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--output-dir', type=Path, default=ROOT/'docs/review_bundle/explainability')
    p.add_argument('--manifest', type=Path, default=ep.DEFAULT_MANIFEST)
    p.add_argument('--random-repeats', type=int, default=5)
    args = p.parse_args()
    if args.per_class < 1 or args.random_repeats < 1:
        p.error('sample size and random repeats must be positive')
    import numpy as np
    import torch
    import torchvision
    import torch.nn.functional as F
    from PIL import Image
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from src.grad_cam import GradCAM, get_target_layer, overlay_cam_on_image
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    (out/'RUNNING').write_text('Generation in progress; do not use older outputs.\n')
    (out/'maps').mkdir(exist_ok=True)
    (out/'cases').mkdir(exist_ok=True)
    manifest = ep.read_manifest(args.manifest)
    test_paths = {r['path']: r for r in manifest if r['split']=='test'}
    provenance = {'manifest_sha256': sha(args.manifest), 'seed': args.seed,
                  'per_class': args.per_class, 'random_repeats': args.random_repeats,
                  'python': platform.python_version(), 'torch': torch.__version__,
                  'torchvision': torchvision.__version__, 'numpy': np.__version__,
                  'device': 'cpu', 'script_sha256': sha(Path(__file__)),
                  'source_sha256': {name: sha(ROOT/name) for name in ['src/grad_cam.py', 'src/experiment_pipeline.py']},
                  'checkpoints': {}}
    all_rows, summaries, gallery = [], {}, []
    for task in ['tumor', 'dementia']:
        checkpoint_path = ep.TASK_DEFAULT_MODEL_PATH[task]
        if checkpoint_path.stat().st_size < 1024:
            raise RuntimeError(f'{checkpoint_path}: missing trained weights (LFS pointer)')
        model, ckpt = ep.load_checkpoint_model(checkpoint_path, torch.device('cpu'))
        if not (ckpt['task'] == task and ckpt['class_names'] == ep.TASK_CLASS_NAMES[task]): raise ValueError("Explainability invariant failed: ckpt['task'] == task and ckpt['class_names'] == ep.TASK_CLASS_NAMES[task]")
        evidence_path=ROOT/f'training_logs/publication_evidence/{task}/probabilities.csv'
        evidence=list(csv.DictReader(evidence_path.open()))
        if not (len({r['image_path'] for r in evidence}) == len(evidence)): raise ValueError("Explainability invariant failed: len({r['image_path'] for r in evidence}) == len(evidence)")
        expected={r['path'] for r in manifest if r['split']=='test' and r['domain']==task}
        if not ({r['image_path'] for r in evidence} == expected): raise ValueError('Evidence/manifest mismatch')
        for r in evidence:
            if not (test_paths[r['image_path']]['subtype'] == r['true_label']): raise ValueError("Explainability invariant failed: test_paths[r['image_path']]['subtype'] == r['true_label']")
        rng=random.Random(args.seed)
        quantitative=[]
        display={}
        for label in ckpt['class_names']:
            pool=sorted([r for r in evidence if r['true_label']==label], key=lambda r:r['image_path'])
            quantitative.extend(rng.sample(pool,min(args.per_class,len(pool))))
            correct=[r for r in pool if r['correct']=='1']
            wrong=[r for r in pool if r['correct']=='0']
            if correct:
                display[min(correct,key=lambda r:(float(r['confidence']),r['image_path']))['image_path']]='lowest-confidence correct'
            if wrong:
                display[max(wrong,key=lambda r:(float(r['confidence']),r['image_path']))['image_path']]='highest-confidence error'
        qpaths={r['image_path'] for r in quantitative}
        selected={r['image_path']:r for r in quantitative}
        selected.update({r['image_path']:r for r in evidence if r['image_path'] in display})
        provenance['checkpoints'][task]={'sha256':sha(checkpoint_path), 'path':str(checkpoint_path.relative_to(ROOT)),
            'created_utc':ckpt.get('created_utc'), 'architecture':ckpt['arch'], 'classes':ckpt['class_names'],
            'image_size':ckpt.get('image_size',224), 'probabilities_sha256':sha(evidence_path),
            'target_layer':'features[-1]', 'native_cam_shape':None}
        size=int(ckpt.get('image_size',224))
        transform=ep.build_transforms(False,size)
        layer=get_target_layer(model)
        task_rows=[]
        for i,(path,ref) in enumerate(sorted(selected.items())):
            with Image.open(ROOT/path) as im:
                original=im.convert('RGB').resize((size,size), resample=Image.Resampling.BILINEAR)
                x=transform(im.convert('RGB')).unsqueeze(0)
            with torch.no_grad():
                probs=model(x).softmax(1)[0]
            target=int(probs.argmax())
            delta=max(abs(float(probs[j])-float(ref[f'prob_{label}'])) for j,label in enumerate(ckpt['class_names']))
            if target!=int(ref['pred_index']) or delta>1e-4:
                raise RuntimeError(f'Checkpoint does not reproduce accepted probabilities: {path}, max difference {delta}')
            cam=GradCAM(model,layer).generate(x,target)
            if not (not layer._forward_hooks and (not layer._backward_hooks)): raise ValueError('Explainability invariant failed: not layer._forward_hooks and (not layer._backward_hooks)')
            row={'task':task,'path':path,'image_sha256':sha(ROOT/path),'true_label':ref['true_label'],
                 'predicted_label':ckpt['class_names'][target],'confidence':float(probs[target]),
                 'correct':int(ref['correct']),'quantitative_sample':int(path in qpaths),
                 'gallery_reason':display.get(path,''),'max_probability_difference':delta,
                 'cam_status':'ok' if cam is not None else 'flat_or_failed',
                 'cam_path':'','top20_drop':None,'random20_drop':None,'paired_drop_advantage':None,
                 'border_mass':None}
            case_id=task+'_'+hashlib.sha256(path.encode()).hexdigest()[:12]
            if cam is not None:
                if not (np.isfinite(cam).all() and cam.min() >= 0 and (cam.max() <= 1)): raise ValueError('Explainability invariant failed: np.isfinite(cam).all() and cam.min() >= 0 and (cam.max() <= 1)')
                provenance['checkpoints'][task]['native_cam_shape']=list(cam.shape)
                np.save(out/'maps'/f'{case_id}.npy',cam)
                row['cam_path']=f'maps/{case_id}.npy'
                up=F.interpolate(torch.from_numpy(cam)[None,None],size=(size,size),mode='bilinear',align_corners=False)[0,0]
                # Exactly floor(20% of pixels); stable ordering makes ties reproducible.
                count=int(.2*size*size)
                order=torch.argsort(up.flatten(),descending=True,stable=True)
                masks=[]
                top=torch.zeros(size*size,dtype=torch.bool); top[order[:count]]=True
                masks.append(top.reshape(size,size))
                local=np.random.default_rng(args.seed+int(case_id.split('_')[1],16))
                for _ in range(args.random_repeats):
                    mask=torch.zeros(size*size,dtype=torch.bool)
                    mask[local.choice(size*size,count,replace=False)]=True
                    masks.append(mask.reshape(size,size))
                batch=x.repeat(len(masks),1,1,1)
                # Zero normalized pixels equals ImageNet RGB mean, not black.
                for j,mask in enumerate(masks): batch[j,:,mask]=0
                with torch.no_grad(): changed=model(batch).softmax(1)[:,target]
                drops=float(probs[target])-changed.numpy()
                row['top20_drop']=float(drops[0]); row['random20_drop']=float(drops[1:].mean())
                row['paired_drop_advantage']=row['top20_drop']-row['random20_drop']
                border=torch.ones_like(up,dtype=torch.bool); width=round(.1*size)
                border[width:-width,width:-width]=False
                row['border_mass']=float(up[border].sum()/up.sum())
                if path in display:
                    fig,axes=plt.subplots(1,3,figsize=(9,3.5))
                    axes[0].imshow(original); axes[0].set_title('Input (resized)')
                    axes[1].imshow(cam,cmap='jet',vmin=0,vmax=1); axes[1].set_title(f'Grad-CAM {cam.shape[0]}×{cam.shape[1]}')
                    axes[2].imshow(overlay_cam_on_image(original,cam)); axes[2].set_title('Overlay (α=0.45)')
                    for ax in axes: ax.axis('off')
                    title=f"{task}: {ref['true_label']} → {row['predicted_label']} | confidence {row['confidence']:.4f}"
                    fig.suptitle(title+'\n'+display[path],fontsize=10)
                    fig.tight_layout(); fig.savefig(out/'cases'/f'{case_id}.png',dpi=140); plt.close(fig)
                    gallery.append((task,case_id,title,path,display[path]))
            all_rows.append(row); task_rows.append(row)
            if i%16==0: print(f'{task}: {i+1}/{len(selected)}',flush=True)
        sampled=[r for r in task_rows if r['quantitative_sample']]
        valid=[r for r in sampled if r['cam_status']=='ok']
        def describe(rows):
            return {k:{'mean':float(np.mean([r[k] for r in rows])), 'sd':float(np.std([r[k] for r in rows],ddof=1)) if len(rows)>1 else None}
                    for k in ['top20_drop','random20_drop','paired_drop_advantage','border_mass']} if rows else {}
        summaries[task]={'test_population':len(evidence),'sample_n':len(sampled),'valid_cam_n':len(valid),
                         'failed_cam_n':len(sampled)-len(valid),'sample_correct':sum(r['correct'] for r in sampled),
                         'metrics':describe(valid),'by_class':{label:{'n':sum(r['true_label']==label for r in valid),
                         'metrics':describe([r for r in valid if r['true_label']==label])} for label in ckpt['class_names']},
                         'gallery_n':sum(g[0]==task for g in gallery),
                         'max_probability_difference':max(r['max_probability_difference'] for r in task_rows)}
    write_csv(out/'per_image.csv',all_rows)
    (out/'summary.json').write_text(json.dumps(summaries,indent=2)+'\n')
    (out/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    esc=html.escape
    content='''<!doctype html><html lang="en"><meta charset="utf-8"><title>Specialist Grad-CAM review</title>
<style>body{max-width:1050px;margin:30px auto;font:16px system-ui;background:#f3f5f8;color:#182338}img{width:100%}article{background:white;padding:20px;margin:24px 0}code{overflow-wrap:anywhere}</style>
<h1>Specialist Grad-CAM review</h1><p>Internal public-dataset benchmark. Exploratory model attribution, not validated anatomical localization or clinical evidence. Softmax scores are model confidence, not calibrated clinical probabilities.</p>
<p>Each class contributes its lowest-confidence correct image and, when present, highest-confidence error in the archived strict-test predictions. Classes with no errors have no error panel. This purposive gallery is separate from the seeded quantitative sample.</p>
<p><a href="summary.json">Quantitative summary</a> · <a href="per_image.csv">Per-image evidence</a> · <a href="provenance.json">Provenance</a></p>'''
    for task,case_id,title,path,reason in gallery:
        content+=f'<article><h2>{esc(title)}</h2><p>{esc(reason)}</p><img src="cases/{case_id}.png" alt="{esc(title)}"><code>{esc(path)}</code></article>'
    (out/'index.html').write_text(content+'</html>')
    # Compact manuscript gallery panels; full resolution cases remain in HTML.
    for task,num in [('dementia','8a'),('tumor','8b')]:
        cases=[g for g in gallery if g[0]==task]
        fig,axes=plt.subplots(len(cases),1,figsize=(10,3.7*len(cases)),squeeze=False)
        for ax,(_,case_id,*_) in zip(axes[:,0],cases):
            ax.imshow(Image.open(out/'cases'/f'{case_id}.png')); ax.axis('off')
        fig.tight_layout(); fig.savefig(out/f'figure{num}_gradcam.png',dpi=140); plt.close(fig)
    lines=['# Quantitative explainability summary','',
           'Exploratory, class-balanced strict-test sample; descriptive image-level statistics only. No patient-independent confidence intervals are inferred.','',
           '| Specialist | Sample / test population | Valid CAMs | Top-20% drop | Random-20% drop | Paired advantage | Border mass |',
           '| --- | --- | --- | --- | --- | --- | --- |']
    for task,s in summaries.items():
        values=[s['metrics'][k]['mean'] for k in ['top20_drop','random20_drop','paired_drop_advantage','border_mass']]
        lines.append(f"| {task} | {s['sample_n']} / {s['test_population']} | {s['valid_cam_n']} | "+' | '.join(f'{v:.4f}' for v in values)+' |')
    lines+=['','Drops are original minus perturbed predicted-class softmax score; negative values mean masking increased the score. Paired advantage is top-CAM drop minus the mean of the configured seeded random masks (default: five). This is a single-fraction perturbation diagnostic, not deletion AUC. Masked pixels use the ImageNet mean RGB baseline. Random masks are spatially scattered and are not shape-matched controls.','',
            'Border mass is the fraction of upsampled CAM mass in the outer 10% image frame (approximately 35.4% of pixels at 224×224). The image frame is not a brain/background segmentation. Neither this measure nor masking establishes lesion localization, causal biological relevance, absence of source bias, or clinical validity.','',
            'See per_image.csv for failures, selection roles, image hashes and archived-probability agreement; summary.json includes per-class counts, means and SDs. The diagnostic gallery is selected separately and excluded from quantitative aggregates unless an image also belongs to the seeded sample.']
    (out/'SUMMARY.md').write_text('\n'.join(lines)+'\n')
    (out/'RUNNING').unlink()
    print(json.dumps(summaries,indent=2),flush=True)


if __name__=='__main__':
    if '--followup' in sys.argv:
        sys.argv.remove('--followup')
        from src.explainability_validation import main as followup_main
        followup_main()
    else:
        main()
