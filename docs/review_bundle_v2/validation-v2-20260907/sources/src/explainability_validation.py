"""Frozen follow-up protocol: perturbation controls and model randomization."""
from __future__ import annotations
import argparse
import hashlib
import json
import shutil
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from src.research_validation import ROOT, require, read_csv, write_csv, write_json, sha, describe, checkpoint_path, seal


def top_mask(cam, fraction):
    require(cam.ndim == 2 and np.isfinite(cam).all(), 'Invalid map')
    count = int(fraction*cam.size)
    require(0 < count < cam.size, 'Invalid masking fraction')
    mask=np.zeros(cam.size,dtype=bool)
    mask[np.argsort(-cam.ravel(),kind='stable')[:count]]=True
    return mask.reshape(cam.shape)


def shape_controls(mask, seed, limit=5):
    variants=[];seen={mask.tobytes()}
    for flipped in [False,True]:
        for rotation in range(4):
            result=np.ascontiguousarray(np.rot90(np.fliplr(mask) if flipped else mask,rotation))
            key=result.tobytes()
            if key not in seen:
                seen.add(key);variants.append((f'flip{int(flipped)}_rot{rotation}',result))
    rng=np.random.default_rng(seed)
    indexes=rng.permutation(len(variants))[:limit]
    return [variants[int(i)] for i in indexes]


def random_controls(mask, seed, count=5):
    rng=np.random.default_rng(seed);result=[]
    for _ in range(count):
        m=np.zeros(mask.size,dtype=bool);m[rng.choice(mask.size,int(mask.sum()),replace=False)]=True
        result.append(m.reshape(mask.shape))
    return result


def map_similarity(original, randomized):
    require(original.shape == randomized.shape, 'Map shape mismatch')
    if np.ptp(randomized)<1e-12 or np.ptp(original)<1e-12:
        return None,None
    rho=float(spearmanr(original.ravel(),randomized.ravel()).statistic)
    a=top_mask(original,.2);b=top_mask(randomized,.2)
    return rho,float((a&b).sum()/(a|b).sum())


def grouped_summary(rows, keys, metrics):
    groups={}
    for row in rows:
        for label in [row['true_label'],'ALL']:
            key='|'.join(str(row[k]) for k in keys)+'|'+label
            groups.setdefault(key,[]).append(row)
    output={}
    for key,group in sorted(groups.items()):
        output[key]={'n':len(group),'failures':sum(r.get('status','ok')!='ok' for r in group),
                     'metrics':{m:describe([r[m] for r in group if r[m] is not None]) for m in metrics}}
    return output


def run(args):
    import torch
    import torch.nn.functional as F
    from torchvision.transforms.functional import gaussian_blur
    from PIL import Image
    from src import experiment_pipeline as ep
    from src.grad_cam import GradCAM,get_target_layer
    out=args.output;out.mkdir(parents=True,exist_ok=False)
    (out/'maps').mkdir();(out/'randomized_maps').mkdir()
    protocol=json.loads(args.protocol.read_text());baseline=ROOT/protocol['baseline']
    source_rows=read_csv(baseline/'per_image.csv')
    write_csv(out/'selection.csv',source_rows)
    samples=[r for r in source_rows if r['quantitative_sample']=='1']
    require(len(samples)==256,'Frozen sample must contain 256 images')
    provenance={'run_id':protocol['run_id'],'protocol_sha256':sha(args.protocol),
                'selection_sha256':sha(baseline/'per_image.csv'),'device':args.device,
                'torch':torch.__version__,'checkpoints':{},'randomized_state_sha256':{},
                'source_sha256':{'src/explainability_validation.py':sha(__file__),
                                 'src/grad_cam.py':sha(ROOT/'src/grad_cam.py')}}
    torch.set_num_threads(4)
    perturbations=[];randomizations=[]
    for task in ['tumor','dementia']:
        cp=checkpoint_path(task,args.checkpoint_dir)
        model,ckpt=ep.load_checkpoint_model(cp,torch.device(args.device))
        require(ckpt['class_names']==ep.TASK_CLASS_NAMES[task], 'Class order mismatch')
        provenance['checkpoints'][task]={'sha256':sha(cp),'classes':ckpt['class_names']}
        transform=ep.build_transforms(False,ckpt['image_size'])
        task_samples=[r for r in samples if r['task']==task]
        cache=[]
        for i,row in enumerate(task_samples):
            require(sha(ROOT/row['path'])==row['image_sha256'],'Frozen image changed')
            with Image.open(ROOT/row['path']) as image:
                x=transform(image.convert('RGB')).unsqueeze(0).to(args.device)
            target=ckpt['class_names'].index(row['predicted_label'])
            with torch.no_grad():prob=model(x).softmax(1)[0]
            require(int(prob.argmax())==target and abs(float(prob[target])-float(row['confidence']))<1e-4,'Accepted probability mismatch')
            native=np.load(baseline/row['cam_path'],allow_pickle=False)
            require(np.isfinite(native).all() and np.ptp(native)>0,'Invalid baseline CAM')
            name=Path(row['cam_path']).name
            np.save(out/'maps'/name,native)
            cache.append((row,x.detach().cpu(),native,target,name))
            up=F.interpolate(torch.from_numpy(native)[None,None],size=x.shape[-2:],mode='bilinear',align_corners=False)[0,0].numpy()
            blurred=gaussian_blur(x,[protocol['blur_kernel']]*2,[protocol['blur_sigma']]*2)
            for fraction in protocol['fractions']:
                seed=42+int(hashlib.sha256((row['path']+str(fraction)).encode()).hexdigest()[:12],16)
                top=top_mask(up,fraction);shapes=shape_controls(top,seed)
                randoms=random_controls(top,seed)
                masks=[top]+randoms+[m for _,m in shapes]
                for replacement,fill in [('imagenet_mean',torch.zeros_like(x)),('gaussian_blur',blurred)]:
                    batch=x.repeat(len(masks),1,1,1)
                    for j,mask in enumerate(masks):
                        m=torch.from_numpy(mask).to(args.device);batch[j,:,m]=fill[0,:,m]
                    with torch.no_grad():scores=model(batch).softmax(1)[:,target].cpu().numpy()
                    drops=float(prob[target])-scores
                    top_drop=float(drops[0]);random_drop=float(drops[1:6].mean())
                    shape_drop=float(drops[6:].mean()) if shapes else None
                    perturbations.append({'task':task,'path':row['path'],'true_label':row['true_label'],
                        'target':target,'fraction':fraction,'replacement':replacement,'seed':seed,
                        'shape_count':len(shapes),'shape_ids':json.dumps([n for n,_ in shapes]),
                        'top_drop':top_drop,'random_drops':json.dumps(drops[1:6].tolist()),
                        'shape_drops':json.dumps(drops[6:].tolist()),'random_drop':random_drop,
                        'shape_drop':shape_drop,'random_advantage':top_drop-random_drop,
                        'shape_advantage':top_drop-shape_drop if shapes else None})
            if i%16==0:print(f'{task} controls {i+1}/128',flush=True)
        original={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        state_hash=lambda state:hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for k,v in sorted(state.items()))).hexdigest()
        original_hash=state_hash(original)
        for seed in protocol['randomization_seeds']:
            torch.manual_seed(seed)
            fresh=ep.build_model(ckpt['arch'],len(ckpt['class_names']),False).state_dict()
            for stage in protocol['randomization_stages']:
                prefix=f'features.{len(model.features)-1}.'
                state={k:(fresh[k] if stage=='full' or k.startswith('classifier.') or
                          (stage=='head_final' and k.startswith(prefix)) else v) for k,v in original.items()}
                digest=state_hash(state);require(digest!=original_hash,'Randomization did not change weights')
                provenance['randomized_state_sha256'][f'{task}|{seed}|{stage}']=digest
                model.load_state_dict(state);model.eval()
                for row,x,native,target,name in cache:
                    cam=GradCAM(model,get_target_layer(model)).generate(x.to(args.device),target)
                    rho,iou=map_similarity(native,cam) if cam is not None else (None,None)
                    map_name=f'{Path(name).stem}_{seed}_{stage}.npy' if cam is not None else ''
                    if cam is not None:np.save(out/'randomized_maps'/map_name,cam)
                    randomizations.append({'task':task,'path':row['path'],'true_label':row['true_label'],
                        'target':target,'seed':seed,'stage':stage,'status':'ok' if rho is not None else 'flat_or_failed',
                        'spearman':rho,'top20_iou':iou,'map':map_name})
                print(f'{task} randomization {seed} {stage} complete',flush=True)
        model.load_state_dict(original)
        require(state_hash(model.state_dict())==original_hash,'Accepted weights changed')
        require(sha(cp)==provenance['checkpoints'][task]['sha256'],'Checkpoint changed on disk')
        del model,cache
    write_csv(out/'perturbations.csv',perturbations);write_csv(out/'randomizations.csv',randomizations)
    summary={'perturbations':grouped_summary(perturbations,['task','fraction','replacement'],
                  ['top_drop','random_drop','shape_drop','random_advantage','shape_advantage','shape_count']),
             'randomizations':grouped_summary(randomizations,['task','seed','stage'],['spearman','top20_iou'])}
    write_json(out/'summary.json',summary);write_json(out/'provenance.json',provenance)
    # Preserve original gallery selection and label its inherited origin explicitly.
    shutil.copytree(baseline/'cases',out/'cases')
    gallery=[r for r in source_rows if r['gallery_reason']]
    html='<html><meta charset="utf-8"><title>Follow-up gallery</title><body><h1>'+protocol['run_id']+'</h1><p>Fixed baseline gallery; already inspected public test data. No anatomical or external validation.</p>'
    for row in gallery:
        html+=f'<img style="width:100%;max-width:1100px" src="cases/{Path(row["cam_path"]).stem}.png">'
    (out/'gallery.html').write_text(html+'</body></html>')
    plot(out,summary,protocol['run_id'])
    seal(out)


def plot(out,summary,run_id):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for ax,task in zip(axes,['tumor','dementia']):
        for replacement in ['imagenet_mean','gaussian_blur']:
            for control in ['random','shape']:
                points=[summary['perturbations'][f'{task}|{f}|{replacement}|ALL']['metrics'][control+'_advantage']['mean'] for f in [.1,.2,.3]]
                ax.plot([10,20,30],points,marker='o',label=f'{replacement} / {control}')
        ax.axhline(0,color='gray',linestyle=':');ax.set(title=task,xlabel='Pixels masked (%)',ylabel='Top-CAM minus control confidence drop');ax.legend(fontsize=7)
    fig.suptitle('Follow-up masking controls | '+run_id);fig.tight_layout();fig.savefig(out/'figure9_controls.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for ax,task in zip(axes,['tumor','dementia']):
        for seed in [42,43,44]:
            values=[summary['randomizations'][f'{task}|{seed}|{stage}|ALL']['metrics']['spearman']['mean'] for stage in ['head','head_final','full']]
            ax.plot(['head','head + final','full'],values,marker='o',label=f'seed {seed}')
        ax.set(title=task,ylabel='Native CAM Spearman correlation',ylim=(-1,1));ax.legend()
    fig.suptitle('Randomization diagnostics (valid maps only) | '+run_id);fig.tight_layout();fig.savefig(out/'figure10_randomization.png',dpi=150);plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol',type=Path,default=ROOT/'docs/validation_protocol_v2.json')
    p.add_argument('--checkpoint-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',choices=['cpu','mps','cuda'],default='cpu')
    run(p.parse_args())
