#!/usr/bin/env python3
"""Isolated dementia experiments. Prepare and smoke locally; train on approved VM.

Legacy stages and source clinical AD/CN diagnoses are distinct endpoints. Test
images never enter checkpoint selection. Production checkpoints are never written.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, math, os, random, time
from collections import Counter, defaultdict
from pathlib import Path

STAGES = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']
SCHEMA = 1


def csv_read(path):
    with Path(path).open(newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def csv_write(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text(''); return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''): h.update(b)
    return h.hexdigest()


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def identities(path):
    from PIL import Image
    with Image.open(path) as im:
        rgb = im.convert('RGB')
        pixel = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
        p = list(rgb.convert('L').resize((9, 8)).getdata())
        dh = 0
        for y in range(8):
            for x in range(8): dh = (dh << 1) | (p[y*9+x] > p[y*9+x+1])
    return pixel, f'{dh:016x}'


def resolve(root, p):
    p = Path(p)
    return p if p.is_absolute() else root / p


def prepare(args):
    root = args.data_root.resolve(); out = args.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    rows = [r for r in csv_read(resolve(root, args.legacy_manifest)) if r['domain'] == 'dementia']
    strict = {r['path']: r['split'] for r in csv_read(resolve(root, args.old_manifest)) if r['domain'] == 'dementia'}
    cohort = resolve(root, args.cohort_dir)
    cohort_paths = sorted(p for p in cohort.iterdir() if p.suffix.lower() in {'.jpg','.jpeg','.png'})
    if len(cohort_paths) != 100: raise ValueError(f'Expected original100 images; found {len(cohort_paths)}')
    banned_pixel, banned_dhash = set(), set()
    for p in cohort_paths:
        a,b = identities(p); banned_pixel.add(a); banned_dhash.add(b)
    parent = list(range(len(rows))); seen = {}
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    for i,r in enumerate(rows):
        a,b = identities(resolve(root,r['path'])); r.update(pixel_sha256=a,dhash=b)
        for k in [('pixel',a),('dhash',b)]:
            if k in seen:
                x,y=find(i),find(seen[k]); parent[y]=x
            else: seen[k]=i
        if (i+1)%5000==0: print(f'Hashed {i+1}/{len(rows)} legacy images',flush=True)
    groups=defaultdict(list)
    for i,r in enumerate(rows): groups[find(i)].append(r)
    excluded=[]; kept=[]; reassigned=0
    rank={'train':0,'val':1,'test':2}
    for g in groups.values():
        reason=''
        if any(r['pixel_sha256'] in banned_pixel or r['dhash'] in banned_dhash for r in g): reason='original100_pixel_or_dhash_connected_group'
        elif len({r['subtype'] for r in g}) != 1: reason='mixed_label_hash_group'
        if reason:
            excluded.extend(dict(r, exclusion_reason=reason) for r in g); continue
        # Preserve existing split, except move any newly connected overlap wholly
        # to the most protected split. Never move test/val data into training.
        split=max((r['split'] for r in g),key=rank.get)
        gid=min(r['pixel_sha256'] for r in g)
        for r in g:
            reassigned+=int(r['split']!=split)
            kept.append(dict(r, prior_split=r['split'],split=split,group_id=gid,
                             label=STAGES.index(r['subtype']),cohort='legacy_augmented_4stage',
                             baseline_strict_split=strict.get(r['path'], 'unknown')))
    csv_write(out/'stage_manifest.csv',kept)
    csv_write(out/'excluded_legacy.csv',excluded)
    comparison=[r for r in kept if r['split']=='test' and r['baseline_strict_split']=='test']
    csv_write(out/'same_case_test_intersection.csv',comparison)
    summary={
        'schema_version':SCHEMA,'source_rows':len(rows),'kept_rows':len(kept),'excluded_rows':len(excluded),
        'exclusion_counts':dict(Counter(r['exclusion_reason'] for r in excluded)),
        'excluded_by_split_class':dict(Counter(r['split']+'::'+r['subtype'] for r in excluded)),
        'remaining_by_split_class':dict(Counter(r['split']+'::'+r['subtype'] for r in kept)),
        'remaining_hash_groups_by_split':{s:len({r['group_id'] for r in kept if r['split']==s}) for s in rank},
        'newly_connected_rows_moved_to_protected_split':reassigned,
        'same_case_test_intersection_rows':len(comparison),
        'same_case_test_intersection_sha256':sha(out/'same_case_test_intersection.csv'),
        'same_case_test_intersection_by_class':dict(Counter(r['subtype'] for r in comparison)),
        'manifest_sha256':sha(out/'stage_manifest.csv'),
        'original100_image_count':len(cohort_paths),'test_used_for_selection':False,
        'limitations':[
            'Legacy patient IDs and upstream augmentation provenance remain unavailable.',
            'Exact-pixel/dHash grouping is a conservative image-similarity sensitivity analysis, not proof of patient or augmentation-family independence.',
            'Original100 was historically inspected and old checkpoint saw66 training images; excluding it now does not make the historical benchmark prospectively blind.',
            'Same-case intersection excludes old strict train/val rows, but cannot erase old checkpoint exposure to unidentified related images.',
            'dHash collisions can group different people; mixed-label groups are removed, and no disease label is inferred from image similarity.'
        ]}
    dump(out/'split_audit.json',summary); print(json.dumps(summary,indent=2))


def clinical_rows(root, path, image_root="data"):
    if not path: return []
    rows=csv_read(resolve(root,path)); subjects=defaultdict(set); labels=defaultdict(set)
    for r in rows:
        r['path']=str(Path(image_root) / (r.get('image_path') or r.get('path')))
        r['cohort']=r.get('dataset') or r.get('cohort')
        r['subject_id']=r.get('group_key') or r.get('subject_key') or (r['cohort']+':'+(r.get('subject_id') or r['source_subject_id']))
        diag=r['diagnosis']
        # Explicit clinical vocabulary only; no mapping of MCI to AD/CN.
        expected={'AD':'1','CN':'0'}.get(diag,'')
        given=str(r.get('binary_label','')).strip()
        if given and given not in {'0','1'}: raise ValueError('binary_label must be0,1,orblank')
        if given != expected: raise ValueError(f'Inconsistent diagnosis/binary label: {diag}/{given}')
        r['label']=int(given) if given else -1
        if r['split'] not in {'train','val','test','external','excluded'}: raise ValueError('Unknown clinical split')
        subjects[r['subject_id']].add(r['split']); labels[r['subject_id']].add(r['label'])
    if any(len(s)>1 for s in subjects.values()): raise ValueError('Clinical subject crosses splits')
    if any(len(s)>1 for s in labels.values()): raise ValueError('Clinical subject has inconsistent label')
    if any('openneuro' in r['cohort'].lower() and r['split']!='external' for r in rows):
        raise ValueError('OpenNeuro must remain wholly external')
    return rows


def imports():
    global np, torch, nn, models, transforms, DataLoader, Dataset
    import numpy as np
    import torch
    from torch import nn
    from torchvision import models, transforms
    from torch.utils.data import DataLoader, Dataset


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def device_for(name):
    if name=='auto': name='cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    return torch.device(name)


def network(pretrained=True, weights_file=None):
    model=models.mobilenet_v3_large(weights=None if weights_file or not pretrained else models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)
    if weights_file:
        if not sha(weights_file).startswith('8738ca79'): raise ValueError('Expected official MobileNetV3-Large ImageNet V1 checkpoint8738ca79')
        model.load_state_dict(torch.load(weights_file,map_location='cpu',weights_only=True))
    model.classifier[3]=nn.Linear(model.classifier[3].in_features,4)
    return model,nn.Linear(model.classifier[3].in_features,2)


def forward(model,aux,x,head):
    z=model.avgpool(model.features(x)).flatten(1)
    z=model.classifier[:-1](z)
    return model.classifier[3](z) if head=='stage' else aux(z)


class ImageRows:
    def __init__(self, rows, root, tf): self.rows,self.root,self.tf=rows,root,tf
    def __len__(self): return len(self.rows)
    def __getitem__(self,i):
        from PIL import Image
        r=self.rows[i]
        with Image.open(resolve(self.root,r['path'])) as im: x=self.tf(im.convert('RGB'))
        return x,int(r['label']),i


def dataset(rows,root,train,image_size):
    steps=[transforms.Resize((image_size,image_size),antialias=True)]
    if train: steps += [transforms.RandomHorizontalFlip(),transforms.RandomRotation(10),transforms.ColorJitter(brightness=.08,contrast=.12)]
    steps += [transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])]
    return ImageRows(rows,root,transforms.Compose(steps))


def metrics(y,p):
    from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score,confusion_matrix,log_loss,roc_auc_score
    p=np.array(p,dtype=float); p=p/p.sum(axis=1,keepdims=True); y=np.array(y); n=p.shape[1]; pred=p.argmax(1)
    d={'n':int(len(y)),'loss':float(log_loss(y,p,labels=list(range(n)))),'accuracy':float(accuracy_score(y,pred)),
       'balanced_accuracy':float(balanced_accuracy_score(y,pred)),'macro_f1':float(f1_score(y,pred,labels=list(range(n)),average='macro',zero_division=0)),
       'confusion_matrix':confusion_matrix(y,pred,labels=list(range(n))).tolist()}
    if n==2 and len(set(y))==2: d['auroc']=float(roc_auc_score(y,p[:,1]))
    return d


def evaluate(model,aux,rows,root,device,head,args):
    loader=DataLoader(dataset(rows,root,False,args.image_size),batch_size=args.batch_size,num_workers=args.workers,pin_memory=device.type=='cuda')
    model.eval();aux.eval(); probs=[]
    with torch.inference_mode():
        for x,_,_ in loader: probs.extend(forward(model,aux,x.to(device),head).softmax(1).cpu().numpy().tolist())
    records=[]
    for r,p in zip(rows,probs):
        records.append(dict(path=r['path'],cohort=r['cohort'],subject_id=r.get('subject_id',''),label=int(r['label']),probabilities=p))
    if head=='aux':
        grouped=defaultdict(list)
        for r in records: grouped[(r['cohort'],r['subject_id'])].append(r)
        records=[dict(cohort=c,subject_id=s,label=g[0]['label'],probabilities=np.mean([r['probabilities'] for r in g],axis=0).tolist(),n_slices=len(g)) for (c,s),g in grouped.items()]
    eligible=[r for r in records if r['label']>=0]
    if not eligible: return {'n':0},records
    result=metrics([r['label'] for r in eligible],[r['probabilities'] for r in eligible])
    result['unit']='subject' if head=='aux' else 'legacy_image'
    result['per_source']={c:metrics([r['label'] for r in eligible if r['cohort']==c],[r['probabilities'] for r in eligible if r['cohort']==c]) for c in sorted({r['cohort'] for r in eligible})}
    return result,records


def curve(out,history,smoke):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for split in ['train','val']:
        axes[0].plot([r['epoch'] for r in history],[r[split+'_stage_loss'] for r in history],label=split)
        axes[1].plot([r['epoch'] for r in history],[r.get(split+'_aux_loss',float('nan')) for r in history],label=split)
    for ax,title in zip(axes,['Legacy4-stage cross-entropy','Clinical AD/CN loss: train slices / val subjects']):
        ax.set_title(title);ax.set_xlabel('Epoch');ax.legend();ax.grid(alpha=.2)
    fig.suptitle('SMOKE TEST ONLY — not research performance' if smoke else 'Raw epoch losses; augmentation on training only; validation selects checkpoints')
    fig.tight_layout();fig.savefig(out/'loss_curves.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for split in ['train','val']:
        for ax,head in zip(axes,['stage','aux']):
            ax.plot([r['epoch'] for r in history],[r.get(split+'_'+head+'_accuracy',float('nan')) for r in history],label=split)
    for ax,title in zip(axes,['Legacy4-stage accuracy','Clinical accuracy: train slices / val subjects']):
        ax.set_title(title);ax.set_xlabel('Epoch');ax.set_ylim(0,1);ax.legend();ax.grid(alpha=.2)
    fig.suptitle('SMOKE TEST ONLY — not research performance' if smoke else 'Raw epoch accuracies; test outcomes do not select checkpoints')
    fig.tight_layout();fig.savefig(out/'accuracy_curves.png',dpi=150);plt.close(fig)


def atomic_save(obj,path):
    tmp=path.with_suffix('.tmp');torch.save(obj,tmp);os.replace(tmp,path)


def train(args):
    imports();root=args.data_root.resolve();out=args.output.resolve()
    if (out/'last.pt').exists() and not args.resume: raise ValueError('Output already has a checkpoint; choose a new output or --resume')
    if args.resume and args.resume.resolve().parent!=out: raise ValueError('Resume output must equal checkpoint parent; preserves prior best.pt provenance')
    out.mkdir(parents=True,exist_ok=True);device=device_for(args.device);seed_all(args.seed)
    stage=csv_read(resolve(root,args.stage_manifest))
    for r in stage: r['label']=int(r['label'])
    clinical=clinical_rows(root,args.clinical_manifest,args.clinical_image_root)
    tr=[r for r in stage if r['split']=='train'];va=[r for r in stage if r['split']=='val']
    atr=[r for r in clinical if r['split']=='train' and r['label']>=0];ava=[r for r in clinical if r['split']=='val' and r['label']>=0]
    if not tr or not va: raise ValueError('Missing stage train or validation data')
    if args.mode=='multitask' and (not atr or not ava): raise ValueError('Multitask requires clinical subject-disjoint train and validation AD/CN')
    for s in ('train','val','test'):
        g={r['group_id'] for r in stage if r['split']==s}
        for t in ('train','val','test'):
            if t!=s and g & {r['group_id'] for r in stage if r['split']==t}: raise ValueError('Legacy hash group crosses splits')
    if set(r['label'] for r in tr)!=set(range(4)) or set(r['label'] for r in va)!=set(range(4)): raise ValueError('All four stage classes required in training and validation')
    if args.smoke: va=[next(r for r in stage if r['split']=='val' and r['label']==l) for l in range(4)]
    model,aux=network(pretrained=not bool(args.resume),weights_file=None if args.resume else args.imagenet_weights);model.to(device);aux.to(device)
    opt=torch.optim.AdamW(list(model.parameters())+(list(aux.parameters()) if args.mode=='multitask' else []),lr=args.learning_rate,weight_decay=args.weight_decay)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=args.epochs,eta_min=args.learning_rate*.05)
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    config.update(schema_version=SCHEMA,torch=torch.__version__,device_actual=str(device),
                  stage_manifest_sha256=sha(resolve(root,args.stage_manifest)),
                  clinical_manifest_sha256=sha(resolve(root,args.clinical_manifest)) if args.clinical_manifest else None,
                  selection='minimum stage validation NLL; clinical auxiliary validation is reported but never used to choose stage checkpoint',
                  training_stage_rows=len(tr),validation_stage_rows=len(va),training_aux_subjects=len({r['subject_id'] for r in atr}),
                  validation_aux_subjects=len({r['subject_id'] for r in ava}),
                  baseline_checkpoint_used_for_initialization=False,test_used_for_selection=False,
                  imagenet_weights_url=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1.url,
                  claim='SMOKE_ONLY' if args.smoke else 'research_candidate_not_deployed')
    start=0;best=float('inf');bad=0;history=[]
    if args.resume:
        ck=torch.load(args.resume,map_location='cpu',weights_only=False)
        for k in ['mode','seed','stage_manifest_sha256','clinical_manifest_sha256','image_size','batch_size','aux_weight','aux_every','learning_rate','weight_decay','epochs','stage_draws','smoke','label_smoothing','min_delta','min_epochs','patience','clinical_image_root']:
            if ck['config'].get(k)!=config.get(k): raise ValueError(f'Resume config mismatch: {k}')
        model.load_state_dict(ck['model_state']);aux.load_state_dict(ck['aux_state']);opt.load_state_dict(ck['optimizer_state']);sched.load_state_dict(ck['scheduler_state'])
        start=ck['epoch'];history=ck['history'];best=ck['best_val_loss'];bad=ck['bad_epochs']
        # Each epoch is explicitly reseeded below. Saved RNG states retain full provenance.
    dump(out/'config.json',config)
    group_counts=Counter(r['group_id'] for r in tr);class_groups=Counter({})
    group_labels={r['group_id']:r['label'] for r in tr}
    class_groups.update(group_labels.values())
    weights=[1/(group_counts[r['group_id']]*class_groups[r['label']]) for r in tr]
    # Full pass is default, with inverse group-size and class-group loss weights.
    # Reduced stochastic draw budgets are explicit opt-ins only.
    scale=len(weights)/sum(weights); weights=[v*scale for v in weights]
    aux_by_subject=defaultdict(list)
    for r in atr: aux_by_subject[r['subject_id']].append(r)
    aux_by_label={l:[s for s,g in aux_by_subject.items() if g[0]['label']==l] for l in [0,1]}
    if args.mode=='multitask' and any(not v for v in aux_by_label.values()): raise ValueError('Both clinical classes required in train')
    ds=dataset(tr,root,True,args.image_size)
    ce=nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    for epoch in range(start,args.epochs):
        seed_all(args.seed+epoch); t=time.time(); model.train();aux.train()
        draws=min(args.batch_size*args.smoke_batches,len(tr)) if args.smoke else (args.stage_draws or len(tr))
        gen=torch.Generator().manual_seed(args.seed+epoch)
        sampler=torch.utils.data.RandomSampler(ds,replacement=True,num_samples=draws,generator=gen) if args.smoke or args.stage_draws else None
        loader=DataLoader(ds,batch_size=args.batch_size,sampler=sampler,shuffle=sampler is None,num_workers=args.workers,pin_memory=device.type=='cuda',generator=gen)
        stage_loss=0.;stage_n=0;aux_loss=0.;aux_n=0;stage_correct=0;aux_correct=0
        for step,(x,y,indices) in enumerate(loader):
            x=x.to(device);y=y.to(device);opt.zero_grad(set_to_none=True)
            logits=forward(model,aux,x,'stage')
            ww=torch.tensor([weights[i] for i in indices.tolist()],device=device)
            loss=(nn.functional.cross_entropy(logits,y,label_smoothing=args.label_smoothing,reduction='none')*ww).mean()
            stage_correct+=int((logits.detach().argmax(1)==y).sum().item())
            stage_loss+=nn.functional.cross_entropy(logits.detach(),y).item()*len(y);stage_n+=len(y)
            if args.mode=='multitask' and step%args.aux_every==0:
                chosen=[]
                for k in range(args.batch_size):
                    sid=random.choice(aux_by_label[k%2]);chosen.append(random.choice(aux_by_subject[sid]))
                ads=dataset(chosen,root,True,args.image_size)
                xx=torch.stack([ads[i][0] for i in range(len(chosen))]).to(device)
                yy=torch.tensor([r['label'] for r in chosen],device=device)
                alog=forward(model,aux,xx,'aux');loss=loss+args.aux_weight*ce(alog,yy)
                aux_loss+=nn.functional.cross_entropy(alog.detach(),yy).item()*len(yy);aux_n+=len(yy)
                aux_correct+=int((alog.detach().argmax(1)==yy).sum().item())
            loss.backward();nn.utils.clip_grad_norm_(list(model.parameters())+list(aux.parameters()),5.);opt.step()
            if (step+1)%100==0: print(f'Epoch{epoch+1} batch{step+1}/{len(loader)}',flush=True)
        vm,_=evaluate(model,aux,va,root,device,'stage',args)
        am,_=evaluate(model,aux,ava,root,device,'aux',args) if args.mode=='multitask' else ({},[])
        row={'epoch':epoch+1,'train_stage_loss':stage_loss/stage_n,'val_stage_loss':vm['loss'],
             'val_stage_accuracy':vm['accuracy'],'val_stage_balanced_accuracy':vm['balanced_accuracy'],
             'train_stage_accuracy':stage_correct/stage_n,'stage_draws':stage_n,'aux_draws':aux_n,'learning_rate':opt.param_groups[0]['lr'],'seconds':time.time()-t}
        if aux_n: row.update(train_aux_loss=aux_loss/aux_n,train_aux_accuracy=aux_correct/aux_n,val_aux_loss=am['loss'],val_aux_accuracy=am['accuracy'],val_aux_balanced_accuracy=am['balanced_accuracy'])
        history.append(row); improved=vm['loss'] < best-args.min_delta
        if improved:best=vm['loss'];bad=0
        else:bad+=1
        sched.step()
        ck={'task':'dementia','arch':'mobilenet_v3_large','class_names':STAGES,'image_size':args.image_size,
            'model_state':model.state_dict(),'aux_state':aux.state_dict(),'aux_classes':['CN','AD'],
            'optimizer_state':opt.state_dict(),'scheduler_state':sched.state_dict(),'epoch':epoch+1,
            'best_val_loss':best,'bad_epochs':bad,'history':history,'config':config,
            'rng_state':{'torch':torch.get_rng_state(),'python':random.getstate(),'numpy':np.random.get_state()},
            'stage_validation':vm,'aux_validation':am,'smoke_only':args.smoke}
        atomic_save(ck,out/'last.pt')
        if improved: atomic_save(ck,out/'best.pt')
        dump(out/'history.json',history);csv_write(out/'history.csv',history)
        dump(out/f'epoch_{epoch+1:03d}_validation.json',{'stage':vm,'aux':am});curve(out,history,args.smoke)
        print(json.dumps(row),flush=True)
        if args.smoke: break
        if epoch+1>=args.min_epochs and bad>=args.patience: break
    print(f'Checkpoint: {out}/best.pt; production model untouched.',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('prepare');a.add_argument('--data-root',type=Path,required=True);a.add_argument('--output',type=Path,required=True)
    a.add_argument('--legacy-manifest',default='training_logs/splits/perceptual_strict_manifest.csv')
    a.add_argument('--old-manifest',default='training_logs/splits/strict_manifest.csv');a.add_argument('--cohort-dir',default='data/evaluation/images')
    a=sub.add_parser('train');a.add_argument('--data-root',type=Path,required=True);a.add_argument('--output',type=Path,required=True)
    a.add_argument('--stage-manifest',required=True);a.add_argument('--clinical-manifest');a.add_argument('--clinical-image-root',default='data',help='Clinical image-path prefix relative to repository data-root');a.add_argument('--mode',choices=['stage_only','multitask'],default='stage_only')
    a.add_argument('--device',choices=['auto','cpu','cuda','mps'],default='auto');a.add_argument('--workers',type=int,default=0)
    a.add_argument('--epochs',type=int,default=100);a.add_argument('--min-epochs',type=int,default=20);a.add_argument('--patience',type=int,default=15);a.add_argument('--min-delta',type=float,default=1e-4)
    a.add_argument('--batch-size',type=int,default=32);a.add_argument('--image-size',type=int,default=224);a.add_argument('--learning-rate',type=float,default=1e-4);a.add_argument('--weight-decay',type=float,default=.001)
    a.add_argument('--label-smoothing',type=float,default=.05);a.add_argument('--seed',type=int,default=20261011)
    a.add_argument('--aux-weight',type=float,default=.25);a.add_argument('--aux-every',type=int,default=4)
    a.add_argument('--stage-draws',type=int,default=0,help='0=every training row once per epoch; nonzero is an explicit reduced stochastic budget')
    a.add_argument('--imagenet-weights',type=Path,help='Optional portable cached torchvision MobileNetV3-Large V1 .pth; otherwise downloads officially')
    a.add_argument('--resume',type=Path);a.add_argument('--smoke',action='store_true');a.add_argument('--smoke-batches',type=int,default=2)
    args=p.parse_args()
    if args.command=='prepare':prepare(args)
    else:train(args)
if __name__=='__main__':main()
