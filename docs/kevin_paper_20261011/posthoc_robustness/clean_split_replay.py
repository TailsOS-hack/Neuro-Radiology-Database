#!/usr/bin/env python3
"""Post-hoc final-checkpoint train/full-test replay for descriptive paper figures.

Train is resubstitution in evaluation mode without random augmentation, not the
logged online training metric. Full new-test contains the primary1058 and is not
an additional independent cohort. No selection or training occurs here.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
from PIL import Image
import torch
import stress_test as s


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--results',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',choices=['mps','cpu'],default='mps')
    parser.add_argument('--batch-size',type=int,default=32)
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args();root=args.root.resolve();results=args.results.resolve();out=args.output.resolve()
    if out.exists():raise ValueError('A new output directory is required')
    if args.device=='mps' and not torch.backends.mps.is_available():raise ValueError('MPS is unavailable')
    _,checkpoints,identities,_,protected,selection=s.prepare(root,results)
    sys.path.insert(0,str(root/'docs/kevin_paper_20261011/training'))
    import train_candidate as c
    import evaluate_candidates as e
    c.imports();c.seed_all(s.SEED);device=torch.device(args.device)
    stagepath=root/'training_logs/experiments/kevin_20261011/prepared/stage_manifest.csv'
    allrows=c.csv_read(stagepath)
    splits={name:[r for r in allrows if r['split']==key] for name,key in [('train_resubstitution','train'),('full_new_test','test')]}
    if {k:len(v) for k,v in splits.items()}!={'train_resubstitution':19023,'full_new_test':5450}:raise ValueError('Unexpected frozen split counts')
    for rows in splits.values():
        if len({r['path'] for r in rows})!=len(rows):raise ValueError('Duplicate path')
        if {int(r['label']) for r in rows}!=set(range(4)):raise ValueError('Class vocabulary incomplete')
        for r in rows:
            if r['subtype']!=s.STAGES[int(r['label'])]:raise ValueError('Stage label mismatch')
    if {r['group_id'] for r in splits['train_resubstitution']} & {r['group_id'] for r in splits['full_new_test']}:raise ValueError('Train/test group overlap')
    out.mkdir(parents=True)
    meta=dict(status='running_posthoc_clean_split_replay',created_utc=datetime.now(timezone.utc).isoformat(),
              script_sha256=s.sha(Path(__file__)),stress_helper_sha256=s.sha(Path(s.__file__)),
              class_order=s.STAGES,device=args.device,torch=torch.__version__,batch_size=args.batch_size,workers=args.workers,
              stage_manifest_sha256=s.sha(stagepath),split_counts={k:len(v) for k,v in splits.items()},
              model_order=[name for name in s.ORDER if name!='old_baseline'],checkpoints={},
              frozen_candidate_identities={name:identities[name] for name in s.ORDER if name!='old_baseline'},protected_inputs_sha256=protected,
              original_validation_preference=selection['exploratory_representative_candidate'],checkpoint_reselection=False,
              primary1058_unchanged=True,training=False,random_augmentation=False,
              interpretations={'train_resubstitution':'Final selected checkpoint scored on its training images in evaluation mode without augmentation. Resubstitution; not a generalization estimate or the online epoch training metric.','full_new_test':'All5450 images in the frozen new test split. Contains the primary1058; post-hoc supplementary evaluation, not a replacement primary or an independent additional cohort. Legacy patient IDs unavailable.'})
    s.dump(out/'protocol_before_inference.json',meta)
    image_checks=[]
    for split,rows in splits.items():
        for r in rows:
            p=root/r['path']
            with Image.open(p) as im:
                rgb=im.convert('RGB');decoded=hashlib.sha256(str(rgb.size).encode()+rgb.tobytes()).hexdigest()
            if decoded!=r['pixel_sha256']:raise ValueError(f'Decoded source changed: {r["path"]}')
            image_checks.append(dict(path=r['path'],split=split,file_sha256=s.sha(p),decoded_rgb_sha256=decoded))
    s.dump(out/'input_image_checksums.json',image_checks)
    t=time.monotonic();args.image_size=224
    for name in meta['model_order']:
        model,aux,ck=e.load(checkpoints[name],device);record=identities[name]
        if ck['epoch']!=record['epoch'] or ck['config']['mode']!=record['mode'] or ck['config']['seed']!=record['seed']:raise ValueError('Checkpoint metadata mismatch')
        if int(ck.get('image_size',224))!=224:raise ValueError('Unexpected checkpoint input size')
        info=dict(checkpoint_sha256=s.sha(checkpoints[name]),mode=record['mode'],seed=record['seed'],selected_epoch=record['epoch'],splits={})
        for split,rows in splits.items():
            mt,pred=c.evaluate(model,aux,rows,root,device,'stage',args)
            if len(pred)!=len(rows):raise ValueError('Incomplete split predictions')
            p=c.np.array([z['probabilities'] for z in pred]);s.check_probs(p)
            exports=[]
            for row,z,pp in zip(rows,pred,p):
                if row['path']!=z['path'] or int(row['label'])!=z['label']:raise ValueError('Output row alignment mismatch')
                rec=dict(path=row['path'],TrueCategory=s.STAGES[int(row['label'])],ModelPrediction=s.STAGES[int(pp.argmax())])
                rec.update({f'P_{cl}':float(v) for cl,v in zip(s.STAGES,pp)});exports.append(rec)
            path=out/f'{name}_{split}_predictions.csv';s.csv_write(path,exports)
            info['splits'][split]=dict(metrics=mt,predictions_file=path.name,predictions_sha256=s.sha(path),n=len(rows))
            print(json.dumps(dict(model=name,split=split,n=len(rows),seconds=round(time.monotonic()-t,2))),flush=True)
        meta['checkpoints'][name]=info
        s.dump(out/'replay_progress.json',meta)
        del model,aux,ck
        if args.device=='mps':torch.mps.empty_cache()
    for p,digest in protected.items():
        if s.sha(p)!=digest:raise ValueError(f'Protected input changed: {p}')
    for r in image_checks:
        if s.sha(root/r['path'])!=r['file_sha256']:raise ValueError('Source image changed during replay')
    meta.update(status='completed_posthoc_clean_split_replay',finished_utc=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.monotonic()-t,protected_input_rechecks_passed=True)
    s.dump(out/'replay_metadata.json',meta)
    manifest=[dict(path=str(p.relative_to(out)),sha256=s.sha(p),bytes=p.stat().st_size) for p in sorted(out.rglob('*')) if p.is_file()]
    s.dump(out/'output_checksums.json',manifest)
    for row in manifest:
        p=out/row['path']
        if s.sha(p)!=row['sha256'] or p.stat().st_size!=row['bytes']:raise ValueError('Output checksum readback failed')
    s.dump(out/'replay_complete.json',dict(status=meta['status'],output_manifest_sha256=s.sha(out/'output_checksums.json'),local_readback_passed=True,cloud_resources_used=False))
    print('CLEAN_SPLIT_REPLAY_COMPLETE',flush=True)


if __name__=='__main__':main()
