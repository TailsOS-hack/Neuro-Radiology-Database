#!/usr/bin/env python3
"""Frozen validation-only temperature fitting followed by full strict-test research evaluation."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import experiment_pipeline as ep
from src.research_validation import (TASKS, checkpoint_path, fit_temperature, hierarchical,
    metrics, require, seal, sha, softmax, write_csv, write_json)


def run(args):
    import torch
    from PIL import Image
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    protocol = json.loads(args.protocol.read_text())
    rows = ep.read_manifest(args.manifest)
    selected = {split: [r for r in rows if r['split'] == split] for split in ['val', 'test']}
    for split, records in selected.items():
        require(len({r['path'] for r in records}) == len(records), 'Duplicate manifest paths')
        write_csv(out/f'{split}_index.csv', records)
    require(not {r['path'] for r in selected['val']} & {r['path'] for r in selected['test']}, 'Val/test overlap')
    torch.set_num_threads(4); torch.manual_seed(42)
    provenance = {'run_id': protocol['run_id'], 'protocol_sha256': sha(args.protocol),
                  'manifest_sha256': sha(args.manifest), 'torch': torch.__version__,
                  'device': args.device, 'checkpoints': {}, 'source_sha256': {
                      str(Path(__file__).relative_to(ROOT)): sha(__file__),
                      'src/research_validation.py': sha(ROOT/'src/research_validation.py')}}
    def export(task, split):
        cp = checkpoint_path(task, args.checkpoint_dir)
        model, ckpt = ep.load_checkpoint_model(cp, torch.device(args.device))
        require(ckpt['class_names'] == ep.TASK_CLASS_NAMES[task] and ckpt['task'] == task, 'Checkpoint classes/task mismatch')
        provenance['checkpoints'][task] = {'sha256': sha(cp), 'classes': ckpt['class_names'], 'image_size': ckpt['image_size']}
        transform = ep.build_transforms(False, ckpt['image_size'])
        values=[]
        with torch.inference_mode():
            for start in range(0, len(selected[split]), args.batch_size):
                batch=[]
                for row in selected[split][start:start+args.batch_size]:
                    with Image.open(ROOT/row['path']) as im: batch.append(transform(im.convert('RGB')))
                z=model(torch.stack(batch).to(args.device)).cpu().numpy()
                require(np.isfinite(z).all(), 'Nonfinite exported logits')
                values.append(z)
                if start % (args.batch_size*50) == 0:
                    print(f'{task} {split}: {start}/{len(selected[split])}',flush=True)
        logits=np.concatenate(values)
        np.save(out/f'{task}_{split}_logits.npy',logits)
        del model
        if args.device == 'mps': torch.mps.empty_cache()
        return logits
    fits={}
    for task in TASKS:
        z=export(task,'val')
        labels=np.array([ep.label_for_row(r,task) if ep.label_for_row(r,task) is not None else -1 for r in selected['val']])
        mask=labels>=0
        fits[task]=fit_temperature(z[mask],labels[mask],split='val',provenance={
            'checkpoint_sha256': provenance['checkpoints'][task]['sha256'],
            'validation_manifest_sha256': sha(out/'val_index.csv'),
            'validation_logits_sha256': sha(out/f'{task}_val_logits.npy'),
            'classes': ep.TASK_CLASS_NAMES[task]})
    # This artifact is frozen before any test inference; no test data enter fitting.
    write_json(out/'temperatures.json',fits)
    frozen_hash=sha(out/'temperatures.json')
    test={task:export(task,'test') for task in TASKS}
    require(sha(out/'temperatures.json') == frozen_hash,'Calibration changed during test evaluation')
    summary={}; probabilities={}
    for mode in ['original','calibrated']:
        probabilities[mode]={task:softmax(test[task],1 if mode=='original' else fits[task]['temperature']) for task in TASKS}
        for task in TASKS:
            labels=np.array([ep.label_for_row(r,task) if ep.label_for_row(r,task) is not None else -1 for r in selected['test']])
            mask=labels>=0
            summary.setdefault(task,{})[mode]=metrics(probabilities[mode][task][mask],labels[mask])
            require(np.array_equal(probabilities['original'][task].argmax(1),probabilities[mode][task].argmax(1)), 'Temperature changed component predictions')
        full,routed=hierarchical(*[probabilities[mode][t] for t in ['binary','tumor','dementia']])
        labels=np.array([ep.label_for_row(r,'eight_class') for r in selected['test']])
        summary.setdefault('hierarchical_argmax',{})[mode]=metrics(full,labels)
        summary.setdefault('hierarchical_routed',{})[mode]=metrics(full,labels,routed)
        np.save(out/f'hierarchical_{mode}_probabilities.npy',full)
        np.save(out/f'hierarchical_{mode}_routed.npy',routed)
    # Concordance with historical predictions is descriptive; never silently replace them.
    concordance={}
    from src.research_validation import read_csv
    for task in TASKS:
        archived=read_csv(ROOT/f'training_logs/publication_evidence/{task}/probabilities.csv')
        indexes={r['path']:i for i,r in enumerate(selected['test'])}
        expected={r['path'] for r in selected['test'] if ep.label_for_row(r,task) is not None}
        require({r['image_path'] for r in archived}==expected,'Archived test population mismatch')
        deltas=[];disagreements=0
        for row in archived:
            current=probabilities['original'][task][indexes[row['image_path']]]
            deltas.append(float(np.max(np.abs(current-np.array([float(row[f'prob_{c}']) for c in ep.TASK_CLASS_NAMES[task]])))))
            disagreements += int(current.argmax()!=int(row['pred_index']))
        concordance[task]={'max_probability_difference':max(deltas),'prediction_disagreements':disagreements,'n':len(archived)}
        require(disagreements==0 and max(deltas)<.001,'Accepted checkpoint inference drift exceeds review tolerance')
    provenance['frozen_temperatures_sha256']=frozen_hash
    provenance['concordance']=concordance
    write_json(out/'provenance.json',provenance);write_json(out/'summary.json',summary)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(12,8))
    for ax,(task,modes) in zip(axes.flat,summary.items()):
        ax.plot([0,1],[0,1],':',color='gray')
        for mode,vals in modes.items():
            b=[b for b in vals['bins'] if b['n']]
            ax.plot([v['confidence'] for v in b],[v['accuracy'] for v in b],marker='o',label=mode)
        ax.set(xlim=(0,1),ylim=(0,1),title=task.replace('_',' '),xlabel='Model confidence',ylabel='Observed correctness')
        ax.legend()
    fig.suptitle('Research-only calibration | '+protocol['run_id'])
    fig.tight_layout();fig.savefig(out/'reliability.png',dpi=150);plt.close(fig)
    seal(out)
    print('Calibration complete',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,default=ep.DEFAULT_MANIFEST)
    p.add_argument('--protocol',type=Path,default=ROOT/'docs/validation_protocol_v2.json')
    p.add_argument('--checkpoint-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',choices=['cpu','mps','cuda'],default='cpu')
    p.add_argument('--batch-size',type=int,default=32)
    run(p.parse_args())
