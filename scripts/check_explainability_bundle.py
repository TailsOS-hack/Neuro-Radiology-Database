#!/usr/bin/env python3
"""Verify frozen baseline artifacts; --full additionally requires original MRI files."""
import argparse
import json
import random
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.research_validation import read_csv,sha,require,describe
OUT=ROOT/'docs/review_bundle/explainability'


def check(full=False, out=OUT):
    require(not (out/'RUNNING').exists(),'Incomplete baseline generation')
    provenance=json.loads((out/'provenance.json').read_text());summary=json.loads((out/'summary.json').read_text())
    snapshots=ROOT/'docs/validation_baseline_sources'
    require(sha(snapshots/'build_explainability_bundle.py')==provenance['script_sha256'],'Historical builder snapshot mismatch')
    for name,digest in provenance['source_sha256'].items():
        require(sha(snapshots/Path(name).name)==digest,f'Historical source mismatch: {name}')
    manifest_path=ROOT/'training_logs/splits/strict_manifest.csv'
    require(provenance['manifest_sha256']==sha(manifest_path),'Manifest changed')
    manifest={r['path']:r for r in read_csv(manifest_path)}
    rows=read_csv(out/'per_image.csv')
    require(len({(r['task'],r['path']) for r in rows})==len(rows),'Duplicate cases')
    for task,s in summary.items():
        all_task=[r for r in rows if r['task']==task];sample=[r for r in all_task if r['quantitative_sample']=='1']
        valid=[r for r in sample if r['cam_status']=='ok'];ckpt=provenance['checkpoints'][task]
        archive=ROOT/f'training_logs/publication_evidence/{task}/probabilities.csv'
        require(sha(archive)==ckpt['probabilities_sha256'],'Historical probabilities changed')
        population=read_csv(archive);rng=random.Random(provenance['seed']);expected=set()
        for label in ckpt['classes']:
            pool=sorted([r for r in population if r['true_label']==label],key=lambda r:r['image_path'])
            expected.update(r['image_path'] for r in rng.sample(pool,min(provenance['per_class'],len(pool))))
        require({r['path'] for r in sample}==expected,'Seeded sample membership mismatch')
        require(len(sample)==s['sample_n'] and len(valid)==s['valid_cam_n'],'Count mismatch')
        require(len(sample)-len(valid)==s['failed_cam_n'],'Failure count mismatch')
        require(sum(int(r['correct']) for r in sample)==s['sample_correct'],'Correct count mismatch')
        require(sum(bool(r['gallery_reason']) for r in all_task)==s['gallery_n'],'Gallery count mismatch')
        for label in ckpt['classes']+['ALL']:
            subset=valid if label=='ALL' else [r for r in valid if r['true_label']==label]
            stats=s['metrics'] if label=='ALL' else s['by_class'][label]['metrics']
            if label!='ALL':require(len(subset)==s['by_class'][label]['n'],'Per-class count mismatch')
            for key,expected_stats in stats.items():
                actual=describe([float(r[key]) for r in subset])
                for measure in ['mean','sd']:
                    a,b=actual[measure],expected_stats[measure]
                    require(a==b or (a is not None and b is not None and abs(a-b)<1e-10),'Summary mean/SD mismatch')
        pointer=ROOT/ckpt['path']
        if pointer.stat().st_size<1024:require(f"oid sha256:{ckpt['sha256']}" in pointer.read_text(),'Checkpoint identity mismatch')
        elif full:require(sha(pointer)==ckpt['sha256'],'Checkpoint identity mismatch')
    for r in rows:
        require(r['path'] in manifest and manifest[r['path']]['split']=='test','Case not in test split')
        require(manifest[r['path']]['subtype']==r['true_label'],'Class mismatch')
        require(float(r['max_probability_difference'])<1e-4,'Probability drift')
        if full:require(sha(ROOT/r['path'])==r['image_sha256'],'Image changed')
        if r['cam_status']=='ok':
            cam=np.load(out/r['cam_path'],allow_pickle=False)
            require(cam.shape==(7,7) and np.isfinite(cam).all() and cam.min()>=0 and cam.max()<=1 and np.ptp(cam)>0,'Invalid native CAM')
            require(0<=float(r['border_mass'])<=1,'Invalid border mass')
            require(abs(float(r['paired_drop_advantage'])-float(r['top20_drop'])+float(r['random20_drop']))<1e-10,'Paired metric mismatch')
    for name in ['index.html','SUMMARY.md','figure8a_gradcam.png','figure8b_gradcam.png']:
        require((out/name).is_file(),f'Missing figure/document: {name}')
    # Original record remains immutable; current implementation hashes are intentionally not substituted.
    inventory=json.loads((snapshots/'baseline_inventory.json').read_text())
    for name,digest in inventory.items():
        require(sha(ROOT/name)==digest,f'Historical baseline changed: {name}')
    return {'status':'pass','mode':'full' if full else 'light','cases':len(rows),'historical_sources':'verified frozen snapshots'}


def main():
    result=check();print(json.dumps(result));return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--full',action='store_true');args=p.parse_args()
    print(json.dumps(check(args.full),indent=2))
