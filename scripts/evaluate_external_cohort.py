#!/usr/bin/env python3
"""Validate or evaluate a separately supplied image cohort. No external cohort ships here."""
import argparse
import json
import sys
import tempfile
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src import experiment_pipeline as ep
from src.external_validation import validate_cohort,cohort_metrics
from src.research_validation import require,read_csv,write_json,write_csv,sha,softmax,seal,publish


def run(args):
    classes=ep.TASK_CLASS_NAMES[args.task]
    mapping=json.loads(args.label_mapping.read_text())
    internal_rows=read_csv(args.internal_manifest)
    paths=[args.internal_root/r['path'] for r in internal_rows]
    rows,identities,warnings=validate_cohort(args.manifest,mapping,classes,paths)
    require(not args.output.exists(),'External output already exists')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix='.external-staging-',dir=args.output.parent))
    fixture=all(r['cohort_id'].startswith('synthetic') for r in rows)
    summary={'status':'synthetic_fixture_only' if fixture else 'readiness_checked_not_evaluated',
             'external_validation':'not performed','images':len(rows),'warnings':warnings,
             'manifest_sha256':sha(args.manifest),'label_mapping_sha256':sha(args.label_mapping),
             'internal_manifest_sha256':sha(args.internal_manifest),'classes':classes,'task':args.task}
    if not args.validate_only:
        require(args.checkpoint is not None,'Evaluation requires a frozen checkpoint')
        import torch
        from PIL import Image
        model,ckpt=ep.load_checkpoint_model(args.checkpoint,torch.device(args.device))
        require(ckpt['task']==args.task and ckpt['class_names']==classes,'Checkpoint task/class mismatch')
        temperature=1.
        if args.calibration:
            artifact=json.loads(args.calibration.read_text())[args.task]
            require(artifact['checkpoint_sha256']==sha(args.checkpoint),'Calibration/checkpoint mismatch')
            require(artifact['classes']==classes and artifact['split']=='val','Invalid calibration provenance')
            temperature=artifact['temperature']
            require(np.isfinite(temperature) and .05<=temperature<=20,'Invalid fitted temperature')
            summary['calibration_sha256']=sha(args.calibration)
        transform=ep.build_transforms(False,ckpt['image_size']);values=[]
        with torch.inference_mode():
            for row in rows:
                with Image.open(row['resolved_path']) as im:x=transform(im.convert('RGB')).unsqueeze(0).to(args.device)
                values.append(model(x).cpu().numpy()[0])
        p=softmax(np.array(values),temperature);np.save(stage/'probabilities.npy',p)
        summary['metrics']=cohort_metrics(p,[r['class_index'] for r in rows],[(r['cohort_id'],r['patient_id']) for r in rows],classes)
        summary['checkpoint_sha256']=sha(args.checkpoint)
        summary['status']='synthetic_fixture_only' if fixture else 'candidate_external_cohort_evaluated'
        summary['external_validation']='not performed' if fixture else 'cohort evaluated; independence and clinical validity require provenance review'
    safe_rows=[{k:v for k,v in r.items() if k!='resolved_path'} for r in rows]
    write_csv(stage/'cohort_index.csv',safe_rows);write_json(stage/'image_identities.json',identities)
    write_json(stage/'summary.json',summary);seal(stage);publish(stage,args.output)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--label-mapping',type=Path,required=True)
    p.add_argument('--task',choices=list(ep.TASK_CLASS_NAMES),default='dementia')
    p.add_argument('--internal-manifest',type=Path,default=ep.DEFAULT_MANIFEST)
    p.add_argument('--internal-root',type=Path,default=ROOT)
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--calibration',type=Path)
    p.add_argument('--validate-only',action='store_true')
    p.add_argument('--device',choices=['cpu','mps','cuda'],default='cpu')
    p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())
