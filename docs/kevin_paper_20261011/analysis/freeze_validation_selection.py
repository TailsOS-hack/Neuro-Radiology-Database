#!/usr/bin/env python3
"""Freeze validation-only preferences BEFORE evaluate_candidates.py is run.

Reads only checkpoint metadata, never images, held-out labels, or model predictions.
Candidate order must be exactly the same as the evaluator's repeated --candidate.
Use this only for locally trained, trusted checkpoints (torch.load deserializes them).
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import torch

from summarize_frozen_evaluation import PROTOCOL, STAGES, digest


def freeze(paths, evaluation_output, sensitivity_manifest=None, sensitivity_audit=None):
    if evaluation_output.exists() and any(evaluation_output.iterdir()):
        raise ValueError('Validation selection must be frozen before held-out outputs exist')
    if len(paths)!=6 or len({p.resolve() for p in paths})!=6: raise ValueError('Require all six distinct candidate checkpoints')
    records,seen,stage_hashes,clinical_hashes = {},set(),set(),set()
    for i,path in enumerate(paths,1):
        ck = torch.load(path,map_location='cpu',weights_only=False)
        cfg = ck.get('config',{}); pair = (cfg.get('mode'),cfg.get('seed'))
        if ck.get('smoke_only') or cfg.get('smoke') or cfg.get('test_used_for_selection') is not False:
            raise ValueError('Refusing smoke or non-validation-selected checkpoint')
        if pair in seen or pair not in {(m,s) for m in PROTOCOL['planned_modes'] for s in PROTOCOL['planned_seeds']}:
            raise ValueError('Unexpected or duplicate condition/seed')
        seen.add(pair)
        if ck.get('arch')!='mobilenet_v3_large' or ck.get('class_names')!=STAGES: raise ValueError('Unexpected architecture/class order')
        if not ck.get('history') or ck.get('epoch')!=ck['history'][-1]['epoch']: raise ValueError('Missing checkpoint epoch provenance')
        loss = ck.get('stage_validation',{}).get('loss',float('nan'))
        if not np.isfinite(loss) or abs(loss-ck.get('best_val_loss',float('inf')))>1e-8: raise ValueError('Not a validation-selected best checkpoint')
        stage_hashes.add(cfg.get('stage_manifest_sha256'))
        if cfg.get('clinical_manifest_sha256'): clinical_hashes.add(cfg['clinical_manifest_sha256'])
        records[f'candidate_{i}'] = {'checkpoint_path':str(path.resolve()),'checkpoint_sha256':digest(path),
            'mode':pair[0],'seed':pair[1],'epoch':ck['epoch'],'stage_validation_nll':float(loss)}
        del ck
    if None in stage_hashes or len(stage_hashes)!=1 or len(clinical_hashes)>1: raise ValueError('Candidates do not share frozen manifests')
    means = {mode:float(np.mean([r['stage_validation_nll'] for r in records.values() if r['mode']==mode])) for mode in PROTOCOL['planned_modes']}
    preferred_mode = min(means,key=lambda m:(means[m],m))
    preferred = min((n for n,r in records.items() if r['mode']==preferred_mode),key=lambda n:(records[n]['stage_validation_nll'],records[n]['seed']))
    if evaluation_output.exists() and any(evaluation_output.iterdir()): raise ValueError('Held-out outputs appeared while freezing selection')
    if bool(sensitivity_manifest)!=bool(sensitivity_audit): raise ValueError('Sensitivity manifest and audit must be supplied together')
    sensitivity = None
    if sensitivity_manifest:
        audit = json.loads(sensitivity_audit.read_text())
        if audit.get('status')!='prespecified_before_heldout_model_evaluation' or audit.get('subset',{}).get('sha256')!=digest(sensitivity_manifest):
            raise ValueError('Sensitivity audit status or subset digest mismatch')
        sensitivity = {'manifest_path':str(sensitivity_manifest.resolve()),'manifest_sha256':digest(sensitivity_manifest),
                       'audit_path':str(sensitivity_audit.resolve()),'audit_sha256':digest(sensitivity_audit)}
    return {'frozen_before_heldout_evaluation':True,'created_utc':datetime.now(timezone.utc).isoformat(),
        'checked_empty_evaluation_output':str(evaluation_output.resolve()),'criterion':PROTOCOL['validation_preference'],
        'candidates':records,'mode_mean_stage_validation_nll':means,'preferred_mode':preferred_mode,
        'exploratory_representative_candidate':preferred,
        'tie_breaking':'mode name lexicographic; seed ascending within mode',
        'interpretation':'Validation-only exploratory preference. Primary reporting includes both conditions and all six seeds/checkpoints. No test result influences this record.',
        'stage_manifest_sha256':next(iter(stage_hashes)), 'clinical_manifest_sha256':next(iter(clinical_hashes),None),
        'old_hash_unexposed_sensitivity':sensitivity,
        'analysis_protocol':PROTOCOL,'analysis_script_sha256':digest(Path(__file__).with_name('summarize_frozen_evaluation.py')),
        'selection_script_sha256':digest(__file__)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidate',action='append',type=Path,required=True)
    p.add_argument('--evaluation-output',type=Path,required=True)
    p.add_argument('--old-hash-unexposed-manifest',type=Path)
    p.add_argument('--baseline-exposure-audit',type=Path)
    p.add_argument('--output',type=Path,required=True)
    args = p.parse_args()
    if args.output.exists(): p.error('Selection output already exists; preserve the frozen record')
    result = freeze(args.candidate,args.evaluation_output,args.old_hash_unexposed_manifest,args.baseline_exposure_audit)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as f: json.dump(result,f,indent=2,allow_nan=False); f.write('\n')
    print(json.dumps({'status':'frozen','output':str(args.output.resolve()),'preferred_mode':result['preferred_mode'],'exploratory_representative_candidate':result['exploratory_representative_candidate']}))


if __name__=='__main__': main()
