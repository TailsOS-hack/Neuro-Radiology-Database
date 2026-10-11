#!/usr/bin/env python3
"""Run the fixed six training experiments; never evaluate a held-out set.

Status and checkpoints persist on /workspace. Re-running this orchestrator is
refused; resume an interrupted individual run with the trainer's guarded resume.
"""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path('/tmp/Neuro-Radiology-Database')
OUT = Path('/workspace/kevin_results')
TRAINER = ROOT / 'docs/kevin_paper_20261011/training/train_candidate.py'
EXPECTED_TRAINER = 'cbf67b7c6e842ce35eeef2e0c0a6a0752b865cec3399d7aa58bd5e8ff0e2a5e2'
SEEDS = [20261011, 20261012, 20261013]

def now():
    return datetime.now(timezone.utc).isoformat()

def save(value):
    path = OUT / 'suite_status.json'
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2)+'\n')
    os.replace(tmp, path)

if __name__ == '__main__':
    assert hashlib.sha256(TRAINER.read_bytes()).hexdigest() == EXPECTED_TRAINER
    if (OUT / 'suite_status.json').exists():
        raise RuntimeError('Suite already launched; inspect status instead of duplicating runs')
    status = dict(started_at=now(), state='training', completed=[], current=None,
                  seeds=SEEDS, modes=['stage_only', 'multitask'], max_epochs=100,
                  min_epochs=20, patience=15, selection='minimum stage validation NLL',
                  heldout_inference_performed=False,
                  note='All six runs are prespecified; report all seeds. Stop cloud compute after verified local retrieval.')
    save(status)
    for seed in SEEDS:
        for mode in status['modes']:
            name = f'{mode}_seed{seed}'
            target = OUT / name
            args = [sys.executable, str(TRAINER), 'train', '--data-root', str(ROOT),
                    '--stage-manifest', 'training_logs/experiments/kevin_20261011/prepared/stage_manifest.csv',
                    '--mode', mode, '--device', 'cuda', '--workers', '4',
                    '--epochs', '100', '--min-epochs', '20', '--patience', '15',
                    '--seed', str(seed), '--output', str(target)]
            if mode == 'multitask':
                args += ['--clinical-manifest', 'data/external/clinical_t1_preprocessed_v1/clinical_slices.csv']
            status.update(current=name, current_started_at=now(), command=args)
            save(status)
            with (OUT / f'{name}.log').open('x') as log:
                result = subprocess.run(args, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                status.update(state='failed', exit_code=result.returncode, finished_at=now())
                save(status)
                sys.exit(result.returncode)
            assert (target / 'best.pt').is_file() and (target / 'last.pt').is_file()
            status['completed'].append(name)
            save(status)
    status.update(state='training_complete_awaiting_frozen_evaluation', current=None, finished_at=now())
    save(status)
    print('ALL_SIX_TRAINING_RUNS_COMPLETE', flush=True)
