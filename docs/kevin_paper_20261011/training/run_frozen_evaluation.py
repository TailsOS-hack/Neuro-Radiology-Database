#!/usr/bin/env python3
"""Guard and execute the prespecified held-out evaluation exactly once.

Invoke manually after all six runs finish. Never start training from this script.
One candidate list supplies both validation freezing and evaluation.
"""
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path('/tmp/Neuro-Radiology-Database')
OUT = Path('/workspace/kevin_results')
PREP = ROOT / 'training_logs/experiments/kevin_20261011/prepared'
DOCS = ROOT / 'docs/kevin_paper_20261011'
RUNS = [f'{mode}_seed{seed}' for seed in (20261011, 20261012, 20261013)
        for mode in ('stage_only', 'multitask')]

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as file:
        for block in iter(lambda: file.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def rows(path):
    with path.open(newline='', encoding='utf-8-sig') as file:
        return list(csv.DictReader(file))

if __name__ == '__main__':
    status = json.loads((OUT / 'suite_status.json').read_text())
    assert status['state'] == 'training_complete_awaiting_frozen_evaluation'
    assert status['completed'] == RUNS
    expected = json.loads((DOCS / 'training/local_verification.json').read_text())
    audit = json.loads((PREP / 'split_audit.json').read_text())
    assert sha(ROOT / 'models/alzheimers_classifier.pt') == expected['production_checkpoint_sha256']
    assert sha(PREP / 'stage_manifest.csv') == expected['prepared_stage_sha256']
    assert sha(ROOT / 'data/external/clinical_t1_preprocessed_v1/clinical_slices.csv') == expected['clinical_manifest_sha256']
    assert sha(PREP / 'same_case_test_intersection.csv') == audit['same_case_test_intersection_sha256']
    eligible = {r['path']: r for r in rows(PREP / 'stage_manifest.csv')
                if r['split'] == 'test' and r['baseline_strict_split'] == 'test'}
    intersection = rows(PREP / 'same_case_test_intersection.csv')
    assert len(intersection) == len(eligible) == audit['same_case_test_intersection_rows'] == 1058
    assert {r['path'] for r in intersection} == set(eligible)
    assert all(r == eligible[r['path']] for r in intersection)
    evaluation = OUT / 'frozen_evaluation_all6'
    selection = OUT / 'validation_selection_before_evaluation.json'
    if (evaluation.exists() and any(evaluation.iterdir())) or selection.exists():
        raise RuntimeError('Frozen evaluation or selection already exists; preserve and inspect it')
    candidate_args = []
    for name in RUNS:
        checkpoint = OUT / name / 'best.pt'
        assert checkpoint.is_file()
        candidate_args.extend(['--candidate', str(checkpoint)])
    launch = dict(launched_at=datetime.now(timezone.utc).isoformat(),
                  candidate_order=RUNS, baseline_sha256=expected['production_checkpoint_sha256'],
                  complete_intersection_verified=True,
                  script_sha256=sha(__file__), status='preflight_passed')
    with (OUT / 'evaluation_launch.json').open('x') as file:
        json.dump(launch, file, indent=2)
    subprocess.run([sys.executable, str(DOCS / 'analysis/freeze_validation_selection.py'),
                    *candidate_args, '--evaluation-output', str(evaluation),
                    '--old-hash-unexposed-manifest', str(DOCS / 'analysis/old_hash_group_unexposed_sensitivity.csv'),
                    '--baseline-exposure-audit', str(DOCS / 'analysis/old_hash_group_unexposed_sensitivity_audit.json'),
                    '--output', str(selection)], cwd=ROOT, check=True)
    with (OUT / 'evaluation.log').open('x') as log:
        subprocess.run([sys.executable, str(DOCS / 'training/evaluate_candidates.py'),
                        '--data-root', str(ROOT), '--prepared', str(PREP),
                        *candidate_args, '--baseline', 'models/alzheimers_classifier.pt',
                        '--clinical-manifest', 'data/external/clinical_t1_preprocessed_v1/clinical_slices.csv',
                        '--device', 'cuda', '--workers', '4', '--output', str(evaluation)],
                       cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    report = json.loads((evaluation / 'evaluation.json').read_text())
    assert len(report['checkpoints']) == 7
    (OUT / 'evaluation_complete.json').write_text(json.dumps(dict(
        completed_at=datetime.now(timezone.utc).isoformat(),
        checkpoints=7, candidate_order=RUNS,
        evaluation_sha256=sha(evaluation / 'evaluation.json'),
        validation_selection_sha256=sha(selection)), indent=2)+'\n')
    print('FROZEN_EVALUATION_COMPLETE', flush=True)
