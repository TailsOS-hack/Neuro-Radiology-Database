"""Research-only metrics, provenance and transactional artifacts. No GUI changes."""
from __future__ import annotations
import csv
import hashlib
import json
import os
from pathlib import Path
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[1]
TASKS = ['binary', 'tumor', 'dementia', 'eight_class']


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_csv(path):
    with Path(path).open(newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    require(bool(rows), f'No rows for {path}')
    with Path(path).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def softmax(logits, temperature=1.0):
    logits = np.asarray(logits, dtype=np.float64)
    require(logits.ndim == 2 and np.isfinite(logits).all(), 'Invalid logits')
    require(np.isfinite(temperature) and temperature > 0, 'Invalid temperature')
    scaled = logits / temperature
    return np.exp(scaled - logsumexp(scaled, axis=1, keepdims=True))


def nll_logits(logits, labels, temperature):
    z = np.asarray(logits, dtype=np.float64) / temperature
    return float(np.mean(logsumexp(z, axis=1) - z[np.arange(len(z)), labels]))


def fit_temperature(logits, labels, *, split, provenance):
    require(split == 'val', 'Calibration fitting accepts validation data only')
    softmax(logits)
    labels = np.asarray(labels, dtype=int)
    require(len(labels) == len(logits) and len(labels) > 0, 'Empty/misaligned fit data')
    require(np.all((labels >= 0) & (labels < logits.shape[1])), 'Invalid fit labels')
    original = nll_logits(logits, labels, 1.)
    result = minimize_scalar(lambda log_t: nll_logits(logits, labels, np.exp(log_t)),
                             bounds=(np.log(.05), np.log(20.)), method='bounded',
                             options={'xatol': 1e-8, 'maxiter': 500})
    candidate = float(np.exp(result.x))
    improved = bool(result.success and np.isfinite(result.fun) and result.fun < original - 1e-12)
    return {'method': 'scalar_temperature', 'temperature': candidate if improved else 1.,
            'candidate_temperature': candidate, 'optimization_success': bool(result.success),
            'optimization_message': str(result.message), 'boundary_solution': candidate <= .05001 or candidate >= 19.999,
            'status': 'fitted' if improved else 'fallback_no_improvement_or_failure',
            'original_val_nll': original, 'fitted_val_nll': float(result.fun) if np.isfinite(result.fun) else None,
            'n': len(labels), 'split': split, 'bounds': [.05, 20.], 'objective': 'NLL', **provenance}


def metrics(probabilities, labels, predicted=None):
    p = np.asarray(probabilities, dtype=float); y = np.asarray(labels, dtype=int)
    require(p.ndim == 2 and len(p) == len(y) and len(y) > 0, 'Invalid metric shape')
    require(np.isfinite(p).all() and (p >= 0).all() and np.allclose(p.sum(1), 1, atol=1e-6), 'Invalid probabilities')
    require(((y >= 0) & (y < p.shape[1])).all(), 'Invalid metric labels')
    pred = p.argmax(1) if predicted is None else np.asarray(predicted, dtype=int)
    confidence = p[np.arange(len(p)), pred]; correct = pred == y
    bins = []
    for i in range(10):
        mask = (confidence >= i/10) & ((confidence < (i+1)/10) if i < 9 else (confidence <= 1))
        bins.append({'low': i/10, 'high': (i+1)/10, 'n': int(mask.sum()),
                     'accuracy': float(correct[mask].mean()) if mask.any() else None,
                     'confidence': float(confidence[mask].mean()) if mask.any() else None})
    return {'n': len(y), 'accuracy': float(correct.mean()),
            'nll': float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-15, 1)).mean()),
            'brier': float(np.mean(np.sum((p-np.eye(p.shape[1])[y])**2, axis=1))),
            'ece': sum(b['n']/len(y)*abs(b['accuracy']-b['confidence']) for b in bins if b['n']),
            'bins': bins}


def hierarchical(router, tumor, dementia):
    require(router.shape[1] == 2 and tumor.shape[1] == dementia.shape[1] == 4, 'Hierarchy class dimensions')
    p = np.concatenate([router[:, :1]*tumor, router[:, 1:]*dementia], axis=1)
    pred = np.where(router.argmax(1) == 0, tumor.argmax(1), 4+dementia.argmax(1))
    require(np.allclose(p.sum(1), 1), 'Hierarchy probabilities not normalized')
    return p, pred


def describe(values):
    a = np.asarray(values, dtype=float)
    require(np.isfinite(a).all(), 'Nonfinite summary values')
    return {'n': len(a), 'mean': float(a.mean()) if len(a) else None,
            'sd': float(a.std(ddof=1)) if len(a) > 1 else None}


def seal(directory):
    directory = Path(directory)
    inventory = {str(p.relative_to(directory)): sha(p) for p in sorted(directory.rglob('*'))
                 if p.is_file() and p.name not in ['COMPLETE.json', 'validation.json']}
    write_json(directory/'COMPLETE.json', {'files': inventory})


def verify_seal(directory):
    directory = Path(directory)
    inventory = json.loads((directory/'COMPLETE.json').read_text())['files']
    require(bool(inventory), 'Empty artifact inventory')
    for name, digest in inventory.items():
        path = directory/name
        require(path.resolve().is_relative_to(directory.resolve()), 'Unsafe artifact path')
        require(path.is_file() and sha(path) == digest, f'Artifact hash mismatch: {name}')
    actual = {str(p.relative_to(directory)) for p in directory.rglob('*')
              if p.is_file() and p.name not in ['COMPLETE.json', 'validation.json']}
    require(actual == set(inventory), 'Unlisted or missing artifacts')


def publish(stage, destination):
    verify_seal(stage)
    destination = Path(destination)
    require(not destination.exists(), f'Refusing to replace completed run: {destination}')
    os.replace(stage, destination)


def checkpoint_path(task, checkpoint_dir):
    from src import experiment_pipeline as ep
    path = Path(checkpoint_dir)/ep.TASK_DEFAULT_MODEL_PATH[task].name
    pointer = ep.TASK_DEFAULT_MODEL_PATH[task]
    expected = pointer.read_text().split('oid sha256:')[1].splitlines()[0] if pointer.stat().st_size < 1024 else sha(pointer)
    require(path.is_file() and sha(path) == expected, f'Checkpoint identity mismatch: {task}')
    return path
