"""Bounded patient-level development; authorization precedes all data reads.

Real fitting is GPU-only. Local synthetic tests exercise the mathematical loop
without making a production authorization or research checkpoint.
"""
from collections import Counter
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import random
import time
import nibabel as nib
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from .models import ModelConfig, MODEL_VERSION, build_model
from .preprocessing import file_hash, object_hash, anatomical_volume_fractions

TRAINING_VERSION = 'round3-patient-training-v1'
SEARCH_SEED = 20261030
FINAL_SEEDS = (20261031, 20261032, 20261033)
RUNTIME_POLICY = {'real_training': 'NVIDIA B300 CUDA; live price verified by external launcher',
                  'requested_gpu_hourly_usd': 8.99,
                  'analysis_statistics_figures': 'local Mac',
                  'paid_launch_authority': 'separate dataset/protocol and live provider gates'}


def bounded_search_space():
    candidates = []
    for family in ('multiplanar', 'resnet3d18'):
        for lr in (0.00003, 0.0001, 0.0003):
            for wd in (0.0001, 0.01):
                for dropout in (0.2, 0.5):
                    candidates.append({'model': asdict(ModelConfig(kind=family, dropout=dropout)),
                                       'learning_rate': lr, 'weight_decay': wd})
    return [dict(candidate_id=f'candidate_{i + 1:02d}', **r) for i, r in enumerate(candidates)]


def anatomical_baseline_spec():
    return {'candidate_id': 'anatomical_fixed_l2',
            'model': asdict(ModelConfig(kind='anatomical_logistic', dropout=0.0)),
            'learning_rate': 0.01, 'weight_decay': 1.0}


@dataclass(frozen=True)
class TrainConfig:
    candidate_id: str
    seed: int = SEARCH_SEED
    expected_protocol_hash: str = ''
    device: str = 'cuda'
    max_epochs: int = 200
    min_epochs: int = 30
    patience: int = 25
    min_delta: float = 0.0001
    batch_size: int = 2
    max_seconds: int = 43200
    heldout_institution: str = ''
    validation_site_id: str = ''

    def validate(self):
        if (self.seed not in (SEARCH_SEED, *FINAL_SEEDS) or self.max_epochs != 200 or
                self.min_epochs != 30 or self.patience != 25 or self.min_delta != 0.0001 or
                self.batch_size != 2 or self.max_seconds != 43200):
            raise ValueError('Training schedule is outside the fixed bounded design')
        if self.candidate_id not in {r['candidate_id'] for r in bounded_search_space()} | {'anatomical_fixed_l2'}:
            raise ValueError('Candidate is outside the fixed 24-configuration search')
        if self.seed == SEARCH_SEED and not self.validation_site_id:
            raise ValueError('Search runs require a named held-out development site')
        if self.heldout_institution:
            raise ValueError('Use validation_site_id; institution independence is verified from its rows')
        if self.seed in FINAL_SEEDS and self.validation_site_id:
            raise ValueError('Final-seed runs use all development training sites')


def _authorize(authfile, rows, action, protocol_hash=None):
    # Deliberately import at call time; the root-owned verifier is the authority.
    from .common import verify_training_authorization, authorize_rows
    auth = verify_training_authorization(authfile, expected_protocol_hash=protocol_hash or None)
    if action not in auth.get('allowed_actions', []):
        raise PermissionError('Action absent from verified authorization')
    authorize_rows(auth, rows, allowed_partitions={'train', 'tune'} if action == 'train' else None)
    if action == 'train' and auth.get('phase') != 'development':
        raise PermissionError('Real training requires development authorization')
    return auth


def balanced_site_diagnosis_weights(rows):
    if not rows or any(r.get('diagnosis') not in (0, 1) or not r.get('site_id') for r in rows):
        raise ValueError('Binary diagnoses and sites required')
    cells = Counter((r['site_id'], r['diagnosis']) for r in rows)
    sites = {r['site_id'] for r in rows}
    if any((site, label) not in cells for site in sites for label in (0, 1)):
        raise ValueError('Each development site must contain both diagnoses; empty cells cannot be balanced')
    return torch.tensor([1.0 / cells[(r['site_id'], r['diagnosis'])] for r in rows], dtype=torch.double)


def validate_partition_rows(rows):
    keys = [r['patient_id'] for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError('Patient appears more than once in development rows')
    if {r['partition'] for r in rows} != {'train', 'tune'}:
        raise ValueError('Training entrypoint accepts exactly train/tune patients')
    hashes = [r['volume_sha256'] for r in rows]
    if len(hashes) != len(set(hashes)):
        raise ValueError('Repeated raw-volume hash in development patients')


def development_fold(rows, config):
    """Keep original authorized row bytes; only dataset membership changes.

    Search holds out an entire site AND its institution. A multi-site institution
    cannot leak into the training fold; it is rejected for explicit redesign.
    Calibration patients never enter a fold. Final seeds use original partitions.
    """
    validate_partition_rows(rows)
    if config.seed == SEARCH_SEED:
        tune = [r for r in rows if r['site_id'] == config.validation_site_id]
        train = [r for r in rows if r['site_id'] != config.validation_site_id]
        if not tune or not train:
            raise ValueError('Held-out development site must define two nonempty folds')
        if {r['institution_id'] for r in train} & {r['institution_id'] for r in tune}:
            raise ValueError('Held-out site shares an institution with training; redesign site grouping')
    else:
        train = [r for r in rows if r['partition'] == 'train']
        tune = [r for r in rows if r['partition'] == 'tune']
    return train, tune


def _verified_json(path, expected):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or file_hash(path) != expected:
        raise ValueError('Missing or changed pinned input metadata')
    return json.loads(path.read_text())


def load_prepared_patient(row, preprocessing_hash, model_config):
    """Private loader; callers must already have passed the authorization gate."""
    if model_config.kind == 'anatomical_logistic':
        obj = _verified_json(row['anatomical_features_path'], row['anatomical_features_sha256'])
        if (obj.get('source_volume_sha256') != row['volume_sha256'] or
                obj.get('units') != 'fraction_of_intracranial_volume' or
                obj.get('qc_status') != 'passed_blinded_review' or
                not obj.get('segmentation_tool_version') or not obj.get('segmentation_source_sha256') or
                not obj.get('segmentation_pipeline_hash') or
                obj.get('features_order') != list(model_config.features)):
            raise ValueError('Actual anatomical segmentation provenance/features required')
        values = np.asarray(obj['values'], np.float32)
        if values.shape != (len(model_config.features),) or not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1):
            raise ValueError('Invalid anatomical volume fractions')
        for key, digest in [('segmentation_volume_path', 'segmentation_source_sha256'), ('icv_mask_path', 'icv_mask_sha256')]:
            path = Path(obj[key])
            if path.is_symlink() or file_hash(path) != obj[digest]:
                raise ValueError('Anatomical segmentation/ICV source bytes changed')
        actual = anatomical_volume_fractions(nib.load(obj['segmentation_volume_path']), nib.load(obj['icv_mask_path']),
                                            obj['labels_by_feature'], model_config.features)
        if not np.allclose(actual, values, rtol=0, atol=1e-7):
            raise ValueError('Anatomical features do not reproduce from actual segmentation')
        return torch.from_numpy(values)
    record = _verified_json(row['preprocessing_record_path'], row['preprocessing_record_sha256'])
    review = _verified_json(row['preprocessing_review_path'], row['preprocessing_review_sha256'])
    if (record.get('status') != 'passed_automated_qc_requires_blinded_review' or
            record.get('source_volume_sha256') != row['volume_sha256'] or
            record.get('preprocessing_hash') != preprocessing_hash or
            review.get('status') != 'passed' or review.get('diagnosis_blinded') is not True or
            review.get('prediction_blinded') is not True or
            review.get('preprocessing_record_sha256') != row['preprocessing_record_sha256']):
        raise ValueError('Unapproved, mismatched or failed preprocessing')
    path = Path(record['processed_volume_path'])
    if path.is_symlink() or file_hash(path) != record['processed_volume_sha256']:
        raise ValueError('Prepared volume checksum differs')
    image = nib.load(path)
    data = image.get_fdata(dtype=np.float32)
    if (len(data.shape) != 3 or list(data.shape) != record['shape'] or not np.isfinite(data).all() or
            not np.allclose(image.affine, record['affine'], atol=1e-5)):
        raise ValueError('Prepared volume geometry/intensity mismatch')
    return torch.from_numpy(data.copy()).unsqueeze(0)


class PatientDataset(Dataset):
    def __init__(self, rows, preprocessing_hash, config):
        self.rows, self.preprocessing_hash, self.config = list(rows), preprocessing_hash, config

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        return load_prepared_patient(row, self.preprocessing_hash, self.config), int(row['diagnosis']), index


def patient_metrics(rows, probabilities):
    p = np.asarray(probabilities, float)
    labels = np.asarray([r['diagnosis'] for r in rows], int)
    if p.shape != (len(rows), 2) or not np.isfinite(p).all() or np.any(p < 0) or not np.allclose(p.sum(1), 1, atol=1e-6):
        raise ValueError('Invalid patient probabilities')
    if set(labels) != {0, 1}:
        raise ValueError('Both diagnoses needed for development metrics')
    losses = -np.log(np.clip(p[np.arange(len(labels)), labels], 1e-12, 1.0))
    sites = sorted({r['site_id'] for r in rows})
    site_nll = {s: float(losses[[r['site_id'] == s for r in rows]].mean()) for s in sites}
    predicted = (p[:, 1] >= .5).astype(int)
    site_ba = {}
    for site in sites:
        member = np.asarray([r['site_id'] == site for r in rows])
        if set(labels[member]) != {0, 1}:
            raise ValueError('Each development site needs both diagnoses for site metrics')
        site_ba[site] = float(np.mean([(predicted[member & (labels == k)] == k).mean() for k in (0, 1)]))
    return {'n': len(rows), 'nll': float(losses.mean()), 'site_macro_nll': float(np.mean(list(site_nll.values()))),
            'site_nll': site_nll, 'site_balanced_accuracy': site_ba,
            'site_macro_balanced_accuracy': float(np.mean(list(site_ba.values()))),
            'accuracy': float((predicted == labels).mean()),
            'balanced_accuracy': float(np.mean([(predicted[labels == k] == k).mean() for k in (0, 1)])),
            'brier': float(np.mean((p[:, 1] - labels) ** 2))}


@torch.no_grad()
def evaluate_patients(model, loader, device):
    model.eval()
    outputs = {}
    for x, _, indices in loader:
        values = model(x.to(device)).softmax(1).cpu().numpy()
        for i, value in zip(indices.tolist(), values):
            if i in outputs:
                raise ValueError('Evaluation visited a patient twice')
            outputs[i] = value.tolist()
    if set(outputs) != set(range(len(loader.dataset))):
        raise ValueError('Incomplete patient evaluation')
    return [outputs[i] for i in range(len(outputs))]


def train_epoch(model, loader, optimizer, device, augment=True, anatomical_l2=0.0):
    model.train()
    loss_sum, objective_sum, count, sampled_indices = 0.0, 0.0, 0, []
    for x, y, indices in loader:
        x, y = x.to(device), y.to(device)
        if augment and x.ndim == 5:
            # Full-volume parameter sharing preserves cross-plane consistency.
            support = x != 0  # Prepared background, not a new anatomical segmentation.
            scale = .9 + .2 * torch.rand((len(x), 1, 1, 1, 1), device=device)
            x = torch.where(support, x * scale + .02 * torch.randn_like(x), torch.zeros_like(x))
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        if logits.shape != (len(y), 2):
            raise ValueError('Model must emit exactly one binary prediction per patient')
        ce = nn.functional.cross_entropy(logits, y)
        # Exact L2 logistic penalty on coefficients only; AdamW decay is not L2.
        penalty = .5 * anatomical_l2 * model.classifier.weight.square().sum() if anatomical_l2 else 0.0
        loss = ce + penalty
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite patient optimization loss')
        loss.backward()
        norm = nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
        if not torch.isfinite(norm):
            raise ValueError('Nonfinite gradients')
        optimizer.step()
        loss_sum += float(ce.detach()) * len(y)
        objective_sum += float(loss.detach()) * len(y)
        count += len(y)
        sampled_indices.extend(indices.tolist())
    if not count:
        raise ValueError('Empty optimization epoch')
    return {'patient_optimization_ce': loss_sum / count, 'penalized_objective': objective_sum / count, 'draws': count,
            'unique_patient_draws': len(set(sampled_indices)), 'sampled_indices': sampled_indices}


def _write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def _runtime_policy_check(config):
    # Must precede the first CUDA matmul; identical across every candidate/seed.
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    if os.environ['CUBLAS_WORKSPACE_CONFIG'] not in {':4096:8', ':16:8'}:
        raise PermissionError('Unsupported deterministic CUDA workspace configuration')
    if config.device != 'cuda' or not torch.cuda.is_available() or 'B300' not in torch.cuda.get_device_name(0):
        raise PermissionError('Real training requires the separately authorized B300; CPU/Mac training blocked')


def train_candidate(rows, config, authfile, output_dir):
    """One prespecified candidate/seed. Never opens calibration/external data."""
    auth = _authorize(authfile, rows, 'train', config.expected_protocol_hash)
    config.validate()
    validate_partition_rows(rows)
    protocol = auth['protocol']
    if protocol.get('search_space_hash') != object_hash(bounded_search_space()) or not protocol.get('preprocessing_hash'):
        raise PermissionError('Search space/preprocessing implementation is not frozen by protocol')
    if config.candidate_id == 'anatomical_fixed_l2':
        if (config.seed != FINAL_SEEDS[0] or protocol.get('anatomical_baseline_hash') != object_hash(anatomical_baseline_spec())):
            raise PermissionError('Fixed anatomical baseline needs its explicit protocol pin and one prespecified final seed')
    elif config.seed in FINAL_SEEDS:
        final = auth.get('final_candidate_selection', {})
        if (final.get('candidate_id') != config.candidate_id or final.get('selection_partitions') != ['train', 'tune'] or
                final.get('protocol_hash') != auth['protocol_hash'] or not final.get('selection_sha256')):
            raise PermissionError('Final seeds require root-verified validation-only candidate selection')
    _runtime_policy_check(config)
    destination = Path(output_dir)
    if destination.exists():
        raise ValueError('Research destination must be new; never overwrite/resume silently')
    candidate = next(r for r in bounded_search_space() + [anatomical_baseline_spec()] if r['candidate_id'] == config.candidate_id)
    model_config = ModelConfig(**candidate['model'])
    train_rows, tune_rows = development_fold(rows, config)
    weights = balanced_site_diagnosis_weights(train_rows)
    balanced_site_diagnosis_weights(tune_rows)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(config.device)
    datasets = {k: PatientDataset(v, protocol['preprocessing_hash'], model_config)
                for k, v in [('train', train_rows), ('tune', tune_rows)]}
    # Validate every prepared input before creating any research checkpoint.
    for dataset in datasets.values():
        for i in range(len(dataset)):
            dataset[i]
    model = build_model(model_config)
    if model_config.kind == 'anatomical_logistic':
        model.fit_normalizer(torch.stack([datasets['train'][i][0] for i in range(len(train_rows))]))
    model.to(device)
    is_anatomical = model_config.kind == 'anatomical_logistic'
    optimizer = (torch.optim.Adam(model.parameters(), lr=candidate['learning_rate']) if is_anatomical else
                 torch.optim.AdamW(model.parameters(), lr=candidate['learning_rate'], weight_decay=candidate['weight_decay']))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=candidate['learning_rate'] * .05)
    clean_loaders = {k: DataLoader(v, batch_size=config.batch_size, shuffle=False, num_workers=0) for k, v in datasets.items()}
    destination.mkdir(parents=True)
    start, best, bad, history = time.monotonic(), math.inf, 0, []
    source_pins = {p.name: file_hash(p) for p in Path(__file__).parent.glob('*.py')}
    common = {'model_version': MODEL_VERSION, 'training_version': TRAINING_VERSION,
              'model_config': asdict(model_config), 'candidate': candidate, 'seed': config.seed,
              'validation_site_id': config.validation_site_id,
              'protocol_hash': auth['protocol_hash'], 'preprocessing_hash': protocol['preprocessing_hash'],
              'source_sha256': source_pins, 'research_mode': 'real_authorized_development',
              'score_interpretation': 'Uncalibrated class-balanced sampling score; calibration is a separate frozen phase'}
    try:
        for epoch in range(config.max_epochs + 1):
            online = None
            if epoch:
                sampler = WeightedRandomSampler(weights, len(train_rows), replacement=True,
                                                 generator=torch.Generator().manual_seed(config.seed + epoch))
                loader = DataLoader(datasets['train'], batch_size=config.batch_size, sampler=sampler, num_workers=0)
                online = train_epoch(model, loader, optimizer, device, augment=not is_anatomical,
                                     anatomical_l2=1.0 if is_anatomical else 0.0)
                scheduler.step()
            probabilities = {k: evaluate_patients(model, loader, device) for k, loader in clean_loaders.items()}
            metrics = {k: patient_metrics(train_rows if k == 'train' else tune_rows, probabilities[k]) for k in probabilities}
            improved = metrics['tune']['site_macro_nll'] < best - config.min_delta
            bad = 0 if improved else bad + 1
            history.append({'epoch': epoch, 'online': online, 'clean': metrics,
                            'learning_rates': [g['lr'] for g in optimizer.param_groups],
                            'selected': improved, 'seconds': time.monotonic() - start})
            _write_json(destination / 'history.json', history)
            _write_json(destination / f'epoch{epoch:03d}_tune.json', [dict(patient_id=r['patient_id'], site_id=r['site_id'], diagnosis=r['diagnosis'], probabilities=p) for r, p in zip(tune_rows, probabilities['tune'])])
            if improved:
                best = metrics['tune']['site_macro_nll']
                checkpoint = dict(common, model={k: v.detach().cpu() for k, v in model.state_dict().items()},
                                  epoch=epoch, selection_metric='site_macro_tune_nll', selected_metrics=metrics['tune'])
                torch.save(checkpoint, destination / 'best.pt')
                _write_json(destination / 'selection.json', {'epoch': epoch, 'score': best, 'checkpoint_sha256': file_hash(destination / 'best.pt'),
                                                            'selection_partitions': ['train', 'tune'] if config.seed == SEARCH_SEED else ['tune'],
                                                            'validation_site_id': config.validation_site_id,
                                                            'selected_metrics': metrics['tune'], 'protocol_hash': auth['protocol_hash']})
            torch.save(dict(common, model=model.state_dict(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                            epoch=epoch, bad_epochs=bad), destination / 'last.pt')
            reason = ('validation_plateau' if epoch >= config.min_epochs and bad >= config.patience else
                      'safety_wallclock_stop' if time.monotonic() - start >= config.max_seconds else
                      'max_epochs_needs_validation_review' if epoch == config.max_epochs else None)
            if reason:
                result = {'status': 'complete' if reason == 'validation_plateau' else 'blocked', 'stop_reason': reason,
                          'completed_epochs': epoch, 'selected_score': best, 'protocol_hash': auth['protocol_hash'],
                          'selection_sha256': file_hash(destination / 'selection.json'), 'history_sha256': file_hash(destination / 'history.json'),
                          'checkpoint_sha256': file_hash(destination / 'best.pt'), 'runtime_policy': RUNTIME_POLICY,
                          'device': torch.cuda.get_device_name(0), 'heldout_outcomes_used': False,
                          'candidate_id': config.candidate_id, 'seed': config.seed,
                          'validation_site_id': config.validation_site_id,
                          'selected_metrics': json.loads((destination / 'selection.json').read_text())['selected_metrics'],
                          'train_n': len(train_rows), 'tune_n': len(tune_rows),
                          'search_space_hash': object_hash(bounded_search_space()),
                          'preprocessing_hash': protocol['preprocessing_hash']}
                _write_json(destination / 'status.json', result)
                return result
    except Exception as exc:
        _write_json(destination / 'status.json', {'status': 'failed', 'failure_type': type(exc).__name__,
                                                'failure_reason': str(exc), 'protocol_hash': auth['protocol_hash']})
        raise


def main():
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--authorization', required=True)
    p.add_argument('--candidate', required=True)
    p.add_argument('--seed', type=int, default=SEARCH_SEED)
    p.add_argument('--output', required=True)
    p.add_argument('--validation-site-id', default='')
    args = p.parse_args()
    # First authorize before obtaining private rows or opening any data file.
    from .common import verify_training_authorization
    auth = verify_training_authorization(args.authorization)
    rows = [r for r in auth['manifest_rows'] if r['partition'] in {'train', 'tune'}]
    result = train_candidate(rows, TrainConfig(args.candidate, args.seed, auth['protocol_hash'], validation_site_id=args.validation_site_id), args.authorization, args.output)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
