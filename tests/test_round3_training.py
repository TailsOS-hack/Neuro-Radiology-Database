"""Synthetic optimization and leakage gates; never authorize patient fitting."""
from dataclasses import asdict
import copy
import json
from unittest.mock import patch
import nibabel as nib
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset
from research.dementia_round3 import training as t
from research.dementia_round3.models import ModelConfig, build_model
from research.dementia_round3.preprocessing import file_hash

torch.set_num_threads(1)


def rows():
    return [dict(patient_id=f'p{i}', site_id=s, institution_id=s, diagnosis=y,
                 partition=part, volume_sha256=f'{i:064x}')
            for i, (s, y, part) in enumerate((('a', 0, 'train'), ('a', 1, 'train'),
                ('a', 0, 'tune'), ('a', 1, 'tune'), ('b', 0, 'train'), ('b', 1, 'train'),
                ('b', 0, 'tune'), ('b', 1, 'tune')))]


def test_exact_bounded_search_and_fixed_baseline_schedule():
    configs = t.bounded_search_space()
    assert len(configs) == 24
    assert len({json.dumps(c, sort_keys=True) for c in configs}) == 24
    assert {c['learning_rate'] for c in configs} == {3e-5, 1e-4, 3e-4}
    assert {c['weight_decay'] for c in configs} == {1e-4, .01}
    assert {c['model']['dropout'] for c in configs} == {.2, .5}
    assert t.anatomical_baseline_spec()['weight_decay'] == 1
    t.TrainConfig('candidate_01', validation_site_id='a').validate()
    t.TrainConfig('candidate_01', seed=t.FINAL_SEEDS[0]).validate()
    for change in [dict(max_epochs=300), dict(patience=5), dict(seed=7), dict(batch_size=3)]:
        with pytest.raises(ValueError):
            t.TrainConfig('candidate_01', validation_site_id='a', **change).validate()
    with pytest.raises(ValueError):
        t.TrainConfig('candidate_01').validate()


def test_loso_pools_development_without_mutating_manifest_rows():
    original = rows(); before = copy.deepcopy(original)
    train, tune = t.development_fold(original, t.TrainConfig('candidate_01', validation_site_id='a'))
    assert {r['site_id'] for r in train} == {'b'}
    assert {r['site_id'] for r in tune} == {'a'}
    assert {r['partition'] for r in tune} == {'train', 'tune'}
    assert original == before
    train, tune = t.development_fold(original, t.TrainConfig('candidate_01', seed=t.FINAL_SEEDS[1]))
    assert {r['partition'] for r in train} == {'train'}
    assert {r['partition'] for r in tune} == {'tune'}
    unsafe = copy.deepcopy(original); unsafe[0]['partition'] = 'calibration'
    with pytest.raises(ValueError, match='exactly'):
        t.development_fold(unsafe, t.TrainConfig('candidate_01', validation_site_id='a'))
    unsafe = copy.deepcopy(original)
    for r in unsafe: r['institution_id'] = 'shared'
    with pytest.raises(ValueError, match='shares an institution'):
        t.development_fold(unsafe, t.TrainConfig('candidate_01', validation_site_id='a'))


def test_sampler_balances_each_site_class_mass_and_rejects_missing_cell():
    roster = rows() + [dict(rows()[0], patient_id='extra')]
    weights = t.balanced_site_diagnosis_weights(roster)
    cell_mass = {(s, y): sum(float(w) for r, w in zip(roster, weights) if (r['site_id'], r['diagnosis']) == (s, y))
                 for s in ('a', 'b') for y in (0, 1)}
    assert set(cell_mass.values()) == {1.0}
    with pytest.raises(ValueError, match='both diagnoses'):
        t.balanced_site_diagnosis_weights([r for r in rows() if not (r['site_id'] == 'a' and r['diagnosis'] == 1)])


def test_metrics_are_patient_and_equal_site_weighted():
    roster = [dict(site_id=s, diagnosis=y) for s, y in [('a', 0), ('a', 1), ('b', 0), ('b', 1)]]
    result = t.patient_metrics(roster, [[.8, .2], [.2, .8], [.3, .7], [.9, .1]])
    assert result['site_balanced_accuracy'] == {'a': 1., 'b': 0.}
    assert result['site_macro_balanced_accuracy'] == .5
    assert result['n'] == 4
    assert result['site_macro_nll'] == pytest.approx(np.mean([-np.log(.8), -np.log(.8), -np.log(.3), -np.log(.1)]))
    with pytest.raises(ValueError, match='probabilities'):
        t.patient_metrics(roster, [[.8, .8]] * 4)


def test_synthetic_patient_optimizer_reduces_loss_and_counts_draws():
    torch.manual_seed(1)
    features = torch.tensor([[-2., -1.], [-1., -2.], [1., 2.], [2., 1.]])
    labels = torch.tensor([0, 0, 1, 1])
    model = torch.nn.Linear(2, 2)
    loader = DataLoader(TensorDataset(features, labels, torch.arange(4)), batch_size=4)
    optimizer = torch.optim.SGD(model.parameters(), lr=.2)
    initial = float(torch.nn.functional.cross_entropy(model(features), labels).detach())
    for _ in range(10):
        record = t.train_epoch(model, loader, optimizer, 'cpu', augment=False)
    final = float(torch.nn.functional.cross_entropy(model(features), labels).detach())
    assert final < initial / 2
    assert record['draws'] == record['unique_patient_draws'] == 4
    assert record['sampled_indices'] == [0, 1, 2, 3]


def test_anatomical_exact_l2_objective_excludes_intercept():
    model = build_model(ModelConfig(kind='anatomical_logistic'))
    x = torch.randn(4, 8); model.fit_normalizer(x)
    model.classifier.weight.data.fill_(.2); model.classifier.bias.data.fill_(2)
    loader = DataLoader(TensorDataset(x, torch.tensor([0, 1, 0, 1]), torch.arange(4)), batch_size=4)
    optimizer = torch.optim.SGD(model.parameters(), lr=0)
    record = t.train_epoch(model, loader, optimizer, 'cpu', augment=False, anatomical_l2=1)
    assert record['penalized_objective'] - record['patient_optimization_ce'] == pytest.approx(.5 * 8 * .2**2, abs=1e-6)


def test_authorization_is_checked_before_any_dataset_or_checkpoint_read(tmp_path):
    with patch('research.dementia_round3.common.verify_training_authorization', side_effect=PermissionError('blocked')), \
         patch.object(t, 'load_prepared_patient', side_effect=AssertionError('opened patient')), \
         patch.object(torch, 'load', side_effect=AssertionError('opened checkpoint')):
        with pytest.raises(PermissionError, match='blocked'):
            t.train_candidate(rows(), t.TrainConfig('candidate_01', validation_site_id='a'), 'missing.json', tmp_path / 'run')
    assert not (tmp_path / 'run').exists()


def test_cpu_is_never_a_real_training_fallback():
    with pytest.raises(PermissionError, match='B300'):
        t._runtime_policy_check(t.TrainConfig('candidate_01', device='cpu'))


def test_prepared_volume_requires_hashes_and_prediction_blind_review(tmp_path):
    volume = tmp_path / 'normalized.nii.gz'
    nib.save(nib.Nifti1Image(np.ones((16, 16, 16), np.float32), np.eye(4)), volume)
    record = dict(status='passed_automated_qc_requires_blinded_review', source_volume_sha256='a' * 64,
        preprocessing_hash='b' * 64, processed_volume_path=str(volume), processed_volume_sha256=file_hash(volume),
        shape=[16, 16, 16], affine=np.eye(4).tolist())
    path = tmp_path / 'record.json'; path.write_text(json.dumps(record))
    review = tmp_path / 'review.json'; review.write_text(json.dumps(dict(status='passed', diagnosis_blinded=True,
        prediction_blinded=True, preprocessing_record_sha256=file_hash(path))))
    row = dict(volume_sha256='a' * 64, preprocessing_record_path=str(path), preprocessing_record_sha256=file_hash(path),
        preprocessing_review_path=str(review), preprocessing_review_sha256=file_hash(review))
    assert t.load_prepared_patient(row, 'b' * 64, ModelConfig()).shape == (1, 16, 16, 16)
    with pytest.raises(ValueError, match='Unapproved'):
        t.load_prepared_patient(row, 'c' * 64, ModelConfig())
    review.write_text('{}')
    with pytest.raises(ValueError, match='changed'):
        t.load_prepared_patient(row, 'b' * 64, ModelConfig())
