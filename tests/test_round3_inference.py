"""Synthetic checkpoint and authorization tests; no real patient prediction."""
from dataclasses import asdict
import hashlib
import importlib.util
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
from PIL import Image
import pytest
import torch
from research.dementia_round3 import inference as i
from research.dementia_round3.models import MODEL_VERSION, ModelConfig, build_model
from research.dementia_round3.preprocessing import PreprocessingConfig, file_hash
from research.dementia_round3.training import FINAL_SEEDS

torch.set_num_threads(1)


def write_json(path, obj):
    path.write_text(json.dumps(obj)); return {'path': str(path), 'sha256': file_hash(path)}


def synthetic_bundle(tmp_path, calibration=None):
    cfg = PreprocessingConfig()
    auth = {'phase': 'development', 'protocol_hash': 'a' * 64, 'protocol': {'preprocessing_hash': cfg.version_hash}}
    refs = []
    for seed in FINAL_SEEDS:
        torch.manual_seed(seed)
        model = build_model(ModelConfig(kind='resnet3d18', width=4, dropout=0))
        path = tmp_path / f'{seed}.pt'
        torch.save(dict(model_version=MODEL_VERSION, protocol_hash=auth['protocol_hash'], preprocessing_hash=cfg.version_hash,
            research_mode='real_authorized_development', seed=seed, epoch=4, selection_metric='site_macro_tune_nll',
            validation_site_id='', model_config=asdict(model.config), candidate={'candidate_id': 'synthetic'}, model=model.state_dict()), path)
        refs.append(dict(path=str(path), sha256=file_hash(path), seed=seed, epoch=4))
    bundle = dict(version='round3_model_bundle_v1', class_order=['CN', 'AD'], protocol_hash=auth['protocol_hash'],
        preprocessing_hash=cfg.version_hash, model_version=MODEL_VERSION, aggregation='equal_probability_mean',
        research_only=True, checkpoints=refs)
    if calibration: bundle['calibration'] = calibration
    path = tmp_path / 'bundle.json'; write_json(path, bundle)
    return cfg, auth, path, bundle


def test_gate_precedes_checkpoint_and_dataset_reads():
    with patch('research.dementia_round3.common.verify_training_authorization', side_effect=PermissionError('dataset gate blocked')), \
         patch.object(torch, 'load', side_effect=AssertionError('checkpoint opened')), \
         patch.object(i, 'load_prepared_patient', side_effect=AssertionError('patient opened')):
        with pytest.raises(PermissionError, match='dataset gate'):
            i.predict_volume({}, 'missing_bundle.json', 'authorization.json', PreprocessingConfig())
        with pytest.raises(PermissionError, match='dataset gate'):
            i.predict_legacy_clinical({}, 'missing_bundle.json', 'authorization.json')


def test_missing_tools_abstain_before_any_checkpoint(tmp_path):
    cfg = PreprocessingConfig()
    auth = {'protocol': {'preprocessing_hash': cfg.version_hash}}
    with patch.object(i, '_authorize', return_value=auth), patch.object(torch, 'load', side_effect=AssertionError('opened')):
        result = i.predict_volume({}, tmp_path / 'does-not-exist', 'auth', cfg)
    assert result['qc_status'] == 'abstained'
    assert result['score'] is None and result['raw_score'] is None
    assert result['dependency_errors']
    assert result['clinical_deployment_permitted'] is False


def test_synthetic_strict_ensemble_inference_preserves_seed_identity(tmp_path):
    cfg, auth, path, bundle = synthetic_bundle(tmp_path)
    x = torch.randn(1, 16, 16, 16)
    with patch.object(i, '_authorize', return_value=auth), patch.object(i, 'preflight', return_value={'status': 'ready'}), \
         patch.object(i, 'load_prepared_patient', return_value=x):
        result = i.predict_volume({}, path, 'auth', cfg)
    assert result['qc_status'] == 'ok', result
    assert result['seeds'] == list(FINAL_SEEDS)
    assert result['score'] == pytest.approx(np.mean(result['individual_scores']))
    assert result['prediction'] is None and result['threshold'] is None
    assert result['score_type'] == 'uncalibrated_development_score'
    bundle['checkpoints'].pop(); write_json(path, bundle)
    with patch.object(i, '_authorize', return_value=auth), patch.object(i, 'preflight', return_value={'status': 'ready'}):
        failure = i.predict_volume({}, path, 'auth', cfg)
    assert failure['qc_status'] == 'failed' and 'seed' in failure['abstention_reason']


def test_changed_checkpoint_and_missing_review_are_not_silently_accepted(tmp_path):
    cfg, auth, path, bundle = synthetic_bundle(tmp_path)
    Path(bundle['checkpoints'][0]['path']).write_bytes(b'changed')
    with patch.object(i, '_authorize', return_value=auth), patch.object(i, 'preflight', return_value={'status': 'ready'}), \
         patch.object(torch, 'load', side_effect=AssertionError('unverified checkpoint loaded')):
        result = i.predict_volume({}, path, 'auth', cfg)
    assert result['qc_status'] == 'failed' and 'checksum' in result['abstention_reason']


def test_frozen_calibration_applies_and_missing_threshold_abstains(tmp_path):
    cal = dict(status='fitted_development_only', role='development_calibration', class_order=['CN', 'AD'],
        score_field='new_score', intercept=.2, slope=.9, provenance={'protocol_sha256': 'a' * 64},
        threshold=.6, threshold_status='available')
    cal_path = tmp_path / 'cal.json'; cal_ref = write_json(cal_path, cal)
    cfg, auth, path, bundle = synthetic_bundle(tmp_path, cal_ref)
    auth['phase'] = 'external_evaluation'
    selection = tmp_path / 'selection.json'
    auth['selection_freeze'] = {'manifest': write_json(selection, {'artifacts': {
        'new_model_bundle': {'path': str(path), 'sha256': file_hash(path)}, 'calibration': cal_ref}})}
    with patch.object(i, '_authorize', return_value=auth), patch.object(i, 'preflight', return_value={'status': 'ready'}), \
         patch.object(i, 'load_prepared_patient', return_value=torch.randn(1, 16, 16, 16)):
        result = i.predict_volume({}, path, 'auth', cfg)
    assert result['qc_status'] == 'ok', result
    assert result['prediction'] == int(result['score'] >= .6)
    from research.dementia_round3.calibration import apply_calibration
    assert result['score'] == apply_calibration([result['raw_score']], cal)[0]
    cal['threshold'] = None; cal['threshold_status'] = 'unavailable'
    bundle['calibration'] = write_json(cal_path, cal); write_json(path, bundle)
    auth['phase'] = 'development'
    with patch.object(i, '_authorize', return_value=auth), patch.object(i, 'preflight', return_value={'status': 'ready'}), \
         patch.object(i, 'load_prepared_patient', return_value=torch.randn(1, 16, 16, 16)):
        result = i.predict_volume({}, path, 'auth', cfg)
    assert result['score'] is None and result['raw_score'] is not None
    assert result['qc_status'] == 'abstained'


def test_legacy_nine_slice_transform_is_exact_and_rejects_changed_pixels(tmp_path):
    slices = []
    pixels = np.repeat(np.tile(np.arange(224, dtype=np.uint8), (224, 1))[:, :, None], 3, axis=2)
    for n, fraction in enumerate([.45, .50, .55, .60, .65, .70, .75, .80, .85]):
        path = tmp_path / f'{n}.png'; Image.fromarray(pixels).save(path)
        slices.append(dict(path=str(path), sha256=file_hash(path), pixel_sha256=hashlib.sha256(pixels.tobytes()).hexdigest(),
            slice_fraction=fraction, slice_index=30+n))
    record = tmp_path / 'legacy.json'
    ref = write_json(record, dict(source_volume_sha256='a'*64, preprocessing_version='exploratory_t1_axial_v1',
        preprocessor_sha256=i.LEGACY_PREPROCESSOR_SHA256, slices=slices))
    review = tmp_path / 'review.json'
    rev = write_json(review, dict(status='passed', diagnosis_blinded=True, prediction_blinded=True,
        legacy_preprocessing_record_sha256=ref['sha256']))
    row = dict(volume_sha256='a'*64, legacy_preprocessing_record_path=ref['path'], legacy_preprocessing_record_sha256=ref['sha256'],
        legacy_review_path=rev['path'], legacy_review_sha256=rev['sha256'])
    tensor = i._legacy_bag(row)
    expected = torch.from_numpy(pixels.astype(np.float32)/255).permute(2,0,1)
    expected = (expected-torch.tensor([.485,.456,.406])[:,None,None])/torch.tensor([.229,.224,.225])[:,None,None]
    assert tensor.shape == (9,3,224,224)
    assert torch.equal(tensor[0], expected)
    Path(slices[0]['path']).write_bytes(b'changed')
    with pytest.raises(ValueError, match='checksum'):
        i._legacy_bag(row)


def test_legacy_aux_forward_matches_frozen_helper_with_synthetic_weights():
    root = Path(__file__).resolve().parents[1]
    path = root/'docs/kevin_paper_20261011/training/train_candidate.py'
    spec = importlib.util.spec_from_file_location('round3_synthetic_legacy_reference', path)
    legacy = importlib.util.module_from_spec(spec); spec.loader.exec_module(legacy); legacy.imports()
    torch.manual_seed(17)
    model, aux = legacy.network(pretrained=False); model.eval(); aux.eval()
    x = torch.randn(2,3,64,64)
    with torch.inference_mode():
        actual = i._legacy_aux_forward(model, aux, x)
        expected = legacy.forward(model, aux, x, 'aux')
    assert torch.equal(actual, expected)
    assert actual.shape == (2,2)


def test_legacy_checkpoint_fixed_identity_rejected_before_pickle(tmp_path):
    path = tmp_path/'unsafe.pt'; path.write_bytes(b'untrusted')
    refs = [dict(path=str(path),sha256=file_hash(path),seed=s) for s in i.LEGACY_CHECKPOINTS]
    with patch.object(torch, 'load', side_effect=AssertionError('unsafe pickle loaded')):
        with pytest.raises(ValueError, match='immutable'):
            i._load_legacy_checkpoints(refs, 'cpu')


def test_batch_authorizes_and_loads_once_then_checks_every_case(tmp_path):
    cfg, auth, path, _ = synthetic_bundle(tmp_path)
    raw = tmp_path/'synthetic_raw'; raw.write_bytes(b'synthetic input whose MRI loader is mocked')
    rows = [dict(patient_id=f'synthetic{n}', volume_path=str(raw), volume_sha256=file_hash(raw)) for n in range(2)]
    with patch.object(i, '_authorize', return_value=auth) as authorize, \
         patch.object(i, 'preflight', return_value={'status': 'ready'}), \
         patch.object(i, '_load_bundle', wraps=i._load_bundle) as loader, \
         patch.object(i, 'load_prepared_patient', return_value=torch.randn(1,16,16,16)) as patient_loader:
        with i.InferenceSession(rows, 'auth', cfg, new_bundle_path=path) as batch:
            output = [batch.predict(r) for r in rows]
        assert authorize.call_count == loader.call_count == 1
        assert patient_loader.call_count == 2
    assert all(r['new']['qc_status'] == 'ok' for r in output)
    assert batch.closed and not batch.loaded
    with pytest.raises(PermissionError, match='closed'):
        batch.predict(rows[0])


def test_batch_rejects_omissions_repeats_changed_rows_and_model_artifacts(tmp_path):
    cfg, auth, path, bundle = synthetic_bundle(tmp_path)
    raw = tmp_path/'synthetic_raw'; raw.write_bytes(b'synthetic')
    row = dict(patient_id='synthetic', volume_path=str(raw), volume_sha256=file_hash(raw))
    with patch.object(i, '_authorize', return_value=auth), patch.object(i, 'preflight', return_value={'status': 'ready'}), \
         patch.object(i, 'load_prepared_patient', return_value=torch.randn(1,16,16,16)):
        with pytest.raises(ValueError, match='omitted'):
            with i.InferenceSession([row], 'auth', cfg, new_bundle_path=path):
                pass
        with pytest.raises(PermissionError, match='outside'):
            with i.InferenceSession([row], 'auth', cfg, new_bundle_path=path) as batch:
                batch.predict({**row, 'diagnosis': 1})
        with pytest.raises(PermissionError, match='repeated'):
            with i.InferenceSession([row], 'auth', cfg, new_bundle_path=path) as batch:
                batch.predict(row); batch.predict(row)
        with pytest.raises(ValueError, match='artifact/code changed'):
            with i.InferenceSession([row], 'auth', cfg, new_bundle_path=path) as batch:
                batch.predict(row)
                Path(bundle['checkpoints'][0]['path']).write_bytes(b'changed after prediction')


def test_batch_keeps_changed_raw_case_as_failure_in_denominator(tmp_path):
    cfg, auth, path, _ = synthetic_bundle(tmp_path)
    raw = tmp_path/'synthetic_raw'; raw.write_bytes(b'synthetic')
    row = dict(patient_id='synthetic', volume_path=str(raw), volume_sha256=file_hash(raw))
    with patch.object(i, '_authorize', return_value=auth), patch.object(i, 'preflight', return_value={'status': 'ready'}), \
         patch.object(i, 'load_prepared_patient', side_effect=AssertionError('changed input loaded')):
        with i.InferenceSession([row], 'auth', cfg, new_bundle_path=path) as batch:
            raw.write_bytes(b'changed')
            result = batch.predict(row)
    assert result['new']['qc_status'] == 'failed'
    assert result['new']['score'] is None
