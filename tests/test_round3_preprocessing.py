"""Synthetic geometry/dependency/QC checks; no external tools or MRI downloaded."""
from dataclasses import replace
import json
from types import SimpleNamespace
from unittest.mock import patch
import nibabel as nib
import numpy as np
import pytest
from research.dementia_round3 import preprocessing as p

META = {'modality': 'T1w', 'acquisition_dimension': '3D', 'source_evidence': 'synthetic-test-metadata'}


def image(shape=(80, 80, 80), spacing=2):
    data = np.random.default_rng(12).uniform(1, 100, shape).astype(np.float32)
    out = nib.Nifti1Image(data, np.diag([spacing, spacing, spacing, 1]))
    out.set_qform(out.affine, 1); out.set_sform(out.affine, 1)
    return out


def test_geometry_metadata_and_native_ras_orientation():
    result = p.inspect_nifti(image(), p.PreprocessingConfig(), META)
    assert result['orientation'] == ['R', 'A', 'S']
    assert result['fov_mm'] == [160., 160., 160.]
    assert result['finite']
    with pytest.raises(ValueError, match='Provider'):
        p.inspect_nifti(image(), p.PreprocessingConfig(), {})
    with pytest.raises(ValueError, match='3D'):
        p.inspect_nifti(image((80, 80, 80, 2)), p.PreprocessingConfig(), META)
    with pytest.raises(ValueError, match='Field of view'):
        p.inspect_nifti(image((64, 64, 64)), p.PreprocessingConfig(), META)


@pytest.mark.parametrize('failure', ['qform_missing', 'qform_conflict', 'nan', 'negative', 'constant', 'spacing', 'shear'])
def test_invalid_acquisition_fails(failure):
    data = image()
    if failure == 'qform_missing':
        data.set_qform(data.affine, 0); data.set_sform(data.affine, 0)
    elif failure == 'qform_conflict':
        other = data.affine.copy(); other[0, 3] += 4; data.set_qform(other, 1)
    elif failure in ('nan', 'negative', 'constant'):
        array = data.get_fdata(dtype=np.float32)
        if failure == 'constant': array[:] = 5
        else: array[0, 0, 0] = np.nan if failure == 'nan' else -1
        data = nib.Nifti1Image(array, data.affine)
    elif failure == 'spacing':
        data = image(spacing=3)
    else:
        affine = data.affine.copy(); affine[0, 1] = .6
        data = nib.Nifti1Image(data.get_fdata(), affine)
    with pytest.raises(ValueError):
        p.inspect_nifti(data, p.PreprocessingConfig(), META)


def test_normalization_is_masked_per_volume_and_preserves_background():
    data = image((16, 16, 16)).get_fdata(dtype=np.float32)
    mask = np.zeros_like(data, bool); mask[2:-2, 2:-2, 2:-2] = True
    normalized, record = p.normalize_brain(data, mask)
    assert np.all(normalized[~mask] == 0)
    assert abs(float(normalized[mask].mean())) < 1e-6
    assert abs(float(normalized[mask].std()) - 1) < 1e-6
    assert record['fit_scope'].startswith('this volume only')
    with pytest.raises(ValueError):
        p.normalize_brain(np.ones_like(data), mask)


def test_mask_qc_detects_truncation_disconnection_and_nonbinary_masks():
    cfg = replace(p.PreprocessingConfig(), min_brain_ml=1, max_brain_ml=100)
    mask = np.zeros((24, 24, 24), bool); mask[3:21, 3:21, 3:21] = True
    affine = np.eye(4)
    assert p.mask_qc(mask, affine, cfg, mask)['template_mask_dice'] == 1
    with pytest.raises(ValueError, match='boundary'):
        p.mask_qc(np.ones_like(mask), affine, cfg)
    disconnected = np.zeros_like(mask); disconnected[1:9, 1:9, 1:9] = True; disconnected[15:23, 15:23, 15:23] = True
    with pytest.raises(ValueError, match='Disconnected'):
        p.mask_qc(disconnected, affine, cfg)
    with pytest.raises(ValueError, match='binary'):
        p.binary_mask(nib.Nifti1Image(mask.astype(np.float32) * .7, affine))


def test_anatomical_features_recompute_actual_labels_not_intensity_threshold():
    labels = np.zeros((8, 8, 8), np.int16)
    labels[1:3, 1:3, 1:3] = 1; labels[4:6, 4:6, 4:6] = 2
    icv = nib.Nifti1Image(np.ones((8,8,8), np.uint8), np.eye(4))
    segmentation = nib.Nifti1Image(labels, np.eye(4))
    result = p.anatomical_volume_fractions(segmentation, icv, {'a': [1], 'b': [2]}, ['a', 'b'])
    np.testing.assert_allclose(result, [8/512, 8/512])
    with pytest.raises(ValueError, match='disjoint'):
        p.anatomical_volume_fractions(segmentation, icv, {'a': [1], 'b': [1]}, ['a', 'b'])
    with pytest.raises(ValueError, match='missing'):
        p.anatomical_volume_fractions(segmentation, icv, {'a': [1], 'b': [3]}, ['a', 'b'])


def test_missing_tools_fail_before_patient_read_and_preserve_failure(tmp_path):
    cfg = p.PreprocessingConfig()
    assert p.preflight(cfg)['status'] == 'blocked'
    with patch.object(p.nib, 'load', side_effect=AssertionError('patient read before dependency gate')):
        record = p.preprocess_volume({'volume_path': 'missing.nii.gz', 'volume_sha256': 'a' * 64}, tmp_path / 'failure', cfg)
    assert record['status'] == 'failed'
    assert 'dependencies' in record['failure_reason']
    assert json.loads((tmp_path / 'failure/preprocessing.json').read_text()) == record
    with pytest.raises(ValueError, match='preserve'):
        p.preprocess_volume({}, tmp_path / 'failure', cfg)


def test_preflight_checks_version_return_code_and_bytes(tmp_path):
    executable = tmp_path / 'fake_tool'
    executable.write_text('#!/bin/sh\necho synthetic-1.0\nexit 1\n'); executable.chmod(0o755)
    asset = tmp_path / 'fake_asset'; asset.write_text('synthetic runtime/model/template pin only')
    pin = {'path': str(executable), 'sha256': p.file_hash(executable), 'version': 'synthetic-1.0',
           'version_args': ['--version'], 'version_returncode': 1}
    cfg = p.PreprocessingConfig(tools={name: pin.copy() for name in p.TOOLS},
        runtime_assets=({'path': str(asset), 'sha256': p.file_hash(asset)},),
        synthstrip_weights=str(asset), synthstrip_weights_sha256=p.file_hash(asset),
        template=str(asset), template_sha256=p.file_hash(asset), template_mask=str(asset), template_mask_sha256=p.file_hash(asset))
    assert p.preflight(cfg)['status'] == 'ready'
    executable.write_text('changed')
    assert p.preflight(cfg)['status'] == 'blocked'


def test_synthetic_external_command_contract_persists_readback_and_qc(tmp_path):
    # Mocked tool outputs exercise sequencing/provenance, not N4/ANTs efficacy.
    source = tmp_path / 'source.nii.gz'; nib.save(image((16, 16, 16)), source)
    mask = np.zeros((16, 16, 16), np.uint8); mask[2:-2, 2:-2, 2:-2] = 1
    template = tmp_path / 'template.nii.gz'; nib.save(image((16, 16, 16)), template)
    maskpath = tmp_path / 'mask.nii.gz'; nib.save(nib.Nifti1Image(mask, np.diag([2, 2, 2, 1])), maskpath)
    cfg = replace(p.PreprocessingConfig(), tools={name: {'path': name} for name in p.TOOLS},
        template=str(template), template_mask=str(maskpath), target_shape=(16, 16, 16),
        min_dimension=16, min_fov_mm=16, min_brain_ml=1)
    dest = tmp_path / 'processed'
    def fake_run(command, **kwargs):
        if command[0] == 'N4BiasFieldCorrection':
            nib.save(nib.load(source), command[command.index('-o') + 1])
        elif command[0] == 'mri_synthstrip':
            nib.save(nib.load(source), command[command.index('-o') + 1])
            nib.save(nib.load(maskpath), command[command.index('-m') + 1])
        elif command[0] == 'antsRegistration':
            nib.save(nib.load(source), dest / 'registered.nii.gz')
        else:
            nib.save(nib.load(maskpath), command[command.index('-o') + 1])
        return SimpleNamespace(returncode=0, stdout='synthetic', stderr='')
    row = {'volume_path': str(source), 'volume_sha256': p.file_hash(source), 't1_metadata': META}
    with patch.object(p, 'preflight', return_value={'status': 'ready'}), patch.object(p.subprocess, 'run', side_effect=fake_run):
        result = p.preprocess_volume(row, dest, cfg)
    assert result['status'] == 'passed_automated_qc_requires_blinded_review', result
    assert len(result['commands']) == 4
    assert result['rigid_mask_volume_ratio'] == 1
    assert result['processed_volume_sha256'] == p.file_hash(dest / 'normalized.nii.gz')
    assert result['anatomical_segmentation_available'] is False
