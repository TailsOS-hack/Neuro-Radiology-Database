"""Pinned external N4/SynthStrip/ANTs preprocessing, with explicit failure records.

No external tool, atlas, segmentation model or weight is downloaded implicitly.
Missing dependencies block processing; the QC here is not clinical adjudication.
Official interfaces: https://surfer.nmr.mgh.harvard.edu/docs/synthstrip/ and
https://github.com/ANTsX/ANTs/wiki/N4BiasFieldCorrection .
"""
from dataclasses import asdict, dataclass, field
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import nibabel as nib
import numpy as np
from scipy import ndimage

PREPROCESSING_VERSION = 'round3-n4-synthstrip-rigid-template-v1'
TOOLS = ('N4BiasFieldCorrection', 'mri_synthstrip', 'antsRegistration', 'antsApplyTransforms')


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def object_hash(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class PreprocessingConfig:
    # Each tool entry requires path, sha256, version, version_args. Runtime
    # libraries/container image must also be pinned in runtime_assets.
    tools: dict = field(default_factory=dict)
    runtime_assets: tuple = ()
    synthstrip_weights: str = ''
    synthstrip_weights_sha256: str = ''
    template: str = ''
    template_sha256: str = ''
    template_mask: str = ''
    template_mask_sha256: str = ''
    target_shape: tuple = (96, 112, 96)
    target_spacing_mm: float = 2.0
    min_dimension: int = 64
    max_dimension: int = 1024
    min_spacing_mm: float = 0.4
    max_spacing_mm: float = 2.0
    min_fov_mm: float = 140.0
    max_fov_mm: float = 350.0
    min_brain_ml: float = 400.0
    max_brain_ml: float = 2200.0
    min_template_mask_dice: float = 0.65
    max_boundary_brain_fraction: float = 0.005
    timeout_seconds: int = 1800

    @property
    def version_hash(self):
        return object_hash({'version': PREPROCESSING_VERSION, 'config': asdict(self)})


def _pinned(path, expected):
    p = Path(path)
    if not path or not p.is_file() or p.is_symlink() or len(expected) != 64 or file_hash(p) != expected:
        raise ValueError('Missing or changed pinned dependency: ' + str(path))
    return p.resolve()


def preflight(config):
    """Checks only dependencies. It never opens a patient image."""
    errors, versions = [], {}
    for name in TOOLS:
        pin = config.tools.get(name, {})
        try:
            p = _pinned(pin.get('path', ''), pin.get('sha256', ''))
            if not os.access(p, os.X_OK) or not pin.get('version') or not isinstance(pin.get('version_args'), list):
                raise ValueError('Missing executable/version pin')
            run = subprocess.run([str(p), *pin['version_args']], capture_output=True, text=True, timeout=30)
            observed = (run.stdout + run.stderr).strip()
            if run.returncode != pin.get('version_returncode', 0) or pin['version'] not in observed:
                raise ValueError('Executable version does not match declared pin')
            versions[name] = observed
        except (ValueError, OSError, subprocess.SubprocessError) as exc:
            errors.append(name + ': ' + str(exc))
    for label, path, expected in [('SynthStrip weights', config.synthstrip_weights, config.synthstrip_weights_sha256),
                                  ('template', config.template, config.template_sha256),
                                  ('template mask', config.template_mask, config.template_mask_sha256)]:
        try:
            _pinned(path, expected)
        except ValueError as exc:
            errors.append(label + ': ' + str(exc))
    if not config.runtime_assets:
        errors.append('Runtime dependency/container lock is not pinned')
    for pin in config.runtime_assets:
        try:
            _pinned(pin['path'], pin['sha256'])
        except (ValueError, KeyError) as exc:
            errors.append('runtime: ' + str(exc))
    return {'status': 'blocked' if errors else 'ready', 'errors': errors,
            'versions': versions, 'preprocessing_version': PREPROCESSING_VERSION,
            'preprocessing_hash': config.version_hash}


def inspect_nifti(image, config, metadata):
    if metadata.get('modality') != 'T1w' or metadata.get('acquisition_dimension') != '3D' or not metadata.get('source_evidence'):
        raise ValueError('Provider-backed 3D T1 acquisition metadata required; filename is insufficient')
    if len(image.shape) != 3 or not all(config.min_dimension <= n <= config.max_dimension for n in image.shape):
        raise ValueError('Not a supported complete 3D volume')
    affine = np.asarray(image.affine, float)
    if affine.shape != (4, 4) or not np.isfinite(affine).all() or not np.allclose(affine[3], [0, 0, 0, 1]):
        raise ValueError('Invalid affine')
    spacing = nib.affines.voxel_sizes(affine)
    if abs(np.linalg.det(affine[:3, :3])) < 1e-8 or np.any(spacing < config.min_spacing_mm) or np.any(spacing > config.max_spacing_mm):
        raise ValueError('Singular affine or unsupported voxel spacing')
    directions = affine[:3, :3] / spacing
    if not np.allclose(directions.T @ directions, np.eye(3), atol=0.03):
        raise ValueError('Unsupported sheared affine')
    qform, qc = image.get_qform(coded=True)
    sform, sc = image.get_sform(coded=True)
    if not qc and not sc:
        raise ValueError('No authoritative qform/sform orientation')
    if qc and sc and not np.allclose(qform, sform, atol=0.1):
        raise ValueError('Conflicting qform and sform')
    fov = spacing * image.shape
    if np.any(fov < config.min_fov_mm) or np.any(fov > config.max_fov_mm):
        raise ValueError('Field of view outside prespecified full-brain bounds')
    data = image.get_fdata(dtype=np.float32)
    if not np.isfinite(data).all() or np.ptp(data) <= 0 or np.count_nonzero(data > 0) < 100:
        raise ValueError('Nonfinite, empty or constant MRI')
    if np.min(data) < 0:
        raise ValueError('Negative T1 intensities unsupported by this pinned N4 workflow')
    return {'shape': list(image.shape), 'spacing_mm': spacing.tolist(), 'fov_mm': fov.tolist(),
            'orientation': list(nib.aff2axcodes(affine)), 'affine': affine.tolist(),
            'qform_code': int(qc), 'sform_code': int(sc), 'finite': True}


def normalize_brain(data, mask):
    data, mask = np.asarray(data, np.float32), np.asarray(mask, bool)
    if data.shape != mask.shape or not np.isfinite(data).all() or mask.sum() < 100:
        raise ValueError('Invalid brain mask/intensities')
    lo, hi = np.percentile(data[mask], [0.5, 99.5])
    if hi <= lo:
        raise ValueError('No brain intensity variation')
    clipped = np.clip(data, lo, hi)
    mean, scale = float(clipped[mask].mean()), float(clipped[mask].std())
    if scale <= 1e-6:
        raise ValueError('Degenerate brain normalization')
    out = np.zeros_like(data)
    out[mask] = (clipped[mask] - mean) / scale
    return out, {'lower': float(lo), 'upper': float(hi), 'mean': mean, 'scale': scale,
                 'fit_scope': 'this volume only; no cohort or diagnosis fitted parameters'}


def binary_mask(image):
    data = image.get_fdata(dtype=np.float32)
    if not np.isfinite(data).all() or not np.isin(data, (0, 1)).all():
        raise ValueError('Expected a finite binary extraction mask')
    return data.astype(bool)


def anatomical_volume_fractions(segmentation, intracranial_mask, labels_by_feature, features):
    """Recompute named fractions from an actual, externally produced label map.

    Label meanings and segmentation/ICV provenance must be pinned separately.
    No intensity threshold is converted into an anatomical structure here.
    """
    labels = segmentation.get_fdata(dtype=np.float32)
    icv = binary_mask(intracranial_mask)
    if (labels.shape != icv.shape or not np.allclose(segmentation.affine, intracranial_mask.affine, atol=1e-5) or
            not np.isfinite(labels).all() or not np.equal(labels, np.rint(labels)).all() or
            np.any(labels < 0) or icv.sum() == 0 or set(labels_by_feature) != set(features)):
        raise ValueError('Invalid anatomical segmentation/ICV geometry or labels')
    used, fractions = set(), []
    for feature in features:
        identifiers = labels_by_feature[feature]
        if (not identifiers or any(type(v) is not int or v <= 0 for v in identifiers) or
                len(set(identifiers)) != len(identifiers) or used.intersection(identifiers)):
            raise ValueError('Anatomical labels must be positive, unique and disjoint')
        used.update(identifiers)
        region = np.isin(labels, identifiers)
        if not region.any() or np.any(region & ~icv):
            raise ValueError('Anatomical region missing or outside intracranial mask')
        fractions.append(float(region.sum() / icv.sum()))
    return np.asarray(fractions, np.float32)


def mask_qc(mask, affine, config, template_mask=None):
    mask = np.asarray(mask, bool)
    volume = float(mask.sum() * abs(np.linalg.det(affine[:3, :3])) / 1000)
    if not config.min_brain_ml <= volume <= config.max_brain_ml:
        raise ValueError('Extracted brain volume outside prespecified basic bounds')
    labels, _ = ndimage.label(mask)
    sizes = np.bincount(labels.ravel())[1:]
    if not len(sizes) or sizes.max() / mask.sum() < 0.98:
        raise ValueError('Disconnected extracted brain mask')
    boundary = np.zeros_like(mask)
    for axis in range(3):
        sl = [slice(None)] * 3
        for index in (0, -1):
            sl[axis] = index
            boundary[tuple(sl)] = True
    fraction = float((mask & boundary).sum() / mask.sum())
    if fraction > config.max_boundary_brain_fraction:
        raise ValueError('Brain touches volume boundary; possible truncated field of view')
    result = {'brain_ml': volume, 'boundary_brain_fraction': fraction}
    if template_mask is not None:
        other = np.asarray(template_mask, bool)
        if other.shape != mask.shape or other.sum() == 0:
            raise ValueError('Invalid template mask')
        dice = float(2 * (mask & other).sum() / (mask.sum() + other.sum()))
        if dice < config.min_template_mask_dice:
            raise ValueError('Registration mask overlap below prespecified QC bound')
        result['template_mask_dice'] = dice
    return result


def preprocess_volume(row, output_dir, config):
    """Processing utility used only after the caller's dataset authorization gate.

    Returns a persisted success/failure record, never an invented segmentation.
    Caller must diagnosis-blindly review registration/extraction before training.
    """
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise ValueError('Preprocessing destination must be new; preserve previous failures')
    output_dir.mkdir(parents=True)
    started = time.monotonic()
    record = {'version': PREPROCESSING_VERSION, 'preprocessing_hash': config.version_hash,
              'source_volume_sha256': row.get('volume_sha256'), 'status': 'failed',
              'commands': [], 'clinical_validity_established': False}
    try:
        check = preflight(config)
        record['preflight'] = check
        if check['status'] != 'ready':
            raise ValueError('Required pinned preprocessing dependencies unavailable')
        source = _pinned(row['volume_path'], row['volume_sha256'])
        image = nib.load(source)
        record['input_qc'] = inspect_nifti(image, config, row.get('t1_metadata', {}))
        canonical = nib.as_closest_canonical(image)
        native = output_dir / 'canonical.nii.gz'
        nib.save(canonical, native)
        template = nib.load(config.template)
        reference_mask = nib.load(config.template_mask)
        if tuple(template.shape) != tuple(config.target_shape) or not np.allclose(nib.affines.voxel_sizes(template.affine), config.target_spacing_mm):
            raise ValueError('Pinned template does not match the declared full-volume grid')
        if reference_mask.shape != template.shape or not np.allclose(reference_mask.affine, template.affine):
            raise ValueError('Template mask geometry differs')
        if not np.isfinite(template.get_fdata(dtype=np.float32)).all():
            raise ValueError('Template has nonfinite intensities')
        template_binary = binary_mask(reference_mask)
        corrected, brain, mask = [output_dir / n for n in ('n4.nii.gz', 'brain.nii.gz', 'brain_mask.nii.gz')]
        registered, warped_mask = output_dir / 'registered.nii.gz', output_dir / 'registered_mask.nii.gz'
        prefix = str(output_dir / 'rigid_')
        tool = lambda n: config.tools[n]['path']
        commands = [
            [tool('N4BiasFieldCorrection'), '-d', '3', '-i', str(native), '-o', str(corrected), '-s', '4', '-b', '[180]', '-c', '[50x50x50x50,0.0]'],
            [tool('mri_synthstrip'), '-i', str(corrected), '-o', str(brain), '-m', str(mask), '--model', config.synthstrip_weights],
            [tool('antsRegistration'), '--dimensionality', '3', '--float', '0', '--output', f'[{prefix},{registered}]', '--interpolation', 'Linear', '--initial-moving-transform', f'[{config.template},{brain},1]', '--transform', 'Rigid[0.1]', '--metric', f'MI[{config.template},{brain},1,32,Regular,0.25]', '--convergence', '[1000x500x250,1e-6,10]', '--shrink-factors', '4x2x1', '--smoothing-sigmas', '2x1x0vox', '--random-seed', '20261030'],
            [tool('antsApplyTransforms'), '-d', '3', '-i', str(mask), '-r', config.template, '-o', str(warped_mask), '-n', 'NearestNeighbor', '-t', prefix + '0GenericAffine.mat']]
        env = {**os.environ, 'ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS': '1'}
        for index, command in enumerate(commands):
            run = subprocess.run(command, capture_output=True, text=True, timeout=config.timeout_seconds, env=env)
            (output_dir / f'command{index}.log').write_text(run.stdout + '\n' + run.stderr)
            record['commands'].append({'argv': command, 'returncode': run.returncode})
            if run.returncode:
                raise ValueError('Pinned preprocessing command failed; log retained')
            if index == 1:
                m = nib.load(mask)
                if m.shape != canonical.shape or not np.allclose(m.affine, canonical.affine):
                    raise ValueError('SynthStrip changed native geometry')
                record['native_mask_qc'] = mask_qc(binary_mask(m), m.affine, config)
        reg, m = nib.load(registered), nib.load(warped_mask)
        if reg.shape != template.shape or m.shape != template.shape or not np.allclose(reg.affine, template.affine) or not np.allclose(m.affine, template.affine):
            raise ValueError('Registered output geometry differs from fixed grid')
        registered_binary = binary_mask(m)
        record['registered_mask_qc'] = mask_qc(registered_binary, reg.affine, config, template_binary)
        # A rigid warp should preserve extracted volume within interpolation
        # error. Large loss or growth suggests clipping or a failed registration.
        ratio = record['registered_mask_qc']['brain_ml'] / record['native_mask_qc']['brain_ml']
        if not .95 <= ratio <= 1.05:
            raise ValueError('Rigid-warp brain volume changed by more than five percent')
        record['rigid_mask_volume_ratio'] = ratio
        normalized, statistics = normalize_brain(reg.get_fdata(dtype=np.float32), registered_binary)
        destination = output_dir / 'normalized.nii.gz'
        nib.save(nib.Nifti1Image(normalized, reg.affine), destination)
        if not np.array_equal(nib.load(destination).get_fdata(dtype=np.float32), normalized):
            raise ValueError('Normalized output readback differs')
        record.update(status='passed_automated_qc_requires_blinded_review', normalization=statistics,
                      processed_volume_path=str(destination.resolve()), processed_volume_sha256=file_hash(destination),
                      registered_mask_path=str(warped_mask.resolve()), registered_mask_sha256=file_hash(warped_mask),
                      affine=reg.affine.tolist(), shape=list(reg.shape),
                      anatomical_segmentation_available=False)
    except (ValueError, KeyError, OSError, subprocess.SubprocessError) as exc:
        record['failure_reason'] = str(exc)
    record['seconds'] = time.monotonic() - started
    (output_dir / 'preprocessing.json').write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')
    return record
