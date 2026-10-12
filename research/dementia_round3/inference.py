"""Gated full-volume research inference, with explicit abstention and provenance.

Bundle schema: version='round3_model_bundle_v1', class_order=['CN','AD'],
protocol_hash, preprocessing_hash, model_version, checkpoints=[{path,sha256,
seed,epoch}], aggregation='equal_probability_mean', research_only=true.
All three fixed final neural seeds are mandatory. The fixed anatomical baseline
may instead use one checkpoint, kind='anatomical_baseline', seed FINAL_SEEDS[0].
Optional calibration is {path,sha256,key}; key indexes a fitted calibration in
that JSON document (omit key when the whole document is the fitted calibration).
External inference requires the exact new_model_bundle/preprocessing/calibration
references already frozen by the root-owned authorization verifier.
"""
import json
import math
from pathlib import Path
import numpy as np
import torch
from .models import MODEL_VERSION, ModelConfig, build_model
from .preprocessing import PREPROCESSING_VERSION, file_hash, object_hash, preflight
from .training import FINAL_SEEDS, _authorize, _verified_json, load_prepared_patient

INFERENCE_VERSION = 'round3-full-volume-inference-v1'
LEGACY_PREPROCESSOR_SHA256 = 'bc3e1c7012cc54e9453af38d554073b3666cbc115fbb2a2072ed10a822f49475'
LEGACY_CHECKPOINTS = {
    20261011: 'dded15b8c2201e8f3d71fab1f0af547363a539f78b79db49f389337eef2654d9',
    20261012: '9435ae63d0ccb38b9474d2aee85f063bcb48f207d88c6f83d6fc495f8b52beac',
    20261013: '87d25d2e878caeb4aef33b99c210ebfdf303fc20058974b1a3c390d3aa5a39fe'}


def _external_artifacts(auth):
    if auth['phase'] != 'external_evaluation':
        return None
    ref = auth['selection_freeze']['manifest']
    return _verified_json(ref['path'], ref['sha256'])['artifacts']


def _same_ref(left, right):
    return (Path(left['path']).resolve() == Path(right['path']).resolve() and
            left['sha256'] == right['sha256'])


def _load_bundle(bundle_path, auth, preprocessing_hash, device):
    """Internal loader. Public entrypoint must authorize rows before invoking."""
    path = Path(bundle_path)
    if path.is_symlink() or not path.is_file():
        raise ValueError('Model bundle is missing or a symlink')
    bundle_ref = {'path': str(path), 'sha256': file_hash(path)}
    external = _external_artifacts(auth)
    if external is not None and not _same_ref(bundle_ref, external['new_model_bundle']):
        raise PermissionError('Bundle differs from frozen external selection')
    bundle = _verified_json(path, bundle_ref['sha256'])
    if (bundle.get('version') != 'round3_model_bundle_v1' or
            bundle.get('class_order') != ['CN', 'AD'] or bundle.get('model_version') != MODEL_VERSION or
            bundle.get('protocol_hash') != auth['protocol_hash'] or
            bundle.get('preprocessing_hash') != preprocessing_hash or
            bundle.get('aggregation') != 'equal_probability_mean' or bundle.get('research_only') is not True):
        raise ValueError('Unsupported or mismatched model bundle')
    refs = bundle.get('checkpoints', [])
    anatomical = bundle.get('kind') == 'anatomical_baseline'
    expected = [FINAL_SEEDS[0]] if anatomical else list(FINAL_SEEDS)
    if len(refs) != len(expected) or sorted(r.get('seed', -1) for r in refs) != expected:
        raise ValueError('Bundle must preserve the prespecified final seed identities')
    models, first_config, candidate_id = [], None, None
    for ref in sorted(refs, key=lambda r: r['seed']):
        checkpoint_path = Path(ref['path'])
        if checkpoint_path.is_symlink() or file_hash(checkpoint_path) != ref['sha256']:
            raise ValueError('Checkpoint missing or checksum changed')
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        if (checkpoint.get('model_version') != MODEL_VERSION or
                checkpoint.get('protocol_hash') != auth['protocol_hash'] or
                checkpoint.get('preprocessing_hash') != preprocessing_hash or
                checkpoint.get('research_mode') != 'real_authorized_development' or
                checkpoint.get('seed') != ref['seed'] or checkpoint.get('epoch') != ref.get('epoch') or
                checkpoint.get('selection_metric') != 'site_macro_tune_nll' or
                checkpoint.get('validation_site_id')):
            raise ValueError('Checkpoint is not the pinned final development model')
        config = ModelConfig(**checkpoint['model_config'])
        if (config.kind == 'anatomical_logistic') != anatomical:
            raise ValueError('Anatomical and volume model bundle types differ')
        current_id = checkpoint['candidate']['candidate_id']
        if first_config is not None and (config != first_config or current_id != candidate_id):
            raise ValueError('Final seeds must use the same validation-selected configuration')
        first_config, candidate_id = config, current_id
        model = build_model(config)
        model.load_state_dict(checkpoint['model'], strict=True)
        if any(not torch.isfinite(t).all() for t in model.state_dict().values()):
            raise ValueError('Nonfinite checkpoint tensors')
        models.append(model.to(device).eval())
    calibration = None
    if bundle.get('calibration'):
        ref = bundle['calibration']
        if external is not None and not _same_ref(ref, external['calibration']):
            raise PermissionError('Calibration differs from external selection freeze')
        calibration = _verified_json(ref['path'], ref['sha256'])
        if ref.get('key'):
            calibration = calibration[ref['key']]
        if (calibration.get('score_field') != 'new_score' or
                (calibration.get('provenance') or {}).get('protocol_sha256') != auth['protocol_hash']):
            raise ValueError('Calibration provenance is not bound to the new model protocol')
    if external is not None and calibration is None:
        raise ValueError('External inference requires frozen development calibration')
    return bundle, models, first_config, calibration, bundle_ref


def _abstention(reason, preprocessing_hash, qc_status='abstained'):
    return {'score': None, 'raw_score': None, 'prediction': None, 'threshold': None,
            'qc_status': qc_status, 'abstention_reason': reason, 'model_version': MODEL_VERSION,
            'preprocessing_version': PREPROCESSING_VERSION, 'preprocessing_hash': preprocessing_hash,
            'inference_version': INFERENCE_VERSION, 'research_only': True,
            'clinical_deployment_permitted': False}


def _predict_new_loaded(row, loaded, prep_hash, device):
    bundle, models, config, calibration, ref = loaded
    x = load_prepared_patient(row, prep_hash, config).unsqueeze(0).to(device)
    with torch.inference_mode():
        probabilities = [m(x).softmax(1).cpu().numpy()[0] for m in models]
    probabilities = np.asarray(probabilities, float)
    if (probabilities.shape != (len(models), 2) or not np.isfinite(probabilities).all() or
            np.any(probabilities < 0) or not np.allclose(probabilities.sum(1), 1, atol=1e-6)):
        raise ValueError('Invalid patient probabilities')
    raw = float(probabilities[:, 1].mean())
    result = _abstention(None, prep_hash, qc_status='ok')
    result.update(raw_score=raw, score=raw, score_type='uncalibrated_development_score',
                  individual_scores=probabilities[:, 1].tolist(), seeds=[r['seed'] for r in sorted(bundle['checkpoints'], key=lambda r: r['seed'])],
                  bundle_sha256=ref['sha256'], aggregation=bundle['aggregation'])
    if calibration is not None:
        from .calibration import apply_calibration
        result.update(score=apply_calibration([raw], calibration)[0], score_type='development_calibrated_score')
        threshold = calibration.get('threshold')
        if (calibration.get('threshold_status') != 'available' or isinstance(threshold, bool) or
                not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or not 0 <= threshold <= 1):
            result.update(score=None, qc_status='abstained', abstention_reason='Frozen calibration threshold is unavailable')
        else:
            result.update(threshold=threshold, prediction=int(result['score'] >= threshold))
    return result


def predict_volume(row, bundle_path, authorization_path, preprocessing_config, *, device='cpu'):
    """Predict one authorized, complete T1 with pinned preprocessing and blind QC.

    No checkpoint, prepared image or patient metadata is read before the root
    authorization/row check. Raw-only inputs abstain pending preprocessing and
    blinded review; use preprocessing.preprocess_volume through the gated CLI.
    This entrypoint cannot accept a JPEG, user-selected slice, or silent fallback.
    """
    auth = _authorize(authorization_path, [row], 'infer')
    prep_hash = preprocessing_config.version_hash
    if auth['protocol'].get('preprocessing_hash') != prep_hash:
        raise PermissionError('Inference preprocessing is not the frozen protocol configuration')
    check = preflight(preprocessing_config)
    if check['status'] != 'ready':
        result = _abstention('Required pinned preprocessing dependencies are unavailable', prep_hash)
        result['dependency_errors'] = check['errors']
        return result
    try:
        bundle, models, config, calibration, ref = _load_bundle(bundle_path, auth, prep_hash, device)
        return _predict_new_loaded(row, (bundle, models, config, calibration, ref), prep_hash, device)
    except (ValueError, KeyError, OSError, RuntimeError, TypeError) as exc:
        return _abstention(type(exc).__name__ + ': ' + str(exc), prep_hash, qc_status='failed')


def _legacy_bag(row):
    """Read nine audited outputs of the unchanged full-raw-T1 legacy pipeline.

    The immutable row must pin legacy_preprocessing_record_path/sha256 and
    legacy_review_path/sha256. The record contains source_volume_sha256,
    preprocessing_version, preprocessor_sha256, slices ordered by slice_fraction
    with path/sha256/pixel_sha256. Review binds the record SHA and both blindings.
    Producing these outputs is a separate preprocessing/QC step, never inference.
    """
    import hashlib
    from PIL import Image
    record = _verified_json(row['legacy_preprocessing_record_path'], row['legacy_preprocessing_record_sha256'])
    review = _verified_json(row['legacy_review_path'], row['legacy_review_sha256'])
    if (record.get('source_volume_sha256') != row['volume_sha256'] or
            record.get('preprocessing_version') != 'exploratory_t1_axial_v1' or
            record.get('preprocessor_sha256') != LEGACY_PREPROCESSOR_SHA256 or
            review.get('status') != 'passed' or review.get('diagnosis_blinded') is not True or
            review.get('prediction_blinded') is not True or
            review.get('legacy_preprocessing_record_sha256') != row['legacy_preprocessing_record_sha256']):
        raise ValueError('Legacy preprocessing or blinded review provenance differs')
    slices = record.get('slices', [])
    if ([r.get('slice_fraction') for r in slices] != [.45, .50, .55, .60, .65, .70, .75, .80, .85] or
            len({r.get('slice_index') for r in slices}) != 9):
        raise ValueError('Legacy comparator needs all nine fixed, distinct axial slices')
    tensors = []
    for item in slices:
        path = Path(item['path'])
        if path.is_symlink() or file_hash(path) != item['sha256']:
            raise ValueError('Legacy slice checksum differs')
        with Image.open(path) as im:
            pixels = np.asarray(im.convert('RGB'))
        if (pixels.shape != (224, 224, 3) or pixels.max() == pixels.min() or
                not np.array_equal(pixels[:, :, 0], pixels[:, :, 1]) or
                not np.array_equal(pixels[:, :, 0], pixels[:, :, 2]) or
                hashlib.sha256(pixels.tobytes()).hexdigest() != item['pixel_sha256']):
            raise ValueError('Legacy prepared pixels differ')
        tensor = torch.from_numpy((pixels.astype(np.float32) / 255).copy()).permute(2, 0, 1)
        tensors.append((tensor - torch.tensor([.485, .456, .406])[:, None, None]) /
                       torch.tensor([.229, .224, .225])[:, None, None])
    return torch.stack(tensors)


def _legacy_aux_forward(model, auxiliary, x):
    # Exact frozen round1 train_candidate.forward(..., 'aux') computation.
    representation = model.avgpool(model.features(x)).flatten(1)
    representation = model.classifier[:-1](representation)
    return auxiliary(representation)


def _load_legacy_checkpoints(refs, device):
    from torchvision.models import mobilenet_v3_large
    from torch import nn
    if len(refs) != 3 or {r.get('seed') for r in refs} != set(LEGACY_CHECKPOINTS):
        raise ValueError('Legacy comparator must preserve c2/c4/c6 identities')
    models = []
    for ref in sorted(refs, key=lambda r: r['seed']):
        path = Path(ref['path'])
        if path.is_symlink() or ref['sha256'] != LEGACY_CHECKPOINTS[ref['seed']] or file_hash(path) != ref['sha256']:
            raise ValueError('Legacy checkpoint does not match immutable prior research bytes')
        # These exact SHA-pinned task-produced archives use the original loader
        # format. No arbitrary or unverified pickle is accepted here.
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        if (checkpoint.get('arch') != 'mobilenet_v3_large' or checkpoint.get('aux_classes') != ['CN', 'AD'] or
                checkpoint.get('config', {}).get('mode') != 'multitask' or
                checkpoint.get('smoke_only') or checkpoint.get('config', {}).get('smoke')):
            raise ValueError('Legacy binary auxiliary endpoint unavailable')
        model = mobilenet_v3_large(weights=None)
        dimensions = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(dimensions, 4)
        auxiliary = nn.Linear(dimensions, 2)
        model.load_state_dict(checkpoint['model_state'], strict=True)
        auxiliary.load_state_dict(checkpoint['aux_state'], strict=True)
        if any(not torch.isfinite(v).all() for network in (model, auxiliary) for v in network.state_dict().values()):
            raise ValueError('Nonfinite legacy model tensor')
        models.append((model.to(device).eval(), auxiliary.to(device).eval()))
    return models


def _load_legacy_bundle(bundle_path, auth, device, recalibrated):
    path = Path(bundle_path)
    ref = {'path': str(path), 'sha256': file_hash(path)}
    external = _external_artifacts(auth)
    artifact = 'recalibrated_old_bundle' if recalibrated else 'old_model_bundle'
    if external is not None and not _same_ref(ref, external[artifact]):
        raise PermissionError('Legacy bundle differs from frozen external comparator')
    bundle = _verified_json(path, ref['sha256'])
    if (bundle.get('version') != 'round3_legacy_binary_bundle_v1' or
            bundle.get('model_id') != 'round1_clinical_ensemble' or
            bundle.get('protocol_hash') != auth['protocol_hash'] or
            bundle.get('class_order') != ['CN', 'AD'] or bundle.get('research_only') is not True or
            bundle.get('aggregation') != 'mean_slice_probability_then_equal_model_mean'):
        raise ValueError('Unsupported legacy binary bundle')
    preprocessor = bundle['preprocessor']
    if (preprocessor['sha256'] != LEGACY_PREPROCESSOR_SHA256 or
            Path(preprocessor['path']).is_symlink() or file_hash(preprocessor['path']) != LEGACY_PREPROCESSOR_SHA256):
        raise ValueError('Historical preprocessing code differs')
    if bool(bundle.get('calibration')) != bool(recalibrated):
        raise ValueError('Original and recalibrated comparator roles cannot be exchanged')
    models = _load_legacy_checkpoints(bundle['checkpoints'], device)
    return bundle, models, ref, external


def _predict_legacy_loaded(row, loaded, auth, device, recalibrated):
    bundle, models, ref, external = loaded
    x = _legacy_bag(row).to(device)
    with torch.inference_mode():
        # Historical order is mean per-slice float64 probabilities, then
        # equal model mean. Do not replace it with mean-logit pooling.
        probabilities = np.asarray([_legacy_aux_forward(m, a, x).softmax(1).cpu().numpy().astype(np.float64).mean(0)
                                    for m, a in models])
    if probabilities.shape != (3, 2) or not np.isfinite(probabilities).all() or not np.allclose(probabilities.sum(1), 1, atol=1e-6):
        raise ValueError('Invalid legacy probabilities')
    raw = float(probabilities[:, 1].mean())
    score, threshold = raw, .5
    if not recalibrated and raw == .5:
        raise ValueError('Historical comparator probability tie needs explicit review; no changed argmax semantics')
    if recalibrated:
        cal_ref = bundle['calibration']
        if external is not None and not _same_ref(cal_ref, external['calibration']):
            raise ValueError('Legacy calibration differs from frozen selection')
        calibration = _verified_json(cal_ref['path'], cal_ref['sha256'])
        if cal_ref.get('key'): calibration = calibration[cal_ref['key']]
        if (calibration.get('score_field') != 'old_score' or
                (calibration.get('provenance') or {}).get('protocol_sha256') != auth['protocol_hash']):
            raise ValueError('Wrong old-score calibration provenance')
        from .calibration import apply_calibration
        score = apply_calibration([raw], calibration)[0]
        threshold = calibration.get('threshold')
        if (calibration.get('threshold_status') != 'available' or isinstance(threshold, bool) or
                not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or not 0 <= threshold <= 1):
            result = _abstention('Frozen old-score calibration threshold unavailable', LEGACY_PREPROCESSOR_SHA256)
            result['raw_score'] = raw
            return result
    result = _abstention(None, LEGACY_PREPROCESSOR_SHA256, 'ok')
    result.update(score=score, raw_score=raw, threshold=threshold, prediction=int(score >= threshold),
        model_version='round1-multitask-clinical-auxiliary-ensemble',
        preprocessing_version='exploratory_t1_axial_v1', bundle_sha256=ref['sha256'],
        individual_scores=probabilities[:, 1].tolist(),
        score_type='development_recalibrated_old_score' if recalibrated else 'unchanged_old_score',
        aggregation=bundle['aggregation'], patient_unit='same complete source T1, historical nine-slice preprocessing')
    return result


def predict_legacy_clinical(row, bundle_path, authorization_path, *, recalibrated=False, device='cpu'):
    """Frozen old binary comparator, retaining its historical image preparation.

    Bundle: version round3_legacy_binary_bundle_v1, model_id
    round1_clinical_ensemble, protocol_hash, class_order CN/AD, research_only true,
    aggregation mean_slice_probability_then_equal_model_mean, preprocessor
    {path,sha256}, three checkpoint refs {path,sha256,seed}. A recalibrated bundle
    additionally pins calibration {path,sha256,key}, fitted on old_score only.
    Original comparator uses its fixed 0.5 threshold; no four-stage relabeling.
    """
    auth = _authorize(authorization_path, [row], 'infer')
    try:
        loaded = _load_legacy_bundle(bundle_path, auth, device, recalibrated)
        return _predict_legacy_loaded(row, loaded, auth, device, recalibrated)
    except (ValueError, KeyError, OSError, RuntimeError, TypeError) as exc:
        return _abstention(type(exc).__name__ + ': ' + str(exc), LEGACY_PREPROCESSOR_SHA256, 'failed')


class InferenceSession:
    """Bounded context for one declared cohort batch, with no global cache.

    Usage::

        with InferenceSession(rows, authorization, config,
                              new_bundle_path=new, old_bundle_path=old) as batch:
            outputs = [batch.predict(row) for row in rows]

    The full authorization gate runs once before any model/image reads. Every
    call checks the exact frozen row digest and its raw/prepared image hashes.
    Each patient may be visited once. Successful context exit requires complete
    membership, rechecks checkpoint/bundle/source-code pins and unchanged model
    tensors. Callers must publish outputs only after successful context exit.
    Constructor errors never create a research result. Missing new preprocessing
    dependencies produce patient abstentions while preserving the old comparator.
    """
    def __init__(self, rows, authorization_path, preprocessing_config, *, new_bundle_path=None,
                 old_bundle_path=None, recalibrated_old_bundle_path=None, device='cpu'):
        self.auth = _authorize(authorization_path, rows, 'infer')
        if not rows or not any((new_bundle_path, old_bundle_path, recalibrated_old_bundle_path)):
            raise ValueError('A bounded nonempty cohort and at least one model bundle are required')
        self.roster = {r['patient_id']: object_hash(r) for r in rows}
        if len(self.roster) != len(rows):
            raise ValueError('Repeated patient in inference batch')
        self.seen, self.closed, self.device = set(), False, device
        self.prep_hash = preprocessing_config.version_hash
        self.loaded, self.failures, self.pins, self.states = {}, {}, {}, []
        self.code_pins = {str(p): file_hash(p) for p in Path(__file__).parent.glob('*.py')}
        if new_bundle_path:
            if self.auth['protocol'].get('preprocessing_hash') != self.prep_hash:
                raise PermissionError('Batch preprocessing differs from frozen protocol')
            check = preflight(preprocessing_config)
            if check['status'] != 'ready':
                self.failures['new'] = _abstention('Required pinned preprocessing dependencies are unavailable', self.prep_hash)
                self.failures['new']['dependency_errors'] = check['errors']
            else:
                self.loaded['new'] = _load_bundle(new_bundle_path, self.auth, self.prep_hash, device)
        for name, path, recalibrated in [('old', old_bundle_path, False), ('recalibrated_old', recalibrated_old_bundle_path, True)]:
            if path:
                self.loaded[name] = _load_legacy_bundle(path, self.auth, device, recalibrated)
        for name, loaded in self.loaded.items():
            bundle, models = loaded[:2]
            ref = loaded[4] if name == 'new' else loaded[2]
            refs = [ref, *bundle['checkpoints']]
            refs += [bundle[k] for k in ('calibration', 'preprocessor') if bundle.get(k)]
            for item in refs:
                self.pins[str(Path(item['path']).resolve())] = item['sha256']
            networks = models if name == 'new' else [network for pair in models for network in pair]
            self.states.extend((m, {k: v.detach().clone() for k, v in m.state_dict().items()}) for m in networks)

    def __enter__(self):
        return self

    def predict(self, row):
        pid = row.get('patient_id')
        if self.closed or pid in self.seen or self.roster.get(pid) != object_hash(row):
            raise PermissionError('Batch is closed or patient is repeated/outside exact authorized rows')
        self.seen.add(pid)
        result = {k: dict(v) for k, v in self.failures.items()}
        try:
            path = Path(row['volume_path'])
            if path.is_symlink() or file_hash(path) != row['volume_sha256']:
                raise ValueError('Raw source T1 changed during inference batch')
        except (ValueError, KeyError, OSError) as exc:
            for name in self.loaded:
                result[name] = _abstention(str(exc), self.prep_hash if name == 'new' else LEGACY_PREPROCESSOR_SHA256, 'failed')
            return result
        for name, loaded in self.loaded.items():
            try:
                result[name] = (_predict_new_loaded(row, loaded, self.prep_hash, self.device) if name == 'new' else
                    _predict_legacy_loaded(row, loaded, self.auth, self.device, name == 'recalibrated_old'))
            except (ValueError, KeyError, OSError, RuntimeError, TypeError) as exc:
                result[name] = _abstention(type(exc).__name__ + ': ' + str(exc),
                    self.prep_hash if name == 'new' else LEGACY_PREPROCESSOR_SHA256, 'failed')
        return result

    def __exit__(self, exc_type, exc, traceback):
        self.closed = True
        if exc_type is not None:
            return False
        if self.seen != set(self.roster):
            raise ValueError('Inference batch omitted eligible patients')
        if any(file_hash(path) != expected for path, expected in {**self.pins, **self.code_pins}.items()):
            raise ValueError('Pinned inference artifact/code changed during batch')
        for model, before in self.states:
            if any(not torch.equal(value, model.state_dict()[key]) for key, value in before.items()) or any(p.grad is not None for p in model.parameters()):
                raise ValueError('Inference mutated a model or accumulated gradients')
        self.loaded.clear()
        self.states.clear()
        return False
