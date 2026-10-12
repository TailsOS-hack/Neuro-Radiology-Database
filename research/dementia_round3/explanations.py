"""Actual-gradient, uncropped research diagnostics; not anatomical ground truth.

These tensor-level functions never open data or checkpoints. Invoke them only
after the same authorization and prepared-volume checks as ordinary inference.
Native maps and full-FOV maps remain available even when the diagnostic fails.
Display normalization is separate from the raw CAM, never a selection criterion.
"""
from contextlib import contextmanager
import copy
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

EXPLANATION_VERSION = 'round3-full-volume-gradcam-v1'


@contextmanager
def _preserve_model(model):
    modes = {m: m.training for m in model.modules()}
    tensors = {k: v.detach().clone() for k, v in model.state_dict().items()}
    gradients = {p: None if p.grad is None else p.grad.detach().clone() for p in model.parameters()}
    try:
        model.eval()
        yield
        if any(not torch.equal(tensors[k], v) for k, v in model.state_dict().items()):
            raise RuntimeError('Explanation mutated model tensors')
        for p, before in gradients.items():
            if (before is None) != (p.grad is None) or (before is not None and not torch.equal(before, p.grad)):
                raise RuntimeError('Explanation mutated existing parameter gradients')
    finally:
        # Restore mode flags individually, including intentionally frozen children.
        for module, flag in modes.items():
            module.training = flag


def _normalize_for_display(raw):
    flat = raw.flatten(1)
    minimum = flat.min(1).values
    maximum = flat.max(1).values
    scale = maximum - minimum
    shape = (-1,) + (1,) * (raw.ndim - 1)
    normalized = (raw - minimum.reshape(shape)) / scale.clamp_min(1e-12).reshape(shape)
    return normalized, (scale <= 1e-12)


def gradcam(model, volume, targets=None, parity_tolerance=1e-6):
    """Standard channel-weighted Grad-CAM at the final residual feature maps.

    Multiplanar maps keep every slice and reconstruct each full-volume view;
    their arithmetic mean is an explicitly defined diagnostic, not segmentation.
    The reported forward parity uses the same eval-mode model and exact input.
    """
    if not hasattr(model, 'explanation_forward'):
        raise ValueError('This model has no spatial representation for Grad-CAM')
    if parity_tolerance != 1e-6:
        raise ValueError('Do not widen the prespecified forward parity tolerance')
    with _preserve_model(model), torch.inference_mode(False), torch.enable_grad():
        x = volume.detach().clone().requires_grad_(True)
        with torch.no_grad():
            reference = model(x).detach()
        logits, features, metadata = model.explanation_forward(x)
        target = reference.argmax(1) if targets is None else torch.as_tensor(targets, dtype=torch.long, device=x.device)
        if target.shape != (len(x),) or not torch.all((target == 0) | (target == 1)):
            raise ValueError('One binary class target is required per patient')
        gradient = torch.autograd.grad(logits.gather(1, target[:, None]).sum(), features)
        native, reconstructed = [], []
        for index, (feature, grad) in enumerate(zip(features, gradient)):
            if not torch.isfinite(grad).all() or not torch.isfinite(feature).all():
                raise ValueError('Nonfinite activation/gradient diagnostic')
            weights = grad.mean(tuple(range(2, grad.ndim)), keepdim=True)
            cam = (weights * feature).sum(1).relu()
            native.append(cam.detach().cpu().numpy())
            if metadata['kind'] == '3d':
                full = F.interpolate(cam[:, None], size=metadata['shape'], mode='trilinear', align_corners=False)[:, 0]
            elif metadata['kind'] == 'multiplanar':
                spatial = metadata['shape']
                others = [d for d in range(3) if d != index]
                planes = F.interpolate(cam[:, None], size=[spatial[d] for d in others], mode='bilinear', align_corners=False)[:, 0]
                arranged = planes.reshape(len(x), spatial[index], spatial[others[0]], spatial[others[1]])
                current_axes = [index, *others]
                full = arranged.permute(0, *[1 + current_axes.index(d) for d in range(3)])
            else:
                raise ValueError('Unsupported spatial explanation representation')
            reconstructed.append(full)
        raw = torch.stack(reconstructed).mean(0)
        display, flat = _normalize_for_display(raw)
        difference = float((reference - logits).abs().max().detach())
        return {'version': EXPLANATION_VERSION, 'targets': target.detach().cpu().tolist(),
                'reference_logits': reference.cpu().tolist(), 'gradient_forward_logits': logits.detach().cpu().tolist(),
                'forward_max_abs_difference': difference, 'forward_parity_passed': difference <= parity_tolerance,
                'parity_tolerance': parity_tolerance, 'native_maps': native,
                'full_volume_view_maps': [v.detach().cpu().numpy() for v in reconstructed],
                'raw_full_volume_map': raw.detach().cpu().numpy(),
                'display_normalized_map': display.detach().cpu().numpy(),
                'flat_map': flat.cpu().tolist(), 'metadata': metadata,
                'diagnostic_only': True, 'localization_validity_established': False}


def _correlation(left, right):
    a, b = np.asarray(left).ravel(), np.asarray(right).ravel()
    if np.ptp(a) <= 1e-12 or np.ptp(b) <= 1e-12:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def randomization_diagnostics(model, volume, seed=20261030):
    """Compare original maps against copied models with randomized parameters.

    The original predicted class stays the explanation target. A flat map is
    explicitly uninformative, not assigned a reassuring correlation score.
    """
    baseline = gradcam(model, volume)
    targets = baseline['targets']
    original_prob = torch.tensor(baseline['reference_logits']).softmax(1).numpy()
    result = {'version': EXPLANATION_VERSION, 'seed': seed, 'original': baseline, 'randomized': {}}
    for offset, scope in enumerate(('classifier', 'last_stage', 'all')):
        copied = copy.deepcopy(model)
        # Model copies alone change; fork_rng restores caller CPU/CUDA RNG states.
        devices = sorted({p.device.index for p in copied.parameters() if p.is_cuda})
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed + offset)
            region = copied.classifier if scope == 'classifier' else copied.encoder.stages[-1] if scope == 'last_stage' else copied
            for module in region.modules():
                if isinstance(module, (nn.Conv2d, nn.Conv3d, nn.Linear, nn.GroupNorm)):
                    module.reset_parameters()
        diagnostic = gradcam(copied, volume, targets=targets)
        probability = torch.tensor(diagnostic['reference_logits']).softmax(1).numpy()
        result['randomized'][scope] = {'diagnostic': diagnostic,
            'cam_pearson_correlation': [_correlation(a, b) for a, b in zip(baseline['raw_full_volume_map'], diagnostic['raw_full_volume_map'])],
            'probability_total_variation': (np.abs(original_prob - probability).sum(1) / 2).tolist()}
    return result


def region_controls(model, volume, regions):
    """Fixed caller-declared masks: erase and retain-only full-FOV diagnostics.

    Masks must be established independently of model predictions. Neither a
    rectangular region nor an intensity mask is called an anatomical segment.
    """
    if volume.ndim != 5 or len(volume) < 1:
        raise ValueError('Complete patient-volume tensor required')
    with _preserve_model(model), torch.inference_mode():
        reference = model(volume).softmax(1)
        result = {'reference_probabilities': reference.cpu().tolist(), 'regions': {},
                  'diagnostic_only': True, 'causal_disease_localization_established': False}
        for name, provided in regions.items():
            mask = torch.as_tensor(provided, device=volume.device)
            if mask.dtype != torch.bool or tuple(mask.shape) != tuple(volume.shape[2:]) or not mask.any() or mask.all():
                raise ValueError('Region must be a nonempty, nonwhole boolean full-volume mask')
            region = mask[None, None]
            erased = model(torch.where(region, torch.zeros_like(volume), volume)).softmax(1)
            only = model(torch.where(region, volume, torch.zeros_like(volume))).softmax(1)
            result['regions'][str(name)] = {'voxel_fraction': float(mask.float().mean()),
                'erased_probabilities': erased.cpu().tolist(), 'only_probabilities': only.cpu().tolist(),
                'erased_probability_tv': ((erased - reference).abs().sum(1) / 2).cpu().tolist(),
                'only_probability_tv': ((only - reference).abs().sum(1) / 2).cpu().tolist()}
        return result
