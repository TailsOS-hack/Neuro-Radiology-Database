"""Volume-level dementia analysis with slice-wise Grad-CAM assembled into 3D.

Pipeline for one scan (DICOM series or NIfTI)::

    MRIVolume ──prepare──► model-domain axial slices ──model──► slice logits
                                                         │
                 volume score = softmax(mean slice logits), ensemble = mean of members
                                                         │
          Grad-CAM per slice (one backward pass per batch, *raw* maps)
                                                         │
        maps put back at their slice position ──► 3D attention volume + affine

Two properties make the stacked maps usable as a volume:

* Slice maps are **not** min-max normalised one by one. A slice the model
  barely used stays dark; the stack is scaled once, by its volume-wide maximum.
* The 3D map carries the affine of the grid it lives on, and can be resampled
  onto the scan's own voxel grid, so it overlays the scan in any NIfTI viewer.

The slice preparation reproduces ``docs/kevin_paper_20261011/preprocess_clinical_t1.py``
(recipe ``exploratory_t1_axial_v1``) operation for operation, because that is
the input the Round 2 clinical AD/CN checkpoints were trained on. Those
checkpoints are research artifacts with a documented external-validation
failure; nothing here upgrades them into a diagnostic device.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence, Union

import numpy as np
from PIL import Image

try:
    from src import volume_viewer
    from src.mri_volume import MRIVolume, PathLike, VolumeError, sha256_file, voxel_sizes
except ModuleNotFoundError:  # executed from inside src/
    import volume_viewer
    from mri_volume import MRIVolume, PathLike, VolumeError, sha256_file, voxel_sizes

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

#: Must stay identical to preprocess_clinical_t1.py (VERSION / FRACTIONS / SPACING).
CLINICAL_RECIPE = "exploratory_t1_axial_v1"
CLINICAL_FRACTIONS = np.linspace(0.45, 0.85, 9)
CLINICAL_SPACING_MM = (1.5, 1.5, 1.5)
MODEL_INPUT_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

CLINICAL_CLASSES = ("CN", "AD")
CLINICAL_THRESHOLD = 0.5
CLINICAL_ARCHITECTURE = "mobilenet_v3_large_features_imagenet_v1_dropout02_linear960x2"
DEFAULT_CLINICAL_CHECKPOINTS = tuple(
    PROJECT_ROOT / "models" / "kevin_round2_20261011" / f"clinical_seed{seed}.pt" for seed in (20261021, 20261022, 20261023)
)
#: Where a reviewed ensemble from ``train_dementia_volume`` is placed to become the default.
TRAINED_VOLUME_MODEL_DIR = PROJECT_ROOT / "models" / "dementia_volume"


def default_checkpoints() -> list:
    """The ensemble used when none is named.

    Fold checkpoints someone has deliberately copied into ``models/dementia_volume/``
    take precedence; otherwise the three Round 2 clinical seeds. Every result
    carries the notice of whichever ensemble produced it.
    """

    placed = sorted(TRAINED_VOLUME_MODEL_DIR.glob("fold*.pt")) if TRAINED_VOLUME_MODEL_DIR.is_dir() else []
    return placed or list(DEFAULT_CLINICAL_CHECKPOINTS)

#: Shown with every result of the Round 2 clinical ensemble (checkpoints that carry no notice of their own).
RESEARCH_USE_NOTICE = (
    "Research output only, not a diagnosis. The AD/CN ensemble was developed on 43 patients from one "
    "source and, on its only fresh external cohort, labelled all 19 participants AD (specificity 0/10); "
    "on a second cohort it labelled all 15 participants CN, missing the one AD case. "
    "The attention map shows what this model responds to in each slice; it is not a localisation of pathology."
)
#: How a model turns slices into one score: the nine frozen protocol slices, or every slice of the trained band.
SLICE_POLICIES = ("protocol9", "band")


class PreparationError(VolumeError):
    """The volume cannot be turned into the slices this model was trained on."""


# --------------------------------------------------------------------------- slice preparation


def _longest_run(indices: np.ndarray) -> np.ndarray:
    runs = np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1)
    return max(runs, key=len)


def _storage_semantics(data: np.ndarray) -> np.ndarray:
    """Array with the dtype family the frozen NIfTI pipeline would have interpolated.

    ``nibabel.processing.resample_to_output`` interpolates in the on-disk dtype,
    so integer scans come out rounded to integers. DICOM/NIfTI magnitude images
    are integer-valued, and reproducing that rounding is what makes these slices
    bit-identical to the ones the checkpoints were trained and evaluated on.
    """

    if np.array_equal(data, np.rint(data)) and np.abs(data).max() < 2**31 - 1:
        return data.astype(np.int32)
    return data.astype(np.float64)


@dataclass
class PreparedVolume:
    """A scan resampled to the model grid plus everything needed to map results back."""

    recipe: str
    grid: np.ndarray  # (X, Y, Z) float32, RAS+, CLINICAL_SPACING_MM
    grid_affine: np.ndarray
    crop: tuple  # x0, x1, y0, y1 on the grid
    cranial_extent: tuple  # zlo, zhi (inclusive) of the broad cranial foreground span
    window: tuple  # low, high intensities mapped to 0..255
    protocol_indices: list  # the nine axial indices the checkpoints were trained on
    notes: list = field(default_factory=list)

    @property
    def side(self) -> int:
        x0, x1, y0, y1 = self.crop
        return max(x1 - x0, y1 - y0)

    @property
    def band_indices(self) -> list:
        """Every axial index between the first and last protocol slice (the trained band)."""

        return list(range(self.protocol_indices[0], self.protocol_indices[-1] + 1))

    def render_slice(self, z: int) -> np.ndarray:
        """uint8 (224, 224) model input for axial index ``z``; anterior up, patient left on image left."""

        x0, x1, y0, y1 = self.crop
        low, high = self.window
        axial = self.grid[x0:x1, y0:y1, int(z)]
        normalized = np.clip((axial - low) / (high - low), 0, 1)
        pixels = np.rint(np.rot90(normalized) * 255).astype(np.uint8)
        tile = Image.fromarray(pixels)
        side = max(tile.size)
        square = Image.new("L", (side, side), 0)
        square.paste(tile, ((side - tile.width) // 2, (side - tile.height) // 2))
        resized = square.resize((MODEL_INPUT_SIZE, MODEL_INPUT_SIZE), resample=Image.Resampling.BILINEAR)
        return np.asarray(resized.convert("RGB"))[:, :, 0].copy()

    def render_slices(self, indices: Sequence[int]) -> np.ndarray:
        return np.stack([self.render_slice(z) for z in indices])

    def maps_to_grid(self, maps: np.ndarray, indices: Sequence[int]) -> np.ndarray:
        """Place per-slice 2D maps (any resolution) at their position on the model grid.

        Exact inverse of :meth:`render_slice`'s geometry: upsample to the padded
        square, drop the padding, undo the display rotation. Voxels outside the
        analysed slices or outside the crop stay zero.
        """

        import torch
        import torch.nn.functional as F

        maps = np.asarray(maps, dtype=np.float32)
        if maps.ndim != 3 or len(maps) != len(indices):
            raise ValueError("Expected one 2D map per analysed slice")
        x0, x1, y0, y1 = self.crop
        width, height, side = x1 - x0, y1 - y0, self.side
        resized = F.interpolate(
            torch.from_numpy(np.ascontiguousarray(maps))[:, None], size=(side, side), mode="bilinear", align_corners=False
        )[:, 0].numpy()
        top, left = (side - height) // 2, (side - width) // 2
        volume = np.zeros(self.grid.shape, dtype=np.float32)
        for plane, z in zip(resized, indices):
            volume[x0:x1, y0:y1, int(z)] = np.rot90(plane[top : top + height, left : left + width], k=-1)
        return volume


def prepare_clinical_volume(volume: MRIVolume) -> PreparedVolume:
    """Resample, find the cranial span and fix the crop/window: recipe ``exploratory_t1_axial_v1``.

    Mirrors ``preprocess_clinical_t1.process`` step for step. It is a heuristic
    foreground crop of a with-skull scan, not brain extraction or registration.
    """

    try:
        import nibabel as nib
        from nibabel.processing import resample_to_output
        from scipy import ndimage
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on the environment
        raise PreparationError("Volume analysis needs nibabel and scipy (pip install nibabel scipy)") from exc

    if min(volume.shape) < 16:
        raise PreparationError(
            f"A {volume.shape} array is not a 3D head volume; volume analysis needs a complete T1 series"
        )
    image = nib.Nifti1Image(_storage_semantics(volume.data), volume.affine)
    aligned = resample_to_output(image, voxel_sizes=CLINICAL_SPACING_MM, order=1, mode="constant", cval=0)
    if tuple(nib.aff2axcodes(aligned.affine)) != ("R", "A", "S"):
        raise PreparationError("Resampled grid is not RAS+")
    grid = np.asarray(aligned.dataobj, dtype=np.float32)
    if not np.isfinite(grid).all():
        raise PreparationError("Resampled volume contains nonfinite voxels")

    positive = grid[grid > 0]
    if positive.size <= 10000:
        raise PreparationError("Too little positive signal to be a head MRI volume")
    p98 = float(np.percentile(positive, 98))
    mask = grid > 0.08 * p98
    labels, count = ndimage.label(mask)
    if count == 0:
        raise PreparationError("No foreground object found")
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    mask = labels == int(sizes.argmax())  # largest connected foreground; NOT skull stripping

    axial_areas = mask.sum(axis=(0, 1))
    broad = np.flatnonzero(axial_areas >= 0.35 * axial_areas.max())
    if broad.size < 15:
        raise PreparationError("Cranial foreground spans too few axial slices; is the head fully covered?")
    span = _longest_run(broad)
    zlo, zhi = int(span[0]), int(span[-1])
    cranial = mask[:, :, zlo : zhi + 1]
    xs, ys, _ = np.where(cranial)
    margin = 4
    x0, x1 = max(0, int(xs.min()) - margin), min(grid.shape[0], int(xs.max()) + margin + 1)
    y0, y1 = max(0, int(ys.min()) - margin), min(grid.shape[1], int(ys.max()) + margin + 1)
    low, high = (float(value) for value in np.percentile(grid[:, :, zlo : zhi + 1][cranial], [1, 99]))
    if not (np.isfinite(low) and np.isfinite(high) and high > low):
        raise PreparationError("Degenerate intensity window")
    indices = [int(round(zlo + float(fraction) * (zhi - zlo))) for fraction in CLINICAL_FRACTIONS]
    if len(set(indices)) != len(indices):
        raise PreparationError("Cranial span too short to give nine distinct axial slices")

    notes = []
    extent_mm = (zhi - zlo + 1) * CLINICAL_SPACING_MM[2]
    if not 110 <= extent_mm <= 260:
        notes.append(
            f"Cranial foreground spans {extent_mm:.0f} mm superior-inferior, outside the usual head range; "
            "slice levels may not match the trained anatomy (partial coverage or a large neck/shoulder field of view?)"
        )
    return PreparedVolume(
        recipe=CLINICAL_RECIPE,
        grid=grid,
        grid_affine=np.asarray(aligned.affine, dtype=np.float64),
        crop=(x0, x1, y0, y1),
        cranial_extent=(zlo, zhi),
        window=(low, high),
        protocol_indices=indices,
        notes=notes,
    )


def resample_array(
    data: np.ndarray, from_affine: np.ndarray, to_affine: np.ndarray, to_shape: Sequence[int], *, order: int = 1
) -> np.ndarray:
    """Resample ``data`` from one voxel grid onto another using their world affines."""

    from scipy import ndimage

    transform = np.linalg.inv(np.asarray(from_affine, dtype=np.float64)) @ np.asarray(to_affine, dtype=np.float64)
    return ndimage.affine_transform(
        np.asarray(data, dtype=np.float32), transform[:3, :3], transform[:3, 3], tuple(int(n) for n in to_shape),
        order=order, mode="constant", cval=0.0,
    ).astype(np.float32)


# --------------------------------------------------------------------------- models


def build_clinical_slice_model():
    """MobileNetV3-Large features -> global average pool -> dropout -> Linear(960, 2).

    Same module layout (and state-dict keys) as ``clinical_bag.SliceModel``; no
    weights are downloaded.
    """

    from torch import nn
    from torchvision import models

    class SliceModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.features = models.mobilenet_v3_large(weights=None).features
            self.avgpool = nn.AdaptiveAvgPool2d(1)
            self.dropout = nn.Dropout(0.2)
            self.head = nn.Linear(960, 2)

        def forward(self, x):
            return self.head(self.dropout(self.avgpool(self.features(x)).flatten(1)))

    return SliceModel()


def load_clinical_checkpoint(path: PathLike, device=None):
    """Strictly load one slice-bag checkpoint (Round 2 clinical, or a ``train_dementia_volume`` fold).

    Returns (model, metadata). Only tensors and plain metadata are unpickled, so
    a checkpoint received from someone else cannot run code when it is opened.
    """

    import pickle

    import torch

    path = Path(path)
    if not path.is_file() or path.stat().st_size < 1024:
        raise FileNotFoundError(
            f"{path} is missing or is a Git LFS pointer. The bundled research checkpoints are fetched with: "
            'git lfs pull --include="models/kevin_round2_20261011/*.pt"'
        )
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except (pickle.UnpicklingError, RuntimeError, EOFError, KeyError, AttributeError) as exc:
        raise ValueError(
            f"{path} is not a loadable slice-bag checkpoint (a dict of tensors and plain metadata is required; "
            f"whole pickled model objects are not accepted here): {type(exc).__name__}"
        ) from exc
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError(f"{path} is not a clinical patient-bag checkpoint")
    if checkpoint.get("architecture") != CLINICAL_ARCHITECTURE:
        raise ValueError(f"{path} has unexpected architecture {checkpoint.get('architecture')!r}")
    model = build_clinical_slice_model()
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    if device is not None:
        model.to(device)
    class_names = tuple(checkpoint.get("class_names", CLINICAL_CLASSES))
    slice_policy = checkpoint.get("slice_policy", "protocol9")
    threshold = float(checkpoint.get("threshold", CLINICAL_THRESHOLD))
    if len(class_names) != 2 or slice_policy not in SLICE_POLICIES or not 0 < threshold < 1:
        raise ValueError(f"{path} has unsupported class names, slice policy or threshold")
    metadata = {
        "file": path.name,
        "sha256": sha256_file(path),
        "seed": checkpoint.get("seed"),
        "selected_epoch": checkpoint.get("epoch"),
        "architecture": checkpoint.get("architecture"),
        # Round 2 files predate these keys; their defaults are that round's frozen protocol.
        "class_names": list(class_names),
        "slice_policy": slice_policy,
        "threshold": threshold,
        "notice": checkpoint.get("notice") or RESEARCH_USE_NOTICE,
    }
    return model, metadata


def default_device():
    """CUDA -> DirectML (Windows) -> CPU, the project's device priority."""

    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    try:
        import torch_directml

        return torch_directml.device()
    except Exception:
        return torch.device("cpu")


def images_to_tensor(images: np.ndarray):
    """uint8 (N, 224, 224) grayscale -> float (N, 3, 224, 224), ImageNet-normalised."""

    import torch

    scaled = torch.from_numpy(np.ascontiguousarray(images)).float() / 255.0
    stacked = scaled[:, None].expand(-1, 3, -1, -1)
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return (stacked - mean) / std


# --------------------------------------------------------------------------- Grad-CAM over a slice stack


def slice_stack_grad_cam(
    model,
    target_layer,
    inputs,
    class_index: int,
    *,
    batch_size: int = 16,
    score: str = "logit",
    parity_tolerance: float = 1e-4,
):
    """Grad-CAM for every slice of a stack, returned raw (ReLU'd, unnormalised).

    Returns ``(maps, logits)`` with ``maps`` shaped (N, h, w) and ``logits``
    (N, classes). In eval mode each slice's score depends only on that slice,
    so differentiating the batch sum yields every slice's own gradient in one
    backward pass. Channel weights are the spatial mean of the gradient within
    each slice, i.e. standard Grad-CAM applied slice by slice.

    ``score="logit"`` explains the class logit (the project's existing
    definition); ``score="margin"`` explains the class logit minus the mean of
    the others, which for a two-class head is what actually sets the probability.

    The caller's module modes and parameter gradients are left untouched, and
    the gradient-enabled forward pass must reproduce the plain forward logits
    within ``parity_tolerance`` or a ``RuntimeError`` is raised.
    """

    import torch

    if score not in ("logit", "margin"):
        raise ValueError("score must be 'logit' or 'margin'")
    if inputs.ndim != 4 or not inputs.is_floating_point():
        raise ValueError("Expected a float tensor shaped (N, C, H, W)")
    captured: dict = {}
    handle = target_layer.register_forward_hook(lambda _module, _inputs, output: captured.__setitem__("value", output))
    modes = [(module, module.training) for module in model.modules()]
    device = next(model.parameters()).device
    maps, all_logits = [], []
    try:
        model.eval()
        for start in range(0, len(inputs), batch_size):
            batch = inputs[start : start + batch_size].to(device)
            with torch.no_grad():
                reference = model(batch)
            with torch.inference_mode(False), torch.enable_grad():
                x = batch.clone().detach().requires_grad_(True)
                logits = model(x)
                if logits.ndim != 2 or not 0 <= class_index < logits.shape[1]:
                    raise ValueError(f"class_index {class_index} is outside logits of shape {tuple(logits.shape)}")
                activation = captured.get("value")
                if not isinstance(activation, torch.Tensor) or activation.ndim != 4:
                    raise ValueError("Grad-CAM target layer must produce a (N, C, h, w) feature map")
                target = logits[:, class_index]
                if score == "margin" and logits.shape[1] > 1:
                    others = (logits.sum(dim=1) - target) / (logits.shape[1] - 1)
                    target = target - others
                (gradient,) = torch.autograd.grad(target.sum(), activation)
                weights = gradient.mean(dim=(2, 3), keepdim=True)
                cam = torch.relu((weights * activation.detach()).sum(dim=1))
            difference = float((reference - logits.detach()).abs().max())
            if not torch.isfinite(cam).all() or not torch.isfinite(logits).all():
                raise RuntimeError("Grad-CAM produced nonfinite values")
            if difference > parity_tolerance:
                raise RuntimeError(
                    f"Gradient-pass logits differ from the plain forward pass by {difference:.3g} "
                    f"(tolerance {parity_tolerance:g}); refusing to return an explanation of a different function"
                )
            maps.append(cam.detach().float().cpu())
            all_logits.append(reference.detach().float().cpu())
    finally:
        handle.remove()
        for module, was_training in modes:
            module.training = was_training
    return torch.cat(maps).numpy(), torch.cat(all_logits).numpy()


# --------------------------------------------------------------------------- analysis result


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / exponent.sum(axis=-1, keepdims=True)


@dataclass
class VolumeAnalysis:
    """Volume-level prediction plus the 3D attention map that explains it."""

    class_names: tuple
    probabilities: np.ndarray  # (classes,) ensemble mean of member patient probabilities
    member_probabilities: np.ndarray  # (members, classes)
    predicted_index: int
    threshold: float
    prepared: PreparedVolume
    cam_indices: list  # axial grid indices that received a Grad-CAM map
    slice_probabilities: np.ndarray  # (len(cam_indices), classes), member-mean per-slice softmax
    cam_grid: Optional[np.ndarray]  # (X, Y, Z) in [0, 1]; 1 is the strongest voxel of the whole volume
    cam_target_index: Optional[int]
    cam_score: str
    slice_policy: str
    notice: str
    members: list
    source: dict
    acquisition: dict
    warnings: list

    @property
    def predicted_label(self) -> str:
        return self.class_names[self.predicted_index]

    @property
    def confidence(self) -> float:
        return float(self.probabilities[self.predicted_index])

    def cam_on_source_grid(self, volume: MRIVolume, *, order: int = 1) -> np.ndarray:
        """The attention volume resampled onto the scan's own voxel grid (same shape as ``volume.data``)."""

        if self.cam_grid is None:
            raise ValueError("This analysis was run without Grad-CAM")
        return np.clip(resample_array(self.cam_grid, self.prepared.grid_affine, volume.affine, volume.shape, order=order), 0, 1)

    def slice_overlay(self, z: int, *, alpha: float = 0.45) -> Image.Image:
        """Model-input view of axial index ``z`` with the volume-scaled heatmap blended in."""

        import matplotlib

        base = self.prepared.render_slice(z).astype(np.float32)
        rgb = np.repeat(base[:, :, None], 3, axis=2)
        if self.cam_grid is None or z not in self.cam_indices:
            return Image.fromarray(rgb.astype(np.uint8))
        x0, x1, y0, y1 = self.prepared.crop
        plane = np.rot90(self.cam_grid[x0:x1, y0:y1, int(z)])
        tile = Image.fromarray(plane.astype(np.float32))
        side = self.prepared.side
        square = Image.new("F", (side, side), 0.0)
        square.paste(tile, ((side - tile.width) // 2, (side - tile.height) // 2))
        heat = np.asarray(square.resize((MODEL_INPUT_SIZE, MODEL_INPUT_SIZE), resample=Image.Resampling.BILINEAR))
        heat = np.clip(heat, 0, 1)
        colour = matplotlib.colormaps["jet"](heat)[:, :, :3] * 255.0
        # Blend in proportion to the map so unattended tissue keeps its grey level.
        weight = (alpha * np.clip(heat * 2.0, 0, 1))[:, :, None]
        blended = rgb * (1 - weight) + colour * weight
        return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))

    def montage(self, *, columns: int = 8, tile: int = 160) -> Image.Image:
        """Contact sheet of every analysed slice (inferior to superior) with its heatmap."""

        indices = self.cam_indices or self.prepared.protocol_indices
        rows = int(np.ceil(len(indices) / columns))
        sheet = Image.new("RGB", (columns * tile, rows * tile), (0, 0, 0))
        for position, z in enumerate(indices):
            image = self.slice_overlay(z).resize((tile, tile), resample=Image.Resampling.BILINEAR)
            sheet.paste(image, ((position % columns) * tile, (position // columns) * tile))
        return sheet

    def projections(self, *, scale: int = 2) -> Image.Image:
        """Axial, coronal and sagittal views: maximum-intensity projection of the map.

        Each panel shows the anatomy plane through the map's centre of mass with
        the map projected along the viewing direction, so the 3D spread of
        attention is visible at a glance. Superior/anterior are up.
        """

        import matplotlib

        if self.cam_grid is None:
            raise ValueError("This analysis was run without Grad-CAM")
        low, high = self.prepared.window
        anatomy = np.clip((self.prepared.grid - low) / (high - low), 0, 1)
        total = float(self.cam_grid.sum())
        if total > 0:
            centre = [int(round(float((self.cam_grid.sum(axis=tuple(a for a in range(3) if a != axis)) * np.arange(n)).sum() / total)))
                      for axis, n in enumerate(self.cam_grid.shape)]
        else:
            centre = [n // 2 for n in self.cam_grid.shape]
        panels = []
        for axis in (2, 1, 0):  # axial, coronal, sagittal
            base = np.rot90(np.take(anatomy, centre[axis], axis=axis))
            heat = np.rot90(self.cam_grid.max(axis=axis))
            rgb = np.repeat((base * 255.0)[:, :, None], 3, axis=2)
            colour = matplotlib.colormaps["jet"](np.clip(heat, 0, 1))[:, :, :3] * 255.0
            weight = (0.6 * np.clip(heat * 2.0, 0, 1))[:, :, None]
            panel = Image.fromarray(np.clip(rgb * (1 - weight) + colour * weight, 0, 255).astype(np.uint8))
            panels.append(panel.resize((panel.width * scale, panel.height * scale), resample=Image.Resampling.BILINEAR))
        sheet = Image.new("RGB", (sum(panel.width for panel in panels), max(panel.height for panel in panels)), (0, 0, 0))
        offset = 0
        for panel in panels:
            sheet.paste(panel, (offset, 0))
            offset += panel.width
        return sheet

    def summary(self) -> dict:
        """JSON-serialisable result. Contains no patient-identifying fields."""

        profile = None
        if self.cam_grid is not None:
            energy = self.cam_grid.sum(axis=(0, 1))
            total = float(energy.sum())
            profile = [round(float(energy[z]) / total, 6) if total > 0 else 0.0 for z in self.cam_indices]
        return {
            "task": f"{self.class_names[1]} versus {self.class_names[0]} (research score)",
            "class_names": list(self.class_names),
            "probabilities": [float(value) for value in self.probabilities],
            "predicted_label": self.predicted_label,
            "positive_class": self.class_names[1],
            "threshold_on_positive_class_probability": self.threshold,
            "member_probabilities": [[float(value) for value in row] for row in self.member_probabilities],
            "members": [{key: value for key, value in member.items() if key != "notice"} for member in self.members],
            "aggregation": (
                f"softmax(mean of the {'nine protocol' if self.slice_policy == 'protocol9' else 'trained-band'} "
                "slice logits) per member; equal-weight mean of member probabilities"
            ),
            "preparation": {
                "recipe": self.prepared.recipe,
                "grid_shape": list(self.prepared.grid.shape),
                "grid_spacing_mm": [float(value) for value in voxel_sizes(self.prepared.grid_affine)],
                "grid_affine": self.prepared.grid_affine.tolist(),
                "crop_x0_x1_y0_y1": list(self.prepared.crop),
                "cranial_extent_z": list(self.prepared.cranial_extent),
                "intensity_window": list(self.prepared.window),
                "protocol_slice_indices": list(self.prepared.protocol_indices),
                # The slices the score is computed from: the nine protocol slices, or the whole trained band.
                "scored_slice_indices": list(
                    self.prepared.protocol_indices if self.slice_policy == "protocol9" else self.prepared.band_indices
                ),
            },
            "grad_cam": None
            if self.cam_grid is None
            else {
                "method": "Grad-CAM at the last feature block, per slice; members scaled individually then averaged; "
                "the stacked map is scaled once, so that its strongest voxel is 1",
                "target_class": self.class_names[self.cam_target_index],
                "score": self.cam_score,
                "slice_indices": list(self.cam_indices),
                "slice_share_of_attention": profile,
                "slice_probabilities": [[float(value) for value in row] for row in self.slice_probabilities],
            },
            "source": self.source,
            "acquisition": self.acquisition,
            "warnings": list(self.warnings),
            "notice": self.notice,
        }

    def save(self, output_dir: PathLike, volume: Optional[MRIVolume] = None) -> dict:
        """Write ``result.json``, the montage, NIfTI volumes and the viewer; returns the written paths.

        ``anatomy.nii.gz`` and ``gradcam.nii.gz`` share the model grid. When the
        source ``volume`` is given, ``gradcam_native.nii.gz`` is the same map on
        the scan's own voxel grid (one value per original voxel). ``viewer.html``
        is a self-contained page for looking at the map in a browser.
        """

        import nibabel as nib

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        written = {}

        def write_nifti(name: str, data: np.ndarray, affine: np.ndarray) -> None:
            image = nib.Nifti1Image(np.asarray(data, dtype=np.float32), affine)
            image.header.set_data_dtype(np.float32)
            image.set_sform(affine, code=1)
            image.set_qform(affine, code=1)
            nib.save(image, str(output_dir / name))
            written[name] = str(output_dir / name)

        write_nifti("anatomy.nii.gz", self.prepared.grid, self.prepared.grid_affine)
        if self.cam_grid is not None:
            write_nifti("gradcam.nii.gz", self.cam_grid, self.prepared.grid_affine)
            if volume is not None:
                write_nifti("gradcam_native.nii.gz", self.cam_on_source_grid(volume), volume.affine)
            self.montage().save(output_dir / "montage.png")
            written["montage.png"] = str(output_dir / "montage.png")
            self.projections().save(output_dir / "projections.png")
            written["projections.png"] = str(output_dir / "projections.png")
        (output_dir / "result.json").write_text(json.dumps(self.summary(), indent=2, allow_nan=False) + "\n")
        written["result.json"] = str(output_dir / "result.json")
        viewer = output_dir / volume_viewer.VIEWER_FILENAME
        try:
            volume_viewer.write_viewer(self, viewer)
            written[volume_viewer.VIEWER_FILENAME] = str(viewer)
        except Exception:
            # The viewer is a convenience on top of the files above; failing to write it must not lose them.
            logger.exception("Could not write the interactive viewer")
            # A viewer left over from an earlier analysis must not sit next to this result.json.
            try:
                viewer.unlink()
            except OSError:
                pass
        return written


# --------------------------------------------------------------------------- analyzer


class DementiaVolumeAnalyzer:
    """Runs the clinical AD/CN ensemble on a whole scan and explains it in 3D."""

    def __init__(self, checkpoints: Optional[Sequence[PathLike]] = None, device=None) -> None:
        self.device = device if device is not None else default_device()
        self.models, self.members = [], []
        for path in checkpoints or default_checkpoints():
            model, metadata = load_clinical_checkpoint(path, self.device)
            self.models.append(model)
            self.members.append(metadata)
        if not self.models:
            raise ValueError("At least one checkpoint is required")
        for key in ("class_names", "slice_policy", "threshold", "notice"):
            if any(member[key] != self.members[0][key] for member in self.members):
                raise ValueError(f"Ensemble members disagree on {key}; they are not one model family")
        self.class_names = tuple(self.members[0]["class_names"])
        self.slice_policy = self.members[0]["slice_policy"]
        self.threshold = self.members[0]["threshold"]
        self.notice = self.members[0]["notice"]

    def _slice_logits(self, images: np.ndarray, batch_size: int = 32) -> np.ndarray:
        """(members, N, classes) logits for uint8 slice images."""

        import torch

        tensor = images_to_tensor(images)
        outputs = []
        with torch.inference_mode():
            for model in self.models:
                model.eval()
                parts = [model(tensor[start : start + batch_size].to(self.device)).float().cpu() for start in range(0, len(tensor), batch_size)]
                outputs.append(torch.cat(parts).numpy())
        return np.stack(outputs)

    def analyze(
        self,
        volume: MRIVolume,
        *,
        grad_cam: bool = True,
        dense: bool = True,
        cam_target: Optional[str] = None,
        cam_score: str = "logit",
        progress: Optional[Callable[[str], None]] = None,
    ) -> VolumeAnalysis:
        """Predict with the checkpoints' own slice policy; explain every slice of the trained band.

        Round 2 checkpoints predict from exactly the frozen nine-slice protocol;
        checkpoints trained with the ``band`` policy predict from every slice of
        the band, the same slices the map explains. With ``dense`` the Grad-CAM
        stack covers every 1.5 mm axial slice between the first and last
        protocol slice (a contiguous slab); otherwise only the nine protocol
        slices. ``cam_target`` defaults to the predicted class.
        """

        say = progress or (lambda _message: None)
        warnings = list(volume.warnings)
        modality = str(volume.acquisition.get("modality") or "").upper()
        if modality and modality != "MR":
            raise PreparationError(f"This is a {modality} series; the model only applies to T1-weighted MR volumes")
        contrast = (volume.acquisition.get("contrast_guess") or {}).get("label")
        if contrast == "unknown":
            warnings.append("The header does not identify this series as T1-weighted; the model was trained on T1 volumes only")
        elif contrast and contrast != "T1w":
            warnings.append(
                f"The series looks like {contrast}, not T1-weighted; the model was only ever trained on T1 volumes"
            )
        if volume.acquisition.get("contrast_agent"):
            warnings.append("Contrast agent recorded in the header; post-contrast T1 is outside the training data")

        say("Preparing slices")
        prepared = prepare_clinical_volume(volume)
        warnings.extend(prepared.notes)

        say("Scoring the scan")
        scored = prepared.protocol_indices if self.slice_policy == "protocol9" else prepared.band_indices
        slice_logits = self._slice_logits(prepared.render_slices(scored))
        member_probabilities = _softmax(slice_logits.mean(axis=1))
        probabilities = member_probabilities.mean(axis=0)
        if not np.isfinite(probabilities).all():
            raise RuntimeError("Model produced nonfinite probabilities")
        predicted = 1 if probabilities[1] >= self.threshold else 0

        cam_grid, cam_indices, slice_probabilities, target = None, [], np.zeros((0, len(self.class_names))), None
        if cam_target is not None and cam_target not in self.class_names:
            raise ValueError(f"cam_target must be one of {list(self.class_names)}, not {cam_target!r}")
        if grad_cam:
            target = predicted if cam_target is None else self.class_names.index(cam_target)
            try:
                cam_grid, cam_indices, slice_probabilities = self._grad_cam_volume(
                    prepared, target, dense, cam_score, say, warnings
                )
            except Exception as exc:  # e.g. a backend without the needed backward op
                # An explanation failure must never cost the user the prediction.
                logger.exception("Grad-CAM failed; returning the prediction without a map")
                warnings.append(f"Grad-CAM could not be computed ({type(exc).__name__}: {exc})")
                cam_grid, cam_indices, slice_probabilities = None, [], np.zeros((0, len(self.class_names)))

        warnings.append(self.notice)
        return VolumeAnalysis(
            class_names=self.class_names,
            probabilities=probabilities,
            member_probabilities=member_probabilities,
            predicted_index=predicted,
            threshold=self.threshold,
            prepared=prepared,
            cam_indices=list(cam_indices),
            slice_probabilities=slice_probabilities,
            cam_grid=cam_grid,
            cam_target_index=target,
            cam_score=cam_score,
            slice_policy=self.slice_policy,
            notice=self.notice,
            members=list(self.members),
            source=dict(volume.source),
            acquisition={key: value for key, value in volume.acquisition.items() if key not in ("series_description", "protocol_name")},
            warnings=warnings,
        )


    def _grad_cam_volume(self, prepared, target, dense, cam_score, say, warnings):
        """Stack of volume-scaled Grad-CAM maps on the model grid, averaged over members."""

        cam_indices = prepared.band_indices if dense else list(prepared.protocol_indices)
        inputs = images_to_tensor(prepared.render_slices(cam_indices))
        member_maps, member_slice_probabilities = [], []
        for number, model in enumerate(self.models, start=1):
            say(f"Grad-CAM, model {number} of {len(self.models)}")
            maps, logits = slice_stack_grad_cam(model, model.features[-1], inputs, target, score=cam_score)
            peak = float(maps.max())
            if peak <= 0:
                warnings.append(f"Member {number} produced an all-zero Grad-CAM map for class {self.class_names[target]}")
            # Members have different feature scales; give each an equal voice.
            member_maps.append(maps / peak if peak > 0 else maps)
            member_slice_probabilities.append(_softmax(logits))
        cam_grid = prepared.maps_to_grid(np.mean(member_maps, axis=0), cam_indices)
        # One scale for the whole volume, fixed after the coarse maps are enlarged (enlarging lowers
        # their peaks a little), so that the strongest voxel of the stack is exactly 1.
        peak = float(cam_grid.max())
        if peak > 0:
            cam_grid = cam_grid / np.float32(peak)
        return cam_grid, cam_indices, np.mean(member_slice_probabilities, axis=0)


def load_any_volume(path: PathLike, *, series_key: Optional[str] = None) -> MRIVolume:
    """Read a NIfTI file, or a DICOM file / folder / zip, into an :class:`MRIVolume`."""

    path = Path(path)
    if path.name.lower().endswith((".nii", ".nii.gz")):
        try:
            from src.mri_volume import load_nifti
        except ModuleNotFoundError:
            from mri_volume import load_nifti
        return load_nifti(path)
    try:
        from src.dicom_io import load_volume
    except ModuleNotFoundError:
        from dicom_io import load_volume
    return load_volume(path, series_key=series_key)


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the research AD/CN ensemble on a DICOM series or NIfTI volume and write a 3D Grad-CAM map."
    )
    parser.add_argument("scan", help="DICOM file, folder or zip, or a .nii/.nii.gz volume")
    parser.add_argument("--output", required=True, help="Directory for result.json, NIfTI maps and the montage")
    parser.add_argument("--series", help="DICOM series key (see: python -m src.dicom_io <path>)")
    parser.add_argument("--checkpoints", nargs="+",
                        help="Ensemble checkpoints; default: models/dementia_volume/fold*.pt if present, else the three Round 2 seeds")
    parser.add_argument("--cam-target", help="Class name to explain (e.g. AD); default: the predicted class")
    parser.add_argument("--cam-score", choices=("logit", "margin"), default="logit")
    parser.add_argument("--protocol-slices-only", action="store_true", help="Grad-CAM on the nine protocol slices only")
    parser.add_argument("--no-grad-cam", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    try:
        volume = load_any_volume(args.scan, series_key=args.series)
        analyzer = DementiaVolumeAnalyzer(args.checkpoints)
        result = analyzer.analyze(
            volume,
            grad_cam=not args.no_grad_cam,
            dense=not args.protocol_slices_only,
            cam_target=args.cam_target,
            cam_score=args.cam_score,
            progress=logger.info,
        )
        written = result.save(args.output, volume)
    except (VolumeError, FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
        return 2
    print(json.dumps({"predicted_label": result.predicted_label, "probabilities": dict(zip(result.class_names, map(float, result.probabilities))), "written": written, "warnings": result.warnings}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
