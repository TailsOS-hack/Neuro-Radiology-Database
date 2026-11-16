"""Patient-level training of the slice-bag dementia model on labelled 3D scans.

One scan is one example and one person is one unit:

* scans are prepared exactly as at inference time (``volume_analysis.
  prepare_clinical_volume``), so a checkpoint trained here reads DICOM series
  through the same code path and its per-slice Grad-CAM stacks into the same
  3D map;
* people, never slices, are assigned to folds, so no person contributes to both
  fitting and testing; every person is scored once, by a model that never saw
  them (stratified K-fold cross-validation), and the K fold models are kept as
  the ensemble;
* each fold selects its epoch on a validation subset of its own training
  people. Test-fold outcomes never influence fitting or selection.

Cohorts::

    python -m src.train_dementia_volume --oasis-root data/external/oasis1 --output training_logs/experiments/dementia_volume/oasis1
    python -m src.train_dementia_volume --manifest cohort.csv --cohort-name my_cohort --output <dir>

A manifest is a CSV with columns ``subject_id``, ``scan`` (NIfTI file, or DICOM
file / folder / zip; relative paths are resolved against the manifest), ``label``
(one of the two class names, or 0/1) and optionally ``stratum``.

The output folder receives ``fold<k>.pt`` (loadable by ``DementiaVolumeAnalyzer``),
``cross_validation.json`` (aggregate results only) and ``private/`` (per-person
predictions; never commit it). Nothing here replaces an existing checkpoint.
Internal cross-validation on one cohort says nothing about other scanners or
sites; the summary and every checkpoint's notice say so.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import logging
import math
import random
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
from PIL import Image, ImageFilter

try:
    from src import volume_analysis as va
    from src.mri_volume import MRIVolume, PathLike, VolumeError, sha256_file
except ModuleNotFoundError:  # executed from inside src/
    import volume_analysis as va
    from mri_volume import MRIVolume, PathLike, VolumeError, sha256_file

logger = logging.getLogger(__name__)

#: SHA-256 of torchvision's official MobileNetV3-Large ImageNet1K V1 weights (the file Round 2 pinned).
IMAGENET_V1_SHA256 = "8738ca797c879b547d18bbd15da5736ff2557b2036a9af72225393ca61759a04"


@dataclass(frozen=True)
class Case:
    """One labelled scan. ``subject_id`` is the unit that is split, sampled and counted."""

    case_id: str
    subject_id: str
    label: int  # index into TrainConfig.class_names
    stratum: str  # only used to balance folds (e.g. CDR level)
    load: Callable[[], MRIVolume] = field(repr=False, compare=False)
    source: str = ""  # where the scan comes from; keeps a shared slice cache from mixing up equally named scans


@dataclass(frozen=True)
class TrainConfig:
    """Fitting schedule. Defaults follow the reviewed Round 2 clinical protocol where one exists."""

    folds: int = 5
    seed: int = 42
    max_epochs: int = 100
    min_epochs: int = 20
    patience: int = 15
    min_delta: float = 1e-4
    warmup_epochs: int = 5  # head only, then the last feature blocks as well
    unfreeze_from: int = 14  # first feature block that is fine-tuned after the warm-up
    slices_per_bag: int = 9  # band slices drawn per person per step
    batch_patients: int = 4
    head_lr: float = 1e-4
    backbone_lr: float = 1e-5
    weight_decay: float = 1e-3
    validation_fraction: float = 0.2
    class_names: tuple = ("CN", "AD")
    threshold: float = 0.5
    bootstrap_repetitions: int = 2000

    def validate(self) -> None:
        if self.folds < 2 or len(self.class_names) != 2 or not 0 < self.threshold < 1:
            raise ValueError("Need at least two folds, two class names and a threshold inside (0, 1)")
        if not 0 < self.validation_fraction < 0.5 or self.slices_per_bag < 1 or self.batch_patients < 1:
            raise ValueError("Invalid validation fraction, bag size or batch size")
        if not 1 <= self.min_epochs <= self.max_epochs or self.patience < 1:
            raise ValueError("Invalid epoch schedule")


# --------------------------------------------------------------------------- splitting by person


def _subject_strata(cases: Sequence[Case]) -> dict:
    subjects: dict = {}
    for case in cases:
        if case.label not in (0, 1):
            raise ValueError("Labels must be 0 or 1")
        known = subjects.setdefault(case.subject_id, (case.label, case.stratum))
        if known != (case.label, case.stratum):
            raise ValueError("A person has scans with different labels or strata; one person needs one label")
    return subjects


def assign_folds(cases: Sequence[Case], folds: int, seed: int) -> dict:
    """Person -> fold. Stratified by (label, stratum), deterministic, sizes differ by at most one."""

    subjects = _subject_strata(cases)
    by_stratum = defaultdict(list)
    for subject, key in subjects.items():
        by_stratum[key].append(subject)
    assignment, position = {}, 0
    for key in sorted(by_stratum):
        members = sorted(by_stratum[key])
        random.Random(f"{seed}|fold|{key}").shuffle(members)
        for subject in members:
            assignment[subject] = position % folds
            position += 1
    if len(set(assignment.values())) < folds:
        raise ValueError(f"{len(subjects)} people cannot fill {folds} folds")
    for fold in range(folds):
        if {subjects[subject][0] for subject, assigned in assignment.items() if assigned == fold} != {0, 1}:
            raise ValueError(f"Fold {fold} would not contain both classes; use fewer folds")
    return assignment


def hold_out_validation(subjects: dict, fraction: float, seed: int) -> set:
    """People held out of fitting for epoch selection, stratified like the folds."""

    by_stratum = defaultdict(list)
    for subject, key in subjects.items():
        by_stratum[key].append(subject)
    held = set()
    for key in sorted(by_stratum):
        members = sorted(by_stratum[key])
        random.Random(f"{seed}|validation|{key}").shuffle(members)
        if len(members) >= 2:
            held.update(members[: max(1, round(len(members) * fraction))])
    for label in (0, 1):
        # A class made only of one-person strata would otherwise never be validated.
        people = sorted(subject for subject, key in subjects.items() if key[0] == label)
        if len(people) >= 2 and not held.intersection(people):
            random.Random(f"{seed}|validation|class{label}").shuffle(people)
            held.add(people[0])
    for name, group in (("validation", held), ("fitting", set(subjects) - held)):
        if {subjects[subject][0] for subject in group} != {0, 1}:
            raise ValueError(f"The {name} people of a fold would not contain both classes; the cohort is too small")
    return held


# --------------------------------------------------------------------------- prepared slices


class SliceCache:
    """Prepared band slices per scan, on disk, so each scan is resampled once."""

    def __init__(self, directory: PathLike) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def _path(self, case: Case) -> Path:
        key = f"{va.CLINICAL_RECIPE}|{case.case_id}|{case.source}"
        return self.directory / (hashlib.sha256(key.encode()).hexdigest()[:24] + ".npz")

    def images(self, case: Case) -> np.ndarray:
        """uint8 (slices, 224, 224): every band slice of the scan, inferior to superior."""

        path = self._path(case)
        if path.is_file():
            with np.load(path, allow_pickle=False) as stored:
                if str(stored["recipe"]) == va.CLINICAL_RECIPE:
                    return stored["images"]
        prepared = va.prepare_clinical_volume(case.load())
        images = prepared.render_slices(prepared.band_indices)
        temporary = path.with_suffix(".tmp.npz")
        np.savez_compressed(temporary, images=images, recipe=np.array(va.CLINICAL_RECIPE))
        temporary.replace(path)
        return images


def prepare_cases(cases: Sequence[Case], cache: SliceCache) -> tuple:
    """Prepare every scan once. Returns (usable cases, failure reasons); failures are counted, not hidden."""

    usable, failures = [], []
    for number, case in enumerate(cases, start=1):
        try:
            cache.images(case)
            usable.append(case)
        except (VolumeError, OSError) as exc:
            failures.append((case.case_id, f"{type(exc).__name__}: {exc}"))
        if number % 25 == 0 or number == len(cases):
            logger.info("Prepared %d of %d scans (%d failed)", number, len(cases), len(failures))
    return usable, failures


def augment_bag(images: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Patient-consistent augmentation of a bag of uint8 slices -> float32 in [0, 1].

    Same family and ranges as the reviewed Round 2 clinical protocol: one small
    rotation (expanded, so no anatomy is cropped), brightness and contrast
    shared by the whole bag, optional blur, and independent pixel noise.
    """

    angle = float(rng.uniform(-5, 5))
    brightness, contrast = float(rng.uniform(0.92, 1.08)), float(rng.uniform(0.9, 1.1))
    sigma = float(rng.uniform(0, 0.01)) if rng.random() < 0.5 else 0.0
    blur = float(rng.uniform(0, 0.5)) if rng.random() < 0.25 else 0.0
    result = []
    for plane in images:
        image = Image.fromarray(plane).rotate(angle, resample=Image.Resampling.BILINEAR, expand=True, fillcolor=0)
        image = image.resize(plane.shape[::-1], resample=Image.Resampling.BILINEAR)
        if blur:
            image = image.filter(ImageFilter.GaussianBlur(blur))
        values = np.asarray(image, dtype=np.float32) / 255.0
        values = ((values - values.mean()) * contrast + values.mean()) * brightness
        if sigma:
            values = values + rng.normal(0, sigma, values.shape).astype(np.float32)
        result.append(np.clip(values, 0, 1))
    return np.stack(result)


def _to_tensor(values: np.ndarray):
    """float32 (N, H, W) in [0, 1] -> ImageNet-normalised (N, 3, H, W)."""

    import torch

    stacked = torch.from_numpy(np.ascontiguousarray(values, dtype=np.float32))[:, None].expand(-1, 3, -1, -1)
    mean = torch.tensor(va.IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(va.IMAGENET_STD).view(1, 3, 1, 1)
    return (stacked - mean) / std


# --------------------------------------------------------------------------- metrics


def exact_interval(successes: int, total: int, level: float = 0.95) -> tuple:
    """Clopper-Pearson interval for a proportion; (None, None) when there are no cases."""

    from scipy.stats import beta

    if total <= 0:
        return (None, None)
    alpha = 1 - level
    low = 0.0 if successes == 0 else float(beta.ppf(alpha / 2, successes, total - successes + 1))
    high = 1.0 if successes == total else float(beta.ppf(1 - alpha / 2, successes + 1, total - successes))
    return (low, high)


def auroc(labels: np.ndarray, scores: np.ndarray) -> Optional[float]:
    """Area under the ROC curve (ties count half); None unless both classes are present."""

    labels, scores = np.asarray(labels, dtype=int), np.asarray(scores, dtype=float)
    positive, negative = scores[labels == 1], scores[labels == 0]
    if not len(positive) or not len(negative):
        return None
    wins = (positive[:, None] > negative[None, :]).sum() + 0.5 * (positive[:, None] == negative[None, :]).sum()
    return float(wins / (len(positive) * len(negative)))


def binary_metrics(labels: Sequence[int], positive_probability: Sequence[float], threshold: float = 0.5) -> dict:
    """Per-person metrics with exact intervals for sensitivity and specificity."""

    y, p = np.asarray(labels, dtype=int), np.asarray(positive_probability, dtype=float)
    if len(y) != len(p) or not len(y) or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Invalid labels or probabilities")
    predicted = (p >= threshold).astype(int)
    true_positive, false_negative = int(((predicted == 1) & (y == 1)).sum()), int(((predicted == 0) & (y == 1)).sum())
    true_negative, false_positive = int(((predicted == 0) & (y == 0)).sum()), int(((predicted == 1) & (y == 0)).sum())
    sensitivity = true_positive / (true_positive + false_negative) if true_positive + false_negative else None
    specificity = true_negative / (true_negative + false_positive) if true_negative + false_positive else None
    clipped = np.clip(np.where(y == 1, p, 1 - p), 1e-12, 1)
    return {
        "n": int(len(y)),
        "positives": int((y == 1).sum()),
        "negatives": int((y == 0).sum()),
        "threshold": threshold,
        "confusion": {"true_positive": true_positive, "false_negative": false_negative,
                      "true_negative": true_negative, "false_positive": false_positive},
        "sensitivity": sensitivity,
        "sensitivity_exact_95ci": list(exact_interval(true_positive, true_positive + false_negative)),
        "specificity": specificity,
        "specificity_exact_95ci": list(exact_interval(true_negative, true_negative + false_positive)),
        "balanced_accuracy": None if sensitivity is None or specificity is None else (sensitivity + specificity) / 2,
        "accuracy": float((predicted == y).mean()),
        "auroc": auroc(y, p),
        "brier": float(np.mean((p - y) ** 2)),
        "nll": float(-np.log(clipped).mean()),
    }


def bootstrap_intervals(labels, positive_probability, groups, threshold: float, seed: int, repetitions: int) -> dict:
    """Percentile 95% intervals from resampling people (all scans of a person move together)."""

    y, p = np.asarray(labels, dtype=int), np.asarray(positive_probability, dtype=float)
    members = defaultdict(list)
    for index, group in enumerate(groups):
        members[group].append(index)
    keys = sorted(members)
    rng = np.random.default_rng(seed)
    balanced, area = [], []
    for _ in range(repetitions):
        chosen = np.concatenate([members[keys[i]] for i in rng.integers(0, len(keys), len(keys))])
        result = binary_metrics(y[chosen], p[chosen], threshold)
        if result["balanced_accuracy"] is not None:
            balanced.append(result["balanced_accuracy"])
            area.append(result["auroc"])
    if not balanced:
        return {"status": "unavailable"}
    return {
        "method": f"percentile 95% interval, {repetitions} resamples of people, seed {seed}; conditional on this cohort",
        "balanced_accuracy": [float(v) for v in np.quantile(balanced, [0.025, 0.975])],
        "auroc": [float(v) for v in np.quantile(area, [0.025, 0.975])],
    }


# --------------------------------------------------------------------------- model and fitting


def initialise_backbone(model, init: str) -> dict:
    """Load ImageNet features into ``model`` and describe exactly what was loaded."""

    import torch

    if init == "random":
        return {"kind": "random", "note": "for software tests only; not a way to train a usable model"}
    if init == "imagenet":
        from torchvision import models

        try:
            source = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)
        except Exception as exc:
            raise RuntimeError(
                "Could not obtain torchvision's MobileNetV3-Large ImageNet weights (no network?). Download "
                "mobilenet_v3_large-8738ca79.pth once and pass it with --init <file>."
            ) from exc
        model.features.load_state_dict(source.features.state_dict(), strict=True)
        return {"kind": "imagenet1k_v1", "source": "torchvision MobileNet_V3_Large_Weights.IMAGENET1K_V1"}
    path = Path(init)
    if not path.is_file():
        raise FileNotFoundError(f"Initial weights not found: {init}")
    digest = sha256_file(path)
    state = torch.load(path, map_location="cpu", weights_only=True)
    features = {key[len("features."):]: value for key, value in state.items() if key.startswith("features.")}
    model.features.load_state_dict(features, strict=True)
    return {"kind": "file", "file": path.name, "sha256": digest,
            "is_official_imagenet1k_v1": digest == IMAGENET_V1_SHA256}


def _set_trainable(model, epoch: int, config: TrainConfig) -> None:
    """Head only during the warm-up, then the last feature blocks too; BatchNorm always frozen."""

    from torch import nn

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.head.parameters():
        parameter.requires_grad_(True)
    if epoch > config.warmup_epochs:
        for block in list(model.features.children())[config.unfreeze_from:]:
            for parameter in block.parameters():
                parameter.requires_grad_(True)
    model.train()
    # A few people per step cannot estimate stable BatchNorm statistics.
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad_(False)


def predict_cases(model, cases: Sequence[Case], cache: SliceCache, device, batch_size: int = 64) -> np.ndarray:
    """(cases, 2) probabilities: softmax of the mean logit over every band slice of each scan."""

    import torch

    model.eval()
    probabilities = []
    with torch.no_grad():
        for case in cases:
            tensor = va.images_to_tensor(cache.images(case))
            logits = torch.cat([model(tensor[start:start + batch_size].to(device)).float().cpu()
                                for start in range(0, len(tensor), batch_size)])
            probabilities.append(torch.softmax(logits.mean(dim=0), dim=0).numpy())
    return np.stack(probabilities)


def frozen_activations(model, cases: Sequence[Case], cache: SliceCache, device, split: int, batch_size: int = 64) -> list:
    """Output of feature blocks ``[:split]`` for every band slice of each scan, computed once.

    Those blocks are never trained (and BatchNorm never updates), so for
    un-augmented slices their output cannot change during fitting. Validation
    then only runs the trained tail each epoch, which is exact, not approximate.
    """

    import torch

    lower = model.features[:split]
    model.eval()
    activations = []
    with torch.no_grad():
        for case in cases:
            tensor = va.images_to_tensor(cache.images(case))
            activations.append(torch.cat([lower(tensor[start:start + batch_size].to(device)).float().cpu()
                                          for start in range(0, len(tensor), batch_size)]))
    return activations


def predict_from_activations(model, activations: Sequence, device, split: int, batch_size: int = 256) -> np.ndarray:
    """Same result as :func:`predict_cases`, starting from :func:`frozen_activations`."""

    import torch

    model.eval()
    probabilities = []
    with torch.no_grad():
        for stored in activations:
            logits = torch.cat([
                model.head(model.dropout(model.avgpool(model.features[split:](stored[start:start + batch_size].to(device))).flatten(1))).float().cpu()
                for start in range(0, len(stored), batch_size)
            ])
            probabilities.append(torch.softmax(logits.mean(dim=0), dim=0).numpy())
    return np.stack(probabilities)


def train_fold(
    fit_cases: Sequence[Case],
    validation_cases: Sequence[Case],
    cache: SliceCache,
    config: TrainConfig,
    seed: int,
    device,
    model_factory: Callable,
    init: str,
) -> tuple:
    """Fit one model; return (model at its best validation epoch, history, initialisation record)."""

    import torch
    from torch import nn

    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    model = model_factory()
    initialisation = initialise_backbone(model, init)
    model.to(device)

    counts = Counter(case.label for case in fit_cases)
    # Inverse-frequency weights: equal to 1 for a balanced cohort, so neither class dominates otherwise.
    weights = torch.tensor([len(fit_cases) / (2 * counts[label]) for label in (0, 1)], dtype=torch.float32, device=device)
    blocks = list(model.features.children())
    optimizer = torch.optim.AdamW(
        [
            {"params": list(model.head.parameters()), "lr": config.head_lr},
            {"params": [p for block in blocks[config.unfreeze_from:] for p in block.parameters()], "lr": config.backbone_lr},
        ],
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda epoch: 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(epoch, config.max_epochs) / config.max_epochs))
    )
    validation_labels = [case.label for case in validation_cases]
    validation_activations = frozen_activations(model, validation_cases, cache, device, config.unfreeze_from)

    best_score, best_state, best_epoch, stale, history = math.inf, None, 0, 0, []
    for epoch in range(config.max_epochs + 1):
        fit_loss = None
        if epoch:
            _set_trainable(model, epoch, config)
            order = list(range(len(fit_cases)))
            random.Random(f"{seed}|order|{epoch}").shuffle(order)
            total = 0.0
            for start in range(0, len(order), config.batch_patients):
                batch = [fit_cases[index] for index in order[start:start + config.batch_patients]]
                bags = []
                for case in batch:
                    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(f"{seed}|{epoch}|{case.case_id}".encode()).digest()[:8], "little"))
                    images = cache.images(case)
                    chosen = np.sort(rng.choice(len(images), size=min(config.slices_per_bag, len(images)), replace=False))
                    bag = augment_bag(images[chosen], rng)
                    if len(bag) < config.slices_per_bag:  # short band: repeat slices so bags stack
                        bag = bag[np.resize(np.arange(len(bag)), config.slices_per_bag)]
                    bags.append(_to_tensor(bag))
                inputs = torch.stack(bags).to(device)  # (people, slices, 3, H, W)
                labels = torch.tensor([case.label for case in batch], device=device)
                optimizer.zero_grad()
                logits = model(inputs.flatten(0, 1)).reshape(len(batch), config.slices_per_bag, 2).mean(dim=1)
                loss = nn.functional.cross_entropy(logits, labels, weight=weights)
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite training loss")
                loss.backward()
                nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 5.0)
                optimizer.step()
                total += float(loss.detach()) * len(batch)
            scheduler.step()
            fit_loss = total / len(fit_cases)

        probabilities = predict_from_activations(model, validation_activations, device, config.unfreeze_from)
        validation = binary_metrics(validation_labels, probabilities[:, 1], config.threshold)
        improved = validation["nll"] < best_score - config.min_delta
        if improved:
            best_score, best_epoch, stale = validation["nll"], epoch, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
        history.append({"epoch": epoch, "fit_loss": fit_loss, "validation_nll": validation["nll"],
                        "validation_balanced_accuracy": validation["balanced_accuracy"], "selected": improved})
        logger.info("  epoch %3d  fit %s  validation NLL %.4f  balanced accuracy %s%s", epoch,
                    "   -   " if fit_loss is None else f"{fit_loss:.4f}", validation["nll"],
                    "-" if validation["balanced_accuracy"] is None else f"{validation['balanced_accuracy']:.3f}",
                    "  *" if improved else "")
        if epoch >= config.min_epochs and stale >= config.patience:
            break
    if best_state is None:
        raise RuntimeError("Validation loss was never finite; nothing to select")
    model.load_state_dict(best_state)
    model.eval()
    return model, {"selected_epoch": best_epoch, "selected_validation_nll": best_score,
                   "completed_epochs": history[-1]["epoch"],
                   "stopped": "validation plateau" if history[-1]["epoch"] < config.max_epochs else "maximum epochs",
                   "epochs": history}, initialisation


# --------------------------------------------------------------------------- cross-validation


#: Stored in every checkpoint next to its notice; the analysis adds it to a result only when that result has a map.
MAP_NOTICE = "The attention map shows what drove this model's score; it is not a localisation of pathology."


def _notice(cohort: str, people: int, folds: int, pooled: dict, class_names: Sequence[str]) -> str:
    confusion = pooled["confusion"]
    return (
        f"Research output only, not a diagnosis. This {class_names[1]}/{class_names[0]} ensemble was trained on "
        f"{people} people from one cohort ({cohort}). In {folds}-fold cross-validation by person on that cohort, "
        f"sensitivity was {confusion['true_positive']}/{pooled['positives']} and specificity "
        f"{confusion['true_negative']}/{pooled['negatives']}. It has not been tested on scans from any other site "
        "or scanner, where performance is usually lower."
    )


def cross_validate(
    cases: Sequence[Case],
    output_dir: PathLike,
    *,
    cohort: str,
    config: TrainConfig = TrainConfig(),
    cache_dir: Optional[PathLike] = None,
    init: str = "imagenet",
    device=None,
    model_factory: Optional[Callable] = None,
) -> dict:
    """Stratified K-fold cross-validation by person. Returns the aggregate summary it also writes."""

    import torch

    config.validate()
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"{output_dir} is not empty; choose a new folder (a run, finished or interrupted, is never overwritten)")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "private").mkdir()
    (output_dir / "private" / ".gitignore").write_text("*\n")  # per-person rows stay out of version control
    device = device if device is not None else va.default_device()
    custom_model = model_factory is not None
    model_factory = model_factory or va.build_clinical_slice_model
    cache = SliceCache(cache_dir if cache_dir is not None else output_dir / "private" / "prepared")

    started = time.time()
    logger.info("Preparing %d scans", len(cases))
    usable, failures = prepare_cases(cases, cache)
    if failures:
        with (output_dir / "private" / "preparation_failures.csv").open("w", newline="") as handle:
            csv.writer(handle).writerows([("case_id", "reason"), *failures])
    subjects = _subject_strata(usable)
    assignment = assign_folds(usable, config.folds, config.seed)
    logger.info("%d people in %d folds on %s", len(subjects), config.folds, device)

    out_of_fold, per_fold, fitted = {}, [], []
    for fold in range(config.folds):
        fold_seed = config.seed + fold
        test_cases = [case for case in usable if assignment[case.subject_id] == fold]
        development = {subject: key for subject, key in subjects.items() if assignment[subject] != fold}
        held = hold_out_validation(development, config.validation_fraction, fold_seed)
        fit_cases = [case for case in usable if case.subject_id in development and case.subject_id not in held]
        validation_cases = [case for case in usable if case.subject_id in held]
        overlap = {c.subject_id for c in test_cases} & ({c.subject_id for c in fit_cases} | {c.subject_id for c in validation_cases})
        if overlap or {c.subject_id for c in fit_cases} & {c.subject_id for c in validation_cases}:
            raise RuntimeError("A person appears on both sides of a split")  # never expected; fail closed
        logger.info("Fold %d: fitting on %d people, selecting on %d, testing on %d", fold,
                    len({c.subject_id for c in fit_cases}), len(held), len({c.subject_id for c in test_cases}))
        model, history, initialisation = train_fold(fit_cases, validation_cases, cache, config, fold_seed, device, model_factory, init)
        probabilities = predict_cases(model, test_cases, cache, device)
        for case, probability in zip(test_cases, probabilities):
            if case.case_id in out_of_fold:
                raise RuntimeError("A scan was tested twice")
            out_of_fold[case.case_id] = (case, fold, float(probability[1]))
        metrics = binary_metrics([case.label for case in test_cases], probabilities[:, 1], config.threshold)
        per_fold.append({"fold": fold, "fit_people": len({c.subject_id for c in fit_cases}), "validation_people": len(held),
                         "test_people": len({c.subject_id for c in test_cases}), "selected_epoch": history["selected_epoch"],
                         "completed_epochs": history["completed_epochs"], "stopped": history["stopped"], "test": metrics})
        (output_dir / f"fold{fold}_history.json").write_text(json.dumps(history, indent=2, allow_nan=False) + "\n")
        fitted.append((fold, fold_seed, {key: value.detach().cpu() for key, value in model.state_dict().items()}, history, initialisation))

    if set(out_of_fold) != {case.case_id for case in usable}:
        raise RuntimeError("Not every scan received exactly one out-of-fold prediction")
    rows = [out_of_fold[case.case_id] for case in usable]
    labels = [case.label for case, _, _ in rows]
    scores = [score for _, _, score in rows]
    pooled = binary_metrics(labels, scores, config.threshold)
    pooled["bootstrap"] = bootstrap_intervals(labels, scores, [case.subject_id for case, _, _ in rows],
                                              config.threshold, config.seed, config.bootstrap_repetitions)
    by_stratum = {}
    for stratum in sorted({case.stratum for case, _, _ in rows}):
        members = [(case.label, score) for case, _, score in rows if case.stratum == stratum]
        correct = sum((score >= config.threshold) == bool(label) for label, score in members)
        by_stratum[stratum] = {"n": len(members), "class": config.class_names[members[0][0]], "correct": correct,
                               "correct_exact_95ci": list(exact_interval(correct, len(members)))}
    with (output_dir / "private" / "out_of_fold_predictions.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["case_id", "subject_id", "fold", "label", "stratum", f"probability_{config.class_names[1]}"])
        writer.writerows([(case.case_id, case.subject_id, fold, case.label, case.stratum, f"{score:.6f}") for case, fold, score in rows])

    notice = _notice(cohort, len(subjects), config.folds, pooled, config.class_names)
    checkpoints = []
    for fold, fold_seed, state, history, initialisation in fitted:
        path = output_dir / f"fold{fold}.pt"
        torch.save(
            {
                "model": state,
                "architecture": "custom" if custom_model else va.CLINICAL_ARCHITECTURE,
                "seed": fold_seed,
                "epoch": history["selected_epoch"],
                "class_names": list(config.class_names),
                "slice_policy": "band",
                "threshold": config.threshold,
                "recipe": va.CLINICAL_RECIPE,
                "initialization": initialisation,
                "notice": notice,
                "map_notice": MAP_NOTICE,
                "training": {"cohort": cohort, "fold": fold, "folds": config.folds, "people": len(subjects),
                             "selection": "lowest validation NLL on people held out of fitting; test fold never used",
                             "config": asdict(config)},
            },
            path,
        )
        checkpoints.append({"file": path.name, "sha256": sha256_file(path), "fold": fold,
                            "selected_epoch": history["selected_epoch"], "initialization": initialisation})

    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "cohort": cohort,
        "unit": "splits and resampling are by person, and every person is tested once, by a model fitted without "
                "them; metric rows are scans (one per person unless a person has repeat scans)",
        "people": len(subjects),
        "scans": len(usable),
        "scans_that_could_not_be_prepared": len(failures),
        "class_names": list(config.class_names),
        "people_by_class": {config.class_names[label]: sum(1 for key in subjects.values() if key[0] == label) for label in (0, 1)},
        "people_by_stratum": dict(sorted(Counter(key[1] for key in subjects.values()).items())),
        "recipe": va.CLINICAL_RECIPE,
        "slice_policy": "band",
        "config": asdict(config),
        "device": str(device),
        "out_of_fold": pooled,
        "by_stratum": by_stratum,
        "per_fold": per_fold,
        "checkpoints": checkpoints,
        "notice": notice,
        "map_notice": MAP_NOTICE,
        "limits": [
            "Internal cross-validation on one cohort: one site, one scanner, one protocol. Not external validation.",
            "The fold models are selected on validation people of the same cohort; the threshold is fixed, not tuned.",
            "Per-stratum rows are descriptive; small strata have very wide intervals.",
            "A research artifact. Not validated for clinical use.",
        ],
        "seconds": round(time.time() - started, 1),
    }
    (output_dir / "cross_validation.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return summary


# --------------------------------------------------------------------------- cohorts


def cases_from_oasis1(root: PathLike, *, min_age: float = 60.0) -> tuple:
    """(cases, aggregate description) for OASIS-1: CDR 0 versus CDR above 0 among adults aged ``min_age``+."""

    try:
        from src import oasis1
    except ModuleNotFoundError:
        import oasis1

    sessions = oasis1.load_index(root)
    summary = oasis1.summarize(sessions, min_age=min_age)
    cases = [
        Case(case_id=session.session_id, subject_id=session.subject_id, label=int(session.cdr > 0),
             stratum=f"CDR {session.cdr_level}", load=lambda session=session: oasis1.load_session_volume(session),
             source=str(session.image_path))
        for session in oasis1.eligible_sessions(sessions, min_age=min_age)
    ]
    return cases, summary


def cases_from_manifest(path: PathLike, class_names: Sequence[str]) -> list:
    """Cases from a CSV manifest (see the module docstring for its columns)."""

    path = Path(path)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    cases = []
    for number, row in enumerate(rows, start=2):
        try:
            subject, scan, label = row["subject_id"].strip(), row["scan"].strip(), row["label"].strip()
        except (KeyError, AttributeError) as exc:
            raise ValueError(f"{path} needs the columns subject_id, scan and label") from exc
        if label in class_names:
            index = list(class_names).index(label)
        elif label in ("0", "1"):
            index = int(label)
        else:
            raise ValueError(f"{path} line {number}: label must be one of {list(class_names)} or 0/1")
        location = Path(scan) if Path(scan).is_absolute() else path.parent / scan
        if not subject or not location.exists():
            raise ValueError(f"{path} line {number}: missing subject_id or scan not found")
        cases.append(Case(case_id=f"{subject}|{scan}", subject_id=subject, label=index,
                          stratum=(row.get("stratum") or class_names[index]).strip(),
                          load=lambda location=location: va.load_any_volume(location),
                          source=str(location.resolve())))
    if not cases:
        raise ValueError(f"{path} lists no scans")
    return cases


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Cross-validated, patient-level training of the dementia slice-bag model.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--oasis-root", help="Extracted OASIS-1 tree (see src/oasis1.py)")
    source.add_argument("--manifest", help="CSV with subject_id, scan, label[, stratum]")
    parser.add_argument("--cohort-name", help="Name recorded with the results (required with --manifest)")
    parser.add_argument("--output", required=True, help="New, empty folder for checkpoints and the summary")
    parser.add_argument("--cache-dir", help="Where prepared slices are kept (default: <output>/private/prepared). Pass a "
                        "folder outside the output to reuse them after an interrupted run; only for unchanged scans")
    parser.add_argument("--init", default="imagenet", help="'imagenet' (torchvision download), a weights file, or 'random' (tests)")
    parser.add_argument("--folds", type=int, default=TrainConfig.folds)
    parser.add_argument("--max-epochs", type=int, default=TrainConfig.max_epochs)
    parser.add_argument("--min-age", type=float, default=60.0, help="OASIS-1 only: youngest age included")
    parser.add_argument("--device", help="Override the device (default: CUDA, then DirectML, then CPU)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    import torch

    config = TrainConfig(folds=args.folds, max_epochs=args.max_epochs, min_epochs=min(TrainConfig.min_epochs, args.max_epochs))
    try:
        if args.oasis_root:
            cases, description = cases_from_oasis1(args.oasis_root, min_age=args.min_age)
            cohort = f"OASIS-1, adults aged {args.min_age:g}+, CDR 0 versus CDR above 0, SUBJ_111 images"
            logger.info("OASIS-1: %s", json.dumps(description))
            if description.get("matches_documented_dementia_counts") is False:
                logger.warning("Dementia counts by CDR differ from the provider's fact sheet; check the tree is complete")
        else:
            if not args.cohort_name:
                parser.error("--cohort-name is required with --manifest")
            cases, cohort = cases_from_manifest(args.manifest, config.class_names), args.cohort_name
        summary = cross_validate(cases, args.output, cohort=cohort, config=config, cache_dir=args.cache_dir, init=args.init,
                                 device=torch.device(args.device) if args.device else None)
    except (VolumeError, ValueError, FileNotFoundError, RuntimeError) as exc:
        logger.error("%s", exc)
        return 2
    pooled = summary["out_of_fold"]
    print(json.dumps({"people": summary["people"], "sensitivity": pooled["sensitivity"], "specificity": pooled["specificity"],
                      "balanced_accuracy": pooled["balanced_accuracy"], "auroc": pooled["auroc"],
                      "summary": str(Path(args.output) / "cross_validation.json")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
