#!/usr/bin/env python3
"""Reproduce the current pipeline on the fixed, reused 100-image benchmark.

This command never trains, selects checkpoints, or changes the label key. The
cohort overlaps training/validation data; its score is NOT independent accuracy.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.experiment_pipeline import DEMENTIA_CLASSES, TASK_DEFAULT_MODEL_PATH


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_cases(key: Path, images: Path, expected_count: int = 100) -> list[dict]:
    with key.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != expected_count:
        raise ValueError(f"Expected {expected_count} cases, got {len(rows)}")
    names = [row.get("FileName", "") for row in rows]
    if len(set(names)) != len(names):
        raise ValueError("Duplicate filenames in the ground-truth key")
    for row in rows:
        name = row.get("FileName", "")
        if not name or Path(name).name != name or "\\" in name:
            raise ValueError(f"Invalid cohort filename: {name!r}")
        if row.get("TrueCategory") not in DEMENTIA_CLASSES:
            raise ValueError(f"Unknown ground-truth class for {name}")
        if not (images / name).is_file():
            raise FileNotFoundError(images / name)
    return rows


def summarize(rows: list[dict], threshold: float) -> dict:
    if not rows:
        raise ValueError("Cannot score an empty cohort")
    correct = sum(row["Correct"] for row in rows)
    cam_ok = sum(row["GradCAMValid"] for row in rows)
    unchanged = sum(row["PredictionUnchangedAfterCAM"] for row in rows)
    classes = sorted({row["TrueCategory"] for row in rows} |
                     {row["ModelPrediction"] for row in rows})
    confusion = [[sum(row["TrueCategory"] == truth and row["ModelPrediction"] == pred
                      for row in rows) for pred in classes] for truth in classes]
    return {
        "total": len(rows), "correct": correct, "accuracy": correct / len(rows),
        "threshold_strictly_greater_than": threshold,
        "accuracy_target_met": correct / len(rows) > threshold,
        "gradcam_valid": cam_ok, "gradcam_failed_or_flat": len(rows) - cam_ok,
        "predictions_unchanged_after_cam": unchanged,
        "technical_checks_passed": cam_ok == unchanged == len(rows),
        "class_order": classes, "confusion_matrix": confusion,
        "per_class": {
            label: {"total": sum(row["TrueCategory"] == label for row in rows),
                    "correct": sum(row["TrueCategory"] == label and row["Correct"] for row in rows)}
            for label in sorted({row["TrueCategory"] for row in rows})
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--key", type=Path, default=ROOT / "data/evaluation/ground_truth/radiologist_test_key.csv")
    parser.add_argument("--images", type=Path, default=ROOT / "data/evaluation/images")
    parser.add_argument("--output", type=Path, required=True, help="New directory; existing outputs are never replaced")
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.95)
    for task in ("binary", "tumor", "dementia"):
        parser.add_argument(f"--{task}-checkpoint", type=Path, default=TASK_DEFAULT_MODEL_PATH[task])
    args = parser.parse_args()
    if not 0 <= args.threshold < 1 or args.threads < 1:
        parser.error("Threshold must be in [0, 1); threads must be positive")
    cases = load_cases(args.key, args.images)
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")

    import numpy as np
    import torch
    import torchvision
    from PIL import Image, __version__ as pillow_version
    from src.grad_cam import GradCAM, get_target_layer
    from src.hierarchical_inference import HierarchicalInferencePipeline

    torch.set_num_threads(args.threads)
    pipeline = HierarchicalInferencePipeline(
        binary_checkpoint=args.binary_checkpoint, tumor_checkpoint=args.tumor_checkpoint,
        dementia_checkpoint=args.dementia_checkpoint, device=torch.device(args.device))
    args.output.mkdir(parents=True)
    results, maps = [], {}
    for case in cases:
        path = args.images / case["FileName"]
        pred = pipeline.predict(path)
        if pred.domain == "dementia":
            model, transform, meta = pipeline.dementia_model, pipeline.dementia_transform, pipeline.dementia_meta
            label = pred.subtype
        else:
            model, transform, meta = pipeline.tumor_model, pipeline.tumor_transform, pipeline.tumor_meta
            # Keep a wrong router decision visible, even for normal scans.
            label = pred.label
        with Image.open(path) as image:
            tensor = transform(image.convert("RGB")).unsqueeze(0).to(pipeline.device)
        target = meta["class_names"].index(pred.subtype)
        with torch.no_grad():
            before = model(tensor).detach().clone()
        cam = GradCAM(model, get_target_layer(model)).generate(tensor, target)
        with torch.no_grad():
            after = model(tensor)
        valid = (cam is not None and cam.ndim == 2 and np.isfinite(cam).all()
                 and float(np.ptp(cam)) > 0 and float(cam.min()) >= 0 and float(cam.max()) <= 1)
        if valid:
            maps[Path(case["FileName"]).stem] = cam
        results.append({**case, "ModelPrediction": label, "Domain": pred.domain,
                        "Correct": label == case["TrueCategory"],
                        "Confidence": pred.confidence, "RouterConfidence": pred.router_confidence,
                        "SpecialistConfidence": pred.specialist_confidence,
                        "GradCAMValid": bool(valid), "GradCAMTarget": pred.subtype,
                        "PredictionUnchangedAfterCAM": bool(torch.equal(before, after)),
                        "ImageSHA256": file_sha256(path)})
        if len(results) % 20 == 0:
            print(f"Evaluated {len(results)}/{len(cases)}", flush=True)

    metrics = summarize(results, args.threshold)
    metrics.update({
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_type": "reused_internal_benchmark_not_independent_test",
        "limitations": ["This cohort overlaps historical training and validation images.",
                        "No retraining or checkpoint selection was performed on this cohort.",
                        "Finite nonflat Grad-CAM checks test software behavior, not anatomical validity."],
        "key_sha256": file_sha256(args.key),
        "checkpoints": {task: {
            "path": str(getattr(args, f"{task}_checkpoint").relative_to(ROOT))
                    if getattr(args, f"{task}_checkpoint").is_relative_to(ROOT)
                    else str(getattr(args, f"{task}_checkpoint")),
            "sha256": file_sha256(getattr(args, f"{task}_checkpoint")),
            "arch": meta["arch"], "image_size": meta["image_size"],
            "class_names": meta["class_names"],
        } for task, meta in [("binary", pipeline.binary_meta), ("tumor", pipeline.tumor_meta),
                             ("dementia", pipeline.dementia_meta)]},
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "torchvision": torchvision.__version__, "numpy": np.__version__,
                        "pillow": pillow_version, "device": args.device, "threads": args.threads},
        "source_sha256": {p: file_sha256(ROOT / p) for p in [
            "scripts/evaluate_radiologist_cohort.py", "src/hierarchical_inference.py",
            "src/experiment_pipeline.py", "src/grad_cam.py"]},
    })
    with (args.output / "predictions.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    np.savez_compressed(args.output / "gradcam_maps.npz", **maps)
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))
    return 0 if metrics["accuracy_target_met"] and metrics["technical_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
