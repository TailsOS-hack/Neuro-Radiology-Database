#!/usr/bin/env python3
"""Verify Grad-CAM on the installed specialist checkpoints, without GUI/network.

Uses each checkpoint's architecture, labels, and image size; writes overlays and
an auditable JSON summary. A missing checkpoint or unusable explanation fails
verification. This checks explanation mechanics, not clinical localization.

Run analytical edge-case tests separately with scripts/test_grad_cam_contract.py.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from PIL import Image

from src.experiment_pipeline import build_transforms, iter_images, load_checkpoint_model, sha256_file
from src.grad_cam import GradCAM, get_target_layer, overlay_cam_on_image


def check_specialist(name, checkpoint_path, image_root, args, output_dir):
    model, checkpoint = load_checkpoint_model(checkpoint_path, args.device)
    transform = build_transforms(False, checkpoint.get("image_size", 224))
    layer = get_target_layer(model)
    samples = []
    for label in checkpoint["class_names"]:
        paths = iter_images(image_root / label)[:args.samples_per_class]
        if len(paths) != args.samples_per_class:
            raise ValueError(f"Need {args.samples_per_class} images in {image_root / label}")
        samples.extend(paths)
    rows = []
    for path in samples:
        with Image.open(path) as source:
            source = source.convert("RGB")
        tensor = transform(source).unsqueeze(0).to(args.device)
        with torch.inference_mode():
            before = model(tensor).clone()
            predicted = int(before.argmax(1).item())
            cam = GradCAM(model, layer).generate(tensor, predicted)
            after = model(tensor)
        if layer._forward_hooks or layer._backward_hooks:
            raise AssertionError(f"{name}: hooks leaked after {path.name}")
        if not torch.equal(before, after):
            raise AssertionError(f"{name}: classification changed during Grad-CAM")
        if any(parameter.grad is not None for parameter in model.parameters()):
            raise AssertionError(f"{name}: Grad-CAM populated parameter gradients")
        if cam is None or cam.ndim != 2 or not np.isfinite(cam).all():
            raise AssertionError(f"{name}: no finite spatial explanation for {path.name}")
        if not (0 <= float(cam.min()) <= float(cam.max()) <= 1):
            raise AssertionError(f"{name}: heatmap out of range for {path.name}")
        overlay = overlay_cam_on_image(source, cam)
        if overlay.size != source.size:
            raise AssertionError(f"{name}: overlay dimensions changed for {path.name}")
        filename = f"{name}_{path.parent.name}_{path.stem}.png"
        overlay.save(output_dir / filename)
        row = {"image": str(path.relative_to(ROOT)), "image_sha256": sha256_file(path),
               "predicted": checkpoint["class_names"][predicted], "cam_shape": list(cam.shape),
               "cam_min": float(cam.min()), "cam_max": float(cam.max()), "overlay": filename}
        rows.append(row)
        print(f"[verify-grad-cam] {name}: {path.name} -> {row['predicted']}, finite CAM {cam.shape}", flush=True)
    for _ in range(args.repeat):
        with torch.no_grad():
            repeated = GradCAM(model, layer).generate(tensor, predicted)
        if repeated is None or not np.allclose(repeated, cam, atol=1e-6):
            raise AssertionError(f"{name}: repeat explanation changed")
    if layer._forward_hooks or layer._backward_hooks:
        raise AssertionError(f"{name}: hooks leaked across repeated calls")
    return {"checkpoint": str(checkpoint_path), "checkpoint_sha256": sha256_file(checkpoint_path),
            "architecture": checkpoint["arch"], "samples": rows, "repeat_calls": args.repeat,
            "classification_unchanged": True, "parameter_gradients_untouched": True, "hooks_clean": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--tumor-checkpoint", type=Path, default=ROOT / "models/brain_tumor_classifier.pt")
    parser.add_argument("--dementia-checkpoint", type=Path, default=ROOT / "models/alzheimers_classifier.pt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--samples-per-class", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.samples_per_class < 1 or args.repeat < 1 or args.threads < 1:
        parser.error("sample count, repeat count, and threads must be positive")
    torch.set_num_threads(args.threads)
    output_dir = args.output_dir or Path(tempfile.mkdtemp(prefix="grad_cam_verify_"))
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {"device": args.device, "source_sha256": sha256_file(ROOT / "src/grad_cam.py"),
              "validation_scope": "mechanics and prediction preservation, not clinical localization",
              "specialists": {}}
    for name, checkpoint, images in [
        ("tumor", args.tumor_checkpoint, ROOT / "data/brain_tumor/Testing"),
        ("dementia", args.dementia_checkpoint, ROOT / "data/alzheimers"),
    ]:
        report["specialists"][name] = check_specialist(name, checkpoint, images, args, output_dir)
    (output_dir / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"[verify-grad-cam] PASS: both trained specialists; results in {output_dir}", flush=True)


if __name__ == "__main__":
    main()
