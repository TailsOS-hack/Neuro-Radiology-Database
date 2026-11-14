#!/usr/bin/env python3
"""Independent checks of the DICOM -> volume -> model-input -> prediction path.

Run from the repository root::

    python scripts/verify_dicom_volume_pipeline.py --output docs/dicom_volume_verification.json \
        --openneuro-root data/external/openneuro_ds007561_v1

Sections (each is skipped, and recorded as skipped, when its inputs are absent):

A. geometry   Synthetic series in all 48 voxel orders x 2 affines must read back
              to the source volume, and must agree with dcm2niix (an independent
              implementation) when it is on PATH. Enhanced multi-frame and
              per-slice rescale are cross-checked the same way.
B. parity     On real T1 volumes (OpenNeuro ds007561, public CC0), the in-memory
              slice preparation must be pixel-identical to the frozen
              ``preprocess_clinical_t1.py`` output, both from the published NIfTI
              and after a DICOM round trip of integer voxels written as a
              sagittal series with shuffled file names.
C. replay     The Round 2 clinical checkpoints, run through the new path, must
              reproduce the per-seed and ensemble counts published in
              ``docs/kevin_paper_20261011/round2_publication/RESULTS.md``.
D. stage      Describes what the production four-class stage model outputs on raw
              (with-skull) slices of the same scans. Descriptive, not a benchmark.

Only aggregate counts are written. No participant-level value leaves this script.
This verifies software behaviour; it is not clinical validation of any model.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import itertools
import json
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

PUBLISHED_OPENNEURO = {  # RESULTS.md, "Separate patient-level AD/control experiment"
    "binary_cases": 15,
    "seed_correct": {"20261021": 14, "20261022": 10, "20261023": 15},
    "ensemble_correct": 14,
    "ensemble_predicted_ad": 0,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dcm2niix_volume(folder: Path, work: Path):
    """Canonical (data, affine) as converted by dcm2niix, or None if it produced no single volume."""

    import nibabel as nib

    target = Path(tempfile.mkdtemp(dir=work))
    subprocess.run(["dcm2niix", "-z", "n", "-b", "n", "-f", "x", "-o", str(target), str(folder)], capture_output=True, text=True)
    files = sorted(target.glob("*.nii"))
    if len(files) != 1:
        return None
    image = nib.as_closest_canonical(nib.load(str(files[0])))
    return image.get_fdata(dtype=np.float32), np.asarray(image.affine)


def check_geometry(work: Path) -> dict:
    import dicom_synth as synth

    from src import dicom_io
    from src.mri_volume import reorient_to_ras

    referee = shutil.which("dcm2niix") is not None
    volume = synth.phantom((24, 30, 18))
    record = {"cases": 0, "source_mismatches": 0, "max_affine_error_vs_source_mm": 0.0, "dcm2niix_available": referee,
              "dcm2niix_compared": 0, "dcm2niix_mismatches": 0, "max_affine_error_vs_dcm2niix_mm": 0.0}

    def compare(folder: Path, expected: np.ndarray, expected_affine: np.ndarray, exact: bool = True) -> None:
        loaded = dicom_io.load_volume(folder)
        record["cases"] += 1
        same = np.array_equal(loaded.data, expected) if exact else np.allclose(loaded.data, expected, rtol=1e-9, atol=1e-9)
        error = float(np.abs(loaded.affine - expected_affine).max())
        record["max_affine_error_vs_source_mm"] = max(record["max_affine_error_vs_source_mm"], error)
        record["source_mismatches"] += int(not same or error > 1e-4)
        if referee:
            reference = dcm2niix_volume(folder, work)
            if reference is None:
                record["dcm2niix_mismatches"] += 1
                return
            record["dcm2niix_compared"] += 1
            error = float(np.abs(reference[1] - loaded.affine).max())
            record["max_affine_error_vs_dcm2niix_mm"] = max(record["max_affine_error_vs_dcm2niix_mm"], error)
            agrees = reference[0].shape == loaded.data.shape and np.allclose(reference[0], loaded.data, rtol=1e-6, atol=1e-3)
            record["dcm2niix_mismatches"] += int(not agrees or error > 1e-3)

    aligned = np.diag([0.9, 1.1, 1.3, 1.0])
    aligned[:3, 3] = (-90.0, 110.0, -70.0)
    for name, base in (("aligned", aligned), ("oblique", synth.oblique_affine())):
        expected, expected_affine = reorient_to_ras(volume.astype(np.float32), base)
        orders = itertools.product(itertools.permutations(range(3)), itertools.product([False, True], repeat=3))
        for case, (permutation, flips) in enumerate(orders):
            data, affine = synth.reoriented(volume, base, permutation, flips)
            folder = work / f"{name}_{case}"
            synth.write_series(folder, data, affine, shuffle=case, reverse_instance_numbers=flips[0])
            compare(folder, expected, expected_affine)
            shutil.rmtree(folder)

    oblique = synth.oblique_affine()
    expected, expected_affine = reorient_to_ras(volume.astype(np.float32), oblique)
    synth.write_enhanced(work / "enhanced" / "e.dcm", volume, oblique)
    compare(work / "enhanced", expected, expected_affine)
    slopes, intercepts = np.linspace(0.5, 2.0, volume.shape[2]), np.linspace(-10.0, 30.0, volume.shape[2])
    synth.write_series(work / "rescale", volume, oblique, slopes=slopes, intercepts=intercepts)
    compare(work / "rescale", reorient_to_ras(volume.astype(np.float64) * slopes + intercepts, oblique)[0], expected_affine, exact=False)
    record["passed"] = record["source_mismatches"] == 0 and record["dcm2niix_mismatches"] == 0
    return record


def openneuro_subjects(root: Path) -> list:
    with (root / "participants.tsv").open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    return [(row["participant_id"], row["diagnosis"], root / row["participant_id"] / "ses-01" / "anat" / f"{row['participant_id']}_ses-01_T1w.nii.gz") for row in rows]


def check_parity(subjects: list, work: Path) -> dict:
    import dicom_synth as synth
    import nibabel as nib
    from PIL import Image

    from src import dicom_io
    from src import volume_analysis as va
    from src.mri_volume import load_nifti

    frozen_path = ROOT / "docs" / "kevin_paper_20261011" / "preprocess_clinical_t1.py"
    spec = importlib.util.spec_from_file_location("frozen_preprocess_clinical_t1", frozen_path)
    frozen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(frozen)

    def frozen_slices(source: Path, tag: str):
        # The frozen script requires its input to live under the data root it is given.
        data_root = work / tag / "data"
        (data_root / "out").mkdir(parents=True)
        (data_root / "in").mkdir()
        staged = data_root / "in" / "scan.nii.gz"
        try:
            staged.symlink_to(source.resolve())
        except OSError:
            shutil.copyfile(source, staged)
        subject = dict(cohort="c", subject_id="s", diagnosis="CN", split="x", age="", sex="", source=staged)
        rows, qc = frozen.process(subject, data_root / "out", data_root)
        return [np.asarray(Image.open(data_root / row["path"]))[:, :, 0] for row in rows], qc

    def matches(prepared, reference, qc) -> bool:
        pixels = all(np.array_equal(mine, theirs) for mine, theirs in zip(prepared.render_slices(prepared.protocol_indices), reference))
        return bool(pixels and prepared.protocol_indices == qc["slice_indices"] and list(prepared.crop) == qc["crop_xy"]
                    and list(prepared.window) == qc["intensity_percentiles_1_99"])

    record = {"scans": 0, "nifti_pixel_identical": 0, "dicom_volume_identical": 0, "dicom_pixel_identical": 0,
              "frozen_preprocessor_sha256": sha256(frozen_path), "recipe": frozen.VERSION}
    for index, (_, _, source) in enumerate(subjects):
        record["scans"] += 1
        # Published NIfTI (real-valued header rescale) through the in-memory path.
        reference, qc = frozen_slices(source, f"a{index}")
        record["nifti_pixel_identical"] += int(matches(va.prepare_clinical_volume(load_nifti(source)), reference, qc))
        # Integer magnitudes, as a scanner stores them, through DICOM (sagittal slices, shuffled names).
        image = nib.load(str(source))
        integers = np.rint(np.asarray(image.dataobj)).astype(np.int16)
        plain = nib.Nifti1Image(integers, image.affine)
        plain.header.set_data_dtype(np.int16)
        plain.header.set_slope_inter(1, 0)
        (work / f"b{index}").mkdir()
        nib.save(plain, str(work / f"b{index}" / "int.nii.gz"))
        reference, qc = frozen_slices(work / f"b{index}" / "int.nii.gz", f"bf{index}")
        data, affine = synth.reoriented(integers, image.affine, (1, 2, 0), (True, False, True))
        synth.write_series(work / f"dicom{index}", data, affine, shuffle=index)
        loaded = dicom_io.load_volume(work / f"dicom{index}")
        canonical = nib.as_closest_canonical(plain)
        record["dicom_volume_identical"] += int(
            np.array_equal(loaded.data, np.asarray(canonical.dataobj)) and float(np.abs(loaded.affine - canonical.affine).max()) < 1e-4
        )
        record["dicom_pixel_identical"] += int(matches(va.prepare_clinical_volume(loaded), reference, qc))
        for name in (f"a{index}", f"b{index}", f"bf{index}", f"dicom{index}"):
            shutil.rmtree(work / name, ignore_errors=True)
    record["passed"] = record["scans"] > 0 and all(
        record[key] == record["scans"] for key in ("nifti_pixel_identical", "dicom_volume_identical", "dicom_pixel_identical")
    )
    return record


def check_replay(subjects: list) -> dict:
    from src import volume_analysis as va
    from src.mri_volume import load_nifti

    analyzer = va.DementiaVolumeAnalyzer()
    truth, members, ensemble = [], [], []
    for _, diagnosis, source in subjects:
        if diagnosis not in ("Control", "AD"):
            continue  # MCI is prediction-only in the published analysis
        result = analyzer.analyze(load_nifti(source), grad_cam=False)
        truth.append(1 if diagnosis == "AD" else 0)
        members.append([float(row[1]) >= 0.5 for row in result.member_probabilities])
        ensemble.append(float(result.probabilities[1]) >= 0.5)
    truth, members, ensemble = np.array(truth), np.array(members), np.array(ensemble)
    record = {
        "binary_cases": int(len(truth)),
        "seed_correct": {str(member["seed"]): int((members[:, index] == truth).sum()) for index, member in enumerate(analyzer.members)},
        "ensemble_correct": int((ensemble == truth).sum()),
        "ensemble_predicted_ad": int(ensemble.sum()),
        "checkpoint_sha256": {member["file"]: member["sha256"] for member in analyzer.members},
        "published": PUBLISHED_OPENNEURO,
    }
    record["passed"] = all(record[key] == PUBLISHED_OPENNEURO[key] for key in PUBLISHED_OPENNEURO)
    return record


def describe_stage_model(subjects: list) -> dict:
    import torch
    from PIL import Image

    from src import volume_analysis as va
    from src.experiment_pipeline import build_model, build_transforms
    from src.mri_volume import load_nifti

    path = ROOT / "models" / "alzheimers_classifier.pt"
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(checkpoint["arch"], len(checkpoint["class_names"]), pretrained=False)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    transform = build_transforms(train=False, image_size=int(checkpoint.get("image_size", 224)))
    names = checkpoint["class_names"]
    votes = np.zeros(len(names), dtype=int)
    controls = controls_with_any_normal_vote = 0
    for _, diagnosis, source in subjects:
        prepared = va.prepare_clinical_volume(load_nifti(source))
        images = [Image.fromarray(image).convert("RGB") for image in prepared.render_slices(prepared.protocol_indices)]
        with torch.no_grad():
            predicted = model(torch.stack([transform(image) for image in images])).argmax(dim=1).numpy()
        counts = np.bincount(predicted, minlength=len(names))
        votes += counts
        if diagnosis == "Control":
            controls += 1
            controls_with_any_normal_vote += int(counts[names.index("NonDemented")] > 0)
    return {
        "input": "nine axial with-skull slices per scan (the clinical recipe's rendering), no skull stripping",
        "scans": len(subjects),
        "slices": int(votes.sum()),
        "slice_votes": dict(zip(names, (int(value) for value in votes))),
        "cognitively_normal_participants": controls,
        "cognitively_normal_participants_with_any_NonDemented_slice": controls_with_any_normal_vote,
        "checkpoint_sha256": sha256(path),
        "interpretation": "Raw scanner slices are outside the stage model's training domain (skull-stripped, atlas-aligned slices).",
    }


def versions() -> dict:
    found = {"python": sys.version.split()[0], "numpy": np.__version__}
    for name in ("pydicom", "nibabel", "scipy", "torch", "torchvision"):
        try:
            found[name] = __import__(name).__version__
        except ModuleNotFoundError:
            found[name] = None
    if shutil.which("dcm2niix"):
        banner = subprocess.run(["dcm2niix", "-v"], capture_output=True, text=True).stdout.strip().splitlines()
        found["dcm2niix"] = banner[-1] if banner else "unknown"
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, help="Write the aggregate JSON record here")
    parser.add_argument("--openneuro-root", type=Path, help="Local copy of OpenNeuro ds007561 v1.0.0 (participants.tsv + sub-*/)")
    parser.add_argument("--limit", type=int, help="Only use the first N scans (quick run)")
    parser.add_argument("--work-dir", type=Path, help="Parent for temporary files (default: the system temp directory)")
    args = parser.parse_args()

    record = {"generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "versions": versions(),
              "scope": "Software verification on public data; aggregate counts only; not clinical validation."}
    if args.work_dir:
        args.work_dir.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="verify_dicom_volume_", dir=args.work_dir))
    try:
        record["geometry"] = check_geometry(work)
        subjects = []
        if args.openneuro_root and (args.openneuro_root / "participants.tsv").is_file():
            subjects = openneuro_subjects(args.openneuro_root)[: args.limit]
            record["cohort"] = {"source": "OpenNeuro ds007561 v1.0.0 (CC0), previously evaluated: historical cohort",
                                "scans": len(subjects), "complete": args.limit is None}
        if not subjects:
            for name in ("frozen_pipeline_parity", "published_result_replay", "stage_model_on_raw_slices"):
                record[name] = {"skipped": "no local OpenNeuro ds007561 copy given (--openneuro-root)"}
        else:
            record["frozen_pipeline_parity"] = check_parity(subjects, work)
            try:
                record["published_result_replay"] = check_replay(subjects) if args.limit is None else {"skipped": "--limit set"}
            except FileNotFoundError as error:
                record["published_result_replay"] = {"skipped": str(error)}
            try:
                record["stage_model_on_raw_slices"] = describe_stage_model(subjects)
            except (FileNotFoundError, KeyError, TypeError, AttributeError) as error:
                record["stage_model_on_raw_slices"] = {"skipped": f"production checkpoint unavailable ({error})"}
    finally:
        shutil.rmtree(work, ignore_errors=True)

    checks = [record[name] for name in ("geometry", "frozen_pipeline_parity", "published_result_replay") if "passed" in record[name]]
    record["all_executed_checks_passed"] = all(check["passed"] for check in checks)
    text = json.dumps(record, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text)
    return 0 if record["all_executed_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
