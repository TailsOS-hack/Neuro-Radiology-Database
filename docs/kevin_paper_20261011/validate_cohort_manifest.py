#!/usr/bin/env python3
"""Read-only staging checks for Kevin's new scan-level cohort manifest.

Exit 0: structural staging checks passed, NOT authorization to train.
Exit 2: not ready, including an empty template or a missing endpoint vocabulary.
This program never decodes images, trains a model, uploads data, or edits inputs.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
LIMITATIONS = [
    "Permission, de-identification, provenance, and label-review statuses are supplied declarations, not independently verified here.",
    "Patient grouping uses supplied cohort patient IDs and optional global links; missing cross-cohort links can hide shared participants.",
    "No image decoding, content hashing, duplicate-image search, DICOM tag inspection, sequence verification, or preprocessing QC is performed.",
    "Counts describe manifest scan rows and supplied patient groups, not independently verified MRI volumes, patients, or slice counts.",
    "Passing staging checks does not establish clinical label validity, cohort independence, adequate sample size, or readiness to train.",
]


def validate(manifest: Path, data_root: Path, endpoint_labels: list[str]) -> dict:
    schema = json.loads((HERE / "data_dictionary.json").read_text())
    fields = schema["columns"]
    errors: list[str] = []
    warnings: list[str] = []
    rows: list[dict] = []
    try:
        with manifest.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            columns = reader.fieldnames or []
            if len(columns) != len(set(columns)):
                errors.append("Duplicate CSV header names.")
            missing = sorted(set(fields) - set(columns))
            if missing:
                errors.append("Missing columns: " + ", ".join(missing))
            for line, raw in enumerate(reader, 2):
                if None in raw or any(value is None for value in raw.values()):
                    errors.append(f"Row {line}: wrong number of CSV fields.")
                    continue
                row = {key: value.strip() for key, value in raw.items()}
                if any(row.values()):
                    rows.append(row)
    except (OSError, csv.Error, UnicodeError) as exc:
        errors.append(f"Cannot read manifest: {exc}")
    if not rows:
        errors.append("Manifest has no scan rows; the empty template is NOT READY.")
    allowed = set(endpoint_labels)
    if not allowed or any(not label.strip() for label in allowed):
        errors.append("Supply the prospectively approved endpoint vocabulary using repeated --endpoint-label arguments.")
    if len(endpoint_labels) != len(allowed):
        errors.append("Endpoint vocabulary contains duplicate labels.")

    patient_splits = defaultdict(set)
    global_splits = defaultdict(set)
    patient_global_links = defaultdict(set)
    cohort_roles = defaultdict(set)
    cohort_splits = defaultdict(set)
    mappings = defaultdict(set)
    scans: set[tuple] = set()
    paths: dict[Path, int] = {}
    path_kinds: dict[Path, str] = {}
    metadata_missing = Counter()
    root = data_root.resolve()
    for index, row in enumerate(rows, 2):
        prefix = f"Row {index}: "
        for name, spec in fields.items():
            value = row.get(name, "")
            if spec.get("required") and not value:
                errors.append(prefix + f"missing {name}.")
            if value and "allowed" in spec and value not in spec["allowed"]:
                errors.append(prefix + f"invalid {name}: {value!r}.")
            if "required_value" in spec and value != spec["required_value"]:
                errors.append(prefix + f"{name} must be {spec['required_value']!r}; pending/unreviewed rows cannot pass.")
        for name in schema["optional_metadata_columns"]:
            if not row.get(name):
                metadata_missing[name] += 1
        for name, low, high in [("age_years", 0, 120), ("field_strength_t", 0.01, 20)]:
            if row.get(name):
                try:
                    value = float(row[name])
                    if not math.isfinite(value) or not low <= value <= high:
                        raise ValueError
                except ValueError:
                    errors.append(prefix + f"{name} must be a finite number from {low} to {high}.")
        cohort = row.get("cohort_id", "")
        release = row.get("cohort_release", "")
        patient = row.get("patient_id", "")
        split = row.get("split", "")
        role = row.get("cohort_role", "")
        if cohort and patient:
            patient_splits[(cohort, patient)].add(split)
            if row.get("global_patient_id"):
                patient_global_links[(cohort, patient)].add(row["global_patient_id"])
        if row.get("global_patient_id"):
            global_splits[row["global_patient_id"]].add(split)
        cohort_roles[cohort].add(role)
        cohort_splits[cohort].add(split)
        if role == "development" and split not in {"train", "validation", "test"}:
            errors.append(prefix + "development cohort must use train, validation, or test.")
        if role == "external" and split != "external_test":
            errors.append(prefix + "external cohort must use external_test exclusively.")
        label = row.get("endpoint_label", "")
        if label and label not in allowed:
            errors.append(prefix + f"endpoint_label {label!r} is outside the approved vocabulary.")
        mappings[(cohort, release, row.get("source_label", ""))].add(label)
        scan = (cohort, release, row.get("scan_id", ""))
        if scan in scans:
            errors.append(prefix + "duplicate scan_id within cohort release.")
        scans.add(scan)
        relative = row.get("relative_file_path", "")
        if not relative:
            continue
        rel_path = Path(relative)
        if rel_path.is_absolute() or ".." in rel_path.parts or "://" in relative:
            errors.append(prefix + "relative_file_path must be local and relative, with no '..' components.")
            continue
        path = (root / rel_path).resolve()
        if not path.is_relative_to(root):
            errors.append(prefix + "resolved path escapes data root (including through a symlink).")
            continue
        if path in paths:
            errors.append(prefix + f"duplicate resolved image/series path (also row {paths[path]}).")
        paths[path] = index
        kind = row.get("input_kind", "")
        path_kinds[path] = kind
        if kind == "volume_file" and not path.is_file():
            errors.append(prefix + "volume_file path is not an existing file.")
        if kind == "dicom_series_directory":
            if not path.is_dir():
                errors.append(prefix + "dicom_series_directory is not an existing directory.")
            elif not any(path.iterdir()):
                errors.append(prefix + "DICOM series directory is empty.")

    for key, splits in patient_splits.items():
        if len(splits) > 1:
            errors.append(f"Patient group {key!r} crosses splits: {sorted(splits)}.")
    for key, links in patient_global_links.items():
        if len(links) > 1:
            errors.append(f"Patient group {key!r} has conflicting global_patient_id values.")
    # Propagate an optional global link to every visit of that local patient.
    for key, links in patient_global_links.items():
        for link in links:
            global_splits[link].update(patient_splits[key])
    for key, splits in global_splits.items():
        if len(splits) > 1:
            errors.append(f"Linked global patient {key!r} crosses splits: {sorted(splits)}.")
    for cohort, roles in cohort_roles.items():
        if len(roles) > 1 or ("external_test" in cohort_splits[cohort] and len(cohort_splits[cohort]) > 1):
            errors.append(f"Cohort {cohort!r} mixes development and external use across rows/releases.")
    for key, labels in mappings.items():
        if len(labels) > 1:
            errors.append(f"Source label {key!r} has inconsistent endpoint mapping.")
    for path in paths:
        for ancestor in path.parents:
            if ancestor in paths and path_kinds[ancestor] == "dicom_series_directory":
                errors.append(f"Rows {paths[path]} and {paths[ancestor]} reference overlapping series paths.")
    if rows and not all(row.get("global_patient_id") for row in rows):
        warnings.append("Global patient linkage is incomplete; cross-cohort shared participants cannot be ruled out.")
    if metadata_missing:
        warnings.append("Some clinical/acquisition metadata are absent; see missing_optional_metadata counts.")
    declared_patient_keys = set()
    for key in patient_splits:
        links = patient_global_links.get(key, set())
        declared_patient_keys.add(("global", next(iter(links))) if len(links) == 1 else ("cohort", *key))
    return {
        "status": "NOT_READY" if errors else "STAGING_CHECKS_PASSED",
        "training_ready": False,
        "mode": "read_only_no_training_no_inference_no_upload",
        "manifest": str(manifest.resolve()),
        "data_root": str(root),
        "approved_endpoint_vocabulary": sorted(allowed),
        "counts": {
            "source_scan_rows": len(rows),
            "volume_file_rows": sum(r.get("input_kind") == "volume_file" for r in rows),
            "dicom_series_rows": sum(r.get("input_kind") == "dicom_series_directory" for r in rows),
            "cohort_patient_groups": len(patient_splits),
            "patient_groups_after_supplied_global_links": len(declared_patient_keys),
            "cohorts": len({r.get("cohort_id") for r in rows if r.get("cohort_id")}),
            "scan_rows_by_split": dict(Counter(r.get("split", "") for r in rows)),
            "unit": "one declared source scan per row; no slices counted or expanded",
        },
        "missing_optional_metadata": dict(metadata_missing),
        "errors": errors,
        "warnings": warnings,
        "limitations": LIMITATIONS,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", nargs="?", type=Path, default=HERE / "cohort_manifest_template.csv")
    parser.add_argument("--data-root", type=Path, help="Root for relative paths; defaults to the manifest directory.")
    parser.add_argument("--endpoint-label", action="append", default=[], help="One approved endpoint class; repeat for every class. Never inferred.")
    args = parser.parse_args()
    report = validate(args.manifest, args.data_root or args.manifest.parent, args.endpoint_label)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 2 if report["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
