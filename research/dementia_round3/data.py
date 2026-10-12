"""Patient inventory, scan/assessment matching and fail-closed data admission.

Inputs are PRIVATE normalized provider records, never inferred filename labels.
Provider-specific code mappings must be reviewed against the delivered dictionary.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import date
import hashlib
import json
from pathlib import Path

PARTITIONS = {"train", "tune", "calibration", "external", "historical"}
HISTORICAL_FAMILIES = {"besga", "harvard29352", "openneuro_ds007561", "legacy_augmented_jpeg"}


def valid_sha256(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def diagnosis_from_assessment(row):
    """Clinical phenotype AND etiology; biomarker positivity is not dementia."""
    status, cause = row.get("cognitive_status"), row.get("etiology")
    if status == "dementia" and cause == "AD":
        return 1
    if status == "normal" and cause == "none":
        return 0
    return None


def match_assessments(scans, assessments, *, maximum_days=180):
    """Nearest assessment, earlier date on ties; preserve every exclusion.

One deterministic earliest eligible scan per patient is primary. Other visits
remain in a separate repeatability list; they never increase primary sample N.
"""
    if maximum_days not in (90, 180):
        raise ValueError("Only the frozen 180-day primary / 90-day sensitivity windows are supported")
    visits = defaultdict(list)
    for assessment in assessments:
        key = assessment["patient_id"]
        if not isinstance(key, str) or not key.strip() or key != key.strip():
            raise ValueError("Patient identities must be canonical nonempty strings")
        date.fromisoformat(assessment["assessment_date"])
        visits[key].append(assessment)
    included, excluded = [], []
    scan_ids = set()
    for scan in scans:
        pid = scan["patient_id"]
        sid = scan["scan_id"]
        if any(not isinstance(x, str) or not x.strip() or x != x.strip() for x in (pid, sid)):
            raise ValueError("Patient/scan identities must be canonical nonempty strings")
        if (pid, sid) in scan_ids:
            raise ValueError("Duplicate patient/scan record")
        scan_ids.add((pid, sid))
        scan_date = date.fromisoformat(scan["scan_date"])
        candidates = sorted(visits.get(pid, []), key=lambda a: (abs((date.fromisoformat(a["assessment_date"]) - scan_date).days), a["assessment_date"]))
        reason = None
        nearest = candidates[0] if candidates else None
        if nearest is None:
            reason = "no_assessment"
        elif abs((date.fromisoformat(nearest["assessment_date"]) - scan_date).days) > maximum_days:
            reason = "assessment_outside_window"
        else:
            same_day = [a for a in candidates if a["assessment_date"] == nearest["assessment_date"]]
            if len({(a.get("cognitive_status"), a.get("etiology")) for a in same_day}) > 1:
                reason = "conflicting_same_day_assessments"
            elif diagnosis_from_assessment(nearest) is None:
                reason = "outside_clinical_AD_CN_endpoint"
            else:
                documented = sorted((a for a in same_day if a.get("reference_standard")), key=digest)
                if not documented:
                    reason = "reference_standard_missing"
                else:
                    nearest = documented[0]
        if reason:
            excluded.append({"patient_id": pid, "scan_id": sid, "reason": reason})
            continue
        included.append({**scan, "diagnosis": diagnosis_from_assessment(nearest), "cognitive_status": nearest["cognitive_status"], "etiology": nearest["etiology"], "assessment_date": nearest["assessment_date"], "assessment_gap_days": abs((date.fromisoformat(nearest["assessment_date"]) - scan_date).days), "reference_standard": nearest["reference_standard"], "matched_assessment_sha256": digest(nearest)})
    by_patient = defaultdict(list)
    for row in included:
        by_patient[row["patient_id"]].append(row)
    primary, repeated = [], []
    for pid in sorted(by_patient):
        rows = sorted(by_patient[pid], key=lambda r: (r["scan_date"], r["scan_id"]))
        primary.append(rows[0])
        repeated.extend(rows[1:])
    return {"maximum_days": maximum_days, "primary": primary, "repeatability": repeated, "exclusions": excluded,
            "counts": {"primary_patients": len(primary), "repeat_scans": len(repeated), "excluded_scans": len(excluded)}}


def validate_manifest(manifest, inventory, *, verify_files=True):
    """Return aggregate blockers, without publishing patient identities/paths."""
    blockers, warnings = [], []
    sources = {s["source_id"]: s for s in inventory.get("sources", [])}
    if len(sources) != len(inventory.get("sources", [])):
        blockers.append("duplicate_source_identity")
    rows = manifest.get("patients", [])
    if not rows:
        blockers.append("no_eligible_patients_acquired")
    matching = manifest.get("scan_assessment_evidence", {})
    if not all(k in matching for k in ("scans", "assessments", "matching_sha256")):
        blockers.append("scan_assessment_evidence_missing")
    elif verify_files:
        try:
            for ref in (matching["scans"], matching["assessments"]):
                if sha256_file(ref["path"]) != ref["sha256"]:
                    raise ValueError("Input bytes changed")
            matched = match_assessments(json.loads(Path(matching["scans"]["path"]).read_text()), json.loads(Path(matching["assessments"]["path"]).read_text()))
            if digest(matched) != matching["matching_sha256"]:
                raise ValueError("Matching changed")
            fields = ("patient_id", "source_id", "site_id", "institution_id", "scan_id", "scan_date", "volume_path", "volume_sha256", "canonical_volume_sha256", "modality", "full_brain", "previously_evaluated", "assessment_date", "diagnosis", "cognitive_status", "etiology", "reference_standard", "matched_assessment_sha256")
            expected = {r["patient_id"]: {k: r[k] for k in fields} for r in matched["primary"]}
            actual = {r["patient_id"]: {k: r[k] for k in fields} for r in rows}
            if actual != expected:
                raise ValueError("Primary membership/diagnosis differs from replayed matching")
        except (OSError, ValueError, KeyError, TypeError):
            blockers.append("scan_assessment_matching_does_not_reproduce")
    ids, hashes = {}, defaultdict(list)
    institutions = defaultdict(set)
    site_institution = {}
    counts = defaultdict(Counter)
    for row in rows:
        required = ("patient_id", "source_id", "site_id", "institution_id", "partition", "volume_path", "volume_sha256", "canonical_volume_sha256", "scan_date", "assessment_date", "reference_standard", "matched_assessment_sha256")
        if any(not isinstance(row.get(k), str) or not row[k] for k in required):
            blockers.append("missing_or_invalid_patient_manifest_fields")
            continue
        if any(row[k] != row[k].strip() for k in ("patient_id", "source_id", "site_id", "institution_id", "partition")):
            blockers.append("noncanonical_identity_whitespace")
        pid, partition = row["patient_id"], row["partition"]
        if pid in ids:
            blockers.append("duplicate_patient_or_repeated_visit_in_primary_manifest")
        ids[pid] = partition
        if partition not in PARTITIONS or type(row.get("diagnosis")) is not int or row["diagnosis"] not in (0, 1):
            blockers.append("invalid_partition_or_diagnosis")
            continue
        if diagnosis_from_assessment(row) != row["diagnosis"]:
            blockers.append("clinical_phenotype_etiology_diagnosis_mismatch")
        if not valid_sha256(row["matched_assessment_sha256"]):
            blockers.append("invalid_matched_assessment_hash")
        try:
            gap = abs((date.fromisoformat(row["scan_date"]) - date.fromisoformat(row["assessment_date"])).days)
            if gap > 180:
                blockers.append("assessment_outside_frozen_window")
        except ValueError:
            blockers.append("invalid_date")
        source = sources.get(row["source_id"])
        if not source:
            blockers.append("unregistered_source")
            continue
        family = source.get("source_family_id")
        if not isinstance(family, str) or not family.strip() or family != family.strip():
            blockers.append("missing_or_noncanonical_source_family")
        if source.get("access_status") != "authorized_and_acquired" or not source.get("terms_sha256") or not source.get("identity_review_sha256") or not source.get("diagnostic_mapping_sha256"):
            blockers.append("source_access_or_label_identity_review_incomplete")
        if any(not valid_sha256(source.get(k)) for k in ("terms_sha256", "identity_review_sha256", "diagnostic_mapping_sha256")):
            blockers.append("invalid_source_review_hash")
        if source.get("source_family_id") == "legacy_augmented_jpeg":
            blockers.append("legacy_jpeg_not_admitted_to_clinical_study")
        if partition == "external" and (source.get("source_family_id") in HISTORICAL_FAMILIES or row.get("previously_evaluated") is not False):
            blockers.append("external_subject_or_source_already_observed")
        if row.get("modality") != "T1w" or row.get("full_brain") is not True:
            blockers.append("unsupported_or_unverified_T1_volume")
        inst, site = row["institution_id"], row["site_id"]
        if site in site_institution and site_institution[site] != inst:
            blockers.append("site_has_conflicting_institution_identity")
        site_institution[site] = inst
        institutions[partition].add(inst)
        counts[(partition, site)][row["diagnosis"]] += 1
        for key in ("volume_sha256", "canonical_volume_sha256"):
            value = row[key]
            if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                blockers.append("invalid_volume_hash")
            hashes[(key, value)].append((pid, partition))
        if verify_files:
            try:
                if sha256_file(row["volume_path"]) != row["volume_sha256"]:
                    blockers.append("raw_volume_checksum_mismatch")
                # Canonical voxels and orientation are independently recomputed.
                import nibabel as nib
                import numpy as np
                image = nib.as_closest_canonical(nib.load(row["volume_path"]))
                data = np.asarray(image.get_fdata(dtype=np.float32), dtype="<f4")
                if data.ndim != 3 or not np.isfinite(data).all():
                    blockers.append("invalid_3D_finite_volume")
                canonical = hashlib.sha256(data.tobytes(order="C") + np.asarray(image.affine, dtype="<f8").tobytes()).hexdigest()
                if canonical != row["canonical_volume_sha256"]:
                    blockers.append("canonical_volume_checksum_mismatch")
            except (OSError, ValueError, ImportError):
                blockers.append("volume_unreadable_or_dependency_missing")
    for group in hashes.values():
        if len({x[0] for x in group}) > 1:
            blockers.append("duplicate_volume_across_patient_identities")
    if len(set(site_institution.values())) != len(site_institution):
        blockers.append("site_ids_must_represent_distinct_institutions_not_scanner_aliases")
    development = set().union(*(institutions[p] for p in ("train", "tune", "calibration")))
    if development & institutions["external"]:
        blockers.append("institution_overlap_development_external")
    if institutions["historical"] & institutions["external"]:
        blockers.append("external_institution_already_observed_historically")
    if set(inventory.get("previously_observed_institutions", [])) & institutions["external"]:
        blockers.append("external_institution_already_observed_historically")
    if len(institutions["train"]) < 2:
        blockers.append("fewer_than_two_development_institutions")
    if len(institutions["external"]) < 2:
        blockers.append("fewer_than_two_fresh_external_institutions")
    for partition in ("train", "tune", "calibration", "external"):
        pc = Counter()
        for (part, _), counter in counts.items():
            if part == partition:
                pc.update(counter)
        if any(pc[label] == 0 for label in (0, 1)):
            blockers.append(f"{partition}_missing_AD_or_CN")
    for (partition, _), counter in counts.items():
        if partition in ("train", "external") and any(counter[label] == 0 for label in (0, 1)):
            blockers.append(f"{partition}_site_missing_AD_or_CN")
    review = manifest.get("cross_source_identity_review", {})
    if review.get("status") != "reviewed" or not valid_sha256(review.get("evidence_sha256")):
        blockers.append("cross_source_patient_identity_review_missing")
    if manifest.get("provider_mapping_review", {}).get("status") != "reviewed":
        blockers.append("provider_field_mapping_not_reviewed")
    if verify_files:
        # Human review cannot be inferred from a status string or a hash-shaped
        # placeholder. Preserve the actual local review artifacts for audit.
        for source_id in {r.get("source_id") for r in rows}:
            source = sources.get(source_id, {})
            for kind in ("terms", "identity_review", "diagnostic_mapping"):
                ref = source.get("review_artifacts", {}).get(kind, {})
                try:
                    path = Path(ref["path"])
                    if path.is_symlink() or sha256_file(path) != source.get(kind+"_sha256") or ref.get("sha256") != source.get(kind+"_sha256"):
                        raise ValueError("Review evidence changed")
                except (OSError, ValueError, KeyError, TypeError):
                    blockers.append("source_review_artifact_missing_or_changed")
        for kind in ("cross_source_identity_review", "provider_mapping_review"):
            review = manifest.get(kind, {})
            try:
                path = Path(review["evidence_path"])
                if not review.get("reviewer") or path.is_symlink() or sha256_file(path) != review.get("evidence_sha256"):
                    raise ValueError("Review evidence unavailable")
            except (OSError, ValueError, KeyError, TypeError):
                blockers.append("manifest_review_artifact_missing_or_changed")
    return {"status": "passed" if not blockers else "blocked", "blockers": sorted(set(blockers)), "warnings": warnings,
            "primary_patients": len(rows), "counts": [{"partition": p, "site_id": s, "CN": c[0], "AD": c[1]} for (p, s), c in sorted(counts.items())],
            "development_institutions": len(development), "external_institutions": len(institutions["external"]),
            "file_checks_performed": verify_files, "clinical_readiness": False}
