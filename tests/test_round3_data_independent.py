"""Independent synthetic tests of identity and phenotype admission boundaries."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from research.dementia_round3.data import digest, match_assessments, sha256_file, validate_manifest


def synthetic_inventory_manifest():
    inventory = {"sources": [{"source_id": "synthetic_source", "source_family_id": "synthetic_family",
        "access_status": "authorized_and_acquired", "terms_sha256": "a"*64,
        "identity_review_sha256": "b"*64, "diagnostic_mapping_sha256": "c"*64}]}
    rows = []
    layout = [("train", "devA"), ("train", "devB"), ("tune", "devA"),
              ("calibration", "devB"), ("external", "externalA"), ("external", "externalB")]
    for partition, institution in layout:
        for label in (0, 1):
            pid = f"synthetic_{partition}_{institution}_{label}"
            assessment = {"patient_id": pid, "assessment_date": "2026-01-01",
                "cognitive_status": "dementia" if label else "normal",
                "etiology": "AD" if label else "none", "reference_standard": "synthetic_clinical_reference"}
            rows.append({**assessment, "source_id": "synthetic_source", "site_id": institution,
                "institution_id": institution, "partition": partition, "diagnosis": label,
                "scan_id": pid+"_scan", "scan_date": "2026-01-02", "assessment_gap_days": 1,
                "matched_assessment_sha256": digest(assessment), "volume_path": "/synthetic/not-read.nii.gz",
                "volume_sha256": digest([pid, "raw"]), "canonical_volume_sha256": digest([pid, "canonical"]),
                "modality": "T1w", "full_brain": True, "previously_evaluated": False})
    return inventory, {"patients": rows,
        "scan_assessment_evidence": {"scans": {"path": "/synthetic/scans.json", "sha256": "e"*64},
                                     "assessments": {"path": "/synthetic/assessments.json", "sha256": "f"*64}, "matching_sha256": "a"*64},
        "cross_source_identity_review": {"status": "reviewed", "evidence_sha256": "d"*64},
        "provider_mapping_review": {"status": "reviewed"}}


class IndependentRound3DataTests(unittest.TestCase):
    def audit(self, inventory, manifest):
        # Metadata-only probes deliberately perform no real-volume read or
        # authorization. common.assess_data_gate independently blocks this mode.
        return validate_manifest(manifest, inventory, verify_files=False)

    def test_positive_synthetic_metadata_and_exact_duplicate_guard(self):
        inv, manifest = synthetic_inventory_manifest()
        self.assertEqual(self.audit(inv, manifest)["status"], "passed")
        manifest["patients"][-1]["patient_id"] = manifest["patients"][0]["patient_id"]
        self.assertEqual(self.audit(inv, manifest)["status"], "blocked")

    def test_patient_whitespace_alias_rejected(self):
        inv, manifest = synthetic_inventory_manifest()
        manifest["patients"][-1]["patient_id"] = manifest["patients"][0]["patient_id"]+" "
        self.assertEqual(self.audit(inv, manifest)["status"], "blocked")

    def test_institution_whitespace_cannot_make_seen_site_fresh(self):
        inv, manifest = synthetic_inventory_manifest()
        for row in manifest["patients"]:
            if row["site_id"] == "externalA":
                row["institution_id"] = "devA "
        self.assertEqual(self.audit(inv, manifest)["status"], "blocked")

    def test_source_family_is_required_for_historical_exclusion(self):
        inv, manifest = synthetic_inventory_manifest()
        del inv["sources"][0]["source_family_id"]
        self.assertEqual(self.audit(inv, manifest)["status"], "blocked")

    def test_assessment_and_review_digests_must_be_sha256(self):
        inv, manifest = synthetic_inventory_manifest()
        manifest["patients"][0]["matched_assessment_sha256"] = "reviewed"
        self.assertEqual(self.audit(inv, manifest)["status"], "blocked")
        inv, manifest = synthetic_inventory_manifest()
        inv["sources"][0]["diagnostic_mapping_sha256"] = "reviewed"
        self.assertEqual(self.audit(inv, manifest)["status"], "blocked")

    def test_historical_institution_cannot_become_never_seen_external(self):
        inv, manifest = synthetic_inventory_manifest()
        historical = copy.deepcopy(manifest["patients"][0])
        historical.update(patient_id="synthetic_prior_study_patient", partition="historical",
            site_id="prior_site_name", institution_id="externalA", previously_evaluated=True,
            volume_sha256=digest("different_prior_raw"), canonical_volume_sha256=digest("different_prior_canonical"))
        manifest["patients"].append(historical)
        self.assertEqual(self.audit(inv, manifest)["status"], "blocked")

    def test_bound_scan_institution_cannot_be_relabelled_in_manifest(self):
        import nibabel as nib
        import numpy as np
        inv, manifest = synthetic_inventory_manifest()
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            for kind in ("terms", "identity_review", "diagnostic_mapping"):
                evidence = folder/f"synthetic_{kind}.json"
                evidence.write_text(json.dumps({"synthetic_test_only": True, "kind": kind}))
                source = inv["sources"][0]
                source[kind+"_sha256"] = sha256_file(evidence)
                source.setdefault("review_artifacts", {})[kind] = {"path": str(evidence), "sha256": sha256_file(evidence)}
            for kind in ("cross_source_identity_review", "provider_mapping_review"):
                evidence = folder/f"synthetic_{kind}.json"
                evidence.write_text(json.dumps({"synthetic_test_only": True, "kind": kind}))
                manifest[kind].update(reviewer="synthetic_fixture", evidence_path=str(evidence), evidence_sha256=sha256_file(evidence))
            scans, assessments = [], []
            for i, row in enumerate(manifest["patients"]):
                voxels = np.full((4, 4, 4), i+1., dtype=np.float32)
                path = folder/f"synthetic_{i}.nii.gz"
                nib.save(nib.Nifti1Image(voxels, np.eye(4)), path)
                row["volume_path"] = str(path)
                row["volume_sha256"] = sha256_file(path)
                row["canonical_volume_sha256"] = hashlib.sha256(voxels.astype("<f4").tobytes(order="C")+np.eye(4, dtype="<f8").tobytes()).hexdigest()
                a = {k: row[k] for k in ("patient_id", "assessment_date", "cognitive_status", "etiology", "reference_standard")}
                row["matched_assessment_sha256"] = digest(a)
                assessments.append(a)
                derived = {"assessment_date", "cognitive_status", "etiology", "reference_standard", "diagnosis", "assessment_gap_days", "matched_assessment_sha256"}
                scans.append({k: v for k, v in row.items() if k not in derived})
            scan_path, assessment_path = folder/"scans.json", folder/"assessments.json"
            scan_path.write_text(json.dumps(scans)); assessment_path.write_text(json.dumps(assessments))
            manifest["scan_assessment_evidence"] = {"scans": {"path": str(scan_path), "sha256": sha256_file(scan_path)},
                "assessments": {"path": str(assessment_path), "sha256": sha256_file(assessment_path)},
                "matching_sha256": digest(match_assessments(scans, assessments))}
            self.assertEqual(validate_manifest(manifest, inv, verify_files=True)["status"], "passed")
            for row in manifest["patients"]:
                if row["site_id"] == "externalA":
                    row["institution_id"] = "unbound_institution_alias"
            result = validate_manifest(manifest, inv, verify_files=True)
            self.assertIn("scan_assessment_matching_does_not_reproduce", result["blockers"])

    def test_manifest_integer_label_must_agree_with_clinical_phenotype(self):
        inv, manifest = synthetic_inventory_manifest()
        manifest["patients"][0]["cognitive_status"] = "normal"
        manifest["patients"][0]["etiology"] = "none"
        manifest["patients"][0]["diagnosis"] = 1
        # Preserve both class counts by selecting another AD row as CN. The
        # rejection must derive from phenotype inconsistency, not absent class.
        manifest["patients"][1]["diagnosis"] = 0
        self.assertEqual(self.audit(inv, manifest)["status"], "blocked")

    def test_same_day_identical_labels_missing_reference_order_invariant(self):
        scan = {"patient_id": "syntheticA", "scan_id": "s1", "scan_date": "2026-01-01"}
        good = {"patient_id": "syntheticA", "assessment_date": "2026-01-01", "cognitive_status": "normal", "etiology": "none", "reference_standard": "clinical"}
        missing = {k: v for k, v in good.items() if k != "reference_standard"}
        self.assertEqual(match_assessments([scan], [good, missing]), match_assessments([scan], [missing, good]))

    def test_frozen_earlier_tie_and_repeatability_unit(self):
        scans = [{"patient_id": "syntheticA", "scan_id": "s2", "scan_date": "2026-01-20"},
                 {"patient_id": "syntheticA", "scan_id": "s1", "scan_date": "2026-01-10"}]
        before = {"patient_id": "syntheticA", "assessment_date": "2026-01-01", "cognitive_status": "normal", "etiology": "none", "reference_standard": "clinical"}
        after = {**before, "assessment_date": "2026-01-19", "cognitive_status": "dementia", "etiology": "AD"}
        result = match_assessments(scans, [after, before])
        self.assertEqual(result["counts"], {"primary_patients": 1, "repeat_scans": 1, "excluded_scans": 0})
        self.assertEqual(result["primary"][0]["scan_id"], "s1")
        self.assertEqual(result["primary"][0]["diagnosis"], 0)
        self.assertEqual(result, match_assessments(list(reversed(scans)), [before, after]))

    def test_conflicting_same_day_diagnosis_is_excluded(self):
        scan = {"patient_id": "syntheticA", "scan_id": "s1", "scan_date": "2026-01-01"}
        cn = {"patient_id": "syntheticA", "assessment_date": "2026-01-01", "cognitive_status": "normal", "etiology": "none", "reference_standard": "clinical"}
        ad = {**cn, "cognitive_status": "dementia", "etiology": "AD"}
        r = match_assessments([scan], [cn, ad])
        self.assertEqual(r["primary"], [])
        self.assertEqual(r["exclusions"][0]["reason"], "conflicting_same_day_assessments")


if __name__ == "__main__":
    unittest.main()
