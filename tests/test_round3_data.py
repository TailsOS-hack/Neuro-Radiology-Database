"""Synthetic patient/visit integrity cases; no acquired MRI or diagnosis claims."""
import copy
import json
from pathlib import Path

import pytest

from research.dementia_round3.data import match_assessments, validate_manifest, diagnosis_from_assessment
from research.dementia_round3.common import assess_data_gate, authorize_rows, canonical_hash, power_receipt_blockers, verify_training_authorization

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = json.loads((ROOT / "docs/kevin_paper_20261011/round3/protocol.json").read_text())


def assessment(day="2020-01-01", **changes):
    return {"patient_id": "invented_patient", "assessment_date": day, "cognitive_status": "dementia", "etiology": "AD", "reference_standard": "invented_fixture", **changes}


def test_phenotype_is_not_biomarker_positivity_or_mci():
    assert diagnosis_from_assessment(assessment()) == 1
    assert diagnosis_from_assessment(assessment(cognitive_status="normal", etiology="none")) == 0
    assert diagnosis_from_assessment(assessment(cognitive_status="normal")) is None
    assert diagnosis_from_assessment(assessment(cognitive_status="MCI")) is None


def test_primary_and_sensitivity_windows_differ_without_relabeling():
    scan = {"patient_id": "invented_patient", "scan_id": "a", "scan_date": "2020-05-01"}
    assert len(match_assessments([scan], [assessment()])["primary"]) == 1
    r = match_assessments([scan], [assessment()], maximum_days=90)
    assert not r["primary"] and r["exclusions"][0]["reason"] == "assessment_outside_window"


def test_nearest_is_chosen_before_endpoint_exclusion():
    scan = {"patient_id": "invented_patient", "scan_id": "a", "scan_date": "2020-01-05"}
    r = match_assessments([scan], [assessment(), assessment("2020-01-05", cognitive_status="MCI")])
    assert not r["primary"]  # Never shop for a farther, more convenient AD label.


def test_equal_distance_earlier_date_and_one_patient_primary():
    scans = [{"patient_id": "invented_patient", "scan_id": s, "scan_date": d} for s, d in [("b", "2020-01-06"), ("a", "2020-01-05")]]
    a = [assessment("2020-01-04"), assessment("2020-01-06", cognitive_status="normal", etiology="none")]
    r = match_assessments(scans, a)
    assert len(r["primary"]) == 1 and len(r["repeatability"]) == 1
    assert r["primary"][0]["scan_id"] == "a" and r["primary"][0]["diagnosis"] == 1


def test_conflicting_same_day_never_cherry_picked():
    s = {"patient_id": "invented_patient", "scan_id": "a", "scan_date": "2020-01-01"}
    r = match_assessments([s], [assessment(), assessment(cognitive_status="normal", etiology="none")])
    assert not r["primary"] and r["exclusions"][0]["reason"] == "conflicting_same_day_assessments"


def test_matching_identical_label_reference_order_is_deterministic():
    s = {"patient_id": "invented_patient", "scan_id": "a", "scan_date": "2020-01-01"}
    visits = [assessment(reference_standard=""), assessment()]
    assert match_assessments([s], visits) == match_assessments([s], visits[::-1])


def test_empty_data_never_authorizes_training():
    r = assess_data_gate(PROTOCOL, {"sources": []}, {"patients": []}, {"status": "pending"}, verify_files=False)
    assert not r["training_authorized"]
    assert "no_eligible_patients_acquired" in r["blockers"]
    assert "volume_integrity_not_verified" in r["blockers"]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1, True, "0.9"])
def test_invalid_power_cannot_satisfy_gate(value):
    p = {"status": "simulated", "assumptions": {}, "joint_gate_power_95ci": [value, 1.]}
    assert "invalid_power_confidence_interval" in power_receipt_blockers(p, PROTOCOL)


def test_patient_authorization_cannot_substitute_site_partition_or_volume():
    row = {"patient_id": "invented_patient", "site_id": "A", "partition": "train", "volume_sha256": "0" * 64}
    auth = {"allowed_partitions": ["train", "tune"], "authorized_rows": {row["patient_id"]: canonical_hash(row)}}
    assert authorize_rows(auth, [row])
    for key, value in [("site_id", "B"), ("volume_sha256", "1" * 64), ("partition", "external")]:
        with pytest.raises(ValueError):
            authorize_rows(auth, [{**row, key: value}])
    with pytest.raises(ValueError):
        authorize_rows(auth, [row, row])


def test_fabricated_success_flag_is_not_training_authorization(tmp_path):
    p = tmp_path / "false_authorization.json"
    p.write_text(json.dumps({"status": "passed", "training_authorized": True}))
    with pytest.raises(ValueError, match="content changed"):
        verify_training_authorization(p)
