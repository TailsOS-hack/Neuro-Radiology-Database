"""Content-addressed scientific gates. A status flag alone never authorizes fitting."""
from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path
from datetime import datetime, timezone

from .data import digest as canonical_hash, sha256_file, validate_manifest


def read_json(path):
    return json.loads(Path(path).read_text(), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"Nonfinite JSON constant: {value}")))


def write_json(path, value, *, overwrite=False):
    path = Path(path)
    content = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if path.exists() and not overwrite:
        if path.read_text() != content:
            raise FileExistsError(f"Refusing to replace existing frozen artifact: {path.name}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents concurrent freezes from silently replacing a file.
    with path.open("w" if overwrite else "x") as stream:
        stream.write(content)


def code_manifest():
    base = Path(__file__).parent
    return {p.name: sha256_file(p) for p in sorted(base.glob("*.py"))}


def validate_protocol(protocol):
    if protocol.get("schema_version") != "dementia_round3_v1":
        raise ValueError("Unsupported protocol")
    if protocol.get("class_order") != ["CN", "AD"]:
        raise ValueError("Clinical class order must be CN, AD")
    if protocol.get("endpoint") != "clinical_AD_vs_CN":
        raise ValueError("Four-class image labels and biological AD are not this endpoint")
    if protocol.get("matching", {}).get("primary_window_days") != 180 or protocol["matching"].get("sensitivity_window_days") != 90:
        raise ValueError("Unexpected scan/diagnosis matching windows")
    if protocol.get("search", {}).get("maximum_configurations") != 24:
        raise ValueError("Search must be bounded to 24 preregistered configurations")
    if protocol.get("execution", {}).get("training_gpu") != "NVIDIA B300" or protocol["execution"].get("analysis_location") != "local_mac":
        raise ValueError("User specified B300 training and local Mac analysis")
    if protocol["execution"].get("maximum_gpu_hourly_usd") != 8.99:
        raise ValueError("Live B300 hourly rate must not exceed the user-selected $8.99")
    if protocol.get("clinical_deployment_permitted") is not False:
        raise ValueError("Research authorization is not clinical deployment authorization")
    return protocol


@lru_cache(maxsize=8)
def _recompute_power(serialized_assumptions, simulations, bootstrap_replicates, seed):
    from .sample_size import simulate_plan
    return simulate_plan(json.loads(serialized_assumptions), simulations=simulations,
                         bootstrap_replicates=bootstrap_replicates, seed=seed)


def power_receipt_blockers(power_plan, protocol):
    problems = []
    if power_plan.get("status") != "simulated":
        return ["sample_size_simulation_not_completed"]
    assumptions = power_plan.get("assumptions", {})
    if power_plan.get("assumptions_sha256") != canonical_hash(assumptions):
        problems.append("sample_size_assumptions_digest_mismatch")
    review = power_plan.get("design_review", {})
    if review.get("status") != "approved_for_research_protocol" or not review.get("reviewer") or not review.get("assumption_rationale"):
        problems.append("sample_size_design_review_missing")
    interval = power_plan.get("joint_gate_power_95ci")
    if (not isinstance(interval, list) or len(interval) != 2 or
        any(type(x) not in (int, float) or not math.isfinite(x) or not 0 <= x <= 1 for x in interval)
        or interval[0] > interval[1]):
        problems.append("invalid_power_confidence_interval")
    elif interval[0] < .80:
        problems.append("insufficient_planned_joint_gate_power_precision")
    simulations, bootstrap = power_plan.get("simulations"), power_plan.get("bootstrap_replicates")
    if type(simulations) is not int or not 1000 <= simulations <= 100000:
        problems.append("sample_size_simulations_outside_reviewed_bounds")
    if type(bootstrap) is not int or bootstrap != protocol["statistics"]["bootstrap_replicates"]:
        problems.append("planning_bootstrap_does_not_match_frozen_evaluation")
    seed = power_plan.get("seed")
    if type(seed) is not int or seed < 0:
        problems.append("invalid_sample_size_seed")
    if not problems:
        try:
            recomputed = _recompute_power(json.dumps(assumptions, sort_keys=True), simulations, bootstrap, seed)
            if any(power_plan.get(key) != value for key, value in recomputed.items()):
                problems.append("sample_size_receipt_does_not_reproduce")
        except (ValueError, TypeError, KeyError):
            problems.append("invalid_sample_size_assumptions")
    return problems


def assess_data_gate(protocol, inventory, manifest, power_plan, *, verify_files=True):
    validate_protocol(protocol)
    audit = validate_manifest(manifest, inventory, verify_files=verify_files)
    blockers = list(audit["blockers"])
    # The actual simulator receipt is required, not a manually entered N or claim.
    blockers.extend(power_receipt_blockers(power_plan, protocol))
    if power_plan.get("status") == "simulated":
        assumptions = power_plan.get("assumptions", {})
        if power_plan.get("assumptions_sha256") != canonical_hash(assumptions):
            blockers.append("sample_size_assumptions_digest_mismatch")
        actual = {r["site_id"]: r for r in audit["counts"] if r["partition"] == "external"}
        expected = {r["site_id"]: r for r in assumptions.get("sites", [])}
        if set(actual) != set(expected) or len(expected) < 2:
            blockers.append("planned_external_sites_do_not_match_manifest")
        for site, spec in expected.items():
            for label in ("CN", "AD"):
                needed = spec.get("classes", {}).get(label, {}).get("n", 0)
                if type(needed) is not int or needed <= 0 or actual.get(site, {}).get(label, 0) < needed:
                    blockers.append("external_site_class_count_below_planned_requirement")
    if not verify_files:
        blockers.append("volume_integrity_not_verified")
    return {**audit, "status": "passed" if not blockers else "blocked", "blockers": sorted(set(blockers)),
            "training_authorized": not blockers, "protocol_hash": canonical_hash(protocol),
            "inventory_hash": canonical_hash(inventory), "manifest_hash": canonical_hash(manifest),
            "power_plan_hash": canonical_hash(power_plan)}


def _bind(path):
    p = Path(path).resolve()
    return {"path": str(p), "sha256": sha256_file(p)}


def freeze_development(*, protocol_path, inventory_path, manifest_path, power_plan_path, output_path):
    if Path(protocol_path).with_suffix(".external_opened.json").exists():
        raise ValueError("External testing already opened for this protocol; fitting is closed")
    refs = {name: _bind(path) for name, path in {
        "protocol": protocol_path, "inventory": inventory_path, "manifest": manifest_path, "power_plan": power_plan_path}.items()}
    content = {name: read_json(ref["path"]) for name, ref in refs.items()}
    gate = assess_data_gate(**content, verify_files=True)
    if gate["status"] != "passed":
        raise ValueError("Training blocked: " + ", ".join(gate["blockers"]))
    from .training import bounded_search_space, load_prepared_patient
    from .models import ModelConfig
    from .preprocessing import PreprocessingConfig, preflight
    protocol = content["protocol"]
    if protocol.get("search_space_hash") != canonical_hash(bounded_search_space()):
        raise ValueError("Execution protocol does not pin the actual bounded search space")
    prep = PreprocessingConfig(**read_json(_verified_ref(protocol.get("preprocessing_config_ref", {}))))
    if prep.version_hash != protocol.get("preprocessing_hash") or preflight(prep)["status"] != "ready":
        raise ValueError("Preprocessing configuration or pinned dependencies are not ready")
    rows = [r for r in content["manifest"]["patients"] if r["partition"] in ("train", "tune", "calibration")]
    for row in rows:
        load_prepared_patient(row, prep.version_hash, ModelConfig())
    # The external marker can have appeared during a long integrity review.
    if Path(protocol_path).with_suffix(".external_opened.json").exists():
        raise ValueError("External testing opened during development freeze")
    authorization = {"schema_version": "round3_authorization_v1", "phase": "development", "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_hash": gate["protocol_hash"], "inputs": refs, "code_sha256": code_manifest(),
        "allowed_actions": ["train", "infer", "calibrate"], "allowed_partitions": ["train", "tune", "calibration"],
        "authorized_rows": {r["patient_id"]: canonical_hash(r) for r in rows}, "gate": gate}
    authorization["authorization_sha256"] = canonical_hash(authorization)
    write_json(output_path, authorization)
    return authorization


def verify_training_authorization(path, expected_protocol_hash=None):
    """Recompute bytes, code, full patient integrity and scientific admission.

This function also validates the separate, read-only external authorization.
Callers must additionally check allowed_actions and authorize_rows.
"""
    authorization = read_json(path)
    signed = {k: v for k, v in authorization.items() if k != "authorization_sha256"}
    if authorization.get("authorization_sha256") != canonical_hash(signed):
        raise ValueError("Authorization content changed")
    if authorization.get("schema_version") != "round3_authorization_v1" or authorization.get("phase") not in ("development", "external_evaluation"):
        raise ValueError("Invalid authorization schema or phase")
    if authorization.get("code_sha256") != code_manifest():
        raise ValueError("Scientific code changed after authorization")
    inputs = {}
    for name in ("protocol", "inventory", "manifest", "power_plan"):
        ref = authorization["inputs"][name]
        if sha256_file(ref["path"]) != ref["sha256"]:
            raise ValueError(f"Bound {name} bytes changed")
        inputs[name] = read_json(ref["path"])
    gate = assess_data_gate(**inputs, verify_files=True)
    if gate["status"] != "passed" or gate["protocol_hash"] != authorization["protocol_hash"]:
        raise ValueError("Scientific admission gate no longer passes")
    if expected_protocol_hash is not None and expected_protocol_hash != gate["protocol_hash"]:
        raise ValueError("Wrong protocol for model artifact")
    phase = authorization["phase"]
    if phase == "development" and Path(authorization["inputs"]["protocol"]["path"]).with_suffix(".external_opened.json").exists():
        raise ValueError("External testing already opened; development authorization revoked")
    partitions = ["train", "tune", "calibration"] if phase == "development" else ["external"]
    actions = ["train", "infer", "calibrate"] if phase == "development" else ["infer", "evaluate"]
    if authorization.get("allowed_partitions") != partitions or authorization.get("allowed_actions") != actions:
        raise ValueError("Authorization scope was widened")
    rows = [r for r in inputs["manifest"]["patients"] if r["partition"] in partitions]
    if authorization.get("authorized_rows") != {r["patient_id"]: canonical_hash(r) for r in rows}:
        raise ValueError("Authorized patient roster differs from bound inventory")
    if phase == "external_evaluation":
        freeze = authorization.get("selection_freeze", {})
        selection_path = _verified_ref(freeze.get("manifest", {}))
        development = _development_replay_context(freeze.get("development_authorization", {}), authorization, inputs)
        marker = read_json(Path(authorization["inputs"]["protocol"]["path"]).with_suffix(".external_opened.json"))
        expected_marker = {"protocol_hash": authorization["protocol_hash"], "selection_sha256": sha256_file(selection_path),
            "development_authorization_sha256": freeze["development_authorization"]["sha256"],
            "status": "external_opened_development_closed"}
        if marker != expected_marker:
            raise ValueError("External opening marker differs from the frozen protocol/selection")
        verify_selection(read_json(selection_path), authorization["protocol_hash"], development_auth=development)
    verified = {**authorization, "manifest_rows": rows, "protocol": inputs["protocol"]}
    if "final_candidate_selection" in authorization:
        raise ValueError("Final selection must be derived from a verified receipt, not supplied as a flag")
    if "final_candidate_selection_ref" in authorization:
        from .workflow import verify_candidate_selection
        ref = authorization["final_candidate_selection_ref"]
        if sha256_file(ref["path"]) != ref["sha256"]:
            raise ValueError("Candidate selection receipt changed")
        verified["final_candidate_selection"] = verify_candidate_selection(read_json(ref["path"]), verified)
    return verified


def bind_final_selection(*, development_authorization_path, selection_path, output_path):
    auth = verify_training_authorization(development_authorization_path)
    if auth["phase"] != "development":
        raise ValueError("Final seed fitting requires development phase")
    from .workflow import verify_candidate_selection
    verify_candidate_selection(read_json(selection_path), auth)
    result = {k: v for k, v in auth.items() if k not in ("manifest_rows", "protocol", "authorization_sha256", "final_candidate_selection")}
    result["final_candidate_selection_ref"] = _bind(selection_path)
    result["authorization_sha256"] = canonical_hash(result)
    write_json(output_path, result)
    return result


def authorize_rows(authorization, rows, allowed_partitions=None):
    seen = set()
    allowed = set(authorization["allowed_partitions"])
    if allowed_partitions is not None:
        allowed &= set(allowed_partitions)
    for row in rows:
        pid = row.get("patient_id")
        if pid in seen or row.get("partition") not in allowed:
            raise ValueError("Duplicate or unauthorized patient partition")
        if authorization["authorized_rows"].get(pid) != canonical_hash(row):
            raise ValueError("Patient row differs from frozen authorization")
        seen.add(pid)
    return True


def _verified_ref(ref):
    if not isinstance(ref, dict) or not isinstance(ref.get("path"), str) or not ref["path"]:
        raise ValueError("Missing artifact reference")
    path = Path(ref["path"])
    if path.is_symlink() or not path.is_file() or sha256_file(path) != ref.get("sha256"):
        raise ValueError("Missing, symlinked or changed artifact")
    return path.resolve()


def _same_ref(left, right):
    return (_verified_ref(left) == _verified_ref(right) and left["sha256"] == right["sha256"])


def _development_replay_context(ref, external, inputs):
    """Read the original development scope without reopening fitting after freeze."""
    development = read_json(_verified_ref(ref))
    unsigned = {k: v for k, v in development.items() if k != "authorization_sha256"}
    if development.get("authorization_sha256") != canonical_hash(unsigned):
        raise ValueError("Original development authorization content changed")
    required = {"schema_version": "round3_authorization_v1", "phase": "development",
        "protocol_hash": external["protocol_hash"], "inputs": external["inputs"],
        "code_sha256": external["code_sha256"], "allowed_actions": ["train", "infer", "calibrate"],
        "allowed_partitions": ["train", "tune", "calibration"]}
    if any(development.get(k) != v for k, v in required.items()) or "final_candidate_selection" in development:
        raise ValueError("Original development authorization scope mismatch")
    rows = [r for r in inputs["manifest"]["patients"] if r["partition"] in required["allowed_partitions"]]
    if development.get("authorized_rows") != {r["patient_id"]: canonical_hash(r) for r in rows}:
        raise ValueError("Original development patient roster changed")
    return {**development, "manifest_rows": rows, "protocol": inputs["protocol"]}


def verify_selection(selection, protocol_hash, *, development_auth=None):
    """Replay training, calibration and comparator identity; never run inference."""
    from . import workflow, inference
    from .calibration import fit_calibration
    from .statistics import validate_rows
    from .preprocessing import PreprocessingConfig, preflight
    if not development_auth or development_auth.get("phase") != "development" or development_auth.get("protocol_hash") != protocol_hash:
        raise ValueError("Selection replay requires the original verified development authorization")
    if selection.get("protocol_hash") != protocol_hash or selection.get("selection_partition") != "development_only":
        raise ValueError("Selection must be fixed using development data only")
    required = {"new_model_bundle", "old_model_bundle", "recalibrated_old_bundle", "calibration", "preprocessing", "selection_receipt"}
    artifacts = selection.get("artifacts", {})
    if set(artifacts) != required:
        raise ValueError("Selection freeze lacks exact comparator, calibration or preprocessing artifacts")
    paths = {name: _verified_ref(ref) for name, ref in artifacts.items()}
    if selection.get("class_order") != ["CN", "AD"] or selection.get("test_outcomes_seen") is not False:
        raise ValueError("Invalid label order or test-informed selection")
    protocol = development_auth["protocol"]
    if not _same_ref(artifacts["preprocessing"], protocol.get("preprocessing_config_ref", {})):
        raise ValueError("Selected preprocessing differs from the development protocol")
    prep = PreprocessingConfig(**read_json(paths["preprocessing"]))
    if prep.version_hash != protocol["preprocessing_hash"]:
        raise ValueError("Preprocessing configuration hash mismatch")
    if preflight(prep)["status"] != "ready":
        raise ValueError("Pinned preprocessing dependencies are unavailable before external opening")
    receipt = workflow.verify_candidate_selection(read_json(paths["selection_receipt"]), development_auth)
    bound_receipt = development_auth.get("final_candidate_selection_ref", {})
    if not _same_ref(bound_receipt, artifacts["selection_receipt"]):
        raise ValueError("Final training authorization did not bind this selected candidate")
    bundles = {name: read_json(paths[name]) for name in ("new_model_bundle", "old_model_bundle", "recalibrated_old_bundle")}
    new = bundles["new_model_bundle"]
    expected_new = {"version": "round3_model_bundle_v1", "class_order": ["CN", "AD"], "protocol_hash": protocol_hash,
        "preprocessing_hash": protocol["preprocessing_hash"], "model_version": "round3-patient-models-v1",
        "research_only": True, "aggregation": "equal_probability_mean"}
    if any(new.get(k) != v for k, v in expected_new.items()):
        raise ValueError("New model bundle identity/scope mismatch")
    workflow.verify_final_runs(new, development_auth, receipt)
    old, recalibrated = bundles["old_model_bundle"], bundles["recalibrated_old_bundle"]
    expected_old = {"version": "round3_legacy_binary_bundle_v1", "model_id": "round1_clinical_ensemble",
        "protocol_hash": protocol_hash, "class_order": ["CN", "AD"], "research_only": True,
        "aggregation": "mean_slice_probability_then_equal_model_mean"}
    if any(old.get(k) != v for k, v in expected_old.items()) or "calibration" in old:
        raise ValueError("Unchanged old comparator identity/scope mismatch")
    if {k: v for k, v in recalibrated.items() if k != "calibration"} != old:
        raise ValueError("Recalibrated old comparator must retain the exact original model")
    _verified_ref(old.get("preprocessor", {}))
    if old["preprocessor"]["sha256"] != inference.LEGACY_PREPROCESSOR_SHA256:
        raise ValueError("Old preprocessing helper differs from the fixed comparator")
    checkpoints = old.get("checkpoints", [])
    if len(checkpoints) != 3 or sorted(r.get("seed", -1) for r in checkpoints) != sorted(inference.LEGACY_CHECKPOINTS):
        raise ValueError("Old comparator requires the exact three original seeds")
    for ref in checkpoints:
        _verified_ref(ref)
        if ref["sha256"] != inference.LEGACY_CHECKPOINTS[ref["seed"]]:
            raise ValueError("Old comparator checkpoint changed")
    for bundle, key in ((new, "new"), (recalibrated, "old")):
        ref = bundle.get("calibration", {})
        if ref.get("key") != key or not _same_ref(ref, artifacts["calibration"]):
            raise ValueError("Model calibration key/reference mismatch")
    calibration = read_json(paths["calibration"])
    if set(calibration) != {"new", "old", "predictions_ref"}:
        raise ValueError("Require both development fits and their score source")
    payload = read_json(_verified_ref(calibration["predictions_ref"]))
    if not isinstance(payload, dict) or payload.get("protocol_hash") != protocol_hash:
        raise ValueError("Calibration score protocol mismatch")
    predictions = validate_rows(payload.get("rows", []))
    reserved = [r for r in development_auth["manifest_rows"] if r["partition"] == "calibration"]
    roster = {r["patient_id"]: r for r in reserved}
    if len(predictions) != len(roster) or {r["patient_id"] for r in predictions} != set(roster):
        raise ValueError("Calibration requires every reserved patient exactly once")
    for row in predictions:
        if any(row.get(k) != roster[row["patient_id"]][k] for k in ("site_id", "diagnosis", "partition")):
            raise ValueError("Calibration patient identity/diagnosis/partition mismatch")
    expected_provenance = {"new_checkpoint_sha256": [r["sha256"] for r in sorted(new["checkpoints"], key=lambda r: r["seed"])],
        "old_checkpoint_sha256": [r["sha256"] for r in sorted(old["checkpoints"], key=lambda r: r["seed"])],
        "preprocessing_hash": protocol["preprocessing_hash"]}
    if payload.get("model_provenance") != expected_provenance:
        raise ValueError("Calibration scores do not identify the frozen model checkpoints")
    for model, final in (("new", new), ("old", old)):
        raw_ref = payload.get("model_bundles", {}).get(model, {})
        raw = read_json(_verified_ref(raw_ref))
        if "calibration" in raw or raw != {k: v for k, v in final.items() if k != "calibration"}:
            raise ValueError("Calibration scores must derive from the exact uncalibrated model")
    provenance = {"protocol_sha256": protocol_hash, "calibration_manifest_sha256": canonical_hash(reserved),
                  "fold_id": "reserved_development_calibration"}
    for model in ("new", "old"):
        fit = fit_calibration(predictions, role="development_calibration", class_order=["CN", "AD"],
                              score_field=model+"_score", minimum_sensitivity=.90, provenance=provenance)
        if fit != calibration[model] or fit.get("status") != "fitted_development_only" or fit.get("threshold_status") != "available":
            raise ValueError("Calibration/threshold does not replay or is unavailable")
    thresholds = selection.get("thresholds", {})
    expected_thresholds = {"old": .5, "new": calibration["new"]["threshold"], "recalibrated_old": calibration["old"]["threshold"]}
    if (set(thresholds) != set(expected_thresholds) or
            any(type(t) not in (int, float) or not math.isfinite(t) or not 0 <= t <= 1 for t in thresholds.values()) or
            thresholds != expected_thresholds):
        raise ValueError("Selected decision thresholds differ from the fixed old or replayed calibration thresholds")
    return True


def freeze_external(*, development_authorization_path, selection_path, output_path):
    auth = verify_training_authorization(development_authorization_path)
    if auth["phase"] != "development":
        raise ValueError("Expected completed development authorization")
    selection = read_json(selection_path)
    verify_selection(selection, auth["protocol_hash"], development_auth=auth)
    manifest = read_json(auth["inputs"]["manifest"]["path"])
    external = [r for r in manifest["patients"] if r["partition"] == "external"]
    result = {k: v for k, v in auth.items() if k not in ("manifest_rows", "protocol", "authorization_sha256", "final_candidate_selection", "final_candidate_selection_ref")}
    result.update(phase="external_evaluation", allowed_actions=["infer", "evaluate"], allowed_partitions=["external"],
        authorized_rows={r["patient_id"]: canonical_hash(r) for r in external},
        selection_freeze={"manifest": _bind(selection_path), "development_authorization": _bind(development_authorization_path)}, created_utc=datetime.now(timezone.utc).isoformat())
    result["authorization_sha256"] = canonical_hash(result)
    write_json(output_path, result)
    marker = Path(auth["inputs"]["protocol"]["path"]).with_suffix(".external_opened.json")
    write_json(marker, {"protocol_hash": auth["protocol_hash"], "selection_sha256": sha256_file(selection_path),
        "development_authorization_sha256": result["selection_freeze"]["development_authorization"]["sha256"],
        "status": "external_opened_development_closed"})
    return result
