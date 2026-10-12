"""Round 3 research commands. No command provisions cloud resources."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import platform
from pathlib import Path
import sys

from . import common
from .data import match_assessments

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs/kevin_paper_20261011/round3"


def emit(value):
    print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))


def _local_analysis():
    if platform.system() != "Darwin":
        raise ValueError("Run analysis on the user's local Mac; the B300 is reserved for training")


def _preprocessing_config(path):
    from .preprocessing import PreprocessingConfig
    return PreprocessingConfig(**common.read_json(path))


def _paths(parser):
    parser.add_argument("--protocol", type=Path, default=DOCS / "protocol.json")
    parser.add_argument("--inventory", type=Path, default=DOCS / "inventory.json")
    parser.add_argument("--manifest", type=Path, default=DOCS / "empty_patient_manifest.json")
    parser.add_argument("--power-plan", type=Path, default=DOCS / "sample_size_pending.json")


def _gate(args):
    return common.assess_data_gate(common.read_json(args.protocol), common.read_json(args.inventory),
        common.read_json(args.manifest), common.read_json(args.power_plan), verify_files=not args.skip_file_checks)


def _verify_predictions(rows, auth, partition):
    expected = {r["patient_id"]: r for r in auth["manifest_rows"] if r["partition"] == partition}
    if len(rows) != len(expected) or {r.get("patient_id") for r in rows} != set(expected):
        raise ValueError("Prediction roster must contain exactly every authorized eligible patient, including failures")
    for row in rows:
        ref = expected[row["patient_id"]]
        if any(row.get(k) != ref[k] for k in ("site_id", "diagnosis", "partition")):
            raise ValueError("Prediction identity/label/partition differs from frozen manifest")
    return expected


def _verify_prediction_models(payload, selection):
    if payload.get("protocol_hash") != selection["protocol_hash"]:
        raise ValueError("Prediction protocol differs from selection")
    expected = {}
    for model in ("old", "new"):
        frozen = selection["artifacts"][model+"_model_bundle"]
        if not common._same_ref(payload.get("model_bundles", {}).get(model, {}), frozen):
            raise ValueError("Prediction model bundle differs from the frozen comparator")
        bundle = common.read_json(frozen["path"])
        expected[model+"_checkpoint_sha256"] = [r["sha256"] for r in sorted(bundle["checkpoints"], key=lambda r: r["seed"])]
        if model == "new":
            expected["preprocessing_hash"] = bundle["preprocessing_hash"]
    if payload.get("model_provenance") != expected:
        raise ValueError("Prediction checkpoint/preprocessing provenance mismatch")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("status", help="Report the current data gate and dependency preflight; never starts training")
    _paths(p)
    p.add_argument("--skip-file-checks", action="store_true", help="Metadata-only audit; cannot authorize training")
    p.add_argument("--output", type=Path)
    p = sub.add_parser("match", help="Match PRIVATE normalized provider scans and assessments")
    p.add_argument("--scans", type=Path, required=True)
    p.add_argument("--assessments", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("plan-sample-size", help="Simulate an explicitly assumed design, never infer an approved sample size")
    p.add_argument("--assumptions", type=Path, required=True)
    p.add_argument("--simulations", type=int, default=1000)
    p.add_argument("--bootstrap-replicates", type=int, default=10000)
    p.add_argument("--seed", type=int, default=20261031)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("freeze-development", help="Authorize fitting only after replaying the data and power gates")
    _paths(p)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("calibrate", help="Fit intercept/slope and threshold on reserved development patients")
    p.add_argument("--authorization", type=Path, required=True)
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--model", choices=("old", "new", "both"), default="both")
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("freeze-external", help="Freeze all pipelines and permanently close development for this protocol")
    p.add_argument("--authorization", type=Path, required=True)
    p.add_argument("--selection", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("evaluate", help="Analyze every eligible frozen external patient on the local computer")
    p.add_argument("--authorization", type=Path, required=True)
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("audit-previous", help="Read saved Round 2 mechanics/distributions without rerunning inference")
    p.add_argument("--root", type=Path, default=ROOT)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("search-manifest", help="Print the exact 24 configurations without any fitting")
    p.add_argument("--output", type=Path)
    for name in ("run-search", "run-final-seeds"):
        p = sub.add_parser(name, help="Run previously authorized training on the existing B300 only")
        p.add_argument("--authorization", type=Path, required=True)
        p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("select-candidate", help="Replay complete development folds and freeze the candidate")
    p.add_argument("--authorization", type=Path, required=True)
    p.add_argument("--search-index", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("bind-final-selection", help="Bind verified candidate selection before the three final seeds")
    p.add_argument("--authorization", type=Path, required=True)
    p.add_argument("--selection", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("prepare", help="Prepare admitted raw volumes before the final development authorization; blinded review remains required")
    _paths(p)
    p.add_argument("--preprocessing-config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("predict-new", help="Predict every patient in one authorized partition, preserving abstentions")
    p.add_argument("--authorization", type=Path, required=True)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--preprocessing-config", type=Path, required=True)
    p.add_argument("--partition", choices=("calibration", "external"), required=True)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("predict-paired", help="Run frozen old/new models on the identical full eligible patient roster")
    p.add_argument("--authorization", type=Path, required=True)
    p.add_argument("--new-bundle", type=Path, required=True)
    p.add_argument("--old-bundle", type=Path, required=True)
    p.add_argument("--preprocessing-config", type=Path, required=True)
    p.add_argument("--partition", choices=("calibration", "external"), required=True)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("report", help="Reproduce frozen statistics and render raw final-seed curves on the local Mac")
    p.add_argument("--authorization", type=Path, required=True)
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        if args.command == "status":
            from .preprocessing import PreprocessingConfig, preflight
            result = {"data_gate": _gate(args), "preprocessing": preflight(PreprocessingConfig()),
                      "paid_resources_started": False, "real_training_performed": False, "analysis_location": "local_mac"}
            if args.output:
                common.write_json(args.output, result)
        elif args.command == "match":
            scans, assessments = common.read_json(args.scans), common.read_json(args.assessments)
            primary = match_assessments(scans, assessments)
            sensitivity = match_assessments(scans, assessments, maximum_days=90)
            result = {"schema_version": "round3_patient_manifest_v1", "patients": primary["primary"],
                      "repeatability_scans": primary["repeatability"], "exclusions": primary["exclusions"],
                      "sensitivity90": sensitivity,
                      "scan_assessment_evidence": {"scans": common._bind(args.scans), "assessments": common._bind(args.assessments), "matching_sha256": common.canonical_hash(primary)},
                      "cross_source_identity_review": {"status": "pending"}, "provider_mapping_review": {"status": "pending"}}
            common.write_json(args.output, result)
            result = {"status": "matched_not_authorized", "primary": primary["counts"], "sensitivity90": sensitivity["counts"], "private_output_written": True}
        elif args.command == "plan-sample-size":
            _local_analysis()
            from .sample_size import simulate_plan
            result = simulate_plan(common.read_json(args.assumptions), simulations=args.simulations, bootstrap_replicates=args.bootstrap_replicates, seed=args.seed)
            result["design_review"] = {"status": "pending"}
            common.write_json(args.output, result)
        elif args.command == "freeze-development":
            result = common.freeze_development(protocol_path=args.protocol, inventory_path=args.inventory, manifest_path=args.manifest, power_plan_path=args.power_plan, output_path=args.output)
            result = {"status": "development_authorized", "protocol_hash": result["protocol_hash"]}
        elif args.command == "calibrate":
            _local_analysis()
            from .calibration import fit_calibration
            auth = common.verify_training_authorization(args.authorization)
            if "calibrate" not in auth["allowed_actions"]:
                raise ValueError("External outcomes cannot fit calibration")
            payload = common.read_json(args.predictions)
            rows = payload["rows"] if isinstance(payload, dict) else payload
            _verify_predictions(rows, auth, "calibration")
            fits = {model: fit_calibration(rows, role="development_calibration", class_order=["CN", "AD"], score_field=model + "_score",
                provenance={"protocol_sha256": auth["protocol_hash"], "calibration_manifest_sha256": common.canonical_hash([r for r in auth["manifest_rows"] if r["partition"] == "calibration"]), "fold_id": "reserved_development_calibration"})
                for model in (("old", "new") if args.model == "both" else (args.model,))}
            result = {**fits, "predictions_ref": common._bind(args.predictions)} if args.model == "both" else fits[args.model]
            common.write_json(args.output, result)
        elif args.command == "freeze-external":
            result = common.freeze_external(development_authorization_path=args.authorization, selection_path=args.selection, output_path=args.output)
            result = {"status": "external_frozen_development_closed", "protocol_hash": result["protocol_hash"]}
        elif args.command in ("evaluate", "report"):
            _local_analysis()
            from .statistics import analyze
            auth = common.verify_training_authorization(args.authorization)
            if "evaluate" not in auth["allowed_actions"] or auth["phase"] != "external_evaluation":
                raise ValueError("External evaluation needs its separate freeze")
            payload = common.read_json(args.predictions)
            if payload.get("selection_manifest_sha256") != auth["selection_freeze"]["manifest"]["sha256"]:
                raise ValueError("Predictions belong to a different model-selection freeze")
            rows = payload["rows"]
            _verify_predictions(rows, auth, "external")
            selection = common.read_json(auth["selection_freeze"]["manifest"]["path"])
            _verify_prediction_models(payload, selection)
            result = analyze(rows, old_threshold=selection["thresholds"]["old"], new_threshold=selection["thresholds"]["new"], bootstrap_replicates=auth["protocol"]["statistics"]["bootstrap_replicates"], seed=auth["protocol"]["statistics"]["seed"])
            result.update(protocol_hash=auth["protocol_hash"], input_predictions_sha256=common.sha256_file(args.predictions))
            if args.command == "report":
                from .report import render_report
                new_bundle = common.read_json(selection["artifacts"]["new_model_bundle"]["path"])
                ref = new_bundle["final_index"]
                if common.sha256_file(ref["path"]) != ref["sha256"]:
                    raise ValueError("Final seed index changed")
                index = common.read_json(ref["path"])
                histories = [Path(ref["path"]).parent/r["run_directory"]/"history.json" for r in index["runs"]]
                result = render_report(rows, result, histories, args.output)
            else:
                common.write_json(args.output, result)
        elif args.command == "audit-previous":
            from .audit_previous import audit
            result = audit(args.root)
            common.write_json(args.output, result)
        elif args.command == "search-manifest":
            from .training import bounded_search_space
            configurations = bounded_search_space()
            result = {"configurations": configurations, "search_space_hash": common.canonical_hash(configurations), "count": len(configurations), "training_performed": False}
            if args.output:
                common.write_json(args.output, result)
        elif args.command in ("run-search", "run-final-seeds"):
            from .workflow import run_search, run_final_seeds
            result = (run_search if args.command == "run-search" else run_final_seeds)(args.authorization, args.output)
        elif args.command == "select-candidate":
            from .workflow import select_candidate
            result = select_candidate(args.authorization, args.search_index)
            common.write_json(args.output, result)
        elif args.command == "bind-final-selection":
            result = common.bind_final_selection(development_authorization_path=args.authorization, selection_path=args.selection, output_path=args.output)
            result = {"status": "final_seed_selection_bound", "protocol_hash": result["protocol_hash"]}
        elif args.command == "prepare":
            from .preprocessing import preprocess_volume, preflight
            protocol = common.read_json(args.protocol)
            inventory = common.read_json(args.inventory)
            manifest = common.read_json(args.manifest)
            gate = common.assess_data_gate(protocol, inventory, manifest, common.read_json(args.power_plan))
            if gate["status"] != "passed":
                raise ValueError("Cannot prepare an unadmitted patient inventory: " + ", ".join(gate["blockers"]))
            config = _preprocessing_config(args.preprocessing_config)
            if protocol.get("preprocessing_hash") != config.version_hash or preflight(config)["status"] != "ready":
                raise ValueError("Preprocessing tools/assets must match the execution protocol")
            if args.output.exists():
                raise ValueError("Preserve prior processing; output directory must be new")
            args.output.mkdir(parents=True)
            updated, counts = [], {}
            for index, row in enumerate(manifest["patients"]):
                directory = args.output / f"volume_{index:06d}"
                # Processing and review templates are deliberately diagnosis-blind.
                inputs = {k: row[k] for k in ("volume_path", "volume_sha256", "t1_metadata") if k in row}
                record = preprocess_volume(inputs, directory, config)
                ref = common._bind(directory / "preprocessing.json")
                review_path = directory / "blinded_review.json"
                common.write_json(review_path, {"status": "pending", "diagnosis_blinded": True,
                    "prediction_blinded": True, "preprocessing_record_sha256": ref["sha256"]})
                updated.append({**row, "preprocessing_record_path": ref["path"],
                    "preprocessing_record_sha256": ref["sha256"], "preprocessing_review_path": str(review_path.resolve())})
                counts[record["status"]] = counts.get(record["status"], 0) + 1
            common.write_json(args.output / "manifest_pending_review.json", {**manifest, "patients": updated})
            result = {"status": "prepared_pending_blinded_review", "counts": counts,
                      "instructions": "Review anatomy without labels/predictions, record actual outcomes and reviewer, then bind each review SHA256 before freeze-development. Retain eligible external failures in the denominator."}
        elif args.command == "predict-new":
            _local_analysis()
            from .inference import InferenceSession
            auth = common.verify_training_authorization(args.authorization)
            if args.partition not in auth["allowed_partitions"] or "infer" not in auth["allowed_actions"]:
                raise ValueError("Partition is not authorized for predictions")
            if args.output.exists():
                raise ValueError("Prediction output already exists; preserve the original predictions")
            config = _preprocessing_config(args.preprocessing_config)
            rows = [r for r in auth["manifest_rows"] if r["partition"] == args.partition]
            if not rows:
                raise ValueError("No authorized patients in this partition")
            predictions = []
            with InferenceSession(rows, args.authorization, config, new_bundle_path=args.bundle) as batch:
                for row in rows:
                    value = batch.predict(row)["new"]
                    predictions.append({**{k: row[k] for k in ("patient_id", "site_id", "partition", "diagnosis")},
                        "new_score": value["score"], "new_status": {"ok": "ok", "failed": "failed", "abstained": "abstain"}[value["qc_status"]],
                        "inference": value})
            common.write_json(args.output, {"protocol_hash": auth["protocol_hash"], "model_bundle": common._bind(args.bundle),
                "selection_manifest_sha256": auth.get("selection_freeze", {}).get("manifest", {}).get("sha256"), "rows": predictions})
            result = {"status": "predictions_written", "eligible_patients": len(rows), "research_only": True}
        elif args.command == "predict-paired":
            _local_analysis()
            from .inference import InferenceSession
            auth = common.verify_training_authorization(args.authorization)
            if args.partition not in auth["allowed_partitions"] or "infer" not in auth["allowed_actions"]:
                raise ValueError("Partition is not authorized for predictions")
            if args.output.exists():
                raise ValueError("Preserve the existing predictions; choose a new destination")
            config = _preprocessing_config(args.preprocessing_config)
            bundles = {"new": common.read_json(args.new_bundle), "old": common.read_json(args.old_bundle)}
            if args.partition == "calibration" and any(b.get("calibration") for b in bundles.values()):
                raise ValueError("Calibration fitting must receive raw model scores")
            rows = [r for r in auth["manifest_rows"] if r["partition"] == args.partition]
            if not rows:
                raise ValueError("No eligible patients in this partition")
            predictions = []
            with InferenceSession(rows, args.authorization, config, new_bundle_path=args.new_bundle,
                                  old_bundle_path=args.old_bundle) as batch:
                for row in rows:
                    values = batch.predict(row)
                    output = {k: row[k] for k in ("patient_id", "site_id", "partition", "diagnosis")}
                    for model, value in values.items():
                        output[model+"_score"] = value["score"]
                        output[model+"_status"] = {"ok": "ok", "abstained": "abstain", "failed": "failed"}[value["qc_status"]]
                        output[model+"_inference"] = value
                    predictions.append(output)
            provenance = {model+"_checkpoint_sha256": [r["sha256"] for r in sorted(bundle["checkpoints"], key=lambda x: x.get("seed", x.get("model_id", "")))] for model, bundle in bundles.items()}
            provenance["preprocessing_hash"] = config.version_hash
            common.write_json(args.output, {"protocol_hash": auth["protocol_hash"], "model_provenance": provenance,
                "selection_manifest_sha256": auth.get("selection_freeze", {}).get("manifest", {}).get("sha256"),
                "model_bundles": {"new": common._bind(args.new_bundle), "old": common._bind(args.old_bundle)}, "rows": predictions})
            result = {"status": "paired_predictions_written", "eligible_patients": len(rows), "research_only": True}
        emit(result)
        return 0
    except (ValueError, OSError, KeyError, ImportError, TypeError) as exc:
        print(json.dumps({"status": "blocked", "error": str(exc)}, allow_nan=False), file=sys.stderr)
        return 2
