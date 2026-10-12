"""Bounded development-only LOSO search and independently replayed selection.

No resource allocation occurs here. Actual training delegates to the separately
authorized B300-only trainer. Calibration/external rows never enter a fold.
"""
import math
from pathlib import Path

import torch

from . import common, training
from .data import digest, sha256_file


def _rows_sites(auth):
    if auth.get("phase") != "development" or "train" not in auth.get("allowed_actions", []):
        raise PermissionError("Development training authorization required")
    rows = [r for r in auth["manifest_rows"] if r["partition"] in ("train", "tune")]
    common.authorize_rows(auth, rows, allowed_partitions={"train", "tune"})
    training.validate_partition_rows(rows)
    training.balanced_site_diagnosis_weights(rows)
    sites = sorted({r["site_id"] for r in rows})
    if len(sites) < 2:
        raise ValueError("LOSO requires at least two development institutions")
    institutions = []
    for site in sites:
        inst = {r["institution_id"] for r in rows if r["site_id"] == site}
        if len(inst) != 1:
            raise ValueError("A development site must identify exactly one institution")
        institutions.extend(inst)
    if len(set(institutions)) != len(sites):
        raise ValueError("Multiple site IDs share an institution; normalize institution groups before LOSO")
    if auth["protocol"].get("search_space_hash") != digest(training.bounded_search_space()):
        raise ValueError("The exact 24-configuration search is not frozen")
    return sorted(rows, key=lambda r: r["patient_id"]), sites


def _file(root, relative):
    p = Path(relative)
    if p.is_absolute() or ".." in p.parts:
        raise ValueError("Run inventory must use contained relative paths")
    result = root/p
    if result.is_symlink() or not result.is_file() or root.resolve() not in result.resolve().parents:
        raise ValueError("Missing or escaping search artifact")
    return result


def _record(directory, base, candidate_id, site):
    history = common.read_json(directory/"history.json")
    names = ["status.json", "selection.json", "history.json", "best.pt"]
    names += [f'epoch{r["epoch"]:03d}_tune.json' for r in history]
    return {"candidate_id": candidate_id, "validation_site_id": site,
            "run_directory": directory.relative_to(base).as_posix(),
            "files": {name: sha256_file(directory/name) for name in names}}


def run_search(authfile, output_dir):
    auth = common.verify_training_authorization(authfile)
    rows, sites = _rows_sites(auth)
    destination = Path(output_dir)
    if destination.exists():
        raise ValueError("Search destination must be new; no silent resume/overwrite")
    destination.mkdir(parents=True)
    index = {"schema_version": "round3_search_v1", "status": "running", "protocol_hash": auth["protocol_hash"],
             "search_space_hash": digest(training.bounded_search_space()), "seed": training.SEARCH_SEED,
             "sites": sites, "runs": [], "selection_partitions": ["train", "tune"]}
    common.write_json(destination/"search_index.json", index)
    try:
        for candidate in training.bounded_search_space():
            for i, site in enumerate(sites):
                directory = destination/candidate["candidate_id"]/f"fold_{i:03d}"
                config = training.TrainConfig(candidate["candidate_id"], seed=training.SEARCH_SEED,
                    expected_protocol_hash=auth["protocol_hash"], validation_site_id=site)
                result = training.train_candidate(rows, config, authfile, directory)
                if result.get("status") != "complete":
                    raise ValueError("Search run did not complete on validation plateau")
                index["runs"].append(_record(directory, destination, candidate["candidate_id"], site))
                common.write_json(destination/"search_index.json", index, overwrite=True)
        index["status"] = "complete"
        common.write_json(destination/"search_index.json", index, overwrite=True)
        # Independently replay every run before declaring search selection ready.
        _selection_receipt(destination/"search_index.json", auth)
        return index
    except Exception as exc:
        index.update(status="blocked", failure_type=type(exc).__name__)
        common.write_json(destination/"search_index.json", index, overwrite=True)
        raise


def _same_metrics(actual, expected):
    if isinstance(expected, dict):
        return isinstance(actual, dict) and set(actual) == set(expected) and all(_same_metrics(actual[k], v) for k, v in expected.items())
    if type(expected) in (int, float):
        return type(actual) in (int, float) and math.isfinite(actual) and math.isclose(actual, expected, abs_tol=1e-10, rel_tol=1e-10)
    return actual == expected


def _replay_run(base, record, candidate, site, rows, auth, *, seed=training.SEARCH_SEED):
    directory = base/record["run_directory"]
    if Path(record["run_directory"]).is_absolute() or ".." in Path(record["run_directory"]).parts or directory.is_symlink() or base.resolve() not in directory.resolve().parents:
        raise ValueError("Unsafe search run directory")
    files = record.get("files", {})
    if not isinstance(files, dict):
        raise ValueError("Missing search artifact inventory")
    for name, expected in files.items():
        if sha256_file(_file(directory, name)) != expected:
            raise ValueError("Search artifact changed")
    required = {"status.json", "history.json", "selection.json", "best.pt"}
    if not required <= set(files):
        raise ValueError("Missing checkpoint/status/history/selection coverage")
    status, history, selected = [common.read_json(directory/n) for n in ("status.json", "history.json", "selection.json")]
    config = training.TrainConfig(candidate["candidate_id"], seed=seed,
                                 expected_protocol_hash=auth["protocol_hash"], validation_site_id=site)
    train_rows, tune_rows = training.development_fold(rows, config)
    expected_status = {"status": "complete", "stop_reason": "validation_plateau", "protocol_hash": auth["protocol_hash"],
                       "candidate_id": candidate["candidate_id"], "seed": seed, "validation_site_id": site,
                       "heldout_outcomes_used": False, "train_n": len(train_rows), "tune_n": len(tune_rows),
                       "search_space_hash": digest(training.bounded_search_space()), "preprocessing_hash": auth["protocol"]["preprocessing_hash"]}
    if any(status.get(k) != v for k, v in expected_status.items()):
        raise ValueError("Search run status/identity/scope mismatch")
    for name, field in [("best.pt", "checkpoint_sha256"), ("history.json", "history_sha256"), ("selection.json", "selection_sha256")]:
        if status.get(field) != files[name]:
            raise ValueError("Run status does not pin its artifacts")
    if not isinstance(history, list) or not history or [r.get("epoch") for r in history] != list(range(len(history))):
        raise ValueError("Training history is missing/reordered epochs")
    last_epoch = len(history)-1
    if last_epoch != status.get("completed_epochs") or not config.min_epochs <= last_epoch <= config.max_epochs:
        raise ValueError("Training completion epoch mismatch")
    expected_files = required | {f"epoch{e:03d}_tune.json" for e in range(last_epoch+1)}
    if set(files) != expected_files:
        raise ValueError("Incomplete or unexpected epoch prediction inventory")
    roster = {r["patient_id"]: r for r in tune_rows}
    best, stale, chosen, selected_metrics = math.inf, 0, None, None
    for epoch, entry in enumerate(history):
        prediction_rows = common.read_json(directory/f"epoch{epoch:03d}_tune.json")
        if not isinstance(prediction_rows, list) or len(prediction_rows) != len(roster):
            raise ValueError("Missing held-out development patient predictions")
        by_id = {}
        for r in prediction_rows:
            pid = r.get("patient_id")
            if pid in by_id or pid not in roster or type(r.get("diagnosis")) is not int or any(r.get(k) != roster[pid][k] for k in ("site_id", "diagnosis")):
                raise ValueError("Mismatched/duplicated held-out development patient")
            by_id[pid] = r
        if set(by_id) != set(roster):
            raise ValueError("Held-out development roster mismatch")
        metrics = training.patient_metrics(tune_rows, [by_id[r["patient_id"]]["probabilities"] for r in tune_rows])
        if not _same_metrics(entry.get("clean", {}).get("tune"), metrics):
            raise ValueError("Stored validation metrics do not reproduce from predictions")
        score = metrics["site_macro_nll"]
        improved = score < best-config.min_delta
        if entry.get("selected") is not improved:
            raise ValueError("Checkpoint selection flags disagree with the frozen min_delta rule")
        stale = 0 if improved else stale+1
        if improved:
            best, chosen, selected_metrics = score, epoch, metrics
        if epoch >= config.min_epochs and stale >= config.patience and epoch != last_epoch:
            raise ValueError("Search continued after its prescribed validation plateau")
    if stale < config.patience:
        raise ValueError("Complete run did not meet validation plateau")
    expected_selection = {"epoch": chosen, "score": best, "checkpoint_sha256": files["best.pt"],
                          "selection_partitions": ["train", "tune"] if seed == training.SEARCH_SEED else ["tune"], "validation_site_id": site,
                          "selected_metrics": selected_metrics, "protocol_hash": auth["protocol_hash"]}
    if not _same_metrics(selected, expected_selection) or not _same_metrics(status.get("selected_metrics"), selected_metrics) or not math.isclose(status.get("selected_score", math.inf), best, rel_tol=1e-10, abs_tol=1e-10):
        raise ValueError("Selected epoch/score/metrics mismatch")
    checkpoint = torch.load(directory/"best.pt", map_location="cpu", weights_only=True)
    expected_checkpoint = {"epoch": chosen, "candidate": candidate, "model_config": candidate["model"], "seed": seed,
                           "validation_site_id": site, "protocol_hash": auth["protocol_hash"],
                           "preprocessing_hash": auth["protocol"]["preprocessing_hash"],
                           "selection_metric": "site_macro_tune_nll", "selected_metrics": selected_metrics,
                           "model_version": training.MODEL_VERSION, "training_version": training.TRAINING_VERSION,
                           "research_mode": "real_authorized_development", "source_sha256": auth["code_sha256"]}
    if any(digest(checkpoint.get(k)) != digest(v) for k, v in expected_checkpoint.items()) or not isinstance(checkpoint.get("model"), dict) or not checkpoint["model"]:
        raise ValueError("Checkpoint metadata differs from replayed selection")
    return {"candidate_id": candidate["candidate_id"], "validation_site_id": site, "selected_epoch": chosen,
            "site_macro_balanced_accuracy": selected_metrics["site_macro_balanced_accuracy"],
            "site_macro_nll": selected_metrics["site_macro_nll"], "run_directory": record["run_directory"],
            "files": files}


def _selection_receipt(search_index_path, auth):
    rows, sites = _rows_sites(auth)
    path = Path(search_index_path).resolve(strict=True)
    if Path(search_index_path).is_symlink():
        raise ValueError("Search index must not be a symlink")
    index = common.read_json(path)
    expected_header = {"schema_version": "round3_search_v1", "status": "complete", "protocol_hash": auth["protocol_hash"],
                       "search_space_hash": digest(training.bounded_search_space()), "seed": training.SEARCH_SEED,
                       "sites": sites, "selection_partitions": ["train", "tune"]}
    if any(index.get(k) != v for k, v in expected_header.items()):
        raise ValueError("Search index scope/status mismatch")
    candidates = training.bounded_search_space()
    expected = {(c["candidate_id"], site) for c in candidates for site in sites}
    records = index.get("runs", [])
    found = [(r.get("candidate_id"), r.get("validation_site_id")) for r in records]
    if len(records) != len(expected) or len(set(found)) != len(found) or set(found) != expected:
        raise ValueError("Require the complete exact 24-by-site search grid")
    if len({r["run_directory"] for r in records}) != len(records):
        raise ValueError("Different folds cannot share a run directory")
    indexed = {key: r for key, r in zip(found, records)}
    folds, scores = [], []
    for candidate in candidates:
        values = [_replay_run(path.parent, indexed[(candidate["candidate_id"], site)], candidate, site, rows, auth) for site in sites]
        folds.extend(values)
        scores.append({"candidate_id": candidate["candidate_id"],
                       "equal_site_balanced_accuracy": sum(v["site_macro_balanced_accuracy"] for v in values)/len(values),
                       "equal_site_nll": sum(v["site_macro_nll"] for v in values)/len(values)})
    winner = min(scores, key=lambda r: (-r["equal_site_balanced_accuracy"], r["equal_site_nll"], r["candidate_id"]))
    receipt = {"schema_version": "round3_candidate_selection_v1", "candidate_id": winner["candidate_id"],
               "selection_partitions": ["train", "tune"], "protocol_hash": auth["protocol_hash"],
               "search_index_ref": {"path": str(path), "sha256": sha256_file(path)}, "fold_refs": folds,
               "candidate_scores": scores, "selection_rule": "highest equal-site balanced accuracy, then lowest equal-site NLL, then candidate ID",
               "all_validation_predictions_replayed": True, "external_or_calibration_outcomes_used": False}
    receipt["selection_sha256"] = digest(receipt)
    return receipt


def select_candidate(authfile, search_index_path):
    return _selection_receipt(search_index_path, common.verify_training_authorization(authfile))


def verify_candidate_selection(receipt, auth):
    if receipt.get("selection_sha256") != digest({k: v for k, v in receipt.items() if k != "selection_sha256"}):
        raise ValueError("Candidate selection receipt content changed")
    ref = receipt.get("search_index_ref", {})
    if not ref or sha256_file(ref["path"]) != ref.get("sha256"):
        raise ValueError("Candidate search index changed")
    recomputed = _selection_receipt(ref["path"], auth)
    if recomputed != receipt:
        raise ValueError("Candidate selection does not independently reproduce")
    return recomputed


def verify_final_runs(bundle, auth, candidate_selection):
    """Replay all final-seed artifacts without constructing a model/inference."""
    rows, _ = _rows_sites(auth)
    ref = bundle.get("final_index", {})
    if not ref or Path(ref["path"]).is_symlink() or sha256_file(ref["path"]) != ref.get("sha256"):
        raise ValueError("Final model bundle requires its unchanged complete final index")
    path = Path(ref["path"]).resolve()
    index = common.read_json(path)
    expected = {"schema_version": "round3_final_runs_v1", "status": "complete", "protocol_hash": auth["protocol_hash"],
                "candidate_id": candidate_selection["candidate_id"], "selection_sha256": candidate_selection["selection_sha256"]}
    if any(index.get(k) != v for k, v in expected.items()):
        raise ValueError("Final index scope/candidate/selection mismatch")
    records, refs = index.get("runs", []), bundle.get("checkpoints", [])
    seeds = list(training.FINAL_SEEDS)
    if len(records) != 3 or sorted(r.get("seed", -1) for r in records) != seeds or len(refs) != 3 or sorted(r.get("seed", -1) for r in refs) != seeds:
        raise ValueError("Require all three fixed final seeds exactly once")
    if len({r["run_directory"] for r in records}) != 3:
        raise ValueError("Final seeds cannot share a directory")
    candidate = next(c for c in training.bounded_search_space() if c["candidate_id"] == candidate_selection["candidate_id"])
    verified = []
    for seed in seeds:
        record = next(r for r in records if r["seed"] == seed)
        checkpoint = next(r for r in refs if r["seed"] == seed)
        if record.get("candidate_id") != candidate["candidate_id"] or record.get("validation_site_id") != "":
            raise ValueError("Final seed retained a search fold or wrong candidate")
        replay = _replay_run(path.parent, record, candidate, "", rows, auth, seed=seed)
        expected_path = (path.parent/record["run_directory"]/"best.pt").resolve()
        if Path(checkpoint["path"]).resolve() != expected_path or checkpoint.get("sha256") != record["files"]["best.pt"] or checkpoint.get("epoch") != replay["selected_epoch"]:
            raise ValueError("Model bundle checkpoint differs from its final run")
        payload = torch.load(expected_path, map_location="cpu", weights_only=True)
        state = payload["model"]
        if any(not isinstance(value, torch.Tensor) or not torch.isfinite(value).all().item() for value in state.values()):
            raise ValueError("Final checkpoint contains malformed or nonfinite model weights")
        model = training.build_model(training.ModelConfig(**candidate["model"]))
        try:
            model.load_state_dict(state, strict=True)
        except (RuntimeError, TypeError) as exc:
            raise ValueError("Final checkpoint weights do not match the selected architecture") from exc
        del model, payload
        verified.append(replay)
    return verified


def run_final_seeds(authfile, output_dir):
    auth = common.verify_training_authorization(authfile)
    rows, _ = _rows_sites(auth)
    selection = verify_candidate_selection(auth.get("final_candidate_selection", {}), auth)
    destination = Path(output_dir)
    if destination.exists():
        raise ValueError("Final-seed destination must be new")
    destination.mkdir(parents=True)
    index = {"schema_version": "round3_final_runs_v1", "status": "running", "protocol_hash": auth["protocol_hash"],
             "candidate_id": selection["candidate_id"], "selection_sha256": selection["selection_sha256"], "runs": []}
    common.write_json(destination/"final_index.json", index)
    try:
        for seed in training.FINAL_SEEDS:
            directory = destination/f"seed{seed}"
            config = training.TrainConfig(selection["candidate_id"], seed=seed, expected_protocol_hash=auth["protocol_hash"])
            status = training.train_candidate(rows, config, authfile, directory)
            if status.get("status") != "complete":
                raise ValueError("Final run did not complete on validation plateau")
            index["runs"].append({"seed": seed, **_record(directory, destination, selection["candidate_id"], "")})
            common.write_json(destination/"final_index.json", index, overwrite=True)
        index["status"] = "complete"
        common.write_json(destination/"final_index.json", index, overwrite=True)
        return index
    except Exception as exc:
        index.update(status="blocked", failure_type=type(exc).__name__)
        common.write_json(destination/"final_index.json", index, overwrite=True)
        raise
