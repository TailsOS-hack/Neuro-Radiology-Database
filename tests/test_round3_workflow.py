"""Synthetic artifact-chain tests. Trainer is replaced; no model fitting occurs."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from research.dementia_round3 import common, training, workflow
from research.dementia_round3.data import digest, sha256_file


def fake_auth():
    rows = []
    for site in ("A", "B"):
        for partition in ("train", "tune", "calibration"):
            for label in (0, 1):
                pid = f"synthetic_{site}_{partition}_{label}"
                rows.append(dict(patient_id=pid, site_id=site, institution_id=site,
                    partition=partition, diagnosis=label, volume_sha256=digest(pid)))
    return {"phase": "development", "allowed_actions": ["train", "infer", "calibrate"],
        "allowed_partitions": ["train", "tune", "calibration"], "manifest_rows": rows,
        "authorized_rows": {r["patient_id"]: digest(r) for r in rows},
        "protocol_hash": "a"*64, "code_sha256": {"synthetic.py": "b"*64},
        "protocol": {"search_space_hash": digest(training.bounded_search_space()), "preprocessing_hash": "c"*64}}


def write_run(rows, config, auth, destination):
    """A tiny, declared synthetic output chain, never a real checkpoint."""
    assert {r["partition"] for r in rows} == {"train", "tune"}
    p = Path(destination);p.mkdir(parents=True)
    candidate = next(c for c in training.bounded_search_space() if c["candidate_id"] == config.candidate_id)
    train_rows, tune_rows = training.development_fold(rows, config)
    strength = .95 if config.candidate_id in ("candidate_02", "candidate_03") else .9
    history = []
    for epoch in range(31):
        confidence = strength if epoch == 0 else strength-.01
        probabilities = [[1-confidence, confidence] if r["diagnosis"] else [confidence, 1-confidence] for r in tune_rows]
        metrics = training.patient_metrics(tune_rows, probabilities)
        history.append({"epoch": epoch, "selected": epoch == 0, "clean": {"tune": metrics}})
        records = [{"patient_id": r["patient_id"], "site_id": r["site_id"], "diagnosis": r["diagnosis"], "probabilities": prob} for r, prob in zip(tune_rows, probabilities)]
        common.write_json(p/f"epoch{epoch:03d}_tune.json", records)
    metrics = history[0]["clean"]["tune"]
    common.write_json(p/"history.json", history)
    state = ({"synthetic_only_tensor": torch.zeros(1)} if config.seed == training.SEARCH_SEED else
             training.build_model(training.ModelConfig(**candidate["model"])).state_dict())
    torch.save({"epoch": 0, "candidate": candidate, "model_config": candidate["model"], "seed": config.seed,
        "validation_site_id": config.validation_site_id, "protocol_hash": auth["protocol_hash"],
        "preprocessing_hash": auth["protocol"]["preprocessing_hash"], "selection_metric": "site_macro_tune_nll",
        "selected_metrics": metrics, "model_version": training.MODEL_VERSION, "training_version": training.TRAINING_VERSION,
        "research_mode": "real_authorized_development", "source_sha256": auth["code_sha256"],
        "model": state}, p/"best.pt")
    selection = {"epoch": 0, "score": metrics["site_macro_nll"], "checkpoint_sha256": sha256_file(p/"best.pt"),
        "selection_partitions": ["train", "tune"] if config.seed == training.SEARCH_SEED else ["tune"],
        "validation_site_id": config.validation_site_id, "selected_metrics": metrics, "protocol_hash": auth["protocol_hash"]}
    common.write_json(p/"selection.json", selection)
    status = {"status": "complete", "stop_reason": "validation_plateau", "completed_epochs": 30,
        "selected_score": metrics["site_macro_nll"], "protocol_hash": auth["protocol_hash"],
        "selection_sha256": sha256_file(p/"selection.json"), "history_sha256": sha256_file(p/"history.json"),
        "checkpoint_sha256": sha256_file(p/"best.pt"), "candidate_id": config.candidate_id, "seed": config.seed,
        "validation_site_id": config.validation_site_id, "selected_metrics": metrics, "train_n": len(train_rows),
        "tune_n": len(tune_rows), "search_space_hash": digest(training.bounded_search_space()),
        "preprocessing_hash": auth["protocol"]["preprocessing_hash"], "heldout_outcomes_used": False}
    common.write_json(p/"status.json", status)
    return status


class Round3WorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.auth = fake_auth()
        with patch.object(common, "verify_training_authorization", return_value=cls.auth), patch.object(training, "train_candidate", side_effect=lambda rows, cfg, authfile, out: write_run(rows, cfg, cls.auth, out)) as trainer:
            cls.index = workflow.run_search("synthetic_authorization", cls.root/"search")
            cls.calls = trainer.call_args_list
            cls.receipt = workflow.select_candidate("synthetic_authorization", cls.root/"search/search_index.json")

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_exact_complete_grid_and_deterministic_metric_tie_selection(self):
        self.assertEqual(len(self.index["runs"]), 48)
        self.assertEqual(len(self.calls), 48)
        self.assertEqual(self.receipt["candidate_id"], "candidate_02")
        self.assertEqual(self.receipt["selection_partitions"], ["train", "tune"])
        self.assertEqual(workflow.verify_candidate_selection(self.receipt, self.auth), self.receipt)
        for call in self.calls:
            self.assertEqual({r["partition"] for r in call.args[0]}, {"train", "tune"})
            self.assertEqual(call.args[1].seed, training.SEARCH_SEED)

    def test_missing_duplicate_extra_grid_and_wrong_seed_rejected(self):
        original = self.root/"search/search_index.json"
        for mutation in ("missing", "duplicate", "extra", "seed"):
            index = copy.deepcopy(self.index)
            if mutation == "missing":index["runs"].pop()
            elif mutation == "duplicate":index["runs"][-1] = index["runs"][0]
            elif mutation == "extra":index["runs"].append(index["runs"][0])
            else:index["seed"] = training.FINAL_SEEDS[0]
            path = original.parent/f"negative_{mutation}.json"
            common.write_json(path, index)
            with self.assertRaises(ValueError):workflow._selection_receipt(path, self.auth)

    def test_receipt_winner_tampering_rejected_even_with_recomputed_receipt_hash(self):
        receipt = copy.deepcopy(self.receipt)
        receipt["candidate_id"] = "candidate_01"
        receipt["selection_sha256"] = digest({k: v for k, v in receipt.items() if k != "selection_sha256"})
        with self.assertRaises(ValueError):workflow.verify_candidate_selection(receipt, self.auth)

    def test_rehashed_patient_label_corruption_is_caught_by_roster(self):
        record = self.index["runs"][0]
        path = self.root/"search"/record["run_directory"]/"epoch000_tune.json"
        original = path.read_bytes()
        records = json.loads(original);records[0]["diagnosis"] = 1-records[0]["diagnosis"]
        path.write_text(json.dumps(records))
        changed = copy.deepcopy(record);changed["files"][path.name] = sha256_file(path)
        rows, _ = workflow._rows_sites(self.auth)
        try:
            with self.assertRaisesRegex(ValueError, "patient"):
                workflow._replay_run(self.root/"search", changed, training.bounded_search_space()[0], "A", rows, self.auth)
        finally:path.write_bytes(original)

    def test_rehashed_nonselected_epoch_probability_change_is_caught(self):
        record = self.index["runs"][0]
        path = self.root/"search"/record["run_directory"]/"epoch001_tune.json"
        original = path.read_bytes()
        records = json.loads(original);records[0]["probabilities"] = [.5, .5]
        path.write_text(json.dumps(records))
        changed = copy.deepcopy(record);changed["files"][path.name] = sha256_file(path)
        rows, _ = workflow._rows_sites(self.auth)
        try:
            with self.assertRaisesRegex(ValueError, "metrics"):
                workflow._replay_run(self.root/"search", changed, training.bounded_search_space()[0], "A", rows, self.auth)
        finally:path.write_bytes(original)

    def test_shared_institution_rejected_before_any_trainer_call(self):
        auth = copy.deepcopy(self.auth)
        for row in auth["manifest_rows"]:row["institution_id"] = "same_institution"
        auth["authorized_rows"] = {r["patient_id"]: digest(r) for r in auth["manifest_rows"]}
        with patch.object(common, "verify_training_authorization", return_value=auth), patch.object(training, "train_candidate") as trainer:
            with self.assertRaisesRegex(ValueError, "share an institution"):
                workflow.run_search("synthetic", self.root/"must_not_exist")
            trainer.assert_not_called()
        self.assertFalse((self.root/"must_not_exist").exists())

    def test_final_three_seeds_require_replayed_candidate_receipt(self):
        auth = copy.deepcopy(self.auth);auth["final_candidate_selection"] = self.receipt
        with patch.object(common, "verify_training_authorization", return_value=auth), patch.object(training, "train_candidate", side_effect=lambda rows, cfg, authfile, out: write_run(rows, cfg, auth, out)) as trainer:
            index = workflow.run_final_seeds("synthetic", self.root/"final")
            self.assertEqual(index["status"], "complete")
            self.assertEqual([r["seed"] for r in index["runs"]], list(training.FINAL_SEEDS))
            self.assertEqual([x.args[1].validation_site_id for x in trainer.call_args_list], ["", "", ""])
        bundle = {"final_index": common._bind(self.root/"final/final_index.json"), "checkpoints": [
            {**common._bind(self.root/"final"/r["run_directory"]/"best.pt"), "seed": r["seed"], "epoch": 0}
            for r in index["runs"]]}
        self.assertEqual(len(workflow.verify_final_runs(bundle, auth, self.receipt)), 3)
        changed = copy.deepcopy(bundle);changed["checkpoints"][0]["epoch"] = 1
        with self.assertRaisesRegex(ValueError, "checkpoint differs"):
            workflow.verify_final_runs(changed, auth, self.receipt)


if __name__ == "__main__":
    unittest.main()
