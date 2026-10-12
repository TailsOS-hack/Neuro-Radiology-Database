"""Synthetic external-freeze contracts; no images/models are inferred or trained."""
import copy
from contextlib import ExitStack
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from research.dementia_round3 import common, calibration, inference, workflow, training, preprocessing
from research.dementia_round3.data import digest


class Round3FreezeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.stack = ExitStack();self.addCleanup(self.stack.close)
        self.preflight = self.stack.enter_context(patch.object(preprocessing, "preflight", return_value={"status": "ready"}))
        self.receipt = {"candidate_id": "candidate_01", "selection_sha256": "b"*64}
        self.replay = self.stack.enter_context(patch.object(workflow, "verify_candidate_selection", return_value=self.receipt))
        self.finals = self.stack.enter_context(patch.object(workflow, "verify_final_runs", return_value=[{}, {}, {}]))
        def put(name, obj):
            path = self.base/name;common.write_json(path, obj, overwrite=True);return common._bind(path)
        self.put = put
        prep_ref = put("preprocessing.json", {})
        prep_hash = preprocessing.PreprocessingConfig().version_hash
        self.protocol = {"preprocessing_hash": prep_hash, "preprocessing_config_ref": prep_ref,
                         "search_space_hash": digest(training.bounded_search_space())}
        self.protocol_hash = digest(self.protocol)
        self.old_prep = self.base/"old_preprocessor.py";self.old_prep.write_text("# synthetic")
        new_ckpts, old_ckpts = [], []
        for label, seeds, refs in (("new", training.FINAL_SEEDS, new_ckpts), ("old", (20261011, 20261012, 20261013), old_ckpts)):
            for seed in seeds:
                p = self.base/f"{label}_{seed}.pt";p.write_bytes(f"synthetic-{label}-{seed}".encode())
                refs.append({**common._bind(p), "seed": seed, **({"epoch": 0} if label == "new" else {})})
        self.stack.enter_context(patch.object(inference, "LEGACY_PREPROCESSOR_SHA256", common.sha256_file(self.old_prep)))
        self.stack.enter_context(patch.object(inference, "LEGACY_CHECKPOINTS", {r["seed"]: r["sha256"] for r in old_ckpts}))
        new = dict(version="round3_model_bundle_v1", class_order=["CN", "AD"], protocol_hash=self.protocol_hash,
            preprocessing_hash=prep_hash, model_version="round3-patient-models-v1", research_only=True,
            aggregation="equal_probability_mean", checkpoints=new_ckpts, final_index=put("final_index.json", {"synthetic": True}))
        old = dict(version="round3_legacy_binary_bundle_v1", model_id="round1_clinical_ensemble", protocol_hash=self.protocol_hash,
            class_order=["CN", "AD"], research_only=True, aggregation="mean_slice_probability_then_equal_model_mean",
            preprocessor=common._bind(self.old_prep), checkpoints=old_ckpts)
        rows = []
        for i, p in enumerate((.1, .3, .5, .7, .9)):
            for j in range(20):
                rows.append(dict(patient_id=f"synthetic_{i}_{j}", site_id="development", partition="calibration",
                    diagnosis=int(j < int(p*20)), old_score=p, new_score=p, old_status="ok", new_status="ok"))
        self.rows = rows
        self.auth = {"phase": "development", "protocol_hash": self.protocol_hash, "protocol": self.protocol,
            "manifest_rows": [{k: r[k] for k in ("patient_id", "site_id", "partition", "diagnosis")} for r in rows],
            "final_candidate_selection_ref": put("candidate_selection.json", self.receipt)}
        payload = {"protocol_hash": self.protocol_hash, "rows": rows,
            "model_provenance": {"new_checkpoint_sha256": [r["sha256"] for r in new_ckpts], "old_checkpoint_sha256": [r["sha256"] for r in old_ckpts], "preprocessing_hash": prep_hash},
            "model_bundles": {"new": put("raw_new.json", new), "old": put("raw_old.json", old)}}
        self.payload = payload
        provenance = dict(protocol_sha256=self.protocol_hash, calibration_manifest_sha256=digest(self.auth["manifest_rows"]), fold_id="reserved_development_calibration")
        fits = {m: calibration.fit_calibration(rows, role="development_calibration", class_order=["CN", "AD"], score_field=m+"_score", provenance=provenance) for m in ("new", "old")}
        fits["predictions_ref"] = put("predictions.json", payload)
        self.fits = fits
        calibration_ref = put("calibration.json", fits)
        new["calibration"] = {**calibration_ref, "key": "new"}
        recalibrated = {**old, "calibration": {**calibration_ref, "key": "old"}}
        self.selection = {"protocol_hash": self.protocol_hash, "selection_partition": "development_only", "class_order": ["CN", "AD"], "test_outcomes_seen": False,
            "thresholds": {"old": .5, "new": fits["new"]["threshold"], "recalibrated_old": fits["old"]["threshold"]},
            "artifacts": {"new_model_bundle": put("new.json", new), "old_model_bundle": put("old.json", old), "recalibrated_old_bundle": put("recalibrated.json", recalibrated),
                "calibration": calibration_ref, "preprocessing": prep_ref, "selection_receipt": self.auth["final_candidate_selection_ref"]}}

    def verify(self):
        return common.verify_selection(self.selection, self.protocol_hash, development_auth=self.auth)

    def replace_calibration_payload(self):
        self.fits["predictions_ref"] = self.put("predictions.json", self.payload)
        ref = self.put("calibration.json", self.fits)
        self.selection["artifacts"]["calibration"] = ref
        for name, key in (("new_model_bundle", "new"), ("recalibrated_old_bundle", "old")):
            obj = common.read_json(self.selection["artifacts"][name]["path"])
            obj["calibration"] = {**ref, "key": key}
            self.selection["artifacts"][name] = self.put(name+".json", obj)

    def test_complete_reserved_replay_invokes_training_verifiers(self):
        self.assertTrue(self.verify());self.replay.assert_called_once();self.finals.assert_called_once()

    def test_original_development_scope_is_mandatory(self):
        with self.assertRaisesRegex(ValueError, "original verified"):
            common.verify_selection(self.selection, self.protocol_hash)
        self.auth["phase"] = "external_evaluation"
        with self.assertRaises(ValueError):self.verify()

    def test_external_opening_requires_ready_pinned_preprocessor(self):
        self.preflight.return_value = {"status": "blocked"}
        with self.assertRaisesRegex(ValueError, "unavailable before external"):self.verify()

    def test_rehashed_missing_and_mislabeled_reserved_scores_rejected(self):
        for mutation in ("missing", "label", "partition", "site"):
            original = copy.deepcopy(self.payload)
            if mutation == "missing":self.payload["rows"].pop()
            elif mutation == "label":self.payload["rows"][0]["diagnosis"] = 1-self.payload["rows"][0]["diagnosis"]
            elif mutation == "partition":self.payload["rows"][0]["partition"] = "external"
            else:self.payload["rows"][0]["site_id"] = "external_institution"
            self.replace_calibration_payload()
            with self.assertRaisesRegex(ValueError, "Calibration"):self.verify()
            self.payload = original

    def test_rehashed_threshold_change_is_recomputed(self):
        self.fits["new"]["threshold"] = .876
        self.replace_calibration_payload()
        with self.assertRaisesRegex(ValueError, "does not replay"):self.verify()

    def test_selection_thresholds_cannot_override_fitted_values(self):
        original = copy.deepcopy(self.selection["thresholds"])
        for value in (.876, float("nan"), True):
            self.selection["thresholds"]["new"] = value
            with self.assertRaisesRegex(ValueError, "decision thresholds"):self.verify()
        self.selection["thresholds"] = original
        del self.selection["thresholds"]["recalibrated_old"]
        with self.assertRaisesRegex(ValueError, "decision thresholds"):self.verify()

    def test_wrong_score_model_and_precalibrated_score_sources_rejected(self):
        self.payload["model_provenance"]["new_checkpoint_sha256"][0] = "f"*64
        self.replace_calibration_payload()
        with self.assertRaisesRegex(ValueError, "checkpoints"):self.verify()

    def test_old_recalibration_cannot_change_model(self):
        obj = common.read_json(self.selection["artifacts"]["recalibrated_old_bundle"]["path"])
        obj["aggregation"] = "wrong"
        self.selection["artifacts"]["recalibrated_old_bundle"] = self.put("changed_old.json", obj)
        with self.assertRaisesRegex(ValueError, "exact original"):self.verify()

    def test_development_preparation_checks_every_development_patient_only(self):
        manifest = {"patients": [{"patient_id": "t", "partition": "train"}, {"patient_id": "v", "partition": "tune"},
            {"patient_id": "c", "partition": "calibration"}, {"patient_id": "e", "partition": "external"}]}
        refs = {name: self.put(name+"_gate.json", obj)["path"] for name, obj in (("protocol", self.protocol), ("inventory", {}), ("manifest", manifest), ("power_plan", {}))}
        gate = {"status": "passed", "protocol_hash": self.protocol_hash}
        with patch.object(common, "assess_data_gate", return_value=gate), patch.object(preprocessing, "preflight", return_value={"status": "ready"}), patch.object(training, "load_prepared_patient") as loader:
            common.freeze_development(**{k+"_path": v for k, v in refs.items()}, output_path=self.base/"auth.json")
            self.assertEqual([c.args[0]["patient_id"] for c in loader.call_args_list], ["t", "v", "c"])
            loader.side_effect = ValueError("Unapproved preprocessing")
            with self.assertRaisesRegex(ValueError, "Unapproved"):
                common.freeze_development(**{k+"_path": v for k, v in refs.items()}, output_path=self.base/"no_auth.json")
            self.assertFalse((self.base/"no_auth.json").exists())

    def test_external_marker_is_required_and_exactly_bound(self):
        rows = self.auth["manifest_rows"] + [{"patient_id": "external_synthetic", "site_id": "E", "diagnosis": 0, "partition": "external"}]
        refs = {name: self.put(name+"_auth.json", obj) for name, obj in (("protocol", self.protocol), ("inventory", {}), ("manifest", {"patients": rows}), ("power_plan", {}))}
        dev = {"schema_version": "round3_authorization_v1", "phase": "development", "protocol_hash": self.protocol_hash,
            "inputs": refs, "code_sha256": common.code_manifest(), "allowed_actions": ["train", "infer", "calibrate"],
            "allowed_partitions": ["train", "tune", "calibration"], "authorized_rows": {r["patient_id"]: digest(r) for r in self.auth["manifest_rows"]},
            "final_candidate_selection_ref": self.auth["final_candidate_selection_ref"]}
        dev["authorization_sha256"] = digest(dev);devref = self.put("development_auth.json", dev)
        selectionref = self.put("selection_freeze.json", self.selection)
        external = {k: v for k, v in dev.items() if k not in ("authorization_sha256", "final_candidate_selection_ref")}
        external.update(phase="external_evaluation", allowed_actions=["infer", "evaluate"], allowed_partitions=["external"],
            authorized_rows={rows[-1]["patient_id"]: digest(rows[-1])}, selection_freeze={"manifest": selectionref, "development_authorization": devref})
        external["authorization_sha256"] = digest(external);extref = self.put("external_auth.json", external)
        marker_path = Path(refs["protocol"]["path"]).with_suffix(".external_opened.json")
        marker = dict(protocol_hash=self.protocol_hash, selection_sha256=selectionref["sha256"], development_authorization_sha256=devref["sha256"], status="external_opened_development_closed")
        with patch.object(common, "assess_data_gate", return_value={"status": "passed", "protocol_hash": self.protocol_hash}):
            with self.assertRaises(OSError):common.verify_training_authorization(extref["path"])
            common.write_json(marker_path, marker)
            self.assertEqual(common.verify_training_authorization(extref["path"])["phase"], "external_evaluation")
            with self.assertRaisesRegex(ValueError, "revoked"):common.verify_training_authorization(devref["path"])
            marker["selection_sha256"] = "e"*64;common.write_json(marker_path, marker, overwrite=True)
            with self.assertRaisesRegex(ValueError, "opening marker"):common.verify_training_authorization(extref["path"])


if __name__ == "__main__":unittest.main()
