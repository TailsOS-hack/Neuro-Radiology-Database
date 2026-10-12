"""Read-only mechanics/distribution audit of saved Round 2 outputs.

Never refits, runs MRI inference, adjusts a threshold or overwrites old artifacts.
"""
from collections import defaultdict
from pathlib import Path
import json
import math

from .common import read_json, sha256_file


def audit(root):
    root = Path(root)
    base = root / "training_logs/experiments/kevin_20261011/round2_clinical_b300_20261011"
    predictions = base / "clinical_evaluation/patient_predictions.json"
    states = base / "clinical_evaluation/model_state_verification.json"
    protocol = root / "docs/kevin_paper_20261011/round2_clinical/prepared/protocol.json"
    rows, state, p = read_json(predictions), read_json(states), read_json(protocol)
    problems, counts, seen = [], defaultdict(lambda: {"n": 0, "AD_labels": 0, "predicted_AD": 0, "correct": 0, "score_min": 1., "score_max": 0.}), set()
    for r in rows:
        key = (r["model"], r["cohort"], r["group_key"], r["scenario"])
        if key in seen:
            problems.append("duplicate_saved_prediction")
        seen.add(key)
        probabilities = r["probabilities"]
        if len(probabilities) != 2 or not all(math.isfinite(x) and 0 <= x <= 1 for x in probabilities) or abs(sum(probabilities)-1) > 1e-5:
            problems.append("invalid_saved_probability")
            continue
        if r["predicted_ad"] != int(probabilities[1] >= .5):
            problems.append("saved_class_threshold_mismatch")
        if {"CN": 0, "AD": 1, "MCI": -1}.get(r["diagnosis"]) != r["label"]:
            problems.append("saved_diagnosis_class_order_mismatch")
        if r["n_slices"] != 9:
            problems.append("incomplete_nine_slice_bag")
        if r["scenario"] != "clean" or r["label"] not in (0, 1):
            continue
        c = counts[(r["cohort"], r["model"])]
        c["n"] += 1
        c["AD_labels"] += int(r["label"] == 1)
        c["predicted_AD"] += int(r["predicted_ad"])
        c["correct"] += int(r["predicted_ad"] == r["label"])
        c["score_min"] = min(c["score_min"], probabilities[1])
        c["score_max"] = max(c["score_max"], probabilities[1])
    if not all(r["all_states_unchanged"] and r["no_parameter_gradients"] for r in state):
        problems.append("model_state_changed_during_old_evaluation")
    return {"status": "completed_saved_output_audit" if not problems else "failed_saved_output_audit", "mechanics_failures": sorted(set(problems)),
            "saved_records_checked": len(rows), "clean_binary_aggregates": [{"cohort": k[0], "model": k[1], **v} for k, v in sorted(counts.items())],
            "source_sha256": {"predictions": sha256_file(predictions), "state_verification": sha256_file(states), "training_protocol": sha256_file(protocol)},
            "training_patient_count": p["train_patients"], "validation_patient_count": p["val_patients"],
            "new_patient_aggregation": p["bag_aggregation"], "old_patient_aggregation": "mean_nine_slice_probabilities",
            "orientation_or_segmentation_validity_established": False,
            "cause_of_external_failure_proven": False,
            "interpretation": "Recorded labels, thresholds, bags and saved model states can be checked mechanically. Opposite source-level decision collapse motivates domain/preprocessing/calibration investigation, but this audit cannot identify its causal mechanism.",
            "new_inference_or_fitting_performed": False, "old_outputs_modified": False}
