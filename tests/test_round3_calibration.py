import json
import unittest

from scipy.special import expit, logit

from research.dementia_round3.calibration import fit_calibration, apply_calibration


def shifted_rows():
    rows = []
    # Each bin's empirical probability is exact; supplied scores are shifted
    # by +0.8 logits, so the fitted intercept should recover -0.8, slope 1.
    for b, positives in enumerate([10, 30, 50, 70, 90]):
        p = float(expit(logit(positives/100)+.8))
        for i in range(100):
            rows.append(dict(patient_id=f"p{b}_{i}", site_id="developmentA", diagnosis=int(i < positives),
                             old_score=p, new_score=p, old_status="ok", new_status="ok", partition="calibration"))
    return rows


class Round3CalibrationTests(unittest.TestCase):
    def test_known_shift_and_deterministic_threshold(self):
        rows = shifted_rows()
        fit = fit_calibration(rows, role="development_calibration", class_order=["CN", "AD"])
        self.assertEqual(fit["status"], "fitted_development_only")
        self.assertAlmostEqual(fit["intercept"], -.8, places=5)
        self.assertAlmostEqual(fit["slope"], 1., places=5)
        self.assertGreaterEqual(fit["development_sensitivity"], .9)
        self.assertEqual(fit, fit_calibration(list(reversed(rows)), role="development_calibration", class_order=["CN", "AD"]))
        self.assertNotIn("patient_id", json.dumps(fit, allow_nan=False))
        self.assertIsNone(apply_calibration([None], fit)[0])
        self.assertAlmostEqual(apply_calibration([rows[0]["new_score"]], fit)[0], .1, places=5)
        scores = apply_calibration([r["new_score"] for r in rows], fit)
        feasible = []
        for threshold in set(scores) | {0., 1.}:
            sensitivity = sum(p >= threshold and r["diagnosis"] == 1 for r, p in zip(rows, scores))/250
            specificity = sum(p < threshold and r["diagnosis"] == 0 for r, p in zip(rows, scores))/250
            if sensitivity >= .9:
                feasible.append((specificity, threshold))
        self.assertEqual((fit["development_specificity"], fit["threshold"]), max(feasible))

    def test_provenance_recorded_without_claiming_membership_verification(self):
        p = {"protocol_sha256": "a"*64, "calibration_manifest_sha256": "b"*64, "fold_id": "development_fold0"}
        fit = fit_calibration(shifted_rows(), role="development_calibration", class_order=["CN", "AD"], provenance=p)
        self.assertEqual(fit["provenance"], p)
        self.assertIn("CLI must verify", fit["provenance_scope"])
        p["patient_ids"] = ["not_allowed"]
        with self.assertRaises(ValueError):
            fit_calibration(shifted_rows(), role="development_calibration", class_order=["CN", "AD"], provenance=p)

    def test_roles_class_order_and_each_partition_checked(self):
        rows = shifted_rows()
        for role, order in [("external", ["CN", "AD"]), ("development_calibration", ["AD", "CN"])]:
            with self.assertRaises(ValueError):
                fit_calibration(rows, role=role, class_order=order)
        for partition in ["external", "train", "validation", None]:
            rows[0]["partition"] = partition
            with self.assertRaises(ValueError):
                fit_calibration(rows, role="development_calibration", class_order=["CN", "AD"])

    def test_failures_cannot_fake_sensitivity_target(self):
        rows = shifted_rows()
        for row in rows:
            if row["diagnosis"] == 1 and int(row["patient_id"].split("_")[1]) % 3 == 0:
                row.update(new_status="failed", new_score=None)
        fit = fit_calibration(rows, role="development_calibration", class_order=["CN", "AD"])
        self.assertEqual(fit["status"], "fitted_development_only")
        self.assertEqual(fit["threshold_status"], "unavailable")
        self.assertIsNone(fit["threshold"])

    def test_unidentifiable_and_nonfinite_explicit(self):
        rows = shifted_rows()
        for row in rows:
            row["new_score"] = .5
        fit = fit_calibration(rows, role="development_calibration", class_order=["CN", "AD"])
        self.assertEqual(fit["status"], "unavailable")
        for row in rows:
            row["new_score"] = .9 if row["diagnosis"] else .1
        self.assertEqual(fit_calibration(rows, role="development_calibration", class_order=["CN", "AD"])["status"], "unavailable")
        rows[0]["new_score"] = float("nan")
        with self.assertRaises(ValueError):
            fit_calibration(rows, role="development_calibration", class_order=["CN", "AD"])


if __name__ == "__main__":
    unittest.main()
