"""The 100-case accuracy gate must never silently drop difficult cases."""
import csv
import tempfile
import unittest
from pathlib import Path

from scripts.evaluate_radiologist_cohort import load_cases, summarize


class CohortEvaluationTests(unittest.TestCase):
    def test_key_requires_unique_existing_cases_and_known_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            key = root / "key.csv"
            (root / "a.jpg").touch()
            (root / "b.jpg").touch()

            def write(rows):
                with key.open("w", newline="") as stream:
                    writer = csv.writer(stream)
                    writer.writerow(["FileName", "TrueCategory"])
                    writer.writerows(rows)

            write([["a.jpg", "NonDemented"], ["b.jpg", "MildDemented"]])
            self.assertEqual(len(load_cases(key, root, expected_count=2)), 2)
            with self.assertRaises(ValueError):
                load_cases(key, root)
            for bad in [["a.jpg", "MildDemented"], ["b.jpg", "Unknown"],
                        ["../a.jpg", "NonDemented"]]:
                write([["a.jpg", "NonDemented"], bad])
                with self.assertRaises(ValueError):
                    load_cases(key, root, expected_count=2)
            write([["a.jpg", "NonDemented"], ["absent.jpg", "MildDemented"]])
            with self.assertRaises(FileNotFoundError):
                load_cases(key, root, expected_count=2)

    def test_above_95_means_at_least_96_correct_and_cam_failure_still_fails(self):
        rows = [{"TrueCategory": "NonDemented", "ModelPrediction": "NonDemented",
                 "Correct": True, "GradCAMValid": True,
                 "PredictionUnchangedAfterCAM": True} for _ in range(100)]
        for row in rows[:5]:
            row.update(ModelPrediction="tumor_notumor", Correct=False)
        result = summarize(rows, .95)
        self.assertFalse(result["accuracy_target_met"])
        self.assertEqual(sum(map(sum, result["confusion_matrix"])), 100)
        rows[4].update(ModelPrediction="NonDemented", Correct=True)
        rows[0]["GradCAMValid"] = False
        result = summarize(rows, .95)
        self.assertTrue(result["accuracy_target_met"])
        self.assertFalse(result["technical_checks_passed"])
        self.assertEqual(result["gradcam_failed_or_flat"], 1)

    def test_explanation_changing_logits_fails_even_with_correct_labels(self):
        result = summarize([{"TrueCategory": "NonDemented", "ModelPrediction": "NonDemented",
                             "Correct": True, "GradCAMValid": True,
                             "PredictionUnchangedAfterCAM": False}], .95)
        self.assertTrue(result["accuracy_target_met"])
        self.assertFalse(result["technical_checks_passed"])
        with self.assertRaises(ValueError):
            summarize([], .95)


if __name__ == "__main__":
    unittest.main()
