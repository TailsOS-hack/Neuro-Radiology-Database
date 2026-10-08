"""Filename/label coverage contracts for the fixed-cohort comparison."""

import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path


@unittest.skipIf(importlib.util.find_spec("pandas") is None,
                 "Pandas is not installed in the artifact-only environment")
class ComparisonCoverageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from data_visualization.compare_rad_vs_ai import load_comparison_data
        cls.load_comparison_data = staticmethod(load_comparison_data)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.truth = self.write("truth.csv", ["FileName", "TrueCategory"], [
            ["one.jpg", "NonDemented"], ["two.jpg", "MildDemented"],
        ])

    def write(self, filename, columns, rows):
        path = self.root / filename
        with path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(columns)
            writer.writerows(rows)
        return path

    def predictions(self, rows):
        return self.write("predictions.csv", ["FileName", "ModelPrediction"], rows)

    def test_full_prediction_export_aligns_by_filename_and_uses_authoritative_key(self):
        model = self.write("predictions.csv", ["FileName", "TrueCategory", "ModelPrediction"], [
            ["two.jpg", "NonDemented", "MildDemented"],
            ["one.jpg", "MildDemented", "NonDemented"],
        ])
        result = self.load_comparison_data(self.truth, model)
        self.assertEqual(list(result.columns), ["FileName", "TrueCategory", "ModelPrediction"])
        self.assertEqual(result["FileName"].tolist(), ["one.jpg", "two.jpg"])
        self.assertEqual(result["TrueCategory"].tolist(), ["NonDemented", "MildDemented"])

    def test_missing_and_extra_rows_cannot_silently_drop_from_inner_join(self):
        for rows in [
            [["one.jpg", "NonDemented"]],
            [["one.jpg", "NonDemented"], ["other.jpg", "MildDemented"]],
            [["one.jpg", "NonDemented"], ["two.jpg", "MildDemented"], ["extra.jpg", "NonDemented"]],
        ]:
            with self.subTest(rows=rows), self.assertRaisesRegex(ValueError, "coverage"):
                self.load_comparison_data(self.truth, self.predictions(rows))

    def test_duplicate_predictions_after_whitespace_normalization_are_rejected(self):
        model = self.predictions([
            ["one.jpg", "NonDemented"], [" one.jpg ", "MildDemented"],
            ["two.jpg", "MildDemented"],
        ])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.load_comparison_data(self.truth, model)

    def test_duplicate_ground_truth_is_rejected(self):
        self.truth = self.write("truth.csv", ["FileName", "TrueCategory"], [
            ["one.jpg", "NonDemented"], ["one.jpg", "MildDemented"],
        ])
        with self.assertRaisesRegex(ValueError, "Ground truth has duplicate"):
            self.load_comparison_data(self.truth, self.predictions([["one.jpg", "NonDemented"]]))

    def test_empty_or_unknown_labels_and_missing_columns_are_rejected(self):
        for label in ["", "Unknown", "tumor_glioma"]:
            with self.subTest(label=label), self.assertRaisesRegex(ValueError, "unknown dementia labels"):
                self.load_comparison_data(self.truth, self.predictions([
                    ["one.jpg", label], ["two.jpg", "MildDemented"],
                ]))
        model = self.write("bad.csv", ["FileName", "WrongColumn"], [["one.jpg", "NonDemented"]])
        with self.assertRaisesRegex(ValueError, "missing columns"):
            self.load_comparison_data(self.truth, model)


if __name__ == "__main__":
    unittest.main()
