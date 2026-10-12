import copy
import json
import unittest

from research.dementia_round3.sample_size import simulate_plan


def assumptions(n, probabilities):
    return {"sites": [{"site_id": s, "classes": {c: {"n": n, "correctness_probabilities": probabilities[:]} for c in ("CN", "AD")}} for s in ("A", "B")]}


class Round3SampleSizeTests(unittest.TestCase):
    def test_perfect_small_cohorts_cannot_pass_exact_floors(self):
        out = simulate_plan(assumptions(5, [0., 0., 1., 0.]), simulations=20, bootstrap_replicates=30)
        self.assertEqual(out["all_criteria_power_estimate"], 0.)
        self.assertEqual(out["total_patients"], 20)
        self.assertFalse(out["clinical_approval"])

    def test_identical_models_cannot_pass_superiority(self):
        out = simulate_plan(assumptions(100, [.95, 0., 0., .05]), simulations=20, bootstrap_replicates=30)
        self.assertEqual(out["all_criteria_power_estimate"], 0.)

    def test_explicit_strong_design_passes_and_is_reproducible(self):
        a = assumptions(40, [0., 0., 1., 0.])
        out = simulate_plan(a, simulations=20, bootstrap_replicates=30, seed=3)
        self.assertEqual(out["all_criteria_power_estimate"], 1.)
        self.assertEqual(out, simulate_plan(a, simulations=20, bootstrap_replicates=30, seed=3))
        b = copy.deepcopy(a);b["sites"][0]["classes"]["AD"]["n"] = 41
        self.assertNotEqual(out["assumptions_sha256"], simulate_plan(b, simulations=2, bootstrap_replicates=5)["assumptions_sha256"])
        json.dumps(out, allow_nan=False)

    def test_no_implicit_effect_or_invalid_probabilities(self):
        for a in [{}, {"sites": []}, assumptions(100, [.2, .2, .2, .2]), assumptions(100, [float("nan"), 0., 0., 1.]), assumptions(0, [0., 0., 1., 0.])]:
            with self.assertRaises(ValueError):
                simulate_plan(a, simulations=2, bootstrap_replicates=3)


if __name__ == "__main__":
    unittest.main()
