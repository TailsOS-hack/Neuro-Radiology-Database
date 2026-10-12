import json
import unittest

from research.dementia_round3.statistics import analyze, exact_interval


def rows_for(sites=2, n=30, new="perfect", old="perfect"):
    rows = []
    for site in range(sites):
        for label in (0, 1):
            for i in range(n):
                def score(kind):
                    return {"perfect": .9 if label else .1, "all_ad": .9, "all_cn": .1,
                            "wrong": .1 if label else .9}[kind]
                rows.append(dict(patient_id=f"p{site}_{label}_{i}", site_id=f"site{site}", diagnosis=label,
                                 old_score=score(old), new_score=score(new), old_status="ok", new_status="ok"))
    return rows


class Round3StatisticsTests(unittest.TestCase):
    def test_all_ad_and_all_cn_do_not_pass(self):
        for kind in ("all_ad", "all_cn"):
            result = analyze(rows_for(new=kind), bootstrap_replicates=100)
            self.assertEqual(result["macro_site_balanced_accuracy"]["new"], .5)
            self.assertFalse(result["research_progression_gate"]["passed"])

    def test_abstentions_count_in_all_eligible_and_pairing(self):
        rows = rows_for(n=20)
        for row in rows[:10]:
            row.update(new_status="abstain", new_score=None)
        result = analyze(rows, bootstrap_replicates=100)
        site = result["sites"]["site0"]["models"]["new"]
        self.assertEqual(site["specificity_cn"]["estimate"], .5)
        self.assertEqual(site["accepted_only_secondary"]["accuracy"]["estimate"], 1)
        self.assertEqual(site["coverage"], .75)
        self.assertFalse(result["research_progression_gate"]["passed"])
        for row in rows:
            row.update(old_status="failed", old_score=None, new_status="failed", new_score=None)
        both = analyze(rows, bootstrap_replicates=100)
        self.assertEqual(both["macro_site_balanced_accuracy"], {"old": 0., "new": 0.})
        self.assertEqual(both["paired95"], [0., 0.])

    def test_duplicate_or_malformed_never_dropped(self):
        for change in ({"patient_id": "p0_0_0"}, {"diagnosis": True}, {"diagnosis": 2},
                       {"new_score": float("nan")}, {"old_score": float("inf")},
                       {"new_score": None}, {"new_status": "unknown"}, {"old_score": "0.3"}):
            rows = rows_for()
            rows[-1].update(change)
            with self.assertRaises(ValueError):
                analyze(rows, bootstrap_replicates=10)

    def test_perfect_small_n_is_not_precise(self):
        result = analyze(rows_for(n=5, old="wrong"), bootstrap_replicates=100)
        self.assertEqual(result["new_minus_old_delta"], 1)
        self.assertEqual(result["paired95"], [1., 1.])
        self.assertLess(result["sites"]["site0"]["models"]["new"]["sensitivity_ad"]["exact95"][0], .8)
        self.assertFalse(result["research_progression_gate"]["passed"])
        self.assertAlmostEqual(exact_interval(0, 1)[1], .975)

    def test_equal_site_weight_not_pooled_size_and_order_invariant(self):
        a = rows_for(sites=1, n=100, new="perfect", old="all_cn")
        b = rows_for(sites=1, n=10, new="wrong", old="all_cn")
        for row in b:
            row["site_id"] = "site1"
            row["patient_id"] = "b"+row["patient_id"]
        r = analyze(a+b, bootstrap_replicates=200, seed=17)
        self.assertEqual(r["macro_site_balanced_accuracy"]["new"], .5)
        self.assertEqual(r, analyze(list(reversed(a+b)), bootstrap_replicates=200, seed=17))
        self.assertNotIn("patient_id", json.dumps(r, allow_nan=False))

    def test_shared_patient_pairs_not_independent_models(self):
        rows = rows_for(n=50)
        for row in rows[::3]:
            row["old_score"] = row["new_score"] = 1-row["new_score"]
        result = analyze(rows, bootstrap_replicates=200)
        self.assertEqual(result["paired95"], [0., 0.])
        improved = analyze(rows_for(n=30, old="wrong"), bootstrap_replicates=100)
        self.assertTrue(improved["research_progression_gate"]["passed"])

    def test_missing_class_and_one_site_are_explicit_nonpasses(self):
        result = analyze(rows_for(sites=1), bootstrap_replicates=30)
        self.assertFalse(result["research_progression_gate"]["components"]["at_least_required_sites"])
        rows = [x for x in rows_for() if not (x["site_id"] == "site1" and x["diagnosis"] == 1)]
        result = analyze(rows, bootstrap_replicates=30)
        self.assertIsNone(result["paired95"])
        self.assertFalse(result["research_progression_gate"]["passed"])


if __name__ == "__main__":
    unittest.main()
