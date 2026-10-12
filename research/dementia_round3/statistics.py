"""Patient-paired, fixed-site research evaluation; never clinical approval."""
from collections import Counter
import math

import numpy as np
from scipy.stats import beta

STATUSES = {"ok", "abstain", "failed"}
DEFAULT_SEED = 20261031


def validate_rows(rows):
    """Validate every eligible row without dropping or exposing patient IDs."""
    if not isinstance(rows, list) or not rows:
        raise ValueError("rows must be a nonempty list")
    seen, result = set(), []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Every row must be a dictionary")
        r = dict(row)
        for field in ("patient_id", "site_id"):
            if not isinstance(r.get(field), str) or not r[field].strip():
                raise ValueError("Patient and site identifiers must be nonempty strings")
            if r[field] != r[field].strip():
                raise ValueError("Identifiers must not contain surrounding whitespace")
        if r["patient_id"] in seen:
            raise ValueError("Duplicate patient: all visits/sites must remain one patient")
        seen.add(r["patient_id"])
        if type(r.get("diagnosis")) is not int or r["diagnosis"] not in (0, 1):
            raise ValueError("diagnosis must be integer 0 (CN) or 1 (AD)")
        for model in ("old", "new"):
            status, field = r.get(model + "_status"), model + "_score"
            if not isinstance(status, str) or status not in STATUSES or field not in r:
                raise ValueError("Each model requires an explicit status and score")
            value = r[field]
            if value is None:
                if status == "ok":
                    raise ValueError("An accepted prediction requires a score")
            elif isinstance(value, (bool, str)) or not isinstance(value, (int, float, np.number)) or not math.isfinite(float(value)) or not 0 <= float(value) <= 1:
                raise ValueError("Every provided score must be finite and within [0,1]")
            else:
                r[field] = float(value)
        result.append(r)
    return result


def _threshold(value):
    if isinstance(value, (bool, str)) or not isinstance(value, (int, float, np.number)) or not math.isfinite(float(value)) or not 0 <= float(value) <= 1:
        raise ValueError("Threshold must be finite and within [0,1]")
    return float(value)


def exact_interval(correct, n):
    """Two-sided 95% Clopper-Pearson interval; no patients => unavailable."""
    if type(n) is not int or type(correct) is not int or not 0 <= correct <= n:
        raise ValueError("Invalid binomial counts")
    if n == 0:
        return None
    return [0.0 if correct == 0 else float(beta.ppf(.025, correct, n-correct+1)),
            1.0 if correct == n else float(beta.ppf(.975, correct+1, n-correct))]


def _rate(k, n):
    return {"correct": int(k), "denominator": int(n), "estimate": float(k/n) if n else None,
            "exact95": exact_interval(int(k), int(n))}


def _metrics(correct, labels):
    rates = [_rate(int(correct[labels == c].sum()), int((labels == c).sum())) for c in (0, 1)]
    return {"specificity_cn": rates[0], "sensitivity_ad": rates[1],
            "balanced_accuracy": float(np.mean([x["estimate"] for x in rates])) if all(x["denominator"] for x in rates) else None,
            "accuracy": _rate(int(correct.sum()), len(correct))}


def _paired_counts(old, new):
    return np.array([np.sum(old & new), np.sum(old & ~new), np.sum(~old & new), np.sum(~old & ~new)], dtype=int)


def _bootstrap_difference(strata, replicates, rng):
    """Multinomial resampling is exactly a paired patient bootstrap of four cells.

    strata contains each site's CN then AD counts. Equal weighting of all
    site/class strata gives equal-site balanced accuracy. Sites stay fixed.
    """
    if any(int(x.sum()) == 0 for x in strata):
        return None
    draws = np.zeros(replicates, dtype=float)
    for counts in strata:
        n = int(counts.sum())
        samples = rng.multinomial(n, counts/n, size=replicates)
        draws += (samples[:, 2] - samples[:, 1])/n
    draws /= len(strata)
    return [float(x) for x in np.quantile(draws, [.025, .975], method="linear")]


def _gate(site_metrics, delta, ci, minimum_sites):
    site_gates = {}
    for site, m in site_metrics.items():
        values = [m[x] for x in ("sensitivity_ad", "specificity_cn")]
        site_gates[site] = bool(all(x["estimate"] is not None and x["estimate"] >= .85 and x["exact95"][0] >= .80 for x in values))
    components = {"at_least_required_sites": len(site_metrics) >= minimum_sites,
                  "macro_site_balanced_accuracy_delta_point_at_least_0_05": delta is not None and delta >= .05 - 1e-12,
                  "paired_95_lower_above_zero": ci is not None and ci[0] > 0,
                  "all_sites_sensitivity_specificity_point_and_exact_lower_floors": bool(site_gates) and all(site_gates.values())}
    return {"passed": all(components.values()), "components": components, "each_site": site_gates,
            "meaning": "Operational research progression only; not clinical approval or proof that true gain is at least 5 percentage points."}


def analyze(rows, *, old_threshold=.5, new_threshold=.5, bootstrap_replicates=2000,
            seed=DEFAULT_SEED, minimum_sites=2):
    rows = validate_rows(rows)
    thresholds = {"old": _threshold(old_threshold), "new": _threshold(new_threshold)}
    if type(bootstrap_replicates) is not int or bootstrap_replicates < 1:
        raise ValueError("bootstrap_replicates must be positive")
    if type(seed) is not int or seed < 0 or type(minimum_sites) is not int or minimum_sites < 2:
        raise ValueError("Require a nonnegative seed and at least two external sites")
    # Stable ordering makes reordering input rows leave deterministic results unchanged.
    rows.sort(key=lambda r: (r["site_id"], r["diagnosis"], r["patient_id"]))
    sites, strata = {}, []
    for site in sorted({r["site_id"] for r in rows}):
        group = [r for r in rows if r["site_id"] == site]
        y = np.array([r["diagnosis"] for r in group], dtype=int)
        models, correct = {}, {}
        for model in ("old", "new"):
            accepted = np.array([r[model+"_status"] == "ok" for r in group], dtype=bool)
            pred = np.array([r[model+"_score"] >= thresholds[model] if r[model+"_status"] == "ok" else False for r in group])
            correct[model] = accepted & (pred == y)
            models[model] = _metrics(correct[model], y)
            models[model].update({"status_counts": dict(Counter(r[model+"_status"] for r in group)),
                                 "coverage": float(accepted.mean()),
                                 "accepted_only_secondary": _metrics(correct[model][accepted], y[accepted])})
        cells = {}
        for c, name in [(0, "CN"), (1, "AD")]:
            counts = _paired_counts(correct["old"][y == c], correct["new"][y == c])
            strata.append(counts)
            cells[name] = dict(zip(["both_correct", "old_only_correct", "new_only_correct", "both_incorrect"], map(int, counts)))
        sites[site] = {"n_eligible": len(group), "models": models, "paired_correctness": cells}
    macro = {}
    for model in ("old", "new"):
        scores = [v["models"][model]["balanced_accuracy"] for v in sites.values()]
        macro[model] = float(np.mean(scores)) if all(v is not None for v in scores) else None
    delta = macro["new"]-macro["old"] if macro["new"] is not None and macro["old"] is not None else None
    ci = _bootstrap_difference(strata, bootstrap_replicates, np.random.default_rng(seed))
    gate = _gate({s: v["models"]["new"] for s, v in sites.items()}, delta, ci, minimum_sites)
    return {"status": "complete_research_statistics", "n_eligible": len(rows), "n_sites": len(sites), "sites": sites,
            "macro_site_balanced_accuracy": macro, "new_minus_old_delta": delta, "paired95": ci,
            "research_progression_gate": gate, "thresholds": thresholds,
            "bootstrap": {"replicates": bootstrap_replicates, "seed": seed, "strata": "fixed site and diagnosis", "paired_patient_resampling": True, "equal_site_weights": True,
                          "degenerate_interval": ci is not None and ci[0] == ci[1],
                          "limitation": "A degenerate conditional interval, especially for perfect/small strata, is not population certainty; exact per-site class intervals remain required."},
            "all_eligible_primary": True, "abstentions_and_failures_counted_incorrect": True,
            "clinical_approval": False, "inference_scope": "Conditional on the observed patients and fixed sites; not a new-site population interval.",
            "external_site_eligibility": "Must be independently verified by the frozen data protocol; site_id alone cannot establish freshness or independence."}
