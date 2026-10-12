"""Explicit-assumption simulations of the complete operational research gate."""
import hashlib
import json
import math

import numpy as np

from .statistics import DEFAULT_SEED, _bootstrap_difference, _gate, _rate, exact_interval


def validate_assumptions(assumptions):
    if not isinstance(assumptions, dict) or set(assumptions) != {"sites"} or not isinstance(assumptions["sites"], list) or len(assumptions["sites"]) < 2:
        raise ValueError("Explicit assumptions require at least two sites")
    result, seen = [], set()
    for site in assumptions["sites"]:
        if not isinstance(site, dict) or set(site) != {"site_id", "classes"}:
            raise ValueError("Each site needs only site_id and classes")
        sid = site["site_id"]
        if not isinstance(sid, str) or not sid.strip() or sid != sid.strip() or sid in seen:
            raise ValueError("Invalid or duplicate site")
        seen.add(sid)
        if not isinstance(site["classes"], dict) or set(site["classes"]) != {"CN", "AD"}:
            raise ValueError("Both CN and AD assumptions are mandatory at each site")
        classes = {}
        for name in ("CN", "AD"):
            c = site["classes"][name]
            if not isinstance(c, dict) or set(c) != {"n", "correctness_probabilities"} or type(c["n"]) is not int or c["n"] < 1:
                raise ValueError("Explicit positive class denominators are required")
            probs = c["correctness_probabilities"]
            if not isinstance(probs, list) or len(probs) != 4 or any(isinstance(x, (bool, str)) or not isinstance(x, (int, float)) or not math.isfinite(x) or not 0 <= x <= 1 for x in probs) or not math.isclose(sum(probs), 1., abs_tol=1e-12, rel_tol=0):
                raise ValueError("Four paired correctness probabilities must sum to one")
            classes[name] = {"n": c["n"], "correctness_probabilities": list(map(float, probs))}
        result.append({"site_id": sid, "classes": classes})
    return {"sites": sorted(result, key=lambda x: x["site_id"])}


def simulate_plan(assumptions, *, simulations=1000, bootstrap_replicates=500, seed=DEFAULT_SEED):
    a = validate_assumptions(assumptions)
    if type(simulations) is not int or simulations < 1 or type(bootstrap_replicates) is not int or bootstrap_replicates < 1 or type(seed) is not int or seed < 0:
        raise ValueError("Positive simulation/bootstrap counts and nonnegative seed required")
    rng = np.random.default_rng(seed)
    passed = 0
    component_counts, site_counts = {}, {s["site_id"]: 0 for s in a["sites"]}
    for _ in range(simulations):
        strata, site_metrics, deltas = [], {}, []
        for site in a["sites"]:
            m = {}
            for name, metric in [("CN", "specificity_cn"), ("AD", "sensitivity_ad")]:
                c = site["classes"][name]
                cells = rng.multinomial(c["n"], c["correctness_probabilities"])
                strata.append(cells)
                m[metric] = _rate(int(cells[0]+cells[2]), c["n"])
                deltas.append(float((cells[2]-cells[1])/c["n"]))
            site_metrics[site["site_id"]] = m
        ci = _bootstrap_difference(strata, bootstrap_replicates, rng)
        gate = _gate(site_metrics, float(np.mean(deltas)), ci, 2)
        passed += gate["passed"]
        for key, value in gate["components"].items():
            component_counts[key] = component_counts.get(key, 0)+int(value)
        for key, value in gate["each_site"].items():
            site_counts[key] += int(value)
    denominators = {s["site_id"]: {k: v["n"] for k, v in s["classes"].items()} for s in a["sites"]}
    power_interval = exact_interval(passed, simulations)
    return {"status": "simulated", "purpose": "research_sample_size_planning_only", "assumptions": a,
            "assumptions_sha256": hashlib.sha256(json.dumps(a, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest(),
            "correctness_cell_order": ["both_correct", "old_only_correct", "new_only_correct", "both_incorrect"],
            "assumed_all_eligible_denominators": denominators,
            "total_patients": sum(sum(v.values()) for v in denominators.values()),
            "joint_gate_power": passed/simulations, "power95lower": power_interval[0], "joint_gate_power_95ci": power_interval,
            "all_criteria_power_estimate": passed/simulations, "monte_carlo_exact95": power_interval,
            "component_pass_proportions": {k: v/simulations for k, v in component_counts.items()},
            "site_floor_pass_proportions": {k: v/simulations for k, v in site_counts.items()},
            "simulations": simulations, "bootstrap_replicates": bootstrap_replicates, "seed": seed,
            "denominator_interpretation": "User-supplied candidate design, not a calculated minimum; vary explicit counts/assumptions and target power before freezing.",
            "limitations": ["Conditional on explicit paired correctness assumptions and fixed sites; no random new-site heterogeneity modeled.",
                            "Cells must already count each system's abstentions/failures as incorrect.",
                            "Simulation assesses the full research gate, not clinical readiness, calibration precision, subgroup power, or reader-study power.",
                            "Inner bootstrap Monte Carlo resolution can affect estimated power; verify stability at the final prespecified replicate count."],
            "clinical_approval": False}
