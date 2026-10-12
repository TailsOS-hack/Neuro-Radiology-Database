"""Logistic score calibration and threshold fitting on reserved development only."""
import math
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit

from .statistics import validate_rows

EPSILON = 1e-6


def _unavailable(reason, n, accepted):
    return {"status": "unavailable", "reason": reason, "n_eligible": n,
            "n_accepted_for_fit": accepted, "threshold": None, "clinical_approval": False}


def apply_calibration(scores, fitted):
    if fitted.get("status") != "fitted_development_only" or fitted.get("role") != "development_calibration" or fitted.get("class_order") != ["CN", "AD"]:
        raise ValueError("Require an identifiable development-calibration fit in CN/AD order")
    a, b = fitted["intercept"], fitted["slope"]
    if not all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in (a, b)):
        raise ValueError("Nonfinite calibration coefficients")
    result = []
    for p in scores:
        if p is None:
            result.append(None)
        elif isinstance(p, (bool, str)) or not isinstance(p, (int, float, np.number)) or not math.isfinite(float(p)) or not 0 <= p <= 1:
            raise ValueError("Invalid probability-like input")
        else:
            result.append(float(expit(a+b*logit(np.clip(float(p), EPSILON, 1-EPSILON)))))
    return result


def fit_calibration(rows, *, role, class_order, score_field="new_score", minimum_sensitivity=.90, provenance=None):
    if role != "development_calibration" or class_order != ["CN", "AD"]:
        raise ValueError("Calibration requires reserved development_calibration role and ['CN','AD']")
    if score_field not in ("new_score", "old_score"):
        raise ValueError("score_field must identify the new or old model")
    if isinstance(minimum_sensitivity, bool) or not isinstance(minimum_sensitivity, (int, float)) or not math.isfinite(minimum_sensitivity) or not 0 < minimum_sensitivity <= 1:
        raise ValueError("Invalid sensitivity target")
    rows = sorted(validate_rows(rows), key=lambda r: r["patient_id"])
    if any(r.get("partition") != "calibration" for r in rows):
        raise ValueError("Every calibration row must come from the reserved calibration partition")
    if provenance is not None:
        if not isinstance(provenance, dict) or set(provenance) != {"protocol_sha256", "calibration_manifest_sha256", "fold_id"}:
            raise ValueError("Provenance requires protocol/manifest SHA256 and fold_id only")
        for key in ("protocol_sha256", "calibration_manifest_sha256"):
            if not isinstance(provenance[key], str) or len(provenance[key]) != 64 or any(c not in "0123456789abcdef" for c in provenance[key]):
                raise ValueError("Invalid calibration provenance digest")
        if not isinstance(provenance["fold_id"], str) or not provenance["fold_id"].strip():
            raise ValueError("Calibration provenance requires a fold identifier")
    model = score_field.split("_")[0]
    accepted = [r for r in rows if r[model+"_status"] == "ok"]
    n, k = len(rows), len(accepted)
    y = np.array([r["diagnosis"] for r in accepted])
    if len(set(y)) != 2:
        return _unavailable("Both diagnoses need accepted predictions for fitting", n, k)
    x = logit(np.clip([r[score_field] for r in accepted], EPSILON, 1-EPSILON))
    if np.ptp(x) < 1e-12:
        return _unavailable("Constant scores: slope is not identifiable", n, k)
    if x[y == 0].max() <= x[y == 1].min() or x[y == 1].max() <= x[y == 0].min():
        return _unavailable("Complete or quasi separation: unpenalized logistic coefficients are not identifiable", n, k)
    design = np.column_stack([np.ones(k), x])
    def loss_gradient(theta):
        z = design @ theta
        return float(np.mean(np.logaddexp(0, z)-y*z)), design.T @ (expit(z)-y)/k
    fitted = minimize(loss_gradient, [0., 1.], jac=True, method="BFGS", options={"gtol": 1e-9, "maxiter": 1000})
    gradient = loss_gradient(fitted.x)[1]
    if not np.isfinite(fitted.x).all() or np.max(np.abs(gradient)) > 1e-6:
        return _unavailable("Logistic fit did not converge", n, k)
    prob = expit(design @ fitted.x)
    hessian = design.T @ ((prob*(1-prob))[:, None]*design)/k
    if np.linalg.eigvalsh(hessian).min() <= 1e-10:
        return _unavailable("Logistic fit is numerically unidentifiable", n, k)
    result = {"status": "fitted_development_only", "role": role, "class_order": class_order,
              "score_field": score_field, "intercept": float(fitted.x[0]), "slope": float(fitted.x[1]),
              "input_clip_epsilon": EPSILON, "n_eligible": n, "n_accepted_for_fit": k,
              "n_abstained_or_failed_excluded_from_fit_but_in_threshold_denominator": n-k,
              "minimum_sensitivity": float(minimum_sensitivity), "threshold": None,
              "threshold_status": "unavailable", "clinical_approval": False,
              "provenance": dict(provenance) if provenance is not None else None,
              "provenance_supplied": provenance is not None,
              "provenance_scope": "Digests recorded; CLI must verify row membership and source bytes against the frozen manifest.",
              "fit_scope": "Accepted development scores only; threshold rates use every eligible development patient."}
    scores = apply_calibration([r[score_field] if r[model+"_status"] == "ok" else None for r in rows], result)
    nad = sum(r["diagnosis"] == 1 for r in rows)
    ncn = n-nad
    feasible = []
    for t in sorted({0., 1.} | {s for s in scores if s is not None}):
        ad = sum(s is not None and s >= t and r["diagnosis"] == 1 for r, s in zip(rows, scores))
        cn = sum(s is not None and s < t and r["diagnosis"] == 0 for r, s in zip(rows, scores))
        sensitivity, specificity = ad/nad, cn/ncn
        if sensitivity >= minimum_sensitivity:
            feasible.append((specificity, t, sensitivity))
    if feasible:
        specificity, threshold, sensitivity = max(feasible)
        result.update(threshold=float(threshold), threshold_status="available", development_sensitivity=float(sensitivity), development_specificity=float(specificity))
    else:
        result["threshold_unavailable_reason"] = "No threshold meets sensitivity on all eligible development patients, including failures/abstentions."
    result["threshold_tie_rule"] = "Highest specificity, then highest threshold; AD when score >= threshold."
    return result
