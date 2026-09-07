# Dr. Cockroft — follow-up review memo (validation-v2-20260907)

Internal research draft; not sent. The original mentor packet remains the historical baseline.

**What changed.** We added shape-preserving masking controls, model-randomization diagnostics, validation-only temperature scaling, and automated evidence checks. The sample, accepted checkpoint weights and original gallery selection were retained.

**Explainability.** At 20% mean replacement, top-CAM masking exceeded shape-control confidence drops by 0.2037 for dementia and 0.0873 for tumor. Comparisons with scattered random pixels were -0.2412 and -0.3702. This contrasts with interpreting the original random-pixel comparison alone. Control design materially affects the result; anatomy has not been validated. Of 2,304 randomized-map attempts, 385 were flat or failed and are reported explicitly.

**Calibration.** Dementia ECE changed from 0.038596 to 0.000581; tumor ECE changed from 0.037228 to 0.012143. Component accuracy was unchanged. These are internal estimates from data already used in model development and review. The application still uses its original confidence scores.

**Claim boundaries.** Public augmented dementia data, absent patient identifiers, possible patient-level leakage, source-domain bias and absent independent external validation remain central limitations. Lower internal ECE does not establish calibrated clinical probabilities. No retraining occurred.

**Review decisions.** Should the control-dependent explainability findings remain supplementary? Is the calibration comparison useful in the main manuscript? Which independent patient-level cohort and reference standard would support the next study?

**Reading.** RESULTS.md contains generated tables and interpretation. Figures 9–11 and the fixed gallery are in index.html. External readiness is documented separately; no external performance result is claimed.
