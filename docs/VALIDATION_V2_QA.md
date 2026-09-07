# Validation follow-up QA

Completed run: `validation-v2-20260907`.

- Original mentor bundle: frozen inventory and historical source snapshots verified unchanged. Accepted checkpoint files were not modified or retrained.
- Explainability: 256 fixed sampled images, 1,536 perturbation settings, 2,304 randomized-map attempts. Flat/failed randomizations are explicit (385); conditional summary denominators are preserved.
- Calibration: 4,952 validation images and 10,251 strict-test images were processed for the four checkpoints; each specialist's fit and task metrics use only its domain rows. All temperatures were frozen before test inference. Historical predicted labels matched for every archived task case.
- Repetition: two complete GPU runs compared 2,187 saved arrays and 19,313 CSV rows. Maximum array difference was zero. Categorical selection and provenance matched. The earlier interrupted repeat directories remained unpublished and were not used as completed evidence.
- Automated tests: 20 passed, including semantic corruption tests after re-sealing artifacts. The analytical Grad-CAM mechanics check also passed.
- Package gates: 18 checks, zero failures, one retained historical calibration warning. The warning continues to apply to the original uncalibrated package; research-only scaling does not change its claims or the application.
- Lightweight execution: a separate environment containing only NumPy, SciPy and Pillow passed the evidence checks with optimized Python. A sparse-checkout simulation without a data directory or Torch passed baseline, follow-up and publication verification. The new GitHub workflow is configured; no remote CI run is claimed here.
- External interface: six synthetic images from four synthetic patients passed both readiness checking and inference with a frozen checkpoint/calibration artifact. This is a software smoke test, not independent MRI evaluation. External validation remains not performed.
- Visual QA: Figures 9–11 were inspected; the final one-page mentor memo was rendered and checked for clipping, overlap and legibility.
- Archive: completed-output checksums and ZIP contents were verified during assembly; the archive SHA-256 is recorded beside the ZIP.

Evidence is in `docs/review_bundle_v2/validation-v2-20260907/`, including `repeatability.json`, `validation_result.json`, `TEST_RESULTS.txt`, per-image results, logits, temperature artifacts and frozen source snapshots. Neither this QA nor the improved internal calibration establishes anatomical correctness, patient independence, absence of source bias or clinical validity. Nothing was sent to Dr. Cockroft.
