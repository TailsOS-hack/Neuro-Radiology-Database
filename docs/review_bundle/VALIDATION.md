# Review bundle validation

Validated locally on September 7, 2026. Exact runtime versions and source hashes are recorded in `explainability/provenance.json`.

- **Accepted checkpoint recovery:** Git LFS objects matched their tracked SHA-256 identifiers. Checkpoint creation timestamps matched the May 12 accepted exact-deduplicated run. Checkpoints were used for evaluation only; no retraining or model-selection changes.
- **Prediction concordance:** all 270 distinct evaluated cases matched archived predicted labels. Maximum absolute probability differences: tumor 0.000023872; dementia 0.000003231 (threshold 0.0001). This is sampled concordance, not a repeat of full-test performance evaluation.
- **Sampling:** 128 randomly selected images per specialist, 32 per true class, seed 42; 256 quantitative images in total. The separate 15-panel gallery adds 14 unique images because one case overlaps the quantitative sample.
- **CAM availability:** 256/256 quantitative cases valid; all gallery maps also valid. No flat maps or failures were silently removed.
- **Repeatability:** two complete CPU passes produced byte-identical `per_image.csv` files across all 270 evaluated cases. Between passes, display interpolation was aligned to evaluation preprocessing and additional provenance/completion checks were added; numerical attribution and perturbation calculations were unchanged.
- **Analytical unit check:** `scripts/test_grad_cam_contract.py` passed known-map calculation, repeatability, operation under `no_grad`, training-mode restoration, hook cleanup, invalid-target handling and flat-map handling. This is a mechanics test only.
- **Integrity:** `scripts/check_explainability_bundle.py` passed manifest membership, class counts, means, image/source/evidence hashes, checkpoint identifiers, probability tolerance and figure presence.
- **Publication package:** `scripts/check_publication_package.py` reported 17 checks, zero failures and one retained calibration warning; overall `READY_WITH_LIMITATIONS`.
- **Visual review:** individual dementia and tumor panels and the assembled dementia figure were inspected. The one-page PDF cover memo was rendered to PNG and checked for clipping, overlap and legibility.

The tracked model files remain their original Git LFS pointers; downloaded checkpoint objects were used locally and are excluded from the review archive. Follow the README recovery commands to hydrate them for another run. No communication was sent to Dr. Cockroft.

These checks establish software and artifact consistency, not anatomical correctness, patient independence, calibrated clinical probability, absence of source bias, or external validity.
