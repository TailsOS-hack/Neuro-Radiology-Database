# Dementia manuscript mentor review bundle

Start with [the draft packet for Dr. Cockroft](DR_COCKROFT_DRAFT_PACKET.md), then open the [Grad-CAM gallery](explainability/index.html) in a browser. The [quantitative summary](explainability/SUMMARY.md) covers both specialists. The dementia-focused review retains the combined tumor/dementia benchmark and its single-head comparator; no new dementia-only or clinical-validation claim is introduced.

## Reproduce

From the repository root, with the public images at the paths in the committed strict manifest:

```bash
# Recover the accepted weights; do not substitute sensitivity or random weights.
git lfs install --local
git lfs pull --include='models/brain_tumor_classifier.pt,models/alzheimers_classifier.pt'
python3 -m venv .venv-review
.venv-review/bin/pip install -r docs/review_bundle/requirements-lock.txt
.venv-review/bin/python scripts/test_grad_cam_contract.py
.venv-review/bin/python scripts/build_explainability_bundle.py --seed 42 --per-class 32 --random-repeats 5
.venv-review/bin/python scripts/check_explainability_bundle.py
.venv-review/bin/python scripts/build_publication_figures.py
.venv-review/bin/python scripts/check_publication_package.py
```

Run the builder into an empty output directory when changing sampling parameters (`--output-dir`). Defaults regenerate the manuscript bundle. CPU inference is deterministic within the recorded environment; numerical differences across runtimes are possible. The builder refuses LFS pointers, incorrect class maps, evidence/manifest mismatches, or sampled probabilities differing from the accepted archive by more than 0.0001. This probability concordance is a sampled consistency check, not a new full-test evaluation. Only trusted project checkpoint files should be loaded because the existing loader deserializes PyTorch checkpoints with `weights_only=False`.

The one-page PDF cover memo is in `output/pdf/Dr_Cockroft_review_memo.pdf`. To rebuild it, install `reportlab==4.4.9` and run `python3 scripts/build_mentor_packet_pdf.py`. Run `python3 scripts/package_mentor_review.py` after all checks to rebuild the ZIP archive and SHA-256 file manifest. The archive preserves repository-relative paths; extract it before opening `docs/review_bundle/explainability/index.html`. It includes review evidence and source code, but not raw datasets or checkpoint binaries. Dataset access and Git LFS weights from the repository are required to rerun model inference.

## Protocol fixed before inspecting CAMs

- Sampling: Python RNG seed 42, 32 images without replacement from each sorted true-class pool, separately for each specialist: 128 images per model. No validation or training images enter the analysis.
- Diagnostic gallery: lowest-confidence correct case and highest-confidence error per true class, using the archived strict-test probabilities. No error panel is invented for classes with zero errors. Gallery cases do not enter the quantitative sample unless independently sampled.
- Target: predicted-class **logit**, final `features[-1]` block, global mean gradient weights, weighted activations followed by ReLU and per-image min/max normalization. Native maps are 7×7; overlays bilinearly upsample the map to the 224×224 input. Input preprocessing matches checkpoint evaluation (RGB, resize, ImageNet normalization); no augmentation.
- Perturbation: mask exactly floor(20% × 224 × 224) pixels of highest upsampled CAM intensity. Replace with ImageNet RGB mean; retain the original predicted target for all scores. Compare the confidence drop with five independently seeded random pixel masks of the same size. Stable pixel ordering resolves ties. No tuning against the test outcomes.
- Aggregation: means and sample SDs across valid CAMs, with class-specific results and all failed/flat maps counted. The class-balanced aggregate is not prevalence weighted. Image-level observations may be dependent, so no patient-level inference, significance tests, or patient-independent confidence intervals are claimed.
- Border diagnostic: mass in a 22-pixel frame (35.43% of the input area), not an anatomical or background mask.

## Evidence and limits

`per_image.csv` records selection roles, predicted and true labels, confidence, metrics, source-image SHA-256, map paths, and probability agreement. `provenance.json` records weights, manifest, evidence and script hashes plus software versions. `maps/` preserves float32 native CAMs. `summary.json` includes per-class counts and SDs; `cases/` holds readable individual panels. Figures 8a–8b are assembled from those panels.

Grad-CAM is a coarse attribution method; see [Selvaraju et al., Grad-CAM](https://arxiv.org/abs/1610.02391). No segmentation labels, expert anatomical scoring, parameter-randomization study, or external cohort are provided here. Random pixel masks are not shape matched to contiguous CAM regions, and replacement can create out-of-distribution inputs. Positive masking effects do not demonstrate causal biological relevance. Negative or weak results must remain reported.

All earlier boundaries remain: public augmented dementia data; unavailable patient identifiers and possible patient-level leakage; source-domain bias; imperfect calibration; absent external validation. This bundle supports internal mentor review only. Example MRI panels are derived from public source images; review their attribution and redistribution terms before public submission. The packet is a draft and has not been sent.
