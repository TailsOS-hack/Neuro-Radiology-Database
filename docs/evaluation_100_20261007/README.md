# Current model and Grad-CAM verification — 7 October 2026

The current binary router and dementia specialist correctly classify **100 of 100** images in the requested radiologist benchmark. All 100 explanations are finite, nonflat 7×7 maps, and specialist logits are identical before and after Grad-CAM. The existing checkpoint weights already exceed the requested 95% threshold; no retraining, test-label fitting, checkpoint selection, or weight changes were needed.

The historical `data/evaluation/model_results/model_predictions.csv` scores 80/100 and predates the current checkpoints. It remains unchanged. The new result belongs to the current pipeline and is saved separately, with image, checkpoint, source-code, and ground-truth hashes:

- [Metrics and environment](current/metrics.json)
- [All 100 predictions](current/predictions.csv)
- [All 100 native Grad-CAM maps](current/gradcam_maps.npz)
- [File and decoded-RGB overlap audit](cohort_audit.json)
- [Both specialist architectures: 16-image verification](specialist_gradcam/verification.json)
- [Current descriptive comparison chart](comparison/accuracy_comparison_comprehensive.png)

| Class | Correct / images |
| --- | ---: |
| MildDemented | 23 / 23 |
| ModerateDemented | 23 / 23 |
| NonDemented | 27 / 27 |
| VeryMildDemented | 27 / 27 |

## What changed

Grad-CAM now uses a direct gradient of the selected logit with respect to the feature map, avoiding module backward hooks that conflict with in-place activations. It works inside both `no_grad()` and `inference_mode()`, leaves existing parameter/input gradients untouched, restores each module's original training mode, and removes its own hooks on success or failure. Invalid inputs and nonfinite or flat maps fail explicitly rather than producing a misleading overlay. The distinction between these gradient contexts follows the [PyTorch autograd documentation](https://docs.pytorch.org/docs/stable/generated/torch.autograd.grad_mode.inference_mode.html).

GUI specialists now apply their own checkpoint's image size and the pipeline normalization to both classification and explanation. Reloading legacy full-model checkpoints resets their preprocessing correctly. Tests exercise nondefault checkpoint sizes and verify that classification and explanation receive identical tensors.

The offline Grad-CAM verifier now loads the actual trained checkpoints, rather than importing the GUI and downloading ImageNet backbones with random classifier heads. Both specialist architectures passed 16 source-image checks (two per class), repeated calls under different gradient contexts, hook cleanup, parameter-gradient preservation, and unchanged logits. Analytical tests cover failure paths and in-place activation compatibility.

All **51 tests pass** in the model environment. The lightweight artifact environment passes its 31 applicable tests and explicitly skips 20 model/plotting dependency tests. Both frozen explainability evidence verifiers pass. The publication package still reports its existing decoded-pixel overlap failure (23 checks: one fail, four warnings); this work does not remove that research limitation.

## Limits of the score

All 100 benchmark images are byte-identical copies of source images. A fresh audit decoded all 51,023 source files and found the same matches using decoded RGB pixels. The historical strict manifest places 66 in training, 9 in validation, and 25 only in test. The perceptual manifest places 63 in training, 12 in validation, and 25 only in test. No label conflicts were found. Checkpoints lack embedded manifest hashes; the default dementia checkpoint's history matches the historical deduplicated regularized run, and the audit records both frozen manifests explicitly.

Consequently, **100% is a reused-benchmark result, not independent held-out or clinical accuracy**. The score does not resolve the [September review's publication limitations](../FINAL_REVIEW_20260920.md). A clean generalization claim requires a separately reserved cohort with patient/original-image-family independence. Further training to optimize these 100 images would not establish that claim.

A finite heatmap establishes working explanation software, not accurate anatomical localization. Native maps are retained; no heatmap was changed to appear more anatomically plausible. The trained-checkpoint runs used CPU; GPU and DirectML behavior were not evaluated in this run.

## Reproduction

Install the project's model dependencies in a Python environment and hydrate the Git LFS checkpoints. Run from the repository root:

```bash
python scripts/evaluate_radiologist_cohort.py --output /tmp/neuro-100-new-run
python scripts/audit_radiologist_cohort.py --output /tmp/neuro-100-new-audit.json
python scripts/test_grad_cam_contract.py
python -m unittest discover -s tests -p 'test_*.py' -v
python scripts/verify_grad_cam.py --output-dir /tmp/neuro-gradcam-new-run
```

The evaluator requires exactly 100 uniquely named, labelled cases and fails for missing images, unknown labels, accuracy at or below 95%, missing/flat/nonfinite maps, or changed specialist logits. It writes into a new directory so archived results cannot be overwritten accidentally. The audit separately verifies complete image/label coverage and fails on unreadable source images.

To plot the new predictions while preserving historical charts:

```bash
python data_visualization/compare_rad_vs_ai.py \
  --model-predictions docs/evaluation_100_20261007/current/predictions.csv \
  --output-dir /tmp/neuro-100-comparison
```

This comparison is descriptive only: the model's prior exposure makes these 100 cases unsuitable for a claim of independent superiority over radiologists.
