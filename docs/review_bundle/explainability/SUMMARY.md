# Quantitative explainability summary

Exploratory, class-balanced strict-test sample; descriptive image-level statistics only. No patient-independent confidence intervals are inferred.

| Specialist | Sample / test population | Valid CAMs | Top-20% drop | Random-20% drop | Paired advantage | Border mass |
| --- | --- | --- | --- | --- | --- | --- |
| tumor | 128 / 1445 | 128 | 0.3027 | 0.6808 | -0.3780 | 0.0754 |
| dementia | 128 / 8806 | 128 | 0.4453 | 0.6901 | -0.2448 | 0.1992 |

Drops are original minus perturbed predicted-class softmax score; negative values mean masking increased the score. Paired advantage is top-CAM drop minus the mean of the configured seeded random masks (default: five). This is a single-fraction perturbation diagnostic, not deletion AUC. Masked pixels use the ImageNet mean RGB baseline. Random masks are spatially scattered and are not shape-matched controls.

Border mass is the fraction of upsampled CAM mass in the outer 10% image frame (approximately 35.4% of pixels at 224×224). The image frame is not a brain/background segmentation. Neither this measure nor masking establishes lesion localization, causal biological relevance, absence of source bias, or clinical validity.

See per_image.csv for failures, selection roles, image hashes and archived-probability agreement; summary.json includes per-class counts, means and SDs. The diagnostic gallery is selected separately and excluded from quantitative aggregates unless an image also belongs to the seeded sample.
