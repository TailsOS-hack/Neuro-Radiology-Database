# Post-hoc fixed-checkpoint image stress audit

This is an additional descriptive analysis after the primary experiment. No checkpoint, threshold, architecture or training choice was changed. All seven frozen models are reported.

The 13 fixed scenarios cover clean inputs, modest intensity/contrast shifts, display noise, blur, lower resolution and JPEG artifacts. They do not simulate MRI acquisition physics or establish clinical robustness.

| Cohort | Model | Local clean accuracy | Lowest scenario accuracy | Scenario(s) | Largest class-flip fraction |
|---|---|---:|---:|---|---:|
| internal_same_case | old_baseline | 0.9991 | 0.9858 | noise_sd0.02 | 0.0132 |
| internal_same_case | candidate_1 | 0.9972 | 0.9887 | blur_sigma1.0 | 0.0104 |
| internal_same_case | candidate_3 | 0.9991 | 0.9868 | noise_sd0.02 | 0.0132 |
| internal_same_case | candidate_5 | 0.9991 | 0.9820 | blur_sigma1.0 | 0.0180 |
| internal_same_case | candidate_2 | 0.9962 | 0.9471 | blur_sigma1.0 | 0.0510 |
| internal_same_case | candidate_4 | 0.9981 | 0.9612 | blur_sigma1.0 | 0.0378 |
| internal_same_case | candidate_6 | 0.9972 | 0.9688 | noise_sd0.02 | 0.0331 |
| original100 | old_baseline | 1.0000 | 1.0000 | clean, brightness_0.90, brightness_1.10, contrast_0.90, contrast_1.10, noise_sd0.01, noise_sd0.02, blur_sigma0.5, blur_sigma1.0, downsample_192, downsample_160, jpeg_quality90, jpeg_quality70 | 0.0000 |
| original100 | candidate_1 | 0.9900 | 0.9600 | noise_sd0.02 | 0.0300 |
| original100 | candidate_3 | 1.0000 | 0.9600 | noise_sd0.02 | 0.0400 |
| original100 | candidate_5 | 1.0000 | 0.9700 | noise_sd0.02 | 0.0300 |
| original100 | candidate_2 | 1.0000 | 0.9400 | blur_sigma1.0 | 0.0600 |
| original100 | candidate_4 | 0.9900 | 0.9500 | noise_sd0.02 | 0.0400 |
| original100 | candidate_6 | 0.9900 | 0.9500 | noise_sd0.02, blur_sigma1.0 | 0.0600 |

The minimum above describes this fixed tested grid, not a selected model or an estimated deployment failure rate.

## Clean reproducibility check

Preset tolerance: absolute 1e-05, relative 0.0001. Across all models and cohorts: 6000 case-model pairs have at least one probability outside tolerance; 0 have different predicted classes. Maximum absolute probability difference: 0.0099551678.

See clean_archived_comparison.json for every discrepancy. The local environment differs from the original CUDA environment; original archived predictions remain authoritative for the primary report.

## Interpretation limits

- Original100 is historically reused and has old training exposure.
- Primary exact-row test membership does not establish patient independence.
- Stress parameters were chosen after the primary experiment; this supplement is descriptive and post-hoc.
- No MRI acquisition physics, patient variation, diagnosis shift, modality shift, prospective setting, or new clinical cohort is simulated.
- No acceptance threshold or model superiority is inferred from this audit.

No confidence intervals or clinical acceptance thresholds were added. Plotted lines connect only the actual measured levels. Corruptions are shared across models and have fixed per-case seeds.

Implementation references: [torchvision intensity transforms](https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.ColorJitter.html), [Pillow GaussianBlur](https://pillow.readthedocs.io/en/stable/reference/ImageFilter.html#PIL.ImageFilter.GaussianBlur), and [CLAIM reporting checklist](https://rsna.github.io/claim/checklist.html).
