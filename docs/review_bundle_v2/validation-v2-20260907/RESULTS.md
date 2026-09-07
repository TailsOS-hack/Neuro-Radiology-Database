# Follow-up validation results — validation-v2-20260907

Research-only follow-up on already inspected public-image test data. The validation split previously influenced checkpoint selection. No new independent cohort or clinical calibration is established.

## Masking controls

The same 128 images per specialist were evaluated. Positive paired differences mean that top-CAM masking reduced the original target-class confidence more than the control; negative differences mean less. All results below are descriptive image-level means.

| Specialist | Fraction | Replacement | Top-CAM drop | Random-pixel drop | Shape-control drop | Paired random difference | Paired shape difference |
| --- | --- | --- | --- | --- | --- | --- | --- |
| dementia | 10% | gaussian_blur | 0.1566 | 0.3689 | 0.0985 | -0.2123 | 0.0582 |
| dementia | 10% | imagenet_mean | 0.1932 | 0.6552 | 0.1153 | -0.4620 | 0.0779 |
| dementia | 20% | gaussian_blur | 0.3927 | 0.5847 | 0.2113 | -0.1920 | 0.1814 |
| dementia | 20% | imagenet_mean | 0.4453 | 0.6865 | 0.2416 | -0.2412 | 0.2037 |
| dementia | 30% | gaussian_blur | 0.5420 | 0.6576 | 0.3608 | -0.1156 | 0.1812 |
| dementia | 30% | imagenet_mean | 0.6037 | 0.7058 | 0.3976 | -0.1021 | 0.2061 |
| tumor | 10% | gaussian_blur | 0.1273 | 0.0493 | 0.0793 | 0.0780 | 0.0480 |
| tumor | 10% | imagenet_mean | 0.1653 | 0.6338 | 0.0959 | -0.4685 | 0.0694 |
| tumor | 20% | gaussian_blur | 0.2589 | 0.2368 | 0.1695 | 0.0221 | 0.0894 |
| tumor | 20% | imagenet_mean | 0.3027 | 0.6729 | 0.2155 | -0.3702 | 0.0873 |
| tumor | 30% | gaussian_blur | 0.3806 | 0.4844 | 0.2880 | -0.1038 | 0.0926 |
| tumor | 30% | imagenet_mean | 0.4548 | 0.6680 | 0.3551 | -0.2132 | 0.0997 |

At 20% masking with ImageNet-mean replacement, paired shape-control differences were 0.2037 for dementia and 0.0873 for tumor. Corresponding random-pixel differences were -0.2412 and -0.3702. These comparisons show sensitivity to the chosen control, not anatomical validation. The v2 random masks are independently seeded by image and fraction; their values need not equal the original v1 masks.

D4 rotations/reflections preserve area and shape but may retain central support. Gaussian blur uses a 31-pixel kernel and sigma 10. Neither baseline eliminates out-of-distribution perturbations. See Figure 9 and per-class means, SDs, control counts and individual drops in the structured evidence.

## Model randomization

Across 2,304 randomized-map attempts, 385 were flat or failed. Such maps are counted and excluded from correlation/overlap averages; they are never represented as zero correlation. The accepted target class remains fixed. Figure 10 shows conditional means among valid maps.

| Specialist | Seed | Stage | Valid / attempted | Spearman mean | Top-20% IoU mean |
| --- | --- | --- | --- | --- | --- |
| dementia | 42 | full | 128 / 128 | 0.0482 | 0.1463 |
| dementia | 42 | head_final | 128 / 128 | 0.1525 | 0.2046 |
| dementia | 42 | head | 128 / 128 | 0.0921 | 0.1902 |
| dementia | 43 | full | 127 / 128 | -0.1220 | 0.0737 |
| dementia | 43 | head_final | 128 / 128 | 0.2145 | 0.2471 |
| dementia | 43 | head | 128 / 128 | 0.0397 | 0.1752 |
| dementia | 44 | full | 128 / 128 | 0.1095 | 0.1251 |
| dementia | 44 | head_final | 128 / 128 | -0.0378 | 0.1319 |
| dementia | 44 | head | 128 / 128 | -0.0006 | 0.1643 |
| tumor | 42 | full | 0 / 128 | unavailable | unavailable |
| tumor | 42 | head_final | 128 / 128 | -0.1553 | 0.0814 |
| tumor | 42 | head | 128 / 128 | 0.1832 | 0.3592 |
| tumor | 43 | full | 0 / 128 | unavailable | unavailable |
| tumor | 43 | head_final | 128 / 128 | 0.2492 | 0.2800 |
| tumor | 43 | head | 128 / 128 | 0.0095 | 0.1936 |
| tumor | 44 | full | 0 / 128 | unavailable | unavailable |
| tumor | 44 | head_final | 128 / 128 | 0.1938 | 0.2618 |
| tumor | 44 | head | 128 / 128 | 0.2684 | 0.3065 |

Randomization is a model-dependence diagnostic, not proof of explanation faithfulness or biological relevance. The three seeds quantify initialization sensitivity; no clinical pass threshold is applied.

## Research-only temperature scaling

| Model / prediction rule | Temperature | Original ECE | Calibrated ECE | Original NLL | Calibrated NLL | Original Brier | Calibrated Brier | Original accuracy | Calibrated accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| binary | 0.05118 | 0.118865 | 0.000000 | 0.127835 | 0.000000 | 0.032546 | 0.000000 | 1.000000 | 1.000000 |
| tumor | 0.63243 | 0.037228 | 0.012143 | 0.150648 | 0.128096 | 0.040875 | 0.038639 | 0.979239 | 0.979239 |
| dementia | 0.46121 | 0.038596 | 0.000581 | 0.041851 | 0.003539 | 0.004085 | 0.001641 | 0.999092 | 0.999092 |
| eight_class | 0.40783 | 0.132689 | 0.001117 | 0.153358 | 0.017837 | 0.028466 | 0.005058 | 0.997171 | 0.997171 |
| hierarchical_argmax | component temperatures | 0.152561 | 0.001732 | 0.185022 | 0.021097 | 0.047376 | 0.006857 | 0.996293 | 0.996293 |
| hierarchical_routed | component temperatures | 0.152561 | 0.001732 | 0.185022 | 0.021097 | 0.047376 | 0.006857 | 0.996293 | 0.996293 |

Temperatures were fitted on validation logits and frozen before test inference. Hierarchical routed predictions are reported separately from the combined-probability argmax; the routed confidence is the joint probability assigned to the routed class. NLL and Brier describe the full probability vector. Zeroes above are rounded, not guarantees of perfect calibration.

Optimization records (including failure, fallback and boundary flags) are preserved for all models. The binary router temperature is near the lower search bound; near-perfect source routing and concentrated probabilities do not rule out dataset-source shortcuts. Figure 11 shows reliability diagrams.

## Limits and external readiness

Follow-up on an already inspected public-image test set; validation influenced checkpoint selection. No patient independence, anatomical validation, absence of source bias, calibrated clinical probability or external validity is established. No retraining or application changes.

External validation remains **not performed**. Only a synthetic cohort fixture has exercised the evaluation interface. Patient identifiers, labels, cohort provenance and suitable imaging must come from a separately acquired cohort. Internal calibration does not change application behavior.

Methods: [Grad-CAM sanity checks](https://proceedings.neurips.cc/paper/2018/hash/294a8ed24b1ad22ec2e7efea049b8737-Abstract.html); [temperature scaling](https://proceedings.mlr.press/v70/guo17a.html).
