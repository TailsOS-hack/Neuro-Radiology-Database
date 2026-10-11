> Portable public derivative of the audited local source. Clinical subject-level probability rows and operational identifiers are omitted. Scientific aggregate results are unchanged. SHA256 values quoted from the source refer to original frozen files; export hashes are in EXPORT_MANIFEST.json.

# Frozen analysis handoff

Both frozen CPU analyses completed with exact alignment/provenance checks and 2,000 paired bootstrap replicates. No model inference or changes to frozen scripts/protocols occurred. Exact commands and output hashes are in each output directory’s execution_record.json.

The individual new models did not improve primary accuracy over the old baseline. The stage-only ensemble corrected the old model’s one primary error, but its paired gain interval includes zero. Clinical performance is weak and unstable; these results do not support an overall stronger or clinically ready model.

## All stage models and reused original100

| Model | Primary1058 | Sensitivity756 | Reused original100 | Original100 BA % | Original100 macro F1 % |
| --- | --- | --- | --- | --- | --- |
| Old baseline | 1057/1058 (99.905%) | 755/756 (99.868%) | 100/100 (100.000%) | 100.000 | 100.000 |
| stage_only seed 20261011 (candidate_1) | 1055/1058 (99.716%) | 753/756 (99.603%) | 99/100 (99.000%) | 99.074 | 99.074 |
| multitask seed 20261011 (candidate_2) | 1054/1058 (99.622%) | 752/756 (99.471%) | 100/100 (100.000%) | 100.000 | 100.000 |
| stage_only seed 20261012 (candidate_3) | 1057/1058 (99.905%) | 755/756 (99.868%) | 100/100 (100.000%) | 100.000 | 100.000 |
| multitask seed 20261012 (candidate_4) | 1056/1058 (99.811%) | 754/756 (99.735%) | 99/100 (99.000%) | 99.074 | 99.074 |
| stage_only seed 20261013 (candidate_5) | 1057/1058 (99.905%) | 755/756 (99.868%) | 100/100 (100.000%) | 100.000 | 100.000 |
| multitask seed 20261013 (candidate_6) | 1055/1058 (99.716%) | 754/756 (99.735%) | 99/100 (99.000%) | 99.074 | 99.074 |
| Stage-only ensemble (supplementary) | 1058/1058 (100.000%) | 756/756 (100.000%) | 100/100 (100.000%) | 100.000 | 100.000 |
| Multitask stage ensemble (supplementary) | 1056/1058 (99.811%) | 754/756 (99.735%) | 100/100 (100.000%) | 100.000 | 100.000 |

## Fixed-reader descriptive comparison on reused original100

| Reader | Correct / 100 | Accuracy % | Balanced accuracy % | Macro F1 % |
| --- | --- | --- | --- | --- |
| Kalapos | 37 | 37.000 | 36.353 | 36.382 |
| Kanekar | 44 | 44.000 | 43.961 | 43.135 |
| Thamburaj | 38 | 38.000 | 39.050 | 38.264 |

All three readers have 100 complete cases. Fixed-panel mean accuracy: 39.667%; balanced accuracy: 39.788%; macro F1: 39.260%. Every old/new model-minus-reader contrast and paired interval is preserved in reader_paired_differences.csv. This image-label task is historically reused; it does not demonstrate clinical superiority over neuroradiologists.

## Prespecified condition means and seed spread

| Cohort | Condition | Accuracy % ± seed SD | Balanced accuracy % ± seed SD | Macro F1 % ± seed SD |
| --- | --- | --- | --- | --- |
| internal_same_case | stage_only | 99.842 ± 0.109 | 99.834 ± 0.131 | 99.831 ± 0.124 |
| internal_same_case | multitask | 99.716 ± 0.095 | 99.696 ± 0.100 | 99.706 ± 0.100 |
| original100 | stage_only | 99.667 ± 0.577 | 99.691 ± 0.535 | 99.691 ± 0.535 |
| original100 | multitask | 99.333 ± 0.577 | 99.383 ± 0.535 | 99.383 ± 0.535 |
| old_hash_group_unexposed_sensitivity | stage_only | 99.780 ± 0.153 | 99.779 ± 0.165 | 99.777 ± 0.160 |
| old_hash_group_unexposed_sensitivity | multitask | 99.647 ± 0.153 | 99.643 ± 0.152 | 99.650 ± 0.152 |

SD describes three fixed training runs, not a confidence interval.

## internal_same_case paired accuracy differences

Differences and 95% conditional paired hash-group bootstrap intervals are percentage points, unadjusted for multiple comparisons.

| Comparison | Difference pp | 95% interval pp |
| --- | --- | --- |
| candidate_1_minus_old_baseline | -0.189 | [-0.570, 0.184] |
| candidate_2_minus_old_baseline | -0.284 | [-0.754, 0.094] |
| candidate_3_minus_old_baseline | 0.000 | [-0.285, 0.281] |
| candidate_4_minus_old_baseline | -0.095 | [-0.389, 0.191] |
| candidate_5_minus_old_baseline | 0.000 | [-0.282, 0.280] |
| candidate_6_minus_old_baseline | -0.189 | [-0.477, 0.000] |
| candidate_2_minus_candidate_1 | -0.095 | [-0.652, 0.379] |
| candidate_4_minus_candidate_3 | -0.095 | [-0.388, 0.192] |
| candidate_6_minus_candidate_5 | -0.189 | [-0.568, 0.187] |
| ensemble_stage_only_minus_old_baseline (supplementary) | 0.095 | [0.000, 0.290] |
| ensemble_multitask_minus_old_baseline (supplementary) | -0.095 | [-0.460, 0.192] |
| ensemble_multitask_minus_ensemble_stage_only (supplementary) | -0.189 | [-0.476, 0.000] |

## old_hash_group_unexposed_sensitivity paired accuracy differences

Differences and 95% conditional paired hash-group bootstrap intervals are percentage points, unadjusted for multiple comparisons.

| Comparison | Difference pp | 95% interval pp |
| --- | --- | --- |
| candidate_1_minus_old_baseline | -0.265 | [-0.797, 0.264] |
| candidate_2_minus_old_baseline | -0.397 | [-1.051, 0.133] |
| candidate_3_minus_old_baseline | 0.000 | [-0.398, 0.396] |
| candidate_4_minus_old_baseline | -0.132 | [-0.652, 0.266] |
| candidate_5_minus_old_baseline | 0.000 | [-0.397, 0.396] |
| candidate_6_minus_old_baseline | -0.132 | [-0.399, 0.000] |
| candidate_2_minus_candidate_1 | -0.132 | [-0.797, 0.531] |
| candidate_4_minus_candidate_3 | -0.132 | [-0.535, 0.266] |
| candidate_6_minus_candidate_5 | -0.132 | [-0.533, 0.266] |
| ensemble_stage_only_minus_old_baseline (supplementary) | 0.132 | [0.000, 0.399] |
| ensemble_multitask_minus_old_baseline (supplementary) | -0.132 | [-0.653, 0.267] |
| ensemble_multitask_minus_ensemble_stage_only (supplementary) | -0.265 | [-0.664, 0.000] |

Primary1058 is exact-row held out; 302 rows have observed old training/validation hash-group mates. Sensitivity756 excludes these but hash groups are not patient identity. Original100 has old exposure of 66 train / 9 validation / 25 test images. Identical perfect original100 predictions yield degenerate paired bootstrap intervals, which do not establish population certainty or equivalence.

## Clinical AD/CN by source and subject

Only multitask heads are applicable; the stage-only models and old baseline have no matching AD/CN endpoint. MCI remains prediction-only. Exact intervals use Clopper–Pearson binomial limits.

### external__openneuro_ds007561

| Model | Accuracy | Balanced accuracy % | Macro F1 % | AD sensitivity successes/n [95% %] | CN specificity successes/n [95% %] |
| --- | --- | --- | --- | --- | --- |
| multitask seed 20261011 (candidate_2) | 14/15 (93.333%) | 50.000 | 48.276 | 0/1 [0.000, 97.500] | 14/14 [76.836, 100.000] |
| multitask seed 20261012 (candidate_4) | 7/15 (46.667%) | 71.429 | 40.000 | 1/1 [2.500, 100.000] | 6/14 [17.661, 71.139] |
| multitask seed 20261013 (candidate_6) | 13/15 (86.667%) | 46.429 | 46.429 | 0/1 [0.000, 97.500] | 13/14 [66.132, 99.819] |
| ensemble_multitask_AD_CN | 12/15 (80.000%) | 42.857 | 44.444 | 0/1 [0.000, 97.500] | 12/14 [57.187, 98.221] |

### test__besga_zenodo_3935636

| Model | Accuracy | Balanced accuracy % | Macro F1 % | AD sensitivity successes/n [95% %] | CN specificity successes/n [95% %] |
| --- | --- | --- | --- | --- | --- |
| multitask seed 20261011 (candidate_2) | 5/11 (45.455%) | 57.143 | 41.071 | 1/7 [0.361, 57.872] | 4/4 [39.764, 100.000] |
| multitask seed 20261012 (candidate_4) | 7/11 (63.636%) | 66.071 | 63.333 | 4/7 [18.405, 90.101] | 3/4 [19.412, 99.369] |
| multitask seed 20261013 (candidate_6) | 4/11 (36.364%) | 44.643 | 34.188 | 1/7 [0.361, 57.872] | 3/4 [19.412, 99.369] |
| ensemble_multitask_AD_CN | 5/11 (45.455%) | 57.143 | 41.071 | 1/7 [0.361, 57.872] | 4/4 [39.764, 100.000] |

External OpenNeuro contains 14 CN, one AD and five unscored MCI subjects. Candidate2 predicts every evaluable subject CN: 93.33% raw accuracy therefore corresponds to 50% balanced accuracy and 0/1 AD sensitivity. The ensemble also misses the sole AD, with sensitivity 0% [0%, 97.5%]. Candidate4 detects that one AD but its exact sensitivity interval is [2.5%, 100%], and it produces eight CN false positives. No stable external clinical conclusion follows.

Besga held-out data contain four CN and seven AD subjects. The ensemble detects only one of seven AD subjects. The aggregate performance cannot justify a real-world robustness or deployment claim.

## Validation-only preference fixed before evaluation

Preferred condition: stage_only; exploratory representative: candidate_3 (stage_only seed20261012). Mean validation NLL stage-only 0.03639474; multitask 0.05031494. This pre-evaluation preference does not select a test winner, and all six runs plus both supplements remain reported.

No further model tuning or reselection was performed after viewing these results.
