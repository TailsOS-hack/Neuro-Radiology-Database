# Round 2 results and remaining clinical limitations

**Clinical superiority and real-world readiness were not demonstrated.** The new fixed clinical ensemble classified all 19 fresh Harvard participants as AD, including all 10 controls. It scored 9/19, versus 10/19 for the old fixed clinical ensemble. All three new individual models made the same all-AD decisions on this cohort. Improved reused-cohort scores do not resolve that external failure. The original production classifier was not replaced.

The research package is complete: all six training runs stopped on validation plateau; the selected checkpoints were frozen before the corresponding evaluations; results were retrieved and verified; all figures were reviewed. This execution completion does not mean that the medical-readiness objective was achieved.

[All 49 new graph pages](combined_graphs/Round2_All_Graphs_Atlas.pdf) · [page index](combined_graphs/atlas_index.csv) · [115 diagnostic Grad-CAM cases](artifacts/stage/attribution_diagnostics/gallery.html) · [complete artifact/curve index](artifacts/README.md) · [Methods](Methods.md) · [data provenance and limitations](DATA_AND_LIMITATIONS.md).

## Separate patient-level AD/control experiment

The prespecified primary comparison is the uniform three-model new ensemble versus the uniform three-model old clinical ensemble. The fixed threshold was 0.5. Old models retain mean slice probabilities; new models use the softmax of mean slice logits. This is a comparison of complete pipelines. The four-class legacy reader labels do not score this separate binary model.

| Clinical cohort | Old ensemble correct / total; accuracy (exact 95% CI) | New ensemble correct / total; accuracy (exact 95% CI) | Old → new balanced accuracy |
|---|---|---|---:|
| Fresh Harvard: 9 AD, 10 CN | 10/19; 52.6% (28.9–75.6) | 9/19; 47.4% (24.4–71.1) | 55.0% → 50.0% |
| Reused Besga test: 7 AD, 4 CN | 5/11; 45.5% (16.7–76.6) | 10/11; 90.9% (58.7–99.8) | 57.1% → 87.5% |
| Reused OpenNeuro: 1 AD, 14 CN | 12/15; 80.0% (51.9–95.7) | 14/15; 93.3% (68.1–99.8) | 42.9% → 50.0% |

On Harvard, new-ensemble AD sensitivity was 100% (exact 95% CI 66.4–100) and CN specificity was **0%** (0–30.8). The old ensemble's specificity was 10% (0.3–44.5). The new ensemble's ROC AUC was 0.8444 versus 0.8222 for the old ensemble, but that ranking measure does not repair the failed fixed-threshold classification or demonstrate superiority. No post-test threshold adjustment was made.

On OpenNeuro, the new ensemble predicted **CN for all 15 participants** and missed the sole AD participant. AD sensitivity was 0%, with exact 95% CI 0–97.5%. High accuracy in this highly imbalanced cohort therefore does not establish AD detection. Its diagnosis-stratified bootstrap repeatedly samples that same AD person; a conditional sensitivity-difference interval of zero width must not be interpreted as population precision.

The paired new-minus-old ensemble accuracy differences were:

| Cohort | Difference in percentage points | Conditional paired 95% interval |
|---|---:|---:|
| Fresh Harvard | −5.3 | −15.8 to 0.0 |
| Reused Besga | +45.5 | +18.2 to +63.6 |
| Reused OpenNeuro | +13.3 | 0.0 to +33.3 |

These intervals use 10,000 diagnosis-stratified patient bootstrap draws. They are conditional, unadjusted exploratory summaries, not confirmatory clinical-superiority evidence. [All eight model/ensemble results with exact intervals](artifacts/clinical/report/README.md), [all 168 scenario metrics](artifacts/clinical/report/all_model_scenario_metrics.csv), and [all 12 patient-paired contrasts](artifacts/clinical/report/analysis.json) remain available.

All new seeds are retained, without choosing a winner from the test results:

| New clinical model | Harvard correct / 19 | Besga correct / 11 | OpenNeuro correct / 15 |
|---|---:|---:|---:|
| Seed 20261021 | 9 | 10 | 14 |
| Seed 20261022 | 9 | 10 | 10 |
| Seed 20261023 | 9 | 9 | 15 |
| Fixed ensemble | 9 | 10 | 14 |

Seed 20261023's 15/15 on the reused OpenNeuro cohort does not justify replacing the prespecified ensemble after observing outcomes. Fifteen MCI participants received prediction-only scores and were excluded from binary accuracy and diagnostic claims. Clinical statistics were independently recomputed from the complete 3,360-record grid. Individual clinical records are excluded from this public package.

## Reused four-class image benchmarks and reader comparison

The validation-selected attribution representative was stage seed **20261011**, fixed before repeated benchmark inference. All three stage models remain visible. Each cell below is correct / total (accuracy).

| Legacy cohort | Original model | Round 1 c3 | Round 2 seed 20261011 | Round 2 seed 20261012 | Round 2 seed 20261013 |
|---|---:|---:|---:|---:|---:|
| Original 100 | 100/100 (100%) | 100/100 (100%) | 100/100 (100%) | 100/100 (100%) | 100/100 (100%) |
| Same-case 1,058 | 1,057/1,058 (99.91%) | 1,057/1,058 (99.91%) | 1,058/1,058 (100%) | 1,057/1,058 (99.91%) | 1,058/1,058 (100%) |
| Conservative 756 subset | 755/756 (99.87%) | 755/756 (99.87%) | 756/756 (100%) | 755/756 (99.87%) | 756/756 (100%) |
| Full reused test 5,450 | 5,446/5,450 (99.93%) | 5,443/5,450 (99.87%) | 5,448/5,450 (99.96%) | 5,446/5,450 (99.93%) | 5,448/5,450 (99.96%) |

The three fixed readers had mean accuracy **39.67%**, with a descriptive range of **37–44%**, on the original 100 cases. [Exact case-aligned model/reader output](artifacts/stage/original100/original100_case_aligned_models_readers.csv) preserves every case. This is a historical label-comparison benchmark, not evidence that these models outperform neuroradiologists clinically. Original100 matches 66 old-training, nine old-validation and 25 old-test rows. The 1,058 comparison is exact-row held out, but 302 rows have a similarity-group mate in old training or validation. The 756 subset removes those observed mates; neither subset establishes patient independence. All four cohorts were already known during Round 2 development.

[Stage metrics and conditional paired uncertainty](artifacts/stage/report/analysis.json) retain all models. Ceiling or degenerate intervals do not overcome reuse or missing patient identity. In the 13 fixed pixel-stress conditions, seed 20261011's original100 minimum was 99/100 under blur, versus 100/100 for the original model throughout this grid; seeds 20261012 and 20261013 remained 100/100 throughout. On the 1,058 images, Round 2 minima were 1,057/1,058, 1,057/1,058 and 1,058/1,058, respectively. These bounded diagnostics do not establish robustness to real MRI acquisition or source shifts.

On the fixed 160-image validation control set, the representative retained 160/160 accuracy after the prescribed corner or outer-border erasures; corner-only inputs yielded 25% accuracy. This supports a limited conclusion about those specific input regions and controls, not absence of every shortcut. [All stress and masking values](artifacts/stage/report/figure_inputs.json) are retained.

## Training curves, attribution, and reproducibility

Stage seeds completed 21/33/35 epochs and selected epochs 6/18/20. Clinical seeds completed 80/53/78 epochs and selected 65/38/63. All stopped on validation plateau. Losses are unsmoothed, with augmented optimization, clean training, validation and stress criteria distinguished. Curves cannot substitute for the negative external result.

- Stage raw curves: [20261011](artifacts/stage/training/stage_only_seed20261011/raw_curves.pdf), [20261012](artifacts/stage/training/stage_only_seed20261012/raw_curves.pdf), [20261013](artifacts/stage/training/stage_only_seed20261013/raw_curves.pdf).
- Clinical raw curves: [20261021](artifacts/clinical/training/clinical_seed20261021/raw_patient_curves.pdf), [20261022](artifacts/clinical/training/clinical_seed20261022/raw_patient_curves.pdf), [20261023](artifacts/clinical/training/clinical_seed20261023/raw_patient_curves.pdf). Each curve's directory contains its unmodified CSV/JSON history.

All 115 new native Grad-CAM maps were valid and all 72 randomization records were retained. However, two original100 probability replays differed from the frozen evaluation by 0.0001177192 and 0.0003736615, exceeding the unchanged 0.0001 tolerance; predicted classes agreed. The [diagnostic gallery](artifacts/stage/attribution_diagnostics/gallery.html), [original blocked marker](artifacts/stage/attribution_diagnostics/source_attribution_complete.json) and [recovery receipt](artifacts/stage/attribution_diagnostics/recovery_receipt.json) preserve this failure. A postprocessing serialization recovery used saved outputs only. Visual review does not clear the failed numerical check or validate anatomical localization.

Training used RTX PRO 6000 Blackwell; the frozen clinical comparison used B300 Blackwell Ultra without retraining. All results and original training states were locally verified before the GPUs and storage were deleted. The [six research checkpoint manifest](../../../models/kevin_round2_20261011/CHECKPOINTS.json), [export fidelity receipt](research_weights_export_verification.json), and [artifact source/output checksums](artifacts/EXPORT_MANIFEST.json) support reproducibility. Synthetic checkpoint equality establishes export fidelity only. Production weights remain unchanged.

## What a further study needs

The verified fresh dataset was [Harvard Dataverse 29352 v2](https://doi.org/10.7910/DVN/29352), with 29 available T1 volumes and only 19 AD/CN participants; one originally reported AD T1 was unavailable. AlzaSet remains quarantined because patient mapping and inventory discrepancies are unresolved. Controlled-access candidates are not acquired cohorts. The [source and access account](DATA_AND_LIMITATIONS.md) documents these boundaries.

A stronger clinical study needs additional verified patient-labeled AD/CN training cohorts and a fresh untouched external cohort, with patient- and source-separated validation. Healthy-only collections or unmapped two-dimensional mirrors cannot fill that multisite AD/CN evidence gap. Harvard is now observed and must not be used to adjust parameters or thresholds while retaining a fresh-test claim. Additional epochs alone are not a solution. Any calibration, abstention, modality/quality rejection, radiologist adjudication and prospective silent evaluation must be developed and evaluated under a separate frozen study with suitable data. The present package supports transparent exploratory reporting, including a negative external result; clinical readiness remains unresolved.
