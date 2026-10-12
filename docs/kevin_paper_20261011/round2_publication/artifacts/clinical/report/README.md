# Separate clinical AD/control pilot

Fresh Harvard results have 19 patients (9 AD, 10 CN); the 10 Harvard and 5 OpenNeuro MCI participants receive predictions only. Besga 11 and OpenNeuro 15 remain separate reused benchmarks. No clinical-superiority, population-risk-calibration or neuroradiologist claim follows.

Accuracy, sensitivity and specificity below include exact 95% binomial intervals.

| Cohort | Model | Accuracy (95% CI) | AD sensitivity (95% CI) | CN specificity (95% CI) | Balanced accuracy |
|---|---|---:|---:|---:|---:|
| Fresh Harvard T1: 19 patients (9 AD, 10 CN) | Old clinical c2 | 68.4% (43.4, 87.4) | 100.0% (66.4, 100.0) | 40.0% (12.2, 73.8) | 70.0% |
| Fresh Harvard T1: 19 patients (9 AD, 10 CN) | Old clinical c4 | 47.4% (24.4, 71.1) | 100.0% (66.4, 100.0) | 0.0% (0.0, 30.8) | 50.0% |
| Fresh Harvard T1: 19 patients (9 AD, 10 CN) | Old clinical c6 | 68.4% (43.4, 87.4) | 100.0% (66.4, 100.0) | 40.0% (12.2, 73.8) | 70.0% |
| Fresh Harvard T1: 19 patients (9 AD, 10 CN) | New seed 20261021 | 47.4% (24.4, 71.1) | 100.0% (66.4, 100.0) | 0.0% (0.0, 30.8) | 50.0% |
| Fresh Harvard T1: 19 patients (9 AD, 10 CN) | New seed 20261022 | 47.4% (24.4, 71.1) | 100.0% (66.4, 100.0) | 0.0% (0.0, 30.8) | 50.0% |
| Fresh Harvard T1: 19 patients (9 AD, 10 CN) | New seed 20261023 | 47.4% (24.4, 71.1) | 100.0% (66.4, 100.0) | 0.0% (0.0, 30.8) | 50.0% |
| Fresh Harvard T1: 19 patients (9 AD, 10 CN) | Old fixed ensemble | 52.6% (28.9, 75.6) | 100.0% (66.4, 100.0) | 10.0% (0.3, 44.5) | 55.0% |
| Fresh Harvard T1: 19 patients (9 AD, 10 CN) | New fixed ensemble | 47.4% (24.4, 71.1) | 100.0% (66.4, 100.0) | 0.0% (0.0, 30.8) | 50.0% |
| Reused Besga test: 11 patients (7 AD, 4 CN) | Old clinical c2 | 45.5% (16.7, 76.6) | 14.3% (0.4, 57.9) | 100.0% (39.8, 100.0) | 57.1% |
| Reused Besga test: 11 patients (7 AD, 4 CN) | Old clinical c4 | 63.6% (30.8, 89.1) | 57.1% (18.4, 90.1) | 75.0% (19.4, 99.4) | 66.1% |
| Reused Besga test: 11 patients (7 AD, 4 CN) | Old clinical c6 | 36.4% (10.9, 69.2) | 14.3% (0.4, 57.9) | 75.0% (19.4, 99.4) | 44.6% |
| Reused Besga test: 11 patients (7 AD, 4 CN) | New seed 20261021 | 90.9% (58.7, 99.8) | 100.0% (59.0, 100.0) | 75.0% (19.4, 99.4) | 87.5% |
| Reused Besga test: 11 patients (7 AD, 4 CN) | New seed 20261022 | 90.9% (58.7, 99.8) | 100.0% (59.0, 100.0) | 75.0% (19.4, 99.4) | 87.5% |
| Reused Besga test: 11 patients (7 AD, 4 CN) | New seed 20261023 | 81.8% (48.2, 97.7) | 85.7% (42.1, 99.6) | 75.0% (19.4, 99.4) | 80.4% |
| Reused Besga test: 11 patients (7 AD, 4 CN) | Old fixed ensemble | 45.5% (16.7, 76.6) | 14.3% (0.4, 57.9) | 100.0% (39.8, 100.0) | 57.1% |
| Reused Besga test: 11 patients (7 AD, 4 CN) | New fixed ensemble | 90.9% (58.7, 99.8) | 100.0% (59.0, 100.0) | 75.0% (19.4, 99.4) | 87.5% |
| Reused OpenNeuro: 15 patients (1 AD, 14 CN) | Old clinical c2 | 93.3% (68.1, 99.8) | 0.0% (0.0, 97.5) | 100.0% (76.8, 100.0) | 50.0% |
| Reused OpenNeuro: 15 patients (1 AD, 14 CN) | Old clinical c4 | 46.7% (21.3, 73.4) | 100.0% (2.5, 100.0) | 42.9% (17.7, 71.1) | 71.4% |
| Reused OpenNeuro: 15 patients (1 AD, 14 CN) | Old clinical c6 | 86.7% (59.5, 98.3) | 0.0% (0.0, 97.5) | 92.9% (66.1, 99.8) | 46.4% |
| Reused OpenNeuro: 15 patients (1 AD, 14 CN) | New seed 20261021 | 93.3% (68.1, 99.8) | 0.0% (0.0, 97.5) | 100.0% (76.8, 100.0) | 50.0% |
| Reused OpenNeuro: 15 patients (1 AD, 14 CN) | New seed 20261022 | 66.7% (38.4, 88.2) | 0.0% (0.0, 97.5) | 71.4% (41.9, 91.6) | 35.7% |
| Reused OpenNeuro: 15 patients (1 AD, 14 CN) | New seed 20261023 | 100.0% (78.2, 100.0) | 100.0% (2.5, 100.0) | 100.0% (76.8, 100.0) | 100.0% |
| Reused OpenNeuro: 15 patients (1 AD, 14 CN) | Old fixed ensemble | 80.0% (51.9, 95.7) | 0.0% (0.0, 97.5) | 85.7% (57.2, 98.2) | 42.9% |
| Reused OpenNeuro: 15 patients (1 AD, 14 CN) | New fixed ensemble | 93.3% (68.1, 99.8) | 0.0% (0.0, 97.5) | 100.0% (76.8, 100.0) | 50.0% |

Fixed new ensemble minus fixed old ensemble: paired differences in percentage points and conditional 95% intervals from 10,000 diagnosis-stratified patient bootstrap draws. These condition on the observed cohort and diagnosis counts.

| Cohort | Accuracy difference | Balanced-accuracy difference | Sensitivity difference | Specificity difference |
|---|---:|---:|---:|---:|
| Fresh Harvard T1: 19 patients (9 AD, 10 CN) | -5.3 (-15.8, +0.0) | -5.0 (-15.0, +0.0) | +0.0 (+0.0, +0.0) | -10.0 (-30.0, +0.0) |
| Reused Besga test: 11 patients (7 AD, 4 CN) | +45.5 (+18.2, +63.6) | +30.4 (+3.6, +50.0) | +85.7 (+57.1, +100.0) | -25.0 (-75.0, +0.0) |
| Reused OpenNeuro: 15 patients (1 AD, 14 CN) | +13.3 (+0.0, +33.3) | +7.1 (+0.0, +17.9) | +0.0 (+0.0, +0.0) | +14.3 (+0.0, +35.7) |

Exact intervals, all patient-paired intervals, discordance counts and seed spread are in analysis.json. McNemar p-values are unadjusted exploratory diagnostics, not clinical-superiority evidence. All 168 scenario metrics are independently recomputed from the complete 3,360-record patient probability grid.

OpenNeuro contains only one AD patient. Diagnosis-stratified bootstrap repeatedly resamples that same person; its conditional sensitivity interval is not population sensitivity precision. Use the exact binomial sensitivity interval and retain this limitation.

Old models retain mean slice probabilities; new models use softmax of mean slice logits. Both ensembles use uniform patient probability means. This compares complete pipelines.

All 19 graph PDFs require visual review before publication. MCI scores are not diagnostic validation or progression predictions.
