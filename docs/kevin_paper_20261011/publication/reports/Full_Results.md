> Portable public derivative of the audited local source. Clinical subject-level probability rows and operational identifiers are omitted. Scientific aggregate results are unchanged. SHA256 values quoted from the source refer to original frozen files; export hashes are in EXPORT_MANIFEST.json.

# Kevin dementia experiment results

This report renders the frozen six-run experiment and its recorded uncertainty analysis. It performs no inference or model selection. All conditions, seeds and benchmark results are retained; numerical differences alone do not establish clinical superiority.

Manuscript Methods draft: [methods_protocol_snapshot.md](Prespecified_Methods.md); SHA256 `b43207ee9a77d3b91136c691f5c5c8546fe540d103e620b0ae8d587ed45bd5b9`.

## Evidence and validation-only preference

Suite status: **training_complete_awaiting_frozen_evaluation**. Six completed runs are recorded. The condition preferred before held-out evaluation was **stage_only**; its exploratory representative was **candidate_3**. This preference used mean stage-validation NLL and then the representative seed's validation NLL, never test outcomes.

| Evidence | SHA256 |
| --- | --- |
| [suite_status.json](../evidence/completion_summary.json) | 40400a103ddce71e8b3322e1313723caf4f5838d43debc2acacf8d035c7519e3 |
| [validation_selection_before_evaluation.json](../results/validation_selection.json) | 5ba9de51fe339ab6e470de496571430ab6f463a814efc858c228db954241c0c7 |
| [evaluation.json](../results/evaluation.json) | 2ad94f62c20b3505f3585abb6165e75aa4a532e916f41d482f7f7f288f2a56d0 |
| [uncertainty_analysis.json](../results/uncertainty_analysis.json) | 56881a4143143dcf4a8bc285ed03ec5a61ea2dcf26882a7eff220d16226189e7 |

Checkpoint selection required an NLL improvement greater than the configured minimum delta. It need not equal the mathematical minimum observed NLL when later improvements were smaller.

## All six training runs and raw curves

| Candidate | Condition | Seed | Epochs completed | Selected epoch | Selected validation NLL | Maximum epochs | Minimum epochs | Patience | Minimum delta | Raw loss curves | Raw accuracy curves | History |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| candidate_1 | stage_only | 20261011 | 38 | 23 | 0.037472 | 100 | 20 | 15 | 0.000100 | [loss](../training/stage_only_seed20261011/loss_curves.png) | [accuracy](../training/stage_only_seed20261011/accuracy_curves.png) | [raw CSV](../training/stage_only_seed20261011/history.csv) |
| candidate_2 | multitask | 20261011 | 43 | 28 | 0.051601 | 100 | 20 | 15 | 0.000100 | [loss](../training/multitask_seed20261011/loss_curves.png) | [accuracy](../training/multitask_seed20261011/accuracy_curves.png) | [raw CSV](../training/multitask_seed20261011/history.csv) |
| candidate_3 | stage_only | 20261012 | 52 | 37 | 0.034771 | 100 | 20 | 15 | 0.000100 | [loss](../training/stage_only_seed20261012/loss_curves.png) | [accuracy](../training/stage_only_seed20261012/accuracy_curves.png) | [raw CSV](../training/stage_only_seed20261012/history.csv) |
| candidate_4 | multitask | 20261012 | 74 | 59 | 0.045311 | 100 | 20 | 15 | 0.000100 | [loss](../training/multitask_seed20261012/loss_curves.png) | [accuracy](../training/multitask_seed20261012/accuracy_curves.png) | [raw CSV](../training/multitask_seed20261012/history.csv) |
| candidate_5 | stage_only | 20261013 | 54 | 39 | 0.036942 | 100 | 20 | 15 | 0.000100 | [loss](../training/stage_only_seed20261013/loss_curves.png) | [accuracy](../training/stage_only_seed20261013/accuracy_curves.png) | [raw CSV](../training/stage_only_seed20261013/history.csv) |
| candidate_6 | multitask | 20261013 | 44 | 29 | 0.054033 | 100 | 20 | 15 | 0.000100 | [loss](../training/multitask_seed20261013/loss_curves.png) | [accuracy](../training/multitask_seed20261013/accuracy_curves.png) | [raw CSV](../training/multitask_seed20261013/history.csv) |

Stage train loss logs unweighted, unsmoothed cross-entropy; optimization uses inverse-group/class weights and label smoothing. Augmentation and dropout are active during training. Clinical train loss uses balanced sampled slices, while validation averages nine slice probabilities per source subject before scoring. The plotted training/validation losses therefore have different sampling and aggregation conditions. Raw curves are linked without cosmetic smoothing.

## Primary exact-row-held-out legacy comparison

**1058 images; 966 resampling units.** Full primary exact-row-held-out comparison; prepared hash groups include old development exposure. Paired hash-group intervals are conditional and do not establish patient independence or equal absence of old image-family exposure.

Observed old-development similarity overlap: 291 primary images have an old training group mate; 302 have an old training or validation group mate. These are coarse image-similarity groups, not patient IDs.

| Model | Correct / n | Accuracy; 95% interval | Balanced accuracy; 95% interval | Macro F1; 95% interval |
| --- | --- | --- | --- | --- |
| Old baseline | 1057/1058 | 99.91%; [99.71%, 100.00%] | 99.89%; [99.66%, 100.00%] | 99.90%; [99.68%, 100.00%] |
| candidate_1: stage_only, seed 20261011 | 1055/1058 | 99.72%; [99.34%, 100.00%] | 99.68%; [99.28%, 100.00%] | 99.69%; [99.28%, 100.00%] |
| candidate_2: multitask, seed 20261011 | 1054/1058 | 99.62%; [99.16%, 99.91%] | 99.60%; [99.14%, 99.91%] | 99.61%; [99.15%, 99.91%] |
| candidate_3: stage_only, seed 20261012 | 1057/1058 | 99.91%; [99.70%, 100.00%] | 99.91%; [99.69%, 100.00%] | 99.91%; [99.69%, 100.00%] |
| candidate_4: multitask, seed 20261012 | 1056/1058 | 99.81%; [99.53%, 100.00%] | 99.80%; [99.49%, 100.00%] | 99.81%; [99.52%, 100.00%] |
| candidate_5: stage_only, seed 20261013 | 1057/1058 | 99.91%; [99.71%, 100.00%] | 99.91%; [99.71%, 100.00%] | 99.90%; [99.69%, 100.00%] |
| candidate_6: multitask, seed 20261013 | 1055/1058 | 99.72%; [99.42%, 100.00%] | 99.68%; [99.31%, 100.00%] | 99.70%; [99.35%, 100.00%] |

Per-class recalls expose class-specific regressions; each cell shows correct class predictions / true class count.

| Model | MildDemented | ModerateDemented | NonDemented | VeryMildDemented |
| --- | --- | --- | --- | --- |
| Old baseline | 277/277 (100.00%) | 299/299 (100.00%) | 229/230 (99.57%) | 252/252 (100.00%) |
| candidate_1: stage_only, seed 20261011 | 277/277 (100.00%) | 299/299 (100.00%) | 228/230 (99.13%) | 251/252 (99.60%) |
| candidate_2: multitask, seed 20261011 | 277/277 (100.00%) | 299/299 (100.00%) | 230/230 (100.00%) | 248/252 (98.41%) |
| candidate_3: stage_only, seed 20261012 | 276/277 (99.64%) | 299/299 (100.00%) | 230/230 (100.00%) | 252/252 (100.00%) |
| candidate_4: multitask, seed 20261012 | 277/277 (100.00%) | 299/299 (100.00%) | 230/230 (100.00%) | 250/252 (99.21%) |
| candidate_5: stage_only, seed 20261013 | 276/277 (99.64%) | 299/299 (100.00%) | 230/230 (100.00%) | 252/252 (100.00%) |
| candidate_6: multitask, seed 20261013 | 277/277 (100.00%) | 299/299 (100.00%) | 228/230 (99.13%) | 251/252 (99.60%) |

Paired changes are candidate minus reference, in percentage points. Every candidate-versus-old and matched-seed multitask-versus-stage-only contrast is retained.

| Candidate | Reference | Paired n | Accuracy change; 95% interval | Balanced accuracy change; 95% interval | Macro F1 change; 95% interval | Candidate only correct | Reference only correct |
| --- | --- | --- | --- | --- | --- | --- | --- |
| candidate_1: stage_only, seed 20261011 | Old baseline | 1058 | -0.19 pp; [-0.57, +0.18] pp | -0.21 pp; [-0.64, +0.16] pp | -0.21 pp; [-0.63, +0.19] pp | 1 | 3 |
| candidate_2: multitask, seed 20261011 | Old baseline | 1058 | -0.28 pp; [-0.75, +0.09] pp | -0.29 pp; [-0.79, +0.11] pp | -0.28 pp; [-0.78, +0.11] pp | 1 | 4 |
| candidate_3: stage_only, seed 20261012 | Old baseline | 1058 | +0.00 pp; [-0.29, +0.28] pp | +0.02 pp; [-0.27, +0.31] pp | +0.01 pp; [-0.29, +0.31] pp | 1 | 1 |
| candidate_4: multitask, seed 20261012 | Old baseline | 1058 | -0.09 pp; [-0.39, +0.19] pp | -0.09 pp; [-0.43, +0.23] pp | -0.09 pp; [-0.40, +0.22] pp | 1 | 2 |
| candidate_5: stage_only, seed 20261013 | Old baseline | 1058 | +0.00 pp; [-0.28, +0.28] pp | +0.02 pp; [-0.26, +0.31] pp | +0.00 pp; [-0.29, +0.30] pp | 1 | 1 |
| candidate_6: multitask, seed 20261013 | Old baseline | 1058 | -0.19 pp; [-0.48, +0.00] pp | -0.21 pp; [-0.53, +0.00] pp | -0.20 pp; [-0.51, +0.00] pp | 0 | 2 |
| candidate_2: multitask, seed 20261011 | candidate_1: stage_only, seed 20261011 | 1058 | -0.09 pp; [-0.65, +0.38] pp | -0.08 pp; [-0.65, +0.44] pp | -0.08 pp; [-0.65, +0.43] pp | 3 | 4 |
| candidate_4: multitask, seed 20261012 | candidate_3: stage_only, seed 20261012 | 1058 | -0.09 pp; [-0.39, +0.19] pp | -0.11 pp; [-0.43, +0.19] pp | -0.09 pp; [-0.40, +0.20] pp | 1 | 2 |
| candidate_6: multitask, seed 20261013 | candidate_5: stage_only, seed 20261013 | 1058 | -0.19 pp; [-0.57, +0.19] pp | -0.23 pp; [-0.63, +0.16] pp | -0.20 pp; [-0.60, +0.19] pp | 1 | 3 |

Condition summaries show the three prespecified seeds. Their standard deviation measures training-run spread and is not a confidence interval.

| Condition | Seeds | Metric | Mean | Sample SD | Minimum | Maximum |
| --- | --- | --- | --- | --- | --- | --- |
| stage_only | 3 | accuracy | 99.84% | 0.11% | 99.72% | 99.91% |
| stage_only | 3 | balanced_accuracy | 99.83% | 0.13% | 99.68% | 99.91% |
| stage_only | 3 | macro_f1 | 99.83% | 0.12% | 99.69% | 99.91% |
| multitask | 3 | accuracy | 99.72% | 0.09% | 99.62% | 99.81% |
| multitask | 3 | balanced_accuracy | 99.70% | 0.10% | 99.60% | 99.80% |
| multitask | 3 | macro_f1 | 99.71% | 0.10% | 99.61% | 99.81% |

## Secondary old-hash-group-unexposed sensitivity analysis

**756 images; 733 resampling units.** Conservative image-similarity sensitivity subset with no observed old train/val mate in the retained prepared hash group; hash groups are not patients, omitted source relationships remain unresolved, and subset selection changes class composition/difficulty.

The secondary subset excludes observed old training/validation group mates and reuses existing primary predictions. The complete primary remains above. Conservative dHash collisions can remove unrelated people; this subset does not establish patient independence.

| Model | Correct / n | Accuracy; 95% interval | Balanced accuracy; 95% interval | Macro F1; 95% interval |
| --- | --- | --- | --- | --- |
| Old baseline | 755/756 | 99.87%; [99.60%, 100.00%] | 99.86%; [99.55%, 100.00%] | 99.86%; [99.57%, 100.00%] |
| candidate_1: stage_only, seed 20261011 | 753/756 | 99.60%; [99.08%, 100.00%] | 99.59%; [99.09%, 100.00%] | 99.59%; [99.11%, 100.00%] |
| candidate_2: multitask, seed 20261011 | 752/756 | 99.47%; [98.93%, 99.87%] | 99.47%; [98.89%, 99.88%] | 99.47%; [98.91%, 99.88%] |
| candidate_3: stage_only, seed 20261012 | 755/756 | 99.87%; [99.60%, 100.00%] | 99.87%; [99.58%, 100.00%] | 99.87%; [99.59%, 100.00%] |
| candidate_4: multitask, seed 20261012 | 754/756 | 99.74%; [99.34%, 100.00%] | 99.73%; [99.31%, 100.00%] | 99.74%; [99.34%, 100.00%] |
| candidate_5: stage_only, seed 20261013 | 755/756 | 99.87%; [99.60%, 100.00%] | 99.87%; [99.58%, 100.00%] | 99.87%; [99.57%, 100.00%] |
| candidate_6: multitask, seed 20261013 | 754/756 | 99.74%; [99.34%, 100.00%] | 99.73%; [99.31%, 100.00%] | 99.73%; [99.33%, 100.00%] |

Per-class recalls expose class-specific regressions; each cell shows correct class predictions / true class count.

| Model | MildDemented | ModerateDemented | NonDemented | VeryMildDemented |
| --- | --- | --- | --- | --- |
| Old baseline | 199/199 (100.00%) | 189/189 (100.00%) | 179/180 (99.44%) | 188/188 (100.00%) |
| candidate_1: stage_only, seed 20261011 | 199/199 (100.00%) | 189/189 (100.00%) | 178/180 (98.89%) | 187/188 (99.47%) |
| candidate_2: multitask, seed 20261011 | 199/199 (100.00%) | 189/189 (100.00%) | 180/180 (100.00%) | 184/188 (97.87%) |
| candidate_3: stage_only, seed 20261012 | 198/199 (99.50%) | 189/189 (100.00%) | 180/180 (100.00%) | 188/188 (100.00%) |
| candidate_4: multitask, seed 20261012 | 199/199 (100.00%) | 189/189 (100.00%) | 180/180 (100.00%) | 186/188 (98.94%) |
| candidate_5: stage_only, seed 20261013 | 198/199 (99.50%) | 189/189 (100.00%) | 180/180 (100.00%) | 188/188 (100.00%) |
| candidate_6: multitask, seed 20261013 | 199/199 (100.00%) | 189/189 (100.00%) | 179/180 (99.44%) | 187/188 (99.47%) |

Paired changes are candidate minus reference, in percentage points. Every candidate-versus-old and matched-seed multitask-versus-stage-only contrast is retained.

| Candidate | Reference | Paired n | Accuracy change; 95% interval | Balanced accuracy change; 95% interval | Macro F1 change; 95% interval | Candidate only correct | Reference only correct |
| --- | --- | --- | --- | --- | --- | --- | --- |
| candidate_1: stage_only, seed 20261011 | Old baseline | 756 | -0.26 pp; [-0.80, +0.26] pp | -0.27 pp; [-0.84, +0.27] pp | -0.27 pp; [-0.84, +0.27] pp | 1 | 3 |
| candidate_2: multitask, seed 20261011 | Old baseline | 756 | -0.40 pp; [-1.05, +0.13] pp | -0.39 pp; [-1.00, +0.14] pp | -0.39 pp; [-1.02, +0.15] pp | 1 | 4 |
| candidate_3: stage_only, seed 20261012 | Old baseline | 756 | +0.00 pp; [-0.40, +0.40] pp | +0.01 pp; [-0.38, +0.40] pp | +0.01 pp; [-0.39, +0.40] pp | 1 | 1 |
| candidate_4: multitask, seed 20261012 | Old baseline | 756 | -0.13 pp; [-0.65, +0.27] pp | -0.13 pp; [-0.61, +0.30] pp | -0.12 pp; [-0.61, +0.29] pp | 1 | 2 |
| candidate_5: stage_only, seed 20261013 | Old baseline | 756 | +0.00 pp; [-0.40, +0.40] pp | +0.01 pp; [-0.37, +0.40] pp | +0.00 pp; [-0.40, +0.40] pp | 1 | 1 |
| candidate_6: multitask, seed 20261013 | Old baseline | 756 | -0.13 pp; [-0.40, +0.00] pp | -0.13 pp; [-0.43, +0.00] pp | -0.13 pp; [-0.40, +0.00] pp | 0 | 1 |
| candidate_2: multitask, seed 20261011 | candidate_1: stage_only, seed 20261011 | 756 | -0.13 pp; [-0.80, +0.53] pp | -0.12 pp; [-0.81, +0.57] pp | -0.12 pp; [-0.79, +0.57] pp | 3 | 4 |
| candidate_4: multitask, seed 20261012 | candidate_3: stage_only, seed 20261012 | 756 | -0.13 pp; [-0.53, +0.27] pp | -0.14 pp; [-0.61, +0.26] pp | -0.13 pp; [-0.57, +0.27] pp | 1 | 2 |
| candidate_6: multitask, seed 20261013 | candidate_5: stage_only, seed 20261013 | 756 | -0.13 pp; [-0.53, +0.27] pp | -0.15 pp; [-0.58, +0.26] pp | -0.13 pp; [-0.55, +0.28] pp | 1 | 2 |

Condition summaries show the three prespecified seeds. Their standard deviation measures training-run spread and is not a confidence interval.

| Condition | Seeds | Metric | Mean | Sample SD | Minimum | Maximum |
| --- | --- | --- | --- | --- | --- | --- |
| stage_only | 3 | accuracy | 99.78% | 0.15% | 99.60% | 99.87% |
| stage_only | 3 | balanced_accuracy | 99.78% | 0.16% | 99.59% | 99.87% |
| stage_only | 3 | macro_f1 | 99.78% | 0.16% | 99.59% | 99.87% |
| multitask | 3 | accuracy | 99.65% | 0.15% | 99.47% | 99.74% |
| multitask | 3 | balanced_accuracy | 99.64% | 0.15% | 99.47% | 99.73% |
| multitask | 3 | macro_f1 | 99.65% | 0.15% | 99.47% | 99.74% |

## Historical original100 comparison

**100 images; 100 resampling units.** Historically reused image-level descriptive benchmark; no independent clinical claim

These historical cases match 66 training, 9 validation and 25 test rows in the old strict manifest. This descriptive benchmark cannot establish independent clinical superiority over readers.

| Model | Correct / n | Accuracy; 95% interval | Balanced accuracy; 95% interval | Macro F1; 95% interval |
| --- | --- | --- | --- | --- |
| Old baseline | 100/100 | 100.00%; [100.00%, 100.00%] (degenerate) | 100.00%; [100.00%, 100.00%] (degenerate) | 100.00%; [100.00%, 100.00%] (degenerate) |
| candidate_1: stage_only, seed 20261011 | 99/100 | 99.00%; [97.00%, 100.00%] | 99.07%; [96.88%, 100.00%] | 99.07%; [96.90%, 100.00%] |
| candidate_2: multitask, seed 20261011 | 100/100 | 100.00%; [100.00%, 100.00%] (degenerate) | 100.00%; [100.00%, 100.00%] (degenerate) | 100.00%; [100.00%, 100.00%] (degenerate) |
| candidate_3: stage_only, seed 20261012 | 100/100 | 100.00%; [100.00%, 100.00%] (degenerate) | 100.00%; [100.00%, 100.00%] (degenerate) | 100.00%; [100.00%, 100.00%] (degenerate) |
| candidate_4: multitask, seed 20261012 | 99/100 | 99.00%; [97.00%, 100.00%] | 99.07%; [96.88%, 100.00%] | 99.07%; [96.90%, 100.00%] |
| candidate_5: stage_only, seed 20261013 | 100/100 | 100.00%; [100.00%, 100.00%] (degenerate) | 100.00%; [100.00%, 100.00%] (degenerate) | 100.00%; [100.00%, 100.00%] (degenerate) |
| candidate_6: multitask, seed 20261013 | 99/100 | 99.00%; [97.00%, 100.00%] | 99.07%; [96.97%, 100.00%] | 99.07%; [96.92%, 100.00%] |

Per-class recalls expose class-specific regressions; each cell shows correct class predictions / true class count.

| Model | MildDemented | ModerateDemented | NonDemented | VeryMildDemented |
| --- | --- | --- | --- | --- |
| Old baseline | 23/23 (100.00%) | 23/23 (100.00%) | 27/27 (100.00%) | 27/27 (100.00%) |
| candidate_1: stage_only, seed 20261011 | 23/23 (100.00%) | 23/23 (100.00%) | 26/27 (96.30%) | 27/27 (100.00%) |
| candidate_2: multitask, seed 20261011 | 23/23 (100.00%) | 23/23 (100.00%) | 27/27 (100.00%) | 27/27 (100.00%) |
| candidate_3: stage_only, seed 20261012 | 23/23 (100.00%) | 23/23 (100.00%) | 27/27 (100.00%) | 27/27 (100.00%) |
| candidate_4: multitask, seed 20261012 | 23/23 (100.00%) | 23/23 (100.00%) | 26/27 (96.30%) | 27/27 (100.00%) |
| candidate_5: stage_only, seed 20261013 | 23/23 (100.00%) | 23/23 (100.00%) | 27/27 (100.00%) | 27/27 (100.00%) |
| candidate_6: multitask, seed 20261013 | 23/23 (100.00%) | 23/23 (100.00%) | 27/27 (100.00%) | 26/27 (96.30%) |

Paired changes are candidate minus reference, in percentage points. Every candidate-versus-old and matched-seed multitask-versus-stage-only contrast is retained.

| Candidate | Reference | Paired n | Accuracy change; 95% interval | Balanced accuracy change; 95% interval | Macro F1 change; 95% interval | Candidate only correct | Reference only correct |
| --- | --- | --- | --- | --- | --- | --- | --- |
| candidate_1: stage_only, seed 20261011 | Old baseline | 100 | -1.00 pp; [-3.00, +0.00] pp | -0.93 pp; [-3.12, +0.00] pp | -0.93 pp; [-3.10, +0.00] pp | 0 | 1 |
| candidate_2: multitask, seed 20261011 | Old baseline | 100 | +0.00 pp; [+0.00, +0.00] pp (degenerate) | +0.00 pp; [+0.00, +0.00] pp (degenerate) | +0.00 pp; [+0.00, +0.00] pp (degenerate) | 0 | 0 |
| candidate_3: stage_only, seed 20261012 | Old baseline | 100 | +0.00 pp; [+0.00, +0.00] pp (degenerate) | +0.00 pp; [+0.00, +0.00] pp (degenerate) | +0.00 pp; [+0.00, +0.00] pp (degenerate) | 0 | 0 |
| candidate_4: multitask, seed 20261012 | Old baseline | 100 | -1.00 pp; [-3.00, +0.00] pp | -0.93 pp; [-3.12, +0.00] pp | -0.93 pp; [-3.10, +0.00] pp | 0 | 1 |
| candidate_5: stage_only, seed 20261013 | Old baseline | 100 | +0.00 pp; [+0.00, +0.00] pp (degenerate) | +0.00 pp; [+0.00, +0.00] pp (degenerate) | +0.00 pp; [+0.00, +0.00] pp (degenerate) | 0 | 0 |
| candidate_6: multitask, seed 20261013 | Old baseline | 100 | -1.00 pp; [-3.00, +0.00] pp | -0.93 pp; [-3.03, +0.00] pp | -0.93 pp; [-3.08, +0.00] pp | 0 | 1 |
| candidate_2: multitask, seed 20261011 | candidate_1: stage_only, seed 20261011 | 100 | +1.00 pp; [+0.00, +3.00] pp | +0.93 pp; [+0.00, +3.12] pp | +0.93 pp; [+0.00, +3.10] pp | 1 | 0 |
| candidate_4: multitask, seed 20261012 | candidate_3: stage_only, seed 20261012 | 100 | -1.00 pp; [-3.00, +0.00] pp | -0.93 pp; [-3.12, +0.00] pp | -0.93 pp; [-3.10, +0.00] pp | 0 | 1 |
| candidate_6: multitask, seed 20261013 | candidate_5: stage_only, seed 20261013 | 100 | -1.00 pp; [-3.00, +0.00] pp | -0.93 pp; [-3.03, +0.00] pp | -0.93 pp; [-3.08, +0.00] pp | 0 | 1 |

Condition summaries show the three prespecified seeds. Their standard deviation measures training-run spread and is not a confidence interval.

| Condition | Seeds | Metric | Mean | Sample SD | Minimum | Maximum |
| --- | --- | --- | --- | --- | --- | --- |
| stage_only | 3 | accuracy | 99.67% | 0.58% | 99.00% | 100.00% |
| stage_only | 3 | balanced_accuracy | 99.69% | 0.53% | 99.07% | 100.00% |
| stage_only | 3 | macro_f1 | 99.69% | 0.53% | 99.07% | 100.00% |
| multitask | 3 | accuracy | 99.33% | 0.58% | 99.00% | 100.00% |
| multitask | 3 | balanced_accuracy | 99.38% | 0.53% | 99.07% | 100.00% |
| multitask | 3 | macro_f1 | 99.38% | 0.53% | 99.07% | 100.00% |

Boundary-sensitive accuracy intervals below additionally assume independent images, which is unproven. Degenerate bootstrap intervals above do not prove certainty or equivalence.

| Model | Correct / n; exact-binomial sensitivity interval |
| --- | --- |
| Old baseline | 100/100 (100.00%); [96.38%, 100.00%] |
| candidate_1: stage_only, seed 20261011 | 99/100 (99.00%); [94.55%, 99.97%] |
| candidate_2: multitask, seed 20261011 | 100/100 (100.00%); [96.38%, 100.00%] |
| candidate_3: stage_only, seed 20261012 | 100/100 (100.00%); [96.38%, 100.00%] |
| candidate_4: multitask, seed 20261012 | 99/100 (99.00%); [94.55%, 99.97%] |
| candidate_5: stage_only, seed 20261013 | 100/100 (100.00%); [96.38%, 100.00%] |
| candidate_6: multitask, seed 20261013 | 99/100 (99.00%); [94.55%, 99.97%] |

## Original100 reader comparisons

Readers form a fixed panel. Comparisons use the same complete cases for each model/reader pair; the panel is not a random sample of neuroradiologists. Missingness and denominators are explicit.

| Reader | Correct / complete n | Excluded cases | Accuracy; 95% interval | Balanced accuracy; 95% interval | Macro F1; 95% interval |
| --- | --- | --- | --- | --- | --- |
| Reader_Kalapos | 37/100 | 0 | 37.00%; [28.00%, 47.00%] | 36.35%; [27.61%, 46.03%] | 36.38%; [27.24%, 45.47%] |
| Reader_Kanekar | 44/100 | 0 | 44.00%; [35.00%, 53.02%] | 43.96%; [34.84%, 52.85%] | 43.13%; [33.75%, 51.80%] |
| Reader_Thamburaj | 38/100 | 0 | 38.00%; [29.00%, 48.00%] | 39.05%; [30.63%, 48.13%] | 38.26%; [29.44%, 46.95%] |

| Model | Reader | Paired n | Accuracy change; 95% interval | Balanced accuracy change; 95% interval | Macro F1 change; 95% interval | Model only correct | Reader only correct |
| --- | --- | --- | --- | --- | --- | --- | --- |
| candidate_1: stage_only, seed 20261011 | Reader_Kalapos | 100 | +62.00 pp; [+51.00, +71.00] pp | +62.72 pp; [+52.61, +71.77] pp | +62.69 pp; [+53.08, +71.92] pp | 63 | 1 |
| candidate_2: multitask, seed 20261011 | Reader_Kalapos | 100 | +63.00 pp; [+53.00, +72.00] pp | +63.65 pp; [+53.97, +72.39] pp | +63.62 pp; [+54.53, +72.76] pp | 63 | 0 |
| candidate_3: stage_only, seed 20261012 | Reader_Kalapos | 100 | +63.00 pp; [+53.00, +72.00] pp | +63.65 pp; [+53.97, +72.39] pp | +63.62 pp; [+54.53, +72.76] pp | 63 | 0 |
| candidate_4: multitask, seed 20261012 | Reader_Kalapos | 100 | +62.00 pp; [+51.00, +71.00] pp | +62.72 pp; [+52.61, +71.77] pp | +62.69 pp; [+53.08, +71.92] pp | 63 | 1 |
| candidate_5: stage_only, seed 20261013 | Reader_Kalapos | 100 | +63.00 pp; [+53.00, +72.00] pp | +63.65 pp; [+53.97, +72.39] pp | +63.62 pp; [+54.53, +72.76] pp | 63 | 0 |
| candidate_6: multitask, seed 20261013 | Reader_Kalapos | 100 | +62.00 pp; [+51.00, +71.00] pp | +62.72 pp; [+52.72, +71.83] pp | +62.69 pp; [+53.33, +72.06] pp | 63 | 1 |
| Old baseline | Reader_Kalapos | 100 | +63.00 pp; [+53.00, +72.00] pp | +63.65 pp; [+53.97, +72.39] pp | +63.62 pp; [+54.53, +72.76] pp | 63 | 0 |
| candidate_1: stage_only, seed 20261011 | Reader_Kanekar | 100 | +55.00 pp; [+46.00, +64.02] pp | +55.11 pp; [+46.11, +64.34] pp | +55.94 pp; [+47.16, +65.44] pp | 55 | 0 |
| candidate_2: multitask, seed 20261011 | Reader_Kanekar | 100 | +56.00 pp; [+46.98, +65.00] pp | +56.04 pp; [+47.15, +65.16] pp | +56.87 pp; [+48.20, +66.25] pp | 56 | 0 |
| candidate_3: stage_only, seed 20261012 | Reader_Kanekar | 100 | +56.00 pp; [+46.98, +65.00] pp | +56.04 pp; [+47.15, +65.16] pp | +56.87 pp; [+48.20, +66.25] pp | 56 | 0 |
| candidate_4: multitask, seed 20261012 | Reader_Kanekar | 100 | +55.00 pp; [+46.00, +64.02] pp | +55.11 pp; [+46.11, +64.34] pp | +55.94 pp; [+47.16, +65.44] pp | 55 | 0 |
| candidate_5: stage_only, seed 20261013 | Reader_Kanekar | 100 | +56.00 pp; [+46.98, +65.00] pp | +56.04 pp; [+47.15, +65.16] pp | +56.87 pp; [+48.20, +66.25] pp | 56 | 0 |
| candidate_6: multitask, seed 20261013 | Reader_Kanekar | 100 | +55.00 pp; [+45.98, +64.00] pp | +55.11 pp; [+46.11, +64.02] pp | +55.94 pp; [+47.24, +65.40] pp | 55 | 0 |
| Old baseline | Reader_Kanekar | 100 | +56.00 pp; [+46.98, +65.00] pp | +56.04 pp; [+47.15, +65.16] pp | +56.87 pp; [+48.20, +66.25] pp | 56 | 0 |
| candidate_1: stage_only, seed 20261011 | Reader_Thamburaj | 100 | +61.00 pp; [+51.00, +70.00] pp | +60.02 pp; [+51.02, +68.67] pp | +60.81 pp; [+52.02, +69.68] pp | 61 | 0 |
| candidate_2: multitask, seed 20261011 | Reader_Thamburaj | 100 | +62.00 pp; [+52.00, +71.00] pp | +60.95 pp; [+51.87, +69.37] pp | +61.74 pp; [+53.05, +70.56] pp | 62 | 0 |
| candidate_3: stage_only, seed 20261012 | Reader_Thamburaj | 100 | +62.00 pp; [+52.00, +71.00] pp | +60.95 pp; [+51.87, +69.37] pp | +61.74 pp; [+53.05, +70.56] pp | 62 | 0 |
| candidate_4: multitask, seed 20261012 | Reader_Thamburaj | 100 | +61.00 pp; [+51.00, +70.00] pp | +60.02 pp; [+51.02, +68.67] pp | +60.81 pp; [+52.02, +69.68] pp | 61 | 0 |
| candidate_5: stage_only, seed 20261013 | Reader_Thamburaj | 100 | +62.00 pp; [+52.00, +71.00] pp | +60.95 pp; [+51.87, +69.37] pp | +61.74 pp; [+53.05, +70.56] pp | 62 | 0 |
| candidate_6: multitask, seed 20261013 | Reader_Thamburaj | 100 | +61.00 pp; [+51.00, +70.00] pp | +60.02 pp; [+50.98, +68.58] pp | +60.81 pp; [+52.05, +69.82] pp | 61 | 0 |
| Old baseline | Reader_Thamburaj | 100 | +62.00 pp; [+52.00, +71.00] pp | +60.95 pp; [+51.87, +69.37] pp | +61.74 pp; [+53.05, +70.56] pp | 62 | 0 |

Fixed panel: 3 readers, 100 common complete images. Reader-case observations are not pooled as independent trials.

| Fixed-panel mean metric | Value |
| --- | --- |
| accuracy | 39.67% |
| balanced_accuracy | 39.79% |
| macro_f1 | 39.26% |

Full original100 case-aligned labels: [original100_case_aligned_old_candidates_readers.csv](../results/original100/original100_case_aligned_old_candidates_readers.csv). Individual prediction CSVs preserve model probabilities.

<details>
<summary>All 100 reference, model and reader labels</summary>

| Case | Reference | Old baseline | candidate_1 | candidate_2 | candidate_3 | candidate_4 | candidate_5 | candidate_6 | Reader_Kalapos | Reader_Kanekar | Reader_Thamburaj |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| image_001.jpg | MildDemented | MildDemented | MildDemented | MildDemented | MildDemented | MildDemented | MildDemented | MildDemented | MildDemented | ModerateDemented | ModerateDemented |
| image_002.jpg | VeryMildDemented | VeryMildDemented | VeryMildDemented | VeryMildDemented | VeryMildDemented | VeryMildDemented | VeryMildDemented | VeryMildDemented | VeryMildDemented | ModerateDemented | MildDemented |
| image_003.jpg | VeryMildDemented | VeryMildDemented | VeryMildDemented | VeryMildDemented | VeryMildDemented | VeryMildDemented | VeryMildDemented | VeryMildDemented | NonDemented | MildDemented | VeryMildDemented |
| image_004.jpg | ModerateDemented | ModerateDemented | ModerateDemented | ModerateDemented | ModerateDemented | ModerateDemented | ModerateDemented | ModerateDemented | NonDemented | MildDemented | MildDemented |
| image_005.jpg | NonDemented | NonDemented | VeryMildDemented | NonDemented | NonDemented | VeryMildDemented | NonDemented | NonDemented | NonDemented | VeryMildDemented | MildDemented |
| image_006.jpg | NonDemented | NonDemented | NonDemented | NonDemented | NonDemented | NonDemented | NonDemented | NonDemented | MildDemented | NonDemented | NonDemented |
| image_007.jpg | MildDemented | MildDemented | MildDemented | MildDemented | MildDemented | MildDemented | MildDemented | MildDemented | NonDemented | ModerateDemented | ModerateDemented |
| image_008.jpg | MildDemented | MildDemented | MildDemented | MildDemented | MildDemented | MildDemented | MildDemented | MildDemented | NonDemented | ModerateDemented | ModerateDemented |
| image_009.jpg | VeryMildDemented | VeryMildDemented | VeryMildDemented | VeryMildDemented | VeryMildDemented | VeryMildDemented | VeryMildDemented | VeryMildDemented | MildDemented | MildDemented | VeryMildDemented |
| image_010.jpg | MildDemented | MildDemented | MildDemented | MildDemented | MildDemented | MildDemented | MildDemented | MildDemented | VeryMildDemented | MildDemented | MildDemented |
| image_011.jpg | ModerateDemented | ModerateDemented | ModerateDemented | ModerateDemented | ModerateDemented | ModerateDemented | ModerateDemented | ModerateDemented | MildDemented | ModerateDemented | ModerateDemented |
| image_012.jpg | VeryMildDemented | VeryMildDemented | VeryMildDemented | VeryMildDemented | VeryMildDemented | VeryMildDemented | VeryMildDemented | VeryMildDemented | MildDemented | MildDemented | ModerateDemented |
| image_013.jpg | VeryMildDemented | VeryMildDemented | VeryMildDemented | VeryMildDemented | VeryMildDemented | VeryMildDemented | VeryMildDemented | VeryMildDemented | MildDemented | ModerateDemented | ModerateDemented |
| image_014.jpg | ModerateDemented | ModerateDemented | ModerateDemented | ModerateDemented | ModerateDemented | ModerateDemented | ModerateDemented | ModerateDemented | MildDemented | ModerateDemented | ModerateDemented |
| image_015.jpg | VeryMildDemented | VeryMildDemented | VeryMildDemented | VeryMildDemented | VeryMildDemented | VeryMildDemented | VeryMildDemented | VeryMildDemented | MildDemented | ModerateDemented | MildDemented |
| image_016.jpg | NonDemented | NonDemented | NonDemented | NonDemented | NonDemented | NonDemented | NonDemented | NonDemented | VeryMildDemented | NonDemented | MildDemented |
| image_017.jpg | ModerateDemented | ModerateDemented | ModerateDemented | ModerateDemented | ModerateDemented | ModerateDemented | ModerateDemented | ModerateDemented | MildDemented | ModerateDemented | ModerateDemented |
| image_018.jpg | ModerateDemented | ModerateDemented | ModerateDemented | ModerateDemented | ModerateDemented | ModerateDemented | ModerateDemented | ModerateDemented | MildDemented | ModerateDemented | ModerateDemented |
| image_019.jpg | NonDemented | NonDemented | NonDemented | NonDemented | NonDemented | NonDemented | NonDemented | NonDemented | NonDemented | MildDemented | MildDemented |
| image_020.jpg | MildDemented | MildDemented | MildDemented | MildDemented | MildDemented | MildDemented | MildDemented | MildDemented | MildDemented | ModerateDemented | MildDemented |
| image_021.jpg | NonDemented | NonDemented | NonDemented | NonDemented | NonDemented | NonDemented | NonDemented | NonDemented | NonDemented | ModerateDemented | ModerateDemented |
| image_022.jpg | NonDemented | NonDemented | NonDemented | NonDemented | NonDemented | NonDemented | NonDemented | NonDemented | NonDemented | NonDemented | NonDemented |
| image_023.jpg | MildDemented | MildDemented | MildDemented | MildDemented | MildDemented | MildDemented | MildDemented | MildDemented | MildDemented | VeryMildDemented | ModerateDemented |
| image_024.jpg | MildDemented | MildDemented | MildDemented | MildDemented | MildDemented | MildDemented | MildDemented | MildDemented | VeryMildDemented | MildDemented | ModerateDemented |
| image_025.jpg | VeryMildDemented | VeryMildDemented | VeryMildDemented | VeryMildDemented | VeryMildDemented | VeryMildDemented | VeryMildDemented | VeryMildDemented | VeryMildDemented | VeryMildDemented | MildDemented |
| image_026.jpg | MildDemented | MildDemented | MildDemented | MildDemented | MildDemented | MildDemented | MildDemented | MildDemented | ModerateDemented | MildDemented | ModerateDemented |
| image_027.jpg | VeryMildDemented | VeryMildDemented | VeryMildDemented | VeryMildDemented | VeryMildDemented | VeryMildDemented | VeryMildDemented | VeryMildDemented | NonDemented | ModerateDemented | MildDemented |
| image_028.jpg | MildDemented | MildDemented | MildDemented | MildDemented | MildDemented | MildDemented | MildDemented | MildDemented | MildDemented | ModerateDemented | ModerateDemented |
| image_029.jpg | MildDemented | MildDemented | MildDemented | MildDemented | MildDemented | MildDemented | MildDemented | MildDemented | VeryMildDemented | MildDemented | MildDemented |
| image_030.jpg | NonDemented | NonDemented | NonDemented | NonDemented | NonDemented | NonDemented | NonDemented | NonDemented | NonDemented | NonDemented | NonDemented |
| image_031.jpg | VeryMildDemented | VeryMildDemented | VeryMildDemented | VeryMildDemented | VeryMildDemented | VeryMildDemented | VeryMildDemented | VeryMildDemented | VeryMildDemented | MildDemented | MildDemented |
| image_032.jpg | VeryMildDemented | VeryMildDemented | VeryMildDemented | VeryMildDemented | VeryMildDemented | VeryMildDemented | VeryMildDemented | NonDemented | VeryMildDemented | MildDemented | MildDemented |
| image_033.jpg | ModerateDemented | ModerateDemented | ModerateDemented | ModerateDemented | ModerateDemented | ModerateDemented | ModerateDemented | ModerateDemented | VeryMildDemented | MildDemented | ModerateDemented |
| image_034.jpg | ModerateDemented | ModerateDemented | ModerateDemented | ModerateDemented | ModerateDemented | ModerateDemented | ModerateDemented | ModerateDemented | NonDemented | MildDemented | ModerateDemented |
| image_035.jpg | NonDemented | NonDemented | NonDemented | NonDemented | NonDemented | NonDemented | NonDemented | NonDemented | NonDemented | NonDemented | NonDemented |
| image_036.jpg | VeryMildDemented | VeryMildDemented | VeryMildDemented | VeryMildDemented | VeryMildDemented | VeryMildDemented | VeryMildDemented | VeryMildDemented | NonDemented | VeryMildDemented | MildDemented |
| image_037.jpg | NonDemented | NonDemented | NonDemented | NonDemented | NonDemented | NonDemented | NonDemented | NonDemented | VeryMildDemented | VeryMildDemented | MildDemented |
| image_038.jpg | VeryMildDemented | VeryMildDemented | VeryMildDemented | VeryMildDemented | VeryMildDemented | VeryMildDemented | VeryMildDemented | VeryMildDemented | NonDemented | NonDemented | MildDemented |
| image_039.jpg | NonDemented | NonDemented | NonDemented | NonDemented | NonDemented | NonDemented | NonDemented | NonDemented | NonDemented | NonDemented | NonDemented |
| image_040.jpg | NonDemented | NonDemented | NonDemented | NonDemented | NonDemented | NonDemented | NonDemented | NonDemented | NonDemented | NonDemented | NonDemented |
| image_041.jpg | ModerateDemented | ModerateDemented | ModerateDemented | ModerateDemented | ModerateDemented | ModerateDemented | ModerateDemented | ModerateDemented | ModerateDemented | ModerateDemented | ModerateDemented |
| image_042.jpg | NonDemented | NonDemented | NonDemented | NonDemented | NonDemented | NonDemented | NonDemented | NonDemented | NonDemented | NonDemented | MildDemented |
| image_043.jpg | VeryMildDemented | VeryMildDemented | VeryMildDemented | VeryMildDemented | VeryMildDemented | VeryMildDemented | VeryMildDemented | VeryMildDemented | NonDemented | NonDemented | VeryMildDemented |
| image_044.jpg | NonDemented | NonDemented | NonDemented | NonDemented | NonDemented | NonDemented | NonDemented | NonDemented | VeryMildDemented | ModerateDemented | ModerateDemented |
| image_045.jpg | ModerateDemented | ModerateDemented | ModerateDemented | ModerateDemented | ModerateDemented | ModerateDemented | ModerateDemented | ModerateDemented | MildDemented | ModerateDemented | ModerateDemented |
| image_046.jpg | VeryMildDemented | VeryMildDemented | VeryMildDemented | VeryMildDemented | VeryMildDemented | VeryMildDemented | VeryMildDemented | VeryMildDemented | NonDemented | MildDemented | MildDemented |
| image_047.jpg | ModerateDemented | ModerateDemented | ModerateDemented | ModerateDemented | ModerateDemented | ModerateDemented | ModerateDemented | ModerateDemented | MildDemented | MildDemented | ModerateDemented |
| image_048.jpg | VeryMildDemented | VeryMildDemented | VeryMildDemented | VeryMildDemented | VeryMildDemented | VeryMildDemented | VeryMildDemented | VeryMildDemented | VeryMildDemented | ModerateDemented | MildDemented |
| image_049.jpg | VeryMildDemented | VeryMildDemented | VeryMildDemented | VeryMildDemented | VeryMildDemented | VeryMildDemented | VeryMildDemented | VeryMildDemented | VeryMildDemented | ModerateDemented | MildDemented |
| image_050.jpg | MildDemented | MildDemented | MildDemented | MildDemented | MildDemented | MildDemented | MildDemented | MildDemented | VeryMildDemented | ModerateDemented | MildDemented |
| image_051.jpg | NonDemented | NonDemented | NonDemented | NonDemented | NonDemented | NonDemented | NonDemented | NonDemented | NonDemented | NonDemented | MildDemented |
| image_052.jpg | NonDemented | NonDemented | NonDemented | NonDemented | NonDemented | NonDemented | NonDemented | NonDemented | VeryMildDemented | NonDemented | VeryMildDemented |
| image_053.jpg | ModerateDemented | ModerateDemented | ModerateDemented | ModerateDemented | ModerateDemented | ModerateDemented | ModerateDemented | ModerateDemented | VeryMildDemented | MildDemented | MildDemented |
| image_054.jpg | NonDemented | NonDemented | NonDemented | NonDemented | NonDemented | NonDemented | NonDemented | NonDemented | NonDemented | NonDemented | NonDemented |
| image_055.jpg | MildDemented | MildDemented | MildDemented | MildDemented | MildDemented | MildDemented | MildDemented | MildDemented | VeryMildDemented | MildDemented | VeryMildDemented |
| image_056.jpg | MildDemented | MildDemented | MildDemented | MildDemented | MildDemented | MildDemented | MildDemented | MildDemented | VeryMildDemented | MildDemented | MildDemented |
| image_057.jpg | VeryMildDemented | VeryMildDemented | VeryMildDemented | VeryMildDemented | VeryMildDemented | VeryMildDemented | VeryMildDemented | VeryMildDemented | MildDemented | MildDemented | MildDemented |
| image_058.jpg | MildDemented | MildDemented | MildDemented | MildDemented | MildDemented | MildDemented | MildDemented | MildDemented | ModerateDemented | ModerateDemented | ModerateDemented |
| image_059.jpg | ModerateDemented | ModerateDemented | ModerateDemented | ModerateDemented | ModerateDemented | ModerateDemented | ModerateDemented | ModerateDemented | ModerateDemented | ModerateDemented | ModerateDemented |
| image_060.jpg | NonDemented | NonDemented | NonDemented | NonDemented | NonDemented | NonDemented | NonDemented | NonDemented | VeryMildDemented | NonDemented | VeryMildDemented |
| image_061.jpg | VeryMildDemented | VeryMildDemented | VeryMildDemented | VeryMildDemented | VeryMildDemented | VeryMildDemented | VeryMildDemented | VeryMildDemented | ModerateDemented | ModerateDemented | ModerateDemented |
| image_062.jpg | ModerateDemented | ModerateDemented | ModerateDemented | ModerateDemented | ModerateDemented | ModerateDemented | ModerateDemented | ModerateDemented | VeryMildDemented | MildDemented | VeryMildDemented |
| image_063.jpg | ModerateDemented | ModerateDemented | ModerateDemented | ModerateDemented | ModerateDemented | ModerateDemented | ModerateDemented | ModerateDemented | VeryMildDemented | MildDemented | MildDemented |
| image_064.jpg | NonDemented | NonDemented | NonDemented | NonDemented | NonDemented | NonDemented | NonDemented | NonDemented | VeryMildDemented | MildDemented | MildDemented |
| image_065.jpg | NonDemented | NonDemented | NonDemented | NonDemented | NonDemented | NonDemented | NonDemented | NonDemented | VeryMildDemented | NonDemented | VeryMildDemented |
| image_066.jpg | VeryMildDemented | VeryMildDemented | VeryMildDemented | VeryMildDemented | VeryMildDemented | VeryMildDemented | VeryMildDemented | VeryMildDemented | MildDemented | ModerateDemented | MildDemented |
| image_067.jpg | MildDemented | MildDemented | MildDemented | MildDemented | MildDemented | MildDemented | MildDemented | MildDemented | MildDemented | ModerateDemented | ModerateDemented |
| image_068.jpg | MildDemented | MildDemented | MildDemented | MildDemented | MildDemented | MildDemented | MildDemented | MildDemented | VeryMildDemented | MildDemented | MildDemented |
| image_069.jpg | MildDemented | MildDemented | MildDemented | MildDemented | MildDemented | MildDemented | MildDemented | MildDemented | MildDemented | MildDemented | ModerateDemented |
| image_070.jpg | NonDemented | NonDemented | NonDemented | NonDemented | NonDemented | NonDemented | NonDemented | NonDemented | VeryMildDemented | MildDemented | MildDemented |
| image_071.jpg | VeryMildDemented | VeryMildDemented | VeryMildDemented | VeryMildDemented | VeryMildDemented | VeryMildDemented | VeryMildDemented | VeryMildDemented | ModerateDemented | ModerateDemented | MildDemented |
| image_072.jpg | NonDemented | NonDemented | NonDemented | NonDemented | NonDemented | NonDemented | NonDemented | NonDemented | ModerateDemented | MildDemented | MildDemented |
| image_073.jpg | MildDemented | MildDemented | MildDemented | MildDemented | MildDemented | MildDemented | MildDemented | MildDemented | ModerateDemented | ModerateDemented | MildDemented |
| image_074.jpg | ModerateDemented | ModerateDemented | ModerateDemented | ModerateDemented | ModerateDemented | ModerateDemented | ModerateDemented | ModerateDemented | VeryMildDemented | NonDemented | VeryMildDemented |
| image_075.jpg | ModerateDemented | ModerateDemented | ModerateDemented | ModerateDemented | ModerateDemented | ModerateDemented | ModerateDemented | ModerateDemented | NonDemented | NonDemented | VeryMildDemented |
| image_076.jpg | ModerateDemented | ModerateDemented | ModerateDemented | ModerateDemented | ModerateDemented | ModerateDemented | ModerateDemented | ModerateDemented | ModerateDemented | ModerateDemented | ModerateDemented |
| image_077.jpg | NonDemented | NonDemented | NonDemented | NonDemented | NonDemented | NonDemented | NonDemented | NonDemented | VeryMildDemented | NonDemented | NonDemented |
| image_078.jpg | MildDemented | MildDemented | MildDemented | MildDemented | MildDemented | MildDemented | MildDemented | MildDemented | VeryMildDemented | ModerateDemented | ModerateDemented |
| image_079.jpg | VeryMildDemented | VeryMildDemented | VeryMildDemented | VeryMildDemented | VeryMildDemented | VeryMildDemented | VeryMildDemented | VeryMildDemented | VeryMildDemented | VeryMildDemented | ModerateDemented |
| image_080.jpg | NonDemented | NonDemented | NonDemented | NonDemented | NonDemented | NonDemented | NonDemented | NonDemented | NonDemented | NonDemented | NonDemented |
| image_081.jpg | NonDemented | NonDemented | NonDemented | NonDemented | NonDemented | NonDemented | NonDemented | NonDemented | NonDemented | NonDemented | NonDemented |
| image_082.jpg | NonDemented | NonDemented | NonDemented | NonDemented | NonDemented | NonDemented | NonDemented | NonDemented | NonDemented | VeryMildDemented | MildDemented |
| image_083.jpg | ModerateDemented | ModerateDemented | ModerateDemented | ModerateDemented | ModerateDemented | ModerateDemented | ModerateDemented | ModerateDemented | ModerateDemented | MildDemented | ModerateDemented |
| image_084.jpg | VeryMildDemented | VeryMildDemented | VeryMildDemented | VeryMildDemented | VeryMildDemented | VeryMildDemented | VeryMildDemented | VeryMildDemented | MildDemented | MildDemented | ModerateDemented |
| image_085.jpg | NonDemented | NonDemented | NonDemented | NonDemented | NonDemented | NonDemented | NonDemented | NonDemented | NonDemented | NonDemented | MildDemented |
| image_086.jpg | MildDemented | MildDemented | MildDemented | MildDemented | MildDemented | MildDemented | MildDemented | MildDemented | MildDemented | VeryMildDemented | ModerateDemented |
| image_087.jpg | MildDemented | MildDemented | MildDemented | MildDemented | MildDemented | MildDemented | MildDemented | MildDemented | MildDemented | VeryMildDemented | MildDemented |
| image_088.jpg | ModerateDemented | ModerateDemented | ModerateDemented | ModerateDemented | ModerateDemented | ModerateDemented | ModerateDemented | ModerateDemented | MildDemented | ModerateDemented | ModerateDemented |
| image_089.jpg | ModerateDemented | ModerateDemented | ModerateDemented | ModerateDemented | ModerateDemented | ModerateDemented | ModerateDemented | ModerateDemented | VeryMildDemented | NonDemented | MildDemented |
| image_090.jpg | NonDemented | NonDemented | NonDemented | NonDemented | NonDemented | NonDemented | NonDemented | NonDemented | NonDemented | NonDemented | MildDemented |
| image_091.jpg | VeryMildDemented | VeryMildDemented | VeryMildDemented | VeryMildDemented | VeryMildDemented | VeryMildDemented | VeryMildDemented | VeryMildDemented | MildDemented | VeryMildDemented | MildDemented |
| image_092.jpg | VeryMildDemented | VeryMildDemented | VeryMildDemented | VeryMildDemented | VeryMildDemented | VeryMildDemented | VeryMildDemented | VeryMildDemented | VeryMildDemented | NonDemented | MildDemented |
| image_093.jpg | MildDemented | MildDemented | MildDemented | MildDemented | MildDemented | MildDemented | MildDemented | MildDemented | ModerateDemented | MildDemented | ModerateDemented |
| image_094.jpg | ModerateDemented | ModerateDemented | ModerateDemented | ModerateDemented | ModerateDemented | ModerateDemented | ModerateDemented | ModerateDemented | ModerateDemented | ModerateDemented | ModerateDemented |
| image_095.jpg | ModerateDemented | ModerateDemented | ModerateDemented | ModerateDemented | ModerateDemented | ModerateDemented | ModerateDemented | ModerateDemented | MildDemented | ModerateDemented | ModerateDemented |
| image_096.jpg | VeryMildDemented | VeryMildDemented | VeryMildDemented | VeryMildDemented | VeryMildDemented | VeryMildDemented | VeryMildDemented | VeryMildDemented | MildDemented | NonDemented | MildDemented |
| image_097.jpg | VeryMildDemented | VeryMildDemented | VeryMildDemented | VeryMildDemented | VeryMildDemented | VeryMildDemented | VeryMildDemented | VeryMildDemented | MildDemented | VeryMildDemented | MildDemented |
| image_098.jpg | ModerateDemented | ModerateDemented | ModerateDemented | ModerateDemented | ModerateDemented | ModerateDemented | ModerateDemented | ModerateDemented | VeryMildDemented | NonDemented | ModerateDemented |
| image_099.jpg | MildDemented | MildDemented | MildDemented | MildDemented | MildDemented | MildDemented | MildDemented | MildDemented | ModerateDemented | VeryMildDemented | ModerateDemented |
| image_100.jpg | VeryMildDemented | VeryMildDemented | VeryMildDemented | VeryMildDemented | VeryMildDemented | VeryMildDemented | VeryMildDemented | VeryMildDemented | MildDemented | VeryMildDemented | VeryMildDemented |

</details>

## Clinical AD/CN endpoint by source and split

Clinical AD/CN is distinct from the four-stage image endpoint. Only multitask candidates have a trained clinical head; no clinical old-versus-new improvement is measured. Each source subject contributes one mean-probability prediction. MCI remains prediction-only and is excluded from every binary metric.

| Multitask candidate | Clinical training subjects | Clinical validation subjects |
| --- | --- | --- |
| candidate_2: multitask, seed 20261011 | 32 | 11 |
| candidate_4: multitask, seed 20261012 | 32 | 11 |
| candidate_6: multitask, seed 20261013 | 32 | 11 |

### external__openneuro_ds007561

| Candidate | All subjects | Binary evaluable | MCI prediction-only | Accuracy correct/n; exact 95% CI | AD sensitivity detected/nAD; exact 95% CI | CN specificity correct/nCN; exact 95% CI | Balanced accuracy | Macro F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| candidate_2: multitask, seed 20261011 | 20 | 15 | 5 | 14/15 (93.33%); [68.05%, 99.83%] | 0/1 (0.00%); [0.00%, 97.50%] | 14/14 (100.00%); [76.84%, 100.00%] | 50.00% | 48.28% |
| candidate_4: multitask, seed 20261012 | 20 | 15 | 5 | 7/15 (46.67%); [21.27%, 73.41%] | 1/1 (100.00%); [2.50%, 100.00%] | 6/14 (42.86%); [17.66%, 71.14%] | 71.43% | 40.00% |
| candidate_6: multitask, seed 20261013 | 20 | 15 | 5 | 13/15 (86.67%); [59.54%, 98.34%] | 0/1 (0.00%); [0.00%, 97.50%] | 13/14 (92.86%); [66.13%, 99.82%] | 46.43% | 46.43% |

### test__besga_zenodo_3935636

| Candidate | All subjects | Binary evaluable | MCI prediction-only | Accuracy correct/n; exact 95% CI | AD sensitivity detected/nAD; exact 95% CI | CN specificity correct/nCN; exact 95% CI | Balanced accuracy | Macro F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| candidate_2: multitask, seed 20261011 | 11 | 11 | 0 | 5/11 (45.45%); [16.75%, 76.62%] | 1/7 (14.29%); [0.36%, 57.87%] | 4/4 (100.00%); [39.76%, 100.00%] | 57.14% | 41.07% |
| candidate_4: multitask, seed 20261012 | 11 | 11 | 0 | 7/11 (63.64%); [30.79%, 89.07%] | 4/7 (57.14%); [18.41%, 90.10%] | 3/4 (75.00%); [19.41%, 99.37%] | 66.07% | 63.33% |
| candidate_6: multitask, seed 20261013 | 11 | 11 | 0 | 4/11 (36.36%); [10.93%, 69.21%] | 1/7 (14.29%); [0.36%, 57.87%] | 3/4 (75.00%); [19.41%, 99.37%] | 44.64% | 34.19% |

An external cohort containing only one AD participant cannot establish stable sensitivity, even with perfect observed accuracy. OpenNeuro was inspected during preprocessing preparation, so it is an exploratory external pilot rather than an entirely unseen pipeline validation. Images can encode age and source differences despite age not being a model input.

## Resource cleanup evidence

The archive and 415 payload files were verified locally before deletion. Task GPU and storage deletion were verified. See [public completion record](../evidence/completion_summary.json); operational identifiers are omitted.

## Interpretation limits and remaining work

- Original100 is a historically reused image-level descriptive benchmark. The old checkpoint had 66 train, 9 validation and 25 test image matches; it is not independent clinical validation.
- Legacy patient IDs and original augmentation families are unknown. Hash grouping is image similarity, not patient identity; cluster intervals do not establish clinical independence.
- The full primary intersection holds out exact rows for both models, but includes hash groups with old training/validation mates. A separately prespecified hash-unexposed sensitivity subset addresses only observed prepared-group overlap, changes sample composition, and does not prove patient independence or absence of other related old images.
- Bootstrap intervals condition on the observed sample and fixed fitted checkpoints. They omit training randomness and dataset/source uncertainty; boundary or identical predictions can yield degenerate intervals that do not prove certainty or equivalence.
- Unadjusted exploratory intervals across multiple candidates, endpoints and readers must not be treated as confirmatory significance or used to select a model on the test set.
- Reader labels concern these fixed images, not prospective clinical diagnosis; a fixed reader panel is not a random sample of neuroradiologists.
- Clinical AD/CN is a separate endpoint from the four legacy stages. MCI remains prediction-only and is excluded from binary accuracy denominators.
- Clinical exact binomial intervals assume independent, representative source subjects. Tiny, selected cohorts, source confounding and preprocessing inspection limit interpretation; one external AD subject cannot establish stable sensitivity.
- The auxiliary comparison tests the complete training regimen, including shared-backbone and batch-normalization updates; it does not isolate only the number of added MRI scans.
- Neither low training loss nor a favorable secondary subset establishes a stronger model overall. Report class-specific regressions and uncertainty; do not select a new winner from held-out results.
- Dedicated probability calibration improvement has not been assessed by this report. NLL and AUROC in the evaluator evidence are separate from the accuracy-based summaries shown here.
- The ten recommended sources are tracked access candidates; the newly acquired source volumes are Besga and OpenNeuro. This experiment does not imply access to or training on all ten. OASIS-1/2/3 should not be pooled without resolving provider overlap constraints.

## Supporting outputs

- [Companion figure: internal_same_case_confusion_all_seven_models.pdf](../figures/evaluation_figures/internal_same_case_confusion_all_seven_models.pdf)
- [Companion figure: internal_same_case_confusion_all_seven_models.png](../figures/evaluation_figures/internal_same_case_confusion_all_seven_models.png)
- [Companion figure: internal_same_case_pr_all_seven_models.pdf](../figures/evaluation_figures/internal_same_case_pr_all_seven_models.pdf)
- [Companion figure: internal_same_case_pr_all_seven_models.png](../figures/evaluation_figures/internal_same_case_pr_all_seven_models.png)
- [Companion figure: internal_same_case_roc_all_seven_models.pdf](../figures/evaluation_figures/internal_same_case_roc_all_seven_models.pdf)
- [Companion figure: internal_same_case_roc_all_seven_models.png](../figures/evaluation_figures/internal_same_case_roc_all_seven_models.png)
- [Companion figure: old_hash_group_unexposed_sensitivity_confusion_all_seven_models.pdf](../figures/evaluation_figures/old_hash_group_unexposed_sensitivity_confusion_all_seven_models.pdf)
- [Companion figure: old_hash_group_unexposed_sensitivity_confusion_all_seven_models.png](../figures/evaluation_figures/old_hash_group_unexposed_sensitivity_confusion_all_seven_models.png)
- [Companion figure: old_hash_group_unexposed_sensitivity_pr_all_seven_models.pdf](../figures/evaluation_figures/old_hash_group_unexposed_sensitivity_pr_all_seven_models.pdf)
- [Companion figure: old_hash_group_unexposed_sensitivity_pr_all_seven_models.png](../figures/evaluation_figures/old_hash_group_unexposed_sensitivity_pr_all_seven_models.png)
- [Companion figure: old_hash_group_unexposed_sensitivity_roc_all_seven_models.pdf](../figures/evaluation_figures/old_hash_group_unexposed_sensitivity_roc_all_seven_models.pdf)
- [Companion figure: old_hash_group_unexposed_sensitivity_roc_all_seven_models.png](../figures/evaluation_figures/old_hash_group_unexposed_sensitivity_roc_all_seven_models.png)
- [Companion figure: original100_confusion_all_seven_models.pdf](../figures/evaluation_figures/original100_confusion_all_seven_models.pdf)
- [Companion figure: original100_confusion_all_seven_models.png](../figures/evaluation_figures/original100_confusion_all_seven_models.png)
- [Companion figure: original100_pr_all_seven_models.pdf](../figures/evaluation_figures/original100_pr_all_seven_models.pdf)
- [Companion figure: original100_pr_all_seven_models.png](../figures/evaluation_figures/original100_pr_all_seven_models.png)
- [Companion figure: original100_reader_confusion.pdf](../figures/evaluation_figures/original100_reader_confusion.pdf)
- [Companion figure: original100_reader_confusion.png](../figures/evaluation_figures/original100_reader_confusion.png)
- [Companion figure: original100_roc_all_seven_models.pdf](../figures/evaluation_figures/original100_roc_all_seven_models.pdf)
- [Companion figure: original100_roc_all_seven_models.png](../figures/evaluation_figures/original100_roc_all_seven_models.png)
- [clinical_exact_intervals.csv](../results/clinical_exact_intervals.csv)
- [paired_differences.csv](../results/paired_differences.csv)
- [reader_paired_differences.csv](../results/reader_paired_differences.csv)
- [stage_metrics.csv](../results/stage_metrics.csv)
- [Old baseline original100 probabilities](../results/original100/old_baseline_original100_predictions.csv)
- [candidate_1: stage_only, seed 20261011 original100 probabilities](../results/original100/candidate_1_original100_predictions.csv)
- [candidate_2: multitask, seed 20261011 original100 probabilities](../results/original100/candidate_2_original100_predictions.csv)
- [candidate_3: stage_only, seed 20261012 original100 probabilities](../results/original100/candidate_3_original100_predictions.csv)
- [candidate_4: multitask, seed 20261012 original100 probabilities](../results/original100/candidate_4_original100_predictions.csv)
- [candidate_5: stage_only, seed 20261013 original100 probabilities](../results/original100/candidate_5_original100_predictions.csv)
- [candidate_6: multitask, seed 20261013 original100 probabilities](../results/original100/candidate_6_original100_predictions.csv)

Report generator SHA256: `8408407421d51b5c12c111301c84cc80f472961a881e6b3926ee79af18824a18`.
