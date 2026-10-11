> Portable public derivative of the audited local source. Clinical subject-level probability rows and operational identifiers are omitted. Scientific aggregate results are unchanged. SHA256 values quoted from the source refer to original frozen files; export hashes are in EXPORT_MANIFEST.json.

# Kevin paper: completed experiment overview

Completed October 11, 2026. All six runs, the old-model comparison, the prespecified ensemble supplement, local retrieval verification and Runpod cleanup are complete.

The archived training orchestrator stopped at `training_complete_awaiting_frozen_evaluation` and was deliberately preserved. That historical status is quoted in the full report; the later [evaluation completion record](../results/evaluation_complete.json) confirms all seven checkpoints were evaluated at 7:10 PM EDT.

The new individual models matched or slightly underperformed the old model on the legacy-image benchmarks. The stage-only ensemble corrected one additional primary image, with a paired uncertainty interval including zero. The clinical AD/control results were weak and unstable. These results do **not** establish an overall stronger or clinically ready model.

## Same 100-image comparison

Counts are correct classifications against the supplied four-class image labels.

| Model | Seed 20261011 | Seed 20261012 | Seed 20261013 |
| --- | ---: | ---: | ---: |
| Stage-only | 99/100 | 100/100 | 100/100 |
| Multitask stage head | 100/100 | 99/100 | 99/100 |

The old baseline scored **100/100**. Both prespecified equal-weight, three-seed stage ensembles scored **100/100**.

The fixed reader scores were Kalapos **37/100**, Thamburaj **38/100**, and Kanekar **44/100**, with all 100 labels present for each reader. This is a historically reused image-label benchmark: 66 images match old training rows, 9 match old validation rows and 25 match old test rows. It cannot establish clinical superiority over neuroradiologists. Perfect observed predictions do not imply population certainty; degenerate bootstrap intervals and exact-binomial sensitivity intervals are disclosed in the full analysis.

## Larger image comparisons and uncertainty

| Model | Primary correct / 1058 | Sensitivity correct / 756 |
| --- | ---: | ---: |
| Old baseline | 1057 | 755 |
| Stage-only, seed 20261011 | 1055 | 753 |
| Multitask, seed 20261011 | 1054 | 752 |
| Stage-only, seed 20261012 | 1057 | 755 |
| Multitask, seed 20261012 | 1056 | 754 |
| Stage-only, seed 20261013 | 1057 | 755 |
| Multitask, seed 20261013 | 1055 | 754 |
| Stage-only ensemble, supplement | 1058 | 756 |
| Multitask ensemble, supplement | 1056 | 754 |

The supplementary stage-only ensemble accuracy gain over the old model was **0.095 percentage points**, with a conditional paired 95% interval of **[0.000, 0.290] percentage points** on the primary. On the sensitivity subset, the gain was **0.132 points [0.000, 0.399]**. Both intervals include zero. All individual models and both fixed ensembles remain reported; none was selected using these outcomes.

The primary is held out by exact rows only; 302 of its images have an observed old training/validation hash-group mate. The prespecified 756-row subset excludes those groups. Neither image hashing nor this exclusion establishes patient independence because legacy participant and augmentation-family identifiers are absent. Intervals use 2,000 paired bootstrap resamples, are exploratory and unadjusted for multiplicity, and do not include training or source uncertainty.

## Separate clinical MRI results

Only the multitask binary AD/control head applies to this endpoint. The old baseline and stage-only heads have no matching clinical endpoint.

| Multitask model | Besga test correct / 11 | OpenNeuro correct / 15 | OpenNeuro AD detected / 1 |
| --- | ---: | ---: | ---: |
| Seed 20261011 | 5 | 14 | 0 |
| Seed 20261012 | 7 | 7 | 1 |
| Seed 20261013 | 4 | 13 | 0 |
| Fixed three-seed ensemble | 5 | 12 | 0 |

The ensemble detected only **1/7** Besga AD subjects and missed the sole OpenNeuro AD subject. Its external sensitivity was **0/1**, with a 95% exact interval of **0–97.5%**. The apparent 14/15 external accuracy of seed 20261011 came from predicting every evaluable subject as control, giving 50% balanced accuracy. Seed 20261012 detected the one external AD subject but falsely labeled eight controls as AD. The five OpenNeuro MCI participants are prediction-only and excluded from binary scoring.

These failures, very small clinical samples, source/age confounding and heuristic preprocessing prevent a real-world-readiness claim. Larger approved cohorts with person identifiers and an untouched clinical evaluation are needed for a separately specified next development study; this experiment was not retuned after its test outcomes.

## Training and datasets

The six runs completed **305 epochs total**: stage-only 38/52/54 and multitask 43/74/44. All stopped under the frozen patience rule; selected epochs were 23/37/39 and 28/59/29 respectively. Both conditions used seeds 20261011, 20261012 and 20261013. Stage-only was preferred from validation before evaluation; seed 20261012 was its exploratory representative.

All recorded curves are unsmoothed. Stage training logs show unweighted, unsmoothed NLL, whereas optimization used weights and label smoothing. Clinical training curves use sampled slices and validation curves use subject-averaged predictions, so the two losses have different aggregation conditions. Low training loss did not establish clinical generalization.

Actual MRI acquisitions were the Besga Zenodo release and OpenNeuro ds007561 v1.0.0: 98 raw T1 volumes acquired, 74 participants included in the prepared AD/control/MCI pipeline. The ten recommended official sources are tracked, but **none of those ten official raw releases was newly acquired**; their access/eligibility requirements remain pending. They must not be described as ten additional training cohorts.

## Paper materials and audit evidence

- [Full six-run report, case-level 100-image comparison, paired intervals and clinical denominators](Full_Results.md)
- [Readable primary and ensemble statistics](Statistical_Handoff.md)
- [Completed Methods addendum](Completed_Methods.md)
- [Stage loss curves: all six runs](../figures/publication_curves/stage_loss_all_six_runs.png)
- [Clinical loss curves: all three seeds](../figures/publication_curves/clinical_loss_all_three_multitask_seeds.png)
- [Evaluation figures and descriptive calibration provenance](../figures/evaluation_figures/evaluation_figure_provenance.json)
- [Figure visual and checksum QA](../evidence/original_figure_qa.json)
- [Frozen ensemble outputs and specifications](../results/ensemble_supplement.json)
- [Dataset recommendation crosswalk](../datasets/recommended_dataset_crosswalk.json)
- [Unresolved readiness requirements](Readiness_Protocol_Snapshot.txt)
- [Verified local retrieval](../evidence/completion_summary.json)
- [GPU and storage deletion evidence](../evidence/completion_summary.json)

The downloaded archive SHA256 is `3b73c98a8c9c2445cb327edf2999d1519eb257ccf123ca3f070d1043f636d2f1`. All **415 payload files**, including checkpoints, were verified against their manifest and read back locally before deletion. Fresh Runpod inventories confirmed the task Pods and network volume absent. No production checkpoint was replaced.
