# Dementia model and dataset handoff for Abid

**Status on October 11, 2026:** Round 1 and Round 2 research experiments are complete. Round 3 software and a study design are prepared, but Round 3 has no acquired eligible patients, trained checkpoints or clinical results. The strongest available independent clinical test exposed a serious failure: the Round 2 ensemble classified every Harvard control as AD. The package supports transparent exploratory reporting, including this negative result; it does not establish clinical readiness or superiority.

## Which model is which

| Model | Task and input | What changed | Interpretation |
|---|---|---|---|
| Original production classifier | Four supplied legacy image labels, using a 2D image | Preserved unchanged; remains the historical reference | Not an independently validated AD/CN patient classifier |
| Round 1 stage-only models, c1/c3/c5 | Four legacy labels; ImageNet-initialized MobileNetV3-Large | New fitting on the conservatively grouped legacy split; three seeds | c3 was the validation-selected representative; all seeds remain reported |
| Round 1 multitask models, c2/c4/c6 | Shared MobileNetV3-Large backbone, separate four-class and binary AD/CN heads | Clinical auxiliary loss weight 0.25 every fourth stage step; subject/slice sampling; clinical patient score averages nine slice probabilities | The fixed mean of these three clinical heads is the **old binary clinical ensemble**. This is distinct from the original production classifier |
| Round 2 stage models | Same four-label legacy task | Continued the three stage-only checkpoints with background-restricted augmentations, intensity/noise/blur controls, validation stress scoring and guards against clean-validation degradation | Tests corner dependence under specified controls; does not prove elimination of all shortcuts |
| Round 2 clinical models | Separate binary AD/CN MobileNetV3-Large models; nine T1 slices per patient | Patient-level loss, one term per training patient per epoch; frozen BatchNorm; head-only warm-up for five epochs, then final three feature blocks plus head; mean slice **logits** before softmax | The new fixed ensemble averages all three patient-probability vectors. This is a whole-pipeline comparison, not proof of benefit from one isolated change |
| Round 3 research pipeline | Planned whole-T1-volume AD/CN support | Full-brain multiplanar versus compact 3D models, verified preprocessing, institution-held-out selection, development calibration and explicit failures/abstentions | Implemented software, **not a newly trained or validated model**; 109 software tests do not measure clinical performance |

The four legacy labels are MildDemented, ModerateDemented, NonDemented and VeryMildDemented. They are not established clinical CDR stages. No mapping of those labels to the separate binary patient endpoint was introduced. Production checkpoint SHA256 remains `a8cc03aea74f5310c1fc7936c72f94a44e55af11e2ada2a8ba5f8edd6d924db5`.

## Data actually used, excluded, and still pending

| Source | Verified size and role | Status and limits |
|---|---|---|
| Legacy Singhal augmented Kaggle derivative | 44,000 source images; conservative exclusions/grouping retained 19,023 training, 2,705 validation and 5,450 test images | Four-class image experiment only. Verified patient and original augmentation-family identities are unavailable; image counts are not patient counts |
| Besga, Zenodo 3935636 | 78 source T1 records: 35 AD, 19 CN, 24 bipolar. AD/CN split: training 32 (21 AD/11 CN), validation 11 (7/4), internal test 11 (7/4) | Only development source for the completed clinical models. Bipolar cases excluded. Test cohort was already observed before Round 2 |
| OpenNeuro ds007561 v1.0.0 | 20 participants: 1 AD, 14 CN, 5 MCI | 15 binary-evaluable external cases, already inspected/evaluated in prior work; five MCI prediction-only. No fitting or checkpoint selection on this cohort |
| Harvard Dataverse 29352 v2 | 29 available T1 volumes: 9 AD, 10 CN, 10 MCI | New source-held-out evaluation in Round 2: 19 binary-evaluable patients, ten MCI prediction-only. Original study reports 30 participants, but one AD T1 is absent from the release. Now an observed historical cohort, never a fresh test again |
| AlzaSet | 12,741 JPEGs; independent patient count unverified | Quarantined, no training or inference: provider article/archive counts differ and patient/series mapping is unresolved |
| NACC/SCAN | Request submitted and provider receipt verified | Approval, delivery, raw MRI entitlement, usable people/sites and derived-model permissions remain unverified. **No NACC data trained this model** |
| BrainLat | Official access route and draft prepared | MRI permission, Synapse account, current participant mapping and usable T1 counts pending. Multimodal catalog totals are not T1 patient counts |
| MIRIAD | Provider catalog: 69 participants (46 AD/23 controls), 708 repeated scans | Not acquired; registration endpoint was unreachable at the recorded check. Repeat scans are not independent patients and this is a single acquisition source |

Besga/OpenNeuro produced 666 prepared slices from 74 included source participants; Harvard added 261 slices from 29 available volumes. Nine slices still represent one person. Do not add cross-repository patient totals as if global identity independence were proved. Raw/canonical/pixel checks found no exact Harvard matches to acquired Besga/OpenNeuro data, but exact-duplicate checks do not establish universal patient independence.

Recorded terms: Besga CC BY 4.0; pinned OpenNeuro and Harvard releases CC0; the legacy derivative/upstream licensing chain remains unresolved. AlzaSet's recorded CC BY license does not resolve its data-quality issues. Controlled-access candidates retain their provider agreements. This handoff contains aggregate clinical results and historical legacy attribution figures, not new restricted raw MRI, clinical subject tables, application identity/signatures or access credentials. Before manuscript submission, authors must verify release-specific attribution, image reuse permission and their actual secondary-data oversight determination; no ethics approval or exemption is asserted here.

## Actual old-versus-new clinical results

Both fixed clinical ensembles used an AD threshold of 0.5 without post-test calibration or threshold adjustment. Old-model aggregation remains mean slice probabilities; new-model aggregation is softmax of mean slice logits.

| Cohort | Old binary ensemble accuracy, exact 95% CI | New binary ensemble accuracy, exact 95% CI | Old → new balanced accuracy |
|---|---|---|---|
| Harvard, fresh at Round 2 evaluation (9 AD/10 CN) | 10/19; 52.6% (28.9–75.6) | 9/19; 47.4% (24.4–71.1) | 55.0% → 50.0% |
| Besga reused internal test (7 AD/4 CN) | 5/11; 45.5% (16.7–76.6) | 10/11; 90.9% (58.7–99.8) | 57.1% → 87.5% |
| OpenNeuro reused external (1 AD/14 CN) | 12/15; 80.0% (51.9–95.7) | 14/15; 93.3% (68.1–99.8) | 42.9% → 50.0% |

On Harvard, the new ensemble and every new individual model predicted AD for all 19 patients. New sensitivity was 100% (66.4–100), but specificity was **0% (0–30.8)**. The old ensemble's specificity was 10% (0.3–44.5). New AUC 0.8444 versus old 0.8222 is a ranking result and does not repair the fixed-threshold failure. The paired new-minus-old accuracy difference was −5.3 percentage points, with a conditional 95% interval of −15.8 to 0.0.

On OpenNeuro, the new ensemble predicted CN for every patient and missed the sole AD case: sensitivity 0%, exact 95% interval 0–97.5%. Its 93.3% accuracy is dominated by controls. Seed 20261023 scored 15/15 here, but selecting that seed after seeing outcomes would invalidate the prespecified comparison. All three seeds and the fixed ensemble are retained. Fifteen MCI patients across Harvard/OpenNeuro are prediction-only; this is not evidence of early AD detection or progression prediction.

The clinical analysis retains all 3,360 patient/model/scenario records in the private source and all 168 aggregate scenario metrics in the public export. Paired intervals use 10,000 diagnosis-stratified patient bootstrap draws. These are conditional, unadjusted exploratory comparisons; resampling OpenNeuro's one AD person does not estimate population sensitivity precisely.

## Historical original100 and other four-class comparisons

| Legacy benchmark | Original classifier | Round 1 representative c3 | Round 2 seeds 20261011 / 20261012 / 20261013 |
|---|---:|---:|---:|
| Original100 | 100/100 | 100/100 | 100/100, 100/100, 100/100 |
| Same-case 1,058 | 1,057/1,058 | 1,057/1,058 | 1,058/1,058, 1,057/1,058, 1,058/1,058 |
| Conservative 756 | 755/756 | 755/756 | 756/756, 755/756, 756/756 |
| Full reused 5,450 test | 5,446/5,450 | 5,443/5,450 | 5,448/5,450, 5,446/5,450, 5,448/5,450 |

The three fixed readers averaged 39.67% on the original100 (descriptive range 37–44%). This is a historical same-label comparison, **not proof of superiority to neuroradiologists clinically**. Original100 overlaps 66 original-training, nine original-validation and 25 original-test rows. In the 1,058 exact-row-held-out set, 302 images have an old training/validation similarity-group mate. The 756 sensitivity set removes those observed mates but cannot establish patient independence. Every legacy benchmark was known during Round 2 development. The Round 1 report contains every individual candidate, including less favorable results omitted from this compact representative table.

## Training and Grad-CAM interpretation

Round 1 stage-only/multitask seeds were 20261011–20261013, with completed/selected epochs c1 38/23, c2 43/28, c3 52/37, c4 74/59, c5 54/39 and c6 44/29. Round 2 stage seeds completed 21/33/35 epochs and selected 6/18/20; clinical seeds 20261021–20261023 completed 80/53/78 and selected 65/38/63. All stopped on validation plateau. Raw unsmoothed losses retain their different training, clean-validation and stress definitions. A lower loss is not evidence of transportability.

The verified training hardware was **NVIDIA RTX PRO 6000 Blackwell Server Edition**, using PyTorch 2.8.0+cu128 and CUDA 12.8. The final clinical evaluation used a different GPU, as distinguished below; the hardware name does not imply medical validity.

| Phase | Recorded GPU | Evidence basis |
|---|---|---|
| Round 1, all six stage-only/multitask runs | NVIDIA RTX PRO 6000 Blackwell Server Edition | Suite runtime environment and six completed run records |
| Round 2 stage seeds 20261011, 20261012 and 20261013 | NVIDIA RTX PRO 6000 Blackwell Server Edition | Active-training GPU snapshots identify each seed; per-seed environments record CUDA 12.8 |
| Round 2 clinical seeds 20261021, 20261022 and 20261023 | NVIDIA RTX PRO 6000 Blackwell Server Edition | Original training runtime and completed clinical suite |
| Final frozen Round 2 clinical comparison | NVIDIA B300 SXM6 AC | Separate evaluation GPU verification; **evaluation only, no retraining** |
| Round 3 | No training performed | Data/runtime gates remain blocked |

The regenerated Grad-CAM collection covers **115 fixed legacy stage-model cases**: original100, eight historical class-balanced examples and seven earlier gallery examples. It compares the original classifier, Round 1 c3 and validation-selected Round 2 stage seed 20261011. These are not clinical binary-model explanations. No model or heatmap was selected for a more anatomical appearance.

All 115 new native 7×7 maps were valid; all 72 parameter-randomization records are retained, including any flat/failed controls. Native cells have broad overlapping receptive fields; an interpolated colored corner is not proof of input-corner dependence. On the fixed 160-image validation control set, the representative retained 160/160 accuracy after prescribed corner/border erasures, while corner-only inputs scored 25%. This is evidence about those specific controls, not proof that every shortcut was eliminated.

**Attribution probability replay remains blocked:** two original100 cases differed from the frozen evaluation by 0.0001177192 and 0.0003736615, exceeding the unchanged 0.0001 tolerance. Classes agreed. A later serialization recovery used saved arrays only and did not rerun inference or remove the failure. Keep warning labels, both numerical exceptions and the full collection; do not present selected attractive maps as validated localization.

## Round 3 changes prepared, not yet demonstrated

The separate Round 3 pipeline encodes patient/visit/source partitions, scan-to-diagnosis matching within 180 days (90-day sensitivity), external-site and overlap gates, an explicit sample-size review, at most 24 whole-brain model configurations, institution-held-out development selection, three fixed final seeds and development-only slope/intercept calibration. Proposed preprocessing uses N4, SynthStrip and rigid template registration with pinned assets and quality checks; that real runtime is still unverified. An anatomical-volume baseline needs independently verified anatomical features. Neither it nor the neural models has a new patient result.

The research progression target requires all external sites to satisfy sensitivity/specificity ≥85% with exact lower 95% bounds ≥80%, plus at least a five-point estimated equal-site balanced-accuracy gain over the old binary ensemble with a paired lower 95% bound above zero. Failures/abstentions count as failures. These are project research targets, not universal clinical standards. No data have yet passed this gate. Clinical use would still require a clinical partner, representative differential-diagnosis patients, fresh assisted-reader research, prospective silent evaluation and supervised deployment validation.

## Files to use when drafting

- `round2_publication/RESULTS.md`, `Methods.md`, `DATA_AND_LIMITATIONS.md`: final scientific account.
- `round2_publication/combined_graphs/Round2_All_Graphs_Atlas.pdf`: 49-page overview; individual export-resolution figures are also included.
- `round2_publication/artifacts/stage/attribution_diagnostics/gallery.html`: all 115 warning-bearing panels and controls; keep its companion files together.
- `round2_publication/artifacts/clinical/report/`: exact intervals, all-seed/scenario metrics and negative external results.
- `publication/reports/Full_Results.md` and `Completed_Methods.md`: complete Round 1 record; an old training-status string in the report is historical, not an unfinished current run.
- `round3/`: proposed next-study design, software verification, unmet data/runtime gates and provider-access status.
- `FIGURE_CAPTION_GUIDANCE.md` and `facts.json`: caption boundaries and source hashes for this handoff.

Paths above are relative to `docs/kevin_paper_20261011` in the repository/source tree. Consult the delivery README for archive layout. The current evidence is suitable for coauthor review and honest exploratory manuscript development; journal acceptance and a clinical-superiority claim remain unresolved.
