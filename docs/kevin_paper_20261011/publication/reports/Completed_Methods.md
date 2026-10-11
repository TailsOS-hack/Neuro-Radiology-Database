> Portable public derivative of the audited local source. Clinical subject-level probability rows and operational identifiers are omitted. Scientific aggregate results are unchanged. SHA256 values quoted from the source refer to original frozen files; export hashes are in EXPORT_MANIFEST.json.

# Kevin dementia experiment: completed-workflow Methods addendum

This addendum records execution facts verified after local retrieval. It supplements, without rewriting, the pre-evaluation protocol snapshot. It does not summarize held-out performance or designate a new model from test outcomes.

## Completed training and frozen selection

Six research runs completed on an NVIDIA RTX PRO 6000 Blackwell Server Edition GPU using PyTorch 2.8.0+cu128: stage-only and multitask conditions at seeds 20261011, 20261012 and 20261013. The retrieved run configurations and all six selected checkpoint hashes match the frozen selection record. Contiguous histories record 305 completed training epochs in total. Every run stopped after 15 epochs without the required stage-validation NLL improvement; none reached the 100-epoch maximum. The 20-epoch stopping floor and minimum improvement of 0.0001 were retained. A selected checkpoint therefore need not have the mathematical minimum observed NLL if later improvements were smaller than 0.0001.

| Candidate | Condition | Seed | Completed epochs | Selected epoch |
| --- | --- | --- | --- | --- |
| candidate_1 | stage_only | 20261011 | 38 | 23 |
| candidate_2 | multitask | 20261011 | 43 | 28 |
| candidate_3 | stage_only | 20261012 | 52 | 37 |
| candidate_4 | multitask | 20261012 | 74 | 59 |
| candidate_5 | stage_only | 20261013 | 54 | 39 |
| candidate_6 | multitask | 20261013 | 44 | 29 |

The validation-only selection record was frozen at 2026-10-11 23:10:27 UTC, after all training runs completed and before held-out model evaluation. Its prespecified rule preferred the stage-only condition and designated candidate_3 (seed 20261012, epoch 37) as the exploratory representative. Both conditions and all six seeds remain in the reported experiment; this preference does not establish clinical superiority.

Both conditions began with ImageNet-initialized MobileNetV3-Large rather than the old study checkpoint. The four-stage head learned the supplied legacy labels. The multitask condition added a distinct AD/control head with auxiliary loss weight 0.25 every fourth stage step, balanced subject/slice sampling, and shared backbone updates. Training used complete shuffled stage passes, batch size 32, AdamW at initial learning rate 0.0001 and weight decay 0.001, label smoothing 0.05, gradient clipping at 5 and cosine learning-rate annealing. Raw logged losses retain their original definitions: unweighted unsmoothed stage cross-entropy, and clinical sampled-slice training loss versus subject-aggregated validation loss. Augmentation and dropout were active during training. These differences must remain explicit when interpreting train/validation curves.

## Evaluation and reproducibility accounting

The guarded evaluation workflow used the old baseline and all six frozen checkpoints, with the same candidate ordering as the selection record. It retained the original100 case-aligned model/reader comparison, the complete 1,058-image primary intersection and separate clinical source/split outputs. The old checkpoint and complete comparison intersection were checked before evaluation. No test outcome was used to alter a checkpoint, threshold, condition or seed. The prespecified 756-image sensitivity analysis uses the existing primary predictions rather than additional model inference.

The prespecified ensemble supplement uniformly averages all three seeds within each condition, without fitted weights, member selection, recalibration or threshold tuning. Only trained multitask AD/control heads contribute to clinical ensembling. Both ensembles and all individual members remain visible. Descriptive Brier scores, fixed-bin ECE and disagreement measures do not constitute a fitted calibration method, validated confidence or clinical robustness. The frozen supplement does not change the primary seven-model comparisons.

The completed-result archive was retrieved and checksum verified, with 415 payload files and local destination readback recorded. This addendum independently matched all six best-checkpoint hashes and selected validation epochs to the retrieved histories. The preserved suite_status.json describes the training-only phase: its “awaiting frozen evaluation” and heldout_inference_performed=false fields are historical orchestrator state, not the current end-to-end workflow status. Final uncertainty outputs, figures and model-performance interpretations are separate artifacts; no held-out outcome metrics were inspected to write this methods addendum.

## Data and interpretation boundaries that remain

The 44,000-image legacy source was augmented upstream and lacks verified patient and original augmentation-family identifiers. The four supplied labels are not a validated mapping to clinical CDR stages. Conservative pixel/dHash grouping and exclusions leave 27,178 images: 19,023 training, 2,705 validation and 5,450 test. Hash grouping is image similarity, not patient identity.

The newly acquired original MRI data comprise 78 Besga and 20 OpenNeuro source participant records/volumes. Besga contributes 54 AD/control participants, split into 32 training, 11 validation and 11 internal-test participants; 24 bipolar participants remain outside the primary endpoint. All 20 OpenNeuro participants remain outside fitting and selection: 15 are binary evaluable (1 AD, 14 control), and 5 MCI participants are prediction-only. The 666 preprocessed slices from 74 included source participants do not increase the independent participant count. Cross-repository participant identity remains unresolved.

Clinical preprocessing used canonical RAS orientation, 1.5-mm resampling, per-volume intensity normalization and nine fixed axial fractions, with heuristic foreground/cropping and 224-pixel rendering. It was not validated skull stripping, atlas registration, scanner harmonization or radiologist quality assurance. Both sources were inspected while the slice-sampling rule was developed, although no model outcomes were used; OpenNeuro is therefore an exploratory external pilot, not an entirely unseen preprocessing validation. Source appearance and age confounding remain.

The primary 1,058-image comparison contains 966 prepared similarity groups. Of those images, 302 have an old training/validation group mate; the 756-image secondary subset excludes the observed exposed groups but still cannot establish patient independence or eliminate unknown relationships. Its changed sample composition must be acknowledged. The original100 set matches 66 old strict training, 9 validation and 25 test rows. It is a historically reused descriptive benchmark with three fixed readers, not independent clinical evidence or a population-level comparison with neuroradiologists.

The ten recommended sources remain acquisition priorities: ADNI, OASIS-1/2/3/4, NACC, AIBL, MIRIAD, NIFD/FTLDNI and ALLFTD. Their official raw releases were not newly acquired for this experiment. Dataset names, portals and overlapping releases must not be counted as additional independent cohorts; OASIS-1/2/3 have specific provider overlap warnings. Clinical etiology, severity, matched visit timing and class availability require release-specific audits before any later inclusion.

The completed computation does not resolve clinical-readiness blockers. Clinical validation still requires appropriate original-data cohorts, source/person overlap checks, clinically adjudicated reference labels, subgroup and quality evaluation, and a defined intended use. Current softmax/argmax inference has no fitted calibration/abstention component or deployed unsupported-modality rejection gate. Any future changes require validation-only development and new untouched evaluation; a prospective silent workflow study with qualified clinical oversight remains a later milestone. Conditional intervals and three-seed spread do not remove these limits.

## Supporting evidence

- [Completed-run evidence and hashes](../evidence/completed_methods_evidence.json)
- [Preserved pre-evaluation Methods protocol](Prespecified_Methods.md)
- [Frozen validation-only selection](../results/validation_selection.json)
- [Verified archive retrieval](../evidence/completion_summary.json)
- [Clinical preprocessing provenance](../datasets/clinical_preprocessing_summary.json)
- [Recommended dataset access and overlap crosswalk](../datasets/recommended_dataset_crosswalk.json)
- [Prespecified secondary sensitivity audit](../evidence/sensitivity_audit.json)
- [Clinical-readiness assessment](../evidence/readiness_protocol_snapshot.json)
- [Prespecified ensemble supplement](../evidence/ensemble_protocol.json)
