# Final review — 20 September 2026

**Submission status: BLOCKED. The internal dementia accuracy target is met; an issue-free publication or independent clinical accuracy is not established.** No retraining was performed: both dementia runs already exceed 95%, and more epochs cannot repair unknown patient/augmentation-family overlap. Accepted models, split manifests, historical predictions and frozen review bundles were preserved.

## Verified results

Recounted every archived test prediction and reconstructed confusion matrices, accuracy and macro F1 for all five models in both result sets (10 evaluations).

| Model | Historical primary accuracy | Perceptual sensitivity accuracy |
| --- | ---: | ---: |
| Binary router | 100.000% | 100.000% |
| Tumor specialist | 97.924% | 95.394% |
| Dementia specialist | 99.909% (8 errors / 8,806) | 99.752% (22 errors / 8,881) |
| Hierarchical | 99.629% | 98.945% |
| Eight-class | 99.717% | 99.055% |

These are internal image-level results. In particular, the primary integrated/tumor results must not be described as free of duplicate leakage. The sensitivity results are separately trained historical experiments, not replacement default checkpoints or independent patient-level validation.

## New data finding

Decoded all 51,023 distinct source files and audited both complete frozen manifests. All images were readable, all manifest paths were unique, and neither manifest had cross-split file-SHA256 matches. There were no conflicting class labels among identical-file or identical-RGB groups.

The primary manifest nevertheless contains **125 cross-split groups / 301 rows with identical RGB pixels**, all in the tumor domain (111 train–test groups and 14 validation–test groups). Different file encodings can defeat file-hash deduplication. This is a substantive limitation of the primary tumor and combined-model evaluations. The exact split combinations and example paths are recorded in [data_audit.json](final_review_20260920/data_audit.json).

Dementia had **zero identical-RGB cross-split rows**, but 21,186 dementia rows participate in identical-dHash cross-split groups in the primary manifest (22,304 rows across both domains). dHash collisions flag similar images, not proven patient identity or mislabeled examples. The sensitivity manifest has zero cross-split file, RGB-pixel and identical-dHash overlap. Within-split dHash collisions can span labels; this does not by itself prove label errors.

Neither audit resolves related slices, augmented copies with changed pixels/hashes, or common patients. Dementia was augmented upstream before receipt; patient and original-image family IDs remain unavailable. Manuscripts, figure caption and workflow wording were corrected to distinguish upstream augmentation from this project's training-time augmentation. The publication gate now fails explicitly on the decoded-pixel finding. Future exact-deduplicated manifest creation now groups both file hashes and decoded RGB hashes; a regression test verifies that re-encoded copies stay together in the official tumor test split. This fixes future split construction without changing historical manifests, checkpoints or accuracy claims. Regenerating a historical manifest in place would invalidate its archived results; use a new experiment directory and retrain when developing a revised model.

## Training curves

[Primary curves](final_review_20260920/primary_training_curves.png) and [sensitivity curves](final_review_20260920/sensitivity_training_curves.png) show every saved epoch, with no smoothing, omitted epochs or edited measurements. Vector SVG versions and hashed source-history references are included in the same folder. Dotted lines mark the last epoch tied for best validation accuracy, matching the selection rule in the training code; checkpoint bytes were not reloaded in this review.

Dementia converged steadily:

| Run | Epochs | First train / validation loss | Final train / validation loss |
| --- | ---: | ---: | ---: |
| Primary | 16 | 0.83895 / 0.66816 | 0.21530 / 0.21155 |
| Sensitivity | 17 | 0.83273 / 0.71638 | 0.21443 / 0.21205 |

Use the machine-readable history summary for exact values. Losses are historical batch-averaged, class-weighted, label-smoothed training objectives; they are not directly comparable to unweighted test NLL. Training also uses augmentation and training mode. The reducer averages batch losses by image count rather than globally summing class-weight denominators, so report these as logged losses, not an exact global weighted objective.

The tumor sensitivity run has a final validation-loss spike to **2.1191**, despite validation accuracy of 98.02%; its best validation-accuracy epoch is earlier. Binary and eight-class models also show train/validation loss gaps. These observations remain visible and should not be hidden by smoothing or by selectively dropping epochs. No global claim of ideal loss curves is warranted.

## Grad-CAM and calibration

Visually inspected both original specialist figure panels: originals, native 7×7 maps, overlays, labels, confidence and error cases are legible. The panels deliberately retain mistakes and low-confidence correct cases. They remain exploratory attribution figures; broad or border-focused activation is not anatomical validation. Do not redraw heatmaps to make them appear medically plausible.

The full baseline integrity check passed for all 270 cases, including original image hashes, native maps, frozen source snapshots, probabilities and historical bundle inventory. The v2 evidence verifier reproduced aggregates for 256 fixed sampled images, 1,536 perturbations, 2,304 randomization attempts and calibration records for 4,952 validation / 10,251 test images. Of the randomized maps, 385 were flat/failed and remain explicitly counted. Dementia top-CAM masking performed better than shape controls but worse than random-pixel controls in the recorded comparisons. This control dependence prevents a blanket faithfulness claim.

The archived dementia ECE changes from 0.038596 to 0.000581 after internal validation-fitted temperature scaling, with unchanged predicted labels. This is research-only calibration on previously inspected internal data; it does not establish clinically calibrated probabilities. No external MRI cohort has been evaluated; the external fixture is synthetic.

## Scope, verification and remaining work

Reviewed the training/splitting/evaluation path, inference/reporting documentation, both model result sets, data provenance, loss histories, explainability/calibration evidence, manuscript claims and publication checks. This was an evidence and source review, not a new end-to-end GUI or GPU training run. Local default model files are Git LFS pointers; fresh checkpoint inference and retraining were not performed. Existing numerical artifacts were verified without claiming a new model evaluation.

- Existing validation and deliberate-corruption suite: 20 tests passed.
- New decoded-pixel re-encoding, future split isolation and unreadable-image regression tests: 3 passed.
- Baseline full source-image integrity and v2 numerical evidence checks passed.
- Publication gate now correctly blocks the historical primary claim on newly detected decoded-pixel leakage; warnings retain patient/family independence, absent external validation, attribution limitations and historical calibration.
- New curve figures and existing Grad-CAM panels were visually inspected. Archived bundles were not rewritten.

Before submission, obtain source/patient/family metadata or transparently narrow the study to an internal augmented-image benchmark; reconcile the primary-versus-sensitivity analysis with the manuscript and checkpoint evidence; obtain an independent cohort for clinical generalization claims; and complete authors, affiliations, venue, references, funding/conflicts and dataset-license review. Live Kaggle pages were requested on 20 September but returned no readable card/license body, so license terms were **not reverified**: [local dataset](https://www.kaggle.com/datasets/aryansinghal10/alzheimers-multiclass-dataset-equal-and-augmented), [upstream dataset](https://www.kaggle.com/datasets/uraninjo/augmented-alzheimer-mri-dataset-v2).

Any future retraining should use a prospectively frozen patient/original-family split, training-only augmentation, validation-based selection and a newly reserved independent test set. Do not keep adapting models to this already inspected test set until a target score appears. The present artifacts cannot establish zero remaining issues or guarantee publication acceptance.

## Additional cohort search

After the user authorized acquisition of additional data, current provider pages were checked for MIRIAD, OASIS, the Nigerian clinical MRI cohort and ADNI. The user then requested a no-registration option. Downloaded the CC BY 4.0 Mendeley cohort `10.17632/kcjt4v658x.2` and decoded all 474 DICOM files in 24 subject folders. Its sequences are T2/FLAIR and labels are AD/MCI/NC, so it is not a matching four-stage external validation set. Raw files are local and Git-ignored; no external performance is claimed. Details, checksum, attribution and limitations are in [the acquisition note](EXTERNAL_COHORT_SEARCH_20260920.md).

## Reproduction

```bash
python scripts/audit_review_data.py
MPLCONFIGDIR=/tmp/neuro-mpl python scripts/build_training_review.py
python -m unittest discover -s tests -p 'test_*.py' -v
python -O scripts/check_explainability_bundle.py --full
python -O scripts/check_validation_review.py docs/review_bundle_v2/validation-v2-20260907
python scripts/check_publication_package.py
```

The final command is expected to exit nonzero while the decoded-pixel primary-split issue remains. NumPy, SciPy, Pillow and Matplotlib are needed; the source audit requires the original local images. Re-running the source audit does not modify a manifest or retrain a model.
