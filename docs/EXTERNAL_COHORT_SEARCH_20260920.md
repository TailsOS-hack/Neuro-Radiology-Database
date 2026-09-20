# External dementia MRI acquisition — 20 September 2026

**Direct-download candidate located: Mendeley kcjt4v658x, version 2, under CC BY 4.0.** Download and inspection results are recorded below. The user authorized searching and downloading additional data. Authoritative sources below were inspected. The initial large/cohort candidates below require registration, a data-use agreement or approved investigator access; the later Mendeley candidate has a direct public download. The user requested avoiding registration/application requirements for now. No access request was submitted and no third-party dataset owner was contacted.

## Candidates

| Cohort | Potential use | Current access / suitability boundary |
| --- | --- | --- |
| [UCL MIRIAD](https://www.ucl.ac.uk/brain-sciences/ion/research/research-centres/dementia-research-centre/research-clinical-trials/minimal-interval-resonance-imaging-alzheimers-disease-miriad) | Longitudinal structural MRI with participant grouping, AD/control labels and clinical assessments; a candidate for a separately sourced evaluation | Requires XNAT registration/login and acceptance of the data-use conditions. No redistribution without permission. Clinical labels and screening CDR require a prespecified mapping; repeated visits cannot be treated as independent patients. |
| [OASIS](https://sites.wustl.edu/oasisbrains/request-access/) | Patient-linked MRI and clinical staging; potentially useful for rebuilding a patient-grouped experiment | Current official access route requires a research statement and agreement; OASIS-3/4 additionally require NITRC accounts. Possible relationship to Kaggle upstream data must be investigated before claiming independence. |
| [Nigerian clinical MRI cohort](https://brainlife.io/pub/675af884becd6464d43296c5) | Geographically distinct clinical dataset with dementia and controls | The [published descriptor](https://www.nature.com/articles/s41597-025-04743-0) requires a signed DUA before access. Age-related dementia labels do not directly establish the four stage labels used by the current classifier. The separate brain-mask publication is not the MRI cohort. |
| [ADNI](https://adni.loni.usc.edu/data-samples/adni-data/) | Large clinical imaging cohort with participant and visit metadata | Requires an investigator application and approved access. The current [DUA](https://adni.loni.usc.edu/wp-content/themes/adni_2023/documents/ADNI_Data_Use_Agreement.pdf) places specific restrictions on use with AI tools and disclosure. This must be addressed in the proposed processing environment before acquisition. |

OpenNeuro searches also returned EEG dementia data and healthy-aging cohorts. These do not supply the MRI four-stage reference standard needed here. More augmented Kaggle/Zenodo derivatives would not repair the missing family and patient identities, so they were not downloaded as purported independent evidence.

## Prepared research statement

> We propose an external research evaluation of a frozen MRI image-classification model previously developed on public image datasets. The study will quantify domain shift and performance limitations using subject-linked structural MRI and clinically recorded reference labels. Before evaluating outcomes, we will specify inclusion/exclusion criteria, preprocessing, clinical label compatibility, participant-level grouping and analysis metrics. We will preserve the external cohort for evaluation, avoid fitting model weights or calibration to its outcomes, and report all failures and uncertainty. No clinical decisions or re-identification attempts are planned. Data and participant-level derivatives will remain within the approved research environment under the provider's access and publication conditions.

For the restricted alternatives only, the lead investigator would need to supply their actual name, affiliation, research contact and provider account, and complete the applicable agreement/access process. These facts cannot be inferred from the repository or the two report-recipient email addresses.

## Evaluation specification to finish before downloading outcomes

1. Confirm the source is independent of the existing Kaggle derivative chain and record access rights and version.
2. Freeze checkpoint SHA-256, image orientation, brain extraction, intensity handling, slice selection and patient aggregation. The current software accepts 2D images, not clinical NIfTI volumes; a documented conversion is needed.
3. Validate reference-label compatibility. AD versus controls is a different endpoint from four dementia stages; do not manufacture absent stage labels or derive them arbitrarily from MMSE. Any binary projection must be explicit and separately reported.
4. Keep all visits and slices from a patient together. For a primary cross-sectional evaluation, choose a prespecified visit/scan per patient and report exclusions.
5. Run source-file and decoded-pixel overlap checks, then patient-cluster confidence intervals and subgroup/coverage reporting. Exact overlap absence alone is not proof of independence.
6. Preserve the cohort as external test evidence. If used for development, reserve a different untouched cohort for the eventual external test.

The existing `scripts/evaluate_external_cohort.py` handles a validated 2D manifest and frozen calibration. It does not supply a clinically justified volumetric preprocessing pipeline or missing patient metadata. Training until an already inspected external score exceeds 95% would invalidate its role as an independent test.

## Direct download completed and inspected

Downloaded [Fathi et al., Alzheimer's disease MRI images, Mendeley V2](https://data.mendeley.com/datasets/kcjt4v658x/2), DOI `10.17632/kcjt4v658x.2`, on 20 September 2026 under CC BY 4.0, without registration. Attribution: Sina Fathi, Ali Ahmadi, Mostafa Almasi-Dooghaee and Melika Sadegh (2023). The 71,145,268-byte ZIP is retained locally at `data/external/mendeley_kcjt4v658x_v2/source.zip`, with extracted files and source metadata beside it. SHA-256 and aggregate inspection are in [external_acquisition.json](final_review_20260920/external_acquisition.json).

All **474 DICOM files decoded successfully** after installing the required lossless JPEG codec. There are 24 subject folders: 10 AD, 10 MCI and 4 NC, plus four JPG thumbnails. The description says 26 subjects, inconsistent with both its class counts and the downloaded folders. Metadata describes **T2-weighted and FLAIR sequences**, not a matching four-stage T1 cohort. Folder names provide candidate grouping, not independently verified clinical adjudication.

**This download is a candidate for separately designed research, not valid four-stage external performance evidence for the current model.** Do not translate MCI into VeryMildDemented or arbitrary AD stages. No model inference, fine-tuning or performance claim was made on it. Sequence/preprocessing compatibility, clinical label mapping, de-identification and independence must be resolved before use. DICOM patient-name, ID and birth-date fields are nonempty (their content was not disclosed or published); raw files and thumbnails stay local and Git-ignored. Nonempty fields alone do not establish that values identify real people, but pseudonymization must be verified before redistribution.

Reproduce the aggregate inspection with `python scripts/inspect_external_mendeley.py` after installing `pydicom`, `pylibjpeg` and `pylibjpeg-libjpeg`. There is no requirement to register for this downloaded candidate. The initial restricted candidates remain alternatives only.
