# Harvard Dataverse dementia T1 cohort

The official [Harvard Dataverse V2 release](https://doi.org/10.7910/DVN/29352), **Resting-state fMRI in dementia patients** (Mascali et al., 2015), is released under **CC0 1.0**. Its [original study](https://doi.org/10.1371/journal.pone.0120988) reports 10 probable-AD, 10 amnestic-MCI and 10 healthy-control participants, with anatomical T1 MDEFT acquired on a Siemens Allegra 3 T system. The published acquisition matrix is 256 × 224 × 176 with approximately 1 mm spacing. Diagnosis is a clinical group label, not the legacy model's four-stage severity label.

**The actual public archive has only 29 structural scans: 9 AD, 10 healthy controls and 10 MCI.** All 30 participants have functional scans. One AD participant's T1 is absent from both the archive and the separate public file list. The planned binary analysis denominator can therefore be at most 19 before preprocessing/QC, with 10 MCI cases separate. The unavailable participant remains in the private subject table; there is no replacement or imputation.

Acquisition used the ordinary Safari download interface without an account or manual challenge solving. Initial command-line API requests received HTTP 403, but the normal browser served both the release and the native metadata API. The API response was saved as Page Source in [official_dataset_metadata.json](official_dataset_metadata.json); the independent DOI registration is [datacite.json](datacite.json). No access control was bypassed.

The 1,495,117,504-byte `DATA.tar.gz` matches the provider MD5 `e4df070acb9a15451e391b9acd00bf04` and local SHA256 `55a88acce2aa77b75b8c587e9ed76fe48b181ab28d13130a7ad9daf2b4cc4481`. All 59 contained NIfTI files also match their independently listed file MD5 and diagnosis category. All structural headers have shape 224 × 256 × 176 and a payload length consistent with the header. Filename suffixes do **not** safely identify diagnostic groups: the mapping uses the original archive hierarchy and provider categories. See [acquisition_summary.json](acquisition_summary.json).

The original archive, extracted structural files, subject mapping, image headers, prepared images, overlap details and QC panels remain local under `data/external/`. Public reporting should use aggregate summaries, source citations and reproducible code. Header/defacing privacy review remains a separate requirement before any raw-image redistribution.

`prepare_external_t1.py` imports the unchanged `process()` function from the existing `preprocess_clinical_t1.py` (SHA256 `bc3e1c7012cc54e9453af38d554073b3666cbc115fbb2a2072ed10a822f49475`). It keeps all participants external and preserves missing or failed cases. It uses the existing nine fixed 45–85% axial fractions, 1.5 mm RAS resampling, per-volume intensity scaling, crop/padding and 224-pixel output. MCI has a blank binary label. It does not call the old script's `main()`, edit historical manifests, train, or perform inference. [preprocessing_summary.json](preprocessing_summary.json) records actual output counts and QA status.

All 29 available T1 scans were prepared successfully into 261 slices. [Independent readback](output_verification.json) checked every image's pixels, membership, source hash and diagnosis mapping. [Gross visual QC](visual_qc.json) inspected every subject's full nine-slice sequence without model outcomes. No subject was excluded. Exact overlap checks found no matches against 98 prior raw Besga/OpenNeuro scans or 666 prior prepared slices, and no cross-subject exact duplicates within Harvard. Historical preprocessing inputs remained unchanged.

Raw bytes, canonical decoded float32 volumes and prepared RGB pixels are compared against previously acquired Besga/OpenNeuro data. These checks address exact duplicates only; they cannot prove patient independence from unidentified legacy sources. Gross all-subject anatomical inspection is an image-quality check, not radiologist adjudication or validated localization. This small, newly acquired source may support a separately frozen external clinical analysis; acquisition and preprocessing alone establish no model performance or clinical readiness.

Reproduction requires obtaining the provider archive through a permitted public route and the existing local clinical comparison data:

```sh
python verify_acquisition.py --download /path/to/DATA.tar.gz
python prepare_external_t1.py
```

The preparation output must be new. Existing raw archives are accepted only if identical. See the scripts for input checks; neither contains credentials or invokes paid resources.
