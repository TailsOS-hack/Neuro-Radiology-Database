# Independent data-gate review

This review used synthetic metadata, synthetic review receipts, and tiny generated NIfTI fixtures only. It did not authorize a dataset, inspect real patient outcomes, fit a model, or allocate compute.

## Findings and resolution

| Finding reproduced during review | Current guard and verification |
| --- | --- |
| Leading/trailing whitespace could create patient or institution aliases. | Noncanonical identity strings are rejected before overlap assessment. Synthetic patient and institution alias probes pass. |
| Missing source-family identifiers could bypass historical-source exclusions. | A canonical source-family identifier is mandatory. Missing-family probe passes. |
| Two same-day assessments could yield input-order-dependent inclusion when one lacked a reference standard. | Matching considers documented assessments deterministically; reversed-order fixtures produce the same selection. Conflicting diagnoses remain excluded. |
| Manifest integer diagnosis could disagree with clinical cognitive-status/etiology evidence. | `diagnosis_from_assessment(row)` must equal the manifest label; the matched assessment is content-addressed. A conflicting-label fixture is rejected even when aggregate class counts remain valid. |
| Hash-shaped review placeholders could substitute for valid digests or preserved evidence. | Digest syntax is validated. File-verifying admission additionally requires actual source permission/identity/mapping artifacts and named manifest review evidence matching their hashes. Metadata-only inspection is not training authorization. |
| A previously observed institution could be renamed as fresh external data. | Historical institution identities and development/external institution overlap are checked, including aliases sharing one institution. |
| Replaying assessment labels alone did not bind the scan's institution or source. | Admission rehashes source scan/assessment files and replays the matching rule, comparing patient/scan identifiers, labels, source/site/institution, volume path and hashes, modality, full-brain declaration, and prior-evaluation status. An institution change in an otherwise rehashed manifest is rejected. |

All 11 tests in `tests/test_round3_data_independent.py` pass. The positive file-verifying fixture now includes explicitly synthetic local review evidence because the root admission gate was strengthened during review. The combined statistics, calibration, sample-size, data, workflow and freeze checks passed 44 tests in the final targeted run.

## Scope and remaining boundaries

Evidence hashes establish which files were reviewed; they do not independently establish legal permission, the truth of a provider diagnosis, or cross-dataset patient identity. These still require documented human review against provider terms, delivered dictionaries, and actual identity information. Equal pixels or hash groups are not patient identity; absence of hash overlap is not proof that subjects differ.

The software rejects site labels that split a single institution into independent validation folds. Site-specific class counts and power assumptions must be reviewed and frozen from the eventual acquired inventory. A planning simulation is conditional on its explicit assumptions and does not prove clinical readiness.

Development authorization now requires pinned preprocessing dependencies and successful loading of every prepared, blinded-review-approved train, tune, and reserved-calibration volume. External failures remain eligible and must enter failure/abstention accounting. External opening additionally replays the complete selection and final training artifact chain, both reserved-data calibration fits and their exact decision thresholds, and the fixed old-model identity. None of these gates replaces a prospective clinical study or clinician acceptance testing.
