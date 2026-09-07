# Independent-cohort validation readiness

**External validation: not performed.** The supplied fixture consists of generated 16×16 color patterns, not MRI or patients. Its predictions cannot support performance claims.

## Cohort requirements

Obtain a separately sourced cohort with documented access/image-use permissions, a clinically meaningful reference standard, pseudonymous patient identifiers, and imaging compatible with the study's image-level RGB resizing pipeline. Record how subjects and images were selected, exclusions, label adjudication, whether controls are appropriate, and independence from the public training datasets. The present input interface accepts PIL-readable images; conversion from DICOM/NIfTI and clinically justified slice selection are outside this implementation and must be specified before a real study.

Required manifest columns: `image_path`, `patient_id`, `reference_label`, `cohort_id`. Paths are relative to the cohort manifest; patient identifiers must be pseudonymous. Optional columns: `site`, `scanner`, `sequence`, `acquisition_date`, `reference_standard`. Missing optional metadata are explicitly flagged. A JSON mapping must map each source label to an exact frozen model class. Do not equate clinical diagnoses with public-image stage labels without reviewing that mapping.

Only one cohort is evaluated per run. Unknown labels, missing required identifiers, unreadable images, repeated paths, and exact file- or decoded-pixel overlap with any internal manifest image stop evaluation. This exact-overlap check does not exclude near duplicates, shared patients under different images, or institutional/source bias. Cohort independence still requires provenance review. Identifiers are not inferred from filenames.

## Readiness command (synthetic fixture only)

```bash
python scripts/evaluate_external_cohort.py \
  --manifest tests/fixtures/external/manifest.csv \
  --label-mapping tests/fixtures/external/label_mapping.json \
  --internal-manifest tests/fixtures/external/internal_manifest.csv \
  --internal-root tests/fixtures/external \
  --validate-only --output /tmp/external-readiness-example
```

For an actual cohort, use its manifest and explicit mapping, retain the default complete internal manifest and project root for overlap checking, and select `--task dementia` or `--task tumor`. Dementia-only evaluation uses the dementia checkpoint directly, without the source-domain router. Remove `--validate-only` and provide `--checkpoint` to run inference; optionally provide the frozen `temperatures.json` through `--calibration`. The evaluator checks task, class order, and calibration/checkpoint identity. Use a new output directory for every run. No calibration fitting or retraining occurs on external data.

## Reporting

Report image-level accuracy, NLL, Brier and ECE with sample and patient counts. Absent reference classes receive unavailable per-class metrics rather than fabricated estimates. Confidence intervals use 1,000 seeded patient-cluster bootstrap replicates, including every image of each resampled patient. They describe image-level accuracy and supported-class macro F1 under patient resampling; they are not patient-level diagnostic performance. Macro F1 in each bootstrap replicate uses its present classes, which can differ for sparse cohorts. Fewer than two patients yields unavailable intervals.

Before interpreting a real cohort result, review the reference standard, selection process, label compatibility, missing metadata, scanner/sequence differences and patient independence. Freeze the model and any internal calibration artifact before evaluation. Keep independent results separate from internal benchmark tables, and report failures and subgroup coverage. No clinician, institution or dataset owner has been contacted through this work.
