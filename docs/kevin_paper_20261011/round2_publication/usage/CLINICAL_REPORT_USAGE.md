<!-- Portable documentation copy; frozen original retained unchanged. -->
# Verified clinical report

Run only after the complete clinical suite and its evaluation have been retrieved and independently verified. This command reads existing predictions and training histories. It performs no model inference, fitting, calibration fitting, threshold tuning, or selection. Keep the private patient manifest and record-level outputs out of public exports.

From the repository root, set `CLINICAL_SUITE` to the verified local directory containing `clinical_evaluation_freeze.json`, `clinical_evaluation/`, and the three `clinical_seed<seed>/history.json` files. For example, the cloud suite is `/workspace/kevin_clinical_results`; use its verified local copy for reporting.

```sh
python \
  docs/kevin_paper_20261011/round2_clinical/clinical_report.py \
  --suite "$CLINICAL_SUITE" \
  --manifest data/external/round2_clinical_evaluation/patients.json \
  --output "$CLINICAL_SUITE/clinical_report"
```

The output directory must not exist. The report rejects incomplete or changed evaluator inventories, changed code/protocol/manifest pins, unverified model identities, incomplete patient/model/scenario grids, duplicate patient keys, metadata mismatches, changed fixed ensembles, and metric discrepancies. Each patient joins by the documented group key, with exact cohort, diagnosis, label and nine-slice membership metadata required; patients are never pooled across seed runs.

Outputs are 19 PNG/PDF figure pairs, `README.md` with every model's exact binomial intervals and primary paired-difference intervals, `analysis.json`, all 168 model/scenario/cohort metric rows, MCI prediction-only rows, and checksums. The figures include all three raw unsmoothed optimization/clean-training/validation histories, all eight models' confusion matrices, ROC/PR, probability distributions, reliability, exact intervals, and all seven fixed synthetic stress conditions for each clinical cohort. Visual review of every actual figure remains mandatory before publication; the completion marker deliberately records that requirement.

Fresh Harvard 19-patient AD/control results (9 AD, 10 CN) are kept separate from reused Besga 11 and OpenNeuro 15. Harvard 10 and OpenNeuro 5 MCI participants are prediction-only. The old three clinical heads and new three patient-bag models, plus their two uniform probability ensembles, are all retained. Old mean-slice-probability pooling and new mean-slice-logit pooling are explicitly distinguished. The fixed primary contrast is new ensemble minus old ensemble; individual seeds are visible and never promoted by test performance.

Exact 95% binomial intervals describe accuracy, sensitivity and specificity. Paired patient bootstrap intervals use the frozen statistics helper, 10,000 draws and diagnosis-stratified resampling. These are conditional on the observed small cohorts. OpenNeuro has only one AD patient: resampling that one individual cannot establish population sensitivity precision. McNemar diagnostics and the multiple model/metric comparisons remain exploratory. No clinical-readiness, clinical-superiority, radiologist-comparison or population-risk-calibration claim is supported.

Synthetic verification (no real outcomes or inference):

```sh
python -m unittest discover \
  -s docs/kevin_paper_20261011/round2_clinical -p test_clinical_report.py -v
python \
  docs/kevin_paper_20261011/round2_review/test_clinical_report_independent.py
MPLCONFIGDIR=/tmp/kevin-clinical-mpl \
python \
  docs/kevin_paper_20261011/round2_clinical/verify_clinical_report_synthetic.py \
  --output /tmp/kevin-clinical-report-synthetic-new
```

The synthetic renderer watermarks every plot. It covers 3,360 patient/model/scenario records, 168 recomputed metric cells, 12 paired 10,000-draw comparisons, 840 prediction-only MCI records and all 19 figure families. Synthetic success is software verification, not evidence about clinical model performance.
