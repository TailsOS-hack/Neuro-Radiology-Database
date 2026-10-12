# From a research paper to evidence for clinician MRI support

The current models are research artifacts. The previously untouched Harvard comparison failed clinically: the Round 2 clinical ensemble predicted AD for every one of 19 binary cases. Its 9/19 accuracy and zero specificity do not support superiority over the old ensemble's 10/19. See the [completed Round 2 results](../round2_publication/RESULTS.md). A successful new retrospective study would address part of this problem; it would not itself establish safe patient-care use.

## 1. Fix the intended use and reference standard

Round 3 targets clinical AD versus cognitively normal controls from supported structural T1 volumes. Define the intended clinician, patient population, clinical information available, output and decision the output might eventually support. A binary case-control study cannot establish discrimination among AD, vascular disease, Lewy body disease, frontotemporal dementia and other causes in an unselected memory clinic.

Document who assigned each diagnosis, the criteria used, the time from assessment to MRI, and available follow-up or biomarker confirmation. The reference must be independent of the model output. Keep a clinical AD phenotype distinct from biologically defined AD; structural MRI alone does not supply the Core 1 biomarker reference described in the [Alzheimer's Association diagnostic criteria](https://doi.org/10.1002/alz.13859). Missing biomarkers must be reported rather than silently treated as positive or negative.

## 2. Acquire eligible research data and preserve an untouched evaluation

Use the [provider-access checklist](access/ACCESS_CHECKLIST.md). An approved download must contain appropriate T1 volumes, patient/visit/scan keys, clinical labels, source/site and acquisition metadata. Count eligible people after these checks, not advertised images or multimodal participants. Keep repeat scans, derivatives and overlapping source-family participants together. A healthy-only collection or an unmapped 2D mirror cannot replace patient-labeled AD/control cohorts.

Reserve external institutions before model outcomes are examined. Freeze development partitions, reference/matching rules, preprocessing assets, model search, calibration/threshold choices, the comparator and the statistical analysis. Harvard, Besga, OpenNeuro and all other already evaluated sets retain historical status. Reusing their outcomes for development creates a need for a genuinely fresh evaluation cohort.

Sample-size planning must use explicit paired performance assumptions and class denominators at each site. Select target precision and a clinically meaningful effect with qualified clinical/statistical input; there is no universal number of scans that makes a model clinically ready. The current simulator assesses the project's full research gate under fixed-site assumptions, not clinical utility, reader-study power or arbitrary new-site variability.

## 3. Complete the retrospective paper

Compare the frozen binary clinical model on the same eligible patients with the old binary clinical ensemble and conventional baselines. Report every planned seed and fixed ensemble, calibration, operating-point sensitivity/specificity, confidence intervals, site/scanner/demographic strata, QC failures, abstention coverage and all-eligible results. Evaluate acquisition stress and repeated-scan stability. Reject unsupported modalities and record processing failures instead of silently dropping them.

Any abstention or confidence calibration mechanism must be selected and checked on development data before external use. A calibrated score is not automatically a validated individual clinical risk. Grad-CAM and other attribution plots are exploratory; prediction preservation, randomization and removal controls do not establish anatomical causality.

[CLAIM 2024](https://pubs.rsna.org/doi/10.1148/ryai.240300) supports transparent imaging/reference-standard, robustness and external-testing reporting. [TRIPOD+AI](https://www.bmj.com/content/385/bmj-2023-078378) supports prediction-model reporting, including outcomes, missing data, calibration and reproducibility. These are reporting recommendations, not performance certificates or regulatory approval.

## 4. Test whether assistance helps clinicians

With a clinical partner and appropriate review, design a multireader/multicase study comparing clinicians with and without the tool. Recommended features are randomized/counterbalanced reading order, a plan to reduce recall, identical available clinical information, an independent reference standard and analysis accounting for both patient and reader variability. Include relevant clinical alternatives and routine image-quality variation.

Prespecify diagnostic performance, clinically important error types, reading time and the effect of disagreement or abstention. The existing original100 four-class JPEG comparison cannot answer whether assistance improves real diagnostic decisions. A model outperforming a stand-alone reader in a constrained task would also not establish a beneficial human–AI team.

## 5. Evaluate actual workflow before live reliance

A recommended next stage is prospective silent evaluation on consecutive eligible patients at intended-use sites, with outputs withheld from clinical decisions. Check inclusion/exclusion, prevalence, scanner drift, missing data, turnaround time, failure handling and subgroup performance under actual workflow conditions.

Then, only with the appropriate clinical, ethics and regulatory arrangements, consider a limited live decision-support pilot with documented responsibilities, human oversight, escalation, safety monitoring and stopping rules. [DECIDE-AI](https://www.nature.com/articles/s41591-022-01772-9) addresses early live clinical evaluation and human factors; a silent or offline study does not by itself fulfill that live-use stage. Larger controlled evaluation may be needed for the eventual claim.

## 6. Establish and maintain an appropriate clinical system

The [FDA's GMLP page and final 2025 IMDRF principles](https://www.fda.gov/medical-devices/artificial-intelligence-enabled-medical-devices/good-machine-learning-practice-medical-device-development-guiding-principles) emphasize representative data, independent testing, suitable reference standards, human–AI performance, clear user information and lifecycle monitoring. Future clinical use also needs version control, security, traceable quality processes, drift monitoring and review of retraining or workflow changes. These principles do not determine the product's regulatory pathway by themselves.

Applicable provider agreements and ethics/regulatory obligations depend on jurisdiction, data and actual intended use. No clearance, exemption or institutional approval is asserted here. FDA lists the 2017 SaMD Clinical Evaluation guidance as withdrawn on January 6, 2026; its January 2025 AI lifecycle guidance remains draft. [FDA withdrawal record](https://www.fda.gov/medical-devices/guidance-documents-medical-devices-and-radiation-emitting-products/withdrawn-or-expired-guidance), [current draft status](https://www.fda.gov/regulatory-information/search-fda-guidance-documents/artificial-intelligence-enabled-device-software-functions-lifecycle-management-and-marketing)

Clinical readiness is therefore an unresolved objective with separate evidence requirements. The current authorized work is a reproducible research implementation and data-access preparation; additional epochs or a faster GPU cannot substitute for appropriate patients, reference standards and clinical evaluation.
