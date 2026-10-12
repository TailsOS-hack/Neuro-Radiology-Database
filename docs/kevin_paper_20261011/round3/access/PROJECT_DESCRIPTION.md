# Proposed research description for provider review

Status: public scientific summary. A NACC-specific version was submitted and acknowledged; BrainLat and MIRIAD requests remain drafts. This text describes a proposed research study; it does not assert access, ethics approval, a clinical partner or clinical readiness. Applicant identity belongs in the separate private form draft.

**Proposed title:** Patient-level and source-separated evaluation of structural MRI models for Alzheimer’s disease classification

The proposed retrospective study will evaluate structural T1-weighted MRI models as a potential future source of support for clinicians assessing cognitive impairment. The immediate objective is a reproducible research paper testing whether patient-level modeling and source-separated development improve generalization over existing image-based classifiers and conventional imaging baselines. No output will guide patient diagnosis or treatment. The initial AD/control research endpoint will be distinguished from a broader dementia differential diagnosis, and MCI and alternative etiologies will have explicit roles.

The requested data are released de-identified or defaced T1 volumes with participant/visit identifiers, diagnosis definitions, scan and assessment timing, source/site information, acquisition metadata and QC. Available demographic variables will support prespecified characterization and subgroup analysis. The initial NACC request omits CSF and PET numerical biomarkers and uses clinical diagnosis as its reference. Eligible numbers will be established from the approved release rather than advertised total cohort sizes. A prespecified temporal linkage rule will connect each scan to an appropriate clinical assessment; clinical diagnosis must not be relabeled as biomarker-confirmed AD.

All repeated scans from a participant will remain in one partition. Duplicate data, overlapping dataset versions and participant/source overlap will be checked before analysis. Development data alone will determine preprocessing, model selection, calibration and operating thresholds. Reserved source/site testing will report patient-level performance, paired comparisons, calibration, uncertainty, subgroups and failure cases against prespecified criteria. Participant-level data will not be placed in a public repository. Collaborator access, external compute, derived-data sharing and checkpoint release will follow the specific approved agreement. Any required ethics determination must be documented before the activity that depends on it.

## NACC-specific requested content

Request actual eligible T1 image volumes from mixed-protocol MRI and available defaced SCAN releases, not only tabular imaging measurements. Include UDS cognitive status **and** etiologic diagnosis, participant/visit identifiers, assessment dates, and scan linkage. Optional released regional/intracranial volumes provide conventional baselines. The [NACC guide](https://www.naccdata.org/the-nacc-researchers-guide) states that the investigator must match imaging dates to UDS visits.

The [SCAN MRI dictionary](https://files.alz.washington.edu/documentation/SCAN-MRI-Imaging-RDD.pdf) documents requested linkage/acquisition categories: `NACCID`, `NACCADC`, `STUDYINSTANCEUID`, `STUDYVISITCODE`, `STUDYDATE`/`SCANDT`, T1 image references, scanner manufacturer/model/field strength, and study/series quality and release indicators. Availability in the actual approved extract must be checked; the dictionary is not an entitlement or complete-case count. Its current revision history includes 2026 changes despite an April 2024 cover label.

## Other-provider adaptation

BrainLat's observed one-page MRI form has no aims field. Keep this description ready for a later legitimate provider request; do not send an unsolicited message. Ask for the current T1 release with matching participant codes, diagnoses, acquisition/site metadata and any clinical-assessment timing. MIRIAD's registration fields are currently unknown; reserve a research-purpose paragraph only if requested after the official endpoint becomes available.
