# Additional unrestricted clinical T1 cohort search

Checked 12 October 2026 UTC. A targeted review found **no additional cohort that could be verified to meet every condition**: original clinical whole-brain T1 MRI, AD/control patient labels and stable identities, lawful access without login/forms, and a new source beyond the cohorts already acquired. This is a bounded search result, not proof that no such dataset exists. No images were acquired, no model inference ran, and the current frozen training/evaluation inputs were unchanged.

The strongest new scientific lead is [ANMerge/AddNeuroMed](https://www.synapse.org/Synapse:syn22252881). Its [original descriptor](https://pmc.ncbi.nlm.nih.gov/articles/PMC7902946/) documents raw 1.5T T1 MRI, AD/MCI/control diagnoses, and unified participant mappings. However, it requires Synapse registration, a signed data-use certificate, a research proposal and ethical-review confirmation. Current anonymous file permissions could not be verified; the public project page is not evidence of permission to download. The 1702-person multimodal total must not be reported as 1702 usable MRI patients.

New freely visible leads failed the endpoint requirement:

| Source | Provider evidence | Why excluded |
|---|---|---|
| [QNLD/OpenNeuro ds006148](https://www.nature.com/articles/s41597-026-07081-x) | Public structural T1/BIDS data; [CC0 provider metadata](https://raw.githubusercontent.com/OpenNeuroDatasets/ds006148/master/dataset_description.json). | Normal cognition was required and MCI/dementia were screened out. Amyloid positivity in cognitively unimpaired participants is not an AD diagnosis. Individual demographics also require an institution-signed DUA. |
| [PEARL-Neuro ds004796](https://openneuro.org/datasets/ds004796/versions/1.0.9) | CC0, 79-person neuroimaging subgroup, public participant metadata. | Healthy middle-aged participants at genetic risk, rather than diagnosed AD/control groups. |
| [OpenNeuro ds004873](https://raw.githubusercontent.com/OpenNeuroDatasets/ds004873/master/README) | Original README describes subject membership and the study. | Main study participants are healthy; no AD group. |
| [Gonzo/Zenodo14266867](https://zenodo.org/records/14266867) | Original provider release includes T1 MRI before/after intrathecal tracer. | One healthy-appearing volunteer with repeated acquisitions; not an AD/control cohort. |
| [Paired low-/high-field MRI/Zenodo19709849](https://zenodo.org/records/19709849) | Leiden provider release includes T1 among healthy-subject acquisitions. | No diagnosed AD group. Scan counts are not independent patient counts. |

Prior access checks for MIRIAD, BrainLat, NeuGRID, BioFIND and OASIS were reused rather than repeating approval routes or treating mirrors as a way around access requirements. AlzaSet remains quarantined because the released JPEGs lack verified patient mapping. The already acquired Besga, OpenNeuro clinical release and Harvard Dataverse 29352 were excluded from the definition of an additional source.

Healthy-only data should not simply be assigned to the control class against AD from a different hospital: that would confound diagnosis with source. Such releases may later support a separately defined normal-aging, acquisition-shift or pretraining experiment after full access/identity review, but they do not meet this task's AD/control training-data requirement. Further parameter comparisons should use a newly frozen development protocol with patient/source partitions and validation-only selection, retaining a genuinely untouched final source. The present search does not justify tuning against the completed test outcomes.

`additional_open_t1_search.json` records the decisions, primary links, unresolved leads and action limits without patient-level records or private correspondence.
