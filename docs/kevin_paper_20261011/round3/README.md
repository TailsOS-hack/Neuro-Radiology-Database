# Round 3: patient-level MRI research

This round is preparation for a stronger research paper on **clinical AD versus cognitively normal control classification from structural T1 MRI**. It is not a deployment authorization. No new source has been approved or acquired for Round 3, no Round 3 training has started, and no paid resources have been provisioned for this round at this documentation snapshot.

The [protocol](protocol.json) records the agreed design in this repository. It is not an external trial registration or a completed data/runtime freeze. The [source inventory](inventory.json), [empty patient manifest](empty_patient_manifest.json) and [pending sample-size record](sample_size_pending.json) deliberately preserve the unmet data gate. Current evidence and implementation boundaries are in [REVIEW_STATUS.md](REVIEW_STATUS.md).

## Access and study scope

The [access package](access/README.md) contains official NACC/SCAN, BrainLat and MIRIAD routes, live form-field observations, a public project description and the access checklist. The NACC request was submitted and acknowledged; actual delivery and MRI entitlement remain unverified. Applicant identity, request identifiers and receipt remain private. Application preparation or submission permission does not equal provider approval, image entitlement or ethics approval.

The initial endpoint uses clinical diagnosis with a prespecified scan/assessment match: 180 days for the primary analysis and 90 days for sensitivity analysis. Cognitive status, etiologic diagnosis and biomarker-defined AD remain distinct. MCI and alternative dementia etiologies are outside the primary binary endpoint; the project cannot claim general dementia differential diagnosis from AD/control performance. [NACC's clinical and imaging linkage guidance](https://www.naccdata.org/the-nacc-researchers-guide)

The intended data design requires at least two development institutions and two untouched external institutions, with every visit and derivative belonging to one global patient partition. Different dataset names do not establish independent people or sites. NACC and SCAN are linked sources; repeated MIRIAD scans are repeated people. All previously evaluated cohorts, including Harvard and the original 100-image set, are historical benchmarks.

## Executable workflow

The research implementation lives in [research/dementia_round3](../../../research/dementia_round3/). Its command-line entry point exists and the top-level and evaluation help have been executed successfully. Run these commands from the repository root in a Python environment with the research dependencies available. Use actual help output for required arguments; do not infer filenames or options from this document:

```bash
python -m research.dementia_round3 --help
python -m research.dementia_round3 status
python -m research.dementia_round3 search-manifest
python -m research.dementia_round3 evaluate --help
```

The core subcommands are `status`, `match`, `plan-sample-size`, `freeze-development`, `calibrate`, `freeze-external`, `evaluate` and `audit-previous`. The executable workflow also registers `search-manifest`, `prepare`, `run-search`, `select-candidate`, `bind-final-selection`, `run-final-seeds`, `predict-new`, `predict-paired` and `report`. The top-level help and the `prepare`, `predict-paired` and `report` help were executed successfully. Inspect a specific command with `python -m research.dementia_round3 COMMAND --help`; there are no generic `train` or `predict` commands. Help registration is not end-to-end verification with clinical data.

The existing research Python environment successfully ran `status` and `search-manifest`: status reported zero eligible patients, a blocked data gate and missing/unpinned preprocessing assets; the search command returned exactly 24 configurations with `training_performed=false`. The system `python3` could display help but returned a missing-`nibabel` dependency block for those operational commands. Activate a suitable research environment rather than assuming the system interpreter has its dependencies. No installation, training, model inference or resource provisioning was performed by these checks; the separately pinned preprocessing runtime remains required.

The local `report` command takes `--authorization`, `--predictions` and a new `--output` directory. It verifies frozen external inputs and statistical replay, then renders raw unsmoothed final-seed loss curves, site confusion matrices with explicit no-output columns, site-specific intervals and accepted-only calibration with explicit coverage/failure boundaries. Actual reports await approved data and a completed frozen workflow; synthetic fixtures are software checks, not clinical outcomes. The [execution contract](EXECUTION_CONTRACT.md) records implementation and remaining study gates.

The required order is source approval and metadata matching, audited patient/site inventory, explicit sample-size planning, development authorization, verified preprocessing and development-only model search, calibration and selection freeze, then external evaluation. `audit-previous` examines existing saved evidence and does not create fresh clinical validation. A `status` check or a manually edited status flag cannot authorize fitting.

The planned search contains 24 configurations across full-brain multiplanar and compact 3D models, with leave-one-development-institution-out selection and three fixed final seeds. The protocol specifies a maximum of 200 epochs, minimum 30, patience 25 and minimum NLL improvement of 0.0001. Longer training requires a development-only review and protocol amendment before external evaluation; more epochs are not an independent measure of clinical quality.

## What counts as improvement

The primary old-model comparator is the unchanged Round 1 **binary clinical ensemble**. The four-class original100 result and reader exercise measure a different task and do not establish a clinical AD comparison. Secondary comparators include the old clinical model recalibrated using development data, the Round 2 clinical ensemble and appropriate conventional/confounding baselines.

The protocol's research progression gate requires all of the following: equal-site mean balanced-accuracy gain of at least 5 percentage points; a paired 95% interval with lower bound above zero; and, at every external site, sensitivity and specificity estimates of at least 0.85 with their separate exact 95% lower bounds at least 0.80. Failures and abstentions count as incorrect in the all-eligible primary denominators. These are project research targets, not universal medical standards. The point-gain requirement plus a lower bound above zero does **not** prove that the true gain is at least five points.

Calibration uses reserved development patients, with a 0.90 development sensitivity target for threshold selection. That target is not guaranteed on external data. Unidentifiable or infeasible fits remain unavailable. The paired bootstrap conditions on the observed sites; it does not estimate performance at a random new hospital. Full clinical follow-up requirements are in [CLINICAL_ROADMAP.md](CLINICAL_ROADMAP.md).

## Runtime and spending boundary

The planned whole-brain preprocessing requires N4 bias correction, reviewed SynthStrip brain extraction, a pinned template and mask, ANTs registration/transforms, QC and provenance. The four required executables were not available on the active PATH during this review, and no complete reviewed runtime/asset freeze has been established. Synthetic tests and geometric checks do not validate anatomical accuracy. Missing tools, weights or template assets must block processing.

The user selected NVIDIA B300 training at a live-verified rate no higher than $8.99/hour, within the $10/hour machine ceiling, with analysis on the local Mac. This is a future execution policy, not a current reservation or guaranteed price. No paid resource is permitted before the data gate; after a future authorized run, results must be retrieved and verified before both GPU and storage are deleted. The production classifier remains unchanged.
