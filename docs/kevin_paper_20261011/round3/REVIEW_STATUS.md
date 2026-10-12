# Round 3 review status

Snapshot: 2026-10-11 America/New_York / 2026-10-12 UTC. This is a development-status record, not authorization to train or evidence of clinical performance.

## Current state

| Area | Verified status | Remaining work |
|---|---|---|
| Scientific design | Repository [protocol](protocol.json) records clinical AD/control scope, matching, partitions, bounded search and research progression criteria. | Complete actual data/runtime/selection freezes; repository registration is not external trial registration. |
| Provider access | [Access evidence](access/source_evidence.json) and exact NACC/BrainLat form fields are documented. MIRIAD's official endpoint timed out. | NACC submission and provider acknowledgment verified after explicit DUA/signature and certification confirmation. Delivered data entitlement and approval remain unverified; no Round 3 source acquisition is recorded. |
| Patient inventory | [Manifest](empty_patient_manifest.json) is empty; sample-size planning is [explicitly pending](sample_size_pending.json). | Approved T1/clinical data, scan/assessment linkage, patient/source overlap review, class/site counts and defensible planning assumptions. |
| Full-brain preprocessing | Source code requires pinned external N4, SynthStrip and ANTs tools/assets and returns explicit blocked/failure records. | Required executables not found on active PATH; reviewed weights, template/mask, runtime lock and anatomical QC remain pending. |
| Statistics/calibration/planning | Independent read-only review of the three modules and 16 synthetic test definitions found no concrete scientific blocker in their implemented estimands. | Root's executable integration, required membership/provenance checks, frozen replicate settings and complete end-to-end verification. This reviewer did not rerun those 16 tests or inspect real Round 3 outcomes. |
| CLI | Top-level/evaluation/prepare/predict-paired/report help, `status` and `search-manifest` verified in the existing research environment. Status reports zero patients and blocked data/preprocessing gates; search lists exactly 24 configurations without training. System Python lacks `nibabel`. | Report source renders genuine frozen statistics and raw curves when available. Complete synthetic acceptance testing and actual data/runtime freeze; read-only command success is not clinical training/inference verification. |
| Paid computation | No Round 3 paid resource or training run has been created. | B300 only after data authorization and live price check; Mac analysis; retrieve/verify then delete GPU and storage. |
| Clinical readiness | Not established. | Independent representative evaluation, clinical partner, appropriately reviewed reader/prospective studies and lifecycle controls. |

## Independent statistical review

Reviewed [statistics.py](../../../research/dementia_round3/statistics.py), [calibration.py](../../../research/dementia_round3/calibration.py), [sample_size.py](../../../research/dementia_round3/sample_size.py), and their synthetic test definitions. No source code was changed by this review.

- Four paired correctness cells preserve the old/new pairing for patient-level correctness resampling. Stratifying within fixed site and diagnosis and weighting every site/class equally matches the stated equal-site balanced-accuracy difference.
- Failures and abstentions remain incorrect in the primary all-eligible denominators; accepted-only results are explicitly secondary. A missing diagnosis at a site prevents the gate from passing.
- Exact binomial intervals for each site's class-specific correctness prevent tiny perfect strata from satisfying the precision floor. They are individual intervals, not a claim of simultaneous population certainty.
- The bootstrap is conditional on the observed fixed sites. Perfect or small strata can yield degenerate intervals; the implementation discloses this rather than claiming certainty about new populations.
- Calibration requires the reserved partition and clinical class order. Constant/separated scores and unidentifiable fits are unavailable; an unattainable all-eligible sensitivity target produces no threshold. Accepted predictions alone fit the logistic calibration, but threshold denominators retain failures/abstentions.
- Sample-size simulation requires explicit site/class denominators and paired correctness assumptions. It reports joint-gate power under those assumptions rather than claiming an automatically calculated minimum or clinical study power.

These module-level checks do not establish that supplied rows belong to approved partitions. CLI integration must verify manifest membership, source bytes, comparator identity and calibration/selection freezes. The final protocol specifies 10,000 bootstrap draws; lower module defaults and synthetic-test settings must not silently replace the frozen planning/evaluation setting.

## Interpretation and release boundaries

The research gate combines a point gain of at least five percentage points, a paired interval excluding zero, and per-site sensitivity/specificity and precision floors. It does not prove a true gain of at least five points, medical readiness, regulatory approval or superiority in an unrestricted dementia differential.

The unchanged Round 1 binary clinical ensemble is the primary clinical comparator; original100 is a reused four-class historical exercise. All already observed outcomes remain known. No model/threshold adjustment based on those outcomes can create a new independent test claim.

Public access documents were checked for applicant names, contact/account identifiers and personal paths; private drafts are ignored. No source has been approved simply because a provider webpage or application form is readable. Credentials, participant-level data and operational cloud details remain outside this public package.

## Final integration verification

The complete Round 3 suite passed **109 tests** (21.12 seconds), with Python compilation and synthetic confusion/interval figure inspection also passing. [implementation_verification.json](implementation_verification.json) records final source/test hashes. The three actual frozen old checkpoints had zero logit and nine-slice probability differences from the original helper on synthetic tensors; [legacy_checkpoint_parity.json](legacy_checkpoint_parity.json) records that adapter check. This is not a real-patient preprocessing or clinical-performance replay. No Round 3 patient training, prediction, anatomical QC or clinical validation has occurred.
