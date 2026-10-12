# Round 3 model pipeline review

Status: implementation foundation reviewed with synthetic tests only. No clinical performance or readiness is established.

## Scope and verification

Reviewed the patient models, preprocessing, training, inference, attribution functions, the development workflow integration, the strengthened selection freeze, and the aggregate report output. No real MRI fitting, patient prediction, cloud action, resource creation, or checkpoint replacement was performed in this review.

The latest scoped run passed **49 tests**: 42 owned model/preprocessing/training/inference cases plus seven development-workflow integration cases, in 4.41 seconds. This count does not describe all repository tests.

```sh
python -m pytest tests/test_round3_models.py tests/test_round3_preprocessing.py tests/test_round3_training.py tests/test_round3_inference.py tests/test_round3_workflow.py -q
```

Checks include CPU forward/backward and strict state reload for both neural families; every full-FOV slice in all three planes; patient-level rather than slice-level losses; balanced site/diagnosis sampling; immutable LOSO roster membership; epoch/search bounds; exact L2 coefficient penalty; finite geometry, intensity and mask checks; preserved preprocessing failures; missing-tool abstention; authorization before checkpoint or patient reads; frozen seed identities; calibration application; and old auxiliary-head computation against the unchanged frozen helper using synthetic weights.

Batch inference is bounded to its declared roster. It verifies full authorization once, loads models once, checks each row digest and raw/prepared source hashes, rejects repeats and omissions, and verifies artifact/code pins and unchanged model state on successful context exit. Outputs must be written after successful exit. Synthetic tests cover these invariants, including a changed raw input retained as a failed case.

Grad-CAM uses actual gradients and reports the forward parity result with a fixed tolerance of 1e-6. It retains native maps, full-volume maps, flat-map flags, parameter-randomization diagnostics and fixed region controls. These tests establish mechanics; they do not establish anatomical or causal disease localization.

## Scientific and operational limits

- The 24 neural configurations use only authorized development train/tune patients. LOSO holds out an entire normalized institutional site without changing the original manifest rows. The selected configuration is refit with all three fixed seeds; each epoch checkpoint is selected by equal-site tuning NLL. Calibration is separate and external outcomes cannot select configurations or epochs.
- The schedule is a 200-epoch ceiling, minimum 30 epochs, patience 25 and NLL improvement threshold 0.0001. Epoch zero remains eligible. Plateau stopping is implemented; a ceiling or safety stop remains blocked pending a new development-only review. No real convergence or loss curve has been observed for Round 3.
- Site/diagnosis sampling is with replacement. Each epoch records draw counts and unique patients; it does not falsely claim every training patient was sampled exactly once. Clean resubstitution and tuning metrics visit each patient once.
- Local N4BiasFieldCorrection, SynthStrip, antsRegistration and antsApplyTransforms and their pinned model/template/runtime assets are unavailable. The mocked external-command test checks sequencing, geometry and checksum persistence only. It does not test actual N4, extraction or registration accuracy. Real preprocessing remains blocked, and automated QC still requires diagnosis/prediction-blinded review.
- The anatomical baseline requires externally produced anatomical labels, intracranial masks and semantic/provenance pins. Its fractions are recomputed from those actual label maps; no intensity threshold is called anatomical segmentation. It requires a separate fixed-baseline protocol hash and cannot run from an arbitrary feature table.
- The old binary adapter preserves the original multitask auxiliary heads, exact checkpoint identities, historical nine-slice preparation, per-slice probability averaging in float64 and equal model averaging. Synthetic-weight parity against the frozen helper passed. Actual old-checkpoint replay parity and real MRI preprocessing parity were not run in Round 3; they remain required before claiming reproduced clinical comparison.
- Real B300 execution, throughput, memory use, CUDA determinism and clean-environment reproduction have not been tested. The trainer rejects CPU/Mac fitting. All local runs above use synthetic tensors. Paid execution remains separately gated by data, protocol, preprocessing and live price checks.
- New suitable patient cohorts, independently documented institutions, adequate class counts and untouched external cases are still required. Historical reader images and all previously evaluated cohorts are not fresh clinical validation. Slice quantity cannot substitute for patient or site independence.

The aggregate report contains metrics, counts, raw curves and hashes rather than patient rows or thumbnails. It explicitly denies clinical readiness and lists the remaining subgroup, stress, repeat-scan, attribution, confounding, sensitivity and reproduction work. Its CLI boundary must preserve the verified prediction/metric/history chain.

The strengthened freeze now checks the complete search/final-run and calibration provenance chain. Independent review additionally requested strict finite state loading at freeze and preprocessing dependency preflight before external authorization. Both checks were added by the workflow/freeze owner and confirmed in source; that owner separately reports passing freeze/workflow tests. Their later verification is distinct from the 49-test run reported above.

## Reviewed implementation identities

These hashes identify this review point, not a declaration that an executable clinical protocol has been frozen.

| File | SHA-256 |
| --- | --- |
| `research/dementia_round3/models.py` | `adcbf8ad3e4c4071f75a344d24015cd9472e8e8e54db6987cf19c201735fb3cc` |
| `research/dementia_round3/preprocessing.py` | `dd915a6844c4c119a7bab037fc3147bb4a4d7903eb2dc7cb83910757bafb401a` |
| `research/dementia_round3/training.py` | `04bff35c9c3c2b8c17bc16de0af553919b61984adf0dd0e736b46d36ac2a214b` |
| `research/dementia_round3/inference.py` | `65cdbd46b84a61e2fd66f84accfd85d19150cb5c00ffe4da6c0c5713460594a1` |
| `research/dementia_round3/explanations.py` | `ab464d74028761ba703b4ce6e7195d6d9668b8d52fbd45832a30ee05245f40d8` |

## Integration follow-up

The root integration review subsequently added explicit Grad-CAM support for callers already under `torch.no_grad()` or `torch.inference_mode()`, with two passing regression cases and unchanged caller gradient state. The hashes above preserve the earlier review point; the release verification manifest records final file identities.

The root subsequently verified the actual three legacy checkpoint weights against the frozen helper on deterministic nine-slice synthetic tensors: maximum logit and aggregated-probability difference 0.0 for every checkpoint. The earlier review limitation above remains for real MRI preprocessing/patient replay, not for this now-completed synthetic-input checkpoint check. See `../legacy_checkpoint_parity.json`.
