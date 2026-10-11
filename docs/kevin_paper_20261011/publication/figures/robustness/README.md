# Post-hoc fixed-checkpoint robustness audit

This folder is separate from the frozen primary experiment. The primary results
were already known when this additional descriptive analysis was requested. No
model selection, retraining, calibration, threshold fitting, or replacement of
the primary predictions is performed.

`stress_test.py` applies the same fixed corruptions to the old model and all six
validation-selected candidates on original100 and primary1058. It imports the
unchanged, checksum-pinned network loader and forward path from the original
experiment. Clean preprocessing is checked against the original transform.
Predictions are FP32; the recorded local device can produce small numerical
differences from the original CUDA environment. Preset tolerances compare each
clean probability to its archived value, and every discrepancy is retained.

Run from the repository root with the project ML environment:

```sh
MPLCONFIGDIR=/private/tmp/kevin-posthoc-mpl \
  [local path omitted] \
  docs/kevin_paper_20261011/posthoc_robustness/stress_test.py \
  --root . \
  --results training_logs/experiments/kevin_20261011/runpod_blackwell_20261011 \
  --output docs/kevin_paper_20261011/posthoc_robustness/results_20261011 \
  --device mps --batch-size 32
```

The output must be a new directory. `protocol_before_inference.json` records the
scenario grid and all protected identities before stress inference. The output
includes per-case probabilities, confusion matrices and scenario metrics,
clean-replay discrepancies, plots, findings, and independently read-back file
checksums. Images/manifests/checkpoints are verified again after inference.

The synthetic checks in `test_stress_test.py` verify transform determinism,
unmodified clean pixels, shape/range contracts, probability rejection, and metric
alignment. They provide software evidence, not clinical validation.

Intensity and contrast factors are 0.90 and 1.10; additive display-noise SDs are
0.01 and 0.02 on the [0,1] scale; Gaussian blur radii are 0.5 and 1.0 pixels;
intermediate resolutions are 192 and 160 pixels; JPEG quality is 90 and 70.
Noise is spatially shared across replicated RGB channels. These perturbations
test a 2D image interface, not the physics of MRI acquisition. Patient, scanner,
site, disease, demographic, and prospective workflow shifts are not represented.
The historically reused reader benchmark and missing legacy patient identities
remain limitations regardless of the stress-test performance.

This fixed diagnostic grid must not be described as validation against MRI-domain
shift. It includes one deterministic noise field per case at two amplitudes, not
an uncertainty analysis across noise realizations, and does not test composed
corruptions. `verify_transfer_inputs.py` separately confirms that all 27,278
legacy/original100 source files match the independently pinned GPU transfer
bundle. That identifies unchanged source bytes without assigning a cause to
cross-environment probability differences.

## Mathematical interpretation of the training-loss plateau

The frozen trainer optimizes cross-entropy with label smoothing 0.05 but logs
ordinary, unsmoothed negative log-likelihood. For a single correctly labeled
example with K classes, the smoothed target puts probability
`1 - 0.05 + 0.05/K` on the labeled class. At that target distribution, the logged
unsmoothed loss is `-log(1 - 0.05 + 0.05/K)`: approximately 0.03822 for four
classes and 0.02532 for the binary auxiliary head. These reference values help
interpret observed training plateaus around 0.039 and 0.026.

This is a mathematical reference, not a hard lower bound, a guarantee of the
optimizer's solution, or evidence of generalization. The full optimization also
includes shared parameters, weighting, regularization, and augmentation. Pushing
the logged unsmoothed loss toward zero is not identical to improving the actual
smoothed objective; clinical validation remains the separate empirical concern.
The [PyTorch 2.8 CrossEntropyLoss documentation](https://docs.pytorch.org/docs/2.8/generated/torch.nn.CrossEntropyLoss.html)
describes label smoothing as a mixture of the original target and a uniform
distribution. No completed result artifact was altered to add this note.

## Additional clean split replay

`clean_split_replay.py` exports all six fixed candidates on the new training
split (19,023 image rows) and full new test split (5,450 image rows). It uses the
unchanged evaluator in evaluation mode and creates separate new files. Training
figures are final-checkpoint resubstitution, not the online epoch metrics. The
full test includes primary1058 and is an overlapping post-hoc supplement, not
an additional independent cohort or a replacement for the primary analysis.

```sh
MPLCONFIGDIR=/private/tmp/kevin-posthoc-mpl \
  [local path omitted] \
  docs/kevin_paper_20261011/posthoc_robustness/clean_split_replay.py \
  --root . \
  --results training_logs/experiments/kevin_20261011/runpod_blackwell_20261011 \
  --output docs/kevin_paper_20261011/posthoc_robustness/clean_split_replay_20261011 \
  --device mps --batch-size 32 --workers 4
```

The replay verifies every source image against the frozen decoded-pixel hashes,
retains all per-case probabilities, records checkpoint identities and per-split
metrics, and independently reads back output checksums. It does not select a new
model or alter the primary evaluation. `replay_metadata.json` links the CSVs to
their exact models and metric summaries.


Public navigation: [stress findings](FINDINGS.md), [scenario metrics](scenario_metrics.csv), [reconstruction guide](../../RECONSTRUCTION.md).
