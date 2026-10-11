# Kevin dementia research checkpoints

These six checkpoints preserve all three seeds in each fixed training condition.
They were selected by stage-validation NLL before the original held-out evaluation.
See [CHECKPOINTS.json](CHECKPOINTS.json) for exact hashes, seeds and selected epochs.
The production `models/alzheimers_classifier.pt` has not been replaced.

Files are stored with Git LFS. After cloning, fetch this study and its baseline:

```sh
git lfs pull --include="models/alzheimers_classifier.pt,models/kevin_20261011/*.pt" --exclude=""
```

Load only trusted checkpoint files: these PyTorch archives contain training metadata in addition to
weights. The frozen architecture/transform code is
[`train_candidate.py`](../../docs/kevin_paper_20261011/training/train_candidate.py).
All models have a four-class stage head; multitask checkpoints additionally have
an `aux_state` for the separate binary AD/control endpoint.

These are research artifacts. The original100 benchmark is reused, legacy patient
identity is unknown, and the clinical pilot showed poor and unstable performance.
No overall superiority or clinical-readiness claim is supported. The stage-only
20261012 seed was the validation-selected exploratory representative; this is not
a winner chosen from test scores. Retain and report every seed.
