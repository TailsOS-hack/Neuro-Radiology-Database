# Reconstructing results for the unchanged research scripts

The public export is sufficient to read the reports, inspect all plotted data,
compare every original-100 prediction, and use the separately committed six
Git-LFS checkpoints. It intentionally excludes newly acquired clinical MRI,
participant rows and clinical subject-level prediction files. It is therefore
**not a byte-identical replacement for the full private frozen result tree**.
The aggregate JSON files under `results/` are labelled derivatives. Do not feed
them to a hash-guarded exact-analysis script and claim the original hashes match.

## Public legacy attribution reconstruction

Hydrate only the dementia model LFS objects and install the existing project
dependencies plus `../training/requirements-analysis.txt` as needed. The legacy
JPEGs are regular Git objects. From
the repository root, the following creates a new local result layout without
inference or copying new clinical data:

```sh
git lfs pull --include="models/alzheimers_classifier.pt,models/kevin_20261011/*.pt" --exclude=""
python3 docs/kevin_paper_20261011/publication/reconstruct_legacy_attribution.py \
  --output /tmp/kevin-public-attribution-inputs
python3 docs/kevin_paper_20261011/posthoc_explainability/run_attribution_audit.py \
  --results-root /tmp/kevin-public-attribution-inputs \
  --output /tmp/kevin-public-attribution-repeat
```

The helper verifies all six model hashes, copies the public original-100 prediction
CSVs, retains the validation-selected identity, and links the exact research
checkpoints into their original condition/seed directory names. The attribution
runner also uses the Git-tracked benchmark JPEGs, historical specialist/gallery
metadata, and baseline checkpoint from the repository. No model is selected from
new outcomes. CPU/CUDA probability discrepancies are reported rather than hidden;
see [the replay amendment](figures/attribution/replay_amendment.json).

## Exact full analysis and post-hoc stress/replay

The unchanged frozen analysis and figure scripts verify original file hashes,
including source clinical manifests and clinical prediction inputs. The unchanged
stress/clean replay scripts also pin the original clinical manifest even though
their inference uses only legacy images. Their protected guards were not weakened
for publication. Exact reruns consequently require the locally retained full
verified result tree and private clinical artifacts, or appropriately obtained
source releases and the same reconstructed inputs with matching hashes.

For the completed local study, `RESULTS_ROOT` below means
`training_logs/experiments/kevin_20261011/runpod_blackwell_20261011`. The legacy
prepared files must also exist at
`training_logs/experiments/kevin_20261011/prepared/`, and the original private
clinical manifest remains at its documented local path. Use a new output folder
for each command. These commands perform inference where stated and were not run
by the export process:

```sh
python3 docs/kevin_paper_20261011/posthoc_robustness/stress_test.py \
  --root . --results RESULTS_ROOT --output /tmp/kevin-stress-repeat --device cpu
python3 docs/kevin_paper_20261011/posthoc_robustness/clean_split_replay.py \
  --root . --results RESULTS_ROOT --output /tmp/kevin-split-repeat --device cpu
python3 docs/kevin_paper_20261011/paper_figures/recreate_quantitative_figures.py \
  --results RESULTS_ROOT --output /tmp/kevin-figures-repeat \
  --clean-replay /tmp/kevin-split-repeat
```

Replace the literal `RESULTS_ROOT` argument with the verified tree's actual path.
The first two commands run fixed-checkpoint inference; the third only plots and
checks saved evidence. Dataset access/licensing and clinical privacy boundaries
are documented in [Data and reproducibility](DATA_AND_REPRODUCIBILITY.md).

The public `figures/quantitative/plot_data.json`, aggregate result CSVs, raw per-seed
history CSVs, confusion matrices, attribution maps and stress/replay outputs remain
available for independent arithmetic and plotting inspection without new model
inference or access to clinical participant rows.
