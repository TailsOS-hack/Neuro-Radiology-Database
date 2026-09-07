# Reproduce the validation follow-up

The frozen protocol is `docs/validation_protocol_v2.json`. The historical mentor bundle under `docs/review_bundle/` is immutable, and its source snapshots and inventory are retained in `docs/validation_baseline_sources/`. The follow-up is research only; it does not modify the application, accepted checkpoints, or historical performance tables.

## One complete build

Use Python 3.13 and the evaluation dependencies recorded in the original review lock file, plus `reportlab==4.4.9`. Recover all four accepted Git LFS weights into a local checkpoint directory. If using the project's `models/` directory, first run `git lfs install --local` and `git lfs pull --include='models/*.pt'`. Do not substitute sensitivity or freshly trained models.

```bash
python scripts/build_validation_review.py \
  --checkpoint-dir models --device cpu --full \
  --output docs/review_bundle_v2/validation-v2-20260907
```

Use `--device mps` on a compatible Apple GPU or `--device cuda` on a compatible GPU runner. CPU is the portable default. The builder runs frozen explainability and calibration protocols, offline tests, the synthetic external-readiness check, generated result tables and memo, integrity verification, and archive checksum verification. It publishes from a staging directory only after all checks pass. Existing versions are never replaced; remove no historical version to make a run fit. Use a new output directory for a repeat and update the protocol run identifier only when creating a genuinely new protocol version.

For independently executed repeat runs, `scripts/compare_validation_repeats.py` compares arrays and numeric CSV fields with absolute tolerance 0.00001 and relative tolerance 0.0001. Categorical values, sample selection, checkpoint identities and protocol hashes must match. Image/PDF metadata is not a numerical reproducibility criterion. Pass completed repeat component paths with `--repeat-calibration` and `--repeat-explainability` when assembling the review package. `--calibration-source` and `--explainability-source` can reuse sealed completed components; their source hashes must match the implementation being packaged.

## Lightweight and full verification

```bash
python -m pip install -r requirements-verification.txt
python -m unittest discover -s tests -p 'test_validation*.py' -v
python -O scripts/check_explainability_bundle.py
python -O scripts/check_validation_review.py docs/review_bundle_v2/validation-v2-20260907
python scripts/check_publication_package.py --no-write
```

The lightweight path needs committed artifacts and frozen source snapshots, not MRI images, Torch, GUI libraries, model binaries or network at runtime. The separate pull-request workflow uses sparse checkout to avoid raw images and disables LFS weight downloads. The existing training workflow remains unchanged.

Full verification adds `--full --checkpoint-dir models` to the follow-up checker. It also verifies checkpoint contents and reads all validation/test image inputs; historical sampled-image hashes are checked. Review raw figure panels and the rendered mentor PDF before sharing. No tool sends the packet automatically.

## Method and scope notes

The 256-image sample and diagnostic gallery selection are inherited from the baseline. This is a follow-up on an already inspected test set. The validation split also influenced checkpoint selection. D4 mask reorientations preserve shape and area but may retain central support; Gaussian blur and mean replacement can both create distribution shifts. Flat/failed randomized maps are counted outside conditional correlation averages. No significance test or clinical threshold is attached to these diagnostics.

Temperature fitting is restricted to validation rows. Four scalar fits minimize validation NLL within [0.05, 20], with temperature 1 retained if optimization fails or does not improve NLL. Fitting artifacts are saved before test inference. Hierarchical hard routing and joint-probability argmax remain separate. Lower internal ECE does not establish calibrated clinical probabilities.

The external interface and acquisition checklist are in `docs/EXTERNAL_VALIDATION_READINESS.md`. No independent cohort exists in this project; external validation remains not performed. No automatic retraining is included.
