<!-- Portable documentation copy; frozen original retained unchanged. -->
# Round2 fixed evaluation and reporting

This package evaluates all three completed round2 stage-only fine-tunes against the original model and the fixed round1 c3 comparator. All four benchmark cohorts have already been seen during this project: these are post-hoc development comparisons, not fresh external or clinical tests. No clinical auxiliary head or newly acquired clinical dataset is evaluated.

The order is fixed: old_baseline, round1_c3, round2_seed20261011, round2_seed20261012, round2_seed20261013. The representative is the smallest eligible selected validation score, breaking ties by ascending seed. The script replays every epoch's clean guards and min_delta selection rule. Epoch0 fallback remains allowed and explicit. It never chooses a model from test or attribution results.

## Phase1: code-only upload and validation freeze

Upload every path in `runtime_code_manifest.json` before freezing. The manifest includes shared helpers, the training protocol's source files, and the two Grad-CAM modules. This transfer contains no held-out pixels. Training must finish all three research runs first.

Cloud repository root: `/tmp/Neuro-Radiology-Database`. Cloud Python: `/tmp/kevin-round2-venv/bin/python`. Run from that repository root; all inference outputs stay under the persistent `/workspace/kevin_round2_results` suite.

```sh
/tmp/kevin-round2-venv/bin/python docs/kevin_paper_20261011/round2_evaluation/round2_pipeline.py freeze \
  --data-root . \
  --suite /workspace/kevin_round2_results \
  --round1-c3 models/kevin_20261011/stage_only_seed20261012.pt
```

Freeze does not read held-out metadata or pixels. It records their known required hashes; the original checkpoint may arrive later with the held-out add-on. The initialization c3 checkpoint must already exist. Freeze exclusively creates `round2_evaluation_freeze.json`, pins all three selected checkpoints, suite/selection metadata, protocol, source code, input metadata and representative identity, and refuses if evaluation has started.

If any run reaches the epoch ceiling without the required plateau, freeze refuses until an explicit validation-only review is provided via `--convergence-review path.json`. It must contain:

```json
{
  "status": "approved_after_validation_only_review",
  "suite_status_sha256": "exact hash of suite_status.json",
  "training_protocol_sha256": "exact hash of suite protocol.json",
  "heldout_outcomes_used": false,
  "reviewed_seeds": [20261011, 20261012, 20261013],
  "rationale": "Concrete review of raw validation histories and convergence."
}
```

A completed maximum-epoch run is not automatically evidence of convergence. No generated approval is supplied by this package.

## Phase2: guarded held-out add-on and one cloud evaluation

After freezing, use `round2_operations/merge_evaluation_addon.py` to verify and merge the separate add-on. It must preserve all training-bundle files and allow preexisting files only when byte-identical. Include every metadata path in `round2_pipeline.INPUT_PINS`, the fixed original checkpoint, all5,450 prepared test images, and the100 original benchmark images. The fixed160 validation images are already in the training bundle.

```sh
/tmp/kevin-round2-venv/bin/python docs/kevin_paper_20261011/round2_evaluation/round2_pipeline.py evaluate \
  --data-root . \
  --suite /workspace/kevin_round2_results \
  --device cuda --batch-size 64
```

This writes `round2_evaluation_launch.json`, then `frozen_round2_evaluation/`. It never automatically reruns an existing launch. The final `evaluation_complete.json` marker requires all155 payloads and checksum readback. An interrupted run remains incomplete and must be preserved for review.

All five models use the same full-precision clean transform. The1,058 same-case and756 sensitivity predictions are exact subsets of the5,450-image pass. All13 unchanged perturbation scenarios are reported for1,058 and100. Fixed160 validation images receive all13 corner/border/center controls. No weight fitting, calibration, threshold changes, or training occur here.

The primary1,058 historical cohort has302 rows with old development hash-group mates. The756 subset removes those observed mates, but hash groups are not patients. Original100 remains the reused image reader benchmark with old66train/9validation/25test exposure. These limits remain in report output.

## Phase3: retrieve, verify, and report locally

Retrieve the entire completed suite with all checkpoint/history/selection files and verify archive plus destination readback before deleting cloud storage. Keep the relative source paths in the local repository. On this Mac, use:

```sh
python \
  docs/kevin_paper_20261011/round2_evaluation/report_round2.py \
  --data-root . --suite <verified-local-suite> \
  --output <new-local-report-directory>

python \
  docs/kevin_paper_20261011/round2_evaluation/round2_attribution.py \
  --data-root . --suite <verified-local-suite> \
  --output <new-local-attribution-directory> --device mps
```

Reporting independently recomputes all215 clean/stress/corner metric records from exact aligned probability rows. It exports28 PNG/PDF pairs:20 cohort confusion/calibration/confidence/ROC/PR figures, comparative accuracy, seed spread, raw training curves, four stress figures, and fixed validation masking controls. `analysis.json` includes all3 seed values, mean/sampleSD, macroF1, balanced accuracy, and paired differences versus both comparators with2,000 deterministic conditional bootstrap draws. Image cohorts use prepared hash groups; original100 uses individual image units. Fixed readers are included on all100 cases; their range is descriptive, not an uncertainty interval. Degenerate intervals at the ceiling do not establish certainty or clinical superiority.

Attribution uses all115 previously fixed cases. Old and c3 arrays are verified and reused unchanged. The new validation-chosen representative produces native7×7 maps and labeled display overlays for its own unmodified predicted class. It checks logits, modes, parameter gradients and checkpoint state; repeats three-stage parameter randomization with three fixed seeds on the same eight designated examples; and renders a native-location audit. Flat/failed maps are counted, not concealed. Cloud/local original100 probability or argmax mismatches cause a blocked replay-review marker. No map is shifted, cropped, or relabeled to make it appear centrally focused.

These scripts deliberately mark real figures as needing visual review after rendering. Inspect all28 plots, both attribution summaries, and all115 case panels before publication. Read the actual robustness/seed comparisons; neither training longer nor a visually favorable heatmap guarantees improvement.

## Synthetic verification

```sh
MPLCONFIGDIR=/tmp/kevin-round2-mpl \
python \
  docs/kevin_paper_20261011/round2_evaluation/test_round2_pipeline.py

MPLCONFIGDIR=/tmp/kevin-round2-mpl \
python \
  docs/kevin_paper_20261011/round2_evaluation/verify_synthetic_pipeline.py \
  --output <new-temporary-synthetic-directory>
```

Seven unit tests and the full synthetic evaluator/report pipeline pass. Synthetic tests use generated noise images and tiny fake networks; no real held-out pixels, outcomes, or models are opened. Synthetic figures are watermarked. The saved verification record lists numerical checks and PDF/page-layout inspection. Do not edit runtime files after the actual validation freeze; source hashes are enforced at each later phase.

After real figures and panels are visually reviewed, run `assemble_atlas.py --report <report> --attribution <attribution> --output <new-atlas-directory>` locally. It creates a bookmarked30-page atlas (28 quantitative/robustness and2 attribution graphs), a checksum/page index, and a link to all115 separate case panels. Keep the historical graph atlas separately linked.
