# Export verified Round 2 research artifacts

`export_artifacts.py` is an inert packaging tool. It does not train, infer, select a checkpoint, regenerate plots, or alter original result files. Run a full export only after both real suites have been retrieved, their numerical outputs verified, and every exported visual inspected. An explicit stage-only release is also supported. Its synthetic tests exercise packaging guards and privacy exclusions; they do not establish medical robustness or visual quality of real figures.

From the repository root, supply the two verified local suite directories and a **new** `artifacts*` child of this directory:

```sh
python docs/kevin_paper_20261011/round2_publication/export_artifacts.py \
  --stage-suite "$STAGE_SUITE" \
  --clinical-suite "$CLINICAL_SUITE" \
  --output docs/kevin_paper_20261011/round2_publication/artifacts_v1
```

The report folders must be direct suite children with `report_complete.json`. In ordinary full mode, the stage attribution folder must likewise contain a passing `attribution_complete.json`. A preserved failed attribution folder without a completion marker is ignored. If there are multiple completed attribution folders, choose the reviewed one explicitly with `--attribution-directory "$STAGE_SUITE/round2_attribution_recovered"`. Recovery receipts are retained; failure logs are excluded. A blocked probability-replay result is never treated as a replay pass.

## Explicit partial and diagnostic releases

The independently completed 28-figure quantitative stage report can be released without waiting for attribution or clinical results:

```sh
python docs/kevin_paper_20261011/round2_publication/export_artifacts.py \
  --stage-suite "$STAGE_SUITE" --stage-quantitative-only \
  --output docs/kevin_paper_20261011/round2_publication/artifacts_stage_quantitative
```

This mode does not read attribution or clinical folders. It includes the stage report, all three raw histories/curve pairs and the original-100 tables. Its README and export manifest explicitly identify a partial release. It cannot be combined with clinical or attribution arguments and does not imply that an omitted check passed. Its numerical and visual review requirements remain intact.

When the attribution probability-replay guard has failed, a **separately prepared and visually reviewed diagnostic appendix** can be included by explicit request:

```sh
python docs/kevin_paper_20261011/round2_publication/export_artifacts.py \
  --stage-suite "$STAGE_SUITE" --clinical-suite "$CLINICAL_SUITE" \
  --include-blocked-attribution-diagnostics \
  --attribution-directory "$STAGE_SUITE/round2_attribution_diagnostics" \
  --output docs/kevin_paper_20261011/round2_publication/artifacts_with_diagnostics
```

This mode accepts the separately sealed `diagnostic_appendix_complete.json` with status `complete_diagnostic_appendix_with_blocked_replay`, its own payload inventory, and warning-bearing figure/gallery derivatives. It preserves the unchanged `source_attribution_complete.json` blocked marker and `source_SHA256SUMS.json`, verification, recovery receipt and derivative provenance. The frozen probability tolerance is not widened. Export is restricted to the reviewed condition in which probability differences exceed the tolerance while predicted classes remain unchanged. Failures and their counts remain visible in the package README, visual-review summary and export manifest. The public directory is named `stage/attribution_diagnostics`, not an ordinary successful attribution result.

Every case panel and both PNG/PDF summary figures must visibly carry the diagnostic warning, and the exact final derivative hashes must be in the visual-review receipt. If a diagnostic atlas is present, its two attribution pages must use those warning-bearing summaries. The optional diagnostic atlas has its own `diagnostic_atlas_verification.json` and `Round2_Diagnostic_Graphs_Atlas.pdf`; the ordinary successful-attribution atlas is not substituted. Visual inspection confirms layout/readability and never clears the failed scientific replay check. No inference or repeat selection occurs during this packaging step.

## Required numerical and visual evidence

The exporter checks the evaluation inventories, embedded frozen selections, all three histories for each task, report inventories and attribution inventory. It verifies all listed payload digests, even private numerical inputs that will not be copied. Clinical evaluation must cover its exact four payload files and the declared 3,360 prediction records / 168 aggregate metrics. The existing reporting pipeline remains responsible for independently recomputing those metrics from individual probabilities. The exporter checks its completion and integrity evidence rather than repeating statistical analysis.

Each suite must have a separately written `publication_visual_qa.json` receipt:

```json
{
  "status": "passed_numeric_and_visual_review",
  "real_data": true,
  "reviewed_files": {
    "report_directory/figure_name.png": "exact SHA256 of the reviewed file",
    "report_directory/figure_name.pdf": "exact SHA256 of the reviewed file"
  }
}
```

This example shows the schema, not a completed receipt. The receipt must cover **every** exported PNG/PDF by its suite-relative path and current SHA256: 28 stage figure pairs, 19 clinical figure pairs, both attribution summary pairs, all 115 case PNGs, all six runs' raw curve pairs, and the optional stage atlas PDF. Record successful review only after inspecting the real artifacts. Original numerical markers can continue to say visual review is required; this separate receipt records that later review without rewriting frozen outputs.

## Public content and exclusions

The export contains all seeds' raw JSON/CSV learning histories, aggregate stage and clinical report data, all figure pairs, an index of graph links, the complete fixed attribution gallery, current native arrays and randomization arrays, the 230 pinned historical old/c3 native arrays, and the six aligned original-100 case tables. Raw history CSV rows must match their pinned JSON counterparts. The original-100 tables must have the same 100 unique filenames. The historical attribution files must match the immutable source hashes used by the attribution audit.

When the separate public six-checkpoint manifest and its fidelity receipt are present, a full export also includes these two metadata files under `research_checkpoints/`. Both must be present, the receipt must match the manifest hash, and both frozen selection hashes and the exact six-seed roster must match these suites. No checkpoint binary enters the artifact export. The separately stored research weights and their synthetic fidelity checks do not establish clinical validity.

If the stage suite has a verified 30-page atlas, its PDF is copied and its CSV source links are rewritten to the exported plots. Exact source digest and filename matching allow a retrieved atlas index to retain historical cloud paths in the original while the public derivative remains portable. The original index is unchanged.

The clinical `analysis.json` is a derivative that recursively removes every `patient_ids` key. Other known participant-record keys cause a failure. Clinical history and aggregate-metric schemas are restricted. The original private analysis remains unchanged and its SHA is recorded. The following are never copied: MCI prediction rows, clinical individual predictions, participant input manifests, MRI images, raw headers, QC panels, per-epoch validation records, selected training/validation patient records, model weights, archives, SSH/resource metadata, or runtime failure logs. The separate research-weight exporter has its own scope and guards.

`EXPORT_MANIFEST.json` records original source hashes, public output hashes, transformations and readback checks. `VISUAL_QA.json` records the source receipt hashes and reviewed visual counts without copying operational context. A list of guarded source names and hashes demonstrates that excluded private numerical files were integrity checked; their contents are absent. All Markdown/HTML local links must resolve within the export. Personal absolute runtime paths and credential-like text are rejected.

Existing outputs are never overwritten. In the unlikely event of an I/O failure during writing, an incomplete directory can remain without the completion manifest; inspect it and choose a fresh output name for the next attempt. Do not represent a directory without a valid completion manifest as a released package.

Methods, limitations drafts, source code and checkpoint exports remain separate sibling artifacts. The public artifact README states the reused-benchmark and small-clinical-pilot boundaries; final outcome prose must still be written from the actual verified results. Packaging success does not establish clinical readiness, anatomical attribution validity, or patient independence of the legacy image set.

## Synthetic verification

```sh
PYTHONPYCACHEPREFIX=/tmp/kevin_export_pycache python -m unittest discover \
  -s docs/kevin_paper_20261011/round2_publication \
  -p test_export_artifacts.py -v
```

The fixtures contain invented scalar metrics, placeholder file bytes and invented case names. They never load real patient inputs or execute a model. Tests cover source preservation, recursive privacy filtering, exact native-map coverage, no overwrite, stale visual receipts, changed payloads, incomplete clinical inventory coverage, JSON/CSV mismatch, unexpected participant fields, symlink rejection, absolute-path rejection, broken links, portable atlas links, recovery-log exclusion, ordinary-mode blocked attribution rejection, explicit diagnostic failure preservation, and stage-only publication without any attribution or clinical input.
