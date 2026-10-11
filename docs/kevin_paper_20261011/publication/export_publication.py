#!/usr/bin/env python3
"""Build a portable, curated paper export without inference or source mutation.

Run from any directory. Only files recorded in this export's prior manifest may
be replaced. Frozen local reports remain unchanged; exported JSON/Markdown is
explicitly a derivative with paths normalized and private records omitted.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shutil

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "docs/kevin_paper_20261011"
RESULTS = ROOT / "training_logs/experiments/kevin_20261011/runpod_blackwell_20261011"
PREPARED = ROOT / "training_logs/experiments/kevin_20261011/prepared"
DROP = object()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def normalize_text(value):
    value = value.replace("Abid's ten recommended official sources", "The ten recommended official sources")
    value = value.replace("Abid's ten recommended sources", "The ten recommended sources")
    value = value.replace("Abid's ten recommendations", "The ten recommended sources")
    value = value.replace("Dataset crosswalk from Abid's recommendations", "Dataset recommendation crosswalk")
    value = value.replace("Abid dataset access", "Recommended dataset access")
    value = value.replace("Abid10_not_claimed", "recommended10_not_claimed")
    value = value.replace("recording/email text", "private correspondence")
    value = value.replace(str(ROOT) + "/", "").replace(str(ROOT), ".")
    value = value.replace("/tmp/Neuro-Radiology-Database/", "").replace("/tmp/Neuro-Radiology-Database", ".")
    value = value.replace("/workspace/kevin_results/", "results/").replace("/workspace/kevin_results", "results")
    value = re.sub(r"/Users/[^\s\"'<>),;]+", "[local path omitted]", value)
    return value


def sanitize(value):
    """Drop clinical row records and operational/private source fields."""
    if isinstance(value, dict):
        if "subject_id" in value:
            return DROP
        omitted = {
            "mci_prediction_only", "email_handling", "files", "private_qc_montages",
            "command", "private_manifest", "private_integrity_report",
            "local_directory", "local_raw_directory", "workspace", "archive", "destination",
        }
        clean = {}
        for key, child in value.items():
            if key in omitted:
                continue
            if key == "environment":
                permitted = {"torch", "torchvision", "numpy", "scipy", "pillow", "matplotlib", "python", "device", "threads", "batch_size", "workers"}
                child = {k:v for k,v in child.items() if k in permitted} if isinstance(child, dict) else {}
            if any(part in key.lower() for part in ("ssh", "private_key", "api_key", "access_token", "refresh_token", "automation", "pod_id", "volume_id", "thread_id", "message_id")):
                continue
            result = sanitize(child)
            if result is not DROP:
                public_key = {"email_names":"recommended_source_count", "email_dataset_name":"recommended_dataset_name"}.get(key, key)
                clean[normalize_text(public_key)] = result
        return clean
    if isinstance(value, list):
        output = []
        for item in value:
            result = sanitize(item)
            if result is not DROP:
                output.append(result)
        return output
    if isinstance(value, str):
        return normalize_text(value)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--results-root", type=Path, default=RESULTS)
    parser.add_argument("--require-complete", action="store_true", help="Require all new quantitative, attribution, stress and replay outputs.")
    args = parser.parse_args()
    out = args.output.resolve()
    results = args.results_root.resolve()
    if args.require_complete:
        required = [
            BASE / "paper_figures/generated/figure_provenance.json",
            BASE / "posthoc_explainability/results/SHA256SUMS.json",
            BASE / "posthoc_robustness/results_20261011/audit_complete.json",
            BASE / "posthoc_robustness/clean_split_replay_20261011/replay_complete.json",
            BASE / "paper_figures/all_graphs_atlas.pdf",
        ]
        if not all(p.exists() for p in required):
            raise ValueError("Final export requires completed diagnostics and graph atlas; existing export preserved")
    out.mkdir(parents=True, exist_ok=True)
    old = out / "EXPORT_MANIFEST.json"
    if old.exists():
        previous = read_json(old)
        for entry in previous["files"]:
            path = out / entry["path"]
            if not path.resolve().is_relative_to(out):
                raise ValueError("Unsafe previous export path")
            if path.exists() and sha(path) != entry["export_sha256"]:
                raise ValueError(f"Export file changed outside generator: {entry['path']}")
        for entry in previous["files"]:
            path = out / entry["path"]
            if path.is_file():
                path.unlink()
        old.unlink()

    entries, mapping, source_hashes = [], {}, {}
    def target(rel):
        p = out / rel
        if not p.resolve().is_relative_to(out):
            raise ValueError("Export target escaped directory")
        if p.exists():
            raise ValueError(f"Refusing to overwrite an unmanaged export file: {rel}")
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def register(source, rel, transformation):
        dest = out / rel
        record = {"path":str(rel), "bytes":dest.stat().st_size, "export_sha256":sha(dest), "transformation":transformation}
        if source is not None:
            source = Path(source).resolve()
            record.update(source=str(source.relative_to(ROOT)), source_sha256=sha(source))
            source_hashes[str(source)] = record["source_sha256"]
            mapping[str(source)] = dest
        entries.append(record)

    def copy(source, rel):
        source = Path(source)
        before = sha(source)
        shutil.copyfile(source, target(rel))
        if sha(out / rel) != before or sha(source) != before:
            raise ValueError("Source changed while copying; rerun after completion")
        register(source, rel, "byte-identical copy")

    def put_json(source, rel, value=None):
        data = sanitize(read_json(source) if value is None else value)
        if data is DROP:
            raise ValueError("Attempt to publish a clinical subject record")
        target(rel).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
        register(source, rel, "portable JSON derivative; local paths normalized; clinical row/private operational fields omitted")

    def put_text(rel, text, source=None, transformation="generated public documentation"):
        target(rel).write_text(text)
        register(source, rel, transformation)

    # Explicit aggregate/provenance inputs. No raw acquired MRI or participant table.
    documents = {
        "current_local_audit.json":"datasets/current_local_audit.json",
        "dataset_registry.json":"datasets/dataset_registry.json",
        "abid_dataset_crosswalk.json":"datasets/recommended_dataset_crosswalk.json",
        "besga_acquisition.json":"datasets/besga_acquisition_summary.json",
        "openneuro_acquisition.json":"datasets/openneuro_acquisition_summary.json",
        "clinical_preprocessing_summary.json":"datasets/clinical_preprocessing_summary.json",
        "analysis/completed_methods_evidence.json":"evidence/completed_methods_evidence.json",
        "analysis/real_world_readiness.json":"evidence/readiness_protocol_snapshot.json",
        "analysis/old_hash_group_unexposed_sensitivity_audit.json":"evidence/sensitivity_audit.json",
        "analysis/final_outcome_numeric_audit.json":"evidence/independent_numeric_audit.json",
        "analysis/final_figure_qa.json":"evidence/original_figure_qa.json",
        "analysis/analysis_protocol.json":"evidence/analysis_protocol.json",
        "analysis/ensemble_protocol.json":"evidence/ensemble_protocol.json",
        "analysis/ensemble_supplement_freeze_record.json":"evidence/ensemble_freeze_record.json",
    }
    for source, rel in documents.items():
        value = None
        if source == "abid_dataset_crosswalk.json":
            value = read_json(BASE / source)
            value["purpose"] = "Ten-source access, clinical-fit and overlap crosswalk for the dementia experiment."
            for item in value.get("consequential_reconciliations", []):
                if item.get("topic") == "OASIS pooling":
                    item["finding"] = "Provider advice warns against pooling OASIS-1, -2 and -3 or presenting one as independent external evaluation of another: people can recur under different anonymized IDs. OASIS-4 is described by the provider as a unique cohort; independence from every other repository still needs verification."
        put_json(BASE / source, rel, value)
    for name in ("checkpoint_verification.json", "test_execution.json"):
        source = BASE / "revision_audit" / name
        if source.exists():
            put_json(source, "evidence/" + name)
    for name in ("figure_qa.json", "all_graphs_atlas_verification.json"):
        source = BASE / "paper_figures" / name
        if source.exists():
            put_json(source, "evidence/" + name)
    for name in ("implementation_verification.json", "stress_output_verification.json", "transfer_input_verification.json", "clean_replay_verification.json"):
        source = BASE / "posthoc_robustness" / name
        if source.exists():
            put_json(source, "evidence/" + name)
    amendment = BASE / "posthoc_explainability/replay_amendment.json"
    if amendment.exists():
        put_json(amendment, "figures/attribution/replay_amendment.json")
    mapping[str((results / "provenance/source_code/analysis/ensemble_protocol.json").resolve())] = out / "evidence/ensemble_protocol.json"
    copy(BASE / "analysis/old_hash_group_unexposed_sensitivity.csv", "prepared/old_hash_group_unexposed_sensitivity.csv")
    for name in ("stage_manifest.csv", "same_case_test_intersection.csv"):
        copy(PREPARED / name, "prepared/" + name)
    put_json(PREPARED / "split_audit.json", "prepared/split_audit.json")
    for source, rel in {
        "validation_selection_before_evaluation.json":"results/validation_selection.json",
        "evaluation_complete.json":"results/evaluation_complete.json",
        "frozen_evaluation_all6/evaluation.json":"results/evaluation.json",
        "frozen_uncertainty_analysis_all6/uncertainty_analysis.json":"results/uncertainty_analysis.json",
        "ensemble_supplement/ensemble_supplement.json":"results/ensemble_supplement.json",
        "ensemble_supplement/ensemble_specifications.json":"results/ensemble_specifications.json",
    }.items():
        put_json(results / source, rel)
    for name in ("stage_metrics.csv", "paired_differences.csv", "reader_paired_differences.csv", "clinical_exact_intervals.csv"):
        copy(results / "frozen_uncertainty_analysis_all6" / name, "results/" + name)
    for folder in ("frozen_evaluation_all6", "ensemble_supplement"):
        for source in sorted((results / folder).glob("*original100*.csv")):
            copy(source, "results/original100/" + source.name)
        for source in sorted((results / folder).glob("*.csv")):
            if any(part in source.name for part in ("_internal_same_case_predictions", "_old_hash_group_unexposed_sensitivity_predictions")):
                copy(source, "results/legacy_predictions/" + source.name)
    copy(ROOT / "data/evaluation/ground_truth/radiologist_test_key.csv", "results/original100/radiologist_test_key.csv")
    for mode in ("stage_only", "multitask"):
        for seed in (20261011, 20261012, 20261013):
            run = f"{mode}_seed{seed}"
            for name in ("history.csv", "loss_curves.png", "accuracy_curves.png"):
                copy(results / run / name, f"training/{run}/{name}")
    for directory in ("publication_curves", "evaluation_figures"):
        for source in sorted((results / directory).iterdir()):
            rel = "figures/" + directory + "/" + source.name
            if source.suffix in (".png", ".pdf", ".csv"):
                copy(source, rel)
            elif source.suffix == ".json":
                put_json(source, rel)

    retrieval = read_json(BASE / "training/results_retrieval_verification.json")
    cleanup = read_json(BASE / "training/runpod_cleanup_evidence.json")
    suite = read_json(results / "suite_status.json")
    completion = {
        "role":"Public aggregate completion record; operational identifiers omitted",
        "archive_sha256":retrieval["archive_sha256"],
        "archive_bytes":cleanup["archive_bytes"],
        "verified_payload_files":retrieval["verified_payload_files"],
        "destination_readback_verified":retrieval["destination_readback_verified"],
        "retrieval_verified_before_deletion":cleanup["retrieval_verified_before_deletion"],
        "all_task_created_paid_resources_absent":cleanup["all_task_created_paid_resources_absent"],
        "historical_training_orchestrator_state":suite["state"],
        "historical_state_is_not_current_evaluation_status":True,
        "evaluation_completion":read_json(results / "evaluation_complete.json"),
        "source_record_sha256":{
            "retrieval":sha(BASE / "training/results_retrieval_verification.json"),
            "cleanup":sha(BASE / "training/runpod_cleanup_evidence.json"),
            "historical_training_status":sha(results / "suite_status.json"),
        },
    }
    put_json(BASE / "training/results_retrieval_verification.json", "evidence/completion_summary.json", completion)
    for p in (BASE / "training/runpod_cleanup_evidence.json", results / "suite_status.json"):
        mapping[str(p.resolve())] = out / "evidence/completion_summary.json"

    # Completion-gated post-hoc families: omit partial executions and operational logs.
    family_specs = [
        ("quantitative", BASE / "paper_figures/generated", "figure_provenance.json", None),
        ("attribution", BASE / "posthoc_explainability/results", "summary.json", None),
        ("robustness", BASE / "posthoc_robustness/results_20261011", "audit_complete.json", "completed_posthoc_descriptive_audit"),
        ("clean_replay", BASE / "posthoc_robustness/clean_split_replay_20261011", "replay_complete.json", "completed_posthoc_clean_split_replay"),
    ]
    families = {}
    excluded_names = {"environment.json", "progress.json", "runtime_estimate.json", "input_image_checksums.json", "output_checksums.json", "SHA256SUMS.json"}
    for family, folder, marker, expected_status in family_specs:
        ready = (folder / marker).exists()
        if ready and expected_status:
            ready = read_json(folder / marker).get("status") == expected_status
        if ready and family == "attribution":
            ready = (folder / "SHA256SUMS.json").exists()
        families[family] = {"complete_marker_present":ready, "source_marker":str((folder / marker).relative_to(ROOT))}
        if not ready:
            continue
        for source in sorted(folder.rglob("*")):
            if not source.is_file() or source.name in excluded_names or "progress" in source.name.lower():
                continue
            if source.suffix not in (".png", ".pdf", ".csv", ".json", ".md", ".html", ".npy"):
                continue
            if "subject_predictions" in source.name or "__pycache__" in source.parts:
                raise ValueError("Unexpected private clinical or cache file in public figure family")
            rel = "figures/" + family + "/" + str(source.relative_to(folder))
            if source.suffix == ".json":
                put_json(source, rel)
            elif source.suffix in (".md", ".html"):
                put_text(rel, normalize_text(source.read_text()), source, "portable text derivative")
            else:
                # All attribution/replay/stress case rows are legacy images.
                copy(source, rel)
    if args.require_complete and not all(x["complete_marker_present"] for x in families.values()):
        raise ValueError("Requested final export, but one or more post-hoc families are incomplete")
    atlas = BASE / "paper_figures/all_graphs_atlas.pdf"
    if atlas.exists():
        copy(atlas, "All_Graphs_Atlas.pdf")
        atlas_index = BASE / "paper_figures/all_graphs_atlas_index.csv"
        if atlas_index.exists():
            rows = list(csv.DictReader(atlas_index.open()))
            for row in rows:
                source_path = (ROOT / row['source_path']).resolve()
                if str(source_path) not in mapping:
                    raise ValueError("Atlas page source is not included in public export")
                row['public_path'] = str(mapping[str(source_path)].relative_to(out))
            with target("All_Graphs_Atlas_Index.csv").open('w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader(); writer.writerows(rows)
            register(atlas_index, "All_Graphs_Atlas_Index.csv", "atlas page/source hashes retained; public paths added")
    runtime_sources = {
        "original_training":results / "stage_only_seed20261012/config.json",
        "posthoc_attribution":BASE / "posthoc_explainability/results/summary.json",
        "posthoc_stress":BASE / "posthoc_robustness/results_20261011/environment.json",
        "posthoc_clean_replay":BASE / "posthoc_robustness/clean_split_replay_20261011/replay_metadata.json",
    }
    runtime = {"interpretation":"Recorded runtime differences are relevant to probability replay. They do not establish the cause of discrepancies. No hostnames, serials or account/connection details are included.", "records":{}}
    runtime_keys = {"torch", "torchvision", "numpy", "scipy", "pillow", "matplotlib", "python", "device", "device_actual", "threads", "batch_size", "workers"}
    for label, path in runtime_sources.items():
        if path.exists():
            source = read_json(path)
            fields = source.get('environment', source)
            runtime['records'][label] = {'versions_and_execution':{k:v for k,v in fields.items() if k in runtime_keys}, 'source':str(path.relative_to(ROOT)), 'source_sha256':sha(path)}
            source_hashes[str(path.resolve())] = sha(path)
    put_json(None, "evidence/runtime_versions.json", runtime)

    def rewrite_links(text, dest):
        def link(match):
            label, raw = match.group(1), match.group(2).strip("<>")
            if raw.startswith(("http://", "https://", "#", "mailto:")):
                return match.group(0)
            if raw.startswith("/"):
                source = str(Path(raw).resolve())
                destination = mapping.get(source)
                if destination is None:
                    candidate = Path(source)
                    if candidate.exists() and candidate.is_relative_to(ROOT) and candidate.suffix == ".py":
                        destination = candidate
                    else:
                        destination = out / "DATA_AND_REPRODUCIBILITY.md"
                return f"[{label}]({os.path.relpath(destination, dest.parent)})"
            return match.group(0)
        return re.sub(r"\[([^\]]+)\]\((<?[^)]+>?)\)", link, text)

    reports = {
        "Kevin_Results_Overview.md":"reports/Overview.md",
        "Kevin_Experiment_Results.md":"reports/Full_Results.md",
        "analysis/manuscript_methods_addendum.md":"reports/Completed_Methods.md",
        "analysis/methods_protocol_snapshot.md":"reports/Prespecified_Methods.md",
        "analysis/real_world_readiness.txt":"reports/Readiness_Protocol_Snapshot.txt",
    }
    for source, rel in reports.items():
        mapping[str((BASE / source).resolve())] = out / rel
    mapping[str((results / "frozen_uncertainty_analysis_all6/analysis_handoff.md").resolve())] = out / "reports/Statistical_Handoff.md"
    reports_extra = [(BASE / source, rel) for source, rel in reports.items()]
    reports_extra.append((results / "frozen_uncertainty_analysis_all6/analysis_handoff.md", "reports/Statistical_Handoff.md"))
    for source, rel in reports_extra:
        text = source.read_text()
        if source.name == "Kevin_Experiment_Results.md":
            text = re.sub(r"MCI descriptive probabilities.*?(?=MCI descriptive probabilities|### test__|## )", "", text, flags=re.S)
            text = re.sub(r"## Resource cleanup evidence\n.*?(?=## Interpretation)", "## Resource cleanup evidence\n\nThe archive and 415 payload files were verified locally before deletion. Task GPU and storage deletion were verified. See [public completion record](../evidence/completion_summary.json); operational identifiers are omitted.\n\n", text, flags=re.S)
        text = rewrite_links(text, out / rel)
        text = normalize_text(text)
        note = "> Portable public derivative of the audited local source. Clinical subject-level probability rows and operational identifiers are omitted. Scientific aggregate results are unchanged. SHA256 values quoted from the source refer to original frozen files; export hashes are in EXPORT_MANIFEST.json.\n\n"
        if source.suffix == ".txt":
            note = "Historical pre-completion readiness protocol snapshot. Completed results are in Overview.md and Full_Results.md. Public derivative; private/runtime details omitted.\n\n"
        put_text(rel, note + text, source, "portable report derivative; private clinical row/operational sections omitted; aggregate numbers preserved")

    attribution_note = BASE / "posthoc_explainability/README.md"
    if attribution_note.exists():
        text = attribution_note.read_text().replace(
            "(results/", "("
        ).replace("(run_attribution_audit.py)", "(../../../posthoc_explainability/run_attribution_audit.py)")
        put_text("figures/attribution/README.md", normalize_text(text), attribution_note, "portable paths; findings unchanged")
    robustness_note = BASE / "posthoc_robustness/README.md"
    if robustness_note.exists():
        text = normalize_text(robustness_note.read_text())
        text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", lambda m:m.group(0) if m.group(2).startswith(('https://','http://','#')) else m.group(1), text)
        text += "\n\nPublic navigation: [stress findings](FINDINGS.md), [scenario metrics](scenario_metrics.csv), [reconstruction guide](../../RECONSTRUCTION.md).\n"
        put_text("figures/robustness/README.md", text, robustness_note, "portable derivative; source-local navigation replaced by public links")

    data_note = """# Data, source licenses and reproducibility

This export reports a research experiment, not a clinically ready system. The
legacy four labels are not verified CDR stages. Legacy person/augmentation-family
identifiers are unavailable, and the reused 100-image comparison cannot establish
clinical superiority over neuroradiologists. The clinical AD/control task is
separate and showed weak generalization with tiny samples.

The legacy dataset is [Singhal's augmented Kaggle derivative](https://www.kaggle.com/datasets/aryansinghal10/alzheimers-multiclass-dataset-equal-and-augmented).
Local provenance records Apache 2.0 for that derivative, LGPL 3.0 for its declared
upstream, and unresolved original-source license compatibility. Existing public
benchmark JPEGs support the requested Grad-CAM panels; this does not resolve the
source chain. No newly acquired clinical MRI is redistributed here.

The acquired clinical sources were [Besga, Zenodo 3935636](https://doi.org/10.5281/zenodo.3935636)
([associated article](https://pmc.ncbi.nlm.nih.gov/articles/PMC7658647/); recorded
CC BY 4.0), and [OpenNeuro ds007561, version 1.0.0](https://openneuro.org/datasets/ds007561/versions/1.0.0)
(DOI 10.18112/openneuro.ds007561.v1.0.0; recorded CC0). Besga acquisition included
78 source participants, of whom 54 AD/control participants entered the primary
clinical task; 24 bipolar participants were excluded. OpenNeuro included 20 source
participants: 15 binary evaluable and 5 MCI prediction-only. Nine slices per person
do not increase the independent sample count. Official raw releases from the ten
recommended sources were not newly acquired.

Source/version/access evidence is in [the dataset registry](datasets/dataset_registry.json)
and [recommendation crosswalk](datasets/recommended_dataset_crosswalk.json).
This export reports recorded license metadata, not a new license determination.
Raw acquired MRI, participant tables, clinical subject prediction rows, private
correspondence, SSH/account/resource metadata, logs and local ZIPs remain local.
Some source references consequently describe private artifacts rather than files
distributed here; omitted navigation points to this explanation.

The three legacy prepared manifests are under `prepared/`; their paths refer to
the existing repository's legacy image files. The original clinical manifest
remains local because it joins source participants and diagnoses to slices.
Its hash, aggregate split counts and preprocessing code are retained. Reproducing
clinical processing requires obtaining the cited releases and rebuilding the
documented private manifest. Public aggregate JSON exports remove only subject
records/private operational fields and normalize paths. They are not substitutes
for exact frozen-input files when running hash-guarded analysis scripts.

The six original trainer/evaluator/analysis source families are preserved in their
repository locations, identified by STAGING_ALLOWLIST.txt and the source hashes.
These are research runners; some frozen orchestrators retain their original
container layout. The public report can be read without that runtime. The six
research checkpoints are separately retained with Git LFS under
`models/kevin_20261011/`; the production baseline is unchanged.

To rebuild this curated export after all diagnostics finish, from the repository:

```sh
python3 docs/kevin_paper_20261011/publication/export_publication.py --require-complete
```

The generator does not perform inference, upload data, change original evidence,
stage Git files or provision resources. It validates source stability, public
links and bounded private-content checks. A clean checkout can read all exported
reports; regeneration additionally requires the locally preserved frozen results.
"""
    put_text("DATA_AND_REPRODUCIBILITY.md", data_note)
    reproduction = """# Reconstructing results for the unchanged research scripts

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
python3 docs/kevin_paper_20261011/publication/reconstruct_legacy_attribution.py \\
  --output /tmp/kevin-public-attribution-inputs
python3 docs/kevin_paper_20261011/posthoc_explainability/run_attribution_audit.py \\
  --results-root /tmp/kevin-public-attribution-inputs \\
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
python3 docs/kevin_paper_20261011/posthoc_robustness/stress_test.py \\
  --root . --results RESULTS_ROOT --output /tmp/kevin-stress-repeat --device cpu
python3 docs/kevin_paper_20261011/posthoc_robustness/clean_split_replay.py \\
  --root . --results RESULTS_ROOT --output /tmp/kevin-split-repeat --device cpu
python3 docs/kevin_paper_20261011/paper_figures/recreate_quantitative_figures.py \\
  --results RESULTS_ROOT --output /tmp/kevin-figures-repeat \\
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
"""
    put_text("RECONSTRUCTION.md", reproduction)

    # Keep frozen source at original paths, with an explicit science-only allowlist.
    allow = [
        "training/train_candidate.py", "training/evaluate_candidates.py", "training/run_training_suite.py",
        "training/run_frozen_evaluation.py", "training/requirements-analysis.txt",
        "analysis/summarize_frozen_evaluation.py", "analysis/freeze_validation_selection.py",
        "analysis/ensemble_supplement.py", "analysis/build_results_report.py", "analysis/plot_training_curves.py",
        "analysis/plot_evaluation_figures.py", "analysis/analysis_protocol.json", "analysis/ensemble_protocol.json",
        "analysis/old_hash_group_unexposed_sensitivity.csv", "analysis/old_hash_group_unexposed_sensitivity_audit.json",
        "analysis/test_synthetic_analysis.py", "analysis/test_synthetic_results_report.py", "analysis/test_ensemble_supplement.py",
        "preprocess_clinical_t1.py", "validate_cohort_manifest.py", "data_dictionary.json", "cohort_manifest_template.csv",
        "paper_figures/recreate_quantitative_figures.py", "paper_figures/prior_figure_inventory.csv", "paper_figures/prior_figure_inventory.json",
        "paper_figures/test_figure_recreation.py", "paper_figures/verify_and_render_figures.py", "paper_figures/assemble_all_graphs_atlas.py",
        "posthoc_explainability/run_attribution_audit.py", "posthoc_robustness/stress_test.py", "posthoc_robustness/clean_split_replay.py",
        "posthoc_robustness/test_stress_test.py", "posthoc_robustness/verify_transfer_inputs.py", "posthoc_robustness/verify_clean_replay.py",
    ]
    safe_sources = ["README.md", ".gitattributes"] + [str((BASE / x).relative_to(ROOT)) for x in allow if (BASE / x).exists()]
    # Existing prepared source paths are kept for hash-guarded reproducibility;
    # the clinical source manifest is deliberately not part of the public list.
    safe_sources += [str((PREPARED / x).relative_to(ROOT)) for x in ("stage_manifest.csv", "same_case_test_intersection.csv", "split_audit.json")]
    model_dir = ROOT / "models/kevin_20261011"
    if (model_dir / "CHECKPOINTS.json").exists():
        check = read_json(model_dir / "CHECKPOINTS.json")
        for row in check["checkpoints"]:
            if sha(model_dir / row["file"]) != row["sha256"]:
                raise ValueError("Public research checkpoint SHA mismatch")
        safe_sources += [str(p.relative_to(ROOT)) for p in sorted(model_dir.iterdir()) if p.suffix in (".pt", ".md", ".json")]
    pngs = sorted(str((out / e["path"]).relative_to(out)) for e in entries if e["path"].endswith(".png"))
    figure_index = "# Figures and case galleries\n\nAll figures are descriptive research outputs. Failed attribution maps and tiny clinical denominators remain disclosed.\n\n"
    if atlas.exists():
        figure_index += "**[Open the complete all-graphs PDF atlas](All_Graphs_Atlas.pdf)** · [page index and source hashes](All_Graphs_Atlas_Index.csv)\n\n"
    figure_index += "[Open all 115 old/new Grad-CAM case panels](figures/attribution/index.html) · [attribution findings](figures/attribution/README.md) · [post-hoc stress findings](figures/robustness/FINDINGS.md)\n\n## Quantitative figures\n\n"
    quant_index = out / 'figures/quantitative/FIGURE_INDEX.csv'
    if quant_index.exists():
        indexed = {}
        for row in csv.DictReader(quant_index.open()):
            indexed.setdefault(row['figure'], []).append(row)
        for slug, rows in indexed.items():
            title = rows[0].get('title') or slug.replace('_', ' ').capitalize()
            links = [f"[{r['format'].upper()}](figures/quantitative/{Path(r['path']).name})" for r in rows]
            figure_index += f"- {title}: " + ' · '.join(links) + '\n'
    for title, folder in [('Post-hoc stress graphs','robustness'),('Attribution controls','attribution')]:
        figure_index += f"\n## {title}\n\n"
        for p in pngs:
            pp = Path(p)
            if pp.parent == Path('figures') / folder:
                figure_index += f"- [{pp.stem.replace('_',' ').capitalize()}]({p})\n"
    figure_index += "\n## Raw curves and history for every seed\n\n| Condition | Seed | Loss | Accuracy | Raw history |\n| --- | --- | --- | --- | --- |\n"
    for mode in ('stage_only','multitask'):
        for seed in (20261011,20261012,20261013):
            directory = f'training/{mode}_seed{seed}'
            figure_index += f"| {mode.replace('_',' ')} | {seed} | [loss]({directory}/loss_curves.png) | [accuracy]({directory}/accuracy_curves.png) | [CSV]({directory}/history.csv) |\n"
    for title, prefix in [('Preserved seven-model evaluation figures','figures/evaluation_figures/'),('Preserved loss overviews','figures/publication_curves/'),('Individual attribution case filenames','figures/attribution/cases/')]:
        figure_index += f"\n<details>\n<summary>{title}</summary>\n\n"
        figure_index += '\n'.join(f'- [{Path(p).name}]({p})' for p in pngs if p.startswith(prefix)) + '\n\n</details>\n'
    put_text("FIGURES.md", figure_index)
    status = "\n".join(f"| {name} | {'complete output present' if data['complete_marker_present'] else 'pending; rerun exporter when complete'} |" for name,data in families.items())
    atlas_link = "- [All quantitative, stress and attribution-control graphs in one PDF](All_Graphs_Atlas.pdf)\n" if atlas.exists() else ""
    readme = f"""# Kevin dementia paper: portable publication materials

Six runs completed 305 epochs. The individual new models matched or slightly
underperformed the old image benchmark. A fixed stage ensemble corrected one
additional primary image, with an uncertainty interval including zero. Clinical
generalization was weak; these results do not establish an overall stronger or
clinically ready model. The production checkpoint remains unchanged.

- [Results overview](reports/Overview.md)
- [Complete six-run results and original 100-case comparison](reports/Full_Results.md)
- [Completed Methods](reports/Completed_Methods.md) and [prespecified Methods](reports/Prespecified_Methods.md)
- [Statistics and ensemble handoff](reports/Statistical_Handoff.md)
- [All figures]({os.path.relpath(out / 'FIGURES.md', out)})
{atlas_link}- [All 115 old/new Grad-CAM case panels](figures/attribution/index.html)
- [Dataset sources, licenses, omissions and reproducibility](DATA_AND_REPRODUCIBILITY.md)
- [Reconstruction and exact-run boundaries](RECONSTRUCTION.md)
- [Attribution findings and replay discrepancies](figures/attribution/README.md)
- [Post-hoc stress findings](figures/robustness/FINDINGS.md)
- [Historical readiness assessment](reports/Readiness_Protocol_Snapshot.txt)
- [Aggregate retrieval and cleanup evidence](evidence/completion_summary.json)
- [Software/device versions relevant to numerical replay](evidence/runtime_versions.json)
- [Six research checkpoints and their hashes]({os.path.relpath(model_dir / 'README.md', out)})

The audited local sources remain unchanged. These public copies normalize links
and omit private clinical rows and operational metadata; original-source and
export hashes are separately recorded in EXPORT_MANIFEST.json. Git tracking of
legacy benchmark images is not proof of dataset license compatibility.

Attribution preserved all 230 original maps' predictions/logits, while 2 of 144
randomized maps failed or were flat. CPU replay preserved all 200 original-100
classes but 43 probability vectors exceeded the preset tolerance. The broader
stress replay also preserved classes but showed numerical probability differences.
Candidate 3 fell from 100/100 on local clean original-100 inputs to 96/100 under
the tested stronger noise setting, while the old model remained 100/100 throughout
that fixed grid. The new candidate was not uniformly more robust. These synthetic
image changes are post-hoc diagnostics, not clinical safety or acquisition tests.

Each seed's original loss plot, accuracy plot and unsmoothed history CSV is linked
in the [six-run results table](reports/Full_Results.md#all-six-training-runs-and-raw-curves).

| Additional diagnostic family | Export status |
| --- | --- |
{status}

Rebuild with `python3 docs/kevin_paper_20261011/publication/export_publication.py --require-complete`.
STAGING_ALLOWLIST.txt is an explicit candidate list, not a command to bulk-stage
the working directory. Large research models require Git LFS; no cloud action is
performed by this exporter.
"""
    put_text("README.md", readme)
    for path, before in source_hashes.items():
        if sha(path) != before:
            raise ValueError(f"Source changed during export: {Path(path).name}; rerun after completion")

    link_errors = []
    for entry in entries:
        path = out / entry["path"]
        if path.suffix not in (".md", ".html", ".txt", ".json", ".csv"):
            continue
        text = path.read_text()
        if re.search(r"/Users/|/workspace/|-----BEGIN (?:OPENSSH|RSA|EC|PRIVATE)|(?:rpa_|sk-proj-|ghp_|github_pat_)[A-Za-z0-9_-]{20,}|openneuro_ds007561__sub-", text):
            raise ValueError(f"Private/runtime content remains in public export: {entry['path']}")
        links = re.findall(r"\]\(<?([^)>]+)>?\)", text) if path.suffix == ".md" else re.findall(r'(?:src|href)="([^\"]+)"', text) if path.suffix == ".html" else []
        for href in links:
            if href.startswith(("https://", "http://", "#", "mailto:")):
                continue
            if not (path.parent / href.split('#')[0]).resolve().exists():
                link_errors.append({"file":entry["path"], "link":href})
    if link_errors:
        raise ValueError("Broken public links: " + json.dumps(link_errors))
    public_files = [str((out / e["path"]).relative_to(ROOT)) for e in entries]
    control_files = [str((out / p).relative_to(ROOT)) for p in ("export_publication.py", "reconstruct_legacy_attribution.py", "EXPORT_MANIFEST.json", "STAGING_ALLOWLIST.txt")]
    put_text("STAGING_ALLOWLIST.txt", "\n".join(sorted(set(safe_sources + public_files + control_files))) + "\n")
    manifest = {
        "role":"Curated public derivative; original frozen sources retained unchanged",
        "exporter_sha256":sha(__file__),
        "no_inference_or_git_or_cloud_actions":True,
        "new_clinical_subject_records_exported":False,
        "clinical_aggregate_metrics_preserved":True,
        "original100_case_table_preserved":True,
        "all_source_hashes_stable_during_export":True,
        "all_public_navigation_links_resolve_locally":True,
        "posthoc_families":families,
        "file_count":len(entries),
        "files":sorted(entries, key=lambda x:x["path"]),
    }
    (out / "EXPORT_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"export":str(out.relative_to(ROOT)), "files":len(entries), "png_figures":len(pngs), "posthoc_families":families}, indent=2))


if __name__ == "__main__":
    main()
