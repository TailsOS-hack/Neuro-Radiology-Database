#!/usr/bin/env python3
"""Rebuild the public legacy attribution input layout; no inference or fitting.

This is not reconstruction of the private clinical evaluation. Public selection
metadata is a derivative; the unchanged attribution runner checks candidate
identity and checkpoint hashes, while full frozen statistical guards additionally
require private original artifacts documented in RECONSTRUCTION.md.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists():
        raise ValueError('Choose a new output directory; existing contents are preserved')
    models = ROOT / 'models/kevin_20261011'
    catalog = json.loads((models / 'CHECKPOINTS.json').read_text())
    selection = json.loads((HERE / 'results/validation_selection.json').read_text())
    if selection['exploratory_representative_candidate'] != 'candidate_3':
        raise ValueError('Validation-selected representative changed')
    for row in catalog['checkpoints']:
        if sha(models / row['file']) != row['sha256']:
            raise ValueError('Checkpoint hash mismatch; hydrate the Git LFS objects')
        if selection['candidates'][row['candidate']]['checkpoint_sha256'] != row['sha256']:
            raise ValueError('Catalog and frozen candidate identity differ')
    baseline = ROOT / 'models/alzheimers_classifier.pt'
    if sha(baseline) != 'a8cc03aea74f5310c1fc7936c72f94a44e55af11e2ada2a8ba5f8edd6d924db5':
        raise ValueError('Baseline hash mismatch; hydrate the Git LFS object')
    paths = sorted((HERE / 'results/original100').glob('*_original100_predictions.csv'))
    individual = [p for p in paths if not p.name.startswith('ensemble_')]
    if len(individual) != 7:
        raise ValueError('Need all seven archived original-100 model CSVs')
    key = {r['FileName']:r['TrueCategory'] for r in csv.DictReader((HERE / 'results/original100/radiologist_test_key.csv').open())}
    for path in individual:
        rows = list(csv.DictReader(path.open()))
        if len(rows) != 100 or {r['FileName']:r['TrueCategory'] for r in rows} != key:
            raise ValueError('Original-100 key membership or labels differ')
    out.mkdir(parents=True)
    (out / 'frozen_evaluation_all6').mkdir()
    shutil.copyfile(HERE / 'results/validation_selection.json', out / 'validation_selection_before_evaluation.json')
    for path in individual:
        shutil.copyfile(path, out / 'frozen_evaluation_all6' / path.name)
    for row in catalog['checkpoints']:
        folder = out / f"{row['condition']}_seed{row['seed']}"
        folder.mkdir()
        os.symlink(models / row['file'], folder / 'best.pt')
    note = {'role':'Public legacy attribution reconstruction only', 'clinical_inputs_reconstructed':False,
            'private_frozen_hash_guarded_analysis_supported':False,
            'checkpoint_hashes_verified':6, 'archived_individual_original100_csvs':7,
            'source_export_manifest_sha256':sha(HERE / 'EXPORT_MANIFEST.json')}
    (out / 'PUBLIC_RECONSTRUCTION.json').write_text(json.dumps(note, indent=2) + '\n')
    print(json.dumps(note, indent=2))


if __name__ == '__main__':
    main()
