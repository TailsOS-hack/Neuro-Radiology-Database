"""Trace a labelled image cohort to frozen training manifests without changing them.

Both file bytes and decoded RGB pixels are checked: a re-encoded copy can have a
different file hash while still being exactly the same model input. Neither check
establishes patient or augmentation-family independence.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
DEFAULT_MANIFESTS = [ROOT / 'training_logs/splits' / f'{name}.csv' for name in
                     ('strict_manifest', 'perceptual_strict_manifest')]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def image_identities(path: Path) -> dict:
    with Image.open(path) as image:
        rgb = image.convert('RGB')
        return {
            'file_sha256': sha256(path),
            'decoded_rgb_sha256': hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest(),
            'width': rgb.width,
            'height': rgb.height,
        }


def read_csv(path: Path, required_columns: set[str]) -> list[dict]:
    with path.open(newline='', encoding='utf-8-sig') as handle:
        reader = csv.DictReader(handle)
        missing = required_columns - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f'{path}: missing columns {sorted(missing)}')
        return list(reader)


def display_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def summarize_matches(cases: list[dict], manifest_name: str, hash_kind: str) -> dict:
    matches = [case['manifest_matches'][manifest_name][hash_kind] for case in cases]
    combinations = Counter(','.join(sorted({row['split'] for row in rows}))
                           for rows in matches if rows)
    return {
        'matched_images': sum(bool(rows) for rows in matches),
        'matched_source_rows': sum(map(len, matches)),
        'unmatched_images': [case['filename'] for case, rows in zip(cases, matches) if not rows],
        'split_combinations': dict(sorted(combinations.items())),
        'train_exposed_images': sum(any(row['split'] == 'train' for row in rows) for rows in matches),
        'validation_exposed_images': sum(any(row['split'] == 'val' for row in rows) for rows in matches),
        'test_only_images': sum(bool(rows) and {row['split'] for row in rows} == {'test'} for rows in matches),
        'label_conflict_images': [case['filename'] for case, rows in zip(cases, matches)
                                  if any(row['subtype'] != case['true_category'] for row in rows)],
    }


def audit_cohort(cohort_dir: Path, labels_path: Path, manifest_paths: list[Path],
                 *, root: Path = ROOT, expected_count: int = 100) -> dict:
    labels = read_csv(labels_path, {'FileName', 'TrueCategory'})
    names = [row['FileName'] for row in labels]
    if len(labels) != expected_count:
        raise ValueError(f'Expected {expected_count} labelled images, found {len(labels)}')
    if len(set(names)) != len(names):
        raise ValueError('Duplicate filenames in cohort labels')
    if any(not name or Path(name).name != name for name in names):
        raise ValueError('Cohort labels must contain plain filenames, without directories')
    if any(not row['TrueCategory'].strip() for row in labels):
        raise ValueError('Empty true category in cohort labels')
    actual_names = {path.name for path in cohort_dir.iterdir()
                    if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS}
    if actual_names != set(names):
        raise ValueError(f'Cohort image/label mismatch: missing={sorted(set(names) - actual_names)}, '
                         f'unlabelled={sorted(actual_names - set(names))}')
    cases = [dict(filename=row['FileName'], true_category=row['TrueCategory'],
                  **image_identities(cohort_dir / row['FileName']), manifest_matches={}) for row in labels]
    cache = {}
    manifests = {}
    for manifest_path in manifest_paths:
        name = display_path(manifest_path, root)
        if name in manifests:
            raise ValueError(f'Duplicate manifest: {name}')
        rows = read_csv(manifest_path, {'path', 'split', 'domain', 'subtype'})
        if len({row['path'] for row in rows}) != len(rows):
            raise ValueError(f'{name}: duplicate source paths')
        if any(row['split'] not in {'train', 'val', 'test'} for row in rows):
            raise ValueError(f'{name}: unsupported split value')
        lookups = {kind: defaultdict(list) for kind in ('file_sha256', 'decoded_rgb_sha256')}
        for row in rows:
            path = (root / row['path']).resolve()
            if path not in cache:
                cache[path] = image_identities(path)
            record = {key: row[key] for key in ('path', 'split', 'domain', 'subtype')}
            for kind, lookup in lookups.items():
                lookup[cache[path][kind]].append(record)
        for case in cases:
            case['manifest_matches'][name] = {kind: lookup.get(case[kind], [])
                                               for kind, lookup in lookups.items()}
        manifests[name] = {
            'manifest_sha256': sha256(manifest_path), 'source_rows': len(rows),
            **{kind: summarize_matches(cases, name, kind) for kind in lookups},
        }
    return {
        'schema_version': 1,
        'created_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'cohort_directory': display_path(cohort_dir, root),
        'labels_path': display_path(labels_path, root),
        'labels_sha256': sha256(labels_path),
        'cohort_size': len(cases),
        'class_counts': dict(sorted(Counter(case['true_category'] for case in cases).items())),
        'unique_file_hashes': len({case['file_sha256'] for case in cases}),
        'unique_decoded_rgb_hashes': len({case['decoded_rgb_sha256'] for case in cases}),
        'source_files_decoded': len(cache),
        'manifests': manifests,
        'limitations': [
            'This cohort is not independent when its images match training or validation rows.',
            'A manifest match documents exposure in that split, not proof of which manifest trained a checkpoint.',
            'No patient or augmentation-family identifiers are available; different hashes do not establish independence.',
            'Repeated model selection using this cohort makes it a development benchmark, even for unmatched images.',
        ],
        'images': cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohort-dir', type=Path, default=ROOT / 'data/evaluation/images')
    parser.add_argument('--labels', type=Path, default=ROOT / 'data/evaluation/ground_truth/radiologist_test_key.csv')
    parser.add_argument('--manifest', type=Path, action='append', help='Repeat for each frozen manifest to audit.')
    parser.add_argument('--root', type=Path, default=ROOT, help='Root for relative source paths inside manifests.')
    parser.add_argument('--expected-count', type=int, default=100)
    parser.add_argument('--output', type=Path, default=ROOT / 'docs/evaluation_100_20261007/cohort_audit.json')
    args = parser.parse_args()
    result = audit_cohort(args.cohort_dir, args.labels, args.manifest or DEFAULT_MANIFESTS,
                          root=args.root, expected_count=args.expected_count)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({key: value for key, value in result.items() if key != 'images'}, indent=2))
    print(f'Saved cohort audit to {args.output}')


if __name__ == '__main__':
    main()
