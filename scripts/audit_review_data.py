"""Audit frozen image splits, including identical RGB pixels in different files.

This does not establish patient or augmentation-family independence. dHash
collisions are similarity flags, not proof of duplicated or mislabeled images.
"""
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'docs/final_review_20260920/data_audit.json'


def identities(path):
    content = path.read_bytes()
    with Image.open(path) as image:
        image.load()
        rgb = image.convert('RGB')
        gray = list(rgb.convert('L').resize((9, 8)).getdata())
        return {
            'file': hashlib.sha256(content).hexdigest(),
            'pixels': hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest(),
            'dhash': ''.join('1' if gray[y*9+x] > gray[y*9+x+1] else '0'
                             for y in range(8) for x in range(8)),
        }


def audit_rows(rows, root, cache=None):
    cache = {} if cache is None else cache
    buckets = {key: defaultdict(list) for key in ['file', 'pixels', 'dhash']}
    bad = []
    for row in rows:
        path = row['path']
        try:
            if path not in cache:
                cache[path] = identities(root / path)
        except (OSError, ValueError) as exc:
            bad.append({'path': path, 'error': str(exc)})
            continue
        for key in buckets:
            buckets[key][cache[path][key]].append(row)
    overlaps, conflicts = {}, {}
    for key, bucket in buckets.items():
        groups = [v for v in bucket.values() if len({r['split'] for r in v}) > 1]
        overlaps[key] = {
            'groups': len(groups), 'rows': sum(map(len, groups)),
            'split_patterns': dict(Counter(','.join(sorted({r['split'] for r in v})) for v in groups)),
            'dementia_rows': sum(r['domain'] == 'dementia' for v in groups for r in v),
            'examples': [[{k: r[k] for k in ['path', 'split', 'subtype']} for r in v] for v in groups[:3]],
        }
        conflicts[key] = sum(len({(r['domain'], r['subtype']) for r in v}) > 1 for v in bucket.values())
    return {
        'n': len(rows), 'unique_paths': len({r['path'] for r in rows}),
        'counts': dict(Counter(f"{r['domain']}/{r['split']}/{r['subtype']}" for r in rows)),
        'unreadable': bad, 'cross_split': overlaps,
        'conflicting_label_groups': conflicts,
    }


def main():
    cache, result = {}, {}
    for name in ['strict_manifest', 'perceptual_strict_manifest']:
        path = ROOT / f'training_logs/splits/{name}.csv'
        with path.open() as handle:
            rows = list(csv.DictReader(handle))
        result[name] = audit_rows(rows, ROOT, cache)
        result[name]['manifest_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        print(name, {key: value['rows'] for key, value in result[name]['cross_split'].items()}, flush=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
