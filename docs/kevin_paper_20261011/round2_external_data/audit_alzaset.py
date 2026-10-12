#!/usr/bin/env python3
"""Non-model source audit. Never infer patient IDs from filename similarity.

The intact provider ZIP stays untouched. JPEG decoding is used only for image
integrity, duplicate and similarity screening; no model inputs are exported.
"""
import csv
import hashlib
import io
import json
import re
import time
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DATA = ROOT / 'data/external/alzaset_zenodo_17084199'
ZIP = DATA / 'raw/AlzaSet.zip'
LEGACY = ROOT / 'docs/kevin_paper_20261011/publication/prepared/stage_manifest.csv'
EXPECTED = '86a61f4e8734b455663f40429ac54ff5fb9d39bac059c815cb3b51c60ecf0065'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()


def write_csv(path, rows, fields):
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def image_ids(b):
    with Image.open(io.BytesIO(b)) as im:
        im.load()
        rgb = im.convert('RGB')
        g = np.asarray(rgb.convert('L').resize((9, 8)))
        dh = ''.join(str(int(x)) for x in (g[:, :-1] > g[:, 1:]).ravel())
        return dict(width=rgb.width, height=rgb.height, mode=im.mode,
                    pixel_sha256=hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest(),
                    dhash=f'{int(dh, 2):016x}',
                    exif_keys='|'.join(str(x) for x in sorted(im.getexif().keys())))


class BKTree:
    def __init__(self, values):
        self.nodes = []
        for value in values:
            if not self.nodes:
                self.nodes.append((value, {})); continue
            idx = 0
            while True:
                v, children = self.nodes[idx]
                d = (v ^ value).bit_count()
                if d == 0:
                    break
                if d in children:
                    idx = children[d]
                else:
                    children[d] = len(self.nodes)
                    self.nodes.append((value, {})); break

    def search(self, value, radius):
        result, stack = [], [0]
        while stack:
            v, children = self.nodes[stack.pop()]
            d = (v ^ value).bit_count()
            if d <= radius:
                result.append((v, d))
            stack.extend(i for edge, i in children.items() if d - radius <= edge <= d + radius)
        return result


def resized_gray(b):
    with Image.open(io.BytesIO(b)) as im:
        return np.asarray(im.convert('L').resize((128, 128)), dtype=np.float32) / 255


def main():
    started = time.monotonic()
    assert sha(ZIP) == EXPECTED, 'Archive changed'
    legacy = list(csv.DictReader(LEGACY.open()))
    assert len(legacy) == 27178 and len({r['path'] for r in legacy}) == 27178
    legacy_pixels, legacy_dhash, legacy_files = defaultdict(list), defaultdict(list), defaultdict(list)
    for r in legacy:
        legacy_pixels[r['pixel_sha256']].append(r)
        legacy_dhash[int(r['dhash'], 16)].append(r)
        legacy_files[sha(ROOT / r['path'])].append(r)
    # Independently confirm hash algorithms agree with supplied legacy manifest.
    for r in legacy[::997]:
        ids = image_ids((ROOT / r['path']).read_bytes())
        assert ids['pixel_sha256'] == r['pixel_sha256'] and ids['dhash'] == r['dhash']
    tree = BKTree(sorted(legacy_dhash))
    rows, exact_matches, flags = [], [], []
    with zipfile.ZipFile(ZIP) as z:
        for info in z.infolist():
            if info.is_dir():
                continue
            parts = Path(info.filename).parts
            assert len(parts) == 5 and parts[1] in ['Axial', 'Coronal', 'Sagittal']
            assert parts[2] in ['train', 'valid', 'test'] and parts[3] in ['AD', 'NC']
            assert Path(info.filename).suffix.lower() == '.jpg'
            b = z.read(info)
            # This extracts a string token only. It is NOT a patient identifier.
            match = re.fullmatch(r'(\d{4})(.+?)SR(\d+)(?:\((\d+)\))?', Path(info.filename).stem)
            row = dict(path=info.filename, plane=parts[1], provider_split=parts[2],
                       provider_label=parts[3], byte_sha256=hashlib.sha256(b).hexdigest(),
                       filename_token=match[2] if match else '',
                       sequence_token=match[3] if match else '',
                       slice_token=match[1] if match else '', **image_ids(b))
            rows.append(row)
            for kind, lookup in [('byte', legacy_files), ('decoded_rgb', legacy_pixels)]:
                key = row['byte_sha256' if kind == 'byte' else 'pixel_sha256']
                for old in lookup.get(key, []):
                    exact_matches.append(dict(path=row['path'], legacy_path=old['path'], kind=kind))
            matches = tree.search(int(row['dhash'], 16), 4)
            if matches:
                a = resized_gray(b)
                for key, distance in matches:
                    # One representative per exact legacy decoded-pixel identity.
                    seen = set()
                    for old in legacy_dhash[key]:
                        if old['pixel_sha256'] in seen:
                            continue
                        seen.add(old['pixel_sha256'])
                        c = resized_gray((ROOT / old['path']).read_bytes())
                        corr = float(np.corrcoef(a.ravel(), c.ravel())[0, 1]) if a.std() and c.std() else None
                        flags.append(dict(path=row['path'], legacy_path=old['path'],
                                          dhash_hamming=distance, resized_gray_rmse=float(np.sqrt(np.mean((a-c)**2))),
                                          resized_gray_correlation=corr, confirmed_duplicate=False))
            if len(rows) % 2000 == 0:
                print(f'Audited {len(rows)} JPEGs; {len(flags)} similarity candidates', flush=True)
    assert len(rows) == 12741
    write_csv(DATA / 'image_integrity_private.csv', rows, list(rows[0]))
    write_csv(DATA / 'legacy_exact_matches.csv', exact_matches, ['path', 'legacy_path', 'kind'])
    flag_fields = ['path', 'legacy_path', 'dhash_hamming', 'resized_gray_rmse', 'resized_gray_correlation', 'confirmed_duplicate']
    write_csv(DATA / 'legacy_similarity_flags_private.csv', flags, flag_fields)
    duplicate_groups = {}
    for kind in ['byte_sha256', 'pixel_sha256']:
        bucket = defaultdict(list)
        for r in rows:
            bucket[r[kind]].append(r)
        groups = [v for v in bucket.values() if len(v) > 1]
        duplicate_groups[kind] = dict(groups=len(groups), images=sum(len(v) for v in groups),
            cross_split_groups=sum(len({x['provider_split'] for x in v}) > 1 for v in groups),
            cross_label_groups=sum(len({x['provider_label'] for x in v}) > 1 for v in groups),
            examples=[[x['path'] for x in v] for v in groups[:8]])
    tokens = defaultdict(list)
    for r in rows:
        if r['filename_token']:
            tokens[r['filename_token']].append(r)
    token_rows = [dict(token=k, images=len(v), planes='|'.join(sorted({r['plane'] for r in v})),
                       provider_splits='|'.join(sorted({r['provider_split'] for r in v})),
                       is_verified_patient_id=False) for k, v in sorted(tokens.items())]
    write_csv(DATA / 'unverified_filename_tokens_private.csv', token_rows, list(token_rows[0]))
    report = dict(status='quarantined_identity_mapping_unresolved', checked_utc=datetime.now(timezone.utc).isoformat(),
        archive_sha256=EXPECTED, legacy_manifest_sha256=sha(LEGACY),
        image_count=len(rows), extensions={'jpg':len(rows)}, raw_volumes=0, metadata_tables=0,
        paper_reported_images=12947, paper_reported_patients=79, independently_verified_patient_count=None,
        counts=dict(sorted(Counter(f"{r['plane']}/{r['provider_split']}/{r['provider_label']}" for r in rows).items())),
        shapes=dict(Counter(f"{r['width']}x{r['height']}" for r in rows)), modes=dict(Counter(r['mode'] for r in rows)),
        images_with_exif=sum(bool(r['exif_keys']) for r in rows),
        filename_tokens=dict(parsed_images=sum(bool(r['filename_token']) for r in rows),
                             unparsed_images=sum(not r['filename_token'] for r in rows),
                             unique_case_sensitive_tokens=len(tokens),
                             tokens_crossing_provider_splits=sum('|' in r['provider_splits'] for r in token_rows),
                             tokens_are_not_verified_patient_ids=True),
        internal_exact_duplicates=duplicate_groups,
        legacy_overlap=dict(compared_images=len(legacy),exact_match_pairs=len(exact_matches),
                            dhash_radius=4,similarity_candidate_pairs=len(flags),
                            candidate_alzaset_images=len({r['path'] for r in flags}),
                            lowest_resized_gray_rmse=min((r['resized_gray_rmse'] for r in flags), default=None),
                            highest_resized_gray_correlation=max((r['resized_gray_correlation'] for r in flags if r['resized_gray_correlation'] is not None), default=None),
                            similarity_flags_are_not_proof=True,
                            patient_overlap_not_excluded=True),
        decision='No training, validation, held-out inference, or patient-level claim until provider-backed patient/series mapping resolves identities and sequence metadata.',
        model_inference_performed=False, training_performed=False, raw_archive_changed=False,
        image_decoding_purpose='Integrity/deduplication only; no model tensors or preprocessed cohort exported.',
        seconds=round(time.monotonic()-started, 3))
    (HERE / 'alzaset_inventory_audit.json').write_text(json.dumps(report, indent=2)+'\n')
    outputs = [DATA / x for x in ['image_integrity_private.csv','legacy_exact_matches.csv','legacy_similarity_flags_private.csv','unverified_filename_tokens_private.csv']]
    (HERE / 'alzaset_inventory_artifact_hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):sha(p) for p in outputs},indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
