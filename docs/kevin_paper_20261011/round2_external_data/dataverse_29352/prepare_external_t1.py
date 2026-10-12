#!/usr/bin/env python3
"""Prepare the acquired Harvard cohort with the unchanged existing process().

All 30 provider subjects are retained in the private subject table, including
the one missing structural scan. No model inference or training is performed.
"""
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import tarfile

import nibabel as nib
import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
DATA = ROOT / 'data'
ACQUIRED = DATA / 'external/dataverse_29352'
OUT = DATA / 'external/clinical_harvard29352_v1'
HELPER = HERE.parents[1] / 'preprocess_clinical_t1.py'
HELPER_SHA = 'bc3e1c7012cc54e9453af38d554073b3666cbc115fbb2a2072ed10a822f49475'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''): h.update(b)
    return h.hexdigest()


def write_csv(path, rows, fields=None):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields or list(rows[0])); writer.writeheader(); writer.writerows(rows)


def decoded_sha(path):
    image = nib.as_closest_canonical(nib.load(path))
    values = np.ascontiguousarray(np.asarray(image.dataobj, dtype='<f4'))
    h = hashlib.sha256(str(values.shape).encode() + b'\n' + values.tobytes())
    return h.hexdigest()


def main():
    if sha(HELPER) != HELPER_SHA:
        raise ValueError('Frozen preprocessing helper changed')
    if OUT.exists():
        raise ValueError('Preserve existing prepared cohort; output must be new')
    old = DATA / 'external/clinical_t1_preprocessed_v1'
    protected = [HELPER, old / 'clinical_slices.csv', old / 'subjects.csv', old / 'preprocessing_qc_private.json',
                 HERE.parents[1] / 'clinical_preprocessing_summary.json']
    before = {str(p.relative_to(ROOT)): sha(p) for p in protected}
    acquisition = json.loads((ACQUIRED / 'acquisition.json').read_text())
    archive_path = ACQUIRED / 'raw/DATA.tar.gz'
    if sha(archive_path) != acquisition['sha256']:
        raise ValueError('Acquired source archive changed')
    inventory = json.loads((ACQUIRED / 'scan_manifest.json').read_text())
    structural = {r['subject_id']: r for r in inventory if r['modality'] == 'structural'}
    all_subjects = {r['subject_id']: r for r in inventory}
    ordered = sorted(all_subjects, key=lambda key: hashlib.sha256(key.encode()).hexdigest())
    ids = {key: f'hv_{i:03d}' for i, key in enumerate(ordered, 1)}
    source_dir = ACQUIRED / 'structural_extracted'
    source_dir.mkdir(exist_ok=True)
    paths = {r['archive_path']: r for r in structural.values()}
    # Extract only the previously verified structural members into named files;
    # never execute or use archive extraction methods on provider paths.
    with tarfile.open(archive_path, 'r|gz') as archive:
        for member in archive:
            record = paths.get(str(Path(member.name)))
            if record is None: continue
            destination = source_dir / (ids[record['subject_id']] + '.nii')
            if not member.isfile() or member.size != record['bytes']:
                raise ValueError('Structural member differs from verified inventory')
            if not destination.exists():
                with archive.extractfile(member) as src, destination.open('xb') as dst:
                    shutil.copyfileobj(src, dst, 1048576)
            if destination.is_symlink() or sha(destination) != record['sha256']:
                raise ValueError('Structural extraction checksum mismatch')
    OUT.mkdir()
    spec = importlib.util.spec_from_file_location('frozen_clinical_t1', HELPER)
    helper = importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)
    rows, qcs, subjects, headers, failures = [], [], [], [], []
    for key in ordered:
        record = all_subjects[key]
        subject = dict(cohort='harvard_dataverse_29352', subject_id=ids[key],
                       diagnosis={'AD': 'AD', 'NCS': 'CN', 'MCI': 'MCI'}[record['provider_group']],
                       split='external', age='', sex='', source=source_dir / (ids[key] + '.nii'))
        private = dict(provider_subject_key=key, provider_subject_id=record['provider_subject_id'],
                       provider_group=record['provider_group'], **subject)
        private['source'] = str(subject['source'].relative_to(DATA)) if key in structural else ''
        if key not in structural:
            private['status'] = 'unavailable_structural_scan'
            failures.append(dict(subject_id=ids[key], status=private['status'], reason='Provider release includes fMRI but no T1 for this subject'))
            subjects.append(private); continue
        image = nib.load(subject['source'])
        headers.append(dict(subject_id=ids[key], header=str(image.header), affine=image.affine.tolist(),
                            extensions=[dict(code=e.get_code(), content_hex=e.get_content().hex()) for e in image.header.extensions]))
        try:
            subject_rows, qc = helper.process(subject, OUT, DATA)
            rows.extend(subject_rows); qcs.append(qc); private['status'] = 'processed'
        except Exception as exc:
            private['status'] = 'preprocessing_failed'
            failures.append(dict(subject_id=ids[key], status=private['status'], reason=f'{type(exc).__name__}: {exc}'))
        subjects.append(private)
        print(json.dumps(dict(prepared=len(qcs), attempted=len(subjects), provider_subjects=len(ordered))), flush=True)
    write_csv(OUT / 'clinical_slices.csv', rows, helper.FIELDS)
    write_csv(OUT / 'subjects_private.csv', subjects)
    (OUT / 'preprocessing_qc_private.json').write_text(json.dumps(dict(version=helper.VERSION, subjects=qcs), indent=2) + '\n')
    (OUT / 'headers_private.json').write_text(json.dumps(headers, indent=2) + '\n')
    (OUT / 'failures_private.json').write_text(json.dumps(failures, indent=2) + '\n')

    # Compare all 98 previously acquired Besga/OpenNeuro subjects, including
    # bipolar raw inputs, and all 666 old prepared slices. This establishes
    # absence/presence of exact data duplicates, never identity independence.
    old_subjects = helper.read_rows(old / 'subjects.csv')
    raw, decoded, pixels = defaultdict(list), defaultdict(list), defaultdict(list)
    for r in old_subjects:
        source = DATA / r['source']
        old_key = r['cohort'] + '__' + r['subject_id']
        raw[sha(source)].append(old_key)
        decoded[decoded_sha(source)].append(old_key)
    old_qc = json.loads((old / 'preprocessing_qc_private.json').read_text())['subjects']
    for q in old_qc:
        for h in q['pixel_sha256']: pixels[h].append(q['group_key'])
    within_raw, within_decoded, within_pixels = defaultdict(set), defaultdict(set), defaultdict(set)
    overlaps = []
    for q in qcs:
        source = DATA / q['source_path']; key = q['group_key']
        ds = decoded_sha(source)
        within_raw[q['source_sha256']].add(key); within_decoded[ds].add(key)
        for kind, h, lookup in [('raw_bytes', q['source_sha256'], raw), ('canonical_decoded_float32_voxels', ds, decoded)]:
            if lookup.get(h): overlaps.append(dict(group_key=key, kind=kind, prior_subjects=lookup[h]))
        for h in q['pixel_sha256']:
            within_pixels[h].add(key)
            if pixels.get(h): overlaps.append(dict(group_key=key, kind='prepared_RGB_pixels', prior_subjects=pixels[h]))
    within = {kind: {h: sorted(keys) for h, keys in lookup.items() if len(keys) > 1}
              for kind, lookup in [('raw_bytes', within_raw), ('decoded_voxels', within_decoded), ('prepared_pixels', within_pixels)]}
    overlap_report = dict(prior_raw_subjects=len(old_subjects), prior_processed_subjects=len(old_qc),
                          prior_processed_slices=sum(len(q['pixel_sha256']) for q in old_qc),
                          cross_cohort_matches=overlaps, within_harvard_cross_subject_duplicates=within,
                          limit='Exact bytes, canonical decoded float32 volumes, and exact prepared RGB pixels only; no proof of patient independence or exclusion of near-duplicate source scans.')
    (OUT / 'overlap_audit_private.json').write_text(json.dumps(overlap_report, indent=2) + '\n')
    qa = OUT / 'qa'; qa.mkdir()
    # Show every available subject's complete nine-slice sequence; labels use
    # opaque subject IDs, with no diagnosis, model output, or selected examples.
    keys = sorted({r['group_key'] for r in rows})
    for page, start in enumerate(range(0, len(keys), 6), 1):
        selected = keys[start:start + 6]
        canvas = Image.new('RGB', (9 * 112, len(selected) * 140), (24, 24, 24)); draw = ImageDraw.Draw(canvas)
        for line, key in enumerate(selected):
            group = [r for r in rows if r['group_key'] == key]
            for col, r in enumerate(group):
                canvas.paste(Image.open(DATA / r['path']).resize((112, 112)), (col * 112, line * 140))
                draw.text((col * 112 + 2, line * 140 + 114), f"{r['subject_id']} {r['slice_fraction']}", fill='white')
        canvas.save(qa / f'all_subjects_nine_slices_{page:02d}.png')
    after = {str(p.relative_to(ROOT)): sha(p) for p in protected}
    if before != after or sha(archive_path) != acquisition['sha256']:
        raise ValueError('Protected historical input changed')
    counts = Counter(q['diagnosis'] for q in qcs)
    report = dict(status='prepared_pending_gross_visual_qc', checked_utc=datetime.now(timezone.utc).isoformat(),
        dataset_doi='10.7910/DVN/29352', license='CC0-1.0', source_archive_sha256=acquisition['sha256'],
        manifest=str((OUT / 'clinical_slices.csv').relative_to(ROOT)), manifest_sha256=sha(OUT / 'clinical_slices.csv'),
        subjects_in_provider_release=len(all_subjects), subjects_with_structural_t1=len(structural),
        subjects_processed=len(qcs), processed_diagnosis_counts=dict(counts), slices_written=len(rows),
        unavailable_t1_subjects=sum(s['status'] == 'unavailable_structural_scan' for s in subjects),
        unavailable_t1_diagnosis_counts=dict(Counter(s['diagnosis'] for s in subjects if s['status'] == 'unavailable_structural_scan')),
        preprocessing_failures=sum(s['status'] == 'preprocessing_failed' for s in subjects),
        all_subjects_retained_in_private_table=True, all_processed_subjects_external=all(r['split'] == 'external' for r in rows),
        mci_binary_labels_blank=all(r['binary_label'] == '' for r in rows if r['diagnosis'] == 'MCI'),
        binary_scoring_subjects=counts['AD'] + counts['CN'],
        preprocessing_version=helper.VERSION, preprocessing_helper_sha256=HELPER_SHA,
        unchanged_slice_fractions=[round(float(f), 3) for f in helper.FRACTIONS],
        protected_historical_files_sha256=before, protected_historical_files_unchanged=True,
        exact_cross_cohort_duplicate_matches=len(overlaps),
        exact_within_cohort_cross_subject_duplicates={k: len(v) for k, v in within.items()},
        prior_raw_subjects_checked=len(old_subjects), prior_processed_slices_checked=overlap_report['prior_processed_slices'],
        gross_visual_qc='pending', qa_pages=len(list(qa.glob('*.png'))),
        clinical_protocol_frozen=False, model_inference_performed=False,
        limitations=['Gross rendering QC does not establish clinical or anatomical validity.',
            'The original fixed heuristic has no atlas registration, validated skull stripping, bias-field correction, or scanner harmonization.',
            'Exact duplicate checks do not establish patient independence against unidentified legacy source data.',
            'A small single-site probable-AD/healthy-control cohort cannot establish real-world clinical readiness.',
            'Missing T1 is retained and reported; no replacement or imputation is performed.'],
        wrapper_sha256=sha(Path(__file__)))
    (HERE / 'preprocessing_summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
