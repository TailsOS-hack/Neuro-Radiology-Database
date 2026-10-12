#!/usr/bin/env python3
"""Independent byte/pixel/membership readback; no model inference."""
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
DATA = ROOT / 'data'
OUT = DATA / 'external/clinical_harvard29352_v1'


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''): h.update(b)
    return h.hexdigest()


def read_csv(p):
    with p.open(newline='') as f: return list(csv.DictReader(f))


def main():
    report = json.loads((HERE / 'preprocessing_summary.json').read_text())
    rows = read_csv(OUT / 'clinical_slices.csv')
    subjects = read_csv(OUT / 'subjects_private.csv')
    qcs = json.loads((OUT / 'preprocessing_qc_private.json').read_text())['subjects']
    inventory = json.loads((DATA / 'external/dataverse_29352/scan_manifest.json').read_text())
    structural = {r['subject_id']: r for r in inventory if r['modality'] == 'structural'}
    assert len(rows) == 261 and len(subjects) == 30 and len(qcs) == 29
    assert sha(OUT / 'clinical_slices.csv') == report['manifest_sha256']
    assert Counter(s['status'] for s in subjects) == {'processed': 29, 'unavailable_structural_scan': 1}
    assert Counter(q['diagnosis'] for q in qcs) == {'AD': 9, 'CN': 10, 'MCI': 10}
    by_subject = defaultdict(list)
    for r in rows: by_subject[r['subject_id']].append(r)
    assert len(by_subject) == 29 and len({r['path'] for r in rows}) == 261
    actual_pngs = {p.relative_to(DATA).as_posix() for p in (OUT / 'images').rglob('*.png')}
    assert actual_pngs == {r['path'] for r in rows}
    qc_by_id = {q['subject_id']: q for q in qcs}
    for s in subjects:
        if s['status'] == 'unavailable_structural_scan':
            assert s['provider_subject_key'] not in structural and not s['source'] and s['subject_id'] not in by_subject
            continue
        source = structural[s['provider_subject_key']]
        assert sha(DATA / s['source']) == source['sha256']
        assert s['diagnosis'] == {'AD': 'AD', 'NCS': 'CN', 'MCI': 'MCI'}[source['provider_group']]
        group = by_subject[s['subject_id']]
        assert len(group) == 9
        assert [r['slice_fraction'] for r in group] == ['0.450','0.500','0.550','0.600','0.650','0.700','0.750','0.800','0.850']
        assert len({r['slice_index'] for r in group}) == 9
        qc = qc_by_id[s['subject_id']]
        for i, r in enumerate(group):
            assert r['split'] == 'external' and r['diagnosis'] == s['diagnosis']
            assert r['binary_label'] == {'AD': '1', 'CN': '0', 'MCI': ''}[s['diagnosis']]
            pixels = np.asarray(Image.open(DATA / r['path']))
            assert pixels.shape == (224, 224, 3) and pixels.dtype == np.uint8
            assert pixels.max() > pixels.min() and np.array_equal(pixels[:,:,0], pixels[:,:,1]) and np.array_equal(pixels[:,:,0], pixels[:,:,2])
            assert hashlib.sha256(pixels.tobytes()).hexdigest() == qc['pixel_sha256'][i]
    for path, digest in report['protected_historical_files_sha256'].items():
        assert sha(ROOT / path) == digest
    result = dict(status='passed_independent_output_readback', checked_utc=datetime.now(timezone.utc).isoformat(),
        subjects_retained=30, subjects_with_verified_t1=29, missing_t1_retained=1,
        all_261_pngs_decoded_and_pixel_hash_checked=True, exact_manifest_file_membership=True,
        source_diagnosis_and_hash_join_verified=True, all_external=True, mci_binary_blank=True,
        historical_inputs_unchanged=True, model_inference_performed=False,
        manifest_sha256=sha(OUT / 'clinical_slices.csv'),
        private_subject_table_sha256=sha(OUT / 'subjects_private.csv'),
        private_preprocessing_qc_sha256=sha(OUT / 'preprocessing_qc_private.json'),
        private_overlap_audit_sha256=sha(OUT / 'overlap_audit_private.json'),
        qa_sha256={p.name: sha(p) for p in sorted((OUT / 'qa').glob('*.png'))},
        script_sha256=sha(Path(__file__)))
    (HERE / 'output_verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
