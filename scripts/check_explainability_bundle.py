#!/usr/bin/env python3
"""Check review evidence integrity without loading models or installing PyTorch."""
import csv
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'docs/review_bundle/explainability'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    assert not (OUT/'RUNNING').exists(), 'Generation incomplete'
    provenance=json.loads((OUT/'provenance.json').read_text())
    for name,digest in provenance['source_sha256'].items(): assert sha(ROOT/name)==digest
    summary=json.loads((OUT/'summary.json').read_text())
    rows=list(csv.DictReader((OUT/'per_image.csv').open()))
    assert provenance['manifest_sha256']==sha(ROOT/'training_logs/splits/strict_manifest.csv')
    assert provenance['script_sha256']==sha(ROOT/'scripts/build_explainability_bundle.py')
    manifest={r['path']:r for r in csv.DictReader((ROOT/'training_logs/splits/strict_manifest.csv').open())}
    assert len({(r['task'],r['path']) for r in rows})==len(rows)
    for task,s in summary.items():
        selected=[r for r in rows if r['task']==task and r['quantitative_sample']=='1']
        valid=[r for r in selected if r['cam_status']=='ok']
        assert len(selected)==s['sample_n'] and len(valid)==s['valid_cam_n']
        assert len(selected)-len(valid)==s['failed_cam_n']
        ckpt=provenance['checkpoints'][task]
        p=ROOT/ckpt['path']
        if p.stat().st_size<1024: assert f"oid sha256:{ckpt['sha256']}" in p.read_text()
        else: assert sha(p)==ckpt['sha256']
        assert sha(ROOT/f'training_logs/publication_evidence/{task}/probabilities.csv')==ckpt['probabilities_sha256']
        for label in ckpt['classes']:
            assert sum(r['true_label']==label for r in selected)==provenance['per_class']
        for key,stats in s['metrics'].items():
            assert abs(sum(float(r[key]) for r in valid)/len(valid)-stats['mean'])<1e-10
    for r in rows:
        assert manifest[r['path']]['split']=='test'
        assert manifest[r['path']]['subtype']==r['true_label']
        assert sha(ROOT/r['path'])==r['image_sha256']
        assert float(r['max_probability_difference'])<1e-4
        if r['cam_status']=='ok':
            assert (OUT/r['cam_path']).is_file()
            assert 0<=float(r['border_mass'])<=1
            assert abs(float(r['paired_drop_advantage'])-(float(r['top20_drop'])-float(r['random20_drop'])))<1e-10
    for name in ['index.html','SUMMARY.md','figure8a_gradcam.png','figure8b_gradcam.png']:
        assert (OUT/name).stat().st_size>0
    print(f'PASS: {len(rows)} unique cases; sample counts, aggregates, source hashes and figure artifacts verified.')

if __name__=='__main__': main()
