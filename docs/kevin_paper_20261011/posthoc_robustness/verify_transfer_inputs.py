#!/usr/bin/env python3
"""Verify legacy/original100 bytes against the independently pinned GPU input bundle."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tarfile
import stress_test as s

EXPECTED_ARCHIVE='b0a7324e58653e138ba520b4188fb529161ef8cd4c96f118987203623e2aad39'


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    root=args.root.resolve();archive=root/'data/external/kevin_runpod_transfer_v1.tar.gz'
    digest=s.sha(archive)
    if digest!=EXPECTED_ARCHIVE:raise ValueError('Original GPU transfer archive checksum mismatch')
    with tarfile.open(archive,'r:gz') as tf:
        raw=tf.extractfile('Neuro-Radiology-Database/bundle_manifest.json').read()
    manifest=json.loads(raw);rows=[r for r in manifest['files'] if r['category'] in ('retained_legacy_image','original100_image')]
    counts={category:sum(r['category']==category for r in rows) for category in ('retained_legacy_image','original100_image')}
    if counts!={'retained_legacy_image':27178,'original100_image':100}:raise ValueError('Bundle image membership changed')
    failures=[]
    for r in rows:
        image=root/r['path']
        if not image.is_file() or image.stat().st_size!=r['bytes'] or s.sha(image)!=r['sha256']:failures.append(r['path'])
    report=dict(status='all_legacy_and_original100_source_bytes_match_original_GPU_bundle' if not failures else 'source_mismatch',created_utc=datetime.now(timezone.utc).isoformat(),script_sha256=s.sha(Path(__file__)),archive_sha256=digest,bundle_manifest_sha256=hashlib.sha256(raw).hexdigest(),counts=counts,checked_files=len(rows),failures=failures,interpretation='The local source image bytes match the independently pinned bundle previously transferred to the GPU. This check does not establish the cause of cross-environment probability drift or patient independence.')
    if args.output.exists():raise ValueError('Use a new output path')
    s.dump(args.output,report);print(json.dumps(report,indent=2))
    if failures:raise RuntimeError('Input image mismatch')


if __name__=='__main__':main()
