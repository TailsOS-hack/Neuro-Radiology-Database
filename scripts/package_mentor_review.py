#!/usr/bin/env python3
"""Package review artifacts, code and supporting evidence; excludes datasets/weights."""
import hashlib
import zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BUNDLE=ROOT/'docs/review_bundle'
assert not (BUNDLE/'explainability/RUNNING').exists(), 'Explainability generation incomplete'
files=set()
for folder in ['docs','scripts','src','training_logs/publication_evidence']:
    for p in (ROOT/folder).rglob('*'):
        if p.is_file() and p.suffix not in ['.zip','.pyc'] and p.name not in ['.DS_Store','PACKAGE_SHA256.txt']:
            files.add(p)
for p in (ROOT/'training_logs/publication_audit').rglob('*'):
    if p.is_file() and p.suffix in ['.json','.md']:files.add(p)
for folder in ['experiments_dedup_regularized','experiments_perceptual_regularized']:
    for p in (ROOT/'training_logs'/folder).rglob('*'):
        if p.is_file() and p.suffix in ['.json','.md']:files.add(p)
files.update(ROOT/p for p in ['training_logs/splits/strict_manifest.csv','training_logs/splits/perceptual_strict_manifest.csv','requirements.txt'])
manifest=BUNDLE/'PACKAGE_SHA256.txt'
manifest.write_text(''.join(f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(ROOT)}\n' for p in sorted(files)))
files.add(manifest)
archive=BUNDLE/'Dr_Cockroft_review_bundle.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
    for p in sorted(files):z.write(p,p.relative_to(ROOT))
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    for line in manifest.read_text().splitlines():
        digest,name=line.split('  ',1)
        assert hashlib.sha256(z.read(name)).hexdigest()==digest
print(f'{archive}: {len(files)} files, {archive.stat().st_size/1e6:.1f} MB, hashes verified')
