#!/usr/bin/env python3
"""Private completed-clinical-suite migration after full original retrieval.

No inference, training, cloud or Git actions. Scientific payloads are never
rewritten. Only the three original optimizer-heavy last.pt files are omitted.
The full original archive remains independently verified and untouched.
"""
from __future__ import annotations
import argparse
import gzip
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import tarfile

BASE='docs/kevin_paper_20261011/round2_clinical/'
PINS={BASE+'clinical_bag.py':'96c4d3a53b717d4e0bfcee25221c8c31ac6b8c84e9ea03652adfa9c45cd88156',
      BASE+'prepared/protocol.json':'58061826e2c1b351845bf3a0a6022552a4f1e60ce637c6d267524f62b8bfe2cf',
      BASE+'prepared/patients.json':'ecac589248aa48dfc117a94b2650d78ea4df945b507d51cf3da0a6b6713289e1'}
SEEDS=[20261021,20261022,20261023]
OMIT={f'clinical_seed{s}/last.pt' for s in SEEDS}
TOP='clinical_completed_migration_v1/'
CONTROL={TOP+n for n in ['manifest.json','original_artifact_manifest.json','original_retrieval_receipt.json']}
HEX=re.compile(r'[0-9a-f]{64}')
MAX_BYTES=2*1024**3


def encoded(value):return (json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
def digest(stream):
    h=hashlib.sha256()
    for block in iter(lambda:stream.read(1048576),b''):h.update(block)
    return h.hexdigest()
def sha(path):
    with Path(path).open('rb') as f:return digest(f)
def parse(value):
    def unique(pairs):
        result={}
        for key,item in pairs:
            if key in result:raise ValueError('Duplicate JSON key')
            result[key]=item
        return result
    return json.loads(value,object_pairs_hook=unique)
def read(path):return parse(Path(path).read_bytes())
def safe(name):
    if not isinstance(name,str) or not name or '\\' in name or '\x00' in name:raise ValueError('Unsafe path')
    p=PurePosixPath(name)
    if p.is_absolute() or '..' in p.parts or str(p)!=name or any(v.startswith('.') for v in p.parts):raise ValueError('Unsafe path')
    return p
def regular(path):
    p=Path(path)
    if p.is_symlink() or not p.is_file():raise ValueError('Expected regular non-symlink file')
    return p
def target(root,name):
    p=root
    for part in safe(name).parts:
        p=p/part
        if p.is_symlink():raise ValueError('Symlink path refused')
    return p


def verify_scientific_suite(root,suite):
    for name,h in PINS.items():
        if sha(regular(target(root,name)))!=h:raise ValueError('Frozen clinical source/prepared input changed')
    # Existing inference markers must be preserved; migration cannot reset the
    # evaluator's single-launch guard or turn a partial test into a fresh test.
    if (suite/'clinical_evaluation_launch.json').exists() or (suite/'clinical_evaluation').exists():
        raise ValueError('Clinical evaluation already launched; do not migrate as unevaluated')
    program="""import json,sys
from pathlib import Path
root,suite=map(Path,sys.argv[1:]);sys.path.insert(0,str(root/'docs/kevin_paper_20261011/round2_clinical'))
import clinical_bag as c
v=c.verify_completed_suite(suite,root/'docs/kevin_paper_20261011/round2_clinical/prepared',root)
if (suite/'clinical_evaluation_freeze.json').exists():
 import clinical_compare as cc
 cc.load_frozen(root,suite,root/'docs/kevin_paper_20261011/round2_clinical/prepared',root/'data/external/round2_clinical_evaluation/patients.json',False)
# Normalize returned display paths in this receipt only; never rewrite a file.
for r in v['selections']:r['checkpoint_path']=str(Path(r['checkpoint_path']).relative_to(suite.resolve()))
print(json.dumps(v))
"""
    run=subprocess.run([sys.executable,'-B','-c',program,str(root),str(suite)],capture_output=True,text=True,timeout=240)
    if run.returncode:raise ValueError('Unchanged clinical suite verification failed: '+run.stderr[-4000:])
    checked=parse(run.stdout)
    if checked.get('status')!='verified_complete_converged_no_inference' or [r['seed'] for r in checked['selections']]!=SEEDS:
        raise ValueError('Unexpected completed-suite verification result')
    for r in checked['selections']:
        if r['checkpoint_path']!=f'clinical_seed{r["seed"]}/best.pt':raise ValueError('Nonportable selected checkpoint path')
    return checked


def original_verified(root,suite,receipt_path,checksum_path):
    receipt_bytes=regular(receipt_path).read_bytes();receipt=parse(receipt_bytes)
    checksum=regular(checksum_path).read_text().strip().splitlines()
    if len(checksum)!=1 or not HEX.fullmatch(checksum[0].split()[0]):raise ValueError('Expected independent archive SHA256')
    expected=checksum[0].split()[0];archive=Path(receipt['archive']);archive=archive if archive.is_absolute() else root/archive
    if (receipt.get('destination_readback_verified') is not True or Path(receipt['destination']).resolve()!=suite
            or receipt.get('archive_sha256')!=expected or sha(regular(archive))!=expected):raise ValueError('Full original retrieval receipt mismatch')
    with tarfile.open(archive,'r:gz') as tar:
        members=tar.getmembers();index={m.name:m for m in members}
        if len(index)!=len(members) or any(not m.isfile() or m.issparse() for m in members):raise ValueError('Unsafe original archive')
        for n in index:
            p=safe(n)
            if len(p.parts)<2 or p.parts[0]!='kevin_results':raise ValueError('Unexpected original archive root')
        if any(str(p) in index for n in index for p in PurePosixPath(n).parents):raise ValueError('Original file/directory collision')
        raw=tar.extractfile('kevin_results/ARTIFACT_SHA256.json').read();manifest=parse(raw);rows=manifest['files']
        records={r['path']:r for r in rows}
        if (manifest.get('version')!=1 or len(records)!=len(rows) or receipt.get('verified_payload_files')!=len(rows)
                or set(index)!=set(records)|{'kevin_results/ARTIFACT_SHA256.json'}
                or regular(suite/'ARTIFACT_SHA256.json').read_bytes()!=raw):raise ValueError('Original full artifact manifest mismatch')
        output=[]
        for name,r in records.items():
            relative=str(safe(name).relative_to('kevin_results'));local=regular(target(suite,relative));member=index[name]
            if (type(r['size']) is not int or member.size!=r['size'] or local.stat().st_size!=r['size']
                    or not HEX.fullmatch(r['sha256']) or sha(local)!=r['sha256']
                    or digest(tar.extractfile(member))!=r['sha256']):raise ValueError('Full original payload verification failed')
            output.append(dict(path=relative,bytes=r['size'],sha256=r['sha256']))
    if not OMIT<={r['path'] for r in output}:raise ValueError('Full original retrieval must include all three last.pt files')
    return sorted(output,key=lambda r:r['path']),raw,receipt_bytes,expected


def add(tar,name,stream,size):
    item=tarfile.TarInfo(name);item.size=size;item.mode=0o600;item.mtime=0;tar.addfile(item,stream)


def build(root,suite,receipt,checksum,output):
    root=root.resolve(strict=True);suite=suite.resolve(strict=True);output=output.absolute();partial=output.with_suffix('.partial')
    if any(p.exists() or p.is_symlink() for p in [output,partial]) or output.resolve().is_relative_to(suite):raise ValueError('Use a new archive outside original suite')
    rows,original,receipt_bytes,archive_sha=original_verified(root,suite,receipt,checksum)
    checked=verify_scientific_suite(root,suite)
    included=[r for r in rows if r['path'] not in OMIT];omitted=[r for r in rows if r['path'] in OMIT]
    manifest=dict(schema='completed_clinical_migration_v1',source_archive_sha256=archive_sha,
        original_artifact_manifest_sha256=hashlib.sha256(original).hexdigest(),original_retrieval_receipt_sha256=hashlib.sha256(receipt_bytes).hexdigest(),
        omitted=omitted,files=included,original_payload_files=len(rows),source_verification=checked,
        scientific_payload_bytes_changed=False,source_files_deleted=False,model_inference_performed=False)
    output.parent.mkdir(parents=True,exist_ok=True)
    with partial.open('xb') as raw,gzip.GzipFile(fileobj=raw,mode='wb',mtime=0) as gz,tarfile.open(fileobj=gz,mode='w|') as tar:
        for name,value in [('manifest.json',encoded(manifest)),('original_artifact_manifest.json',original),('original_retrieval_receipt.json',receipt_bytes)]:
            add(tar,TOP+name,io.BytesIO(value),len(value))
        for r in included:
            with regular(target(suite,r['path'])).open('rb') as f:add(tar,TOP+'suite/'+r['path'],f,r['bytes'])
    verify_archive(partial,sha(partial))
    for r in rows:
        if sha(regular(target(suite,r['path'])))!=r['sha256']:raise ValueError('Original result changed during migration packaging')
    if verify_scientific_suite(root,suite)!=checked:raise ValueError('Clinical verification changed during packaging')
    os.link(partial,output);partial.unlink()
    return dict(status='migration_archive_ready_originals_preserved',archive=str(output),archive_sha256=sha(output),bytes=output.stat().st_size,
                full_original_archive_sha256=archive_sha,included_payloads=len(included),omitted_last_checkpoints=3,
                omitted_bytes=sum(r['bytes'] for r in omitted),scientific_bytes_changed=False,model_inference_performed=False)


def verify_archive(archive,expected):
    if not HEX.fullmatch(expected) or sha(regular(archive))!=expected:raise ValueError('Migration archive checksum mismatch')
    with tarfile.open(archive,'r:gz') as tar:
        index={};values={};actual={};total=0
        for m in tar:
            safe(m.name)
            if not m.name.startswith(TOP) or not m.isfile() or m.issparse() or m.name in index or m.size<0:raise ValueError('Unsafe migration TAR member')
            total+=m.size
            if total>MAX_BYTES:raise ValueError('Migration archive payload limit exceeded')
            index[m.name]=m
            with tar.extractfile(m) as f:
                if m.name in CONTROL:
                    if m.size>32*1024**2:raise ValueError('Oversized migration control')
                    values[m.name]=f.read();h=hashlib.sha256(values[m.name]).hexdigest()
                else:h=digest(f)
            actual[m.name]=dict(bytes=m.size,sha256=h)
        if set(values)!=CONTROL:raise ValueError('Missing migration controls')
        if any(str(p) in index for n in index for p in PurePosixPath(n).parents):raise ValueError('Migration file/directory collision')
        manifest=parse(values[TOP+'manifest.json']);original=parse(values[TOP+'original_artifact_manifest.json']);receipt=parse(values[TOP+'original_retrieval_receipt.json'])
        if (manifest.get('schema')!='completed_clinical_migration_v1' or manifest.get('scientific_payload_bytes_changed') is not False
                or manifest.get('source_files_deleted') is not False or manifest.get('model_inference_performed') is not False):raise ValueError('Invalid migration scope')
        if (actual[TOP+'original_artifact_manifest.json']['sha256']!=manifest['original_artifact_manifest_sha256']
                or actual[TOP+'original_retrieval_receipt.json']['sha256']!=manifest['original_retrieval_receipt_sha256']
                or receipt.get('archive_sha256')!=manifest['source_archive_sha256'] or receipt.get('destination_readback_verified') is not True):raise ValueError('Original receipt provenance changed')
        original_records={str(safe(r['path']).relative_to('kevin_results')):dict(path=str(PurePosixPath(r['path']).relative_to('kevin_results')),bytes=r['size'],sha256=r['sha256']) for r in original['files']}
        included={r['path']:r for r in manifest['files']};omitted={r['path']:r for r in manifest['omitted']}
        if (original.get('version')!=1 or len(original_records)!=len(original['files'])
                or len(included)!=len(manifest['files']) or len(omitted)!=len(manifest['omitted']) or len(omitted)!=3 or set(omitted)!=OMIT
                or set(included)&set(omitted) or original_records!={**included,**omitted}
                or len(original_records)!=manifest['original_payload_files'] or len(original_records)!=receipt['verified_payload_files']):raise ValueError('Only exact three last.pt files may be omitted')
        if set(index)!=CONTROL|{TOP+'suite/'+n for n in included}:raise ValueError('Migration payload membership mismatch')
        for n,r in included.items():
            safe(n)
            if actual[TOP+'suite/'+n]!={'bytes':r['bytes'],'sha256':r['sha256']}:raise ValueError('Migrated payload bytes changed')
    return manifest,values


def install(archive,expected,root,suite,evidence):
    root=root.resolve(strict=True);suite=suite.absolute();evidence=evidence.absolute()
    if any(p.exists() or p.is_symlink() for p in [suite,evidence]) or suite.resolve().is_relative_to(root) or evidence.resolve().is_relative_to(suite.resolve()):raise ValueError('Use new suite and separate evidence directories outside repository')
    for p in [suite,evidence]:
        for parent in p.parents:
            if parent.is_symlink():raise ValueError('Destination parent symlink refused')
    manifest,values=verify_archive(archive,expected)
    for n,h in PINS.items():
        if sha(regular(target(root,n)))!=h:raise ValueError('Install frozen clinical development inputs first')
    suite.mkdir(parents=True,mode=0o700)
    with tarfile.open(archive,'r:gz') as tar:
        for r in manifest['files']:
            p=target(suite,r['path']);p.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
            with tar.extractfile(TOP+'suite/'+r['path']) as src,p.open('xb') as dst:
                for block in iter(lambda:src.read(1048576),b''):dst.write(block)
    for r in manifest['files']:
        p=regular(target(suite,r['path']))
        if p.stat().st_size!=r['bytes'] or sha(p)!=r['sha256']:raise ValueError('Migration destination readback failed')
    checked=verify_scientific_suite(root,suite)
    if checked!=manifest['source_verification']:raise ValueError('Source/destination clinical verification differs')
    if sha(archive)!=expected:raise ValueError('Migration archive changed during installation')
    receipt=dict(status='completed_clinical_suite_migrated_and_reverified',migration_archive_sha256=expected,
        full_original_archive_sha256=manifest['source_archive_sha256'],payloads=len(manifest['files']),omitted_last_checkpoints=3,
        all_destination_bytes_verified=True,unchanged_verifier_passed=True,scientific_payload_bytes_changed=False,
        source_verification=checked,model_inference_performed=False)
    evidence.mkdir(parents=True,mode=0o700)
    for n,value in values.items():
        with (evidence/PurePosixPath(n).name).open('xb') as f:f.write(value)
    (evidence/'destination_verification.json').write_bytes(encoded(receipt))
    return receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('build')
    for n in ['root','suite','receipt','checksum','output']:p.add_argument('--'+n,type=Path,required=True)
    p=sub.add_parser('verify');p.add_argument('--archive',type=Path,required=True);p.add_argument('--expected-sha256',required=True)
    p=sub.add_parser('install')
    for n in ['archive','root','suite','evidence']:p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--expected-sha256',required=True);a=parser.parse_args()
    if a.command=='build':result=build(a.root,a.suite,a.receipt,a.checksum,a.output)
    elif a.command=='install':result=install(a.archive,a.expected_sha256,a.root,a.suite,a.evidence)
    else:
        m,_=verify_archive(a.archive,a.expected_sha256);result=dict(status='migration_archive_verified',payloads=len(m['files']),omitted_last_checkpoints=3)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
