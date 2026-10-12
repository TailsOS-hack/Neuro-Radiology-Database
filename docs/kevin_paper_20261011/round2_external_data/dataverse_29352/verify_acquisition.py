#!/usr/bin/env python3
"""Verify the original public CC0 archive downloaded through the normal browser.

No model inference, preprocessing, extraction, splitting, or network access.
Copies the completed archive into local external-data storage only after the
provider's byte count and MD5 match, then audits NIfTI membership and headers.
"""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import struct
import tarfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
OUT = ROOT / 'data/external/dataverse_29352'
EXPECTED_BYTES = 1495117504
EXPECTED_MD5 = 'e4df070acb9a15451e391b9acd00bf04'


def hashes(path):
    md5, sha = hashlib.md5(), hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            md5.update(block)
            sha.update(block)
    return dict(md5=md5.hexdigest(), sha256=sha.hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download', type=Path, required=True)
    args = parser.parse_args()
    metadata_path = HERE / 'official_dataset_metadata.json'
    version = json.loads(metadata_path.read_text())['data']['latestVersion']
    if version['versionState'] != 'RELEASED' or version['license']['rightsIdentifier'] != 'CC0-1.0':
        raise ValueError('Expected released CC0 provider version')
    archive_record = next(f for f in version['files'] if f['label'] == 'DATA.tar.gz')
    if (archive_record['restricted'] or archive_record['dataFile']['filesize'] != EXPECTED_BYTES
            or archive_record['dataFile']['checksum'] != {'type': 'MD5', 'value': EXPECTED_MD5}):
        raise ValueError('Archive access or checksum metadata changed')
    source = args.download
    if source.is_symlink() or not source.is_file() or source.stat().st_size != EXPECTED_BYTES:
        raise ValueError('Download is not a complete regular file of the provider-declared size')
    initial = hashes(source)
    if initial['md5'] != EXPECTED_MD5:
        raise ValueError('Provider archive MD5 mismatch')
    raw = OUT / 'raw'
    raw.mkdir(parents=True, exist_ok=True)
    target = raw / 'DATA.tar.gz'
    if target.exists():
        if target.is_symlink() or hashes(target) != initial:
            raise ValueError('Existing local archive conflicts')
    else:
        with source.open('rb') as src, target.open('xb') as dst:
            shutil.copyfileobj(src, dst, 1024 * 1024)
    readback = hashes(target)
    if readback != initial or target.stat().st_size != EXPECTED_BYTES:
        raise ValueError('Copied archive readback failed')

    # The provider independently lists 29 structural NIfTI files and 30 fMRI
    # files in addition to the archive. Match by bytes, not ambiguous suffixes.
    published_by_md5 = {f['dataFile']['checksum']['value']: f for f in version['files']
                        if f['dataFile']['filename'].endswith('.nii')}
    members, scans, paths = [], [], set()
    with tarfile.open(target, 'r|gz') as archive:
        for info in archive:
            p = PurePosixPath(info.name)
            if p.is_absolute() or '..' in p.parts or str(p) in paths or not (info.isfile() or info.isdir()):
                raise ValueError('Unsafe or duplicate archive path')
            paths.add(str(p))
            members.append(dict(path=str(p), bytes=info.size, is_directory=info.isdir()))
            if info.isdir():
                continue
            match = re.fullmatch(r'DATI/(AD|MCI|NCS)/(subj_\d{2})/\2_(structural|functional)\.nii', str(p))
            if not match:
                raise ValueError('Unexpected archive payload: ' + str(p))
            group, subject, modality = match.groups()
            stream = archive.extractfile(info)
            header = stream.read(352)
            endian = '<' if struct.unpack('<i', header[:4])[0] == 348 else '>'
            if struct.unpack(endian + 'i', header[:4])[0] != 348 or header[344:348] != b'n+1\x00':
                raise ValueError('Not a single-file NIfTI-1 image')
            dims = struct.unpack(endian + '8h', header[40:56])
            pixdim = struct.unpack(endian + '8f', header[76:108])
            offset = int(struct.unpack(endian + 'f', header[108:112])[0])
            bitpix = struct.unpack(endian + 'h', header[72:74])[0]
            if offset < 352 or offset > 1024 * 1024:
                raise ValueError('Unexpected NIfTI data offset')
            extensions = stream.read(offset - 352)
            voxel_count = 1
            for dim in dims[1:dims[0] + 1]:
                if dim < 1:
                    raise ValueError('Invalid NIfTI dimension')
                voxel_count *= dim
            if bitpix not in (8, 16, 32, 64) or offset + voxel_count * bitpix // 8 != info.size:
                raise ValueError('NIfTI dimensions do not match actual payload length')
            md5, sha = hashlib.md5(), hashlib.sha256()
            for block in (header, extensions):
                md5.update(block); sha.update(block)
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                md5.update(block); sha.update(block)
            published = published_by_md5.get(md5.hexdigest())
            if published and published.get('categories') != [f'DATI-{group}-{subject}']:
                raise ValueError('Provider diagnosis tag differs from archive hierarchy')
            scans.append(dict(subject_id=f'DVN29352_{group}_{subject}', provider_subject_id=subject,
                provider_group=group, diagnosis={'AD': 'probable_AD', 'MCI': 'aMCI', 'NCS': 'healthy_control'}[group],
                modality=modality, archive_path=str(p), bytes=info.size, md5=md5.hexdigest(), sha256=sha.hexdigest(),
                shape=list(dims[1:dims[0] + 1]), voxel_spacing=list(pixdim[1:dims[0] + 1]),
                nifti_extension_bytes=len(extensions),
                header_description_nonempty=bool(header[148:228].strip(b'\x00')),
                header_aux_file_nonempty=bool(header[228:252].strip(b'\x00')),
                extension_attribute_names=[s.decode() for s in re.findall(rb'atr_name\s*=\s*"([^"]+)"', extensions)],
                provider_individual_file_md5_matched=published is not None,
                provider_file_doi=published['dataFile']['persistentId'] if published else None))
    if hashes(target) != initial:
        raise ValueError('Archive changed during audit')
    structural = [s for s in scans if s['modality'] == 'structural']
    counts = Counter(s['provider_group'] for s in structural)
    if len({s['subject_id'] for s in structural}) != len(structural):
        raise ValueError('Multiple structural images per subject require a separate audit')
    for name, records in [('archive_members.csv', members)]:
        with (OUT / name).open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=records[0]); writer.writeheader(); writer.writerows(records)
    (OUT / 'scan_manifest.json').write_text(json.dumps(scans, indent=2) + '\n')
    report = dict(status='downloaded_and_verified_header_inventory', checked_utc=datetime.now(timezone.utc).isoformat(),
        dataset_doi='10.7910/DVN/29352', dataset_version='2.0', license='CC0-1.0',
        archive_file_doi=archive_record['dataFile']['persistentId'], provider_file_id=2544368,
        provider_archive_bytes=EXPECTED_BYTES, provider_md5=EXPECTED_MD5, **readback,
        access_route='Ordinary Safari public dataset download; official API metadata saved through Safari. No account, challenge solving, access bypass, or contact.',
        metadata_sha256=hashes(metadata_path)['sha256'], member_count=len(members), nifti_count=len(scans),
        structural_count=len(structural), structural_group_counts=dict(counts),
        functional_count=sum(s['modality'] == 'functional' for s in scans),
        independent_file_md5_matches=sum(s['provider_individual_file_md5_matched'] for s in scans),
        individual_structural_records=sum('structural' in f['label'] for f in version['files']),
        header_payload_lengths_verified=True, structural_shapes=sorted({str(s['shape']) for s in structural}),
        unique_structural_sha256=len({s['sha256'] for s in structural}),
        t1_weighting_basis='Provider related publication and Clark et al. specify MDEFT anatomical T1; filenames and NIfTI dimensions support structural scans, not independent sequence verification.',
        subject_mapping_basis='Original archive group/subject hierarchy, independently matched against diagnosis-tagged file MD5 where listed; subject numbers repeat across groups and must be prefixed.',
        role='Untouched candidate for a separately frozen clinical external analysis; not training, validation selection, or a four-stage label source.',
        eligibility_limits=['Small source-specific cohort; candidate AD-versus-healthy-control denominator must use available T1 counts, with MCI separate.',
            'Patient independence against unresolved legacy source cannot be proved from pseudonymous IDs alone.',
            'Age, sex, severity, reference-standard adjudication, image quality, defacing and residual header privacy still require clinical/data review.',
            'No clinical-readiness claim; acquisition is not external model validation.'],
        model_inference_performed=False, preprocessing_performed=False, extraction_performed=False,
        source_script_sha256=hashes(Path(__file__))['sha256'])
    (OUT / 'acquisition.json').write_text(json.dumps(report, indent=2) + '\n')
    (HERE / 'acquisition_summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
