#!/usr/bin/env python3
"""Acquire public provider archive only; preserve an untouched evaluation candidate.
No model inference, preprocessing, data splitting or archive extraction occurs.
"""
import csv,hashlib,json,os,stat,time,urllib.request,zipfile
from datetime import datetime,timezone
from pathlib import Path,PurePosixPath
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
OUT=ROOT/'data/external/alzaset_zenodo_17084199';RAW=OUT/'raw';RAW.mkdir(parents=True,exist_ok=True)
URL='https://zenodo.org/api/records/17084199/files/AlzaSet.zip/content'
EXPECTED_MD5='c62f001cb24bb7eaa344a37fde32d6c8';EXPECTED_BYTES=314906645

def digest(path,kind):
 h=hashlib.new(kind)
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()

def main():
 metadata=json.loads((HERE/'access_metadata/alzaset_zenodo_api.json').read_text())
 files=json.loads((HERE/'access_metadata/alzaset_files_api.json').read_text())
 html=(HERE/'access_metadata/alzaset_record_html.html').read_text()
 if metadata['metadata']['access_right']!='open' or metadata['metadata']['license']['id']!='cc-by-4.0':raise ValueError('Provider not openly licensed')
 if not files['enabled'] or files['entries'][0]['access']['hidden']:raise ValueError('Files not public')
 if '"embargo": {"active": false' not in html or '"files": "public"' not in html:raise ValueError('Provider record does not explicitly confirm inactive embargo/public files')
 final=RAW/'AlzaSet.zip';started=time.monotonic();downloaded=False
 if not final.exists():
  part=RAW/'AlzaSet.zip.part';md5=hashlib.md5();sha=hashlib.sha256();n=0
  request=urllib.request.Request(URL,headers={'User-Agent':'AcademicResearchDatasetAcquisition/1.0'})
  with urllib.request.urlopen(request,timeout=60) as response,part.open('wb') as out:
   if response.status!=200:raise ValueError('Not a public successful response')
   for block in iter(lambda:response.read(1048576),b''):
    out.write(block);md5.update(block);sha.update(block);n+=len(block)
    if n>EXPECTED_BYTES:raise ValueError('Provider size mismatch')
    if n%(25*1048576)==0:print(f'Downloaded {n}/{EXPECTED_BYTES} bytes',flush=True)
  if n!=EXPECTED_BYTES or md5.hexdigest()!=EXPECTED_MD5:raise ValueError('Provider checksum mismatch')
  os.replace(part,final);stream_sha=sha.hexdigest();downloaded=True
 else:stream_sha=digest(final,'sha256')
 actual_md5=digest(final,'md5');actual_sha=digest(final,'sha256')
 if final.stat().st_size!=EXPECTED_BYTES or actual_md5!=EXPECTED_MD5 or actual_sha!=stream_sha:raise ValueError('Destination readback mismatch')
 members=[];small_text={}
 with zipfile.ZipFile(final) as z:
  for info in z.infolist():
   path=PurePosixPath(info.filename)
   if path.is_absolute() or '..' in path.parts or stat.S_ISLNK(info.external_attr>>16):raise ValueError('Unsafe archive member')
   members.append(dict(path=info.filename,bytes=info.file_size,compressed_bytes=info.compress_size,crc32=f'{info.CRC:08x}',is_directory=info.is_dir()))
   if path.suffix.lower() in ['.csv','.tsv','.json','.txt','.md'] and info.file_size<=262144:
    small_text[info.filename]=z.read(info).decode('utf-8',errors='replace')
  if z.testzip() is not None:raise ValueError('ZIP CRC verification failed')
 with (OUT/'archive_members.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=members[0].keys());w.writeheader();w.writerows(members)
 (OUT/'small_metadata_files.json').write_text(json.dumps(small_text,indent=2)+'\n')
 report=dict(status='downloaded_and_verified' if downloaded else 'existing_archive_reverified',checked_utc=datetime.now(timezone.utc).isoformat(),provider_concept_doi='10.5281/zenodo.15564968',provider_version_doi='10.5281/zenodo.17084199',url=URL,license='CC-BY-4.0',provider_open_record_and_public_files_confirmed=True,embargo_active=False,provider_publication_date='2027-01-01',date_discrepancy='Current provider openly serves the archive with inactive embargo despite a future publication date; 2025 paper described an embargo. No restriction was bypassed.',bytes=EXPECTED_BYTES,provider_md5=EXPECTED_MD5,readback_md5=actual_md5,sha256=actual_sha,zip_crc_verified=True,member_count=len(members),file_count=sum(not m['is_directory'] for m in members),uncompressed_bytes=sum(m['bytes'] for m in members),small_metadata_names=list(small_text),seconds=time.monotonic()-started,role='Untouched candidate external cohort pending patient/label/provenance audit and separately frozen clinical analysis protocol',model_inference_performed=False,preprocessing_performed=False,extraction_performed=False)
 (OUT/'acquisition.json').write_text(json.dumps(report,indent=2)+'\n');(HERE/'alzaset_acquisition_summary.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps(report,indent=2));print('FIRST MEMBERS',json.dumps(members[:30],indent=2))
if __name__=='__main__':main()
