#!/usr/bin/env python3
"""Read-only public access metadata checks; never download patient images."""
import hashlib,json,urllib.request,urllib.error
from pathlib import Path
from datetime import datetime,timezone
OUT=Path(__file__).resolve().parent/'access_metadata';OUT.mkdir(exist_ok=True)
URLS={
'alzaset_zenodo_api':'https://zenodo.org/api/records/15564968',
'alzaset_files_api':'https://zenodo.org/api/records/17084199/files',
'alzaset_record_html':'https://zenodo.org/records/17084199',
'brainlat_request_form':'https://docs.google.com/forms/d/e/1FAIpQLSfSH-vHFvfIg4G0xCQlpfOGRHU4AutftzTQLLTp9GYtMk99aA/viewform',
'miriad_original':'https://www.ucl.ac.uk/drc/research/miriad/Database',
'miriad_xnat':'https://miriad.drc.ion.ucl.ac.uk/',
'neugrid_portfolio':'https://www.neugrid2.eu/index.php/data-portfolio/',
'miriad_ucl':'https://www.ucl.ac.uk/ion/research/research-departments/dementia-research-centre/research/research-methods/minimal-interval-resonance-imaging-alzheimers-disease-miriad',
'brainlat_wiki':'https://repo-prod.prod.sagebase.org/repo/v1/entity/syn51549340/wiki/624187',
'brainlat_permissions':'https://repo-prod.prod.sagebase.org/repo/v1/entity/syn53972968/permissions',
'brainlat_access_requirements':'https://repo-prod.prod.sagebase.org/repo/v1/entity/syn53972968/accessRequirement',
}
results=[]
for name,url in URLS.items():
 try:
  req=urllib.request.Request(url,headers={'User-Agent':'AcademicResearchMetadataCheck/1.0'})
  with urllib.request.urlopen(req,timeout=35) as response:
   payload=response.read(2_000_001)
   if len(payload)>2_000_000:raise ValueError('Metadata response too large; stopped')
   contenttype=response.headers.get('content-type','');actual=response.url;status=response.status
  suffix='.json' if 'json' in contenttype else '.html'
  file=OUT/(name+suffix);file.write_bytes(payload)
  results.append(dict(name=name,url=url,resolved_url=actual,status=status,file=file.name,bytes=len(payload),sha256=hashlib.sha256(payload).hexdigest()))
 except Exception as e:results.append(dict(name=name,url=url,error=f'{type(e).__name__}: {e}'))
(OUT/'request_log.json').write_text(json.dumps(dict(checked_utc=datetime.now(timezone.utc).isoformat(),requests=results),indent=2)+'\n')
print(json.dumps(results,indent=2))
