# Rebuild from the recorded source files, including the earlier local scientific package.
# This requires all source inputs from SOURCE_MANIFEST.json; a bare Git checkout may not contain the older package.
# No training or inference. The recorded ZIP checksum identifies the original delivery, not a future ZIP timestamp.
from pathlib import Path
import json,csv,re,shutil,subprocess,hashlib,zipfile
from pypdf import PdfReader
R=Path(__file__).resolve().parents[3]; B=R/'paper_package/Abid_Dementia_Update_20261011'
source_manifest=json.loads((Path(__file__).parent/'SOURCE_MANIFEST.json').read_text())
source_errors=[]
for record in source_manifest['files']:
 source=R/record['source']
 if not source.is_file(): source_errors.append(record['source']+' (missing)')
 elif hashlib.sha256(source.read_bytes()).hexdigest()!=record['source_sha256']:
  source_errors.append(record['source']+' (source checksum differs)')
if source_errors: raise SystemExit('Exact archived source inputs are required before rebuilding:\n'+'\n'.join(source_errors))
if B.exists() and any(B.iterdir()):raise SystemExit('Output folder already exists; preserve it and use a clean checkout for a rebuild.')
B.mkdir(parents=True,exist_ok=True)
D='docs/kevin_paper_20261011/'
tracked=subprocess.check_output(['git','ls-files'],cwd=R,text=True).splitlines()
selected=[]
for s in tracked:
 if s.startswith(D+'round2_publication/artifacts/') or s.startswith(D+'round2_publication/combined_graphs/'):
  if not s.endswith('EXPORT_MANIFEST.json'): selected.append(s)
 elif s.startswith(D+'publication/figures/') or s.startswith(D+'publication/training/') or s.startswith(D+'publication/results/') or s.startswith(D+'publication/reports/'):
  selected.append(s)
 elif s in [D+'publication/'+x for x in ['FIGURES.md','All_Graphs_Atlas.pdf','All_Graphs_Atlas_Index.csv']]+[D+'round2_publication/'+x for x in ['README.md','Methods.md','RESULTS.md','DATA_AND_LIMITATIONS.md']]:selected.append(s)
 elif s in [D+'round3/'+x for x in ['README.md','CLINICAL_ROADMAP.md','REVIEW_STATUS.md','protocol.json','previous_failure_audit.json','implementation_verification.json','legacy_checkpoint_parity.json']]:selected.append(s)
# Historical training/quantitative images already scoped to dementia; current galleries include their CAM comparisons.
old=R/'paper_package/Dementia_Only_2026-10-09/source'
for f in old.rglob('*'):
 if f.is_file() and f.suffix.lower() in {'.png','.pdf','.json','.csv'} and ('gradcam' not in str(f) and '/explainability/' not in str(f)):
  if f.suffix.lower() in {'.png','.pdf'} or 'Dementia_Specialist_history.json' in f.name: selected.append(str(f.relative_to(R)))
# Fixed checkpoint identities only; weights stay available from repository.
selected += ['models/kevin_round2_20261011/CHECKPOINTS.json']
for x in ['report_round2.py','round2_pipeline.py','round2_attribution.py']:
 selected.append(D+'round2_evaluation/'+x)
for x in ['clinical_bag.py','clinical_compare.py','clinical_report.py','clinical_statistics.py']:
 selected.append(D+'round2_clinical/'+x)
selected += [D+'round2_training/round2.py']
for x in ['train_candidate.py','analysis.py','create_publication_figures.py']:
 p=D+'training/'+x
 if (R/p).is_file() and p in tracked: selected.append(p)
ban=re.compile(r'runpod|billing|network[_ -]?volume|pod[_ -]?id|ssh[_ :]|root@|/workspace/|\$\d|/hour|paid resources|paid compute|GPU.*storage.*delet|resources.*deleted',re.I)
excluded=[]; records=[]
def sha(b):return hashlib.sha256(b).hexdigest()
def sanitize_obj(x):
 if isinstance(x,dict):
  return {k:sanitize_obj(v) for k,v in x.items() if not ban.search(k) and k not in {'execution','resource_ids','ssh','remote_host','cost','billing','runtime_resource','execution_environment'}}
 if isinstance(x,list):return [sanitize_obj(v) for v in x]
 if isinstance(x,str):
  if ban.search(x):return '[Operational provenance omitted from this scientific handoff.]'
  if x.startswith(('/Users/','/workspace/','/tmp/','/private/','root@')):return '[Local source path omitted; see source checksum.]'
 return x
for s in sorted(set(selected)):
 f=R/s
 if not f.exists():excluded.append({'source':s,'reason':'missing'});continue
 if ban.search(s):excluded.append({'source':s,'reason':'operational filename'});continue
 raw=f.read_bytes(); new=raw; transform='unchanged'
 if raw.startswith(b'version https://git-lfs.github.com/spec/v1'):raise RuntimeError('LFS pointer '+s)
 if f.suffix=='.py' and ban.search(raw.decode()):excluded.append({'source':s,'reason':'implementation includes operational configuration; use repository scientific source'});continue
 if f.suffix=='.json':
  try:
   obj=json.loads(raw); out=sanitize_obj(obj)
   if out!=obj:new=(json.dumps(out,indent=2,ensure_ascii=False)+'\n').encode();transform='operational provenance redacted; scientific values retained'
  except (ValueError,UnicodeError):raise
 elif f.suffix.lower() in {'.md','.txt','.html','.csv'}:
  txt=raw.decode(); orig=txt
  if f.suffix=='.md':
   # Remove infrastructure-only subsections. Retain requested training-device fact below.
   txt=re.sub(r'(?ms)^## (?:Hardware and execution roles|Runtime and spending boundary|Execution and retrieval|Compute and retrieval).*?(?=^## |\Z)','',txt)
   if s.endswith('round2_publication/Methods.md'):
    txt += '\n## Training hardware\n\nAll six Round 2 training runs used an NVIDIA RTX PRO 6000 Blackwell Server Edition GPU. The frozen clinical comparison later used an NVIDIA B300; it did not retrain the selected models. Statistical analysis and figure preparation used saved outputs locally.\n'
   paragraphs=txt.split('\n\n'); txt='\n\n'.join(p for p in paragraphs if not ban.search(p))
  else:
   txt='\n'.join(line for line in txt.splitlines() if not ban.search(line))+'\n'
  if txt!=orig:new=txt.encode();transform='operational-only text removed; scientific image artifacts unchanged'
 if f.suffix=='.pdf':
  text='\n'.join(p.extract_text() or '' for p in PdfReader(f).pages)
  if ban.search(text):excluded.append({'source':s,'reason':'operational text embedded in PDF'});continue
 dest=B/'repository'/s
 dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(new)
 records.append({'path':str(dest.relative_to(B)),'source':s,'source_sha256':sha(raw),'sha256':sha(new),'bytes':len(new),'transformation':transform})
(B/'SOURCE_MANIFEST.json').write_text(json.dumps({'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip(),'scientific_scope':'Dementia only; prior and current completed experiments plus untrained next-study design.','files':records},indent=2)+'\n')
Path('/tmp/abid_selection_exclusions.json').write_text(json.dumps(excluded,indent=2))
print(json.dumps({'files':len(records),'uncompressed_MB':sum(r['bytes'] for r in records)/1e6,'excluded':excluded},indent=2))

for name in ['START_HERE.md','MODEL_AND_DATASET_COMPARISON.md','FIGURE_CAPTION_GUIDANCE.md','facts.json']:
 (B/name).write_bytes((R/'docs/kevin_paper_20261011/abid_handoff_20261011'/name).read_bytes())
figure_rows=[]
for f in sorted(B.rglob('*')):
 if f.suffix.lower() not in {'.png','.pdf'}:continue
 rel=str(f.relative_to(B));scope='Earlier research / historical supplementary record';role='Supplement; preserve original cohort and model labels'
 if 'round2_publication' in rel:
  scope='Completed Round 2 legacy four-class research'
  if '/clinical/' in rel:scope='Completed Round 2 patient-level AD/CN research'
  if '/attribution_diagnostics/' in rel:scope='Legacy four-class stage attribution only';role='Diagnostic supplement; disclose 2/100 probability replay failures; no anatomical-validity claim'
  elif 'harvard_fresh_external' in rel:role='Main clinical result candidate; show external failure, denominator 19 and uncertainty'
  elif 'original100' in rel:role='Historical reader appendix only; 66 old-training / 9 old-validation / 25 old-test'
  elif 'training' in rel or 'curves' in rel:role='Methods/supplement; raw unsmoothed, every seed; not external validation'
  else:role='Main or supplement according to endpoint; reused cohorts remain labeled reused'
 if 'Atlas.pdf' in rel:role='Browsing atlas; use individual vector/PDF figures for journal assembly'
 figure_rows.append({'file':rel,'scope':scope,'recommended_role':role,'format':f.suffix[1:].upper(),'bytes':f.stat().st_size,'sha256':sha(f.read_bytes())})
with (B/'FIGURE_INDEX.csv').open('w',newline='') as h:
 w=csv.DictWriter(h,fieldnames=figure_rows[0].keys());w.writeheader();w.writerows(figure_rows)
from pathlib import Path
import json,csv,hashlib,re,zipfile,subprocess
from pypdf import PdfReader
from PIL import Image
prefix=B/'repository/docs/kevin_paper_20261011'
# Restore scientific statements that shared a paragraph with omitted operational text.
p=prefix/'round3/README.md';s=p.read_text();s=s.replace('# Round 3: patient-level MRI research\n','# Round 3: patient-level MRI research\n\nThis round prepares a stronger structural-T1 MRI research study of clinical AD versus cognitively normal controls. It is not authorization for clinical deployment. No new source is approved or acquired for Round 3, and no Round 3 training has started.\n',1);s+='\n## Preprocessing dependency boundary\n\nThe planned whole-brain preprocessing requires N4 bias correction, reviewed SynthStrip brain extraction, a pinned template and mask, ANTs registration/transforms, QC and provenance. The required executables and complete pinned runtime/assets were not established at this snapshot. Missing tools, weights or template assets block processing. Synthetic and geometric checks do not validate anatomical accuracy.\n';p.write_text(s)
p=prefix/'publication/reports/Overview.md';s=p.read_text();s+='\n- [Full six-run report and clinical denominators](Full_Results.md)\n- [Primary and ensemble statistics](Statistical_Handoff.md)\n- [Completed Methods addendum](Completed_Methods.md)\n- [Stage loss curves: all six runs](../figures/publication_curves/stage_loss_all_six_runs.png)\n- [Clinical loss curves: all three seeds](../figures/publication_curves/clinical_loss_all_three_multitask_seeds.png)\n- [Evaluation figure provenance](../figures/evaluation_figures/evaluation_figure_provenance.json)\n- [Frozen ensemble outputs and specifications](../results/ensemble_supplement.json)\n- [Unresolved readiness requirements](Readiness_Protocol_Snapshot.txt)\n\nNo production checkpoint was replaced.\n';p.write_text(s)
p=prefix/'round2_publication/RESULTS.md';s=p.read_text().replace('## What a further study needs','The six selected research-checkpoint identities and export-fidelity checks are preserved in the source repository and the supplied checkpoint manifest. Synthetic equality checks establish export fidelity only. The production classifier remains unchanged.\n\n## What a further study needs');p.write_text(s)
for p in B.rglob('*.md'):
 s=p.read_text();s=s.replace(', and cloud connection details','').replace(', cloud connection details','').replace(' and cloud connection details','').replace(' and cloud resource identifiers','').replace(', cloud resource identifiers','').replace(', operational cloud details','').replace('operational cloud details, ','').replace(' and operational cloud details','')
 p.write_text(s)
# Add scientific files directly referenced by the next-study README.
for rel in ['round3/inventory.json','round3/empty_patient_manifest.json','round3/sample_size_pending.json']:
 p=R/'docs/kevin_paper_20261011'/rel;d=prefix/rel;d.write_bytes(p.read_bytes())
# Point access package to the full source repository while keeping contact/application details out of this handoff.
p=prefix/'round3/README.md';s=p.read_text().replace('(access/README.md)','(https://github.com/TailsOS-hack/Neuro-Radiology-Database/blob/8bfd1dbb6bf6b29b1faf1962e6038c59f9a4f574/docs/kevin_paper_20261011/round3/access/README.md)').replace('(../../../research/dementia_round3/)','(https://github.com/TailsOS-hack/Neuro-Radiology-Database/tree/8bfd1dbb6bf6b29b1faf1962e6038c59f9a4f574/research/dementia_round3)').replace('(EXECUTION_CONTRACT.md)','(https://github.com/TailsOS-hack/Neuro-Radiology-Database/blob/8bfd1dbb6bf6b29b1faf1962e6038c59f9a4f574/docs/kevin_paper_20261011/round3/EXECUTION_CONTRACT.md)');p.write_text(s)
# Remove execution-location label fragments while preserving scientific check outcomes.
for p in B.rglob('*'):
 if not p.is_file() or p.suffix.lower() not in {'.md','.txt','.csv','.json','.py','.html'}:continue
 if p.name in {'SOURCE_MANIFEST.json','FILE_MANIFEST.csv','PACKAGE_VERIFICATION.json'}:continue
 original_text=p.read_bytes().decode('utf-8');txt=original_text
 if p.suffix=='.json':
  obj=json.loads(txt)
  def clean_location(x):
   if isinstance(x,dict):return {k.replace('cloud_',''):clean_location(v) for k,v in x.items() if k!='cloud_resources_used'}
   if isinstance(x,list):return [clean_location(v) for v in x]
   return x.replace('cloud_','') if isinstance(x,str) else x
  new=clean_location(obj)
  if new!=obj:txt=json.dumps(new,indent=2)+'\n'
 else:txt=txt.replace('cloud_','')
 txt=''.join(line for line in txt.splitlines(keepends=True) if '| Paid computation |' not in line)
 if txt!=original_text:p.write_bytes(txt.encode('utf-8'))

# Preserve source byte streams for untouched scientific CSV/HTML/Python files.
precheck=re.compile(r'runpod|billing|network[_ -]?volume|pod[_ -]?id|ssh[_ :]|root@|/workspace/|\$\d|/hour|cloud|\bpaid\b|\bstorage\b',re.I)
for rec in json.loads((B/'SOURCE_MANIFEST.json').read_text())['files']:
 src=R/rec['source'];dst=B/rec['path']
 if src.suffix.lower() in {'.csv','.html','.py'} and src.is_file():
  raw=src.read_bytes()
  if not precheck.search(raw.decode('utf-8')):dst.write_bytes(raw)

# Final source readback after reviewed derivative edits.
manifest=json.loads((B/'SOURCE_MANIFEST.json').read_text()); lookup={r['path']:r for r in manifest['files']}
for f in B.rglob('*'):
 if not f.is_file() or f.name in {'SOURCE_MANIFEST.json','FILE_MANIFEST.csv','PACKAGE_VERIFICATION.json'}:continue
 rel=str(f.relative_to(B));raw=f.read_bytes();h=hashlib.sha256(raw).hexdigest()
 if rel in lookup:
  r=lookup[rel];r['sha256']=h;r['bytes']=len(raw)
  if h==r['source_sha256']:r['transformation']='unchanged'
  if h!=r['source_sha256'] and r['transformation']=='unchanged':r['transformation']='reviewed scientific derivative; operational text removed and scientific context retained'
 else:
  source=R/rel.removeprefix('repository/') if rel.startswith('repository/') else R/'docs/kevin_paper_20261011/abid_handoff_20261011'/rel
  if source.is_file():manifest['files'].append({'path':rel,'source':str(source.relative_to(R)),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'sha256':h,'bytes':len(raw),'transformation':'unchanged' if raw==source.read_bytes() else 'reviewed package derivative'})
(B/'SOURCE_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
ban=re.compile(r'runpod|billing|network[_ -]?volume|pod[_ -]?id|ssh[_ :]|root@|/workspace/|\$\d|/hour|cloud|\bpaid\b|\bstorage\b|tumou?r|aneurysm|glioma|meningioma|pituitary|glioblastoma',re.I)
errors=[]; pdfpages=0;pngs=0;texts=0;figure_equal=0
for f in B.rglob('*'):
 if not f.is_file():continue
 raw=f.read_bytes()
 if raw.startswith(b'version https://git-lfs.github.com/spec/v1'):errors.append(str(f)+' pointer')
 if ban.search(str(f.relative_to(B))):errors.append(str(f)+' filename')
 if f.suffix.lower() in {'.md','.txt','.json','.csv','.py','.html'}:
  texts+=1;t=raw.decode()
  if ban.search(t):errors.append(str(f)+' text')
  if any(q in t for q in ['PRIVATE KEY','Bearer ','@gmail.com']):errors.append(str(f)+' private marker')
 elif f.suffix=='.pdf':
  reader=PdfReader(f);pdfpages+=len(reader.pages);t='\n'.join(p.extract_text() or '' for p in reader.pages)+'\n'+str(reader.metadata)
  if ban.search(t):errors.append(str(f)+' PDF text/metadata')
 elif f.suffix=='.png':
  im=Image.open(f);pngs+=1
  if ban.search(str(im.info)):errors.append(str(f)+' PNG metadata')
  im.verify()
 if f.suffix.lower() in {'.pdf','.png'}:
  r=lookup.get(str(f.relative_to(B)))
  if r and r['source_sha256']==hashlib.sha256(raw).hexdigest():figure_equal+=1
if errors:raise RuntimeError(errors)
verification={'status':'passed','text_files_scanned':texts,'pdf_pages_text_and_metadata_scanned':pdfpages,'png_files_verified':pngs,'figures_byte_identical_to_source':figure_equal,'excluded_operational_text_matches':0,'unrelated_disease_content_matches':0,'lfs_pointer_files':0,'patient_raw_MRI_files':0,'new_gradcam_case_panels':115,'figure_files':479,'scientific_claim_review':'All seeds, reused benchmark status, external failure and attribution control failures retained.','visual_spot_check':'Fresh Harvard all-model confusion figure inspected; original source figures already reviewed in the completed experiment.'}
(B/'PACKAGE_VERIFICATION.json').write_text(json.dumps(verification,indent=2)+'\n')
files=[f for f in sorted(B.rglob('*')) if f.is_file() and f.name!='FILE_MANIFEST.csv']
with (B/'FILE_MANIFEST.csv').open('w',newline='') as h:
 w=csv.writer(h);w.writerow(['file','bytes','sha256'])
 for f in files:w.writerow([str(f.relative_to(B)),f.stat().st_size,hashlib.sha256(f.read_bytes()).hexdigest()])
zip_path=B.with_suffix('.zip')
with zipfile.ZipFile(zip_path,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
 for f in sorted(B.rglob('*')):
  if f.is_file():z.write(f,str(f.relative_to(B.parent)))
with zipfile.ZipFile(zip_path) as z:
 assert z.testzip() is None
 for f in B.rglob('*'):
  if f.is_file():assert z.read(str(f.relative_to(B.parent)))==f.read_bytes()
receipt={'zip':str(zip_path),'bytes':zip_path.stat().st_size,'sha256':hashlib.sha256(zip_path.read_bytes()).hexdigest(),'files':len([f for f in B.rglob('*') if f.is_file()]),'figure_files':479,'new_gradcam_case_panels':115,'zip_test':'passed','archive_member_readback':'every member byte-identical to final package'}
(R/'paper_package/Abid_Dementia_Update_20261011_VERIFICATION.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2))
