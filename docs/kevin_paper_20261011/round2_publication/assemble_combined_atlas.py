#!/usr/bin/env python3
"""Combine already reviewed stage and clinical graph PDFs without changing pages."""
import argparse,csv,hashlib,json
from pathlib import Path
from pypdf import PdfReader,PdfWriter
import subprocess,tempfile
from PIL import Image

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--stage-atlas',type=Path,required=True);p.add_argument('--clinical-report',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--stage-qa',type=Path,required=True);p.add_argument('--clinical-qa',type=Path,required=True);a=p.parse_args()
 if a.output.exists():raise ValueError('Preserve existing atlas directory')
 clinical=[a.clinical_report/'all_training_curves.pdf']
 for cohort in ['harvard_fresh_external','besga_reused_test','openneuro_reused_external']:
  for kind in ['confusion','exact_intervals','roc_pr','reliability','confidence','all_stress']:clinical.append(a.clinical_report/f'{cohort}_{kind}.pdf')
 sources=[a.stage_atlas]+clinical
 qa=[json.loads(x.read_text()) for x in [a.stage_qa,a.clinical_qa]]
 assert all(x['status']=='passed_numeric_and_visual_review' for x in qa)
 reviewed=set().union(*(set(x['reviewed_files'].values()) for x in qa))
 assert all(sha(f) in reviewed for f in sources), 'Every source PDF must have an exact visual review hash'
 assert len(clinical)==19 and len(PdfReader(a.stage_atlas).pages)==30
 a.output.mkdir(parents=True);writer=PdfWriter();index=[]
 for f in sources:
  r=PdfReader(f);start=len(writer.pages)
  if f!=a.stage_atlas and len(r.pages)!=1:raise ValueError('Unexpected graph page count')
  writer.append(r,outline_item='Stage graphs and flagged Grad-CAM diagnostics' if f==a.stage_atlas else f.stem)
  for n in range(len(r.pages)):index.append({'page':start+n+1,'source':f.name,'source_page':n+1,'source_sha256':sha(f)})
 assert len(writer.pages)==49
 target=a.output/'Round2_All_Graphs_Atlas.pdf'
 writer.add_metadata({'/Title':'Kevin Round 2: stage comparisons, flagged Grad-CAM diagnostics, and clinical external failure','/Subject':'Research-only evidence; no clinical readiness or superiority established'})
 with target.open('xb') as f:writer.write(f)
 with (a.output/'atlas_index.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(index[0]));w.writeheader();w.writerows(index)
 checks=[];i=0
 with tempfile.TemporaryDirectory(prefix='kevin-atlas-qa-') as temp:
  temp=Path(temp)
  subprocess.run(['pdftoppm','-r','72','-png',str(target),str(temp/'dest')],check=True,capture_output=True)
  destinations=sorted(temp.glob('dest-*.png'),key=lambda p:int(p.stem.rsplit('-',1)[1]))
  assert len(destinations)==49
  for number,source in enumerate(sources):
   prefix=temp/f'source{number}'
   subprocess.run(['pdftoppm','-r','72','-png',str(source),str(prefix)],check=True,capture_output=True)
   pages=sorted(temp.glob(f'source{number}-*.png'),key=lambda p:int(p.stem.rsplit('-',1)[1]))
   for page in pages:
    with Image.open(page) as im:src=im.convert('RGB');src_size=src.size;src_bytes=src.tobytes()
    with Image.open(destinations[i]) as im:dest=im.convert('RGB');dest_size=dest.size;dest_bytes=dest.tobytes()
    assert src_size==dest_size and src_bytes==dest_bytes,('Page render changed',i+1)
    checks.append({'page':i+1,'render_sha256':hashlib.sha256(dest_bytes).hexdigest(),'source_and_destination_render_equal':True});i+=1
 record={'status':'all49_pages_rendered_and_equal_to_reviewed_sources','pages':49,'atlas_sha256':sha(target),'source_pdf_count':20,'visual_review_receipt_sha256':[sha(a.stage_qa),sha(a.clinical_qa)],'page_checks':checks,'clinical_subject_images_or_records_included':False,'attribution_replay_failure_preserved':True,'visual_qa_basis':'Source figures reviewed separately; every combined page rendered and compared pixel-for-pixel to its reviewed source PDF.'}
 (a.output/'atlas_verification.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps({k:v for k,v in record.items() if k!='page_checks'}))
if __name__=='__main__':main()
