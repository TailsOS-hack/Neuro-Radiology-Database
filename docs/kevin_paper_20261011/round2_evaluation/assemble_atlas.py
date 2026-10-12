#!/usr/bin/env python3
"""Combine reviewed round2 graph PDFs; do not modify any source figure."""
import argparse,csv,hashlib,json
from pathlib import Path
from pypdf import PdfReader,PdfWriter


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--report',type=Path,required=True);ap.add_argument('--attribution',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args();output=args.output.resolve()
    if output.exists():raise ValueError('Atlas output must be new')
    records=[];writer=PdfWriter();page=0
    for group,directory,marker,n in [('Quantitative and robustness',args.report,'report_complete.json',28),('Attribution diagnostics',args.attribution,'attribution_complete.json',2)]:
        complete=json.loads((directory/marker).read_text());manifest=json.loads((directory/'SHA256SUMS.json').read_text())
        if complete['status']!='complete_numerically_verified_visual_review_required' or complete['manifest_sha256']!=sha(directory/'SHA256SUMS.json'):raise ValueError('Incomplete source report or blocked replay')
        files=sorted(directory.glob('*.pdf'))
        if len(files)!=n:raise ValueError('Unexpected graph count')
        parent=writer.add_outline_item(group,page)
        for f in files:
            if sha(f)!=manifest[f.name]:raise ValueError('Source figure changed')
            title=f.stem.replace('_',' ').capitalize();reader=PdfReader(f)
            if len(reader.pages)!=1:raise ValueError('Expected one-page source graph')
            writer.add_page(reader.pages[0]);writer.add_outline_item(title,page,parent=parent);page+=1
            records.append(dict(page=page,group=group,title=title,source=str(f.resolve()),source_sha256=sha(f)))
    if len(list((args.attribution/'cases').glob('*.png')))!=115:raise ValueError('Incomplete115-case gallery')
    output.mkdir(parents=True)
    with (output/'Round2_Graphs_Atlas.pdf').open('xb') as f:writer.write(f)
    if len(PdfReader(output/'Round2_Graphs_Atlas.pdf').pages)!=30:raise ValueError('Atlas readback page count differs')
    with (output/'atlas_index.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    (output/'README.md').write_text('# Round2 graph atlas\n\n[All30 graph pages](Round2_Graphs_Atlas.pdf) include the raw unsmoothed learning curves and every fixed comparison. [Page index](atlas_index.csv).\n\n[All115 fixed Grad-CAM cases]('+str((args.attribution/'gallery.html').resolve())+') are linked separately. This assembly confirms checksums and page boundaries; source plots and case panels still require documented human visual review. Historical round1 figures remain in their unchanged separate atlas.\n')
    result=dict(pages=30,bookmarks=32,source_pdf_count=30,gradcam_case_count=115,atlas_sha256=sha(output/'Round2_Graphs_Atlas.pdf'),sources=records,visual_review_not_implied_by_assembly=True)
    (output/'atlas_verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='sources'}))

if __name__=='__main__':main()
