#!/usr/bin/env python3
"""Read-only numeric/file QA, PDF rasterization, and combined figure atlas."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np
import pdfplumber
from PIL import Image,ImageDraw

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]


def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--figures',type=Path,default=HERE/'generated');parser.add_argument('--review-dir',type=Path);args=parser.parse_args()
    folder=args.figures.resolve();qa=args.review_dir or Path(tempfile.mkdtemp(prefix='kevin-paper-final-review-'))
    qa.mkdir(parents=True,exist_ok=True)
    meta=json.loads((folder/'figure_provenance.json').read_text());data=json.loads((folder/'plot_data.json').read_text())
    assert digest(ROOT/meta['source_script'])==meta['source_script_sha256']
    assert all(digest(ROOT/p)==sha for p,sha in meta['inputs_sha256'].items())
    assert all(digest(ROOT/r['path'])==r['sha256'] for r in meta['figures'])
    assert len(data['metric_and_calibration_values'])==27 and len(data['confidence_histograms'])==27
    for item in data['confidence_histograms']:
        point=next(v for v in data['metric_and_calibration_values'] if v['model']==item['model'] and v['cohort']==item['cohort'])
        assert sum(item['correct_counts'])==point['point']['correct']
        assert sum(item['correct_counts'])+sum(item['error_counts'])==point['n']
        assert sum(b['n'] for b in point['calibration']['bins'])==point['n']
        assert 0<=point['calibration']['multiclass_brier_sum']<=2
    with (folder/'classwise_roc_ap.csv').open() as f:curves=list(csv.DictReader(f))
    # Average precision can exceed one by machine epsilon in scikit-learn.
    # Preserve source values; allow only floating-point roundoff in the check.
    assert len(curves)==216 and all(-1e-12<=float(r['value'])<=1+1e-12 for r in curves)
    pdfs=sorted(folder.glob('*.pdf'));pngs=sorted(folder.glob('*.png'))
    assert len(pdfs)==len(pngs) and len(pdfs) in (32,34)
    png_checks=[]
    for p in pngs:
        with Image.open(p) as im:
            im.load();assert im.width>=2500 and im.height>=1000
            assert np.asarray(im.convert('RGB')).std()>10
            png_checks.append({'path':str(p.relative_to(ROOT)),'dimensions':list(im.size),'sha256':digest(p)})
    if not shutil.which('pdftoppm'):raise RuntimeError('Poppler pdftoppm required')
    def render(p):
        with pdfplumber.open(p) as doc:
            assert len(doc.pages)==1
            page=doc.pages[0];w,h=page.width,page.height;words=page.extract_words()
            assert len(words)>20
            outside=[e['text'] for e in words if e['x0'] < -.5 or e['top'] < -.5 or e['x1'] > w+.5 or e['bottom'] > h+.5]
        if outside:raise ValueError(f'{p.name}: PDF text outside page: {outside}')
        prefix=qa/p.stem
        subprocess.run(['pdftoppm','-r','90','-singlefile','-png',str(p),str(prefix)],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
        return {'path':str(p.relative_to(ROOT)),'sha256':digest(p),'page_count':1,'text_words':len(words),'page_bounds_pass':True,'render':str(prefix.with_suffix('.png'))}
    with ThreadPoolExecutor(max_workers=4) as pool:pdf_checks=list(pool.map(render,pdfs))
    sheets=[]
    for offset in range(0,len(pdf_checks),4):
        sheet=Image.new('RGB',(2000,1700),'#dddddd');draw=ImageDraw.Draw(sheet)
        for j,rec in enumerate(pdf_checks[offset:offset+4]):
            with Image.open(rec['render']) as source:
                im=source.convert('RGB');im.thumbnail((980,790));x=(j%2)*1000+(1000-im.width)//2;y=(j//2)*850+35
                sheet.paste(im,(x,y));draw.text(((j%2)*1000+10,(j//2)*850+10),Path(rec['path']).stem,fill='black')
        path=qa/f'contact_{offset//4+1:02}.png';sheet.save(path);sheets.append(str(path))
    from pypdf import PdfWriter,PdfReader
    atlas=HERE/'quantitative_figure_atlas.pdf'
    writer=PdfWriter()
    for p in pdfs:writer.append(str(p),outline_item=p.stem.replace('_',' '))
    with atlas.open('wb') as file:writer.write(file)
    assert len(PdfReader(atlas).pages)==len(pdfs)
    record={'status':'numeric_and_render_checks_passed_visual_review_pending','png_figures':len(pngs),'pdf_figures':len(pdfs),'all_source_and_output_hashes_match':True,'numeric_values_verified':{'stage_model_cohort_summaries':27,'confidence_histograms_and_calibration_bin_denominators':27,'classwise_auc_ap_values':216},'pdf_checks':pdf_checks,'png_checks':png_checks,'contact_sheets':sheets,'atlas':{'path':str(atlas.relative_to(ROOT)),'sha256':digest(atlas),'pages':len(pdfs)},'model_inference':False,'review_tool_script_sha256':digest(__file__)}
    (HERE/'figure_qa.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({'status':record['status'],'figures':len(pdfs),'review_directory':str(qa),'atlas':str(atlas.relative_to(ROOT))}))


if __name__=='__main__':main()
