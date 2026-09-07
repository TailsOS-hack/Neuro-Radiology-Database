#!/usr/bin/env python3
"""Build the concise cover memo; requires reportlab. Gallery stays in companion bundle."""
import json
from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.pagesizes import letter
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'docs/review_bundle/output/pdf'
OUT.mkdir(parents=True,exist_ok=True)
s=json.loads((ROOT/'docs/review_bundle/explainability/summary.json').read_text())
styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='Memo',fontName='Helvetica',fontSize=10,leading=13,spaceAfter=8,textColor=colors.HexColor('#24344a')))
styles.add(ParagraphStyle(name='Label',fontName='Helvetica-Bold',fontSize=10,leading=13,spaceBefore=5,spaceAfter=5,textColor=colors.HexColor('#205d7b')))
styles.add(ParagraphStyle(name='Meta',fontName='Helvetica',fontSize=9,leading=12,textColor=colors.HexColor('#607083'),spaceAfter=14))
flow=[]
def para(text,style='Memo'):flow.append(Paragraph(text,styles[style]))
para('Dementia manuscript | Mentor review', 'Title')
para('Prepared for Dr. Cockroft - September 7, 2026 - Internal draft, not sent','Meta')
para('Dear Dr. Cockroft, please review the dementia component of our leakage-audited public brain MRI benchmark. The tumor specialist and single-head comparator remain part of the original study design.')
para('Evidence retained','Label')
para('The accepted dementia specialist reached <b>0.9991 accuracy on 8,806 internal test images</b>; the conservative dHash sensitivity run reached 0.9975 on 8,881 images. The accepted single 8-class model remains slightly stronger than the hierarchical model (0.9972 vs 0.9963). These are public-image benchmark results, not clinical dementia diagnosis.')
para('New explainability result: a negative control comparison','Label')
para('All 128 sampled images per specialist yielded valid Grad-CAMs. Masking the highest-CAM 20% of pixels reduced confidence <b>less</b> than equal-count random masking:')
rows=[['Specialist','Top-CAM drop','Random drop','Paired difference']]
for task in ['dementia','tumor']:
 m=s[task]['metrics'];rows.append([task.capitalize()]+[f"{m[k]['mean']:.4f}" for k in ['top20_drop','random20_drop','paired_drop_advantage']])
t=Table(rows,colWidths=[119,110,110,153],hAlign='LEFT')
t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e9f0f5')),('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),('FONTSIZE',(0,0),(-1,-1),9),('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7),('LINEBELOW',(0,-1),(-1,-1),.5,colors.HexColor('#cad6df'))]))
flow.extend([t,Spacer(1,9)])
para('This test does not support preferential sensitivity to the highlighted regions. Spatially different masks and out-of-distribution perturbations limit interpretation. The separate gallery contains seven dementia and eight tumor low-confidence/error examples, chosen by a fixed rule before inspecting maps. No expert anatomical validation was performed.')
para('Boundaries unchanged','Label')
para('The dementia data are publicly sourced and augmented. Patient identifiers and acquisition metadata are absent; patient-level leakage and source shortcuts remain possible. Confidence is not calibrated clinical probability (integrated ECE: 0.1327 single-head; 0.1526 hierarchical). <b>No independent external cohort has been evaluated.</b> Accepted checkpoints were recovered and checked against archived predictions; no retraining or test-based tuning was performed.')
para('Decisions requested','Label')
para('1. Keep the negative perturbation comparison and galleries as a transparent supplement?<br/>2. Retain the combined benchmark framing, or emphasize dementia within that design?<br/>3. Which venue and independent patient-level cohort should guide the next study?')
para('Companion reading','Label')
para('Full manuscript: Methods 2.9, Results 3.6 and limitations. The companion bundle contains the browser gallery, Figures 8a-8b, quantitative summary, per-image evidence, provenance hashes and reproduction scripts. Start with <b>docs/review_bundle/README.md</b>. Authorship, venue formatting, formal references, image-use terms and institutional ethics wording still need completion.','Memo')
path=OUT/'Dr_Cockroft_review_memo.pdf'
def footer(c,d):
 c.setFont('Helvetica',8);c.setFillColor(colors.HexColor('#607083'))
 c.drawString(60,30,'Internal public-dataset benchmark - not clinical or external validation')
 c.drawRightString(552,30,str(d.page))
SimpleDocTemplate(str(path),pagesize=letter,rightMargin=60,leftMargin=60,topMargin=35,bottomMargin=45,title='Dementia manuscript mentor review',author='Neuro-Radiology-Database project').build(flow,onFirstPage=footer,onLaterPages=footer)
print(path)
