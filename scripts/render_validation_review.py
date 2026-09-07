#!/usr/bin/env python3
"""Generate follow-up tables, manuscript text and mentor memo from completed numerical results."""
from __future__ import annotations
import html
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.research_validation import require


def texts(root):
    protocol=json.loads((root/'protocol.json').read_text());run=protocol['run_id']
    exp=json.loads((root/'explainability/summary.json').read_text())
    cal=json.loads((root/'calibration/summary.json').read_text())
    fit=json.loads((root/'calibration/temperatures.json').read_text())
    failures=sum(v['failures'] for k,v in exp['randomizations'].items() if k.endswith('|ALL'))
    def value(task,key):return exp['perturbations'][f'{task}|0.2|imagenet_mean|ALL']['metrics'][key]['mean']
    result=[f'# Follow-up validation results — {run}', '',
        'Research-only follow-up on already inspected public-image test data. The validation split previously influenced checkpoint selection. No new independent cohort or clinical calibration is established.', '',
        '## Masking controls', '',
        'The same 128 images per specialist were evaluated. Positive paired differences mean that top-CAM masking reduced the original target-class confidence more than the control; negative differences mean less. All results below are descriptive image-level means.', '',
        '| Specialist | Fraction | Replacement | Top-CAM drop | Random-pixel drop | Shape-control drop | Paired random difference | Paired shape difference |',
        '| --- | --- | --- | --- | --- | --- | --- | --- |']
    for key,group in exp['perturbations'].items():
        if not key.endswith('|ALL'):continue
        task,fraction,replacement,_=key.split('|');m=group['metrics']
        result.append(f'| {task} | {float(fraction):.0%} | {replacement} | '+ ' | '.join(f"{m[k]['mean']:.4f}" if m[k]['mean'] is not None else 'unavailable' for k in ['top_drop','random_drop','shape_drop','random_advantage','shape_advantage'])+' |')
    result+=['',f'At 20% masking with ImageNet-mean replacement, paired shape-control differences were {value("dementia","shape_advantage"):.4f} for dementia and {value("tumor","shape_advantage"):.4f} for tumor. Corresponding random-pixel differences were {value("dementia","random_advantage"):.4f} and {value("tumor","random_advantage"):.4f}. These comparisons show sensitivity to the chosen control, not anatomical validation. The v2 random masks are independently seeded by image and fraction; their values need not equal the original v1 masks.', '',
        'D4 rotations/reflections preserve area and shape but may retain central support. Gaussian blur uses a 31-pixel kernel and sigma 10. Neither baseline eliminates out-of-distribution perturbations. See Figure 9 and per-class means, SDs, control counts and individual drops in the structured evidence.', '',
        '## Model randomization', '',
        f'Across 2,304 randomized-map attempts, {failures} were flat or failed. Such maps are counted and excluded from correlation/overlap averages; they are never represented as zero correlation. The accepted target class remains fixed. Figure 10 shows conditional means among valid maps.', '',
        '| Specialist | Seed | Stage | Valid / attempted | Spearman mean | Top-20% IoU mean |',
        '| --- | --- | --- | --- | --- | --- |']
    for key,g in exp['randomizations'].items():
        if key.endswith('|ALL'):
            task,seed,stage,_=key.split('|');m=g['metrics']
            result.append(f"| {task} | {seed} | {stage} | {g['n']-g['failures']} / {g['n']} | "+' | '.join(f"{m[k]['mean']:.4f}" if m[k]['mean'] is not None else 'unavailable' for k in ['spearman','top20_iou'])+' |')
    result+=['','Randomization is a model-dependence diagnostic, not proof of explanation faithfulness or biological relevance. The three seeds quantify initialization sensitivity; no clinical pass threshold is applied.','',
        '## Research-only temperature scaling','',
        '| Model / prediction rule | Temperature | Original ECE | Calibrated ECE | Original NLL | Calibrated NLL | Original Brier | Calibrated Brier | Original accuracy | Calibrated accuracy |',
        '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for task,modes in cal.items():
        orig,new=modes['original'],modes['calibrated'];temperature=f"{fit[task]['temperature']:.5f}" if task in fit else 'component temperatures'
        result.append(f'| {task} | {temperature} | '+' | '.join(f'{v:.6f}' for k in ['ece','nll','brier','accuracy'] for v in [orig[k],new[k]])+' |')
    result+=['','Temperatures were fitted on validation logits and frozen before test inference. Hierarchical routed predictions are reported separately from the combined-probability argmax; the routed confidence is the joint probability assigned to the routed class. NLL and Brier describe the full probability vector. Zeroes above are rounded, not guarantees of perfect calibration.', '',
        'Optimization records (including failure, fallback and boundary flags) are preserved for all models. The binary router temperature is near the lower search bound; near-perfect source routing and concentrated probabilities do not rule out dataset-source shortcuts. Figure 11 shows reliability diagrams.', '',
        '## Limits and external readiness', '',protocol['claim_limits'], '',
        'External validation remains **not performed**. Only a synthetic cohort fixture has exercised the evaluation interface. Patient identifiers, labels, cohort provenance and suitable imaging must come from a separately acquired cohort. Internal calibration does not change application behavior.', '',
        'Methods: [Grad-CAM sanity checks](https://proceedings.neurips.cc/paper/2018/hash/294a8ed24b1ad22ec2e7efea049b8737-Abstract.html); [temperature scaling](https://proceedings.mlr.press/v70/guo17a.html).']
    memo=[f'# Dr. Cockroft — follow-up review memo ({run})','',
        'Internal research draft; not sent. The original mentor packet remains the historical baseline.','',
        '**What changed.** We added shape-preserving masking controls, model-randomization diagnostics, validation-only temperature scaling, and automated evidence checks. The sample, accepted checkpoint weights and original gallery selection were retained.','',
        f'**Explainability.** At 20% mean replacement, top-CAM masking exceeded shape-control confidence drops by {value("dementia","shape_advantage"):.4f} for dementia and {value("tumor","shape_advantage"):.4f} for tumor. Comparisons with scattered random pixels were {value("dementia","random_advantage"):.4f} and {value("tumor","random_advantage"):.4f}. This contrasts with interpreting the original random-pixel comparison alone. Control design materially affects the result; anatomy has not been validated. Of 2,304 randomized-map attempts, {failures} were flat or failed and are reported explicitly.','',
        f'**Calibration.** Dementia ECE changed from {cal["dementia"]["original"]["ece"]:.6f} to {cal["dementia"]["calibrated"]["ece"]:.6f}; tumor ECE changed from {cal["tumor"]["original"]["ece"]:.6f} to {cal["tumor"]["calibrated"]["ece"]:.6f}. Component accuracy was unchanged. These are internal estimates from data already used in model development and review. The application still uses its original confidence scores.','',
        '**Claim boundaries.** Public augmented dementia data, absent patient identifiers, possible patient-level leakage, source-domain bias and absent independent external validation remain central limitations. Lower internal ECE does not establish calibrated clinical probabilities. No retraining occurred.','',
        '**Review decisions.** Should the control-dependent explainability findings remain supplementary? Is the calibration comparison useful in the main manuscript? Which independent patient-level cohort and reference standard would support the next study?','',
        '**Reading.** RESULTS.md contains generated tables and interpretation. Figures 9–11 and the fixed gallery are in index.html. External readiness is documented separately; no external performance result is claimed.']
    captions=f'''# Follow-up figure captions — {run}

Figure 9. Paired top-CAM-minus-control confidence drops across three masking fractions, two replacement baselines and two control types. Means use the fixed class-balanced sample of 128 images per specialist. Geometric controls preserve binary-mask shape and area, not anatomical meaning.

Figure 10. Native CAM Spearman correlations after progressive model randomization, with the original predicted target held fixed. Means include valid maps only; flat/failed counts are reported in RESULTS.md. This is not a clinical attribution-validation test.

Figure 11. Original and temperature-scaled internal strict-test reliability diagrams for four component models and both hierarchical prediction rules. Temperatures were fitted on validation data only. Internal reliability does not establish calibrated clinical probability.
'''
    return {'RESULTS.md':'\n'.join(result)+'\n','MENTOR_MEMO.md':'\n'.join(memo)+'\n','FIGURE_CAPTIONS.md':captions}


def render(root):
    for name,text in texts(root).items():(root/name).write_text(text)
    run=json.loads((root/'protocol.json').read_text())['run_id']
    links='<p><a href="RESULTS.md">Results and limitations</a> · <a href="MENTOR_MEMO.md">Mentor memo</a> · <a href="mentor_memo.pdf">One-page PDF memo</a> · <a href="explainability/gallery.html">Fixed gallery</a> · <a href="FIGURE_CAPTIONS.md">Captions</a></p>'
    body=f'<!doctype html><html lang="en"><meta charset="utf-8"><title>{run}</title><style>body{{max-width:1100px;margin:30px auto;font:16px system-ui;color:#20334b}}img{{width:100%}}p{{line-height:1.5}}</style><h1>Validation follow-up</h1><p>{run}</p><p>Internal public-image benchmark; no anatomical validation, patient independence or external validation established.</p>'+links
    for caption,path in [('Figure 9: masking controls','explainability/figure9_controls.png'),('Figure 10: randomization diagnostics','explainability/figure10_randomization.png'),('Figure 11: research-only calibration','calibration/reliability.png')]:
        body+=f'<h2>{caption}</h2><img src="{path}" alt="{caption}">'
    (root/'index.html').write_text(body+'</html>')
    from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer
    from reportlab.lib import colors
    styles=getSampleStyleSheet();styles.add(ParagraphStyle(name='MemoText',fontName='Helvetica',fontSize=10,leading=14,spaceAfter=10))
    flow=[Paragraph('Dementia manuscript: validation follow-up',styles['Title']),Paragraph('Prepared for Dr. Cockroft - '+run,styles['Normal']),Spacer(1,14)]
    import re
    for paragraph in texts(root)['MENTOR_MEMO.md'].split('\n\n')[1:]:
        content=html.escape(paragraph).replace('—','-').replace('–','-')
        content=re.sub(r'\*\*(.*?)\*\*',r'<b>\1</b>',content)
        flow.append(Paragraph(content,styles['MemoText']))
    def footer(canvas,doc):
        canvas.setFont('Helvetica',8);canvas.setFillColor(colors.grey)
        canvas.drawString(50,25,'Internal mentor review - no external validation - '+run)
        canvas.drawRightString(560,25,str(doc.page))
    SimpleDocTemplate(str(root/'mentor_memo.pdf'),title='Dementia validation follow-up',pagesize=(612,792),leftMargin=50,rightMargin=50,topMargin=40,bottomMargin=45).build(flow,onFirstPage=footer,onLaterPages=footer)


if __name__=='__main__':render(Path(sys.argv[1]))
