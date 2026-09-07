#!/usr/bin/env python3
"""Idempotently add generated follow-up passages without rewriting historical results."""
import argparse
import json
import re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def blocks(bundle):
    protocol=json.loads((bundle/'protocol.json').read_text());run=protocol['run_id']
    exp=json.loads((bundle/'explainability/summary.json').read_text())
    cal=json.loads((bundle/'calibration/summary.json').read_text())
    get=lambda task:exp['perturbations'][f'{task}|0.2|imagenet_mean|ALL']['metrics']['shape_advantage']['mean']
    relative=bundle.relative_to(ROOT/'docs')
    methods=f'''<!-- validation-v2-methods:start -->
### 2.10 Follow-up Explainability Controls And Research Calibration

The frozen follow-up protocol ({run}) reused the original 256-image sample and gallery selection. Top-CAM masking at 10%, 20% and 30% was compared with five scattered-random masks and up to five distinct nonidentity rotations/reflections of the same binary mask, using ImageNet-mean and Gaussian-blur replacement (kernel 31, sigma 10). Progressive randomization of the classifier head, head plus final feature block, and full model used seeds 42–44 with the original predicted target held fixed; native-map Spearman correlation and top-20% IoU were summarized only for valid maps, with failures counted. These diagnostics do not validate anatomy.

Four scalar temperatures were fitted by minimizing validation NLL over [0.05, 20] and frozen before complete strict-test inference. Unsuccessful or non-improving fits retain temperature 1. Hard-routed hierarchical classification and joint-probability argmax are reported separately. The validation split had already influenced checkpoint selection, and the test set had already been inspected; this is not independent validation. The application and accepted model weights were unchanged. Reproduction and external-cohort readiness are documented in `docs/VALIDATION_V2_REPRODUCTION.md` and `docs/EXTERNAL_VALIDATION_READINESS.md`.
<!-- validation-v2-methods:end -->'''
    results=f'''<!-- validation-v2-results:start -->
### 3.7 Follow-up Controls And Calibration

In the follow-up run {run}, top-CAM masking at 20% with ImageNet-mean replacement exceeded shape-preserving control confidence drops by {get('dementia'):.4f} for dementia and {get('tumor'):.4f} for tumor. The random-pixel comparisons remained control dependent; the original negative comparison is retained as historical evidence. Figures 9 and 10 show masking-control and randomization diagnostics, with all flat/failed-map counts in the supplement. These results do not establish anatomical correctness or absence of source shortcuts.

Dementia ECE changed from {cal['dementia']['original']['ece']:.6f} to {cal['dementia']['calibrated']['ece']:.6f}, and tumor ECE from {cal['tumor']['original']['ece']:.6f} to {cal['tumor']['calibrated']['ece']:.6f}, on the complete internal strict test. Component class predictions were unchanged. Figure 11 presents reliability diagrams; full NLL, Brier, ECE, accuracy and fitting-status records are in the [generated follow-up results]({relative}/RESULTS.md). Lower internal ECE does not establish calibrated clinical probability. External validation remains not performed.
<!-- validation-v2-results:end -->'''
    references=f'''<!-- validation-v2-reference:start -->
## Versioned validation follow-up — {run}

The original results and mentor packet remain the historical baseline. [Follow-up results]({relative}/RESULTS.md), [Figures 9–11 and gallery]({relative}/index.html), and the [updated mentor memo]({relative}/MENTOR_MEMO.md) are generated from the same completed run. These research-only analyses retain all public-data, patient-metadata, source-bias, calibration and absent-external-validation limitations.
<!-- validation-v2-reference:end -->'''
    return methods,results,references


def replace_block(text,block,tag,anchor=None):
    pattern=rf'<!-- {tag}:start -->.*?<!-- {tag}:end -->'
    if re.search(pattern,text,flags=re.S):return re.sub(pattern,lambda _:block,text,flags=re.S)
    if anchor:return text.replace(anchor,block+'\n\n'+anchor,1)
    return text.rstrip()+'\n\n'+block+'\n'


def update(bundle,verify=False):
    methods,results,references=blocks(bundle)
    paths=[ROOT/'docs'/name for name in ['MANUSCRIPT_FULL_DRAFT.md','MANUSCRIPT_DRAFT.md','PUBLICATION_EVIDENCE_RESULTS.md','PUBLICATION_FIGURES.md','FIGURE_CAPTIONS.md']]
    for path in paths:
        text=path.read_text()
        if path.name=='MANUSCRIPT_FULL_DRAFT.md':
            expected=replace_block(text,methods,'validation-v2-methods','## 3. Results')
            expected=replace_block(expected,results,'validation-v2-results','## 4. Discussion')
        else:
            block=references
            if path.name=='FIGURE_CAPTIONS.md':
                block=block.replace('<!-- validation-v2-reference:end -->', (bundle/'FIGURE_CAPTIONS.md').read_text().replace('# Follow-up figure captions','### Follow-up figure captions')+'\n<!-- validation-v2-reference:end -->')
            expected=replace_block(text,block,'validation-v2-reference')
        if verify:
            if text!=expected:raise ValueError(f'Stale follow-up manuscript references: {path.name}')
        else:path.write_text(expected)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('bundle',type=Path);p.add_argument('--verify',action='store_true');a=p.parse_args();update(a.bundle.resolve(),a.verify)
