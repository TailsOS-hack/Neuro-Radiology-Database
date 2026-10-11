#!/usr/bin/env python3
"""Assemble individually reviewed graph PDFs without modifying their pages."""
import csv
import hashlib
import json
from pathlib import Path

from pypdf import PdfReader,PdfWriter

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    qa=json.loads((HERE/'figure_qa.json').read_text())
    if qa.get('status')!='passed_numeric_and_visual_review' or qa['pdf_figures']!=34:
        raise ValueError('Require complete reviewed 34-figure quantitative set')
    folder=HERE/'generated';provenance=json.loads((folder/'figure_provenance.json').read_text())
    expected={ROOT/r['path']:r['sha256'] for r in provenance['figures'] if r['path'].endswith('.pdf')}
    first=['dementia_study_workflow','dementia_two_condition_architecture','stage_loss_all_six_runs','stage_accuracy_all_six_runs','clinical_loss_all_three_multitask_seeds','clinical_accuracy_all_three_seeds','train_resubstitution_confusion_all_six','stage_validation_confusion_all_six','full_new_test_confusion_all_six']
    def order(path):
        if path.stem in first:return (0,first.index(path.stem),'')
        for i,co in enumerate(['internal_same_case','old_hash_group_unexposed_sensitivity','original100']):
            if path.stem.startswith(co):return (1,i,path.stem)
        return (2,0,path.stem)
    with (folder/'FIGURE_INDEX.csv').open() as f:
        titles={r['figure']:r['title'] for r in csv.DictReader(f)}
    sources=[(p,'Completed quantitative figures; post-hoc analyses labeled on page') for p in sorted(expected,key=order)]
    stress=ROOT/'docs/kevin_paper_20261011/posthoc_robustness/results_20261011'
    sources += [(stress/(name+'.pdf'),'Post-hoc image perturbation stress analysis') for name in ['internal_same_case_accuracy_stress_curves','internal_same_case_label_flip_fraction_stress_curves','original100_accuracy_stress_curves','original100_label_flip_fraction_stress_curves']]
    attribution=ROOT/'docs/kevin_paper_20261011/posthoc_explainability/results'
    sources += [(attribution/(name+'.pdf'),'Exploratory attribution controls') for name in ['figure9_attribution_controls','figure10_parameter_randomization']]
    if len(sources)!=40 or len({p for p,_ in sources})!=40:raise ValueError('Atlas source count/uniqueness mismatch')
    writer=PdfWriter();records=[]
    for page,(path,category) in enumerate(sources,1):
        sha=digest(path)
        if path in expected and expected[path]!=sha:raise ValueError('Quantitative figure changed after provenance')
        if len(PdfReader(path).pages)!=1:raise ValueError('Each graph source must be one page')
        title=titles.get(path.stem,path.stem.replace('_',' '))
        writer.append(str(path),outline_item=title)
        records.append({'page':page,'title':title,'category':category,'source_path':str(path.relative_to(ROOT)),'source_sha256':sha})
    output=HERE/'all_graphs_atlas.pdf'
    with output.open('wb') as f:writer.write(f)
    reread=PdfReader(output)
    if len(reread.pages)!=40 or len(reread.outline)!=40:raise ValueError('Combined atlas page/bookmark count mismatch')
    for actual,(source,_) in zip(reread.pages,sources):
        original=PdfReader(source).pages[0]
        if list(actual.mediabox)!=list(original.mediabox) or actual.extract_text()!=original.extract_text():
            raise ValueError('Atlas page boundaries/text differ from reviewed source')
    with (HERE/'all_graphs_atlas_index.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    result={'status':'passed','source_figures':40,'pages':40,'bookmarks':40,'duplicate_sources':0,'source_page_bounds_and_text_preserved':True,'atlas_sha256':digest(output),'index_sha256':digest(HERE/'all_graphs_atlas_index.csv'),'script_sha256':digest(__file__),'sources':records,'quantitative_review':'figure_qa.json','additional_source_review':'Parent confirmed all four stress figures and both attribution-control figures individually reviewed before atlas assembly.'}
    (HERE/'all_graphs_atlas_verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='sources'}))


if __name__=='__main__':main()
