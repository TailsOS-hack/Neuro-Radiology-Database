#!/usr/bin/env python3
"""Export only verified public science artifacts; never infer, train or select.

This is inert until explicitly invoked after real result retrieval and visual QA.
Private clinical records, input manifests, MRI panels, MCI predictions, model
weights, operational metadata and whole result directories are never copied.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
from urllib.parse import unquote, urlsplit

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2]
NUMERICAL='complete_numerically_verified_visual_review_required'
QA_STATUS='passed_numeric_and_visual_review'
DIAGNOSTIC_STATUS='complete_diagnostic_appendix_with_blocked_replay'
BLOCKED_STATUS='blocked_cloud_probability_replay_review'
STAGE_SEEDS=[20261011,20261012,20261013]
CLINICAL_SEEDS=[20261021,20261022,20261023]
HEX=re.compile(r'[0-9a-f]{64}')
CLINICAL_HISTORY_FIELDS={'epoch','train_aug_loss','train_aug_accuracy','train_clean_loss',
    'train_clean_accuracy','val_loss','val_accuracy','val_balanced_accuracy','head_lr',
    'backbone_lr','selected','stale_epochs','seconds'}
PRIVATE_KEYS={'patient_id','patient_ids','subject_id','subject_ids','group_key','group_keys',
              'patients','patient_predictions','source_path','source_paths','headers'}
CLINICAL_METRIC_FIELDS={'model','scenario','cohort','n_patients','n_ad','n_cn','correct',
    'accuracy','sensitivity_ad','specificity_cn','precision_ad','negative_predictive_value',
    'balanced_accuracy','macro_f1','roc_auc','average_precision','nll','binary_brier',
    'ad_probability_ece_5_bins','threshold','calibration_fitted','unit',
    'mean_probability_total_variation','label_flip_fraction'}
HISTORICAL='docs/kevin_paper_20261011/posthoc_explainability/results'
HISTORICAL_MANIFEST_SHA='abf11186722dd43ab4fbea7b97eb7752007436c5dacd8bfc27dd0bfe80f2b7e7'
HISTORICAL_CASES_SHA='71e94e550471989ff4316bb5742e185379add3ef7e1237bcfdb901d538b378a5'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def encoded(value):return (json.dumps(value,indent=2,allow_nan=False)+'\n').encode()
def rel(name):
    if not isinstance(name,str) or not name or '\\' in name or '\x00' in name:raise ValueError('Invalid relative artifact path')
    p=PurePosixPath(name)
    if p.is_absolute() or '..' in p.parts or str(p)!=name:raise ValueError('Unsafe artifact path')
    return p
def regular(root,name):
    p=root
    for part in rel(name).parts:
        p=p/part
        if p.is_symlink():raise ValueError('Artifact symlinks are not permitted')
    if not p.is_file() or not p.resolve().is_relative_to(root.resolve()):raise ValueError('Missing regular artifact: '+name)
    return p


def inventory(directory,marker,status,plan=None,suite=None,kind=None):
    complete=read(regular(directory,marker));manifest_path=regular(directory,'SHA256SUMS.json');manifest=read(manifest_path)
    if complete.get('status')!=status or complete.get('manifest_sha256')!=sha(manifest_path) or complete.get('payloads')!=len(manifest):raise ValueError('Incomplete or unverified numerical package: '+str(directory))
    if not manifest:raise ValueError('Empty numerical inventory')
    for name,h in manifest.items():
        path=regular(directory,name)
        if not isinstance(h,str) or not HEX.fullmatch(h) or sha(path)!=h:raise ValueError('Numerical payload changed: '+name)
        if plan is not None and plan.pin(path,suite,kind)!=h:raise ValueError('Source changed while reading numerical inventory')
    if plan is not None:
        for name in [marker,'SHA256SUMS.json']:plan.pin(regular(directory,name),suite,kind)
    return complete,manifest


def one_directory(suite,marker):
    candidates=[p for p in suite.iterdir() if p.is_dir() and not p.is_symlink() and (p/marker).is_file()]
    if len(candidates)!=1:raise ValueError(f'Require one direct child containing {marker}; found {len(candidates)} under {suite}')
    return candidates[0]


def strip_patient_ids(value):
    if isinstance(value,dict):return {k:strip_patient_ids(v) for k,v in value.items() if k!='patient_ids'}
    if isinstance(value,list):return [strip_patient_ids(v) for v in value]
    return value


def no_private_keys(value):
    if isinstance(value,dict):
        if set(value)&PRIVATE_KEYS:raise ValueError('Private clinical record key in public derivative')
        for v in value.values():no_private_keys(v)
    elif isinstance(value,list):
        for v in value:no_private_keys(v)


def clinical_history(path):
    rows=read(path)
    if not isinstance(rows,list) or not rows or [r.get('epoch') for r in rows]!=list(range(len(rows))):raise ValueError('Clinical history is not contiguous')
    for r in rows:
        if not set(r)<=CLINICAL_HISTORY_FIELDS or any(isinstance(v,(dict,list,str)) for v in r.values()):raise ValueError('Unexpected clinical history content')
    return rows


def check_history_csv(json_path,csv_path):
    rows=read(json_path);actual=read_csv(csv_path)
    if not isinstance(rows,list) or not rows or not all(isinstance(r,dict) for r in rows):raise ValueError('Invalid raw history')
    fields=list(dict.fromkeys(k for r in rows for k in r))
    expected=[{k:'' if r.get(k) is None else str(r[k]) for k in fields} for r in rows]
    if actual!=expected:raise ValueError('Raw history JSON/CSV disagree')


def read_csv(path):
    with Path(path).open(newline='') as f:return list(csv.DictReader(f))


def portability(text):
    if re.search(r'(?:/Users/|/home/|/workspace/|/private/|file://|[A-Za-z]:\\)',text):raise ValueError('Absolute/private runtime path in public text')
    if re.search(r'(?i)(?:BEGIN[^\n]*PRIVATE KEY|ssh-ed25519 |ssh-rsa |api[_-]?key\s*[:=])',text):raise ValueError('Credential-like content in public text')


class Plan:
    def __init__(self):self.items={};self.sources={};self.source_names={};self.destinations={};self.figure_requirements={'stage':{},'clinical':{}}
    def pin(self,path,root,kind,label=None):
        name=str(path.relative_to(root));regular(root,name);h=sha(path)
        if path in self.sources and self.sources[path]!=h:raise ValueError('Source changed while planning')
        self.sources[path]=h;self.source_names[path]=(label or kind+'_suite')+'/'+name
        return h
    def add(self,path,dest,root,kind,transform=None,visual=False,label=None):
        name=str(path.relative_to(root));regular(root,name);rel(dest)
        if dest in self.items:raise ValueError('Duplicate public destination')
        h=self.pin(path,root,kind,label);raw=path.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=h:raise ValueError('Source changed while reading public payload')
        data=raw if transform is None else transform(raw)
        if Path(dest).suffix.lower() in {'.md','.html','.json','.csv','.txt'}:portability(data.decode('utf-8'))
        self.items[dest]=dict(data=data,source=self.source_names[path],source_sha256=h,
            transformation='identity' if transform is None else 'documented_public_derivative')
        self.destinations[path.resolve()]=dest
        if visual:self.figure_requirements[kind][name]=h
    def package(self,directory,manifest,names,prefix,suite,kind):
        for name in names:
            if name not in manifest:raise ValueError('Export file lacks numerical inventory coverage: '+name)
            self.add(regular(directory,name),prefix+'/'+name,suite,kind,visual=name.endswith(('.png','.pdf')))
    def qa(self,suite,kind):
        path=regular(suite,'publication_visual_qa.json');q=read(path)
        if q.get('status')!=QA_STATUS or q.get('real_data') is not True or not isinstance(q.get('reviewed_files'),dict):raise ValueError('Real-result visual review receipt is absent or incomplete')
        for name,h in self.figure_requirements[kind].items():
            if q['reviewed_files'].get(name)!=h:raise ValueError('Export visual lacks exact reviewed SHA: '+name)
        self.pin(path,suite,kind)
        return dict(status=QA_STATUS,real_data=True,source_receipt_sha256=sha(path),
                    reviewed_export_visuals=len(self.figure_requirements[kind]))


def figure_names(directory,manifest,count):
    pdfs=sorted(n for n in manifest if '/' not in n and n.endswith('.pdf'))
    pngs=sorted(n for n in manifest if '/' not in n and n.endswith('.png'))
    if len(pdfs)!=count or {Path(n).stem for n in pdfs}!={Path(n).stem for n in pngs}:raise ValueError('Incomplete required PNG/PDF figure families')
    return pdfs+pngs


def numerical_context(stage,clinical,plan):
    _,sm=inventory(stage/'frozen_round2_evaluation','evaluation_complete.json','complete_verified',plan,stage,'stage')
    if 'evaluation.json' not in sm:raise ValueError('Incomplete evaluation inventory coverage')
    se=read(stage/'frozen_round2_evaluation/evaluation.json');sf=read(regular(stage,'round2_evaluation_freeze.json'))
    plan.pin(stage/'round2_evaluation_freeze.json',stage,'stage')
    if se.get('status')!='complete' or se.get('freeze')!=sf or se.get('selection_freeze_sha256')!=sha(stage/'round2_evaluation_freeze.json'):raise ValueError('Stage evaluation/freeze mismatch')
    if sf.get('status')!='frozen_before_any_round2_evaluation':raise ValueError('Invalid scientific selection boundary')
    if set(sf.get('all_three_runs',{}))!={f'round2_seed{s}' for s in STAGE_SEEDS}:raise ValueError('Incomplete three-seed selections')
    if clinical is None:return sf,None,sm,None
    _,cm=inventory(clinical/'clinical_evaluation','evaluation_complete.json','complete_verified',plan,clinical,'clinical')
    if set(cm)!={'evaluation.json','metrics.json','patient_predictions.json','model_state_verification.json'}:raise ValueError('Incomplete evaluation inventory coverage')
    ce=read(clinical/'clinical_evaluation/evaluation.json');cf=read(regular(clinical,'clinical_evaluation_freeze.json'))
    plan.pin(clinical/'clinical_evaluation_freeze.json',clinical,'clinical')
    if ce.get('status')!='complete' or ce.get('freeze')!=cf or ce.get('freeze_sha256')!=sha(clinical/'clinical_evaluation_freeze.json'):raise ValueError('Clinical evaluation/freeze mismatch')
    if cf.get('status')!='frozen_before_clinical_heldout_inference' or cf.get('heldout_outcomes_used_for_selection') is not False:raise ValueError('Invalid scientific selection boundary')
    if set(cf.get('selections',{}))!={f'clinical_seed{s}' for s in CLINICAL_SEEDS}:raise ValueError('Incomplete three-seed selections')
    if ce.get('fresh_binary_patients')!=19 or ce.get('mci_not_scored') is not True or ce.get('total_patient_scenario_records')!=3360 or ce.get('metric_records')!=168:raise ValueError('Clinical endpoint or complete grid changed')
    return sf,cf,sm,cm


def add_historical_maps(plan,cases,verification,prefix='stage/attribution'):
    source=REPO/HISTORICAL
    manifest_path=regular(source,'SHA256SUMS.json');case_path=regular(source,'per_case.csv')
    if sha(manifest_path)!=HISTORICAL_MANIFEST_SHA or sha(case_path)!=HISTORICAL_CASES_SHA:raise ValueError('Pinned historical attribution sources changed')
    if verification.get('historical_sources')!={'case_inventory_sha256':HISTORICAL_CASES_SHA,'baseline_payload_manifest_sha256':HISTORICAL_MANIFEST_SHA}:raise ValueError('Round 2 attribution used different historical sources')
    manifest=read(manifest_path);historical=read_csv(case_path);by={r['case_id']:r for r in cases}
    if len(historical)!=230 or len({(r['model'],r['case_id']) for r in historical})!=230:raise ValueError('Incomplete historical attribution alignment')
    plan.pin(manifest_path,REPO,'stage','repository')
    for r in historical:
        if r['model'] not in {'old_baseline','candidate_3'} or r['case_id'] not in by or any(r[k]!=by[r['case_id']][k] for k in ['path','label','role','image_sha256']):raise ValueError('Historical attribution case membership changed')
        name=f'maps/{r["model"]}_{r["case_id"]}.npy'
        path=regular(source,name)
        if sha(path)!=manifest.get(name):raise ValueError('Historical native map changed')
        plan.add(path,prefix+'/baseline_maps/'+path.name,REPO,'stage',label='repository')
    plan.add(case_path,prefix+'/historical_per_case.csv',REPO,'stage',label='repository')


def add_original100(plan,stage,se_inventory):
    keys=None
    for name in ['original100_case_aligned_models_readers.csv']+[n+'_original100_predictions.csv' for n in ['old_baseline','round1_c3']+[f'round2_seed{s}' for s in STAGE_SEEDS]]:
        if name not in se_inventory:raise ValueError('Missing original100 case output')
        path=regular(stage/'frozen_round2_evaluation',name);rows=read_csv(path);found={r['FileName'] for r in rows}
        if len(rows)!=100 or len(found)!=100 or (keys is not None and found!=keys):raise ValueError('Original100 export rows are incomplete or misaligned')
        keys=found;plan.add(path,'stage/original100/'+name,stage,'stage')


def stage_quantitative_plan(stage):
    """Explicit partial release; neither reads nor needs attribution/clinical data."""
    plan=Plan();sf,_,se_inventory,_=numerical_context(stage,None,plan)
    sr=one_directory(stage,'report_complete.json');_,sm=inventory(sr,'report_complete.json',NUMERICAL,plan,stage,'stage')
    analysis=read(sr/'analysis.json')
    if analysis.get('freeze_sha256')!=sha(stage/'round2_evaluation_freeze.json') or analysis.get('evaluation_sha256')!=se_inventory['evaluation.json']:raise ValueError('Stage report is not aligned to this evaluation')
    plan.package(sr,sm,figure_names(sr,sm,28)+['README.md','analysis.json','probability_curve_metrics.csv','figure_inputs.json'],'stage/report',stage,'stage')
    for seed in STAGE_SEEDS:
        selected=sf['all_three_runs'][f'round2_seed{seed}'];folder=f'stage_only_seed{seed}'
        if selected['checkpoint']!=folder+'/best.pt' or plan.pin(regular(stage,folder+'/history.json'),stage,'stage')!=selected['history_sha256']:raise ValueError('Stage training history changed')
        check_history_csv(stage/folder/'history.json',regular(stage,folder+'/history.csv'))
        for name in ['history.json','history.csv','raw_curves.png','raw_curves.pdf']:plan.add(regular(stage,folder+'/'+name),'stage/training/'+folder+'/'+name,stage,'stage',visual=name.endswith(('.png','.pdf')))
    add_original100(plan,stage,se_inventory)
    return plan,{'stage':plan.qa(stage,'stage')}


def make_plan(stage,clinical,attribution=None,diagnostic=False):
    plan=Plan();sf,cf,se_inventory,_=numerical_context(stage,clinical,plan)
    sr=one_directory(stage,'report_complete.json');cr=one_directory(clinical,'report_complete.json')
    marker='diagnostic_appendix_complete.json' if diagnostic else 'attribution_complete.json'
    attr=one_directory(stage,marker) if attribution is None else Path(attribution).resolve(strict=True)
    if attr.parent!=stage or not (attr/marker).is_file():raise ValueError('Attribution must be a sealed direct child of the stage suite')
    prefix='stage/attribution_diagnostics' if diagnostic else 'stage/attribution'
    _,sm=inventory(sr,'report_complete.json',NUMERICAL,plan,stage,'stage');_,cm=inventory(cr,'report_complete.json',NUMERICAL,plan,clinical,'clinical');_,am=inventory(attr,marker,DIAGNOSTIC_STATUS if diagnostic else NUMERICAL,plan,stage,'stage')
    sfig=figure_names(sr,sm,28);cfig=figure_names(cr,cm,19);afig=figure_names(attr,am,2)
    stage_analysis=read(sr/'analysis.json')
    if stage_analysis.get('freeze_sha256')!=sha(stage/'round2_evaluation_freeze.json') or stage_analysis.get('evaluation_sha256')!=se_inventory['evaluation.json']:raise ValueError('Stage report is not aligned to this evaluation')
    plan.package(sr,sm,sfig+['README.md','analysis.json','probability_curve_metrics.csv','figure_inputs.json'],'stage/report',stage,'stage')
    plan.package(cr,cm,cfig+['README.md','all_model_scenario_metrics.csv'],'clinical/report',clinical,'clinical')
    if 'analysis.json' not in cm:raise ValueError('Clinical analysis lacks SHA coverage')
    clinical_analysis=read(cr/'analysis.json')
    if clinical_analysis.get('status')!='complete_verified_predictions_only' or clinical_analysis.get('fresh_external_binary_n')!=19 or clinical_analysis.get('mci_scored') is not False or clinical_analysis.get('freeze_sha256')!=sha(clinical/'clinical_evaluation_freeze.json'):raise ValueError('Unexpected clinical analysis endpoint or freeze')
    metrics=read_csv(cr/'all_model_scenario_metrics.csv')
    if len(metrics)!=168 or len({(r['model'],r['scenario'],r['cohort']) for r in metrics})!=168:raise ValueError('Incomplete aggregate clinical metric grid')
    if any(not set(r)<=CLINICAL_METRIC_FIELDS for r in metrics):raise ValueError('Unexpected field in aggregate clinical metrics')
    no_private_keys(metrics)
    def public_clinical(raw):
        v=strip_patient_ids(json.loads(raw));no_private_keys(v);return encoded(v)
    plan.add(cr/'analysis.json','clinical/report/analysis.json',clinical,'clinical',public_clinical)
    for seed in STAGE_SEEDS:
        selected=sf['all_three_runs'][f'round2_seed{seed}'];folder=f'stage_only_seed{seed}'
        if selected['checkpoint']!=folder+'/best.pt' or plan.pin(regular(stage,folder+'/history.json'),stage,'stage')!=selected['history_sha256']:raise ValueError('Stage training history changed')
        check_history_csv(stage/folder/'history.json',regular(stage,folder+'/history.csv'))
        for name in ['history.json','history.csv','raw_curves.png','raw_curves.pdf']:
            plan.add(regular(stage,folder+'/'+name),'stage/training/'+folder+'/'+name,stage,'stage',visual=name.endswith(('.png','.pdf')))
    for seed in CLINICAL_SEEDS:
        folder=f'clinical_seed{seed}';selected=cf['selections'][folder]
        if selected['checkpoint_path']!=folder+'/best.pt' or plan.pin(regular(clinical,folder+'/history.json'),clinical,'clinical')!=selected['history_sha256']:raise ValueError('Clinical training history changed')
        clinical_history(clinical/folder/'history.json')
        check_history_csv(clinical/folder/'history.json',regular(clinical,folder+'/history.csv'))
        for name in ['history.json','history.csv','raw_patient_curves.png','raw_patient_curves.pdf']:
            plan.add(regular(clinical,folder+'/'+name),'clinical/training/'+folder+'/'+name,clinical,'clinical',visual=name.endswith(('.png','.pdf')))
    cases=read_csv(regular(attr,'per_case.csv'))
    if len(cases)!=115 or len({r['case_id'] for r in cases})!=115 or sum(r['role']=='reused_original100' for r in cases)!=100:raise ValueError('Changed fixed attribution case membership')
    # Only the legacy image benchmark is included; never newly acquired MRI.
    for r in cases:
        image_path=str(rel(r['path']))
        if image_path.startswith('data/external/') or 'clinical' in image_path.lower():raise ValueError('Clinical image entered public attribution scope')
    verification=read(regular(attr,'verification.json'))
    if verification.get('case_count')!=115 or verification.get('all_logits_modes_gradients_unchanged') is not True or verification.get('original_checkpoint_state_restored') is not True:raise ValueError('Attribution mechanics are incomplete')
    failures=verification.get('cloud_probability_or_argmax_failures')
    if not diagnostic and failures:raise ValueError('Attribution replay is blocked')
    names=afig+['gallery.html','per_case.csv','native_map_mass.csv','randomization.csv','verification.json']
    if diagnostic:
        if not failures or verification.get('cloud_probability_tolerance')!=1e-4 or any(r.get('argmax_agreement') is not True or r.get('max_probability_difference',0)<=1e-4 for r in failures):raise ValueError('Diagnostic replay failure differs from the authorized unchanged-class probability condition')
        source_marker=read(regular(attr,'source_attribution_complete.json'));source_manifest=regular(attr,'source_SHA256SUMS.json')
        if source_marker.get('status')!=BLOCKED_STATUS or source_marker.get('manifest_sha256')!=sha(source_manifest):raise ValueError('Original blocked attribution marker is absent or changed')
        source_inventory=read(source_manifest)
        if source_marker.get('payloads')!=len(source_inventory) or source_inventory.get('verification.json')!=sha(attr/'verification.json'):raise ValueError('Diagnostic verification differs from original blocked output')
        names+=['source_attribution_complete.json','source_SHA256SUMS.json','diagnostic_provenance.json','ATTRIBUTION_REPLAY_LIMITATION.md']
        provenance=read(regular(attr,'diagnostic_provenance.json'))
        if provenance.get('original_marker_sha256')!=sha(attr/'source_attribution_complete.json'):raise ValueError('Diagnostic source-marker provenance changed')
        visual_names={n for n in am if n.endswith(('.png','.pdf')) and (n.startswith('cases/') or '/' not in n)}
        warnings=provenance.get('warning_visuals_sha256')
        if not isinstance(warnings,dict) or set(warnings)!=visual_names or any(warnings[n]!=am[n] for n in visual_names):raise ValueError('Diagnostic warning-bearing visual coverage is incomplete')
        if not (attr/'recovery_receipt.json').is_file():raise ValueError('Diagnostic recovery receipt is missing')
        if 'replay' not in (attr/'gallery.html').read_text().lower() or 'diagnostic' not in (attr/'gallery.html').read_text().lower():raise ValueError('Diagnostic gallery warning is absent')
    if (attr/'recovery_receipt.json').exists():
        if read(regular(attr,'recovery_receipt.json')).get('status')!='postprocessing_only_serialization_recovery':raise ValueError('Unrecognized attribution recovery')
        names.append('recovery_receipt.json')
    expected_cases={f'cases/{r["case_id"]}.png' for r in cases}
    if {n for n in am if n.startswith('cases/')}!=expected_cases:raise ValueError('Incomplete 115-case attribution gallery')
    names+=sorted(expected_cases)
    names+=sorted(n for n in am if (n.startswith('maps/') or n.startswith('randomized_maps/')) and n.endswith('.npy'))
    plan.package(attr,am,names,prefix,stage,'stage')
    plan.add(attr/marker,prefix+'/'+marker,stage,'stage')
    # The new panels reuse these pinned public legacy-benchmark maps. Include
    # those native arrays too so every displayed system has accessible data.
    add_historical_maps(plan,cases,verification,prefix)
    # Every original100 case remains available for the exact reader comparison.
    add_original100(plan,stage,se_inventory)
    # Optional existing atlas: its source paths become portable exported paths.
    atlas_marker='diagnostic_atlas_verification.json' if diagnostic else 'atlas_verification.json'
    atlas_name='Round2_Diagnostic_Graphs_Atlas.pdf' if diagnostic else 'Round2_Graphs_Atlas.pdf'
    atlases=[p for p in stage.iterdir() if p.is_dir() and not p.is_symlink() and (p/atlas_marker).is_file()]
    if len(atlases)>1:raise ValueError('Ambiguous stage atlas')
    if atlases:
        atlas=atlases[0];av=read(atlas/atlas_marker);pdf=regular(atlas,atlas_name)
        if av.get('pages')!=30 or av.get('atlas_sha256')!=sha(pdf):raise ValueError('Stage atlas verification mismatch')
        if diagnostic:
            if av.get('status')!='complete_diagnostic_atlas_with_blocked_replay' or av.get('attribution_replay_status')!=BLOCKED_STATUS or av.get('quantitative_pages')!=28 or av.get('diagnostic_pages')!=2 or av.get('diagnostic_manifest_sha256')!=sha(attr/'SHA256SUMS.json') or av.get('report_manifest_sha256')!=sha(sr/'SHA256SUMS.json'):raise ValueError('Diagnostic atlas source/status mismatch')
        elif av.get('source_pdf_count')!=30:raise ValueError('Stage atlas page-source count mismatch')
        records=read_csv(regular(atlas,'atlas_index.csv'))
        if len(records)!=30 or [int(r['page']) for r in records]!=list(range(1,31)):raise ValueError('Incomplete atlas index')
        expected_pdfs={dest for dest in plan.items if dest.startswith(('stage/report/',prefix+'/')) and dest.endswith('.pdf')}
        seen=set()
        for r in records:
            # The original index may retain cloud paths after retrieval. Resolve
            # its exact digest + basename against already verified public plots.
            source_name=r['source_file'] if diagnostic else Path(r['source']).name
            matches=[dest for dest in expected_pdfs if Path(dest).name==source_name and plan.items[dest]['source_sha256']==r['source_sha256']]
            if len(matches)!=1 or matches[0] in seen:raise ValueError('Atlas source is absent, duplicated or ambiguous')
            seen.add(matches[0]);r['source']='../'+matches[0].removeprefix('stage/')
            if diagnostic:r['source_directory']=str(PurePosixPath(r['source']).parent);r['source_file']=PurePosixPath(r['source']).name
        if seen!=expected_pdfs:raise ValueError('Atlas does not contain every stage graph exactly once')
        plan.pin(atlas/atlas_marker,stage,'stage')
        stream=io.StringIO();writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
        plan.add(pdf,'stage/atlas/'+atlas_name,stage,'stage',visual=True)
        plan.add(atlas/'atlas_index.csv','stage/atlas/atlas_index.csv',stage,'stage',lambda _:stream.getvalue().encode())
    qa={'stage':plan.qa(stage,'stage'),'clinical':plan.qa(clinical,'clinical')}
    if diagnostic:qa['stage']['attribution_probability_replay_status']=BLOCKED_STATUS
    return plan,qa


def link_check(contents):
    for name,data in contents.items():
        if not name.endswith(('.md','.html')):continue
        text=data.decode();links=re.findall(r'\]\(([^)]+)\)',text)+re.findall(r'(?:src|href)=["\']([^"\']+)["\']',text)
        for link in links:
            link=link.strip('<>');parsed=urlsplit(link)
            if parsed.scheme in {'http','https','mailto'} or not parsed.path:continue
            if parsed.scheme or parsed.path.startswith('/'):raise ValueError('Nonportable artifact link: '+link)
            parts=[]
            for part in (PurePosixPath(name).parent/PurePosixPath(unquote(parsed.path))).parts:
                if part=='..':
                    if not parts:raise ValueError('Artifact link escapes output')
                    parts.pop()
                elif part!='.':parts.append(part)
            if '/'.join(parts) not in contents:raise ValueError('Broken public artifact link: '+name+' -> '+link)


def add_research_weight_metadata(plan,stage,clinical):
    """Optional already-public fidelity evidence, never the checkpoint binaries."""
    manifest=REPO/'models/kevin_round2_20261011/CHECKPOINTS.json'
    receipt=HERE/'research_weights_export_verification.json'
    if not manifest.exists() and not receipt.exists():return False
    if not manifest.exists() or not receipt.exists():raise ValueError('Incomplete public checkpoint metadata pair')
    m=read(manifest);r=read(receipt)
    if r.get('status')!='six_fixed_research_exports_verified' or r.get('checkpoint_manifest_sha256')!=sha(manifest):raise ValueError('Research checkpoint metadata is unverified')
    expected={'stage':sha(stage/'round2_evaluation_freeze.json'),'clinical':sha(clinical/'clinical_evaluation_freeze.json')}
    if m.get('frozen_selection_sha256')!=expected:raise ValueError('Research checkpoint metadata belongs to different frozen selections')
    entries=m.get('checkpoints',[])
    if len(entries)!=6 or {(x.get('kind'),x.get('seed')) for x in entries}!={('stage',s) for s in STAGE_SEEDS}|{('clinical',s) for s in CLINICAL_SEEDS}:raise ValueError('Incomplete fixed research checkpoint roster')
    if r.get('production_baseline_unchanged') is not True or r.get('real_image_inference_performed') is not False:raise ValueError('Unexpected research checkpoint export scope')
    no_private_keys(m);no_private_keys(r)
    for path,dest in [(manifest,'research_checkpoints/CHECKPOINTS.json'),(receipt,'research_checkpoints/export_verification.json')]:
        plan.add(path,dest,REPO,'stage',label='repository')
    return True


def export(stage,clinical,output,attribution=None,diagnostic=False,stage_only=False):
    if stage_only and (clinical is not None or attribution is not None or diagnostic):raise ValueError('Stage-only export cannot be combined with clinical/attribution arguments')
    if not stage_only and clinical is None:raise ValueError('Full export requires a clinical suite')
    stage=Path(stage).resolve(strict=True);clinical=None if clinical is None else Path(clinical).resolve(strict=True);output=Path(output).resolve()
    if output.exists() or output.parent!=HERE.resolve() or not output.name.startswith('artifacts'):raise ValueError('Output must be a new artifacts* child of round2_publication')
    plan,qa=stage_quantitative_plan(stage) if stage_only else make_plan(stage,clinical,attribution,diagnostic)
    weights_metadata=False if stage_only else add_research_weight_metadata(plan,stage,clinical)
    contents={n:r['data'] for n,r in plan.items.items()}
    lines=['# Round 2 research artifacts','','[Stage results](stage/report/README.md)']
    if stage_only:
        lines+=['','**Explicit partial release: stage quantitative results only.** Clinical and attribution artifacts are not part of this package. No attribution replay success is asserted.']
    else:
        prefix='stage/attribution_diagnostics' if diagnostic else 'stage/attribution'
        lines+=['','[Clinical results](clinical/report/README.md) · [All 115 attribution cases]('+prefix+'/gallery.html)']
        if diagnostic:
            v=json.loads(contents[prefix+'/verification.json']);failures=v['cloud_probability_or_argmax_failures']
            lines+=['',f'**Attribution is a diagnostic appendix: probability replay remains BLOCKED.** {len(failures)}/100 cases exceed the frozen maximum absolute probability-difference tolerance of {v["cloud_probability_tolerance"]:g}; predicted classes are unchanged. The tolerance was not widened and inference was not rerun. Successful visual review does not turn this failed replay check into a pass. See the [unchanged blocked marker]('+prefix+'/source_attribution_complete.json) and [recovery evidence]('+prefix+'/recovery_receipt.json).']
    lines+=['','Stage results are reused development benchmarks. No clinical-readiness claim follows.']
    if not stage_only:lines+=['The separate clinical pilot preserves the fresh Harvard 19-patient endpoint and reused Besga/OpenNeuro cohorts.']
    if weights_metadata:lines+=['','[Six fixed research checkpoint identities](research_checkpoints/CHECKPOINTS.json) and [export fidelity evidence](research_checkpoints/export_verification.json). Checkpoint binaries are stored separately in the repository model directory; synthetic export equality does not establish medical validity.']
    lines+=['','## Figure and curve index','']
    if 'stage/atlas/Round2_Graphs_Atlas.pdf' in contents:lines+=['[Stage 30-page graph atlas](stage/atlas/Round2_Graphs_Atlas.pdf)','']
    if 'stage/atlas/Round2_Diagnostic_Graphs_Atlas.pdf' in contents:lines+=['[Stage graph atlas with explicitly flagged diagnostic attribution pages](stage/atlas/Round2_Diagnostic_Graphs_Atlas.pdf)','']
    for kind in (['stage'] if stage_only else ['stage','clinical']):
        lines+=['### '+kind.capitalize(),'']
        for n in sorted(contents):
            if n.startswith(kind+'/') and n.endswith('.pdf'):lines.append(f'- [{Path(n).stem.replace("_"," ")}]({n})')
        lines.append('')
    contents['README.md']=('\n'.join(lines)+'\n').encode();contents['VISUAL_QA.json']=encoded(qa)
    link_check(contents)
    for path,h in plan.sources.items():
        if sha(path)!=h:raise ValueError('Source changed before public export')
    output.mkdir(mode=0o755)
    records=[]
    for name,data in contents.items():
        p=output/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
        record=dict(path=name,output_sha256=sha(p),bytes=len(data))
        if name in plan.items:record.update({k:v for k,v in plan.items[name].items() if k!='data'})
        else:record.update(source='generated_navigation_or_QA_derivative',source_sha256=None,transformation='generated')
        records.append(record)
    if any(sha(path)!=h for path,h in plan.sources.items()):raise ValueError('Original source changed during export')
    manifest=dict(status='complete_public_artifact_export',created_utc=datetime.now(timezone.utc).isoformat(),
        export_scope='stage_quantitative_only' if stage_only else 'full_with_blocked_attribution_diagnostic_appendix' if diagnostic else 'full_with_replay_verified_attribution',
        attribution_replay_status='not_part_of_export' if stage_only else BLOCKED_STATUS if diagnostic else 'passed_frozen_probability_guard',
        files=records,source_files_unchanged=True,private_clinical_rows_exported=False,
        guarded_sources=[dict(source=plan.source_names[path],sha256=h) for path,h in sorted(plan.sources.items())],
        clinical_derivative=None if stage_only else 'Removed every patient_ids key recursively; all other permitted aggregate fields retained.',
        research_checkpoint_metadata_included=weights_metadata,
        original100_case_outputs_included=True,visual_qa=qa,no_inference_or_selection=True)
    (output/'EXPORT_MANIFEST.json').write_bytes(encoded(manifest))
    for r in records:
        if sha(output/r['path'])!=r['output_sha256']:raise ValueError('Public output readback failed')
    return dict(status=manifest['status'],files=len(records),output=str(output),manifest_sha256=sha(output/'EXPORT_MANIFEST.json'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage-suite',type=Path,required=True);p.add_argument('--clinical-suite',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--attribution-directory',type=Path,help='Optional explicit completed direct child of the stage suite, e.g. a sealed recovery output')
    mode=p.add_mutually_exclusive_group();mode.add_argument('--include-blocked-attribution-diagnostics',action='store_true');mode.add_argument('--stage-quantitative-only',action='store_true')
    a=p.parse_args();print(json.dumps(export(a.stage_suite,a.clinical_suite,a.output,a.attribution_directory,a.include_blocked_attribution_diagnostics,a.stage_quantitative_only),indent=2))


if __name__=='__main__':main()
