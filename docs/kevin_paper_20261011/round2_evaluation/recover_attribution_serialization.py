#!/usr/bin/env python3
"""Seal one known final-JSON failure from saved artifacts; never run inference."""
from __future__ import annotations
import argparse,json,re,shutil
from pathlib import Path
import numpy as np
from PIL import Image
import round2_attribution as a
import round2_pipeline as p
import report_round2 as report
from src.explainability_validation import map_similarity

TRACE_SHA='86e0cd39e12fb661cefd34666049eef54d130a5b64622e793a674c3ad774d953'
ATTRIBUTION_SHA='5d44bcc8b22f337619fe7f20e307271f41a7669901540294e04f53dd9328c074'
TRANSITIVE_SHA='051c2c33fde7fd10b7daff72f775e299f47a6a083b21137d079acc1cb559969c'
SUMMARY_STEMS=['native_location_comparison','parameter_randomization']


def require(value,message):
    if not value:raise ValueError(message)


def boolean(value):
    require(value in ['True','False'],'Invalid CSV Boolean')
    return value=='True'


def near(value,expected,context,atol=1e-8):
    actual=float(value);require(np.isfinite(actual) and np.isclose(actual,expected,atol=atol,rtol=0),'Changed numeric field: '+context)


def inventory(directory):
    require(directory.is_dir() and not directory.is_symlink(),'Source must be a regular directory')
    result={}
    for f in sorted(directory.rglob('*')):
        require(not f.is_symlink(),'Symlink in artifact tree')
        if f.is_file():result[f.relative_to(directory).as_posix()]=p.sha(f)
        else:require(f.is_dir(),'Nonregular artifact member')
    return result


def check_trace(log):
    text=log.read_text()
    require(p.sha(log)==TRACE_SHA,'Failure log differs from the independently preserved exact traceback')
    require(text.rstrip().endswith('TypeError: Object of type int64 is not JSON serializable'),'Wrong exception')
    require('round2_attribution.py", line 123, in main' in text and "p.dump(out/'verification.json'" in text and text.count('Traceback (most recent call last):')==1,'Failure did not occur at the known final serializer')
    return text


def load_map(path,nonflat=True):
    arr=np.load(path,allow_pickle=False)
    require(arr.shape==(7,7) and np.issubdtype(arr.dtype,np.floating) and np.isfinite(arr).all() and arr.min()>=0 and arr.max()<=1,'Invalid saved native map')
    if nonflat:require(np.ptp(arr)>0 and arr.sum()>0,'Expected nonflat saved map')
    return arr


def validate_saved(source,cases,baseline,provenance,protocol,freeze,checkpoints,evaluation,suite):
    chosen=freeze['representative']
    expected_protocol=dict(representative=chosen,checkpoint_sha256=p.sha(checkpoints[chosen]),freeze_sha256=p.sha(suite/'round2_evaluation_freeze.json'),historical_sources=provenance,class_target='Each model own unmodified argmax; new target fixed through parameter randomization',clean_probability_tolerance=1e-4,randomization_seeds=[42,43,44],randomization_stages=['head','head_final','full'],case_count=115,prior_benchmark_known=True,clinical_anatomical_validation=False,source_code_sha256=p.code_hashes())
    require(protocol==expected_protocol,'Attribution protocol/selection/source identity mismatch')
    require(len(cases)==115 and len({r['case_id'] for r in cases})==115 and len(baseline)==230,'Incomplete historical fixed case set')
    truth={r['case_id']:r for r in cases};rows=p.stress.csv_read(source/'per_case.csv');by={r['case_id']:r for r in rows}
    require(len(rows)==115 and set(by)==set(truth),'Incomplete or duplicated115-case table')
    cloud=p.stress.csv_read(evaluation/f'{chosen}_original100_predictions.csv');cloud_by={r['path']:r for r in cloud}
    originals=[r for r in cases if r['role']=='reused_original100']
    require(len(cloud)==100 and len(cloud_by)==100 and set(cloud_by)=={r['path'] for r in originals},'Changed original100 replay membership')
    expected_files={'protocol_before_attribution.json','per_case.csv','native_map_mass.csv','randomization.csv','gallery.html'}
    expected_files|={stem+'.'+ext for stem in SUMMARY_STEMS for ext in ['png','pdf']}
    maps={};failures=[];valid_count=0
    for case in cases:
        key=case['case_id'];row=by[key]
        require(all(row[k]==v for k,v in case.items()) and row['model']==chosen,'Case metadata/model differs')
        prob=np.array([float(row['P_'+cl]) for cl in p.stress.STAGES]);p.stress.check_probs(prob[None])
        require(row['prediction']==p.stress.STAGES[int(prob.argmax())] and boolean(row['prediction_and_logits_unchanged']),'Prediction/state flag mismatch')
        near(row['confidence'],prob.max(),'confidence',1e-12)
        valid=boolean(row['cam_valid']);native_name=f'maps/{chosen}_{key}.npy'
        if valid:maps[key]=load_map(source/native_name);expected_files.add(native_name);valid_count+=1
        else:require(not (source/native_name).exists(),'Failed native map unexpectedly exists');maps[key]=None
        if case['role']=='reused_original100':
            previous=cloud_by[case['path']];prior=np.array([float(previous['P_'+cl]) for cl in p.stress.STAGES]);p.stress.check_probs(prior[None])
            require(previous['TrueCategory']==case['label'],'Replay source label mismatch')
            delta=float(np.abs(prob-prior).max());agreement=int(prob.argmax())==int(prior.argmax());near(row['cloud_max_probability_difference'],delta,'replay delta',1e-12)
            require(boolean(row['cloud_argmax_agreement'])==agreement,'Replay argmax flag mismatch')
            if delta>1e-4 or not agreement:failures.append(dict(case_id=key,max_probability_difference=delta,argmax_agreement=agreement))
        else:require(row['cloud_max_probability_difference']==row['cloud_argmax_agreement']=='','Unexpected replay fields for historical example')
        panel=f'cases/{key}.png';expected_files.add(panel)
        with Image.open(source/panel) as image:require(image.size==(1680,328) and image.format=='PNG','Panel dimensions/format changed');image.verify()
    mass=p.stress.csv_read(source/'native_map_mass.csv');mass_by={(r['case_id'],r['model']):r for r in mass}
    models=['old_baseline','round1_c3','round2_representative']
    require(len(mass)==345 and set(mass_by)=={(k,n) for k in truth for n in models},'Incomplete native-map mass grid')
    regions=p.corner.native_regions()
    for key,case in truth.items():
        for name in models:
            arr=maps[key] if name=='round2_representative' else baseline[key,name][1];row=mass_by[key,name]
            require(row['role']==case['role'] and boolean(row['valid'])==(arr is not None),'Native-mass metadata mismatch')
            if arr is None:require(all(row[k]=='' for k in ['corners4_mass','edge_ring24_mass','center25_mass','peak_in_any_corner']),'Failed native-map mass must remain missing');continue
            normalized=arr/arr.sum()
            for region,mask in regions.items():near(row[region+'_mass'],float(normalized[mask].sum()),'native '+region)
            require(boolean(row['peak_in_any_corner'])==bool(((arr==arr.max())&regions['corners4']).any()),'Native peak flag differs')
    random=p.stress.csv_read(source/'randomization.csv');controls={r['case_id'] for r in cases if r['role']=='historical_class_balanced_controls'}
    random_by={(r['case_id'],int(r['seed']),r['stage']):r for r in random};wanted={(k,seed,stage) for k in controls for seed in [42,43,44] for stage in ['head','head_final','full']}
    require(len(controls)==8 and len(random)==72 and set(random_by)==wanted,'Incomplete72-row randomization grid')
    random_failed=0
    for (key,seed,stage),row in random_by.items():
        require(row['target']==by[key]['prediction'],'Randomization target changed')
        name=f'randomized_maps/{key}_{seed}_{stage}.npy';path=source/name;arr=None
        if path.exists():arr=load_map(path,nonflat=False);expected_files.add(name)
        rho,iou=map_similarity(maps[key],arr) if maps[key] is not None and arr is not None else (None,None)
        require(row['status']==('valid' if rho is not None else 'flat_or_failed'),'Randomization validity differs')
        if rho is None:random_failed+=1;require(row['spearman']==row['top20_iou']=='','Failed randomization metrics must remain missing')
        else:near(row['spearman'],rho,'randomization rank');near(row['top20_iou'],iou,'randomization overlap')
    gallery=(source/'gallery.html').read_text();require(gallery.count('<img ')==115 and all(gallery.count('src="cases/'+r['case_id']+'.png"')==1 for r in cases),'Incomplete gallery panel references')
    require(chosen in gallery,'Gallery representative differs')
    for stem in SUMMARY_STEMS:
        with Image.open(source/(stem+'.png')) as image:require(image.format=='PNG' and min(image.size)>500,'Invalid summary PNG');image.verify()
        pdf=(source/(stem+'.pdf')).read_bytes()
        require(pdf.startswith(b'%PDF-') and pdf.rstrip().endswith(b'%%EOF') and len(re.findall(rb'/Type\s*/Page(?!s)\b',pdf))==1,'Invalid or non-single-page summary PDF')
    actual=inventory(source);require(set(actual)==expected_files,'Missing or unexpected attribution files: '+str(sorted(set(actual)^expected_files)))
    return dict(case_count=115,valid_maps=int(valid_count),all_logits_modes_gradients_unchanged=True,original_checkpoint_state_restored=True,randomization_records=72,cloud_probability_tolerance=1e-4,cloud_probability_or_argmax_failures=failures,historical_sources=provenance,clinical_anatomical_validation=False,visual_review_required=True),dict(saved_native_maps=valid_count,failed_native_maps=115-valid_count,randomization_records=72,failed_randomization_records=random_failed,replayed_original100_probabilities=100,recomputed_native_mass_rows=345,verified_historical_maps=230,panel_count=115,summary_pdf_png_pairs=2),actual


def recover(args):
    root=args.data_root.resolve();suite=args.suite.resolve();source=args.source.resolve();output=args.output.resolve();log=args.failure_log.resolve()
    require(not output.exists() and not args.output.is_symlink(),'Recovery output must be new')
    require(source.parent==suite and output.parent==suite and source!=output,'Source and recovery output must be distinct direct suite children')
    require(not args.source.is_symlink(),'Source directory cannot be a symlink')
    check_trace(log)
    require(p.sha(Path(a.__file__))==ATTRIBUTION_SHA and p.sha(root/'src/research_validation.py')==TRANSITIVE_SHA,'Known attribution or transitive source changed')
    before=inventory(source)
    require(not {'verification.json','SHA256SUMS.json','attribution_complete.json'}&set(before),'Source is not the exact unsealed final-serialization failure')
    evaluation,_=report.verified_outputs(suite);freeze,checkpoints=p.load_frozen(root,suite);cases,baseline,provenance=a.verified_cases(root)
    verification,checks,validated=validate_saved(source,cases,baseline,provenance,p.read(source/'protocol_before_attribution.json'),freeze,checkpoints,evaluation,suite)
    require(validated==before and inventory(source)==before,'Source changed during recovery validation')
    receipt=dict(status='postprocessing_only_serialization_recovery',created_utc=p.now(),source_directory=source.name,output_directory=output.name,helper_sha256=p.sha(Path(__file__)),frozen_attribution_source_sha256=ATTRIBUTION_SHA,transitive_source_sha256=TRANSITIVE_SHA,failure_log_sha256=TRACE_SHA,freeze_sha256=p.sha(suite/'round2_evaluation_freeze.json'),original_payload_sha256=before,independently_recomputed=checks,original_runtime_assertion_evidence=dict(basis='Exact preserved traceback reaches the known final serializer after all115 per-case checks, model-restoration assertion, frozen-source rechecks and both plot writes; the pinned source defines this control flow.',per_case_csv_flags_all_true=True,original_state_assertions_reexecuted=False,in_memory_model_state_available=False),no_new_inference=True,no_model_construction=True,no_checkpoint_or_source_change=True,no_probability_tolerance_change=True,original_directory_unchanged=True,visual_review_required=True)
    output.mkdir()
    for name,digest in before.items():
        target=output/name;target.parent.mkdir(parents=True,exist_ok=True)
        with (source/name).open('rb') as reader,target.open('xb') as writer:shutil.copyfileobj(reader,writer)
        require(p.sha(target)==digest,'Copied attribution payload changed')
    require(inventory(source)==before,'Source changed during copy')
    with (output/'serialization_failure.log').open('xb') as f:f.write(log.read_bytes())
    require(p.sha(output/'serialization_failure.log')==TRACE_SHA,'Failure evidence changed during copy')
    verification['recovery_receipt']='recovery_receipt.json';verification['state_assertion_basis']='Original runtime assertion evidence; artifact checks independently recomputed without inference.'
    p.dump(output/'recovery_receipt.json',receipt);p.dump(output/'verification.json',verification)
    payloads=inventory(output);p.dump(output/'SHA256SUMS.json',payloads)
    require(all(p.sha(output/n)==h for n,h in payloads.items()) and inventory(source)==before,'Final source/destination readback differs')
    marker=dict(status='blocked_cloud_probability_replay_review' if verification['cloud_probability_or_argmax_failures'] else 'complete_numerically_verified_visual_review_required',payloads=len(payloads),manifest_sha256=p.sha(output/'SHA256SUMS.json'),recovery_receipt_sha256=p.sha(output/'recovery_receipt.json'),postprocessing_recovery=True)
    p.dump(output/'attribution_complete.json',marker);return marker


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--data-root',type=Path,required=True);ap.add_argument('--suite',type=Path,required=True);ap.add_argument('--source',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--failure-log',type=Path,required=True);args=ap.parse_args();print(json.dumps(recover(args)))

if __name__=='__main__':main()
