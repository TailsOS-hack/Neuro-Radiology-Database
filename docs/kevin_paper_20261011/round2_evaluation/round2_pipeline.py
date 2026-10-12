#!/usr/bin/env python3
"""Freeze all round2 validation choices, then run one post-hoc image evaluation.

No training, clinical-head evaluation, outcome-based selection, or cloud actions.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,sys,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from PIL import Image

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
for folder in ['training','posthoc_robustness','round2_training','round2_corner_audit']:
    sys.path.insert(0,str(HERE.parent/folder))
import train_candidate as c
import evaluate_candidates as e
import stress_test as stress
import round2 as trainer
import corner_audit as corner

SEEDS=[20261011,20261012,20261013]
NEW=[f'round2_seed{s}' for s in SEEDS]
ORDER=['old_baseline','round1_c3']+NEW
R1='training_logs/experiments/kevin_20261011/runpod_blackwell_20261011'
PREP='training_logs/experiments/kevin_20261011/prepared'
INPUT_PINS={
 f'{PREP}/stage_manifest.csv':stress.PINS[f'{PREP}/stage_manifest.csv'],
 f'{PREP}/same_case_test_intersection.csv':stress.PINS[f'{PREP}/same_case_test_intersection.csv'],
 'data/evaluation/ground_truth/radiologist_test_key.csv':stress.PINS['data/evaluation/ground_truth/radiologist_test_key.csv'],
 'docs/kevin_paper_20261011/analysis/old_hash_group_unexposed_sensitivity.csv':'da1288c26944a10d7fdbda0ffc87886d77e48cb5ca2d7186a603e2492239bcba',
 'docs/kevin_paper_20261011/analysis/old_hash_group_unexposed_sensitivity_audit.json':'44db9789f8f5479fef700abfdeb45e1a1cb2cef487ccde2f5a2907b6c731576c',
 f'{R1}/frozen_evaluation_all6/original100_case_aligned_old_candidates_readers.csv':'219d2e5f9102e47a70b9ac4a2141b7aad9871a2b9dbed7a4673e12d38e717b8c',
 'docs/kevin_paper_20261011/round2_corner_audit/results/validation_sample.csv':'b0832e126b405c00c659c6f7e944eecaebc2ad90e6308f98c1e8e54e2f2fd8cb',
}

def sha(p):return stress.sha(p)
def read(p):return json.loads(Path(p).read_text())
def dump(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def now():return datetime.now(timezone.utc).isoformat()
def root_path(root,p):return Path(p) if Path(p).is_absolute() else root/Path(p)
def relative(root,p):return str(Path(p).resolve().relative_to(root.resolve()))
def code_hashes():
    files=[HERE/n for n in ['round2_pipeline.py','report_round2.py','round2_attribution.py','protocol.json']]
    files += [Path(c.__file__),Path(e.__file__),Path(stress.__file__),Path(corner.__file__),Path(trainer.__file__),ROOT/'src/grad_cam.py',ROOT/'src/explainability_validation.py',HERE.parent/'analysis/summarize_frozen_evaluation.py',HERE.parent/'analysis/analysis_protocol.json',HERE.parent/'analysis/plot_evaluation_figures.py']
    return {relative(ROOT,p):sha(p) for p in files}

def replay_selection(history,validations,baseline,cfg):
    if [r['epoch'] for r in history]!=list(range(len(history))):raise ValueError('Noncontiguous epoch history')
    best=None;epoch=None
    for row in history:
        v=validations[row['epoch']];actual=trainer.selection(v['clean'],v['stress'],baseline,cfg)
        if actual!=v['selection'] or abs(actual['score']-row['score'])>1e-12 or actual['eligible']!=row['eligible']:
            raise ValueError('Validation criterion/guard differs from history')
        improved=best is None or (actual['eligible'] and actual['score']<best-cfg['min_delta'])
        if row['selected']!=improved:raise ValueError('Saved min_delta checkpoint sequence is inconsistent')
        if improved:best=actual['score'];epoch=row['epoch']
    return epoch,best

def convergence_gate(status,selection,review,suite_status_sha,protocol_sha):
    required=bool(status.get('needs_validation_convergence_review') or selection.get('needs_validation_convergence_review') or any(r.get('needs_validation_convergence_review') for r in status.get('runs',{}).values()))
    if required and (not review or review.get('status')!='approved_after_validation_only_review' or review.get('suite_status_sha256')!=suite_status_sha or review.get('training_protocol_sha256')!=protocol_sha or review.get('heldout_outcomes_used') is not False or not review.get('rationale') or set(review.get('reviewed_seeds',[]))!=set(SEEDS)):
        raise ValueError('Validation convergence review required before any round2 test inference')

def verify_suite(root,suite,review=None):
    status=read(suite/'suite_status.json');protocol=read(suite/'protocol.json');suite_selection=read(suite/'validation_selection_before_any_round2_test.json')
    psha=sha(suite/'protocol.json')
    convergence_gate(status,suite_selection,review,sha(suite/'suite_status.json'),psha)
    if status.get('status')!='complete' or set(status.get('completed_seeds',[]))!=set(SEEDS) or set(status.get('runs',{}))!=set(map(str,SEEDS)):
        raise ValueError('Require all three completed research runs')
    if not status.get('research_only') or status.get('expected_seeds')!=SEEDS:raise ValueError('Invalid expected research suite')
    if not suite_selection.get('complete') or suite_selection.get('selection_used_only_fixed_validation') is not True or set(suite_selection['selections'])!=set(map(str,SEEDS)):
        raise ValueError('Incomplete validation-only suite selection')
    if status['selection_sha256']!=sha(suite/'validation_selection_before_any_round2_test.json') or status['protocol_sha256']!=psha or suite_selection['protocol_sha256']!=psha:
        raise ValueError('Suite selection/protocol hash mismatch')
    for path,digest in protocol['source_hashes'].items():
        if sha(root/path)!=digest:raise ValueError('Round2 training source changed: '+path)
    checked={};c.imports()
    for seed,name in zip(SEEDS,NEW):
        folder=suite/f'stage_only_seed{seed}';run=read(folder/'run_status.json');selection=read(folder/'selection.json');history=read(folder/'history.json');config=read(folder/'config.json')
        cp=folder/'best.pt';cp_sha=sha(cp)
        if run['status']!='complete' or run.get('smoke_only') or selection.get('smoke_only') or config.get('smoke'):raise ValueError('Incomplete or smoke run')
        if run!=status['runs'][str(seed)] or selection!=suite_selection['selections'][str(seed)]:raise ValueError('Per-run suite provenance mismatch')
        if any(v!=psha for v in [run['protocol_sha256'],selection['protocol_sha256'],config['protocol_sha256'],sha(folder/'protocol.json')]):raise ValueError('Per-run protocol drift')
        if cp_sha!=run['best_checkpoint_sha256'] or cp_sha!=selection['checkpoint_sha256'] or sha(folder/'selection.json')!=run['selection_sha256']:raise ValueError('Selected checkpoint digest mismatch')
        if config.get('initial_checkpoint_sha256')!=trainer.INIT_SHAS[seed] or selection.get('initial_checkpoint_sha256')!=trainer.INIT_SHAS[seed]:raise ValueError('Initialization identity differs')
        if config.get('round')!=2 or config.get('mode')!='stage_only' or config.get('seed')!=seed or config.get('test_used_for_selection') is not False or config['stage_manifest_sha256']!=trainer.STAGE_SHA:raise ValueError('Invalid round2 checkpoint provenance')
        validations={r['epoch']:read(folder/f'epoch_{r["epoch"]:03d}_validation.json') for r in history}
        epoch,score=replay_selection(history,validations,validations[0]['clean'],protocol['defaults'])
        selected=selection['selected']
        if epoch!=selected['epoch'] or epoch!=run['selected_epoch'] or len(history)-1!=run['completed_epochs'] or selected['selection']['score']!=score or not selected['selection']['eligible']:
            raise ValueError('Selected epoch not the validation-only optimum')
        if selected['clean']!=validations[epoch]['clean'] or selected['stress']!=validations[epoch]['stress']:raise ValueError('Selected validation metrics mismatch')
        model,aux,ck=e.load(cp,c.torch.device('cpu'))
        if ck.get('round')!=2 or ck['config']!=config or ck['epoch']!=epoch or ck['selection']!=selected['selection'] or ck['best_selection_score']!=score or ck['stage_validation']!=selected['clean'] or ck['stress_validation']!=selected['stress'] or ck['history']!=history[:epoch+1]:raise ValueError('Checkpoint metadata disagrees with frozen validation choice')
        del model,aux,ck
        checked[name]=dict(seed=seed,checkpoint=str(cp.relative_to(suite)),checkpoint_sha256=cp_sha,epoch=epoch,validation_score=score,epoch0_fallback=epoch==0,history_sha256=sha(folder/'history.json'),selection_sha256=sha(folder/'selection.json'))
    representative=min(checked,key=lambda n:(checked[n]['validation_score'],checked[n]['seed']))
    return checked,representative,psha

def freeze(args):
    root=args.data_root.resolve();suite=args.suite.resolve();output=suite/'round2_evaluation_freeze.json'
    def started():return output.exists() or (suite/'round2_evaluation_launch.json').exists() or (suite/'frozen_round2_evaluation').exists()
    if started():raise ValueError('Freeze/evaluation already exists; preserve it')
    review=read(args.convergence_review) if args.convergence_review else None
    checked,representative,psha=verify_suite(root,suite,review)
    c3=root_path(root,args.round1_c3);old=root/'models/alzheimers_classifier.pt'
    baselines={'old_baseline':dict(path=relative(root,old),sha256=stress.PINS['models/alzheimers_classifier.pt']),'round1_c3':dict(path=relative(root,c3),sha256=trainer.INIT_SHAS[20261012])}
    # The old baseline may arrive with the held-out add-on after this freeze.
    # Its known historical digest is pinned here and required at evaluation.
    for name,r in baselines.items():
        path=root/r['path']
        if (name=='round1_c3' or path.exists()) and sha(path)!=r['sha256']:raise ValueError('Baseline checkpoint mismatch')
    record=dict(status='frozen_before_any_round2_evaluation',created_utc=now(),all_three_runs=checked,representative=representative,representative_rule='Smallest eligible selected validation score; ties use ascending fixed seed.',training_protocol_sha256=psha,suite_status_sha256=sha(suite/'suite_status.json'),validation_selection_sha256=sha(suite/'validation_selection_before_any_round2_test.json'),source_code_sha256=code_hashes(),input_pins=INPUT_PINS,baselines=baselines,model_order=ORDER,protocol=read(HERE/'protocol.json'),prior_benchmarks_seen=True,clinical_head_evaluated=False,selection_used_only_validation=True,validation_convergence_review=review)
    if started():raise ValueError('Evaluation appeared while freezing')
    with output.open('x') as f:json.dump(record,f,indent=2);f.write('\n')
    print(json.dumps(dict(status='frozen',path=str(output),sha256=sha(output),representative=representative)))

def load_frozen(root,suite):
    f=suite/'round2_evaluation_freeze.json';record=read(f)
    if record.get('status')!='frozen_before_any_round2_evaluation' or record['source_code_sha256']!=code_hashes() or record['input_pins']!=INPUT_PINS:raise ValueError('Evaluation source/protocol changed after freeze')
    for file,key in [('suite_status.json','suite_status_sha256'),('validation_selection_before_any_round2_test.json','validation_selection_sha256'),('protocol.json','training_protocol_sha256')]:
        if sha(suite/file)!=record[key]:raise ValueError('Frozen training suite metadata changed: '+file)
    for name,r in record['all_three_runs'].items():
        folder=suite/Path(r['checkpoint']).parent
        if sha(folder/'history.json')!=r['history_sha256'] or sha(folder/'selection.json')!=r['selection_sha256']:raise ValueError('Frozen run metadata changed')
    for p,digest in INPUT_PINS.items():
        if sha(root/p)!=digest:raise ValueError('Pinned evaluation input changed: '+p)
    cps={n:root/r['path'] for n,r in record['baselines'].items()}
    cps.update({n:suite/r['checkpoint'] for n,r in record['all_three_runs'].items()})
    for name,path in cps.items():
        expected=record['baselines'][name]['sha256'] if name in record['baselines'] else record['all_three_runs'][name]['checkpoint_sha256']
        if sha(path)!=expected:raise ValueError('Frozen checkpoint changed: '+name)
    return record,cps

def rows_for(root):
    stage=stress.csv_read(root/f'{PREP}/stage_manifest.csv');by={r['path']:r for r in stage};full=[r for r in stage if r['split']=='test'];primary=stress.csv_read(root/f'{PREP}/same_case_test_intersection.csv');sensitivity=stress.csv_read(root/'docs/kevin_paper_20261011/analysis/old_hash_group_unexposed_sensitivity.csv')
    if len(full)!=5450 or len(primary)!=1058 or len(sensitivity)!=756:raise ValueError('Unexpected cohort counts')
    eligible={r['path'] for r in full if r['baseline_strict_split']=='test'}
    if {r['path'] for r in primary}!=eligible or not {r['path'] for r in sensitivity}<=eligible:raise ValueError('Invalid paired cohort membership')
    for row in primary+sensitivity:
        if row!=by[row['path']]:raise ValueError('Cohort source fields differ')
    gt=stress.csv_read(root/'data/evaluation/ground_truth/radiologist_test_key.csv')
    hundred=[dict(path='data/evaluation/images/'+r['FileName'],label=str(stress.STAGES.index(r['TrueCategory'])),FileName=r['FileName'],group_id='original100:'+r['FileName']) for r in gt]
    if len(hundred)!=100 or len({r['path'] for r in hundred})!=100:raise ValueError('Invalid original100')
    validation=stress.csv_read(root/'docs/kevin_paper_20261011/round2_corner_audit/results/validation_sample.csv')
    if len(validation)!=160 or len({r['group_id'] for r in validation})!=160 or any(r!=by.get(r['path']) or r['split']!='val' for r in validation):raise ValueError('Validation corner sample changed')
    return dict(full_new_test=full,internal_same_case=primary,old_hash_group_unexposed_sensitivity=sensitivity,original100=hundred,corner_validation=validation)

def export_predictions(out,name,cohort,rows,p):
    stress.check_probs(p)
    if len(p)!=len(rows) or len({r['path'] for r in rows})!=len(rows):raise ValueError('Prediction export alignment mismatch')
    records=[]
    for row,pp in zip(rows,p):
        r=dict(path=row['path'],TrueCategory=stress.STAGES[int(row['label'])],ModelPrediction=stress.STAGES[int(pp.argmax())],group_id=row['group_id']);r.update({'P_'+cl:float(v) for cl,v in zip(stress.STAGES,pp)})
        if cohort=='original100':r['FileName']=row['FileName']
        records.append(r)
    stress.csv_write(out/f'{name}_{cohort}_predictions.csv',records)

def evaluate(args):
    root=args.data_root.resolve();suite=args.suite.resolve();out=suite/'frozen_round2_evaluation';launch=suite/'round2_evaluation_launch.json'
    if out.exists() or launch.exists():raise ValueError('Evaluation already launched; no automatic rerun')
    if args.batch_size<1:raise ValueError('Positive batch size required')
    record,cps=load_frozen(root,suite);rows=rows_for(root);c.imports();device=c.device_for(args.device)
    c.torch.set_float32_matmul_precision('highest')
    if device.type=='cuda':c.torch.backends.cudnn.benchmark=False
    models={};states={}
    # Validate/load every checkpoint before opening held-out pixels.
    for name in ORDER:
        model,aux,ck=e.load(cps[name],device)
        if int(ck.get('image_size',224))!=224:raise ValueError('Checkpoint input size differs')
        model.eval();aux.eval();models[name]=(model,aux);states[name]={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};del ck
    if out.exists() or launch.exists():raise ValueError('Concurrent evaluation launch')
    with launch.open('x') as f:json.dump(dict(started_utc=now(),selection_freeze_sha256=sha(suite/'round2_evaluation_freeze.json'),source_code_sha256=code_hashes()),f,indent=2)
    out.mkdir();(out/'stress_predictions').mkdir();start=time.monotonic()
    from torchvision import transforms
    resize=transforms.Resize((224,224),antialias=True);norm=transforms.Compose([transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    imgs={};image_shas={}
    for row in rows['full_new_test']+rows['original100']+rows['corner_validation']:
        p=root/row['path'];image_shas[row['path']]=sha(p)
        with Image.open(p) as im:
            rgb=im.convert('RGB');pixel=hashlib.sha256(str(rgb.size).encode()+rgb.tobytes()).hexdigest()
            if row.get('pixel_sha256') and pixel!=row['pixel_sha256']:raise ValueError('Decoded image changed')
            imgs[row['path']]=resize(rgb)
    for row in [rows['full_new_test'][0],rows['original100'][-1]]:
        frozen=c.dataset([row],root,False,224)[0][0]
        if not c.torch.equal(norm(imgs[row['path']]),frozen):raise ValueError('Clean transform differs from frozen pipeline')
    def infer(name,tensor):
        values=[];model,aux=models[name]
        with c.torch.inference_mode():
            for i in range(0,len(tensor),args.batch_size):values.append(c.forward(model,aux,tensor[i:i+args.batch_size].to(device),'stage').softmax(1).cpu().numpy())
        p=np.concatenate(values).astype(float);stress.check_probs(p);return p
    clean={};metrics=[]
    for cohort in ['full_new_test','original100']:
        tensor=c.torch.stack([norm(imgs[r['path']]) for r in rows[cohort]])
        for name in ORDER:
            p=infer(name,tensor);clean[(name,cohort)]=p;export_predictions(out,name,cohort,rows[cohort],p)
        del tensor
    full_index={r['path']:i for i,r in enumerate(rows['full_new_test'])}
    for cohort in ['internal_same_case','old_hash_group_unexposed_sensitivity']:
        idx=[full_index[r['path']] for r in rows[cohort]]
        for name in ORDER:
            p=clean[(name,'full_new_test')][idx];clean[(name,cohort)]=p;export_predictions(out,name,cohort,rows[cohort],p)
    for (name,cohort),p in clean.items():metrics.append(dict(model=name,cohort=cohort,**stress.metrics(np.array([int(r['label']) for r in rows[cohort]]),p,p)))
    stress_rows=[dict(r,cohort=co) for co in ['internal_same_case','original100'] for r in rows[co]];stress_metrics=[]
    for scenario in stress.SCENARIOS:
        # Reuse the same-device clean probabilities; later scenarios add no selection.
        tensor=None if scenario['name']=='clean' else c.torch.stack([norm(stress.perturb(imgs[r['path']],scenario,r['path'])) for r in stress_rows])
        for name in ORDER:
            p=np.concatenate([clean[(name,co)] for co in ['internal_same_case','original100']]) if tensor is None else infer(name,tensor)
            exported=[]
            for row,pp in zip(stress_rows,p):exported.append(dict(cohort=row['cohort'],path=row['path'],TrueCategory=stress.STAGES[int(row['label'])],ModelPrediction=stress.STAGES[int(pp.argmax())],**{'P_'+cl:float(v) for cl,v in zip(stress.STAGES,pp)}))
            stress.csv_write(out/'stress_predictions'/f'{name}__{scenario["name"]}.csv',exported)
            for co in ['internal_same_case','original100']:
                mask=np.array([r['cohort']==co for r in stress_rows]);mt=stress.metrics(np.array([int(r['label']) for r in rows[co]]),p[mask],clean[(name,co)])
                stress_metrics.append(dict(model=name,cohort=co,scenario=scenario['name'],family=scenario['family'],value=scenario['value'],**mt))
        print(json.dumps(dict(phase='stress',scenario=scenario['name'],seconds=time.monotonic()-start)),flush=True)
    # Fixed validation-only causal corner controls; same160 sample as prior audit.
    x=c.torch.stack([norm(imgs[r['path']]) for r in rows['corner_validation']]);black=c.torch.tensor([-.485/.229,-.456/.224,-.406/.225])[None,:,None,None];masks=corner.masks();corner_metrics=[]
    scenarios=[('clean',None,None)]
    for region in ['corners32','border16']:
        for fill in ['black','imagenet_mean']:
            scenarios += [(region+'_erase_'+fill,masks[region],fill),(region+'_only_'+fill,~masks[region],fill),(region+'_matched_center_erase_'+fill,masks[region+'_matched_center'],fill)]
    (out/'corner_predictions').mkdir()
    for name in ORDER:
        cp=None
        for scen,mask,fill in scenarios:
            tensor=x if mask is None else c.torch.where(c.torch.from_numpy(mask)[None,None],black if fill=='black' else c.torch.zeros(1,3,1,1),x)
            p=infer(name,tensor)
            if cp is None:cp=p.copy()
            export_predictions(out/'corner_predictions',name,scen,rows['corner_validation'],p)
            corner_metrics.append(dict(model=name,scenario=scen,**stress.metrics(np.array([int(r['label']) for r in rows['corner_validation']]),p,cp)))
    checks={}
    for name,(model,aux) in models.items():
        if not all(c.torch.equal(v,model.state_dict()[k].cpu()) for k,v in states[name].items()) or any(p.grad is not None for p in model.parameters()):raise ValueError('Checkpoint state or gradients changed')
        checks[name]=dict(state_unchanged=True,checkpoint_sha256=sha(cps[name]))
    if any(sha(root/p)!=digest for p,digest in image_shas.items()):raise ValueError('Image changed during inference')
    load_frozen(root,suite)
    reader=stress.csv_read(root/f'{R1}/frozen_evaluation_all6/original100_case_aligned_old_candidates_readers.csv');reader_by={r['FileName']:r for r in reader};aligned=[]
    if set(reader_by)!={r['FileName'] for r in rows['original100']}:raise ValueError('Reader alignment mismatch')
    for i,row in enumerate(rows['original100']):
        old=reader_by[row['FileName']]
        if old['TrueCategory']!=stress.STAGES[int(row['label'])]:raise ValueError('Reader reference differs')
        r={k:v for k,v in old.items() if k in ['FileName','TrueCategory'] or k.startswith('Reader_')};r.update({name:stress.STAGES[int(clean[(name,'original100')][i].argmax())] for name in ORDER});aligned.append(r)
    stress.csv_write(out/'original100_case_aligned_models_readers.csv',aligned)
    dump(out/'clean_metrics.json',metrics);dump(out/'stress_metrics.json',stress_metrics);dump(out/'corner_metrics.json',corner_metrics)
    dump(out/'evaluation.json',dict(status='complete',created_utc=now(),selection_freeze_sha256=sha(suite/'round2_evaluation_freeze.json'),freeze=record,model_order=ORDER,checkpoint_checks=checks,cohort_sizes={k:len(v) for k,v in rows.items()},image_sha256=image_shas,source_code_sha256=code_hashes(),device=str(device),torch=c.torch.__version__,mixed_precision=False,elapsed_seconds=time.monotonic()-start,heldout_results_used_for_selection=False,all_benchmarks_retrospective=True,clinical_head_evaluated=False))
    outputs={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file()};dump(out/'SHA256SUMS.json',outputs)
    if any(sha(out/p)!=digest for p,digest in outputs.items()):raise ValueError('Output readback failed')
    dump(out/'evaluation_complete.json',dict(status='complete_verified',payloads=len(outputs),manifest_sha256=sha(out/'SHA256SUMS.json'),freeze_sha256=sha(suite/'round2_evaluation_freeze.json'),all_five_models_reported=True))
    print(json.dumps(read(out/'evaluation_complete.json')))

def main():
    p=argparse.ArgumentParser(description=__doc__);sp=p.add_subparsers(dest='command',required=True)
    for name in ['freeze','evaluate']:
        a=sp.add_parser(name);a.add_argument('--data-root',type=Path,default=ROOT);a.add_argument('--suite',type=Path,required=True)
        if name=='freeze':a.add_argument('--round1-c3',default=R1+'/stage_only_seed20261012/best.pt');a.add_argument('--convergence-review',type=Path)
        else:a.add_argument('--device',choices=['cpu','mps','cuda'],default='cuda');a.add_argument('--batch-size',type=int,default=64)
    args=p.parse_args();globals()[args.command](args)

if __name__=='__main__':main()
