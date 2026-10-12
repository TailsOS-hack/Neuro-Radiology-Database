#!/usr/bin/env python3
"""Guarded patient-level clinical comparison; separate from the legacy stage task."""
from __future__ import annotations
import argparse, collections, csv, hashlib, json, sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from PIL import Image, ImageFilter
import clinical_statistics as stats

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
SEEDS=[20261021,20261022,20261023]
NEW=[f'clinical_seed{s}' for s in SEEDS]
OLD={
 'round1_c2':('models/kevin_20261011/multitask_seed20261011.pt','dded15b8c2201e8f3d71fab1f0af547363a539f78b79db49f389337eef2654d9'),
 'round1_c4':('models/kevin_20261011/multitask_seed20261012.pt','9435ae63d0ccb38b9474d2aee85f063bcb48f207d88c6f83d6fc495f8b52beac'),
 'round1_c6':('models/kevin_20261011/multitask_seed20261013.pt','87d25d2e878caeb4aef33b99c210ebfdf303fc20058974b1a3c390d3aa5a39fe')}
ORDER=list(OLD)+NEW+['round1_clinical_ensemble','new_clinical_ensemble']
SCENARIOS=['clean','intensity_0.9','intensity_1.1','gaussian_noise_0.01','gaussian_noise_0.02','blur_0.5','blur_1.0']
OLD_MANIFEST='data/external/clinical_t1_preprocessed_v1/clinical_slices.csv'
HARVARD_MANIFEST='data/external/clinical_harvard29352_v1/clinical_slices.csv'
HARVARD_SHA='504d2b04d223637fcd92ee9072341b43a2f7ed4e44e7d0043b28c70137d8585d'
EVALUATION_MANIFEST_SHA='7275db1cefdc29790981cf4821110cf9a8dc6fda32c0bf273f708ecc8bb3ba74'
QC_BASE='docs/kevin_paper_20261011/round2_external_data/dataverse_29352'

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def dump(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def now():return datetime.now(timezone.utc).isoformat()
def csv_rows(p):
 with Path(p).open(newline='') as f:return list(csv.DictReader(f))
def safe(root,name):
 p=Path(name)
 if p.is_absolute() or '..' in p.parts or not (root/p).resolve().is_relative_to(root.resolve()) or (root/p).is_symlink():raise ValueError('Unsafe payload path')
 return root/p

def code_hashes():
 files=[HERE/n for n in ['clinical_compare.py','clinical_statistics.py','clinical_bag.py','evaluation_protocol.json','clinical_report.py','test_clinical_report.py']]
 files += [HERE.parent/'training/train_candidate.py',HERE.parent/'training/evaluate_candidates.py',HERE.parent/'preprocess_clinical_t1.py']
 return {str(p.relative_to(ROOT)):sha(p) for p in files}

def prepare(args):
 root=args.data_root.resolve();out=args.manifest.resolve()
 if out.exists():raise ValueError('Preserve existing evaluation manifest')
 if sha(root/HARVARD_MANIFEST)!=HARVARD_SHA:raise ValueError('Changed Harvard source manifest')
 qa=read(root/QC_BASE/'visual_qc.json');verified=read(root/QC_BASE/'output_verification.json')
 if qa.get('status')!='gross_all_subject_rendering_review_complete' or qa['subjects_reviewed']!=29 or qa['slices_reviewed']!=261 or qa['excluded_subjects']!=0 or qa['model_outcomes_inspected']:
  raise ValueError('Harvard pre-inference gross-QC gate not met')
 if verified.get('status')!='passed_independent_output_readback' or verified['manifest_sha256']!=HARVARD_SHA or not verified['source_diagnosis_and_hash_join_verified']:
  raise ValueError('Provider join/integrity gate not met')
 groups=collections.defaultdict(list)
 for source in [OLD_MANIFEST,HARVARD_MANIFEST]:
  for r in csv_rows(root/source):
   if source==OLD_MANIFEST and r['split'] not in {'test','external'}:continue
   groups[r['group_key']].append(r)
 patients=[];seen_pixels=collections.defaultdict(set)
 for key,rows in sorted(groups.items()):
  if len(rows)!=9 or any(len({r[f] for r in rows})!=1 for f in ['cohort','subject_id','diagnosis','binary_label','split','preprocessing_version']):raise ValueError('Incomplete or inconsistent patient bag')
  rows.sort(key=lambda r:float(r['slice_fraction']));r=rows[0]
  if [float(v['slice_fraction']) for v in rows]!=[.45,.5,.55,.6,.65,.7,.75,.8,.85] or len({v['slice_index'] for v in rows})!=9:raise ValueError('Changed fixed slice selection')
  diagnosis=r['diagnosis'];label={'CN':0,'AD':1,'MCI':-1}.get(diagnosis)
  if label is None or r['binary_label']!=('' if label==-1 else str(label)):raise ValueError('Invalid diagnosis mapping')
  mapping={'besga_zenodo_3935636':'besga_reused_test','openneuro_ds007561':'openneuro_reused_external','harvard_dataverse_29352':'harvard_fresh_external'}
  if r['cohort'] not in mapping or r['preprocessing_version']!='exploratory_t1_axial_v1':raise ValueError('Unknown clinical source or preprocessing')
  cohort=mapping[r['cohort']]
  if (cohort=='besga_reused_test' and r['split']!='test') or (cohort!='besga_reused_test' and r['split']!='external'):raise ValueError('Patient entered wrong role')
  paths=[];byte_shas=[];pixel_shas=[]
  for v in rows:
   path='data/'+v['path'];p=safe(root,path)
   with Image.open(p) as im:
    a=np.asarray(im.convert('RGB'))
   if a.shape!=(224,224,3) or not np.array_equal(a[:,:,0],a[:,:,1]) or not np.array_equal(a[:,:,0],a[:,:,2]) or a.max()==a.min():raise ValueError('Invalid prepared MRI slice')
   ph=hashlib.sha256(a.tobytes()).hexdigest();seen_pixels[ph].add(key)
   paths.append(path);byte_shas.append(sha(p));pixel_shas.append(ph)
  patients.append(dict(group_key=key,subject_id=r['subject_id'],cohort=r['cohort'],evaluation_cohort=cohort,label=label,diagnosis=diagnosis,split=r['split'],paths=paths,sha256=byte_shas,pixel_sha256=pixel_shas))
 if any(len(v)>1 for v in seen_pixels.values()):raise ValueError('Cross-patient exact pixel duplicate')
 counts=collections.Counter((r['evaluation_cohort'],r['diagnosis']) for r in patients)
 expected={('besga_reused_test','AD'):7,('besga_reused_test','CN'):4,('openneuro_reused_external','AD'):1,('openneuro_reused_external','CN'):14,('openneuro_reused_external','MCI'):5,('harvard_fresh_external','AD'):9,('harvard_fresh_external','CN'):10,('harvard_fresh_external','MCI'):10}
 if counts!=expected or len(patients)!=60:raise ValueError('Unexpected patient denominator')
 pins={p:sha(root/p) for p in [OLD_MANIFEST,HARVARD_MANIFEST,QC_BASE+'/visual_qc.json',QC_BASE+'/output_verification.json',QC_BASE+'/preprocessing_summary.json',QC_BASE+'/acquisition_summary.json']}
 out.parent.mkdir(parents=True,exist_ok=True)
 dump(out,dict(created_utc=now(),status='prepared_without_model_inference',patients=patients,source_sha256=pins,counts=[dict(cohort=c,diagnosis=d,n=n) for (c,d),n in sorted(counts.items())],model_inference_performed=False))
 print(json.dumps(dict(status='prepared',patients=60,slices=540,manifest_sha256=sha(out))))

def verify_manifest(root,path,require_images):
 if sha(path)!=EVALUATION_MANIFEST_SHA:raise ValueError('Prepared clinical evaluation membership/labels/QC changed')
 manifest=read(path)
 if manifest.get('status')!='prepared_without_model_inference' or len(manifest['patients'])!=60:raise ValueError('Invalid prepared heldout manifest')
 if len({r['group_key'] for r in manifest['patients']})!=60:raise ValueError('Duplicate patient identities')
 for p,h in manifest['source_sha256'].items():
  if require_images and sha(safe(root,p))!=h:raise ValueError('Frozen clinical source/QC changed')
 for r in manifest['patients']:
  if len(r['paths'])!=9 or len(r['sha256'])!=9 or len(r['pixel_sha256'])!=9:raise ValueError('Incomplete bag')
  if require_images:
   for p,h,ph in zip(r['paths'],r['sha256'],r['pixel_sha256']):
    path=safe(root,p)
    if sha(path)!=h:raise ValueError('Clinical image bytes changed')
    with Image.open(path) as im:a=np.asarray(im.convert('RGB'))
    if a.shape!=(224,224,3) or hashlib.sha256(a.tobytes()).hexdigest()!=ph:raise ValueError('Clinical pixels changed')
 return manifest

def bag_tensor(row,root,scenario):
 import torch
 xs=[]
 for j,path in enumerate(row['paths']):
  with Image.open(root/path) as im:
   im=im.convert('RGB')
   if scenario.startswith('blur_'):im=im.filter(ImageFilter.GaussianBlur(float(scenario.split('_')[-1])))
   a=np.asarray(im,dtype=np.float32)/255.
  if scenario.startswith('intensity_'):a=np.clip(a*float(scenario.split('_')[-1]),0,1)
  if scenario.startswith('gaussian_noise_'):
   seed=int.from_bytes(hashlib.sha256(f'{row["group_key"]}|{j}|{scenario}|20261021'.encode()).digest()[:8],'big')
   noise=np.random.default_rng(seed).normal(0,float(scenario.split('_')[-1]),a.shape[:2]+(1,)).astype(np.float32)
   a=np.clip(a+noise,0,1)
  x=torch.from_numpy(a.copy()).permute(2,0,1)
  xs.append((x-torch.tensor([.485,.456,.406])[:,None,None])/torch.tensor([.229,.224,.225])[:,None,None])
 return torch.stack(xs)

def freeze(args):
 import clinical_bag as cb
 root=args.data_root.resolve();suite=args.suite.resolve();target=suite/'clinical_evaluation_freeze.json'
 if target.exists() or (suite/'clinical_evaluation_launch.json').exists() or (suite/'clinical_evaluation').exists():raise ValueError('Existing clinical freeze/evaluation must be preserved')
 checked=cb.verify_completed_suite(suite,args.prepared.resolve(),root)
 if checked.get('status')!='verified_complete_converged_no_inference' or {r['seed'] for r in checked['selections']}!=set(SEEDS):raise ValueError('Require verified complete clinical suite')
 manifest=verify_manifest(root,args.manifest,False)
 selections={}
 for row in checked['selections']:
  cp=Path(row['checkpoint_path'])
  if not cp.is_absolute():cp=suite/cp
  if sha(cp)!=row['checkpoint_sha256']:raise ValueError('Clinical selection checksum mismatch')
  selections[f'clinical_seed{row["seed"]}']={**row,'checkpoint_path':str(cp.relative_to(suite))}
 record=dict(status='frozen_before_clinical_heldout_inference',created_utc=now(),training_verification={k:v for k,v in checked.items() if k!='selections'},
   selections=selections,source_code_sha256=code_hashes(),evaluation_manifest_sha256=sha(args.manifest),input_source_sha256=manifest['source_sha256'],
   protocol=read(HERE/'evaluation_protocol.json'),old_checkpoints={n:dict(path=p,sha256=h) for n,(p,h) in OLD.items()},
   primary_new='new_clinical_ensemble',primary_comparator='round1_clinical_ensemble',representative_seed=checked['representative_seed'],heldout_outcomes_used_for_selection=False)
 with target.open('x') as f:json.dump(record,f,indent=2,allow_nan=False);f.write('\n')
 print(json.dumps(dict(status='frozen',freeze_sha256=sha(target),path=str(target))))

def load_frozen(root,suite,prepared,manifest_path,images):
 import clinical_bag as cb
 f=read(suite/'clinical_evaluation_freeze.json')
 if f.get('status')!='frozen_before_clinical_heldout_inference' or f['source_code_sha256']!=code_hashes() or f['evaluation_manifest_sha256']!=sha(manifest_path) or f['protocol']!=read(HERE/'evaluation_protocol.json'):
  raise ValueError('Clinical frozen code/protocol/manifest changed')
 checked=cb.verify_completed_suite(suite,prepared,root)
 if {k:v for k,v in checked.items() if k!='selections'}!=f['training_verification']:raise ValueError('Clinical training verification changed')
 expected={}
 for row in checked['selections']:
  cp=Path(row['checkpoint_path']);cp=cp if cp.is_absolute() else suite/cp
  expected[f'clinical_seed{row["seed"]}']={**row,'checkpoint_path':str(cp.relative_to(suite))}
 if expected!=f['selections']:raise ValueError('Clinical selected checkpoint metadata changed')
 manifest=verify_manifest(root,manifest_path,images)
 if manifest['source_sha256']!=f['input_source_sha256']:raise ValueError('Clinical source/QC identities changed')
 cps={n:suite/r['checkpoint_path'] for n,r in f['selections'].items()}
 cps.update({n:root/r['path'] for n,r in f['old_checkpoints'].items()})
 if images:
  for n,cp in cps.items():
   wanted=f['selections'][n]['checkpoint_sha256'] if n in NEW else f['old_checkpoints'][n]['sha256']
   if sha(cp)!=wanted:raise ValueError('Frozen clinical comparator/checkpoint changed')
 return f,manifest,cps

def evaluate(args):
 import torch
 import clinical_bag as cb
 sys.path.insert(0,str(HERE.parent/'training'))
 import train_candidate as c
 import evaluate_candidates as e
 root=args.data_root.resolve();suite=args.suite.resolve();out=suite/'clinical_evaluation';launch=suite/'clinical_evaluation_launch.json'
 if out.exists() or launch.exists():raise ValueError('Refuse repeated clinical test inference')
 f,manifest,cps=load_frozen(root,suite,args.prepared.resolve(),args.manifest,True)
 rows=manifest['patients'];device=torch.device(args.device)
 if device.type=='cuda' and not torch.cuda.is_available():raise ValueError('Requested CUDA is unavailable')
 torch.set_num_threads(4);torch.set_float32_matmul_precision('highest');torch.backends.cudnn.benchmark=False
 # Verify the exact unperturbed transform independently, for every patient.
 for r in rows:
  if not torch.equal(bag_tensor(r,root,'clean'),cb.preprocess_bag(r,root,train=False)):raise ValueError('Clinical evaluation and training-clean transforms differ')
 with launch.open('x') as handle:json.dump(dict(started_utc=now(),freeze_sha256=sha(suite/'clinical_evaluation_freeze.json'),device=str(device),torch=torch.__version__,cuda=torch.version.cuda),handle,indent=2)
 out.mkdir();c.imports();predictions=[];probabilities={};states=[]
 for name in list(OLD)+NEW:
  if name in OLD:
   model,aux,ck=e.load(cps[name],device)
   if ck['config']['mode']!='multitask':raise ValueError('Old clinical auxiliary head was not trained')
   aux.eval()
  else:
   ck=torch.load(cps[name],map_location='cpu',weights_only=False)
   model=cb.build_model();model.load_state_dict(ck['model']);model.to(device);aux=None
  model.eval();before={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
  aux_before={k:v.detach().cpu().clone() for k,v in aux.state_dict().items()} if aux is not None else {}
  for scenario in SCENARIOS:
   pp=[]
   with torch.inference_mode():
    for r in rows:
     x=bag_tensor(r,root,scenario).to(device)
     logits=c.forward(model,aux,x,'aux') if name in OLD else model(x)
     # Preserve the historical Python-list / NumPy float64 slice pooling exactly.
     if name in OLD:
      prob=logits.softmax(1).detach().cpu().numpy().astype(np.float64).mean(axis=0)
     else:
      prob=logits.mean(0).softmax(0).detach().cpu().numpy()
     pp.append(prob)
   pp=np.array(pp)
   if pp.shape!=(60,2) or not np.isfinite(pp).all() or np.any(pp<0) or np.any(pp>1) or not np.allclose(pp.sum(1),1,atol=1e-6,rtol=0):raise ValueError('Invalid patient probability vectors')
   if name in OLD and (np.any(pp[:,1]==.5) or np.any((pp[:,1]>=.5)!=pp.argmax(1))):raise ValueError('Old comparator decision differs from historical argmax; explicit review required')
   probabilities[name,scenario]=pp
   print(json.dumps(dict(model=name,scenario=scenario,patients=len(rows),inference_complete=True)),flush=True)
  if any(not torch.equal(v,model.state_dict()[k].detach().cpu()) for k,v in before.items()) or any(p.grad is not None for p in model.parameters()):raise ValueError('Clinical inference changed model state')
  if aux is not None and (any(not torch.equal(v,aux.state_dict()[k].detach().cpu()) for k,v in aux_before.items()) or any(p.grad is not None for p in aux.parameters())):raise ValueError('Clinical inference changed old auxiliary state')
  states.append(dict(model=name,checkpoint_sha256=sha(cps[name]),all_states_unchanged=True,no_parameter_gradients=True))
  del model,aux,ck
  if device.type=='cuda':torch.cuda.empty_cache()
 for scenario in SCENARIOS:
  probabilities['round1_clinical_ensemble',scenario]=np.mean([probabilities[n,scenario] for n in OLD],axis=0)
  probabilities['new_clinical_ensemble',scenario]=np.mean([probabilities[n,scenario] for n in NEW],axis=0)
  old_ensemble=probabilities['round1_clinical_ensemble',scenario]
  if np.any(old_ensemble[:,1]==.5) or np.any((old_ensemble[:,1]>=.5)!=old_ensemble.argmax(1)):raise ValueError('Old ensemble decision differs from historical argmax; review required')
 metrics=[]
 for name in ORDER:
  for scenario in SCENARIOS:
   pp=probabilities[name,scenario]
   for r,prob in zip(rows,pp):predictions.append(dict(group_key=r['group_key'],cohort=r['evaluation_cohort'],diagnosis=r['diagnosis'],label=r['label'],model=name,scenario=scenario,n_slices=9,probabilities=prob.tolist(),predicted_ad=bool(prob[1]>=.5),binary_evaluable=r['label']>=0))
   for cohort in f['protocol']['cohorts']:
    indices=[i for i,r in enumerate(rows) if r['evaluation_cohort']==cohort and r['label']>=0]
    y=np.array([rows[i]['label'] for i in indices]);m=stats.metrics(y,pp[indices]);clean=probabilities[name,'clean'][indices]
    m.update(mean_probability_total_variation=float(np.abs(pp[indices]-clean).sum(1).mean()/2),label_flip_fraction=float(np.mean((pp[indices,1]>=.5)!=(clean[:,1]>=.5))))
    metrics.append(dict(model=name,scenario=scenario,cohort=cohort,**m))
 if len(predictions)!=3360 or len(metrics)!=168:raise ValueError('Incomplete clinical inference grid')
 load_frozen(root,suite,args.prepared.resolve(),args.manifest,True)
 dump(out/'patient_predictions.json',predictions);dump(out/'metrics.json',metrics);dump(out/'model_state_verification.json',states)
 dump(out/'evaluation.json',dict(status='complete',freeze=f,freeze_sha256=sha(suite/'clinical_evaluation_freeze.json'),model_order=ORDER,scenarios=SCENARIOS,total_patient_scenario_records=len(predictions),metric_records=len(metrics),fresh_binary_patients=19,mci_not_scored=True,old_probability_ties=0,no_training_or_selection=True))
 inventory={p.name:sha(p) for p in out.iterdir() if p.is_file()};dump(out/'SHA256SUMS.json',inventory)
 for name,h in inventory.items():
  if sha(out/name)!=h:raise ValueError('Clinical output readback failed')
 dump(out/'evaluation_complete.json',dict(status='complete_verified',payloads=len(inventory),manifest_sha256=sha(out/'SHA256SUMS.json'),completed_utc=now()))
 print(json.dumps(read(out/'evaluation_complete.json')),flush=True)

def main():
 ap=argparse.ArgumentParser(description=__doc__);sp=ap.add_subparsers(dest='command',required=True)
 a=sp.add_parser('prepare');a.add_argument('--data-root',type=Path,default=ROOT);a.add_argument('--manifest',type=Path,required=True)
 for name in ['freeze','evaluate']:
  a=sp.add_parser(name);a.add_argument('--data-root',type=Path,default=ROOT);a.add_argument('--manifest',type=Path,required=True);a.add_argument('--suite',type=Path,required=True);a.add_argument('--prepared',type=Path,required=True)
  if name=='evaluate':a.add_argument('--device',choices=['cpu','mps','cuda'],default='cuda')
 args=ap.parse_args();globals()[args.command](args)

if __name__=='__main__':main()
