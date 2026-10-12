#!/usr/bin/env python3
"""Descriptive native-map audit and fixed validation-only input controls.

No fitting, checkpoint selection, held-out inference, or edits to prior results.
"""
from pathlib import Path
import argparse,csv,hashlib,json,sys,time
import numpy as np
from PIL import Image,ImageDraw,ImageFont

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(HERE.parent/'training'))
sys.path.insert(0,str(HERE.parent/'posthoc_robustness'))
import stress_test as s
import train_candidate as c
import evaluate_candidates as e

SEED=20261012
N_PER_CLASS=40
MODELS=['old_baseline','candidate_3']

def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def rel(path):return str(Path(path).resolve().relative_to(ROOT))

def masks(n=224):
    corner=np.zeros((n,n),bool)
    for a,b in [(0,0),(0,n-32),(n-32,0),(n-32,n-32)]:corner[a:a+32,b:b+32]=True
    border=np.ones((n,n),bool);border[16:-16,16:-16]=False
    yy,xx=np.indices((n,n));distance=(yy-(n-1)/2)**2+(xx-(n-1)/2)**2
    order=np.argsort(distance.ravel(),kind='stable');result={'corners32':corner,'border16':border}
    for name,mask in list(result.items()):
        center=np.zeros(n*n,bool);center[order[:int(mask.sum())]]=True;result[name+'_matched_center']=center.reshape(n,n)
    return result

def native_regions():
    corner=np.zeros((7,7),bool);corner[0,0]=corner[0,-1]=corner[-1,0]=corner[-1,-1]=True
    edge=np.ones((7,7),bool);edge[1:-1,1:-1]=False
    return {'corners4':corner,'edge_ring24':edge,'center25':~edge}

def sample(rows):
    chosen=[];used=set()
    for label in range(4):
        available=sorted([r for r in rows if r['split']=='validation' and int(r['label'])==label],key=lambda r:hashlib.sha256(f'{SEED}|{r["path"]}'.encode()).hexdigest())
        # Current prepared manifests use the short split spelling.
        if not available:available=sorted([r for r in rows if r['split']=='val' and int(r['label'])==label],key=lambda r:hashlib.sha256(f'{SEED}|{r["path"]}'.encode()).hexdigest())
        take=[]
        for r in available:
            if r['group_id'] in used:continue
            take.append(r);used.add(r['group_id'])
            if len(take)==N_PER_CLASS:break
        if len(take)!=N_PER_CLASS:raise ValueError('Insufficient validation groups')
        chosen+=take
    assert len({r['path'] for r in chosen})==160 and len(used)==160
    return chosen

def map_audit(out,source):
    inventory=json.loads((source/'SHA256SUMS.json').read_text());rows=s.csv_read(source/'per_case.csv')
    expected={f'{m}_{r["case_id"]}.npy' for m in MODELS for r in rows if r['model']==m}
    if len(rows)!=230 or len(expected)!=230:raise ValueError('Unexpected historical map set')
    results=[];regions=native_regions();cases={};inputs={rel(source/'SHA256SUMS.json'):s.sha(source/'SHA256SUMS.json')}
    for r in rows:
        file=source/'maps'/f'{r["model"]}_{r["case_id"]}.npy'
        if s.sha(file)!=inventory[str(file.relative_to(source))]:raise ValueError('Archived native map changed')
        a=np.load(file,allow_pickle=False)
        if a.shape!=(7,7) or not np.isfinite(a).all() or a.min()<0 or a.sum()<=0:raise ValueError('Invalid native map')
        mass=a/a.sum();peak=np.unravel_index(a.argmax(),a.shape)
        rec={k:r[k] for k in ['case_id','role','path','model','prediction','confidence']}
        rec.update({k+'_mass':float(mass[mask].sum()) for k,mask in regions.items()})
        rec.update({k+'_area_enrichment':float(mass[mask].sum()/mask.mean()) for k,mask in regions.items()})
        rec.update(peak_row=int(peak[0]),peak_col=int(peak[1]),peak_in_corner=bool(regions['corners4'][peak]),peak_in_edge=bool(regions['edge_ring24'][peak]))
        for name,index in [('top_left',(0,0)),('top_right',(0,6)),('bottom_left',(6,0)),('bottom_right',(6,6))]:rec[name+'_mass']=float(mass[index])
        results.append(rec);cases.setdefault(r['case_id'],{'record':r})[r['model']]=a
        inputs[rel(file)]=s.sha(file)
    s.csv_write(out/'native_map_mass.csv',results)
    groups=[]
    for role in sorted({r['role'] for r in results}):
        for model in MODELS:
            values=[r for r in results if r['role']==role and r['model']==model]
            groups.append(dict(role=role,model=model,n=len(values),**{k:float(np.mean([r[k] for r in values])) for k in values[0] if k.endswith('_mass') or k.endswith('_area_enrichment') or k.startswith('peak_in_')}))
    s.csv_write(out/'native_map_summary.csv',groups)
    # Regenerate every fixed case from verified native arrays; no model inference.
    import matplotlib;matplotlib.use('Agg')
    from matplotlib import colormaps
    lut=colormaps['inferno'];font=ImageFont.load_default(size=16)
    (out/'native_gallery').mkdir()
    for case_id,item in cases.items():
        r=item['record'];file=ROOT/r['path']
        if s.sha(file)!=r['image_sha256']:raise ValueError('Source image changed')
        im=Image.open(file).convert('RGB').resize((224,224));panel=Image.new('RGB',(1200,304),'white');d=ImageDraw.Draw(panel)
        d.text((5,4),f'{case_id} | {r["label"]} | {r["role"]}',fill='black',font=font)
        columns=[('Input',im)]
        for model in MODELS:
            a=item[model];rgb=(lut(a)[...,:3]*255).astype(np.uint8)
            coarse=Image.fromarray(rgb).resize((224,224),Image.Resampling.NEAREST)
            dc=ImageDraw.Draw(coarse)
            for j in range(1,7):dc.line((j*32,0,j*32,223),fill='white',width=1);dc.line((0,j*32,223,j*32),fill='white',width=1)
            for x,y in [(0,0),(192,0),(0,192),(192,192)]:dc.rectangle((x,y,x+31,y+31),outline='#00ffff',width=2)
            smooth=Image.fromarray(rgb).resize((224,224),Image.Resampling.BILINEAR)
            columns += [(model+' native 7x7',coarse),(model+' display overlay',Image.blend(im,smooth,.38))]
        for j,(title,pic) in enumerate(columns):d.text((j*240+5,29),title,fill='black',font=ImageFont.load_default(size=13));panel.paste(pic,(j*240+8,51))
        d.text((5,283),'Per-map normalized scores; cyan = native corner cells. Coarse receptive fields are not anatomical localization.',fill='black',font=ImageFont.load_default(size=14))
        panel.save(out/'native_gallery'/f'{case_id}.png')
    html='<meta charset="utf-8"><h1>Native 7×7 attribution: every fixed historical case</h1><p>Maps are reused, hash-verified arrays; no held-out inference. Cyan outlines show native corner cells. Upsampling is display only, not fine anatomical evidence.</p>'
    for case_id in cases:html+=f'<h2>{case_id}</h2><img src="native_gallery/{case_id}.png" width="1200" alt="{case_id} native maps and overlays">'
    (out/'native_gallery.html').write_text(html)
    return groups,inputs

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--device',choices=['cpu','mps'],default='mps');parser.add_argument('--output',type=Path,default=HERE/'results');args=parser.parse_args();out=args.output.resolve()
    if out.exists():raise ValueError('Output must be new')
    c.imports();import torch
    if args.device=='mps' and not torch.backends.mps.is_available():raise ValueError('MPS unavailable')
    torch.set_num_threads(4);torch.manual_seed(SEED);device=torch.device(args.device)
    resultroot=ROOT/'training_logs/experiments/kevin_20261011/runpod_blackwell_20261011'
    selection=json.loads((resultroot/'validation_selection_before_evaluation.json').read_text());chosen=selection['candidates']['candidate_3']
    assert selection['exploratory_representative_candidate']=='candidate_3' and chosen['mode']=='stage_only' and chosen['seed']==20261012
    checkpoints={'old_baseline':ROOT/'models/alzheimers_classifier.pt','candidate_3':resultroot/'stage_only_seed20261012/best.pt'}
    pins={'old_baseline':s.PINS['models/alzheimers_classifier.pt'],'candidate_3':chosen['checkpoint_sha256']}
    if any(s.sha(p)!=pins[m] for m,p in checkpoints.items()):raise ValueError('Checkpoint SHA mismatch')
    manifest=ROOT/'training_logs/experiments/kevin_20261011/prepared/stage_manifest.csv'
    if s.sha(manifest)!=s.PINS[str(manifest.relative_to(ROOT))]:raise ValueError('Manifest changed')
    allrows=s.csv_read(manifest);rows=sample(allrows);ms=masks();scenarios=[('clean',None,None)]
    for region in ['corners32','border16']:
        for fill in ['black','imagenet_mean']:
            scenarios += [(region+'_erase_'+fill,ms[region],fill),(region+'_only_'+fill,~ms[region],fill),(region+'_matched_center_erase_'+fill,ms[region+'_matched_center'],fill)]
    protocol=dict(seed=SEED,n=160,class_order=s.STAGES,per_class=40,unique_hash_groups=160,selection='SHA256(seed|path) ordering, one new-validation image per prepared hash group per class; no outcome-based sample selection',models=pins,scenarios=[dict(name=n,masked_fraction=0 if m is None else float(m.mean()),fill=f) for n,m,f in scenarios],evaluation_only=True,random_control='Not added: its four-class performance would not meaningfully test learned shortcut reliance. Existing parameter-randomization controls remain available.',limitations=['Post-hoc validation diagnostic, not a new clinical or independent patient test.','New validation images may have old-baseline development exposure; comparisons are descriptive.','Native 7x7 map mass uses per-map min/max normalized nonnegative scores; it is not an anatomical probability.','Native corner cells have broad overlapping receptive fields; high corner map mass alone does not prove input-corner reliance.','A nonblack-pixel threshold is only a display support proxy, not a brain mask.','Erasure creates out-of-distribution images; location and fill controls help but cannot prove absence of all shortcuts.'],stage_manifest_sha256=s.sha(manifest),script_sha256=s.sha(__file__))
    out.mkdir(parents=True);write(out/'protocol_before_inference.json',protocol);s.csv_write(out/'validation_sample.csv',rows)
    y=np.array([int(r['label']) for r in rows]);tf=c.dataset([],ROOT,False,224).tf
    xs=[];support=[];source_shas={rel(p):s.sha(p) for p in [manifest,*checkpoints.values(),Path(c.__file__),Path(e.__file__),Path(s.__file__)]}
    for r in rows:
        p=ROOT/r['path'];im=Image.open(p).convert('RGB');decoded=hashlib.sha256(str(im.size).encode()+im.tobytes()).hexdigest()
        if decoded!=r['pixel_sha256']:raise ValueError('Validation source changed')
        x=tf(im);xs.append(x);raw=x*np.array([.229,.224,.225],dtype=np.float32)[:,None,None]+np.array([.485,.456,.406],dtype=np.float32)[:,None,None]
        bright=raw.mean(0).numpy()>.08
        support.append(dict(path=r['path'],**{region+'_nonblack_fraction':float(bright[mask].mean()) for region,mask in ms.items()},whole_nonblack_fraction=float(bright.mean())))
        source_shas[rel(p)]=s.sha(p)
    x=torch.stack(xs);s.csv_write(out/'validation_display_support.csv',support)
    black=torch.tensor([-.485/.229,-.456/.224,-.406/.225],dtype=x.dtype)[None,:,None,None]
    allpred=[];metrics=[];model_checks={};t=time.monotonic()
    for name,path in checkpoints.items():
        model,aux,ck=e.load(path,device);model.eval();before={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};clean=None
        for scen,mask,fill in scenarios:
            batch=x if mask is None else torch.where(torch.from_numpy(mask)[None,None],black if fill=='black' else torch.zeros(1,3,1,1),x)
            values=[]
            with torch.inference_mode():
                for j in range(0,len(batch),32):values.append(model(batch[j:j+32].to(device)).softmax(1).cpu().numpy())
            prob=np.concatenate(values);s.check_probs(prob)
            if clean is None:clean=prob.copy()
            mt=s.metrics(y,prob,clean);mt.update(model=name,scenario=scen,mean_clean_target_probability_drop=float(np.mean(clean.max(1)-prob[np.arange(len(y)),clean.argmax(1)])))
            metrics.append(mt)
            for r,pp,cp in zip(rows,prob,clean):allpred.append(dict(model=name,scenario=scen,path=r['path'],TrueCategory=s.STAGES[int(r['label'])],ModelPrediction=s.STAGES[int(pp.argmax())],clean_prediction=s.STAGES[int(cp.argmax())],total_variation=float(abs(pp-cp).sum()/2),clean_target_probability_drop=float(cp.max()-pp[cp.argmax()]),**{'P_'+cl:float(v) for cl,v in zip(s.STAGES,pp)}))
        assert all(torch.equal(v,model.state_dict()[k].cpu()) for k,v in before.items()) and all(p.grad is None for p in model.parameters()) and not model.training
        model_checks[name]={'checkpoint_sha256':s.sha(path),'state_unchanged':True,'no_gradients':True}
        print(name+' validation controls complete',flush=True);del model,aux,ck,before
    s.csv_write(out/'validation_predictions.csv',allpred);write(out/'validation_metrics.json',metrics)
    groups,map_inputs=map_audit(out,HERE.parent/'posthoc_explainability/results');source_shas.update(map_inputs)
    if any(s.sha(ROOT/p)!=v for p,v in source_shas.items()):raise ValueError('Protected input changed during audit')
    summary=dict(protocol=protocol,model_checks=model_checks,metrics=metrics,native_map_summaries=groups,input_sha256=source_shas,elapsed_seconds=time.monotonic()-t,device=args.device,old_baseline_exposure_counts={k:sum(r['baseline_strict_split']==k for r in rows) for k in sorted({r['baseline_strict_split'] for r in rows})},display_support={region:dict(mean_nonblack_fraction=float(np.mean([r[region+'_nonblack_fraction'] for r in support])),max_nonblack_fraction=float(max(r[region+'_nonblack_fraction'] for r in support)),images_with_any_nonblack=sum(r[region+'_nonblack_fraction']>0 for r in support)) for region in ms},training=False,heldout_inference=False)
    write(out/'summary.json',summary)
    manifest={str(p.relative_to(out)):s.sha(p) for p in sorted(out.rglob('*')) if p.is_file()};write(out/'SHA256SUMS.json',manifest)
    print(json.dumps({'status':'complete','validation_predictions':len(allpred),'native_maps':230,'gallery_cases':115,'seconds':summary['elapsed_seconds']}),flush=True)

if __name__=='__main__':main()
