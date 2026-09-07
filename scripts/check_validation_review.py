#!/usr/bin/env python3
"""Light/full verification of completed follow-up artifacts, independent of model inference."""
import argparse
import json
import re
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src import experiment_pipeline as ep
from src.research_validation import (require,sha,read_csv,softmax,fit_temperature,metrics,
    hierarchical,verify_seal,write_json)
from src.explainability_validation import grouped_summary,map_similarity,shape_controls


def equal(a,b,path='summary'):
    if isinstance(a,dict):
        require(isinstance(b,dict) and set(a)==set(b),f'Keys differ: {path}')
        for k in a:equal(a[k],b[k],path+'.'+k)
    elif isinstance(a,list):
        require(isinstance(b,list) and len(a)==len(b),f'Lengths differ: {path}')
        for x,y in zip(a,b):equal(x,y,path)
    elif isinstance(a,(float,np.floating)):
        require(b is not None and np.isclose(a,b,atol=1e-7,rtol=1e-6),f'Numeric mismatch: {path}')
    else:require(a==b,f'Value mismatch: {path}')


def check_explainability(out,protocol,full=False):
    verify_seal(out)
    provenance=json.loads((out/'provenance.json').read_text())
    require(provenance['run_id']==protocol['run_id'],'Explainability run mismatch')
    frozen=ROOT/protocol['baseline']/'per_image.csv'
    require(sha(out/'selection.csv')==sha(frozen)==provenance['selection_sha256'],'Selection changed')
    selection=[r for r in read_csv(out/'selection.csv') if r['quantitative_sample']=='1']
    lookup={(r['task'],r['path']):r for r in selection}
    require(len(lookup)==256,'Wrong sample size')
    for r in selection:
        cam=np.load(out/r['cam_path'],allow_pickle=False)
        require(np.array_equal(cam,np.load(ROOT/protocol['baseline']/r['cam_path'],allow_pickle=False)),'Baseline map changed')
        if full:require(sha(ROOT/r['path'])==r['image_sha256'],'Source image changed')
    perturb=read_csv(out/'perturbations.csv');random=read_csv(out/'randomizations.csv')
    require(len(perturb)==256*6,'Perturbation count mismatch')
    require(len(random)==256*9,'Randomization count mismatch')
    require(len({(r['task'],r['path'],r['fraction'],r['replacement']) for r in perturb})==len(perturb),'Duplicate perturbations')
    require(len({(r['task'],r['path'],r['seed'],r['stage']) for r in random})==len(random),'Duplicate randomization rows')
    for r in perturb:
        ref=lookup[(r['task'],r['path'])]
        for key in ['fraction','top_drop','random_drop','shape_drop','random_advantage','shape_advantage']:
            r[key]=float(r[key]) if r[key] else None
        for key in ['seed','shape_count','target']:r[key]=int(r[key])
        require(r['fraction'] in protocol['fractions'] and r['replacement'] in protocol['replacements'],'Unexpected control setting')
        require(r['true_label']==ref['true_label'] and ep.TASK_CLASS_NAMES[r['task']][r['target']]==ref['predicted_label'],'Control target/class changed')
        rd=json.loads(r['random_drops']);sd=json.loads(r['shape_drops']);ids=json.loads(r['shape_ids'])
        require(len(rd)==5 and len(sd)==len(ids)==r['shape_count']<=5,'Control availability mismatch')
        equal(r['random_drop'],float(np.mean(rd)));equal(r['random_advantage'],r['top_drop']-r['random_drop'])
        equal(r['shape_drop'],float(np.mean(sd)) if sd else None)
        equal(r['shape_advantage'],r['top_drop']-r['shape_drop'] if sd else None)
        allowed={f'flip{flip}_rot{rotation}' for flip in [0,1] for rotation in range(4)}-{'flip0_rot0'}
        require(len(ids)==len(set(ids)) and set(ids)<=allowed,'Invalid or duplicate shape controls')
    for r in random:
        ref=lookup[(r['task'],r['path'])]
        r['seed']=int(r['seed']);r['target']=int(r['target'])
        require(r['seed'] in protocol['randomization_seeds'] and r['stage'] in protocol['randomization_stages'],'Unexpected randomization')
        require(r['true_label']==ref['true_label'] and ep.TASK_CLASS_NAMES[r['task']][r['target']]==ref['predicted_label'],'Randomization target changed')
        for key in ['spearman','top20_iou']:r[key]=float(r[key]) if r[key] else None
        if r['status']=='ok':
            m=np.load(out/'randomized_maps'/r['map'],allow_pickle=False)
            require(m.shape==(7,7) and np.isfinite(m).all() and m.min()>=0 and m.max()<=1,'Invalid randomized map')
            original=np.load(out/ref['cam_path'],allow_pickle=False)
            rho,iou=map_similarity(original,m);equal(r['spearman'],rho);equal(r['top20_iou'],iou)
        else:
            require(r['status']=='flat_or_failed' and r['spearman'] is None and r['top20_iou'] is None,'Invalid failure row')
    actual={'perturbations':grouped_summary(perturb,['task','fraction','replacement'],['top_drop','random_drop','shape_drop','random_advantage','shape_advantage','shape_count']),
            'randomizations':grouped_summary(random,['task','seed','stage'],['spearman','top20_iou'])}
    equal(actual,json.loads((out/'summary.json').read_text()))
    return {'sample_n':len(lookup),'perturbations':len(perturb),'randomizations':len(random)}


def check_calibration(out,protocol):
    verify_seal(out)
    prov=json.loads((out/'provenance.json').read_text());fits=json.loads((out/'temperatures.json').read_text())
    require(prov['run_id']==protocol['run_id'],'Calibration run mismatch')
    require(sha(out/'temperatures.json')==prov['frozen_temperatures_sha256'],'Calibration artifact changed')
    manifest=ep.read_manifest(ep.DEFAULT_MANIFEST)
    index={s:read_csv(out/f'{s}_index.csv') for s in ['val','test']}
    for split in index:equal(index[split],[r for r in manifest if r['split']==split],split+' membership')
    probabilities={};summary={}
    for task in ep.TASK_CLASS_NAMES:
        fit=fits[task]
        require(fit['split']=='val' and fit['classes']==ep.TASK_CLASS_NAMES[task],'Invalid calibration provenance')
        require(fit['checkpoint_sha256']==prov['checkpoints'][task]['sha256'],'Fit checkpoint mismatch')
        pointer=ep.TASK_DEFAULT_MODEL_PATH[task]
        expected=pointer.read_text().split('oid sha256:')[1].splitlines()[0] if pointer.stat().st_size<1024 else sha(pointer)
        require(fit['checkpoint_sha256']==expected,'Not the accepted checkpoint')
        require(fit['validation_manifest_sha256']==sha(out/'val_index.csv'),'Fit manifest mismatch')
        require(fit['validation_logits_sha256']==sha(out/f'{task}_val_logits.npy'),'Fit logit mismatch')
        z=np.load(out/f'{task}_val_logits.npy',allow_pickle=False)
        require(z.shape==(len(index['val']),len(fit['classes'])),'Validation logits shape')
        y=np.array([ep.label_for_row(r,task) if ep.label_for_row(r,task) is not None else -1 for r in index['val']]);keep=y>=0
        recomputed=fit_temperature(z[keep],y[keep],split='val',provenance={k:fit[k] for k in ['checkpoint_sha256','validation_manifest_sha256','validation_logits_sha256','classes']})
        equal(recomputed,fit)
        test=np.load(out/f'{task}_test_logits.npy',allow_pickle=False)
        require(test.shape==(len(index['test']),len(fit['classes'])),'Test logits shape')
        probabilities[task]={'original':softmax(test),'calibrated':softmax(test,fit['temperature'])}
        y=np.array([ep.label_for_row(r,task) if ep.label_for_row(r,task) is not None else -1 for r in index['test']]);keep=y>=0
        summary[task]={mode:metrics(p[keep],y[keep]) for mode,p in probabilities[task].items()}
        require(np.array_equal(probabilities[task]['original'].argmax(1),probabilities[task]['calibrated'].argmax(1)),'Component accuracy changed')
    labels=np.array([ep.label_for_row(r,'eight_class') for r in index['test']])
    for mode in ['original','calibrated']:
        p,routed=hierarchical(*[probabilities[t][mode] for t in ['binary','tumor','dementia']])
        equal(p.tolist(),np.load(out/f'hierarchical_{mode}_probabilities.npy',allow_pickle=False).tolist())
        require(np.array_equal(routed,np.load(out/f'hierarchical_{mode}_routed.npy',allow_pickle=False)),'Routed predictions mismatch')
        summary.setdefault('hierarchical_argmax',{})[mode]=metrics(p,labels)
        summary.setdefault('hierarchical_routed',{})[mode]=metrics(p,labels,routed)
    equal(summary,json.loads((out/'summary.json').read_text()))
    return {'validation_n':len(index['val']),'test_n':len(index['test']),'models':4}


def check(directory,full=False,checkpoint_dir=None):
    protocol=json.loads((directory/'protocol.json').read_text())
    result={'run_id':protocol['run_id'],'mode':'full' if full else 'light','external_validation':'not performed'}
    result['explainability']=check_explainability(directory/'explainability',protocol,full)
    result['calibration']=check_calibration(directory/'calibration',protocol)
    if full:
        from src.research_validation import checkpoint_path
        require(checkpoint_dir is not None, 'Full verification requires --checkpoint-dir')
        for task in ep.TASK_CLASS_NAMES: checkpoint_path(task, checkpoint_dir)
        from PIL import Image
        for split in ['val', 'test']:
            for row in read_csv(directory/'calibration'/f'{split}_index.csv'):
                with Image.open(ROOT/row['path']) as image: image.verify()
    for component in ['explainability','calibration']:
        prov=json.loads((directory/component/'provenance.json').read_text())
        require(prov['protocol_sha256']==sha(directory/'protocol.json'),'Protocol hash mismatch')
    if (directory/'sources').exists():
        for component in ['explainability','calibration']:
            prov=json.loads((directory/component/'provenance.json').read_text())
            for name,digest in prov['source_sha256'].items():
                require(sha(directory/'sources'/name)==digest,'Recorded source snapshot mismatch')
    if (directory/'COMPLETE.json').exists():
        verify_seal(directory)
        from scripts.render_validation_review import texts
        for name,expected in texts(directory).items():
            require((directory/name).read_text()==expected,'Generated manuscript/memo text mismatch: '+name)
        for name in ['RESULTS.md','MENTOR_MEMO.md','FIGURE_CAPTIONS.md','index.html']:
            require(protocol['run_id'] in (directory/name).read_text(),f'Document run mismatch: {name}')
        for p in directory.rglob('*.html'):
            for ref in re.findall(r'(?:src|href)="([^"]+)"',p.read_text()):
                if '://' not in ref and not ref.startswith('#'):require((p.parent/ref).exists(),f'Broken figure/link: {ref}')
    result['status']='pass';return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path);p.add_argument('--full',action='store_true');p.add_argument('--checkpoint-dir',type=Path);p.add_argument('--json-output',type=Path)
    args=p.parse_args();result=check(args.directory,args.full,args.checkpoint_dir)
    if args.json_output:write_json(args.json_output,result)
    print(json.dumps(result,indent=2))
