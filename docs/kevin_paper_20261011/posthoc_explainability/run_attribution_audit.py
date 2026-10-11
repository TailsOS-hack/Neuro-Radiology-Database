#!/usr/bin/env python3
"""Post-hoc attribution diagnostics; fixed checkpoints, no fitting or selection.

Recreates all 115 dementia attribution case slots from the prior paper package.
Controls use the eight previously fixed class-balanced specialist examples.
Maps explain model scores, not validated anatomy or hippocampal atrophy.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'docs/kevin_paper_20261011/training'))

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''): h.update(b)
    return h.hexdigest()

def read_csv(p):
    with Path(p).open(newline='') as f: return list(csv.DictReader(f))

def write_csv(p, rows):
    with p.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results-root', type=Path, default=ROOT / 'training_logs/experiments/kevin_20261011/runpod_blackwell_20261011')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(); out = args.output
    if out.exists(): raise ValueError('Preserve prior diagnostics; output must be new')
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/kevin-attribution-mpl')
    import numpy as np
    import torch
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image, ImageDraw, ImageFont
    from torchvision.transforms.functional import gaussian_blur
    import train_candidate as c
    from src.grad_cam import GradCAM, get_target_layer, overlay_cam_on_image
    from src.explainability_validation import top_mask, shape_controls, random_controls, map_similarity
    c.imports(); torch.set_num_threads(4); torch.manual_seed(20261011)
    selection = json.loads((args.results_root / 'validation_selection_before_evaluation.json').read_text())
    identity = selection['exploratory_representative_candidate']
    chosen = selection['candidates'][identity]
    assert chosen['mode'] == 'stage_only' and chosen['seed'] == 20261012
    cp = args.results_root / Path(chosen['checkpoint_path']).parent.name / 'best.pt'
    checkpoints = {'old_baseline': ROOT / 'models/alzheimers_classifier.pt', identity: cp}
    expected = {'old_baseline': 'a8cc03aea74f5310c1fc7936c72f94a44e55af11e2ada2a8ba5f8edd6d924db5', identity: chosen['checkpoint_sha256']}
    assert all(sha(v) == expected[k] for k,v in checkpoints.items())
    key = ROOT / 'data/evaluation/ground_truth/radiologist_test_key.csv'
    cases = [dict(case_id=Path(r['FileName']).stem, path='data/evaluation/images/'+r['FileName'], label=r['TrueCategory'], role='reused_original100') for r in read_csv(key)]
    assert len(cases) == 100 and len({r['case_id'] for r in cases}) == 100
    verification = ROOT / 'docs/evaluation_100_20261007/specialist_gradcam/verification.json'
    extra = json.loads(verification.read_text())['specialists']['dementia']['samples']
    assert len(extra) == 8
    for i,r in enumerate(extra):
        assert sha(ROOT/r['image']) == r['image_sha256']
        cases.append(dict(case_id=f'historical_specialist_{i+1:02d}', path=r['image'], label=Path(r['image']).parent.name, role='historical_class_balanced_controls'))
    review = ROOT / 'docs/review_bundle/explainability/per_image.csv'
    gallery = [r for r in read_csv(review) if r['task']=='dementia' and r['gallery_reason']]
    assert len(gallery) == 7
    for i,r in enumerate(gallery):
        assert sha(ROOT/r['path']) == r['image_sha256']
        cases.append(dict(case_id=f'historical_review_{i+1:02d}', path=r['path'], label=r['true_label'], role='historical_selected_gallery_not_random'))
    out.mkdir(parents=True); (out/'cases').mkdir(); (out/'maps').mkdir(); (out/'randomized_maps').mkdir()
    protocol = {'role':'Post-hoc descriptive software and attribution diagnostics, not independent clinical validation', 'models':expected,
                'representative_selection':'candidate_3 chosen from stage-validation before original evaluation; unchanged here',
                'case_count':len(cases), 'control_case_count':8, 'control_selection':'The eight fixed historical class-balanced specialist examples; no new outcome selection',
                'mask_fractions':[0.1,0.2,0.3], 'replacement':['imagenet_mean','gaussian_blur_sigma5'],
                'random_masks':5, 'shape_controls':'Up to five nonduplicate dihedral transforms of each top-CAM mask',
                'randomization_seeds':[42,43,44], 'randomization_stages':['head','head_final','full'],
                'class_target':'Original unmodified model argmax; held fixed through perturbation and randomization',
                'native_layer':'features[-1]', 'clean_probability_tolerance':1e-4,
                'sources':{str(x.relative_to(ROOT)):sha(x) for x in [key,verification,review,Path(__file__),ROOT/'src/grad_cam.py',ROOT/'src/explainability_validation.py']}}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    records=[]; controls=[]; randomized=[]; renders={}; model_checks={}
    for model_id, checkpoint in checkpoints.items():
        ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
        assert ck['class_names']==c.STAGES and ck['arch']=='mobilenet_v3_large'
        model,_=c.network(pretrained=False); model.load_state_dict(ck['model_state']); model.eval()
        original={k:v.detach().clone() for k,v in model.state_dict().items()}
        frozen_rows=read_csv(args.results_root/'frozen_evaluation_all6'/f'{model_id}_original100_predictions.csv')
        assert len(frozen_rows)==100 and len({r['FileName'] for r in frozen_rows})==100
        assert {r['FileName']:r['TrueCategory'] for r in frozen_rows}=={Path(r['path']).name:r['label'] for r in cases if r['role']=='reused_original100'}
        source_probs={r['FileName']:np.array([float(r['P_'+cl]) for cl in c.STAGES]) for r in frozen_rows}
        cache=[]; max_delta=0.0
        tf=c.dataset([],ROOT,False,224).tf
        for i,case in enumerate(cases):
            with Image.open(ROOT/case['path']) as im: rgb=im.convert('RGB'); x=tf(rgb).unsqueeze(0)
            with torch.inference_mode(): before=model(x).detach().clone(); prob=before.softmax(1)[0].numpy()
            target=int(prob.argmax()); modes=[m.training for m in model.modules()]
            native=GradCAM(model,get_target_layer(model)).generate(x,target)
            with torch.inference_mode(): after=model(x)
            unchanged=bool(torch.equal(before,after)); assert unchanged
            assert modes==[m.training for m in model.modules()] and all(p.grad is None for p in model.parameters())
            valid=native is not None and np.isfinite(native).all() and np.ptp(native)>0
            delta=None; reference_agreement=None; reference_tolerance=None
            if case['role']=='reused_original100':
                prior=source_probs[Path(case['path']).name]; delta=float(np.max(np.abs(prob-prior))); max_delta=max(max_delta,delta)
                reference_agreement=int(prior.argmax())==target
                reference_tolerance=delta<=protocol['clean_probability_tolerance']
            rec={**case,'model':model_id,'image_sha256':sha(ROOT/case['path']),'prediction':c.STAGES[target], 'confidence':float(prob[target]), 'cam_valid':bool(valid),'prediction_and_logits_unchanged':unchanged,'clean_archived_max_probability_difference':delta,'archived_prediction_agreement':reference_agreement,'archived_probability_within_1e4_tolerance':reference_tolerance}
            records.append(rec)
            if valid: np.save(out/'maps'/f'{model_id}_{case["case_id"]}.npy',native)
            shown=rgb.resize((224,224)); overlay=overlay_cam_on_image(shown,native) if valid else shown
            renders.setdefault(case['case_id'],{'image':shown,'case':case})[model_id]=(overlay,c.STAGES[target],bool(valid))
            if case['role']=='historical_class_balanced_controls' and valid:
                cache.append((case,x,native,target,float(prob[target])))
                expanded=torch.nn.functional.interpolate(torch.from_numpy(native)[None,None],size=(224,224),mode='bilinear',align_corners=False)[0,0].numpy()
                for fraction in protocol['mask_fractions']:
                    mask=top_mask(expanded,fraction); seed=20261011+i*100+int(fraction*100)
                    randoms=random_controls(mask,seed,5); shapes=shape_controls(mask,seed,5)
                    masks=[mask]+randoms+[m for _,m in shapes]
                    for replacement in protocol['replacement']:
                        fill=torch.zeros_like(x) if replacement=='imagenet_mean' else gaussian_blur(x,[31,31],[5.,5.])
                        batch=x.repeat(len(masks),1,1,1); stack=torch.from_numpy(np.stack(masks))[:,None]
                        batch=torch.where(stack,fill.expand_as(batch),batch)
                        with torch.inference_mode(): scores=model(batch).softmax(1)[:,target].numpy()
                        drops=float(prob[target])-scores
                        controls.append(dict(model=model_id,case_id=case['case_id'],fraction=fraction,replacement=replacement,target=c.STAGES[target],top_drop=float(drops[0]),random_drop=float(drops[1:6].mean()),shape_drop=float(drops[6:].mean()) if shapes else None,shape_count=len(shapes),random_advantage=float(drops[0]-drops[1:6].mean()),shape_advantage=float(drops[0]-drops[6:].mean()) if shapes else None))
            if (i+1)%25==0: print(f'{model_id}: {i+1}/{len(cases)} attribution cases',flush=True)
        for seed in protocol['randomization_seeds']:
            torch.manual_seed(seed); fresh,_=c.network(pretrained=False); fresh=fresh.state_dict()
            prefix=f'features.{len(model.features)-1}.'
            for stage in protocol['randomization_stages']:
                state={k:(fresh[k] if stage=='full' or k.startswith('classifier.') or (stage=='head_final' and k.startswith(prefix)) else v) for k,v in original.items()}
                model.load_state_dict(state); model.eval()
                for case,x,native,target,_ in cache:
                    cam=GradCAM(model,get_target_layer(model)).generate(x,target)
                    rho,iou=map_similarity(native,cam) if cam is not None else (None,None)
                    if cam is not None: np.save(out/'randomized_maps'/f'{model_id}_{case["case_id"]}_{seed}_{stage}.npy',cam)
                    randomized.append(dict(model=model_id,case_id=case['case_id'],seed=seed,stage=stage,status='valid' if rho is not None else 'flat_or_failed',spearman=rho,top20_iou=iou))
        model.load_state_dict(original)
        assert all(torch.equal(v,model.state_dict()[k]) for k,v in original.items()) and sha(checkpoint)==expected[model_id]
        model_checks[model_id]={'checkpoint_sha256':expected[model_id],'state_restored':True,'archived_clean_max_probability_difference':max_delta,'archived_prediction_disagreements':sum(r['model']==model_id and r['archived_prediction_agreement'] is False for r in records),'archived_probability_tolerance_failures':sum(r['model']==model_id and r['archived_probability_within_1e4_tolerance'] is False for r in records),'valid_case_maps':sum(r['model']==model_id and r['cam_valid'] for r in records),'designated_control_cases':8,'valid_control_cases':len(cache)}
    font=ImageFont.load_default(size=17)
    for case_id,item in renders.items():
        panel=Image.new('RGB',(720,305),'white'); draw=ImageDraw.Draw(panel)
        columns=[('Input / supplied label',item['image'],item['case']['label'])]
        for model_id in checkpoints:
            overlay,label,valid=item[model_id]; columns.append((('Old baseline' if model_id=='old_baseline' else 'Validation-selected new'),overlay,label if valid else 'Flat/failed map'))
        for j,(heading,im,label) in enumerate(columns):
            draw.text((j*240+5,3),heading,fill='black',font=font);panel.paste(im,(j*240+8,28));draw.text((j*240+5,255),label,fill='black',font=font)
        draw.text((5,282),'Exploratory attribution; no anatomical or hippocampal validation',fill='black',font=ImageFont.load_default(size=14))
        panel.save(out/'cases'/f'{case_id}_gradcam.png')
    write_csv(out/'per_case.csv',records);write_csv(out/'perturbation_controls.csv',controls);write_csv(out/'randomization.csv',randomized)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for ax,replacement in zip(axes,protocol['replacement']):
        for model_id in checkpoints:
            for control,style in [('random','-'),('shape','--')]:
                values=[np.mean([r[control+'_advantage'] for r in controls if r['model']==model_id and r['replacement']==replacement and r['fraction']==f and r[control+'_advantage'] is not None]) for f in protocol['mask_fractions']]
                ax.plot(protocol['mask_fractions'],values,style,marker='o',label=f'{model_id}: {control}')
        ax.axhline(0,color='gray',lw=.8); ax.set(title=replacement,xlabel='Masked pixel fraction',ylabel='Top-CAM drop minus control drop');ax.legend(fontsize=8)
    fig.suptitle('Post-hoc masking diagnostics | 8 designated cases/model; valid maps only');fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(out/f'figure9_attribution_controls.{ext}',dpi=200)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for ax,model_id in zip(axes,checkpoints):
        for seed in protocol['randomization_seeds']:
            means=[]
            for stage in protocol['randomization_stages']:
                group=[r for r in randomized if r['model']==model_id and r['seed']==seed and r['stage']==stage and r['spearman'] is not None]
                means.append(np.mean([r['spearman'] for r in group]) if group else np.nan)
            ax.plot(protocol['randomization_stages'],means,marker='o',label=f'seed {seed}')
        ax.set(title=model_id,ylabel='Mean native-map Spearman correlation',ylim=(-1,1));ax.legend()
    fig.suptitle('Post-hoc parameter randomization | valid maps only; failures recorded');fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(out/f'figure10_parameter_randomization.{ext}',dpi=200)
    plt.close(fig)
    summary={'protocol':protocol,'models':model_checks,'case_panels':len(renders),'total_case_maps':len(records),'finite_nonflat_maps':sum(r['cam_valid'] for r in records),'unchanged_predictions_and_logits':sum(r['prediction_and_logits_unchanged'] for r in records),'perturbation_rows':len(controls),'randomization_rows':len(randomized),'randomization_failed_or_flat':sum(r['status']!='valid' for r in randomized),'limitations':['Post-hoc diagnostics; no checkpoint tuning or test-based reselection.','Historical extra examples are not new independent samples.','Random and dihedral mask controls have different spatial properties; masking can create out-of-distribution inputs.','Per-map normalization and coarse native resolution prevent quantitative anatomical claims.','Valid maps, score drops and randomization sensitivity do not establish faithful anatomy or clinical readiness.'],'environment':{'torch':torch.__version__,'device':'cpu','threads':4},'source_sha256':sha(__file__)}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    html='<!doctype html><meta charset="utf-8"><title>Kevin dementia attribution audit</title><h1>Old versus validation-selected new model: 115 fixed case slots</h1><p>Post-hoc exploratory attribution. Original100 is reused. No anatomical localization or atrophy claims. All cases retained, including any failed maps.</p>'
    for case_id,item in renders.items():html+=f'<h2>{case_id} — {item["case"]["role"]}</h2><img src="cases/{case_id}_gradcam.png" width="720" alt="{case_id} old and new Grad-CAM"><br>'
    (out/'index.html').write_text(html)
    manifest={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file()}
    (out/'SHA256SUMS.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__': main()
