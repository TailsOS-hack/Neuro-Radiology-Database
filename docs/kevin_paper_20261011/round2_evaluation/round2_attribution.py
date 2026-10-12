#!/usr/bin/env python3
"""Exploratory native Grad-CAM for the validation-chosen round2 representative.

Reuses every fixed historical case and old/c3 native array. Does not select models.
"""
from __future__ import annotations
import argparse,html,sys
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import round2_pipeline as p
import report_round2 as report
sys.path.insert(0,str(p.ROOT))

SOURCE='docs/kevin_paper_20261011/posthoc_explainability/results'


def verified_cases(root):
    source=root/SOURCE
    if p.sha(source/'SHA256SUMS.json')!='abf11186722dd43ab4fbea7b97eb7752007436c5dacd8bfc27dd0bfe80f2b7e7' or p.sha(source/'per_case.csv')!='71e94e550471989ff4316bb5742e185379add3ef7e1237bcfdb901d538b378a5':raise ValueError('Pinned historical attribution source changed')
    inventory=p.read(source/'SHA256SUMS.json');rows=p.stress.csv_read(source/'per_case.csv')
    if p.sha(source/'per_case.csv')!=inventory['per_case.csv']:raise ValueError('Historical case inventory changed')
    baseline={};cases=[]
    for r in rows:
        if r['model'] not in ['old_baseline','candidate_3']:raise ValueError('Unexpected historical model')
        name='round1_c3' if r['model']=='candidate_3' else 'old_baseline';key=(r['case_id'],name)
        if key in baseline:raise ValueError('Duplicate historical case')
        path=source/'maps'/f'{r["model"]}_{r["case_id"]}.npy'
        if p.sha(path)!=inventory[str(path.relative_to(source))] or p.sha(root/r['path'])!=r['image_sha256']:raise ValueError('Historical map/image changed')
        arr=np.load(path,allow_pickle=False)
        if arr.shape!=(7,7) or not np.isfinite(arr).all() or arr.min()<0 or arr.max()>1 or arr.sum()<=0:raise ValueError('Invalid historical native map')
        baseline[key]=(r,arr)
        if name=='old_baseline':cases.append({k:r[k] for k in ['case_id','path','label','role','image_sha256']})
    if len(cases)!=115 or len(baseline)!=230 or sum(r['role']=='reused_original100' for r in cases)!=100 or sum(r['role']=='historical_class_balanced_controls' for r in cases)!=8:raise ValueError('Changed fixed case set')
    for case in cases:
        for name in ['old_baseline','round1_c3']:
            r,_=baseline[case['case_id'],name]
            if any(r[k]!=case[k] for k in ['path','label','role','image_sha256']):raise ValueError('Baseline case alignment differs')
    return cases,baseline,dict(case_inventory_sha256=p.sha(source/'per_case.csv'),baseline_payload_manifest_sha256=p.sha(source/'SHA256SUMS.json'))


def render_panel(path,case,shown,maps,labels):
    from matplotlib import colormaps
    lut=colormaps['inferno'];panel=Image.new('RGB',(1680,328),'white');d=ImageDraw.Draw(panel)
    d.text((5,3),case['case_id']+' | supplied label: '+case['label']+' | '+case['role'],fill='black',font=ImageFont.load_default(size=17))
    columns=[('Input',shown,'Supplied: '+case['label'])]
    for name in ['old_baseline','round1_c3','round2_representative']:
        a=maps[name];label=labels[name];short={'old_baseline':'Old','round1_c3':'Round1 c3','round2_representative':'Round2 selected'}[name]
        if a is None:coarse=Image.new('RGB',(224,224),'#eeeeee');smooth=coarse
        else:
            colors=np.uint8(lut(a)[...,:3]*255);coarse=Image.fromarray(colors).resize((224,224),Image.Resampling.NEAREST);smooth=Image.fromarray(colors).resize((224,224),Image.Resampling.BILINEAR);dc=ImageDraw.Draw(coarse)
            for j in range(1,7):dc.line((j*32,0,j*32,223),fill='white');dc.line((0,j*32,223,j*32),fill='white')
            for x,y in [(0,0),(192,0),(0,192),(192,192)]:dc.rectangle((x,y,x+31,y+31),outline='#00ffff',width=2)
        columns += [(short+' native7×7',coarse,label if a is not None else 'Flat/failed map'),(short+' overlay',Image.blend(shown,smooth,.38),label)]
    for j,(title,pic,label) in enumerate(columns):d.text((j*240+5,28),title,fill='black',font=ImageFont.load_default(size=15));panel.paste(pic,(j*240+8,50));d.text((j*240+5,279),label,fill='black',font=ImageFont.load_default(size=14))
    d.text((5,305),'Own predicted-class targets; cyan marks corner cells. Native cells have broad receptive fields; overlays are display only. No anatomical validation.',fill='black',font=ImageFont.load_default(size=14));panel.save(path)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--suite',type=Path,required=True);ap.add_argument('--data-root',type=Path,default=p.ROOT);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--device',choices=['cpu','mps','cuda'],default='cpu');args=ap.parse_args();suite=args.suite.resolve();root=args.data_root.resolve();out=args.output.resolve()
    if out.exists():raise ValueError('Attribution output must be new')
    evaluation,evaluation_record=report.verified_outputs(suite);freeze,checkpoints=p.load_frozen(root,suite);cases,baseline,provenance=verified_cases(root);chosen=freeze['representative'];p.c.imports();torch=p.c.torch;torch.set_num_threads(4);device=p.c.device_for(args.device)
    from src.grad_cam import GradCAM,get_target_layer
    from src.explainability_validation import map_similarity
    model,aux,ck=p.e.load(checkpoints[chosen],device);model.eval();aux.eval();original={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};tf=p.c.dataset([],root,False,224).tf
    source_rows=p.stress.csv_read(evaluation/f'{chosen}_original100_predictions.csv');source_by={r['path']:np.array([float(r['P_'+cl]) for cl in p.stress.STAGES]) for r in source_rows}
    out.mkdir(parents=True);(out/'maps').mkdir();(out/'cases').mkdir();(out/'randomized_maps').mkdir()
    p.dump(out/'protocol_before_attribution.json',dict(representative=chosen,checkpoint_sha256=p.sha(checkpoints[chosen]),freeze_sha256=p.sha(suite/'round2_evaluation_freeze.json'),historical_sources=provenance,class_target='Each model own unmodified argmax; new target fixed through parameter randomization',clean_probability_tolerance=1e-4,randomization_seeds=[42,43,44],randomization_stages=['head','head_final','full'],case_count=115,prior_benchmark_known=True,clinical_anatomical_validation=False,source_code_sha256=p.code_hashes()))
    rows=[];mass=[];cache=[];randomized=[];regions=p.corner.native_regions();tolerance_failures=[]
    for i,case in enumerate(cases):
        with Image.open(root/case['path']) as im:rgb=im.convert('RGB');x=tf(rgb).unsqueeze(0).to(device);shown=rgb.resize((224,224))
        with torch.inference_mode():before=model(x).detach().clone();prob=before.softmax(1)[0].cpu().numpy()
        target=int(prob.argmax());modes=[m.training for m in model.modules()];cam=GradCAM(model,get_target_layer(model)).generate(x,target)
        with torch.inference_mode():after=model(x)
        if not torch.equal(before,after) or modes!=[m.training for m in model.modules()] or any(q.grad is not None for q in model.parameters()):raise ValueError('GradCAM changed logits, gradients or modes')
        valid=cam is not None and cam.shape==(7,7) and np.isfinite(cam).all() and np.ptp(cam)>0
        if not valid:cam=None
        if cam is not None:np.save(out/'maps'/f'{chosen}_{case["case_id"]}.npy',cam)
        delta=None;agreement=None
        if case['role']=='reused_original100':
            if case['path'] not in source_by:raise ValueError('Original100 prediction missing')
            prior=source_by[case['path']];delta=float(np.abs(prob-prior).max());agreement=int(prior.argmax())==target
            if delta>1e-4 or not agreement:tolerance_failures.append(dict(case_id=case['case_id'],max_probability_difference=delta,argmax_agreement=agreement))
        row=dict(**case,model=chosen,prediction=p.stress.STAGES[target],confidence=float(prob[target]),cam_valid=valid,prediction_and_logits_unchanged=True,cloud_max_probability_difference=delta,cloud_argmax_agreement=agreement,**{'P_'+cl:float(v) for cl,v in zip(p.stress.STAGES,prob)});rows.append(row)
        maps={'round2_representative':cam};labels={'round2_representative':p.stress.STAGES[target]}
        for name in ['old_baseline','round1_c3']:rr,a=baseline[case['case_id'],name];maps[name]=a;labels[name]=rr['prediction']
        for name,a in maps.items():
            if a is None:mass.append(dict(case_id=case['case_id'],role=case['role'],model=name,valid=False,corners4_mass=None,edge_ring24_mass=None,center25_mass=None,peak_in_any_corner=None));continue
            normalized=a/a.sum();peak=a==a.max();mass.append(dict(case_id=case['case_id'],role=case['role'],model=name,valid=True,**{k+'_mass':float(normalized[m].sum()) for k,m in regions.items()},peak_in_any_corner=bool((peak&regions['corners4']).any())))
        render_panel(out/'cases'/f'{case["case_id"]}.png',case,shown,maps,labels)
        if case['role']=='historical_class_balanced_controls':cache.append((case,x,cam,target))
        if (i+1)%25==0:print(f'{chosen}: {i+1}/115 fixed attribution cases',flush=True)
    if len(cache)!=8:raise ValueError('Changed designated randomization cases')
    for seed in [42,43,44]:
        torch.manual_seed(seed);fresh,_=p.c.network(pretrained=False);fresh=fresh.state_dict();prefix=f'features.{len(model.features)-1}.'
        for stage in ['head','head_final','full']:
            state={k:(fresh[k] if stage=='full' or k.startswith('classifier.') or (stage=='head_final' and k.startswith(prefix)) else v) for k,v in original.items()};model.load_state_dict(state);model.eval()
            for case,x,native,target in cache:
                cam=GradCAM(model,get_target_layer(model)).generate(x,target);rho,iou=map_similarity(native,cam) if native is not None and cam is not None else (None,None)
                if cam is not None:np.save(out/'randomized_maps'/f'{case["case_id"]}_{seed}_{stage}.npy',cam)
                randomized.append(dict(case_id=case['case_id'],seed=seed,stage=stage,target=p.stress.STAGES[target],status='valid' if rho is not None else 'flat_or_failed',spearman=rho,top20_iou=iou))
    model.load_state_dict(original);model.eval()
    if not all(torch.equal(v,model.state_dict()[k].cpu()) for k,v in original.items()) or any(q.grad is not None for q in model.parameters()):raise ValueError('Randomization did not restore original state')
    p.load_frozen(root,suite);verified_cases(root)
    p.stress.csv_write(out/'per_case.csv',rows);p.stress.csv_write(out/'native_map_mass.csv',mass);p.stress.csv_write(out/'randomization.csv',randomized)
    html_text='<meta charset="utf-8"><h1>All115 fixed attribution cases</h1><p>Old and round1c3 maps are unchanged verified arrays. Round2 representative '+html.escape(chosen)+' was selected only by validation. Native7×7 maps use each model’s own predicted class; comparison does not assume the same target. These are exploratory visualizations, not anatomical proof. Cyan marks corner cells.</p>'
    for case in cases:html_text+='<h2>'+html.escape(case['case_id'])+'</h2><img width="1680" src="cases/'+case['case_id']+'.png" alt="Native maps and display overlays">'
    (out/'gallery.html').write_text(html_text)
    plt=report.plots.setup_plotting();fig,axes=plt.subplots(1,2,figsize=(12.5,5.4));labels=['Old','Round1 c3','Round2 selected']
    for ax,role in zip(axes,['reused_original100','historical_class_balanced_controls']):
        for j,name in enumerate(['old_baseline','round1_c3','round2_representative']):
            vals=[r for r in mass if r['role']==role and r['model']==name and r['valid']];ax.bar(np.arange(3)+(j-1)*.24,[100*np.mean([r[k+'_mass'] for r in vals]) if vals else 0 for k in regions],width=.24,label=labels[j])
        for k,reg in enumerate(regions.values()):ax.plot([k-.4,k+.4],[100*reg.mean()]*2,color='black',ls=':')
        ax.set_xticks(range(3),['4 corner cells','24 outer cells','25 center cells']);ax.set(title=role.replace('_',' '),ylabel='Mean fraction of normalized map mass (%)');ax.legend(fontsize=8)
    fig.suptitle('Native-grid location audit | all fixed cases; valid maps only',fontweight='bold');fig.text(.02,.02,'Dotted lines: uniform area fractions. Outer cells overlap corner cells. Map mass is not anatomical probability or causal input importance.',fontsize=8);fig.tight_layout(rect=(0,.07,1,.93));report.plots.save_figure(fig,out,'native_location_comparison');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11.8,4.8))
    for ax,metric in zip(axes,['spearman','top20_iou']):
        for j,stage in enumerate(['head','head_final','full']):
            vals=[r[metric] for r in randomized if r['stage']==stage and r[metric] is not None];ax.scatter(j+np.linspace(-.12,.12,len(vals)),vals,color='#8e44ad',alpha=.6,s=18)
            if vals:ax.plot([j-.2,j+.2],[np.mean(vals)]*2,color='black')
        ax.set_xticks(range(3),['Head','Head + final block','Full']);ax.set_ylabel(metric.replace('_',' '));ax.grid(axis='y',alpha=.2)
    fig.suptitle('Round2 parameter randomization | 8 fixed cases × 3 seeds',fontweight='bold');fig.text(.02,.02,'Original predicted class stays fixed. Points exclude flat/failed maps, whose counts remain in randomization.csv. Model state is restored.',fontsize=8);fig.tight_layout(rect=(0,.08,1,.93));report.plots.save_figure(fig,out,'parameter_randomization');plt.close(fig)
    p.dump(out/'verification.json',dict(case_count=115,valid_maps=sum(r['cam_valid'] for r in rows),all_logits_modes_gradients_unchanged=True,original_checkpoint_state_restored=True,randomization_records=len(randomized),cloud_probability_tolerance=1e-4,cloud_probability_or_argmax_failures=tolerance_failures,historical_sources=provenance,clinical_anatomical_validation=False,visual_review_required=True))
    manifest={str(f.relative_to(out)):p.sha(f) for f in sorted(out.rglob('*')) if f.is_file()};p.dump(out/'SHA256SUMS.json',manifest);p.dump(out/'attribution_complete.json',dict(status='complete_numerically_verified_visual_review_required' if not tolerance_failures else 'blocked_cloud_probability_replay_review',payloads=len(manifest),manifest_sha256=p.sha(out/'SHA256SUMS.json')));print(p.read(out/'attribution_complete.json'))

if __name__=='__main__':main()
