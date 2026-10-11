#!/usr/bin/env python3
"""Evaluate validation-selected frozen checkpoints once. Never train or select.

Original100 and legacy image test outcomes are descriptive, not patient-independent
clinical validation. Original100 includes old-checkpoint training exposure.
"""
import argparse, importlib.util, json
from pathlib import Path
import train_candidate as c


def load(path,device):
    ck=c.torch.load(path,map_location='cpu',weights_only=False)
    if ck.get('smoke_only') or ck.get('config',{}).get('smoke'): raise ValueError('Refusing research evaluation of SMOKE checkpoint')
    if ck.get('arch')!='mobilenet_v3_large': raise ValueError('Unexpected architecture')
    if ck.get('class_names')!=c.STAGES: raise ValueError('Unexpected stage vocabulary/order')
    m,a=c.network(pretrained=False);m.load_state_dict(ck['model_state'])
    if ck.get('config',{}).get('mode')=='multitask': a.load_state_dict(ck['aux_state'])
    m.to(device);a.to(device)
    return m,a,ck


def readers(root):
    import pandas as pd
    path=root/'data_visualization/compare_rad_vs_ai.py'
    spec=importlib.util.spec_from_file_location('existing_reader_loader',path);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    result={}; provenance={}
    for p in sorted((root/'data/evaluation/radiologist_results').glob('*.xlsx')):
        for sheet in pd.ExcelFile(p).sheet_names:
            df=mod.load_clean_sheet(p,sheet)
            if df is None: continue
            if df['FileName'].duplicated().any(): raise ValueError(f'Duplicate reader case in {p}/{sheet}')
            mapped={r.FileName:r.RadPrediction_Mapped for r in df.itertuples()}
            mapped={k: v if isinstance(v,str) and v in c.STAGES else 'Unknown' for k,v in mapped.items()}
            if sheet in result and result[sheet]!=mapped: raise ValueError(f'Conflicting duplicated reader {sheet}')
            result[sheet]=mapped;provenance.setdefault(sheet,[]).append({'file':str(p.relative_to(root)),'sha256':c.sha(p),'sheet':sheet})
    return result,provenance


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,required=True);p.add_argument('--prepared',type=Path,required=True)
    p.add_argument('--candidate',type=Path,action='append',required=True,help='Repeat for each validation-selected candidate; all are reported')
    p.add_argument('--baseline',type=Path,default=Path('models/alzheimers_classifier.pt'))
    p.add_argument('--clinical-manifest');p.add_argument('--clinical-image-root',default='data')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='auto');p.add_argument('--workers',type=int,default=0);p.add_argument('--batch-size',type=int,default=32)
    args=p.parse_args();c.imports();root=args.data_root.resolve();out=args.output.resolve();device=c.device_for(args.device)
    if out.exists() and any(out.iterdir()): raise ValueError('Evaluation output must be empty; preserve prior frozen evidence')
    intersection=c.csv_read(c.resolve(root,args.prepared)/'same_case_test_intersection.csv')
    if not intersection: raise ValueError('No same-case test intersection')
    frozen_stage=c.csv_read(c.resolve(root,args.prepared)/'stage_manifest.csv')
    stage_digest=c.sha(c.resolve(root,args.prepared)/'stage_manifest.csv')
    frozen_test={r['path']:r for r in frozen_stage if r['split']=='test' and r['baseline_strict_split']=='test'}
    if len({r['path'] for r in intersection})!=len(intersection): raise ValueError('Duplicate intersection row')
    for r in intersection:
        if r['path'] not in frozen_test or any(r.get(k)!=frozen_test[r['path']].get(k) for k in ('split','label','group_id','baseline_strict_split','pixel_sha256')): raise ValueError('Intersection is not a subset of frozen eligible test rows')
    gt=c.csv_read(root/'data/evaluation/ground_truth/radiologist_test_key.csv')
    hundred=[dict(path='data/evaluation/images/'+r['FileName'],label=c.STAGES.index(r['TrueCategory']),cohort='historically_reused100',filename=r['FileName']) for r in gt]
    if len(hundred)!=100 or len({r['filename'] for r in hundred})!=100: raise ValueError('Invalid original100 key')
    clinical=c.clinical_rows(root,args.clinical_manifest,args.clinical_image_root)
    heldclinical=[r for r in clinical if r['split'] in {'test','external'}]
    result={'checkpoints':{},'limitations':[
        'No test-based checkpoint selection; each input candidate must already be frozen by validation.',
        'Legacy patient IDs/upstream augmentation families unknown; internal image independence not established.',
        'Original100 historically reused, old checkpoint matched66 train/9 validation/25 test images.',
        'Reader benchmark uses fixed image labels, not independent clinical diagnosis; no clinical superiority claim follows.',
        'Clinical AD/CN auxiliary endpoint is distinct from legacy4-stage endpoint; MCI is prediction-only.'
    ]}
    aligned={r['filename']:{'FileName':r['filename'],'TrueCategory':c.STAGES[r['label']]} for r in hundred}
    reader,prov=readers(root);result['reader_provenance']=prov
    for name,labels in reader.items():
        for case in aligned:aligned[case]['Reader_'+name]=labels.get(case,'Missing')
        known=[(r['label'],labels[r['filename']]) for r in hundred if labels.get(r['filename']) in c.STAGES]
        result.setdefault('reader_metrics',{})[name]={'known_labels':len(known),'missing_or_unknown':100-len(known),'accuracy_known_only':sum(c.STAGES[y]==z for y,z in known)/len(known) if known else None}
    checkpoints=[('old_baseline',c.resolve(root,args.baseline))]+[(f'candidate_{i+1}',c.resolve(root,p)) for i,p in enumerate(args.candidate)]
    # Validate every input and state dict before reading any held-out image.
    for name,path in checkpoints:
        m,a,ck=load(path,c.torch.device('cpu'))
        if name!='old_baseline':
            cfg=ck.get('config',{})
            if cfg.get('stage_manifest_sha256')!=stage_digest: raise ValueError('Candidate stage manifest changed')
            wanted=cfg.get('clinical_manifest_sha256')
            if wanted and (not args.clinical_manifest or c.sha(c.resolve(root,args.clinical_manifest))!=wanted): raise ValueError('Candidate clinical manifest changed')
            if cfg.get('mode') not in {'stage_only','multitask'}: raise ValueError('Unexpected candidate mode')
            if cfg.get('test_used_for_selection') is not False: raise ValueError('Candidate lacks validation-only selection provenance')
            if not ck.get('history') or ck['epoch']!=ck['history'][-1]['epoch']: raise ValueError('Missing candidate epoch provenance')
            if abs(ck['stage_validation']['loss']-ck['best_val_loss'])>1e-8: raise ValueError('Candidate is not its validation-selected best checkpoint')
        del m,a,ck
    out.mkdir(parents=True,exist_ok=True)
    for name,path in checkpoints:
        m,a,ck=load(path,device);args.image_size=int(ck.get('image_size',224))
        info={'path':str(path),'sha256':c.sha(path),'selection_config':ck.get('config',{}),'epoch':ck.get('epoch')}
        for cohort,rows in [('internal_same_case',intersection),('original100',hundred)]:
            mt,pred=c.evaluate(m,a,rows,root,device,'stage',args);info[cohort]=mt
            exported=[]
            for r,z in zip(rows,pred):
                rec=dict(path=r['path'],TrueCategory=c.STAGES[int(r['label'])],ModelPrediction=c.STAGES[int(c.np.argmax(z['probabilities']))])
                rec.update({f'P_{cl}':v for cl,v in zip(c.STAGES,z['probabilities'])})
                if cohort=='original100':
                    rec['FileName']=r['filename'];aligned[r['filename']][name]=rec['ModelPrediction']
                exported.append(rec)
            c.csv_write(out/f'{name}_{cohort}_predictions.csv',exported)
        # Stage-only auxiliary head was never trained, so it is never evaluated.
        if ck.get('config',{}).get('mode')=='multitask' and heldclinical:
            info['clinical_aux_AD_CN']={}
            for split,source in sorted({(r['split'],r['cohort']) for r in heldclinical}):
                selected=[r for r in heldclinical if r['split']==split and r['cohort']==source]
                mt,pred=c.evaluate(m,a,selected,root,device,'aux',args)
                key=split+'__'+source;info['clinical_aux_AD_CN'][key]=mt
                mt['diagnosis_subject_counts']={d:len({r['subject_id'] for r in selected if r['diagnosis']==d}) for d in ['CN','AD','MCI']}
                mt['interpretation']='Exploratory small-cohort result. Report internal and external cohorts separately;1externalAD cannot establish stable sensitivity.'
                c.csv_write(out/f'{name}_{key}_subject_predictions.csv',[dict(cohort=r['cohort'],subject_id=r['subject_id'],binary_label=r['label'] if r['label']>=0 else '',P_CN=r['probabilities'][0],P_AD=r['probabilities'][1],n_slices=r['n_slices']) for r in pred])
        result['checkpoints'][name]=info
    c.csv_write(out/'original100_case_aligned_old_candidates_readers.csv',list(aligned.values()))
    c.dump(out/'evaluation.json',result);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
