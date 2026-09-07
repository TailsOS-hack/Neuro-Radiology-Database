"""External-cohort readiness and patient-cluster evaluation; no cohort is supplied by this project."""
from __future__ import annotations
import hashlib
from pathlib import Path
import numpy as np
from PIL import Image
from src.research_validation import require, read_csv, sha, metrics

REQUIRED = ['image_path','patient_id','reference_label','cohort_id']
OPTIONAL = ['site','scanner','sequence','acquisition_date','reference_standard']


def image_identity(path):
    with Image.open(path) as im:
        im.load(); im=im.convert('RGB')
        payload=f'{im.width}x{im.height}:RGB:'.encode()+im.tobytes()
    return sha(path),hashlib.sha256(payload).hexdigest()


def validate_cohort(manifest, label_mapping, classes, internal_paths):
    require(bool(internal_paths),'Internal image inventory is required for overlap checking')
    known=set()
    for path in internal_paths:
        file_hash,pixel_hash=image_identity(path)
        known.update([file_hash,pixel_hash])
    rows=read_csv(manifest);require(bool(rows),'Empty external cohort')
    seen=set();warnings=[];identities=[]
    for row in rows:
        for name in REQUIRED:require(bool(row.get(name,'').strip()),f'Missing required {name}')
        require(row['reference_label'] in label_mapping,'Unknown reference label')
        mapped=label_mapping[row['reference_label']]
        require(mapped in classes,'Mapping targets unknown model class')
        path=(Path(manifest).parent/row['image_path']).resolve()
        require(path not in seen,'Duplicate external image path');seen.add(path)
        try:file_hash,pixel_hash=image_identity(path)
        except (OSError,ValueError) as exc:raise ValueError(f'Unreadable external input: {path.name}') from exc
        require(file_hash not in known and pixel_hash not in known,'Exact internal/external overlap')
        row['resolved_path']=str(path);row['class_index']=classes.index(mapped)
        identities.append({'file_sha256':file_hash,'pixel_sha256':pixel_hash})
    for name in OPTIONAL:
        missing=sum(not r.get(name,'').strip() for r in rows)
        if missing:warnings.append(f'{name} unavailable for {missing}/{len(rows)} images')
    require(len({r['cohort_id'] for r in rows})==1,'Evaluate one cohort per run')
    return rows,identities,warnings


def supported_macro_f1(y,pred,classes):
    vals=[]
    for c in classes:
        tp=np.sum((y==c)&(pred==c));fp=np.sum((y!=c)&(pred==c));fn=np.sum((y==c)&(pred!=c))
        vals.append(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.)
    return float(np.mean(vals))


def cohort_metrics(probabilities, labels, patient_keys, classes, seed=42, repetitions=1000):
    y=np.asarray(labels,dtype=int);pred=np.argmax(probabilities,axis=1)
    require(len(y)==len(patient_keys),'Patient/image alignment mismatch')
    report=metrics(probabilities,y)
    report['unit']='image-level predictions; patient-cluster resampling, not patient diagnosis'
    report['per_class']={}
    for c,name in enumerate(classes):
        count=int(np.sum(y==c))
        if count==0:
            report['per_class'][name]={'support':0,'recall':None,'precision':None,'f1':None,'status':'unavailable_no_reference_examples'}
        else:
            tp=int(np.sum((y==c)&(pred==c)));fp=int(np.sum((y!=c)&(pred==c)));fn=count-tp
            report['per_class'][name]={'support':count,'recall':tp/count,'precision':tp/(tp+fp) if tp+fp else 0.,
                                       'f1':2*tp/(2*tp+fp+fn),'status':'available'}
    groups={key:[] for key in patient_keys}
    for i,key in enumerate(patient_keys):groups[key].append(i)
    report['patients']=len(groups)
    if len(groups)<2:
        report['cluster_intervals']={'status':'unavailable_fewer_than_two_patients'}
        return report
    rng=np.random.default_rng(seed);keys=list(groups);accuracy=[];f1=[]
    for _ in range(repetitions):
        sampled=np.concatenate([groups[keys[i]] for i in rng.integers(0,len(keys),len(keys))])
        yy=y[sampled];pp=pred[sampled]
        accuracy.append(float((yy==pp).mean()))
        # Explicitly use supported classes of the resampled replicate.
        f1.append(supported_macro_f1(yy,pp,np.unique(yy)))
    report['cluster_intervals']={'status':'available','seed':seed,'repetitions':repetitions,
        'method':'percentile 95%; resample patients with all images; macro F1 uses classes present in each replicate',
        'accuracy':np.quantile(accuracy,[.025,.975]).tolist(),
        'supported_macro_f1':np.quantile(f1,[.025,.975]).tolist()}
    return report
