#!/usr/bin/env python3
"""Exploratory clinical T1 preparation; all patient-level output stays in ignored data/.

This deterministic pipeline does not perform validated brain segmentation,
registration to a brain atlas, bias-field correction, or clinical validation.
Run with the repository data directory as --data-root. No model is trained.
"""
from __future__ import annotations
import argparse, collections, csv, hashlib, json, math, random
from pathlib import Path
import nibabel as nib
from nibabel.processing import resample_to_output
import numpy as np
from scipy import ndimage
from PIL import Image, ImageDraw

VERSION = 'exploratory_t1_axial_v1'
SEED = 1729
# Fixed before model fitting after gross anatomical QC of the first preprocessing pass.
# Fractions reference a full cranial foreground span, not a validated brain mask.
FRACTIONS = np.linspace(0.45, 0.85, 9)
SPACING = (1.5, 1.5, 1.5)
FIELDS = ['path','cohort','subject_id','group_key','diagnosis','binary_label','split','age','sex','view','slice_index','slice_fraction','preprocessing_version']

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()

def read_rows(path):
    with path.open(newline='',encoding='utf-8-sig') as f: return list(csv.DictReader(f,delimiter='\t' if path.suffix=='.tsv' else ','))

def source_subjects(data_root, allow_missing=False):
    subjects=[]
    on=data_root/'external/openneuro_ds007561_v1'
    for r in read_rows(on/'participants.tsv'):
        identifier=r['participant_id']
        p=on/identifier/'ses-01/anat'/f'{identifier}_ses-01_T1w.nii.gz'
        assert p.is_file(),f'Missing T1: {p}'
        subjects.append(dict(cohort='openneuro_ds007561',subject_id=identifier,diagnosis={'Control':'CN','MCI':'MCI','AD':'AD'}[r['diagnosis']],split='external',age=r['age'],sex=r['gender'],source=p))
    assert len(subjects)==20 and len({r['subject_id'] for r in subjects})==20
    besga=data_root/'external/besga_zenodo_3935636/acquired'
    clinical=besga/'Clinical_data/clinical_data_id_age_gender.csv'
    if not clinical.is_file():
        if allow_missing: return subjects, ['Besga acquisition has not completed; only OpenNeuro processed.']
        raise FileNotFoundError(f'Besga approved clinical table missing: {clinical}')
    clinical_rows=read_rows(clinical)
    assert len(clinical_rows)==78
    scans={p.name[:-7]:p for p in (besga/'T1_original').glob('*.nii.gz')}
    keys=[r['Subject'].strip()+'_'+r['id'].strip() for r in clinical_rows]
    assert len(set(keys))==78 and set(keys)==set(scans),'Besga imaging/clinical join mismatch'
    diagnoses={'ea':'AD','crl':'CN','tb':'BIPOLAR'}
    for r,identifier in zip(clinical_rows,keys):
        subjects.append(dict(cohort='besga_zenodo_3935636',subject_id=identifier,diagnosis=diagnoses[r['Subject'].strip()],split='excluded_primary' if r['Subject'].strip()=='tb' else '',age=r['Age'].strip(),sex=r['gender'].strip(),source=scans[identifier]))
    assert collections.Counter(r['diagnosis'] for r in subjects if r['cohort']=='besga_zenodo_3935636')=={'AD':35,'CN':19,'BIPOLAR':24}
    for label in ('AD','CN'):
        group=sorted([r for r in subjects if r['cohort']=='besga_zenodo_3935636' and r['diagnosis']==label],key=lambda r:r['subject_id'])
        random.Random(SEED + (1 if label=='AD' else 2)).shuffle(group)
        ntrain=math.floor(0.60*len(group));nval=round(0.20*len(group))
        for i,r in enumerate(group): r['split']='train' if i<ntrain else 'val' if i<ntrain+nval else 'test'
    return subjects, []

def longest_run(indices):
    runs=np.split(indices,np.flatnonzero(np.diff(indices)>1)+1)
    return max(runs,key=lambda r:len(r))

def process(subject, output, data_root):
    key=f"{subject['cohort']}__{subject['subject_id']}"
    image=nib.load(subject['source'])
    assert len(image.shape)==3, f'Expected 3D T1: {key}'
    assert np.isfinite(image.affine).all() and abs(np.linalg.det(image.affine[:3,:3]))>1e-8
    original_shape=list(image.shape);original_orientation=list(nib.aff2axcodes(image.affine))
    canonical=nib.as_closest_canonical(image)
    # An orthogonal RAS grid makes the third axis anatomically superior even for oblique source affines.
    aligned=resample_to_output(canonical,voxel_sizes=SPACING,order=1,mode='constant',cval=0)
    assert nib.aff2axcodes(aligned.affine)==('R','A','S')
    volume=np.asarray(aligned.dataobj,dtype=np.float32)
    assert np.isfinite(volume).all(), f'Nonfinite voxels: {key}'
    positive=volume[volume>0]
    assert positive.size>10000,f'Insufficient positive foreground: {key}'
    p98=float(np.percentile(positive,98))
    assert p98>0
    mask=volume>0.08*p98
    # Retain only the largest connected foreground object. This is NOT skull stripping.
    labels,n=ndimage.label(mask)
    assert n>0
    counts=np.bincount(labels.ravel());counts[0]=0
    mask=labels==int(counts.argmax())
    axial_areas=mask.sum(axis=(0,1))
    broad=np.flatnonzero(axial_areas>=0.35*axial_areas.max())
    assert broad.size>=15, f'Insufficient cranial foreground extent: {key}'
    span=longest_run(broad); zlo,zhi=int(span[0]),int(span[-1])
    # Avoid inferior neck by using the broad cranial span; cropped brain-only inputs are also supported.
    cranial=mask[:,:,zlo:zhi+1]
    xs,ys,zs=np.where(cranial)
    margin=4
    x0=max(0,int(xs.min())-margin);x1=min(volume.shape[0],int(xs.max())+margin+1)
    y0=max(0,int(ys.min())-margin);y1=min(volume.shape[1],int(ys.max())+margin+1)
    robust=volume[:,:,zlo:zhi+1][cranial]
    low,high=(float(v) for v in np.percentile(robust,[1,99]))
    assert math.isfinite(low) and math.isfinite(high) and high>low
    indices=[int(round(zlo+float(frac)*(zhi-zlo))) for frac in FRACTIONS]
    assert len(set(indices))==9, f'Duplicate axial slice indices: {key}'
    image_dir=output/'images'/subject['cohort']/subject['subject_id'];image_dir.mkdir(parents=True,exist_ok=True)
    rows=[]; pixel_hashes=[]
    for number,(index,fraction) in enumerate(zip(indices,FRACTIONS)):
        axial=volume[x0:x1,y0:y1,index]
        normalized=np.clip((axial-low)/(high-low),0,1)
        # np.rot90 places anterior at the top and anatomical left on image left (neurological convention).
        pixels=np.rint(np.rot90(normalized)*255).astype(np.uint8)
        assert pixels.max()>pixels.min()
        tile=Image.fromarray(pixels,mode='L')
        side=max(tile.size);square=Image.new('L',(side,side),0)
        square.paste(tile,((side-tile.width)//2,(side-tile.height)//2))
        rgb=square.resize((224,224),resample=Image.Resampling.BILINEAR).convert('RGB')
        destination=image_dir/f'axial_{number:02d}.png';rgb.save(destination)
        check=np.asarray(Image.open(destination))
        assert check.shape==(224,224,3) and np.isfinite(check).all()
        assert np.array_equal(check[:,:,0],check[:,:,1]) and np.array_equal(check[:,:,0],check[:,:,2])
        pixel_hashes.append(hashlib.sha256(check.tobytes()).hexdigest())
        rows.append(dict(path=destination.relative_to(data_root).as_posix(),cohort=subject['cohort'],subject_id=subject['subject_id'],group_key=key,diagnosis=subject['diagnosis'],binary_label={'AD':'1','CN':'0'}.get(subject['diagnosis'],''),split=subject['split'],age=subject['age'],sex=subject['sex'],view='axial_RAS_neurological',slice_index=index,slice_fraction=f'{fraction:.3f}',preprocessing_version=VERSION))
    private_qc=dict(group_key=key,cohort=subject['cohort'],subject_id=subject['subject_id'],diagnosis=subject['diagnosis'],split=subject['split'],source_path=subject['source'].relative_to(data_root).as_posix(),source_sha256=sha(subject['source']),source_shape=original_shape,source_orientation=original_orientation,aligned_shape=list(volume.shape),aligned_spacing=list(SPACING),foreground_threshold=0.08*p98,cranial_z_extent=[zlo,zhi],crop_xy=[x0,x1,y0,y1],intensity_percentiles_1_99=[low,high],slice_indices=indices,pixel_sha256=pixel_hashes)
    return rows,private_qc

def montage(rows,output):
    qa=output/'qa';qa.mkdir(exist_ok=True)
    # These files are generated only by this script. Remove superseded QA sheets
    # so a prior sampling pass cannot be mistaken for the frozen final protocol.
    for previous in qa.glob('*.jpg'): previous.unlink()
    cohorts=sorted({r['cohort'] for r in rows})
    written=[]
    for cohort in cohorts:
        representatives=[r for r in rows if r['cohort']==cohort and r['slice_fraction']==f'{FRACTIONS[4]:.3f}']
        columns=6;thumb=160;cellh=192
        for page,start in enumerate(range(0,len(representatives),30),1):
            subset=representatives[start:start+30]
            canvas=Image.new('RGB',(columns*thumb,math.ceil(len(subset)/columns)*cellh),(25,25,25));draw=ImageDraw.Draw(canvas)
            for i,r in enumerate(subset):
                image=Image.open(output.parents[1]/r['path']).resize((thumb,thumb))
                x=(i%columns)*thumb;y=(i//columns)*cellh
                canvas.paste(image,(x,y));draw.text((x+3,y+thumb+3),r['subject_id'],fill='white')
            target=qa/f'{cohort}_central_all_{page}.jpg';canvas.save(target,quality=92);written.append(str(target))
        ordered=sorted(representatives,key=lambda r:r['subject_id'])
        # Representative full-slice grids are selected without diagnosis labels.
        for position in sorted({0,len(ordered)//2,len(ordered)-1}):
            chosen=ordered[position]
            subset=[r for r in rows if r['group_key']==chosen['group_key']]
            canvas=Image.new('RGB',(224*3,250*3),(25,25,25));draw=ImageDraw.Draw(canvas)
            for i,r in enumerate(subset):
                x=(i%3)*224;y=(i//3)*250
                canvas.paste(Image.open(output.parents[1]/r['path']),(x,y));draw.text((x+4,y+226),f"{r['subject_id']} fraction {r['slice_fraction']}",fill='white')
            target=qa/f"{cohort}_{chosen['subject_id']}_nine_slices.jpg";canvas.save(target,quality=92);written.append(str(target))
    return written

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-root',type=Path,required=True)
    ap.add_argument('--allow-missing-besga',action='store_true')
    args=ap.parse_args();data_root=args.data_root.resolve();output=data_root/'external/clinical_t1_preprocessed_v1'
    output.mkdir(parents=True,exist_ok=True)
    subjects,limitations=source_subjects(data_root,args.allow_missing_besga)
    selected=[r for r in subjects if r['diagnosis']!='BIPOLAR']
    assert len({f"{r['cohort']}__{r['subject_id']}" for r in subjects})==len(subjects)
    rows=[];private=[]
    for i,subject in enumerate(selected,1):
        subject_rows,qc=process(subject,output,data_root);rows.extend(subject_rows);private.append(qc)
        print(json.dumps({'processed':i,'of':len(selected),'cohort':subject['cohort'],'diagnosis':subject['diagnosis'],'source_orientation':qc['source_orientation'],'source_shape':qc['source_shape']}),flush=True)
    by_subject=collections.defaultdict(set)
    for r in rows: by_subject[r['group_key']].add(r['split'])
    assert all(len(v)==1 for v in by_subject.values()),'A patient crosses splits'
    assert all(sum(r['group_key']==key for r in rows)==9 for key in by_subject)
    assert all(r['split']=='external' for r in rows if r['cohort']=='openneuro_ds007561')
    assert all(r['binary_label']=='' and r['split']=='external' for r in rows if r['diagnosis']=='MCI')
    all_hashes=collections.defaultdict(set)
    for r in private:
        for h in r['pixel_sha256']: all_hashes[h].add(r['group_key'])
    cross_patient_duplicates={h:sorted(keys) for h,keys in all_hashes.items() if len(keys)>1}
    assert not cross_patient_duplicates,'Exact duplicate processed pixels across patients'
    manifest=output/'clinical_slices.csv'
    with manifest.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=FIELDS);writer.writeheader();writer.writerows(rows)
    with (output/'subjects.csv').open('w',newline='') as f:
        fields=['cohort','subject_id','diagnosis','split','age','sex','source']
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for r in subjects: writer.writerow({**r,'source':r['source'].relative_to(data_root).as_posix()})
    (output/'preprocessing_qc_private.json').write_text(json.dumps({'version':VERSION,'subjects':private},indent=2)+'\n')
    montage_paths=montage(rows,output)
    counts=collections.Counter((r['cohort'],r['split'],r['diagnosis']) for r in selected)
    summary={'protocol_version':VERSION,'status':'exploratory_preprocessed_pending_visual_qc','data_root':'data','manifest':'data/external/clinical_t1_preprocessed_v1/clinical_slices.csv','seed':SEED,'subjects_processed':len(selected),'slices_written':len(rows),'bipolar_subjects_retained_raw_excluded_primary':sum(r['diagnosis']=='BIPOLAR' for r in subjects),'subject_counts':[{'cohort':c,'split':s,'diagnosis':d,'subjects':n,'slices':n*9} for (c,s,d),n in sorted(counts.items())],'checks':{'all_images_224_rgb':True,'all_pixels_finite':True,'all_rgb_channels_equal':True,'source_clinical_join_complete':True,'patient_split_disjoint':True,'nine_slices_per_subject':True,'exact_cross_patient_processed_pixel_duplicates':len(cross_patient_duplicates),'all_openneuro_held_external':True,'mci_binary_labels_blank':True},'image_normalization':'Per-volume 1st/99th percentiles of intensity-thresholded largest connected cranial foreground; no cohort-fitted preprocessing.','orientation':'Canonical RAS then orthogonal 1.5 mm isotropic RAS resampling; axial neurological display.','cranial_foreground':'Largest connected component above 8% of 98th percentile of positive intensities. Longest axial run with area at least 35% of maximum; nine fixed 45–85% fractions of cranial extent. This heuristic is not validated brain segmentation.','resize':'Fixed per-volume xy crop with 6 mm margin; preserve aspect ratio with black square padding, resize 224 bilinear, repeat grayscale RGB.','selection_policy':'Besga provider groups ea=AD,crl=CN; tb bipolar excluded. Deterministic subject-stratified 60/20/20 using seed 1729; OpenNeuro entire cohort external. No forced dementia-severity mapping.','clinical_metadata_policy':'Age and source sex/gender retained locally for subgroup descriptions only, not image model inputs. Besga gender coding not inferred. Original NIfTI headers and private biomarker tables are not exported.','evaluation_policy':'Only Besga validation may guide auxiliary model selection. Besga test and all OpenNeuro labels are excluded from selection. MCI may be predicted but excluded from binary AD/CN scoring.','limitations':limitations+['Exploratory preprocessing only; visual review is gross QC, not validated anatomical or clinical quality assurance.','No atlas registration, validated skull stripping, bias-field correction, scanner harmonization or clinical validation.','Preprocessing failures stop the pipeline rather than silently dropping cases.','Small single-source Besga development sample and only one OpenNeuro AD case; no robust external validation claim.','Source/cohort overlap and patient-identity deduplication beyond exact image pixels remain unresolved.','No model training or label harmonization to the existing severity task performed.'],'private_qc_montages':len(montage_paths),'visual_qc':'pending'}
    destination=data_root.parent/'docs/kevin_paper_20261011/clinical_preprocessing_summary.json'
    summary.update(
        preprocessing_script_sha256=sha(Path(__file__)),
        clinical_manifest_sha256=sha(manifest),
        final_slice_fractions=[round(float(f),3) for f in FRACTIONS],
        protocol_history=[{
            'pass':'superseded draft',
            'slice_fractions':[0.2,0.275,0.35,0.425,0.5,0.575,0.65,0.725,0.8],
            'reason_changed':'Gross visual anatomical QC showed excess inferior cerebellum/neck slices; initial sheets displayed source diagnoses but no model predictions or outcomes were inspected.',
            'model_training_performed':False,
            'files_retained':False
        },{
            'pass':'final before model training',
            'slice_fractions':[round(float(f),3) for f in FRACTIONS],
            'applied_uniformly_to_both_cohorts':True,
            'qa_labels':'Subject IDs and slice fraction only; no diagnoses or model outcomes.',
            'png_policy':'Every manifest PNG regenerated; superseded QA sheets removed.'
        }])
    assert len(list((output/'images').glob('*/*/*.png')))==len(rows)
    destination.write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({'manifest':str(manifest),'summary':str(destination),'subjects':len(selected),'slices':len(rows),'montages':montage_paths}),flush=True)

if __name__=='__main__':main()
