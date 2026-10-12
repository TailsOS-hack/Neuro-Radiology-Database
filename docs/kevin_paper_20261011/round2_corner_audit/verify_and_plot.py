#!/usr/bin/env python3
"""Recompute exported audit metrics, verify hashes, and render summaries."""
import json,hashlib
from pathlib import Path
import numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import corner_audit as a

OUT=a.HERE/'results'

def save(fig,name):
    for ext in ['png','pdf']:fig.savefig(OUT/f'{name}.{ext}',dpi=250,facecolor='white')
    plt.close(fig)

def main():
    v=json.loads((OUT/'summary.json').read_text());manifest=json.loads((OUT/'SHA256SUMS.json').read_text())
    assert all(a.s.sha(OUT/p)==sha for p,sha in manifest.items())
    assert all(a.s.sha(a.ROOT/p)==sha for p,sha in v['input_sha256'].items())
    assert a.s.sha(a.__file__)==v['protocol']['script_sha256']
    source=a.s.csv_read(OUT/'validation_sample.csv');expected={r['path']:int(r['label']) for r in source}
    assert len(expected)==160 and all(r['split'] in ['val','validation'] for r in source)
    predictions=a.s.csv_read(OUT/'validation_predictions.csv');lookup={}
    for r in predictions:
        k=(r['model'],r['scenario']);lookup.setdefault(k,{})
        assert r['path'] not in lookup[k] and r['TrueCategory']==a.s.STAGES[expected[r['path']]]
        lookup[k][r['path']]=r
    assert len(lookup)==26
    y=np.array(list(expected.values()))
    for mt in v['metrics']:
        rows=lookup[(mt['model'],mt['scenario'])];assert set(rows)==set(expected)
        # Preserve the inference float32 dtype when independently recomputing NLL.
        p=np.array([[float(rows[i]['P_'+cl]) for cl in a.s.STAGES] for i in expected],dtype=np.float32)
        clean=np.array([[float(lookup[(mt['model'],'clean')][i]['P_'+cl]) for cl in a.s.STAGES] for i in expected],dtype=np.float32)
        assert all(rows[i]['ModelPrediction']==a.s.STAGES[j] for i,j in zip(expected,p.argmax(1)))
        actual=a.s.metrics(y,p,clean)
        for key in actual:
            if isinstance(actual[key],list):assert actual[key]==mt[key]
            else:assert np.isclose(actual[key],mt[key],atol=1e-12,rtol=0),(key,actual[key],mt[key])
    masks=a.masks()
    assert masks['corners32'].sum()==4*32*32 and masks['border16'].sum()==224*224-192*192
    assert all(masks[name].sum()==masks[name+'_matched_center'].sum() for name in ['corners32','border16'])
    assert not (masks['corners32']&masks['corners32_matched_center']).any()
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    groups=v['native_map_summaries'];roles=['reused_original100','historical_class_balanced_controls','historical_selected_gallery_not_random'];regions=['corners4','edge_ring24','center25'];colors=['#444444','#0072B2']
    fig,axes=plt.subplots(3,3,figsize=(12,8),sharey=True)
    for i,role in enumerate(roles):
        for j,region in enumerate(regions):
            ax=axes[i,j];rows=[next(r for r in groups if r['role']==role and r['model']==model) for model in a.MODELS]
            ax.bar([0,1],[r[region+'_mass']*100 for r in rows],color=colors)
            area=a.native_regions()[region].mean()*100;ax.axhline(area,color='#999999',ls=':',label='Uniform score area share')
            for k,r in enumerate(rows):ax.text(k,r[region+'_mass']*100+1,f"{r[region+'_mass']*100:.2f}%",ha='center',fontsize=9)
            ax.set_xticks([0,1],['Old','New c3']);ax.set_ylim(0,100)
            if i==0:ax.set_title(['Four native corner cells','Outer native grid ring','Inner native grid'][j])
            if j==0:ax.set_ylabel(['Reused100','Historical8','Selected historical7'][i]+'\nMean score mass (%)')
    fig.suptitle('Native 7×7 Grad-CAM location | all 230 archived maps',weight='bold')
    fig.text(.045,.015,'Dotted line: region share of 49 cells (corners 8.16%, ring 48.98%, center 51.02%). Scores were normalized per map, not calibrated anatomical probabilities.\nNo map peaks in a corner cell. Broad overlapping receptive fields and display interpolation prevent a pixel-local causal interpretation.',fontsize=9)
    fig.subplots_adjust(left=.08,right=.98,top=.91,bottom=.12,hspace=.3,wspace=.2);save(fig,'native_location_mass')
    fig,axes=plt.subplots(2,2,figsize=(13,8),sharex=True)
    stems=['clean','corners32_erase','corners32_only','corners32_matched_center_erase','border16_erase','border16_only','border16_matched_center_erase'];labels=['Clean','Erase corners','Corners only','Erase matched\ncenter (8.2%)','Erase border','Border only','Erase matched\ncenter (26.5%)']
    for j,fill in enumerate(['black','imagenet_mean']):
        for model,color in zip(a.MODELS,colors):
            rows=[next(r for r in v['metrics'] if r['model']==model and r['scenario']==('clean' if s=='clean' else s+'_'+fill)) for s in stems]
            for i,metric in enumerate(['accuracy','mean_total_variation']):axes[i,j].plot(range(7),[r[metric]*100 for r in rows],marker='o',color=color,label='Old baseline' if model=='old_baseline' else 'New c3')
        axes[0,j].set_title('Fill: '+fill.replace('_',' '));axes[0,j].axhline(25,ls=':',color='#999999');axes[0,j].set_ylim(0,104);axes[1,j].set_ylim(0,104)
        for i in range(2):axes[i,j].set_xticks(range(7),labels,rotation=25,ha='right',fontsize=8);axes[i,j].grid(alpha=.15)
        axes[0,j].legend(frameon=False)
    axes[0,0].set_ylabel('Accuracy on 160 validation images (%)');axes[1,0].set_ylabel('Mean probability total variation (%)')
    fig.suptitle('Fixed validation-only masking controls | 40 images per class',weight='bold')
    fig.text(.045,.015,'No model selection or training. Corners: four 32×32 squares; border: outer 16 pixels. Center controls mask exactly the same number of pixels.\nThe clean sample is balanced; 25% dotted line is a reference, not a significance test. Erasure is artificial; hash groups are not patient identities.\nAll images are new-model validation rows; old-baseline strict exposure was 99 training / 23 validation / 38 test images.',fontsize=9)
    fig.subplots_adjust(left=.07,right=.98,top=.90,bottom=.21,hspace=.15,wspace=.18);save(fig,'validation_masking_controls')
    # Show every region definition and both fills for the first fixed validation image.
    first=source[0];im=np.asarray(Image.open(a.ROOT/first['path']).convert('RGB').resize((224,224),Image.Resampling.BILINEAR))/255
    fig,axes=plt.subplots(3,5,figsize=(13,8))
    configs=[('Input',np.zeros((224,224),bool)),('Erase corners',masks['corners32']),('Corners only',~masks['corners32']),('Erase border',masks['border16']),('Border only',~masks['border16'])]
    for j,(name,mask) in enumerate(configs):
        axes[0,j].imshow(mask,cmap='gray',vmin=0,vmax=1);axes[0,j].set_title(name+f'\nwhite = masked ({mask.mean()*100:.1f}%)',fontsize=9)
        for i,fill in enumerate([np.zeros(3),np.array([.485,.456,.406])],1):axes[i,j].imshow(np.where(mask[...,None],fill,im))
        for i in range(3):axes[i,j].set_xticks([]);axes[i,j].set_yticks([])
    axes[1,0].set_ylabel('Black fill');axes[2,0].set_ylabel('ImageNet mean fill')
    fig.suptitle('Input-region definitions | first deterministically selected validation image',weight='bold')
    fig.text(.045,.015,'Representative display only; all 160 fixed validation images were measured. Native Grad-CAM cells are not these input masks.\nDisplay image: '+first['path'],fontsize=8);fig.subplots_adjust(left=.04,right=.99,top=.89,bottom=.09,wspace=.07,hspace=.08);save(fig,'input_mask_definitions')
    report={'status':'passed','validation_images':160,'unique_hash_groups':160,'per_class':40,'models':2,'scenarios_per_model':13,'predictions_recomputed_and_verified':4160,'metric_records_verified':26,'all_prior_map_and_input_hashes_match':True,'checkpoint_states_unchanged':all(r['state_unchanged'] for r in v['model_checks'].values()),'native_gallery_panels':115,'inference_on_original100_or_test':False,'training':False,'mask_area_and_center_control_checks':True,'verifier_sha256':a.s.sha(__file__)}
    a.write(OUT/'verification.json',report)
    a.write(OUT/'FINAL_SHA256SUMS.json',{str(p.relative_to(OUT)):a.s.sha(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='FINAL_SHA256SUMS.json'})
    print(json.dumps(report))

if __name__=='__main__':main()
