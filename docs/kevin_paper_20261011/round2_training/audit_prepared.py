#!/usr/bin/env python3
"""Visual and array audit of predetermined TRAIN/VAL support examples only."""
import json
from pathlib import Path
import round2 as r
r.dependencies()
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    prepared=r.HERE/'prepared';out=r.HERE/'verification';out.mkdir(exist_ok=True)
    rows=r.c.csv_read(prepared/'stress_validation_manifest.csv')
    selected=[row for label in range(4) for row in [q for q in rows if int(q['label'])==label][:2]]
    scenarios=['clean','background_flat','background_edge_pattern','noise_002','blur_100']
    reports=[]
    for page in range(2):
        fig,axs=plt.subplots(4,6,figsize=(12,8.7))
        for rownum,row in enumerate(selected[page*4:(page+1)*4]):
            x=r.resize_rgb(r.ROOT/row['path'])
            with r.Image.open(prepared/row['mask_file']) as im:m=r.np.asarray(im)>0
            arrays={name:r.transform(x,m,name,row['path']) for name in scenarios}
            for name in ['background_flat','background_edge_pattern']:
                if not r.np.array_equal(arrays[name][m],arrays['clean'][m]):raise AssertionError('Protected support changed')
            r.np.savez_compressed(out/f'example_{page*4+rownum:02d}.npz',rgb_uint8=x,support=m,**arrays)
            shown=[arrays['clean'],m,arrays['background_flat'],arrays['background_edge_pattern'],arrays['noise_002'],arrays['blur_100']]
            for col,(array,label) in enumerate(zip(shown,['Original','Protected support','Safe background .10','Safe edge pattern','Noise SD .02','Blur radius1'])):
                ax=axs[rownum,col];ax.imshow(array,cmap='gray',vmin=0,vmax=1);ax.axis('off')
                if rownum==0:ax.set_title(label,fontsize=9)
            reports.append(dict(path=row['path'],label=row['label'],protected_fraction=float(m.mean()),safe_edge_fraction=float((~m&r.edge_region()).mean()),background_preserves_all_protected_pixels=True))
        fig.suptitle('Fixed validation examples: entire field of view; support is a heuristic, not brain segmentation',fontsize=11)
        fig.tight_layout();fig.savefig(out/f'augmentation_examples_{page+1}.png',dpi=140);plt.close(fig)
    r.dump(out/'prepared_audit.json',dict(status='passed',examples=reports,selection='First two frozen stress-manifest rows per class, no model outcomes inspected',preparation_sha256=r.c.sha(prepared/'preparation.json'),raw_arrays_saved=True))
if __name__=='__main__':main()
