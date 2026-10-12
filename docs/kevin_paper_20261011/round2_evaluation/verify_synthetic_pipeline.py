#!/usr/bin/env python3
"""End-to-end synthetic inference and all report figures; no research outcomes."""
import argparse,sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from PIL import Image
import round2_pipeline as p
import report_round2 as r
from test_round2_pipeline import histories


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);args=ap.parse_args();root=args.output.resolve()
    if root.exists():raise ValueError('Synthetic output must be new')
    root.mkdir(parents=True);suite=root/'suite';suite.mkdir();p.c.imports();torch=p.c.torch;torch.set_num_threads(2)
    rows=[]
    for i in range(24):
        path=f'images/{i:02d}.png';im=root/path;im.parent.mkdir(exist_ok=True);Image.fromarray(np.uint8(np.random.default_rng(i).uniform(0,255,(31,37,3)))).save(im);rows.append(dict(path=path,label=str(i%4),group_id=f'g{i}',FileName=f'{i:02d}.png'))
    cohorts=dict(full_new_test=rows[:12],internal_same_case=rows[:8],old_hash_group_unexposed_sensitivity=rows[:4],original100=rows[12:16],corner_validation=rows[16:])
    reader=root/p.R1/'frozen_evaluation_all6/original100_case_aligned_old_candidates_readers.csv';reader.parent.mkdir(parents=True);p.stress.csv_write(reader,[dict(FileName=row['FileName'],TrueCategory=p.stress.STAGES[int(row['label'])],Reader_A=p.stress.STAGES[(int(row['label'])+1)%4],Reader_B=p.stress.STAGES[int(row['label'])],Reader_C=p.stress.STAGES[(int(row['label'])+2)%4]) for row in cohorts['original100']])
    cps={};selected={}
    for name in p.ORDER:
        cp=suite/(name+'.fake');cp.write_text('SYNTHETIC '+name);cps[name]=cp
    for name,seed in zip(p.NEW,p.SEEDS):
        folder=suite/f'stage_only_seed{seed}';folder.mkdir();hist,_=histories();p.dump(folder/'history.json',hist);selected[name]=dict(checkpoint=f'{folder.name}/best.pt',epoch=1,history_sha256=p.sha(folder/'history.json'))
    freeze=dict(status='frozen_before_any_round2_evaluation',source_code_sha256=p.code_hashes(),all_three_runs=selected,representative=p.NEW[0]);p.dump(suite/'round2_evaluation_freeze.json',freeze)
    class Fake(torch.nn.Module):
        def __init__(self):super().__init__();self.weight=torch.nn.Parameter(torch.arange(12,dtype=torch.float32).reshape(3,4)/50)
        def forward(self,x):return x.mean((2,3))@self.weight
    def load(path,device):return Fake().to(device),torch.nn.Identity().to(device),{'image_size':224}
    with patch.object(p,'load_frozen',return_value=(freeze,cps)),patch.object(p,'rows_for',return_value=cohorts),patch.object(p.e,'load',side_effect=load),patch.object(p.c,'forward',side_effect=lambda model,aux,x,head:model(x)):
        p.evaluate(SimpleNamespace(data_root=root,suite=suite,device='cpu',batch_size=5))
        with patch.object(sys,'argv',['report_round2.py','--suite',str(suite),'--data-root',str(root),'--output',str(root/'report')]):
            save=r.plots.save_figure
            def marked(fig,output,stem):fig.text(.5,.5,'SYNTHETIC QA',fontsize=40,ha='center',alpha=.16,rotation=20);return save(fig,output,stem)
            with patch.object(r.plots,'save_figure',side_effect=marked):r.main()
    verified=r.verified_outputs(suite);analysis=p.read(root/'report/analysis.json')
    if len(analysis['metric_recomputation'])!=215 or len(list((root/'report').glob('*.pdf')))!=28:raise ValueError('Unexpected synthetic verification/figure count')
    # Corrupt a numeric field and establish that recomputation catches it even
    # when the caller has already verified package checksums.
    corner=suite/'frozen_round2_evaluation/corner_metrics.json';original=corner.read_bytes();data=p.read(corner);data[0]['accuracy']=.12345;p.dump(corner,data)
    try:
        try:r.verify_stress_corner(suite/'frozen_round2_evaluation',cohorts)
        except ValueError:pass
        else:raise ValueError('Failed to reject corrupt scenario metric')
    finally:corner.write_bytes(original)
    result=dict(status='passed_synthetic_end_to_end',real_models_or_heldout_outcomes_used=False,models=5,cohorts=4,clean_metric_records=20,stress_metric_records=130,corner_metric_records=65,all_215_metric_records_recomputed=True,all_report_figure_families_rendered=True,figure_pairs=28,corrupted_metric_rejected=True,output_manifest_readback=True)
    p.dump(root/'synthetic_verification.json',result);print(result)

if __name__=='__main__':main()
