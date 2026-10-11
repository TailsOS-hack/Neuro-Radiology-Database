#!/usr/bin/env python3
"""Independent CSV denominator/metric/readback audit; no model inference."""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import stress_test as s


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--replay',type=Path,required=True)
    p.add_argument('--stress',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();root=args.root.resolve();replay=args.replay.resolve();stress=args.stress.resolve()
    if args.output.exists():raise ValueError('Use a new verification output path')
    complete=json.loads((replay/'replay_complete.json').read_text())
    if complete['status']!='completed_posthoc_clean_split_replay' or complete['output_manifest_sha256']!=s.sha(replay/'output_checksums.json'):raise ValueError('Replay is not verified complete')
    outputs=json.loads((replay/'output_checksums.json').read_text())
    for r in outputs:
        f=replay/r['path']
        if s.sha(f)!=r['sha256'] or f.stat().st_size!=r['bytes']:raise ValueError('Replay output changed')
    meta=json.loads((replay/'replay_metadata.json').read_text())
    if len(meta['checkpoints'])!=6:raise ValueError('All six candidates required')
    source=s.csv_read(root/'training_logs/experiments/kevin_20261011/prepared/stage_manifest.csv')
    expected={name:{r['path']:r for r in source if r['split']==key} for name,key in [('train_resubstitution','train'),('full_new_test','test')]}
    checks=[];primary=[];total=0
    for name,model in meta['checkpoints'].items():
        for split,info in model['splits'].items():
            f=replay/info['predictions_file']
            if s.sha(f)!=info['predictions_sha256']:raise ValueError('Prediction checksum mismatch')
            rows=s.csv_read(f);cm=[[0]*4 for _ in range(4)];loss=0
            if len(rows)!=len(expected[split]) or len({r['path'] for r in rows})!=len(rows) or {r['path'] for r in rows}!=set(expected[split]):raise ValueError('Split membership mismatch')
            for r in rows:
                yy=int(expected[split][r['path']]['label']);prob=[float(r['P_'+cl]) for cl in s.STAGES]
                if not all(math.isfinite(v) and 0<=v<=1 for v in prob) or abs(sum(prob)-1)>2e-6:raise ValueError('Invalid probability')
                pred=prob.index(max(prob))
                if r['TrueCategory']!=s.STAGES[yy] or r['ModelPrediction']!=s.STAGES[pred]:raise ValueError('Label/argmax mismatch')
                cm[yy][pred]+=1;loss-=math.log(max(prob[yy]/sum(prob),1e-15))
            n=len(rows);accuracy=sum(cm[i][i] for i in range(4))/n
            ba=sum(cm[i][i]/sum(cm[i]) for i in range(4))/4
            f1=sum(2*cm[i][i]/(sum(cm[i])+sum(cm[j][i] for j in range(4))) for i in range(4))/4
            calculated=dict(accuracy=accuracy,balanced_accuracy=ba,macro_f1=f1,loss=loss/n)
            mt=info['metrics']
            if cm!=mt['confusion_matrix'] or mt['n']!=n or any(abs(v-mt[k])>1e-10 for k,v in calculated.items()):raise ValueError('Exported metrics disagree with raw CSV')
            checks.append(dict(model=name,split=split,n=n,metrics_recomputed=True,confusion_matrix=cm,**calculated));total+=n
            if split=='full_new_test':
                current={r['path']:r for r in rows};prior=[r for r in s.csv_read(stress/'predictions'/f'{name}__clean.csv') if r['cohort']=='internal_same_case']
                if len(prior)!=1058 or len({r['path'] for r in prior})!=1058:raise ValueError('Invalid stress primary membership')
                maxdiff=0.;outside=0;argmax=0
                for r in prior:
                    q=current[r['path']];a=[float(q['P_'+cl]) for cl in s.STAGES];b=[float(r['P_'+cl]) for cl in s.STAGES]
                    d=[abs(x-y) for x,y in zip(a,b)];maxdiff=max(maxdiff,max(d));outside+=any(dd>s.ATOL+s.RTOL*abs(y) for dd,y in zip(d,b));argmax+=q['ModelPrediction']!=r['ModelPrediction']
                primary.append(dict(model=name,overlap_n=1058,max_abs_probability_difference=maxdiff,cases_outside_same_preset_tolerance=outside,argmax_disagreements=argmax))
    if total!=6*(19023+5450):raise ValueError('Total denominator mismatch')
    report=dict(status='completed_independent_csv_and_denominator_verification',created_utc=datetime.now(timezone.utc).isoformat(),script_sha256=s.sha(Path(__file__)),replay_metadata_sha256=s.sha(replay/'replay_metadata.json'),output_manifest_sha256=s.sha(replay/'output_checksums.json'),files_readback_checked=len(outputs),prediction_csv_files=12,prediction_rows_checked=total,checks=checks,primary_overlap_local_clean_replay_comparison=primary,tolerance={'atol':s.ATOL,'rtol':s.RTOL},interpretation='Training is final-checkpoint resubstitution. Full5450 contains primary1058 and is an overlapping post-hoc descriptive analysis; primary results remain unchanged. Any numerical differences between local clean passes are retained above, not corrected or used for model selection.')
    s.dump(args.output,report);print(json.dumps(dict(path=str(args.output),sha256=s.sha(args.output),rows=total,checks=len(checks),primary_overlap_argmax_disagreements=sum(r['argmax_disagreements'] for r in primary)),indent=2))


if __name__=='__main__':main()
