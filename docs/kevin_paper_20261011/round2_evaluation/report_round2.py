#!/usr/bin/env python3
"""Read verified round2 predictions; generate conditional statistics and figures."""
from __future__ import annotations
import argparse,sys
from pathlib import Path
import numpy as np
import round2_pipeline as p
sys.path.insert(0,str(p.HERE.parent/'analysis'))
import summarize_frozen_evaluation as stats
import plot_evaluation_figures as plots
from sklearn.metrics import roc_curve,roc_auc_score,precision_recall_curve,average_precision_score

TITLES={'full_new_test':'Full reused image test (5,450)','internal_same_case':'Reused same-case comparison (1,058)','old_hash_group_unexposed_sensitivity':'Reused hash-unexposed subset (756)','original100':'Reused reader benchmark (100)'}
DISPLAY={'old_baseline':'Old model','round1_c3':'Round 1 c3',**{n:f'Round 2 seed {s}' for n,s in zip(p.NEW,p.SEEDS)}}
COLORS=['#555555','#2980b9','#d35400','#239b56','#8e44ad']
NOTE='Post-hoc repeated image benchmarks; hash groups are not patients. No clinical or independent-test claim.'


def verified_outputs(suite):
    out=suite/'frozen_round2_evaluation';complete=p.read(out/'evaluation_complete.json');manifest=p.read(out/'SHA256SUMS.json');report=p.read(out/'evaluation.json')
    if complete.get('status')!='complete_verified' or complete['manifest_sha256']!=p.sha(out/'SHA256SUMS.json') or complete['payloads']!=len(manifest):raise ValueError('Incomplete evaluation archive')
    if any(p.sha(out/f)!=h for f,h in manifest.items()):raise ValueError('Evaluation payload checksum mismatch')
    if report['status']!='complete' or report['model_order']!=p.ORDER or report['selection_freeze_sha256']!=p.sha(suite/'round2_evaluation_freeze.json') or complete['freeze_sha256']!=report['selection_freeze_sha256']:raise ValueError('Evaluation/selection provenance differs')
    freeze=p.read(suite/'round2_evaluation_freeze.json')
    if report['freeze']!=freeze or freeze['source_code_sha256']!=p.code_hashes():raise ValueError('Frozen code or selection changed')
    return out,report


def aligned_probabilities(rows,expected):
    by={r['path']:r for r in rows}
    if len(by)!=len(rows) or set(by)!={r['path'] for r in expected}:raise ValueError('Missing, duplicate, or extra prediction cases')
    probabilities=[]
    for source in expected:
        r=by[source['path']];truth=stats.STAGES[int(source['label'])]
        if r['TrueCategory']!=truth or r.get('group_id',source['group_id'])!=source['group_id']:raise ValueError('Prediction truth/group mismatch')
        prob=stats.validate_probabilities(r,['P_'+cl for cl in stats.STAGES],'round2')
        if r['ModelPrediction']!=stats.STAGES[int(prob.argmax())]:raise ValueError('Argmax prediction differs from probabilities')
        probabilities.append(prob)
    return np.asarray(probabilities)


def verify_stress_corner(source,cohorts):
    """Recompute every scenario metric from complete aligned probabilities."""
    checked=[]
    specs=[('stress_metrics.json','stress_predictions'),('corner_metrics.json','corner_predictions')]
    for metrics_name,folder in specs:
        records=p.read(source/metrics_name)
        wanted=130 if folder=='stress_predictions' else 65
        keys={(r['model'],r.get('cohort','corner_validation'),r['scenario']) for r in records}
        if len(records)!=wanted or len(keys)!=wanted:raise ValueError('Missing or duplicate stress/corner metrics')
        for r in records:
            name=r['model'];cohort=r.get('cohort','corner_validation');expected=cohorts[cohort]
            if name not in p.ORDER:raise ValueError('Unknown stress/corner model')
            if folder=='stress_predictions':
                rows=[v for v in p.stress.csv_read(source/folder/f"{name}__{r['scenario']}.csv") if v['cohort']==cohort]
                clean=aligned_probabilities(p.stress.csv_read(source/f'{name}_{cohort}_predictions.csv'),expected)
            else:
                rows=p.stress.csv_read(source/folder/f"{name}_{r['scenario']}_predictions.csv")
                clean=aligned_probabilities(p.stress.csv_read(source/folder/f'{name}_clean_predictions.csv'),expected)
            prob=aligned_probabilities(rows,expected);y=np.array([int(v['label']) for v in expected]);actual=p.stress.metrics(y,prob,clean)
            for k,v in actual.items():
                if not np.allclose(np.asarray(v),np.asarray(r[k]),atol=1e-10,rtol=1e-9):raise ValueError('Stress/corner metric mismatch: '+k)
            checked.append(dict(model=name,cohort=cohort,scenario=r['scenario'],n=len(y),metrics_recomputed=True))
    return checked


def calibration_panels(y,probabilities,cohort,output,kind):
    plt=plots.setup_plotting();fig,axes=plt.subplots(2,3,figsize=(12.5,7.9));records={}
    for ax,(name,prob),color in zip(axes.flat,probabilities.items(),COLORS):
        cal=plots.calibration(y,prob);records[name]=cal
        if kind=='reliability':
            bins=[b for b in cal['bins'] if b['n']];ax.plot([0,1],[0,1],color='#888888',ls=':');ax.scatter([b['mean_confidence'] for b in bins],[b['accuracy'] for b in bins],s=[20+150*b['n']/len(y) for b in bins],color=color);ax.set(xlim=(0,1.02),ylim=(0,1.04),xlabel='Mean confidence in fixed bin',ylabel='Bin accuracy')
            ax.text(.04,.07,f"Brier={cal['multiclass_brier_sum']:.4f}\nECE={cal['top_label_ece_10_equal_width_bins']:.4f}",transform=ax.transAxes,fontsize=9)
        else:
            conf=prob.max(1);correct=prob.argmax(1)==y
            ax.hist([conf[correct],conf[~correct]],bins=np.linspace(0,1,21),stacked=True,color=[color,'#bd3c3c'],label=['Correct','Incorrect']);ax.set(xlim=(0,1),xlabel='Maximum class probability',ylabel='Image count');ax.legend(fontsize=8)
        ax.set_title(DISPLAY[name],fontsize=10)
    axes.flat[-1].axis('off');fig.suptitle(TITLES[cohort]+' | '+kind.capitalize(),fontweight='bold');fig.text(.04,.025,NOTE,fontsize=8);fig.tight_layout(rect=(0,.065,1,.94));plots.save_figure(fig,output,cohort+'_'+kind);plt.close(fig)
    return records


def probability_curves(y,probabilities,cohort,output,kind):
    plt=plots.setup_plotting();fig,axes=plt.subplots(2,2,figsize=(11.5,9.1));scores=[];handles=[]
    for k,ax in enumerate(axes.flat):
        binary=(y==k).astype(int)
        for (name,prob),color in zip(probabilities.items(),COLORS):
            if len(np.unique(binary))<2:continue
            if kind=='roc':xx,yy,_=roc_curve(binary,prob[:,k],drop_intermediate=False);score=roc_auc_score(binary,prob[:,k]);line,=ax.plot(xx,yy,color=color,lw=1.4)
            else:yy,xx,_=precision_recall_curve(binary,prob[:,k]);score=average_precision_score(binary,prob[:,k]);line,=ax.step(xx,yy,where='post',color=color,lw=1.4)
            if k==0:handles.append(line)
            scores.append(dict(cohort=cohort,model=name,stage=stats.STAGES[k],metric='ROC_AUC' if kind=='roc' else 'average_precision',value=float(score)))
        ax.set(title=stats.STAGES[k]+f' vs rest (positive n={int(binary.sum())})',xlabel='False-positive rate' if kind=='roc' else 'Recall',ylabel='Sensitivity' if kind=='roc' else 'Precision',xlim=(0,1),ylim=(0,1.02))
        if kind=='roc':ax.plot([0,1],[0,1],ls=':',color='#999999')
    fig.suptitle(TITLES[cohort]+' | '+('ROC' if kind=='roc' else 'Precision–recall'),fontweight='bold');fig.legend(handles,[DISPLAY[n] for n in probabilities],loc='lower center',bbox_to_anchor=(.5,.05),ncol=3,fontsize=9);fig.text(.03,.02,NOTE,fontsize=8);fig.tight_layout(rect=(0,.12,1,.94));plots.save_figure(fig,output,cohort+'_'+kind);plt.close(fig);return scores


def comparison_figures(analyses,output):
    plt=plots.setup_plotting();fig,axes=plt.subplots(2,2,figsize=(13.5,9))
    for ax,(cohort,data) in zip(axes.flat,analyses.items()):
        names=list(data['models']);points=[data['models'][n]['point']['accuracy'] for n in names]
        cis=[data['models'][n]['bootstrap']['accuracy']['ci95'] for n in names]
        errors=np.array([[max(0,v-ci[0]),max(0,ci[1]-v)] if ci else [0,0] for v,ci in zip(points,cis)]).T
        colors=COLORS+['#9b8c72']*(len(names)-5);ax.barh(range(len(names)),100*np.array(points),color=colors);ax.errorbar(100*np.array(points),range(len(names)),xerr=100*errors,fmt='none',ecolor='black',capsize=3)
        ax.set_yticks(range(len(names)),[DISPLAY.get(n,n.replace('Reader_','Reader ')) for n in names],fontsize=9);ax.invert_yaxis();ax.set(title=TITLES[cohort],xlabel='Accuracy (%)',xlim=(0,105))
        for i,v in enumerate(points):ax.text(min(100*v+1,100),i,f'{100*v:.2f}',va='center',fontsize=8)
    fig.suptitle('All fixed models and readers | conditional 95% intervals',fontweight='bold');fig.text(.025,.02,'Paired 2,000-draw hash-group bootstrap; original100 image units. Degenerate intervals do not prove certainty.\nThree-reader range is descriptive across fixed readers; repeated benchmarks cannot establish independent superiority.',fontsize=8);fig.tight_layout(rect=(0,.07,1,.94));plots.save_figure(fig,output,'all_cohort_accuracy');plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,5.4))
    for ax,metric in zip(axes,['accuracy','balanced_accuracy','macro_f1']):
        for j,(cohort,data) in enumerate(analyses.items()):
            vals=[data['models'][n]['point'][metric] for n in p.NEW];ax.scatter([j-.12,j,j+.12],np.array(vals)*100,c=COLORS[2:]);ax.plot([j-.15,j+.15],[np.mean(vals)*100]*2,color='black',lw=2)
        ax.set_xticks(range(4),['5,450','1,058','756','100']);ax.set(xlabel='Reused cohort size',ylabel=metric.replace('_',' ')+' (%)');ax.grid(axis='y',alpha=.2)
    fig.suptitle('All three round2 seeds | points and within-condition mean',fontweight='bold');fig.text(.03,.03,'Seed variation is descriptive; the validation representative is fixed independently of these outcomes.',fontsize=9);fig.tight_layout(rect=(0,.08,1,.92));plots.save_figure(fig,output,'round2_seed_spread');plt.close(fig)


def training_figures(suite,freeze,output):
    plt=plots.setup_plotting();fig,axes=plt.subplots(3,3,figsize=(14,10.8))
    for j,(name,selection) in enumerate(freeze['all_three_runs'].items()):
        folder=suite/Path(selection['checkpoint']).parent;hist=p.read(folder/'history.json')
        if p.sha(folder/'history.json')!=selection['history_sha256']:raise ValueError('Training history changed')
        ep=[r['epoch'] for r in hist];train=[r for r in hist if 'train_raw_nll' in r]
        axes[0,j].plot([r['epoch'] for r in train],[r['train_raw_nll'] for r in train],label='Training raw NLL');axes[0,j].plot(ep,[r['clean_nll'] for r in hist],label='Clean validation NLL');axes[0,j].plot(ep,[r['stress_nll'] for r in hist],label='Validation stress NLL');axes[0,j].set_title(DISPLAY[name]);axes[0,j].set_ylabel('Raw cross-entropy NLL');axes[0,j].legend(fontsize=8)
        axes[1,j].plot(ep,[r['score'] for r in hist],color=COLORS[j+2]);bad=[r for r in hist if not r['eligible']];axes[1,j].scatter([r['epoch'] for r in bad],[r['score'] for r in bad],marker='x',color='#bd3c3c',label='Clean guard rejected');axes[1,j].set_ylabel('Fixed validation selection score');axes[1,j].legend(fontsize=8)
        axes[2,j].plot([r['epoch'] for r in train],[100*r['train_accuracy'] for r in train],label='Training draw accuracy');axes[2,j].plot(ep,[100*r['clean_accuracy'] for r in hist],label='Clean validation');axes[2,j].plot(ep,[100*r['stress_accuracy'] for r in hist],label='Stress validation');axes[2,j].set_ylabel('Accuracy (%)');axes[2,j].legend(fontsize=8)
        for ax in axes[:,j]:ax.axvline(selection['epoch'],ls=':',color='black');ax.set_xlabel('Epoch (0 = initialization)');ax.grid(alpha=.2)
    fig.suptitle('Round2 raw learning curves | all three runs, selected epoch marked',fontweight='bold');fig.text(.03,.018,'Training includes random augmentation; validation is fixed. Raw NLL differs from the weighted, label-smoothed optimization objective.\nEpoch0 fallback, if selected, is retained; longer training is not evidence of a better model.',fontsize=9);fig.tight_layout(rect=(0,.07,1,.95));plots.save_figure(fig,output,'raw_learning_curves');plt.close(fig)


def stress_figures(records,output):
    plt=plots.setup_plotting();families=list(dict.fromkeys(r['family'] for r in records if r['family']!='clean'))
    for cohort in ['internal_same_case','original100']:
        for metric in ['accuracy','label_flip_fraction']:
            fig,axes=plt.subplots(2,3,figsize=(14,8.4))
            for ax,family in zip(axes.flat,families):
                for name,color in zip(p.ORDER,COLORS):
                    clean=next(r for r in records if r['cohort']==cohort and r['model']==name and r['scenario']=='clean');rr=[r for r in records if r['cohort']==cohort and r['model']==name and r['family']==family];values=[clean]+rr
                    ax.plot(range(len(values)),[100*r[metric] for r in values],marker='o',ms=3,color=color,label=DISPLAY[name]);ax.set_xticks(range(len(values)),['Clean']+[str(r['value']) for r in rr]);ax.set(title=family.replace('_',' '),ylabel=metric.replace('_',' ')+' (%)');ax.grid(alpha=.2)
            for ax in list(axes.flat)[len(families):]:ax.axis('off')
            fig.suptitle(TITLES[cohort]+' | fixed stress '+metric.replace('_',' '),fontweight='bold');handles,labels=axes.flat[0].get_legend_handles_labels();fig.legend(handles,labels,ncol=3,loc='lower center',bbox_to_anchor=(.5,.04),fontsize=9);fig.text(.02,.013,'Same-device clean reference; all13 unchanged scenarios. Synthetic pixel stress is not MRI acquisition or clinical validation.',fontsize=8);fig.tight_layout(rect=(0,.12,1,.94));plots.save_figure(fig,output,cohort+'_'+metric+'_stress');plt.close(fig)


def corner_figure(records,output):
    plt=plots.setup_plotting();fig,axes=plt.subplots(1,2,figsize=(14,6.2));names=list(dict.fromkeys(r['scenario'] for r in records));labels=[n.replace('imagenet_mean','mean').replace('_',' ') for n in names]
    for name,color in zip(p.ORDER,COLORS):
        by={r['scenario']:r for r in records if r['model']==name}
        for ax,metric in zip(axes,['accuracy','mean_total_variation']):
            ax.plot(range(len(names)),[by[n][metric]*(100 if metric=='accuracy' else 1) for n in names],marker='o',ms=3,color=color,label=DISPLAY[name]);ax.set_xticks(range(len(names)),labels,rotation=62,ha='right',fontsize=7);ax.set_ylabel('Accuracy (%)' if metric=='accuracy' else 'Mean probability total variation');ax.grid(alpha=.2)
    fig.suptitle('Fixed160 validation images | causal input masking controls',fontweight='bold');fig.legend(*axes[0].get_legend_handles_labels(),ncol=3,loc='lower center',bbox_to_anchor=(.5,.04),fontsize=9);fig.text(.02,.012,'Erasure is out of distribution. Matched center areas and both fills are controls; these results cannot exclude all shortcuts.',fontsize=8);fig.tight_layout(rect=(0,.14,1,.94));plots.save_figure(fig,output,'validation_corner_controls');plt.close(fig)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--suite',type=Path,required=True);ap.add_argument('--data-root',type=Path,default=p.ROOT);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args();suite=args.suite.resolve();root=args.data_root.resolve();output=args.output.resolve()
    if output.exists():raise ValueError('Report output must be new')
    source,evaluation=verified_outputs(suite);p.load_frozen(root,suite);cohorts=p.rows_for(root);analyses={};calibration=[];curves=[];seed_spread={};verification=[];output.mkdir(parents=True)
    clean_rows=p.read(source/'clean_metrics.json');clean={(r['model'],r['cohort']):r for r in clean_rows}
    if len(clean)!=20 or len(clean_rows)!=20:raise ValueError('Missing or duplicate clean metric records')
    verification+=verify_stress_corner(source,cohorts)
    for cohort in TITLES:
        expected=cohorts[cohort];y=np.array([int(r['label']) for r in expected]);probs={n:aligned_probabilities(p.stress.csv_read(source/f'{n}_{cohort}_predictions.csv'),expected) for n in p.ORDER};preds={n:v.argmax(1) for n,v in probs.items()};pairs=[(n,ref) for n in p.NEW for ref in ['old_baseline','round1_c3']]
        for name,prob in probs.items():
            m=p.stress.metrics(y,prob,prob)
            for field,value in m.items():
                if not np.allclose(np.asarray(value),np.asarray(clean[name,cohort][field]),atol=1e-10,rtol=1e-9):raise ValueError('Stored metric differs: '+name+' '+cohort+' '+field)
            verification.append(dict(cohort=cohort,model=name,n=len(y),metrics_recomputed=True))
        if cohort=='original100':
            rr=p.stress.csv_read(source/'original100_case_aligned_models_readers.csv');by={r['FileName']:r for r in rr};names=[k for k in rr[0] if k.startswith('Reader_')]
            if len(names)!=3 or len(by)!=len(rr) or set(by)!={r['FileName'] for r in expected}:raise ValueError('Reader completeness mismatch')
            for name in names:
                if any(by[r['FileName']]['TrueCategory']!=stats.STAGES[int(r['label'])] for r in expected):raise ValueError('Reader truth mismatch')
                preds[name]=np.array([stats.STAGES.index(by[r['FileName']][name]) for r in expected]);pairs += [(n,name) for n in p.NEW]
        analysis=stats.bootstrap_analysis(y,preds,[r['group_id'] for r in expected],'round2_fixed_'+cohort,pairs);analysis['interpretation']=p.read(p.HERE/'protocol.json')['cohorts'][cohort]['interpretation'];analyses[cohort]=analysis
        seed_spread[cohort]={metric:dict(values={n:analysis['models'][n]['point'][metric] for n in p.NEW},mean=float(np.mean([analysis['models'][n]['point'][metric] for n in p.NEW])),sample_sd=float(np.std([analysis['models'][n]['point'][metric] for n in p.NEW],ddof=1))) for metric in stats.METRICS}
        plots.confusion_panels(y,preds,{n:DISPLAY.get(n,n.replace('Reader_','Reader ')) for n in preds},TITLES[cohort]+' | confusion',NOTE,output,cohort+'_confusion',False)
        cal=calibration_panels(y,probs,cohort,output,'reliability');calibration_panels(y,probs,cohort,output,'confidence')
        for name,v in cal.items():calibration.append(dict(cohort=cohort,model=name,**v))
        for kind in ['roc','pr']:curves+=probability_curves(y,probs,cohort,output,kind)
    comparison_figures(analyses,output);training_figures(suite,evaluation['freeze'],output);stress_figures(p.read(source/'stress_metrics.json'),output);corner_figure(p.read(source/'corner_metrics.json'),output)
    reader_points=[v['point']['accuracy'] for k,v in analyses['original100']['models'].items() if k.startswith('Reader_')]
    report=dict(status='complete',evaluation_sha256=p.sha(source/'evaluation.json'),freeze_sha256=p.sha(suite/'round2_evaluation_freeze.json'),all_models=p.ORDER,representative=evaluation['freeze']['representative'],cohorts=analyses,round2_seed_spread=seed_spread,readers=dict(n_readers=3,mean_accuracy=float(np.mean(reader_points)),range_accuracy=[min(reader_points),max(reader_points)],interpretation='Descriptive fixed-reader range, not confidence interval'),calibration=calibration,metric_recomputation=verification,protocol=p.read(p.HERE/'protocol.json'),clinical_head_evaluated=False)
    p.dump(output/'analysis.json',report);p.stress.csv_write(output/'probability_curve_metrics.csv',curves);p.dump(output/'figure_inputs.json',dict(stress=p.read(source/'stress_metrics.json'),corner=p.read(source/'corner_metrics.json')))
    lines=['# Round2 repeated-benchmark comparison','','All three runs are shown. The representative was selected only by the fixed validation criterion. Every test cohort was previously seen during project development; none is a fresh clinical test.','','| Cohort | Old | Round1 c3 | Seed11 | Seed12 | Seed13 |','|---|---:|---:|---:|---:|---:|']
    for co,a in analyses.items():lines.append('| '+TITLES[co]+' | '+' | '.join(f"{100*a['models'][n]['point']['accuracy']:.2f}%" for n in p.ORDER)+' |')
    lines += ['',f"Reader accuracy: mean {100*np.mean(reader_points):.2f}%; range {100*min(reader_points):.2f}–{100*max(reader_points):.2f}% across these three fixed readers.",'','Intervals in analysis.json are conditional paired percentile intervals; hash groups are not patient identities, and ceiling/degenerate intervals do not remove dataset limitations. No outcome-based winner or clinical-readiness claim is made.','', 'All figures require visual inspection before publication.']
    (output/'README.md').write_text('\n'.join(lines)+'\n');manifest={f.name:p.sha(f) for f in sorted(output.iterdir()) if f.is_file()};p.dump(output/'SHA256SUMS.json',manifest);p.dump(output/'report_complete.json',dict(status='complete_numerically_verified_visual_review_required',figures=len(list(output.glob('*.pdf'))),payloads=len(manifest),manifest_sha256=p.sha(output/'SHA256SUMS.json')));print(p.read(output/'report_complete.json'))

if __name__=='__main__':main()
