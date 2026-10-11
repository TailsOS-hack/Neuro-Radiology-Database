#!/usr/bin/env python3
"""Recreate Kevin's quantitative paper figures from verified, frozen artifacts.

No model forward passes, fitting, selection, threshold changes or cloud actions.
Trusted, hash-checked best.pt files are read only for stored validation matrices.
Run from any directory; all default paths and output provenance are repo-relative.
"""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import sys
import textwrap

import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(HERE.parent/'analysis'))
import summarize_frozen_evaluation as stats
import ensemble_supplement as ensemble
import plot_evaluation_figures as prior_plot
import plot_training_curves as curves
from sklearn.metrics import roc_curve,roc_auc_score,precision_recall_curve,average_precision_score

COHORTS=['internal_same_case','old_hash_group_unexposed_sensitivity','original100']
TITLE={'internal_same_case':'Primary 1,058 images: exact-row held out',
       'old_hash_group_unexposed_sensitivity':'Secondary sensitivity: 756 images',
       'original100':'Original 100: historically reused image benchmark'}
NOTE={'internal_same_case':'302 images have old development hash-group mates. Hash groups are not patient identities.',
      'old_hash_group_unexposed_sensitivity':'Excludes observed old development hash-group mates; patient independence remains unknown.',
      'original100':'Old-model exposure: 66 training, 9 validation, 25 test images. No independent clinical claim.'}
SHORT=['Mild','Moderate','Non','Very mild']
ENSEMBLES=['ensemble_stage_only','ensemble_multitask']
COLORS=['#252525','#0072B2','#56B4E9','#009E73','#D55E00','#CC79A7','#E69F00','#003F5C','#7A1E5D']
RESULTS=Path('training_logs/experiments/kevin_20261011/runpod_blackwell_20261011')
FIGURE_TITLES={
    'clinical_accuracy_all_three_seeds':'Clinical AD/CN training and validation accuracy: all three multitask seeds',
    'clinical_loss_all_three_multitask_seeds':'Clinical AD/CN training and validation loss: all three multitask seeds',
    'clinical_confusion_all_multitask':'Clinical subject confusion matrices: all multitask seeds and supplementary ensemble',
    'clinical_exact_intervals':'Clinical accuracy, AD sensitivity and CN specificity with exact 95% intervals',
    'stage_accuracy_all_six_runs':'Four-stage training and validation accuracy: all six runs',
    'stage_loss_all_six_runs':'Four-stage training and validation loss: all six runs',
    'stage_seed_spread':'Three-seed variability for both conditions across all image benchmarks',
    'stage_validation_confusion_all_six':'Stored validation confusion matrices: all six selected checkpoints',
    'train_resubstitution_confusion_all_six':'Post-hoc training resubstitution confusion matrices: all six checkpoints',
    'full_new_test_confusion_all_six':'Post-hoc full new test confusion matrices: all six checkpoints',
    'original100_models_and_fixed_readers':'Reused original100: models, individual readers and descriptive reader range',
    'original100_per_class_models_readers':'Reused original100: per-class recall for all models and fixed readers',
    'original100_fixed_reader_confusion':'Reused original100: confusion matrices for the three fixed readers',
    'paired_accuracy_differences':'Paired accuracy differences from the old baseline with conditional 95% intervals',
    'dementia_study_workflow':'Dementia study workflow and separate image and clinical endpoints',
    'dementia_two_condition_architecture':'MobileNetV3 architecture: stage-only and multitask conditions',
}


def figure_title(stem):
    if stem in FIGURE_TITLES:return FIGURE_TITLES[stem]
    families={'confusion_all_nine':'confusion matrices','reliability_all_nine':'reliability diagrams',
              'confidence_all_nine':'confidence distributions','roc_all_nine':'one-versus-rest ROC curves',
              'pr_all_nine':'precision-recall curves','errors_all_nine':'complete observed-error audit'}
    for cohort in COHORTS:
        if stem.startswith(cohort+'_'):
            return TITLE[cohort]+' | '+families[stem[len(cohort)+1:]]+' for all nine models'
    raise ValueError('Missing figure title: '+stem)


def relative(path):return str(Path(path).resolve().relative_to(ROOT))


def relocate(path):
    """Relocate archived absolute provenance paths using repository path anchors."""
    p=Path(path)
    if not p.is_absolute():return ROOT/p
    parts=p.parts
    for anchor in ('training_logs','docs','data'):
        if anchor in parts:return ROOT/Path(*parts[parts.index(anchor):])
    raise ValueError('Input provenance is not a supported repository artifact: '+str(path))


def load_inputs(result_root):
    args=argparse.Namespace(evaluation_dir=result_root/'frozen_evaluation_all6',
        prepared=ROOT/'training_logs/experiments/kevin_20261011/prepared',
        original100_key=ROOT/'data/evaluation/ground_truth/radiologist_test_key.csv',
        validation_selection=result_root/'validation_selection_before_evaluation.json',
        old_hash_unexposed_manifest=HERE.parent/'analysis/old_hash_group_unexposed_sensitivity.csv',
        baseline_exposure_audit=HERE.parent/'analysis/old_hash_group_unexposed_sensitivity_audit.json',
        baseline_checkpoint_sha256=ensemble.BASELINE_SHA256,synthetic_qa=False)
    base=prior_plot.collect(args)
    for helper in (stats,ensemble,prior_plot,curves):
        base['inputs_sha256'][str(Path(helper.__file__).resolve())]=stats.digest(helper.__file__)
    ppath=result_root/'frozen_uncertainty_analysis_all6/uncertainty_analysis.json'
    epath=result_root/'ensemble_supplement/ensemble_supplement.json'
    primary=json.loads(ppath.read_text());supplement=json.loads(epath.read_text())
    for item in primary['input_manifest']:
        p=relocate(item['path'])
        if stats.digest(p)!=item['sha256']:raise ValueError('Frozen primary input digest changed: '+relative(p))
        base['inputs_sha256'][str(p)]=item['sha256']
    if primary['software']['analysis_script_sha256']!=stats.digest(stats.__file__):raise ValueError('Frozen primary helper changed')
    if supplement['provenance']['primary_analysis_sha256']!=stats.digest(ppath):raise ValueError('Ensemble binds different primary analysis')
    if supplement['provenance']['code_hashes']!=ensemble.code_hashes():raise ValueError('Frozen ensemble code/protocol changed')
    for p in (ppath,epath,result_root/'ensemble_supplement_freeze.json'):
        base['inputs_sha256'][str(p)]=stats.digest(p)
    if supplement['provenance']['supplement_freeze_sha256']!=stats.digest(result_root/'ensemble_supplement_freeze.json'):
        raise ValueError('Ensemble freeze mismatch')
    for name in ENSEMBLES:
        base['display'][name]=('Stage ensemble' if name.endswith('stage_only') else 'Multitask ensemble')+' | supplementary'
    base['names']+=ENSEMBLES
    numerical=[]
    for cohort,dataset in base['datasets'].items():
        for name in ENSEMBLES:
            file=result_root/'ensemble_supplement'/f'{name}_{cohort}_predictions.csv'
            rows=stats.index_rows(stats.read_csv(file,['case_id','TrueCategory','ModelPrediction']),'case_id',file.name)
            stats.require_same(rows,dataset['ids'],file.name)
            prob=np.asarray([stats.validate_probabilities(rows[i],['P_'+cl for cl in stats.STAGES],file.name) for i in dataset['ids']])
            specs=supplement['ensemble_specifications'][name]
            members=[m['evaluation_name'] for m in specs['members']]
            if len(members)!=3 or any(m['weight_fraction']!='1/3' for m in specs['members']):raise ValueError('Nonuniform ensemble')
            expected=np.mean([dataset['models'][m]['prob'] for m in members],axis=0)
            if not np.allclose(prob,expected,atol=1e-14,rtol=0):raise ValueError('Exported ensemble differs from fixed member mean')
            pred=prob.argmax(1)
            for j,i in enumerate(dataset['ids']):
                if rows[i]['TrueCategory']!=stats.STAGES[dataset['y'][j]] or rows[i]['ModelPrediction']!=stats.STAGES[pred[j]]:raise ValueError('Ensemble label mismatch')
            dataset['models'][name]={'pred':pred,'prob':prob}
            base['inputs_sha256'][str(file)]=stats.digest(file)
        for name,m in dataset['models'].items():
            expected=(supplement if name in ENSEMBLES else primary)['stage'][cohort]['models'][name]['point']
            actual=stats.class_metrics(dataset['y'],m['pred'],4)
            if actual['confusion_matrix']!=expected['confusion_matrix'] or any(not np.isclose(actual[k],expected[k],atol=1e-14,rtol=0) for k in stats.METRICS):
                raise ValueError('Plot values differ from frozen statistics')
            numerical.append({'cohort':cohort,'model':name,'n':len(dataset['y']),'point':actual,
                              'calibration':prior_plot.calibration(dataset['y'],m['prob'])})
    clinical=stats.clinical_subjects(ROOT/'data/external/clinical_t1_preprocessed_v1/clinical_slices.csv')
    for cohort,expected in clinical.items():
        for name,reference in primary['clinical'][cohort].items():
            file=result_root/'frozen_evaluation_all6'/f'{name}_{cohort}_subject_predictions.csv'
            actual=stats.clinical_analysis(file,expected)
            if any(actual[k]!=reference[k] for k in ('point','accuracy_exact','AD_sensitivity_exact','CN_specificity_exact')):
                raise ValueError('Clinical plot metrics differ from source subject predictions')
        file=result_root/'ensemble_supplement'/f'ensemble_multitask_AD_CN_{cohort}_subject_predictions.csv'
        rows=stats.index_rows(stats.read_csv(file,['subject_id','binary_label','P_CN','P_AD']),'subject_id',file.name)
        stats.require_same(rows,expected,file.name)
        ids=sorted(expected);prob=np.asarray([stats.validate_probabilities(rows[i],['P_CN','P_AD'],file.name) for i in ids])
        for i in ids:
            if rows[i]['binary_label']!=expected[i]['label']:raise ValueError('Clinical ensemble reference label mismatch')
        scored=[j for j,i in enumerate(ids) if expected[i]['label']!=''];y=np.array([int(expected[ids[j]]['label']) for j in scored]);pred=prob[scored].argmax(1)
        point=stats.class_metrics(y,pred,2);reference=supplement['clinical'][cohort]
        if point!=reference['point']:raise ValueError('Clinical ensemble plot differs from CSV')
        for metric,k,n in [('accuracy_exact',int((y==pred).sum()),len(y)),('AD_sensitivity_exact',int((pred[y==1]==1).sum()),int((y==1).sum())),('CN_specificity_exact',int((pred[y==0]==0).sum()),int((y==0).sum()))]:
            if stats.exact_interval(k,n)!=reference[metric]:raise ValueError('Clinical ensemble exact interval mismatch')
        base['inputs_sha256'][str(file)]=stats.digest(file)
    _,runs=curves.load_runs(result_root,args.validation_selection)
    for (mode,seed),run in runs.items():
        for file in ('config.json','history.json','best.pt'):
            path=result_root/f'{mode}_seed{seed}'/file;base['inputs_sha256'][str(path)]=stats.digest(path)
    # Only trusted checkpoints, already verified against the pre-evaluation hashes.
    import torch
    validation={}
    for (mode,seed),run in runs.items():
        ck=torch.load(result_root/f'{mode}_seed{seed}/best.pt',map_location='cpu',weights_only=False)
        metric=ck['stage_validation'];cm=np.asarray(metric['confusion_matrix'])
        if cm.shape!=(4,4) or cm.sum()!=metric['n'] or not np.isclose(cm.trace()/cm.sum(),metric['accuracy']):raise ValueError('Invalid saved validation matrix')
        if ck['epoch']!=run['selected_epoch'] or ck['config']['mode']!=mode or ck['config']['seed']!=seed:raise ValueError('Validation checkpoint identity mismatch')
        validation[run['candidate']]=metric
        del ck
    return base,primary,supplement,runs,validation,numerical


def plotting():
    plt=prior_plot.setup_plotting()
    plt.rcParams.update({'font.size':9,'axes.titleweight':'normal','savefig.dpi':300})
    return plt


def finish(fig,out,stem,title,note):
    fig.suptitle(title,fontsize=15,fontweight='bold',y=.985)
    fig.text(.045,.018,'\n'.join(textwrap.wrap(note,155)),fontsize=8,va='bottom',linespacing=1.35)
    paths=[]
    for ext in ('png','pdf'):
        path=out/f'{stem}.{ext}';fig.savefig(path,dpi=300,facecolor='white');paths.append(path)
    plotting().close(fig)
    return paths


def reliability(dataset,names,display,cohort,out):
    plt=plotting();fig,axes=plt.subplots(3,3,figsize=(12.5,10),sharex=True,sharey=True)
    for ax,name,color in zip(axes.flat,names,COLORS):
        cal=prior_plot.calibration(dataset['y'],dataset['models'][name]['prob']);bins=[b for b in cal['bins'] if b['n']]
        ax.plot([0,1],[0,1],':',color='#999999',lw=1)
        ax.plot([b['mean_confidence'] for b in bins],[b['accuracy'] for b in bins],'-',color=color,lw=1)
        ax.scatter([b['mean_confidence'] for b in bins],[b['accuracy'] for b in bins],s=[20+120*b['n']/len(dataset['y']) for b in bins],color=color)
        ax.text(.04,.08,f"Brier sum {cal['multiclass_brier_sum']:.4f}\nECE {cal['top_label_ece_10_equal_width_bins']:.4f}",transform=ax.transAxes,fontsize=8)
        ax.set(title=display[name].replace(' | ','\n'),xlim=(-.03,1.03),ylim=(-.03,1.03));ax.grid(alpha=.15)
    for ax in axes[-1]:ax.set_xlabel('Mean top-class probability in bin')
    for ax in axes[:,0]:ax.set_ylabel('Fraction correct in bin')
    fig.subplots_adjust(left=.07,right=.985,bottom=.13,top=.88,wspace=.22,hspace=.55)
    return finish(fig,out,cohort+'_reliability_all_nine',TITLE[cohort]+' | reliability',NOTE[cohort]+' Ten fixed equal-width bins; empty bins omitted; marker area reflects count. Descriptive only; no recalibration or fitted thresholds. Ensembles are supplementary.')


def confidence(dataset,names,display,cohort,out):
    plt=plotting();fig,axes=plt.subplots(3,3,figsize=(12.5,10),sharex=True)
    records=[]
    for ax,name in zip(axes.flat,names):
        data=dataset['models'][name];conf=data['prob'].max(1);correct=data['pred']==dataset['y']
        ax.hist([conf[correct],conf[~correct]],bins=np.linspace(0,1,21),stacked=True,color=['#4D859E','#D55E00'],label=['Correct','Error'])
        ax.set(title=display[name].replace(' | ','\n'),xlim=(0,1));ax.grid(axis='y',alpha=.15)
        ax.text(.04,.93,f'n={len(conf)}; errors={int((~correct).sum())}',transform=ax.transAxes,va='top',fontsize=8)
        counts=[np.histogram(conf[mask],bins=np.linspace(0,1,21))[0].tolist() for mask in (correct,~correct)]
        records.append({'cohort':cohort,'model':name,'bin_edges':np.linspace(0,1,21).tolist(),'correct_counts':counts[0],'error_counts':counts[1]})
    for ax in axes[-1]:ax.set_xlabel('Maximum predicted probability')
    for ax in axes[:,0]:ax.set_ylabel('Image count')
    fig.legend(*axes.flat[0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,.945),ncol=2,frameon=False)
    fig.subplots_adjust(left=.07,right=.985,bottom=.12,top=.86,wspace=.24,hspace=.6)
    return finish(fig,out,cohort+'_confidence_all_nine',TITLE[cohort]+' | confidence distributions',NOTE[cohort]+' Fixed 0.05-wide bins; panel count axes vary. Confidence is not validated clinical certainty. Both ensembles are supplementary.'),records


def probability_curves(dataset,names,display,cohort,out,kind):
    plt=plotting();fig,axes=plt.subplots(2,2,figsize=(12.5,10.5));scores=[]
    for k,ax in enumerate(axes.flat):
        y=(dataset['y']==k).astype(int)
        for j,name in enumerate(names):
            p=dataset['models'][name]['prob'][:,k]
            if kind=='roc':x,v,_=roc_curve(y,p);score=roc_auc_score(y,p)
            else:v,x,_=precision_recall_curve(y,p);score=average_precision_score(y,p)
            draw=ax.plot if kind=='roc' else ax.step
            draw(x,v,color=COLORS[j],ls=':' if name in ENSEMBLES else ('--' if 'Multitask' in display[name] else '-'),lw=1.6,label=display[name],**({} if kind=='roc' else {'where':'post'}))
            scores.append({'cohort':cohort,'model':name,'class':stats.STAGES[k],'metric':'ROC_AUC' if kind=='roc' else 'average_precision','value':float(score)})
        if kind=='roc':ax.plot([0,1],[0,1],':',color='#BBBBBB');ax.set(xlabel='False-positive rate',ylabel='True-positive rate')
        else:ax.axhline(y.mean(),ls=':',color='#BBBBBB');ax.set(xlabel='Recall',ylabel='Precision')
        ax.set(title=f'{SHORT[k]} versus rest | positive n={y.sum()}',xlim=(-.015,1.015),ylim=(-.015,1.025));ax.grid(alpha=.15)
    fig.legend(*axes.flat[0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,.945),ncol=3,frameon=False,fontsize=8)
    fig.subplots_adjust(left=.07,right=.985,bottom=.13,top=.8,hspace=.33,wspace=.22)
    return finish(fig,out,cohort+'_'+kind+'_all_nine',TITLE[cohort]+(' | ROC' if kind=='roc' else ' | precision–recall'),NOTE[cohort]+' Fixed model order; overlapping near-ceiling curves remain in the numeric table. Curves use frozen probabilities, with no threshold selection. Ensembles are supplementary.'),scores


def errors(dataset,names,display,cohort,out):
    plt=plotting();matrix=np.asarray([dataset['models'][n]['pred']!=dataset['y'] for n in names]);any_error=np.flatnonzero(matrix.any(0))
    fig,(left,right)=plt.subplots(1,2,figsize=(14,7),gridspec_kw={'width_ratios':[1,1.8]})
    left.barh(range(len(names)),matrix.sum(1),color=COLORS);left.set_yticks(range(len(names)),[display[n].replace(' | ','\n') for n in names],fontsize=8);left.invert_yaxis()
    left.set(xlabel='Number of image errors',xlim=(0,max(matrix.sum(1))+1));left.grid(axis='x',alpha=.15)
    for i,c in enumerate(matrix.sum(1)):left.text(c+.06,i,str(c),va='center')
    if len(any_error):
        right.imshow(matrix[:,any_error],vmin=0,vmax=1,cmap='Oranges',aspect='auto')
        codes=['M','Mod','N','VM']
        right.set_xticks(range(len(any_error)),[f'E{i+1}\n{codes[dataset["y"][p]]}' for i,p in enumerate(any_error)],fontsize=8)
        compact={n:(display[n].replace(' | seed 202610',' seed') if n not in ENSEMBLES else display[n].split(' | ')[0]) for n in names}
        right.set_yticks(range(len(names)),[compact[n] for n in names],fontsize=8)
        for i in range(len(names)):
            for j,p in enumerate(any_error):right.text(j,i,codes[dataset['models'][names[i]]['pred'][p]] if matrix[i,p] else '✓',ha='center',va='center',fontsize=8,color='white' if matrix[i,p] else 'black')
        right.set_xlabel('Union of misclassified cases; reference class beneath case code')
    else:right.text(.5,.5,'No observed errors in this cohort',ha='center');right.axis('off')
    fig.subplots_adjust(left=.17,right=.985,bottom=.2,top=.86,wspace=.48)
    rows=[{'cohort':cohort,'case_code':f'E{i+1}','case_id':dataset['ids'][p],'reference':stats.STAGES[dataset['y'][p]],**{n:stats.STAGES[dataset['models'][n]['pred'][p]] for n in names}} for i,p in enumerate(any_error)]
    return finish(fig,out,cohort+'_errors_all_nine',TITLE[cohort]+' | error audit',NOTE[cohort]+' Every missed case is shown; orange means error. M=Mild, Mod=Moderate, N=Non, VM=Very mild. Case identities: error_cases.csv. Both ensembles are supplementary.'),rows


def reader_figures(base,primary,supplement,out):
    plt=plotting();names=base['names']+list(base['readers']);display={**base['display'],**{n:n.replace('Reader_','') for n in base['readers']}}
    points={n:stats.class_metrics(base['datasets']['original100']['y'],(base['datasets']['original100']['models'][n]['pred'] if n in base['names'] else base['readers'][n]),4) for n in names}
    fig,axes=plt.subplots(1,3,figsize=(15,7),sharey=True)
    for ax,metric in zip(axes,stats.METRICS):
        vals=[points[n][metric]*100 for n in names];ax.barh(range(len(names)),vals,color=COLORS+['#888888']*3)
        for i,v in enumerate(vals):ax.text(v+.6,i,f'{v:.1f}',va='center',fontsize=7)
        mean=np.mean([points[n][metric] for n in base['readers']])*100;ax.axvline(mean,color='#555555',ls=':',label=f'Fixed-reader mean {mean:.1f}%')
        reader_values=[points[n][metric]*100 for n in base['readers']]
        ax.axvspan(min(reader_values),max(reader_values),color='#888888',alpha=.12,label=f'Reader range {min(reader_values):.1f}-{max(reader_values):.1f}%')
        ax.set(xlim=(0,108),xlabel=metric.replace('_',' ').title()+' (%)');ax.grid(axis='x',alpha=.15);ax.legend(loc='lower right',fontsize=7,frameon=False)
    axes[0].set_yticks(range(len(names)),[display[n].replace(' | ','\n') for n in names],fontsize=8);axes[0].invert_yaxis()
    fig.subplots_adjust(left=.2,right=.985,bottom=.16,top=.88,wspace=.16)
    paths=finish(fig,out,'original100_models_and_fixed_readers','Original100 | all models and three fixed readers',NOTE['original100']+' Dotted lines and shaded bands show means and min-max ranges across these three readers, not confidence intervals. Both ensembles are supplementary. Reader-case rows are not pooled as independent observations.')
    fig,ax=plt.subplots(figsize=(11,7));recalls=np.asarray([np.diag(points[n]['confusion_matrix'])/np.asarray(points[n]['class_counts']) for n in names])
    im=ax.imshow(recalls,vmin=0,vmax=1,cmap='Blues',aspect='auto')
    for i,n in enumerate(names):
        cm=np.asarray(points[n]['confusion_matrix'])
        for k in range(4):ax.text(k,i,f'{cm[k,k]}/{cm[k].sum()} ({recalls[i,k]*100:.1f}%)',ha='center',va='center',color='white' if recalls[i,k]>.55 else 'black',fontsize=9)
    ax.set_xticks(range(4),stats.STAGES);ax.set_yticks(range(len(names)),[display[n].replace(' | ','\n') for n in names],fontsize=8)
    fig.colorbar(im,ax=ax,pad=.02,label='Recall for each image reference category')
    fig.subplots_adjust(left=.26,right=.985,bottom=.16,top=.86)
    paths+=finish(fig,out,'original100_per_class_models_readers','Original100 | per-class image recall',NOTE['original100']+' These legacy image categories are not a patient-level clinical staging reference. Counts preserve class denominators; ensembles are supplementary.')
    return paths


def seed_spread(primary,out):
    plt=plotting();fig,axes=plt.subplots(3,3,figsize=(12.5,10),sharex=True)
    for i,co in enumerate(COHORTS):
        for j,metric in enumerate(stats.METRICS):
            ax=axes[i,j]
            for x,(mode,color) in enumerate([('stage_only','#0072B2'),('multitask','#D55E00')]):
                v=primary['stage'][co]['primary_condition_summaries'][mode]['metrics'][metric]
                vals=np.array(list(v['per_candidate'].values()))*100
                ax.scatter(x+np.linspace(-.12,.12,3),vals,color=color,s=25,zorder=4)
                ax.errorbar(x,v['mean']*100,yerr=v['sample_sd']*100,fmt='_',color=color,capsize=5,markersize=15,lw=1.3)
            old=primary['stage'][co]['models']['old_baseline']['point'][metric]*100;ax.axhline(old,color='#333333',ls=':',label='Old baseline')
            ax.set_xticks([0,1],['Stage only','Multitask']);ax.set_ylim(98.5,100.5);ax.grid(axis='y',alpha=.15)
            if i==0:ax.set_title(metric.replace('_',' ').title())
            if j==0:ax.set_ylabel(['Primary1058','Sensitivity756','Reused100'][i]+' (%)')
    fig.subplots_adjust(left=.08,right=.985,bottom=.13,top=.87,hspace=.3,wspace=.22)
    return finish(fig,out,'stage_seed_spread','All three prespecified seeds | mean and spread','Each dot is one fixed seed; mark is its condition mean and whiskers are sample SD, not confidence intervals (SD bars may exceed 100%). Dotted line: old baseline. Axes show 98.5-100.5%. Primary has old development group exposure; sensitivity groups are not patients; original100 is reused.')


def paired_differences(base,primary,supplement,out):
    plt=plotting();fig,axes=plt.subplots(1,3,figsize=(14,7),sharey=True);names=base['names'][1:]
    for j,(ax,co) in enumerate(zip(axes,COHORTS)):
        for i,name in enumerate(names):
            source=supplement if name in ENSEMBLES else primary
            v=source['stage'][co]['paired_comparisons'][name+'_minus_old_baseline']['difference']['accuracy']
            estimate=v['estimate']*100;low,high=np.array(v['ci95'])*100
            ax.errorbar(estimate,i,xerr=[[estimate-low],[high-estimate]],fmt='D' if name in ENSEMBLES else 'o',color=COLORS[i+1],capsize=3,markersize=5)
        ax.axvline(0,color='#333333',ls=':',lw=1);ax.set_xlabel('Accuracy difference (percentage points)');ax.grid(axis='x',alpha=.15)
        ax.set_title(['Primary1058','Sensitivity756','Reused original100'][j]);ax.margins(x=.15)
    axes[0].set_yticks(range(len(names)),[base['display'][n].replace(' | ','\n') for n in names],fontsize=9);axes[0].invert_yaxis()
    fig.subplots_adjust(left=.20,right=.985,bottom=.19,top=.86,wspace=.20)
    return finish(fig,out,'paired_accuracy_differences','Paired accuracy differences from the old baseline','Points and 95% conditional paired bootstrap intervals are copied from frozen analyses (2,000 replicates, no multiplicity adjustment). Primary1058 includes 302 old development group mates; sensitivity756 excludes them. Hash groups are not patients; original100 is reused. Horizontal scales differ; zero-width intervals do not prove equivalence. Diamonds are supplementary ensembles.')


def clinical_figures(primary,supplement,out):
    plt=plotting();cohorts=['test__besga_zenodo_3935636','external__openneuro_ds007561'];names=['candidate_2','candidate_4','candidate_6','ensemble_multitask_AD_CN'];labels=['Multitask seed11','Multitask seed12','Multitask seed13','Multitask ensemble\n(supplementary)']
    fig,axes=plt.subplots(2,3,figsize=(13,7.8),sharex=True,sharey=True)
    metrics=['accuracy_exact','AD_sensitivity_exact','CN_specificity_exact']
    for i,co in enumerate(cohorts):
        values={**primary['clinical'][co],'ensemble_multitask_AD_CN':supplement['clinical'][co]}
        for j,metric in enumerate(metrics):
            ax=axes[i,j]
            for k,name in enumerate(names):
                v=values[name][metric];point=v['estimate']*100;lo,hi=np.array(v['ci95'])*100
                ax.errorbar(point,k,xerr=[[point-lo],[hi-point]],fmt='D' if k==3 else 'o',color=COLORS[[4,5,6,8][k]],capsize=3)
                ax.text(2,k-.17,f"{v['successes']}/{v['n']}",fontsize=8,ha='left',va='bottom')
            ax.set_xlim(-3,103);ax.set_xticks([0,25,50,75,100]);ax.grid(axis='x',alpha=.15)
            if i==0:ax.set_title(['Accuracy','AD sensitivity','CN specificity'][j])
            if j==0:ax.set_yticks(range(4),labels,fontsize=9);ax.set_ylabel(['Besga test: 11 subjects','OpenNeuro external: 15 evaluable'][i],labelpad=18)
            if i==1:ax.set_xlabel('Estimate and exact 95% interval (%)')
    axes[0,0].invert_yaxis();fig.subplots_adjust(left=.21,right=.985,bottom=.17,top=.86,hspace=.32,wspace=.15)
    paths=finish(fig,out,'clinical_exact_intervals','Clinical AD/CN | subject-level exact intervals','Counts label every estimate. Clopper–Pearson intervals remain very wide: the external cohort has only one AD and 14 CN. Five external MCI subjects are prediction-only. Old and stage-only models have no matching clinical head. The fixed ensemble is supplementary; no model was selected from these outcomes.')
    fig,axes=plt.subplots(2,4,figsize=(13,7.5))
    for i,co in enumerate(cohorts):
        values={**primary['clinical'][co],'ensemble_multitask_AD_CN':supplement['clinical'][co]}
        for j,name in enumerate(names):
            ax=axes[i,j];point=values[name]['point'];cm=np.array(point['confusion_matrix']);fraction=cm/cm.sum(1,keepdims=True)
            ax.imshow(fraction,vmin=0,vmax=1,cmap='Blues')
            for x in range(2):
                for y in range(2):ax.text(y,x,str(cm[x,y]),ha='center',va='center',color='white' if fraction[x,y]>.55 else 'black',fontsize=15)
            ax.set_xticks([0,1],['CN','AD']);ax.set_yticks([0,1],['CN','AD'])
            ax.set_title(labels[j]+f"\nBA {point['balanced_accuracy']*100:.1f}% | macro F1 {point['macro_f1']:.3f}",fontsize=9)
            if j==0:ax.set_ylabel(['Besga test\nReference','OpenNeuro external\nReference'][i])
            if i==1:ax.set_xlabel('Predicted')
    fig.subplots_adjust(left=.075,right=.985,bottom=.15,top=.84,hspace=.55,wspace=.3)
    paths+=finish(fig,out,'clinical_confusion_all_multitask','Clinical AD/CN | all seeds and fixed ensemble','Cells show subject counts; colors show within-reference-class recall. Besga: four CN and seven AD. OpenNeuro: 14 CN and one AD; five MCI excluded. High raw accuracy on this imbalanced pilot can coexist with missing every AD subject. Clinical claims remain exploratory.')
    return paths


def accuracy_curves(runs,out,clinical=False):
    plt=plotting();modes=['multitask'] if clinical else ['stage_only','multitask'];fig,axes=plt.subplots(len(modes),3,figsize=(13,4.9 if clinical else 8.1),squeeze=False,sharex=True,sharey=True)
    for i,mode in enumerate(modes):
        for j,seed in enumerate(curves.SEEDS):
            run=runs[(mode,seed)];h=run['history'];epochs=[v['epoch'] for v in h];ax=axes[i,j];prefix='aux' if clinical else 'stage'
            ax.plot(epochs,[v['train_'+prefix+'_accuracy']*100 for v in h],color='#0072B2',label='Train: augmented slices' if clinical else 'Train: augmented images')
            ax.plot(epochs,[v['val_'+prefix+'_accuracy']*100 for v in h],color='#D55E00',label='Validation: subject means' if clinical else 'Validation: images')
            if clinical:ax.plot(epochs,[v['val_aux_balanced_accuracy']*100 for v in h],color='#7A1E5D',ls=':',label='Validation balanced accuracy')
            ax.axvline(run['selected_epoch'],ls='--',color='#333333',lw=1,label='Stage-validation selected epoch')
            ax.set(title=mode.replace('_',' ')+' | '+str(seed),ylim=(0,103),xlabel='Epoch');ax.grid(alpha=.15)
            if j==0:ax.set_ylabel('Accuracy (%)')
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,.925 if clinical else .95),ncol=2 if clinical else 3,fontsize=9,frameon=False)
    fig.subplots_adjust(left=.07,right=.985,bottom=.23 if clinical else .14,top=.76 if clinical else .84,hspace=.33,wspace=.18)
    return finish(fig,out,'clinical_accuracy_all_three_seeds' if clinical else 'stage_accuracy_all_six_runs',('Clinical AD/CN' if clinical else 'Four-stage image head')+' | raw accuracy by epoch',('Training uses augmented slices and validation averages nine slices per subject; these are different units. ' if clinical else 'Every epoch is shown without smoothing; all six fixed runs remain visible. ')+'Dashed line is the frozen stage-validation-selected checkpoint. Curves do not establish patient-independent clinical generalization.')


def validation_confusion(validation,display,out):
    y,preds=None,{}
    # Construct count-equivalent vectors solely to reuse the panel renderer.
    # These are aggregate stored validation counts, never synthetic measurements.
    for name,m in validation.items():
        cm=np.asarray(m['confusion_matrix']);current=np.repeat(np.arange(4),cm.sum(1))
        if y is None:y=current
        elif not np.array_equal(y,current):raise ValueError('Validation class totals differ')
        preds[name]=np.concatenate([np.repeat(np.arange(4),row) for row in cm])
    files,_=prior_plot.confusion_panels(y,preds,display,'Validation at each frozen selected checkpoint','Stored best.pt validation counts; these development data selected the checkpoints. No additional inference. All six candidates shown; old/ensemble validation matrices unavailable.',out,'stage_validation_confusion_all_six',False)
    return files


def load_clean_replay(folder,base):
    """Validate separately documented post-hoc clean inference, never run it."""
    path=folder/'replay_metadata.json';meta=json.loads(path.read_text())
    manifest=ROOT/'training_logs/experiments/kevin_20261011/prepared/stage_manifest.csv'
    if meta['stage_manifest_sha256']!=stats.digest(manifest) or meta['class_order']!=stats.STAGES:raise ValueError('Replay manifest/class identity mismatch')
    wanted=set(base['selection']['candidates'])
    stats.require_same(meta['checkpoints'],wanted,'clean replay candidates')
    stage=stats.read_csv(manifest,['path','label','split']);data={}
    base['inputs_sha256'][str(path)]=stats.digest(path)
    for split,stage_split,count in [('train_resubstitution','train',19023),('full_new_test','test',5450)]:
        expected={r['path']:stats.STAGES[int(r['label'])] for r in stage if r['split']==stage_split}
        if len(expected)!=count:raise ValueError('Unexpected replay split size')
        ids=sorted(expected);y=np.asarray([stats.STAGES.index(expected[i]) for i in ids]);predictions={}
        for name in base['names']:
            if name not in wanted:continue
            record=meta['checkpoints'][name];frozen=base['selection']['candidates'][name]
            if any(record[k]!=frozen[k] for k in ('checkpoint_sha256','mode','seed')) or record['selected_epoch']!=frozen['epoch']:
                raise ValueError('Replay checkpoint identity mismatch')
            item=record['splits'][split];file=folder/f'{name}_{split}_predictions.csv'
            if Path(item['predictions_file']).name!=file.name or stats.digest(file)!=item['predictions_sha256'] or item['n']!=count:raise ValueError('Replay prediction digest/count mismatch')
            pred=np.asarray(stats.load_stage_predictions(file,expected,'path'));actual=stats.class_metrics(y,pred,4)
            if actual['confusion_matrix']!=item['metrics']['confusion_matrix'] or any(not np.isclose(actual[k],item['metrics'][k],atol=1e-14,rtol=0) for k in stats.METRICS):raise ValueError('Replay matrix differs from CSV')
            predictions[name]=pred;base['inputs_sha256'][str(file)]=stats.digest(file)
        data[split]={'y':y,'predictions':predictions}
    return data


def replay_confusions(data,display,out):
    files=[]
    for split,values in data.items():
        title=('Training19023 | final-checkpoint resubstitution' if split=='train_resubstitution' else 'Full new test5450 | additional post-hoc evaluation')
        note=('Clean final-checkpoint predictions on training images; not augmented epoch draws or generalization evidence. ' if split=='train_resubstitution' else 'Additional clean evaluation of the entire new test split; it does not replace the frozen1058 primary or756 sensitivity. ')
        note+='All six frozen candidates; no model selection. Image groups are not patient identities.'
        paths,_=prior_plot.confusion_panels(values['y'],values['predictions'],display,title,note,out,split+'_confusion_all_six',False);files+=paths
    return files


def diagrams(base,out):
    plt=plotting()
    from matplotlib.patches import FancyBboxPatch
    def box(ax,x,y,w,h,text,color):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.015',facecolor=color,edgecolor='#55626E',lw=1))
        ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=10,linespacing=1.5)
    def arrow(ax,start,end):ax.annotate('',xy=end,xytext=start,arrowprops={'arrowstyle':'->','lw':1.3,'color':'#445566'})
    fig,ax=plt.subplots(figsize=(13,7));ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
    box(ax,.02,.67,.28,.23,'Legacy four-stage images\nPixel/dHash grouping\nFixed train / validation / test','#EAF2F8')
    box(ax,.02,.28,.28,.23,'Clinical T1 MRI\nBesga: 32 train / 11 val / 11 test\nOpenNeuro: 20 external subjects','#E8F4ED')
    box(ax,.38,.46,.24,.27,'Train / validation only\nThree seeds per condition\nStage-only and multitask\nSelect: stage validation NLL','#FCF1E4')
    arrow(ax,(.30,.78),(.37,.66));arrow(ax,(.30,.39),(.37,.54))
    ax.text(.354,.37,'Besga\ntrain/val only',ha='center',va='top',fontsize=8,linespacing=1.25)
    box(ax,.71,.67,.27,.23,'Paired image benchmarks\nPrimary1058 / sensitivity756\nOriginal100 + three fixed readers','#EAF2F8')
    box(ax,.71,.28,.27,.23,'Clinical AD/CN endpoint\nNine-slice subject means\nMCI: prediction-only','#E8F4ED')
    arrow(ax,(.62,.66),(.70,.78));arrow(ax,(.62,.54),(.70,.39))
    ax.text(.5,.13,'Report every frozen run and both fixed supplementary ensembles\nConditional paired intervals for images; exact intervals for clinical subjects',ha='center',fontsize=10,linespacing=1.5)
    fig.subplots_adjust(left=.03,right=.97,bottom=.10,top=.9)
    paths=finish(fig,out,'dementia_study_workflow','Kevin dementia study | data and evaluation workflow','Test/external cases enter evaluation only. Original100 is reused; hash groups do not prove patient identity. Clinical cohorts are small and source-limited. All evaluations remain research-only.')
    fig,ax=plt.subplots(figsize=(13,7));ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
    box(ax,.02,.36,.21,.29,'Input\n224 × 224 RGB\nImageNet normalization\nFrozen evaluation transform','#EAF2F8')
    box(ax,.30,.36,.25,.29,'MobileNetV3-Large\nImageNet initialization\nShared feature representation','#EDF1F4');arrow(ax,(.23,.51),(.29,.51))
    box(ax,.63,.62,.34,.23,'Stage head: four classes\nMild / Moderate / Non / Very mild\nTrained in both conditions','#EAF2F8')
    box(ax,.63,.23,.34,.23,'Clinical auxiliary head: CN / AD\nSupervised only in multitask condition\nSubject score: mean of nine slices','#E8F4ED')
    arrow(ax,(.55,.55),(.62,.73));arrow(ax,(.55,.46),(.62,.34))
    ax.text(.425,.24,'Stage-only: stage supervision\nMultitask: stage + clinical supervision\nEach condition: seeds 20261011 / 12 / 13',ha='center',fontsize=10,linespacing=1.5)
    fig.subplots_adjust(left=.03,right=.97,bottom=.1,top=.9)
    paths+=finish(fig,out,'dementia_two_condition_architecture','Dementia model | two prespecified training conditions','The legacy stage head and clinical AD/CN head have different reference labels and evaluation units. Stage-only auxiliary weights are not evaluated clinically. Uniform three-seed probability ensembles are supplementary postprocessing, with no fitted weights.')
    return paths


def write_csv(path,rows):
    if not rows:return
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--results',type=Path,default=RESULTS);p.add_argument('--output',type=Path,default=HERE/'generated');p.add_argument('--clean-replay',type=Path);args=p.parse_args()
    result_root=args.results if args.results.is_absolute() else ROOT/args.results
    out=args.output if args.output.is_absolute() else ROOT/args.output
    if out.exists():raise ValueError('Output must be a new directory; preserve previous figures')
    base,primary,supplement,runs,validation,numerical=load_inputs(result_root)
    replay=load_clean_replay(args.clean_replay if args.clean_replay.is_absolute() else ROOT/args.clean_replay,base) if args.clean_replay else None
    out.mkdir(parents=True);files=[];histograms=[];scores=[];error_rows=[]
    for cohort in COHORTS:
        d=base['datasets'][cohort];names=base['names'];display=base['display']
        f,_=prior_plot.confusion_panels(d['y'],{n:d['models'][n]['pred'] for n in names},display,TITLE[cohort],NOTE[cohort]+' Both fixed ensembles are supplementary.',out,cohort+'_confusion_all_nine',False);files+=f
        files+=reliability(d,names,display,cohort,out)
        f,h=confidence(d,names,display,cohort,out);files+=f;histograms+=h
        for kind in ('roc','pr'):
            f,s=probability_curves(d,names,display,cohort,out,kind);files+=f;scores+=s
        f,rows=errors(d,names,display,cohort,out);files+=f;error_rows+=rows
    files+=reader_figures(base,primary,supplement,out)
    files+=seed_spread(primary,out)
    files+=paired_differences(base,primary,supplement,out)
    files+=clinical_figures(primary,supplement,out)
    for clinical in (False,True):
        files+=accuracy_curves(runs,out,clinical)
        files+=curves.plot_panels(runs,out,clinical,False)
    files+=validation_confusion(validation,base['display'],out)
    if replay:files+=replay_confusions(replay,base['display'],out)
    f,_=prior_plot.confusion_panels(base['datasets']['original100']['y'],base['readers'],{n:n.replace('Reader_','') for n in base['readers']},'Original100 | three fixed reader confusion matrices',NOTE['original100'],out,'original100_fixed_reader_confusion',False);files+=f
    files+=diagrams(base,out)
    write_csv(out/'error_cases.csv',error_rows);write_csv(out/'classwise_roc_ap.csv',scores)
    plot_data={'models_in_fixed_order':base['names'],'model_display':base['display'],'metric_and_calibration_values':numerical,'confidence_histograms':histograms,'saved_validation_metrics':validation,'clinical':primary['clinical'],'supplementary_clinical':supplement['clinical']}
    (out/'plot_data.json').write_text(json.dumps(plot_data,indent=2,allow_nan=False)+'\n')
    if any(stats.digest(path)!=sha for path,sha in base['inputs_sha256'].items()):raise ValueError('Source artifact changed during rendering')
    provenance={'source_script':relative(__file__),'source_script_sha256':stats.digest(__file__),'inputs_sha256':{relative(p):sha for p,sha in base['inputs_sha256'].items()},'model_inference':False,'new_fitting_or_selection':False,'frozen_sources_modified':False,'figures':[{'path':relative(f),'sha256':stats.digest(f)} for f in files],'numeric_checks':{'cohort_model_point_summaries_match_frozen_analysis':27,'clinical_model_cohort_points_and_exact_intervals':8,'fixed_ensemble_probability_files_equal_member_means':6,'saved_validation_matrices_validated':6,'posthoc_clean_replay_matrices_validated':12 if replay else 0},'limitations':list(NOTE.values())+['Clinical AD/CN is exploratory, with only one external AD subject.','Seed spread is SD, not a confidence interval; ensembles remain supplementary.','Training resubstitution and full-new-test replay are post-hoc supplementary displays, not the primary held-out analysis.' if replay else 'Training/full-new-test matrices are omitted until the separate clean replay is supplied.']}
    (out/'figure_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    write_csv(out/'FIGURE_INDEX.csv',[{'figure':f.stem,'title':figure_title(f.stem),'format':f.suffix[1:],'path':relative(f),'cohort':next((co for co in COHORTS if f.stem.startswith(co)),'see_caption'),'sha256':stats.digest(f)} for f in files])
    print(json.dumps({'status':'complete','png_figures':sum(f.suffix=='.png' for f in files),'pdf_figures':sum(f.suffix=='.pdf' for f in files),'output':relative(out),'new_inference':False}))


if __name__=='__main__':main()
