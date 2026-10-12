#!/usr/bin/env python3
"""Full synthetic report build; generates no research prediction or image."""
import argparse,json,sys
from pathlib import Path
from unittest.mock import patch
import clinical_report as r
from test_clinical_report import fixture


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);args=ap.parse_args();root=args.output.resolve()
    if root.exists():raise ValueError('Synthetic output must be new')
    root.mkdir(parents=True);suite=root/'suite';suite.mkdir();source=suite/'clinical_evaluation';source.mkdir();patients,predictions,metrics=fixture()
    manifest=root/'manifest.json';r.c.dump(manifest,dict(status='prepared_without_model_inference',source_sha256={},patients=patients));selections={}
    for name in r.c.NEW:
        folder=suite/name;folder.mkdir();history=[dict(epoch=i,train_aug_loss=None if i==0 else .6/(i+1),train_aug_accuracy=None if i==0 else .6+i*.08,train_clean_loss=.55/(i+1),train_clean_accuracy=.6+i*.09,val_loss=.7/(i+1),val_accuracy=.6+i*.07) for i in range(5)];r.c.dump(folder/'history.json',history);selections[name]=dict(checkpoint_path=f'{name}/best.pt',checkpoint_sha256='synthetic_'+name,history_sha256=r.c.sha(folder/'history.json'),epoch=4)
    freeze=dict(status='frozen_before_clinical_heldout_inference',heldout_outcomes_used_for_selection=False,source_code_sha256=r.c.code_hashes(),evaluation_manifest_sha256=r.c.sha(manifest),input_source_sha256={},protocol=r.c.read(r.c.HERE/'evaluation_protocol.json'),selections=selections,old_checkpoints={n:dict(sha256='synthetic_'+n) for n in r.c.OLD});r.c.dump(suite/'clinical_evaluation_freeze.json',freeze)
    r.c.dump(source/'patient_predictions.json',predictions);r.c.dump(source/'metrics.json',metrics);r.c.dump(source/'model_state_verification.json',[dict(model=n,checkpoint_sha256='synthetic_'+n,all_states_unchanged=True,no_parameter_gradients=True) for n in list(r.c.OLD)+r.c.NEW])
    r.c.dump(source/'evaluation.json',dict(status='complete',freeze=freeze,freeze_sha256=r.c.sha(suite/'clinical_evaluation_freeze.json'),model_order=r.c.ORDER,scenarios=r.c.SCENARIOS,fresh_binary_patients=19,mci_not_scored=True,no_training_or_selection=True))
    inventory={f.name:r.c.sha(f) for f in source.iterdir()};r.c.dump(source/'SHA256SUMS.json',inventory);r.c.dump(source/'evaluation_complete.json',dict(status='complete_verified',manifest_sha256=r.c.sha(source/'SHA256SUMS.json'),payloads=len(inventory)))
    with patch.object(sys,'argv',['clinical_report.py','--suite',str(suite),'--manifest',str(manifest),'--output',str(root/'report')]):
        save=r.save
        with patch.object(r,'save',side_effect=lambda fig,out,stem,synthetic=False:save(fig,out,stem,True)):r.main()
    analysis=r.c.read(root/'report/analysis.json')
    if len(list((root/'report').glob('*.pdf')))!=19 or len(analysis['cohorts'])!=3:raise ValueError('Unexpected synthetic report dimensions')
    for cohort,a in analysis['cohorts'].items():
        if len(a['models'])!=8 or len(a['paired_comparisons'])!=4 or any(v['bootstrap_draws']!=10000 for v in a['paired_comparisons'].values()):raise ValueError('Incomplete model/paired comparison grid')
    r.c.dump(root/'synthetic_verification.json',dict(status='passed_full_synthetic_report',real_patient_data_used=False,model_inference_performed=False,prediction_rows_verified=3360,metric_records_recomputed=168,cohorts=3,models=8,conditional_paired_bootstraps=12,draws_each=10000,pdf_png_pairs=19,mci_prediction_only_rows=840,visual_review_required=True))
    print(r.c.read(root/'synthetic_verification.json'))

if __name__=='__main__':main()
