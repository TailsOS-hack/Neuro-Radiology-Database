#!/usr/bin/env python3
"""Synthetic-only tests; never opens held-out pixels or real model outcomes."""
import copy,json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import round2_pipeline as p
import report_round2 as r
import round2_attribution as a


def validation(loss=.1,accuracy=1.,stress_loss=1.):
    clean=dict(loss=loss,accuracy=accuracy,balanced_accuracy=accuracy)
    st={name:dict(loss=stress_loss,accuracy=accuracy) for name,_,_ in p.trainer.SCENARIOS[1:]}
    return dict(clean=clean,stress=st)


def histories():
    vals=[validation(),validation(.09,1.,.8),validation(.08,.9,.1),validation(.09,1.,.7999)]
    history=[];best=None
    for ep,v in enumerate(vals):
        sel=p.trainer.selection(v['clean'],v['stress'],vals[0]['clean']);v['selection']=sel
        improved=best is None or (sel['eligible'] and sel['score']<best-p.trainer.DEFAULTS['min_delta'])
        if improved:best=sel['score']
        history.append(dict(epoch=ep,score=sel['score'],eligible=sel['eligible'],selected=improved,clean_nll=v['clean']['loss'],clean_accuracy=v['clean']['accuracy'],stress_nll=sel['mean_stress_family_nll'],stress_accuracy=v['clean']['accuracy'],**(dict(train_raw_nll=.1/(ep+1),train_accuracy=.98) if ep else {})))
    return history,dict(enumerate(vals))


def predictions():
    expected=[dict(path=f'case_{i}.png',label=str(i%4),group_id=f'g{i//2}') for i in range(12)]
    rows=[]
    for row in expected:
        prob=np.ones(4)*.05;prob[int(row['label'])]=.85
        rows.append(dict(path=row['path'],TrueCategory=p.stress.STAGES[int(row['label'])],ModelPrediction=p.stress.STAGES[int(prob.argmax())],group_id=row['group_id'],**{'P_'+cl:str(v) for cl,v in zip(p.stress.STAGES,prob)}))
    return expected,rows


class PipelineTests(unittest.TestCase):
    def test_validation_guard_and_min_delta_replay(self):
        hist,vals=histories();self.assertEqual(p.replay_selection(hist,vals,vals[0]['clean'],p.trainer.DEFAULTS)[0],1)
        for field,value in [('selected',True),('eligible',True),('score',0.)]:
            bad=copy.deepcopy(hist);bad[2][field]=value
            with self.assertRaises(ValueError):p.replay_selection(bad,vals,vals[0]['clean'],p.trainer.DEFAULTS)
        bad=copy.deepcopy(hist);bad[3]['epoch']=4
        with self.assertRaises(ValueError):p.replay_selection(bad,vals,vals[0]['clean'],p.trainer.DEFAULTS)

    def test_convergence_requires_pinned_validation_only_review(self):
        status={'runs':{'20261011':{'needs_validation_convergence_review':True}}}
        for review in [None,{},dict(status='approved_after_validation_only_review',suite_status_sha256='wrong')]:
            with self.assertRaises(ValueError):p.convergence_gate(status,{},review,'statushash','protocolhash')
        review=dict(status='approved_after_validation_only_review',suite_status_sha256='statushash',training_protocol_sha256='protocolhash',heldout_outcomes_used=False,rationale='Synthetic validation plateau review.',reviewed_seeds=p.SEEDS)
        p.convergence_gate(status,{},review,'statushash','protocolhash')
        review['heldout_outcomes_used']=True
        with self.assertRaises(ValueError):p.convergence_gate(status,{},review,'statushash','protocolhash')
        p.convergence_gate({'runs':{}},{},None,'statushash','protocolhash')

    def test_complete_alignment_labels_and_probabilities(self):
        expected,rows=predictions();prob=r.aligned_probabilities(rows[::-1],expected);self.assertEqual(prob.shape,(12,4))
        for bad in [rows[:-1],rows+[rows[0]],rows+[dict(rows[0],path='extra')]]:
            with self.assertRaises(ValueError):r.aligned_probabilities(bad,expected)
        for field,value in [('TrueCategory','wrong'),('group_id','wrong'),('ModelPrediction','wrong'),('P_MildDemented','nan'),('P_MildDemented','1.4')]:
            bad=copy.deepcopy(rows);bad[0][field]=value
            with self.assertRaises(ValueError):r.aligned_probabilities(bad,expected)

    def test_export_refuses_length_and_duplicate(self):
        expected,_=predictions();prob=np.ones((12,4))*.25
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):p.export_predictions(Path(d),'m','c',expected,prob[:-1])
            with self.assertRaises(ValueError):p.export_predictions(Path(d),'m','c',expected+[expected[0]],np.ones((13,4))*.25)

    def test_metric_calibration_known_values_and_paired_units(self):
        expected,rows=predictions();prob=r.aligned_probabilities(rows,expected);y=np.array([int(z['label']) for z in expected]);cal=r.plots.calibration(y,prob)
        self.assertAlmostEqual(cal['multiclass_brier_sum'],.03);self.assertAlmostEqual(cal['top_label_ece_10_equal_width_bins'],.15)
        out=r.stats.bootstrap_analysis(y,{'a':y,'b':np.roll(y,1)},[z['group_id'] for z in expected],'synthetic_round2',[('a','b')])
        self.assertEqual(out['n_resampling_units'],6);self.assertEqual(out['paired_comparisons']['a_minus_b']['difference']['accuracy']['estimate'],1.)
        self.assertEqual(out['models']['a']['bootstrap']['accuracy']['requested_resamples'],2000)

    def test_suite_gate_and_freeze_before_addon(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);suite=root/'suite';suite.mkdir();hist,vals=histories();protocol={'defaults':p.trainer.DEFAULTS,'source_hashes':{}};p.dump(suite/'protocol.json',protocol);psha=p.sha(suite/'protocol.json');runs={};selections={};checkpoints={}
            for seed in p.SEEDS:
                folder=suite/f'stage_only_seed{seed}';folder.mkdir();cp=folder/'best.pt';cp.write_text('synthetic '+str(seed));p.dump(folder/'protocol.json',protocol)
                cfg=dict(round=2,mode='stage_only',seed=seed,smoke=False,initial_checkpoint_sha256=p.trainer.INIT_SHAS[seed],protocol_sha256=psha,stage_manifest_sha256=p.trainer.STAGE_SHA,test_used_for_selection=False)
                sel=dict(seed=seed,selected=dict(epoch=1,**vals[1]),epoch0_fallback=False,initial_checkpoint_sha256=p.trainer.INIT_SHAS[seed],protocol_sha256=psha,checkpoint_sha256=p.sha(cp),smoke_only=False)
                p.dump(folder/'config.json',cfg);p.dump(folder/'selection.json',sel);p.dump(folder/'history.json',hist)
                for ep,v in vals.items():p.dump(folder/f'epoch_{ep:03d}_validation.json',v)
                run=dict(status='complete',smoke_only=False,protocol_sha256=psha,best_checkpoint_sha256=p.sha(cp),selection_sha256=p.sha(folder/'selection.json'),selected_epoch=1,completed_epochs=3,needs_validation_convergence_review=False)
                p.dump(folder/'run_status.json',run);runs[str(seed)]=run;selections[str(seed)]=sel
                checkpoints[str(cp)]=dict(round=2,config=cfg,epoch=1,selection=vals[1]['selection'],best_selection_score=vals[1]['selection']['score'],stage_validation=vals[1]['clean'],stress_validation=vals[1]['stress'],history=hist[:2])
            p.dump(suite/'validation_selection_before_any_round2_test.json',dict(complete=True,selection_used_only_fixed_validation=True,selections=selections,protocol_sha256=psha))
            status=dict(status='complete',completed_seeds=p.SEEDS,expected_seeds=p.SEEDS,runs=runs,research_only=True,protocol_sha256=psha,selection_sha256=p.sha(suite/'validation_selection_before_any_round2_test.json'));p.dump(suite/'suite_status.json',status)
            with patch.object(p.e,'load',side_effect=lambda cp,device:(None,None,checkpoints[str(cp)])):
                selected,representative,_=p.verify_suite(root,suite);self.assertEqual(representative,p.NEW[0]);self.assertEqual(len(selected),3)
                # No heldout inputs or old baseline exist. Only the required initial c3 is present.
                c3=root/'c3.pt';c3.write_text('synthetic init')
                with patch.dict(p.trainer.INIT_SHAS,{20261012:p.sha(c3)}):
                    # verify_suite already checked its own real initialization pins above.
                    with patch.object(p,'verify_suite',return_value=(selected,representative,psha)):
                        p.freeze(SimpleNamespace(data_root=root,suite=suite,round1_c3='c3.pt',convergence_review=None))
                frozen=p.read(suite/'round2_evaluation_freeze.json');self.assertEqual(frozen['input_pins'],p.INPUT_PINS);self.assertEqual(frozen['suite_status_sha256'],p.sha(suite/'suite_status.json'))
                with self.assertRaises(ValueError):p.freeze(SimpleNamespace(data_root=root,suite=suite,round1_c3='c3.pt',convergence_review=None))
            status['status']='incomplete_preserve_diagnostics';p.dump(suite/'suite_status.json',status)
            with self.assertRaises(ValueError):p.verify_suite(root,suite)

    def test_native_panel_does_not_change_arrays(self):
        maps={n:np.linspace(0,1,49).reshape(7,7) for n in ['old_baseline','round1_c3','round2_representative']};before={k:v.copy() for k,v in maps.items()}
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'panel.png';a.render_panel(path,dict(case_id='SYNTHETIC',label='MildDemented',role='synthetic QA'),a.Image.new('RGB',(224,224)),maps,{k:'MildDemented' for k in maps});
            with a.Image.open(path) as image:self.assertEqual(image.size,(1680,328))
        for k in maps:np.testing.assert_array_equal(maps[k],before[k])

if __name__=='__main__':unittest.main(verbosity=2)
