"""Synthetic clinical report validation; no MRI or real model outcomes."""
import copy,json,tempfile,unittest
from pathlib import Path
import numpy as np
import clinical_report as r


def fixture():
    patients=[]
    specs=[('harvard_fresh_external',9,10,10),('besga_reused_test',7,4,0),('openneuro_reused_external',1,14,5)]
    for cohort,nad,ncn,nmci in specs:
        for diagnosis,n in [('AD',nad),('CN',ncn),('MCI',nmci)]:
            for i in range(n):patients.append(dict(group_key=f'{cohort}_{diagnosis}_{i:02d}',evaluation_cohort=cohort,label={'AD':1,'CN':0,'MCI':-1}[diagnosis],diagnosis=diagnosis))
    probabilities={};predictions=[];metrics=[];rng=np.random.default_rng(932)
    for scenario in r.c.SCENARIOS:
        for name in list(r.c.OLD)+r.c.NEW:
            values=[]
            for row in patients:
                center=.7 if row['label']==1 else (.3 if row['label']==0 else .5)
                q=float(np.clip(center+rng.normal(0,.2),.01,.99));values.append([1-q,q])
            probabilities[name,scenario]=np.array(values)
        for ensemble,names in [('round1_clinical_ensemble',list(r.c.OLD)),('new_clinical_ensemble',r.c.NEW)]:probabilities[ensemble,scenario]=np.mean([probabilities[n,scenario] for n in names],axis=0)
    for name in r.c.ORDER:
        for scenario in r.c.SCENARIOS:
            pp=probabilities[name,scenario]
            for row,p in zip(patients,pp):predictions.append(dict(group_key=row['group_key'],cohort=row['evaluation_cohort'],label=row['label'],diagnosis=row['diagnosis'],model=name,scenario=scenario,n_slices=9,probabilities=p.tolist(),predicted_ad=bool(p[1]>=.5),binary_evaluable=row['label']>=0))
            for cohort in r.TITLES:
                idx=[i for i,row in enumerate(patients) if row['evaluation_cohort']==cohort and row['label']>=0];y=np.array([patients[i]['label'] for i in idx]);m=r.s.metrics(y,pp[idx]);clean=probabilities[name,'clean'][idx];m.update(mean_probability_total_variation=float(np.abs(pp[idx]-clean).sum(1).mean()/2),label_flip_fraction=float(np.mean((pp[idx,1]>=.5)!=(clean[:,1]>=.5))))
                metrics.append(dict(model=name,scenario=scenario,cohort=cohort,**m))
    return patients,predictions,metrics


class ClinicalReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.patients,cls.predictions,cls.metrics=fixture()

    def test_full_grid_recomputation_and_actual_denominators(self):
        data,metrics=r.validate_predictions(self.predictions[::-1],self.patients[::-1],self.metrics[::-1]);self.assertEqual(len(metrics),168)
        self.assertEqual([len(data[c]['y']) for c in r.TITLES],[19,11,15]);self.assertEqual(sum(row['label']==-1 for row in self.patients),15)
        self.assertEqual(set(data['harvard_fresh_external']['models']),set(r.c.ORDER))

    def test_duplicate_missing_and_metadata_mismatch(self):
        for bad in [self.predictions[:-1],self.predictions+[self.predictions[0]]]:
            with self.assertRaises(ValueError):r.validate_predictions(bad,self.patients,self.metrics)
        for field,value in [('label',-1),('diagnosis','MCI'),('cohort','wrong'),('n_slices',8),('binary_evaluable',False),('probabilities',[.2,.2]),('predicted_ad','True')]:
            bad=copy.deepcopy(self.predictions);bad[0][field]=value
            with self.assertRaises(ValueError):r.validate_predictions(bad,self.patients,self.metrics)

    def test_ensemble_and_metric_corruption_rejected(self):
        bad=copy.deepcopy(self.predictions);row=next(z for z in bad if z['model']=='new_clinical_ensemble');row['probabilities']=[.1,.9];row['predicted_ad']=True
        with self.assertRaises(ValueError):r.validate_predictions(bad,self.patients,self.metrics)
        bad=copy.deepcopy(self.metrics);bad[0]['accuracy']=.12345
        with self.assertRaises(ValueError):r.validate_predictions(self.predictions,self.patients,bad)

    def test_exact_intervals_and_mci_never_scored(self):
        self.assertAlmostEqual(r.s.exact_interval(19,19)[0],.82353308819)
        with self.assertRaises(ValueError):r.s.metrics([-1],[[.5,.5]])
        self.assertEqual(len([z for z in self.predictions if not z['binary_evaluable']]),840)

    def test_paired_bootstrap_uses_patient_not_seed_count(self):
        y=np.array([0]*10+[1]*9);p=np.column_stack([1-(.2+.6*y),.2+.6*y]);result=r.s.paired_comparison(y,p,p,'clinical_report_identical_synthetic',draws=10000)
        self.assertEqual(result['bootstrap_draws'],10000);self.assertEqual(result['new_only_correct'],0)
        self.assertTrue(all(z['point']==0 and z['conditional95']==[0.,0.] for z in result['differences'].values()))

    def test_safe_payload_and_output_preconditions(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):r.c.safe(Path(d),'../escape')
            with self.assertRaises(ValueError):r.c.safe(Path(d),'/absolute')

if __name__=='__main__':unittest.main(verbosity=2)
