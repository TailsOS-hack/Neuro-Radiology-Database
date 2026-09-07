"""Meaningful offline contract checks; no MRI data, checkpoints, training or network."""
import json
import shutil
import tempfile
import unittest
from pathlib import Path
import numpy as np
from PIL import Image
from src.research_validation import (ROOT,fit_temperature,softmax,hierarchical,seal,verify_seal,publish,metrics,write_json)
from src.explainability_validation import top_mask,shape_controls,random_controls,map_similarity
from src.external_validation import validate_cohort,cohort_metrics


class ExplainabilityTests(unittest.TestCase):
    def test_control_area_geometry_and_determinism(self):
        cam=np.arange(49).reshape(7,7)
        for fraction in [.1,.2,.3]:
            mask=top_mask(cam,fraction)
            self.assertEqual(mask.sum(),int(49*fraction))
            first=shape_controls(mask,42);second=shape_controls(mask,42)
            self.assertEqual([k for k,_ in first],[k for k,_ in second])
            candidates=[np.rot90(np.fliplr(mask) if flip else mask,k) for flip in [False,True] for k in range(4)]
            for _,control in first:
                self.assertEqual(control.sum(),mask.sum())
                self.assertFalse(np.array_equal(control,mask))
                self.assertTrue(any(np.array_equal(control,c) for c in candidates))
            for a,b in zip(random_controls(mask,42),random_controls(mask,42)):
                np.testing.assert_array_equal(a,b);self.assertEqual(a.sum(),mask.sum())
    def test_symmetric_masks_report_no_controls(self):
        self.assertEqual(shape_controls(np.ones((7,7),dtype=bool),42),[])
    def test_similarity_and_flat_maps(self):
        x=np.arange(49,dtype=float).reshape(7,7)
        rho,iou=map_similarity(x,x)
        self.assertAlmostEqual(rho,1);self.assertEqual(iou,1)
        self.assertEqual(map_similarity(x,np.zeros_like(x)),(None,None))
        self.assertLess(map_similarity(x,-x)[0],0)


class CalibrationTests(unittest.TestCase):
    def setUp(self):
        self.z=np.array([[5.,-2.],[4.,-1.],[-2.,4.],[-3.,6.]])
        self.y=np.array([0,1,1,0])
    def test_validation_only_and_improvement(self):
        with self.assertRaises(ValueError):fit_temperature(self.z,self.y,split='test',provenance={})
        fit=fit_temperature(self.z,self.y,split='val',provenance={})
        self.assertGreaterEqual(fit['temperature'],.05);self.assertLessEqual(fit['temperature'],20)
        self.assertLessEqual(fit['fitted_val_nll'],fit['original_val_nll'])
        np.testing.assert_array_equal(softmax(self.z).argmax(1),softmax(self.z,fit['temperature']).argmax(1))
    def test_identity_and_fallback(self):
        np.testing.assert_allclose(softmax(self.z,1),softmax(self.z))
        fit=fit_temperature(np.zeros((4,2)),self.y,split='val',provenance={})
        self.assertEqual(fit['temperature'],1.)
        with self.assertRaises(ValueError):softmax(self.z,0)
        with self.assertRaises(ValueError):softmax(np.array([[np.nan,1]]))
    def test_hierarchy_routes_and_probability_argmax_are_distinct(self):
        router=np.array([[.51,.49]])
        tumor=np.array([[.26,.25,.25,.24]]);dementia=np.array([[.97,.01,.01,.01]])
        p,routed=hierarchical(router,tumor,dementia)
        np.testing.assert_allclose(p.sum(1),1)
        self.assertEqual(routed[0],0);self.assertEqual(p.argmax(1)[0],4)
        self.assertNotEqual(metrics(p,[4])['accuracy'],metrics(p,[4],routed)['accuracy'])


class ExternalTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        fixture=ROOT/'tests/fixtures/external'
        shutil.copytree(fixture,self.root/'fixture');self.f=self.root/'fixture'
        self.mapping=json.loads((self.f/'label_mapping.json').read_text())
        self.classes=['MildDemented','ModerateDemented','NonDemented','VeryMildDemented']
    def tearDown(self):self.temp.cleanup()
    def validate(self):
        return validate_cohort(self.f/'manifest.csv',self.mapping,self.classes,[self.f/'internal.png'])
    def test_fixture_and_missing_optional_metadata(self):
        rows,_,warnings=self.validate();self.assertEqual(len(rows),6);self.assertEqual(len(warnings),5)
    def test_unknown_label_and_patient_rejected(self):
        p=self.f/'manifest.csv';text=p.read_text();p.write_text(text.replace(',mild,',',unknown,'))
        with self.assertRaises(ValueError):self.validate()
        p.write_text(text.replace('synthetic_patient_0',''))
        with self.assertRaises(ValueError):self.validate()
    def test_internal_duplicate_and_unreadable_rejected(self):
        shutil.copyfile(self.f/'internal.png',self.f/'synthetic_0.png')
        with self.assertRaises(ValueError):self.validate()
        (self.f/'synthetic_0.png').write_bytes(b'invalid image')
        with self.assertRaises(ValueError):self.validate()
    def test_pixel_identical_reencoding_rejected(self):
        with Image.open(self.f/'internal.png') as im:im.save(self.f/'synthetic_0.png',compress_level=0)
        with self.assertRaises(ValueError):self.validate()
    def test_cluster_bootstrap_and_missing_class(self):
        p=np.array([[.9,.1,0,0],[.4,.6,0,0],[.1,.9,0,0]])
        keys=[('cohort','patient_a'),('cohort','patient_a'),('cohort','patient_b')]
        a=cohort_metrics(p,[0,0,1],keys,self.classes,repetitions=30)
        b=cohort_metrics(p,[0,0,1],keys,self.classes,repetitions=30)
        self.assertEqual(a,b);self.assertEqual(a['patients'],2)
        self.assertIsNone(a['per_class']['NonDemented']['recall'])
        c=cohort_metrics(p,[0,0,1],[('cohort','a')]*3,self.classes,repetitions=30)
        self.assertEqual(c['cluster_intervals']['status'],'unavailable_fewer_than_two_patients')


class ArtifactTests(unittest.TestCase):
    def test_corruption_and_incomplete_staging(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);stage=root/'staging';stage.mkdir();dest=root/'final'
            (stage/'metrics.json').write_text('{"n": 256}')
            with self.assertRaises(FileNotFoundError):publish(stage,dest)
            self.assertFalse(dest.exists())
            seal(stage);verify_seal(stage)
            (stage/'metrics.json').write_text('{"n": 255}')
            with self.assertRaises(ValueError):verify_seal(stage)
            seal(stage);publish(stage,dest)
            self.assertTrue(dest.exists());self.assertFalse(stage.exists())
    def test_each_artifact_type_is_hash_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for name in ['counts.json','map.npy','figures.html','temperatures.json']:
                (root/name).write_bytes(b'original')
            seal(root)
            for name in ['counts.json','map.npy','figures.html','temperatures.json']:
                (root/name).write_bytes(b'corrupt')
                with self.assertRaises(ValueError):verify_seal(root)
                (root/name).write_bytes(b'original')
            verify_seal(root)


if __name__=='__main__':unittest.main()
