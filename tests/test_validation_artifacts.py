"""Tamper with completed artifacts, re-seal, and ensure semantic checks still reject them."""
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
import numpy as np
from src.research_validation import ROOT,seal,read_csv,write_csv
from scripts.check_validation_review import check,check_explainability,check_calibration

BUNDLE=Path(os.environ.get('VALIDATION_TEST_BUNDLE',str(ROOT/'docs/review_bundle_v2/validation-v2-20260907')))


@unittest.skipUnless(BUNDLE.exists(),'Run after versioned evidence is generated')
class CompletedEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.protocol=json.loads((BUNDLE/'protocol.json').read_text())
    def tearDown(self):self.tmp.cleanup()
    def component(self,name):
        path=self.root/name;shutil.copytree(BUNDLE/name,path);return path
    def test_count_corruption_after_reseal(self):
        out=self.component('explainability');rows=read_csv(out/'perturbations.csv')
        write_csv(out/'perturbations.csv',rows[:-1]);seal(out)
        with self.assertRaises(ValueError):check_explainability(out,self.protocol)
    def test_sd_corruption_after_reseal(self):
        out=self.component('explainability');path=out/'summary.json';s=json.loads(path.read_text())
        next(iter(s['perturbations'].values()))['metrics']['top_drop']['sd']+=.01
        path.write_text(json.dumps(s));seal(out)
        with self.assertRaises(ValueError):check_explainability(out,self.protocol)
    def test_native_map_corruption_after_reseal(self):
        out=self.component('explainability');p=next((out/'maps').glob('*.npy'))
        np.save(p,np.full((7,7),np.nan));seal(out)
        with self.assertRaises(ValueError):check_explainability(out,self.protocol)
    def test_calibration_split_corruption_after_reseal(self):
        out=self.component('calibration');p=out/'temperatures.json';s=json.loads(p.read_text());s['tumor']['split']='test';p.write_text(json.dumps(s));seal(out)
        with self.assertRaises(ValueError):check_calibration(out,self.protocol)
    def test_shape_identity_corruption_after_reseal(self):
        out=self.component('explainability');rows=read_csv(out/'perturbations.csv')
        ids=json.loads(rows[0]['shape_ids']);ids[0]='flip0_rot0';rows[0]['shape_ids']=json.dumps(ids)
        write_csv(out/'perturbations.csv',rows);seal(out)
        with self.assertRaises(ValueError):check_explainability(out,self.protocol)
    def test_document_link_corruption_after_reseal(self):
        out=self.root/'bundle';shutil.copytree(BUNDLE,out)
        p=out/'index.html';p.write_text(p.read_text().replace('calibration/reliability.png','calibration/missing.png'));seal(out)
        with self.assertRaises(ValueError):check(out)
    def test_manuscript_number_corruption_after_reseal(self):
        out=self.root/'bundle';shutil.copytree(BUNDLE,out)
        p=out/'RESULTS.md';p.write_text(p.read_text().replace('128 images','999 images'));seal(out)
        with self.assertRaises(ValueError):check(out)


if __name__=='__main__':unittest.main()
