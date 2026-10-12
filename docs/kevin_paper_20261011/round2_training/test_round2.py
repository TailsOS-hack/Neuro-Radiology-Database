#!/usr/bin/env python3
"""Local correctness checks; no held-out images and no optimization runs."""
import tempfile, unittest
from pathlib import Path
import round2 as r
r.dependencies()
np=r.np

class Round2Tests(unittest.TestCase):
    def example(self):
        x=np.zeros((224,224,3),np.uint8)
        x[25:200,35:195]=120;x[65:100,80:120]=0
        x[0:10,0:10]=220 # Conservative support must preserve bright corner text too.
        return x
    def test_support_holes_dilation_and_corner_protection(self):
        x=self.example();m=r.support_mask(x)
        self.assertTrue(m[80,90]);self.assertTrue(m[20,35]);self.assertTrue(m[0,0]);self.assertFalse(m[223,223])
    def test_every_background_edit_preserves_protected_rgb_exactly(self):
        x=self.example();original=x.copy();m=r.support_mask(x)
        for edge in [False,True]:
            for pattern in [False,True]:
                y=r.background_edit(x,m,np.random.default_rng(17),edge,.12,pattern)
                self.assertTrue(np.array_equal(y[m],x[m]));self.assertTrue(np.array_equal(x,original))
                self.assertEqual(y.shape,x.shape)
                if edge:self.assertTrue(np.array_equal(y[~r.edge_region()],x[~r.edge_region()]))
    def test_clean_exact_and_stress_fixed_across_models_epochs(self):
        x=self.example();m=r.support_mask(x)
        self.assertTrue(np.array_equal(r.transform(x,m,'clean','a'),x.astype(np.float32)/255))
        for name,_,_ in r.SCENARIOS:
            a=r.transform(x,m,name,'a',20261011,1);b=r.transform(x,m,name,'a',20261013,99)
            self.assertTrue(np.array_equal(a,b));self.assertEqual(a.shape,(224,224,3))
            self.assertTrue(np.isfinite(a).all());self.assertGreaterEqual(a.min(),0);self.assertLessEqual(a.max(),1)
        self.assertFalse(np.array_equal(r.transform(x,m,'noise_001','a'),r.transform(x,m,'noise_001','b')))
    def test_train_rng_changes_by_seed_epoch_not_call_order(self):
        x=self.example();m=r.support_mask(x)
        a=r.transform(x,m,'train','a',20261011,1)
        self.assertTrue(np.array_equal(a,r.transform(x,m,'train','a',20261011,1)))
        self.assertFalse(np.array_equal(a,r.transform(x,m,'train','a',20261011,2)))
        self.assertFalse(np.array_equal(a,r.transform(x,m,'train','a',20261012,1)))
    def test_clean_guard_and_family_weighting(self):
        base=dict(loss=.04,accuracy=.999,balanced_accuracy=.998)
        st={n:dict(loss=.1*(i+1)) for i,(n,_,_) in enumerate(r.SCENARIOS[1:])}
        result=r.selection(base,st,base)
        self.assertTrue(result['eligible']);self.assertAlmostEqual(result['score'],.5*.04+.5*.45)
        for field,value in [('accuracy',.996),('balanced_accuracy',.995),('loss',.051)]:
            self.assertFalse(r.selection(dict(base,**{field:value}),st,base)['eligible'])
    def test_epoch_sampler_full_pass_no_replacement(self):
        s=r.EpochSampler(113,20261011);s.epoch=3
        a=list(s);self.assertEqual(sorted(i for i,e in a),list(range(113)))
        self.assertEqual({e for i,e in a},{3});self.assertEqual(a,list(s))
        s.epoch=4;self.assertNotEqual([i for i,e in a],[i for i,e in s])
    def test_stress_subset_validation_only_class_balanced_group_unique(self):
        rows=[dict(path=f'{l}/{i}',label=str(l),group_id=f'{l}/{i//2}',split='val') for l in range(4) for i in range(12)]
        selected=r.select_stress(rows,4)
        self.assertEqual(len(selected),16);self.assertEqual(len({x['group_id'] for x in selected}),16)
        self.assertEqual(selected,r.select_stress(list(reversed(rows)),4))
    def test_timeout_and_near_empty_support(self):
        self.assertTrue(r.support_mask(np.zeros((224,224,3),np.uint8)).all())
        with self.assertRaises(r.Deadline):r.check_deadline(0)
    def test_resize_matches_frozen_eval_transform(self):
        x=self.example()
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'x.png';r.Image.fromarray(x).resize((270,190)).save(path)
            actual=r.resize_rgb(path).astype(np.float32)/255
            with r.Image.open(path) as im:
                expected=r.c.transforms.Compose([r.c.transforms.Resize((224,224),antialias=True),r.c.transforms.ToTensor()])(im.convert('RGB')).numpy().transpose(1,2,0)
            self.assertTrue(np.array_equal(actual,expected))
if __name__=='__main__':unittest.main()
