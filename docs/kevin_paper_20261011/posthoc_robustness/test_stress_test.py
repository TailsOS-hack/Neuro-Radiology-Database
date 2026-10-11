"""Synthetic implementation checks; these are not model performance evidence."""
import unittest
import numpy as np
from PIL import Image
import stress_test as s


class StressChecks(unittest.TestCase):
    def setUp(self):
        a=np.tile(np.arange(224,dtype=np.uint8),(224,1))
        self.im=Image.fromarray(np.repeat(a[:,:,None],3,axis=2),'RGB')

    def test_clean_is_exact_and_does_not_mutate_source(self):
        original=np.array(self.im);result=s.perturb(self.im,s.SCENARIOS[0],'test')
        self.assertTrue(np.array_equal(original,np.array(result)))
        self.assertIsNot(result,self.im)
        for scenario in s.SCENARIOS:s.perturb(self.im,scenario,'test')
        self.assertTrue(np.array_equal(original,np.array(self.im)))

    def test_noise_reproducible_case_specific_and_grayscale(self):
        scenario=next(x for x in s.SCENARIOS if x['name']=='noise_sd0.01')
        first=np.array(s.perturb(self.im,scenario,'one'))
        np.random.seed(55);np.random.random(100)
        repeated=np.array(s.perturb(self.im,scenario,'one'))
        second=np.array(s.perturb(self.im,scenario,'two'))
        self.assertTrue(np.array_equal(first,repeated))
        self.assertFalse(np.array_equal(first,second))
        self.assertTrue(np.array_equal(first[:,:,0],first[:,:,1]))
        self.assertTrue(np.array_equal(first[:,:,1],first[:,:,2]))

    def test_all_scenarios_keep_shape_range_and_pixel_contract(self):
        for scenario in s.SCENARIOS:
            with self.subTest(scenario=scenario['name']):
                im=s.perturb(self.im,scenario,'same-case');a=np.asarray(im)
                self.assertEqual(im.mode,'RGB');self.assertEqual(im.size,(224,224))
                self.assertEqual(a.dtype,np.uint8);self.assertTrue(np.isfinite(a).all())
                self.assertGreaterEqual(int(a.min()),0);self.assertLessEqual(int(a.max()),255)
        with self.assertRaises(ValueError):s.perturb(self.im.resize((100,100)),s.SCENARIOS[0],'case')

    def test_probability_guards_reject_nan_negative_wrong_sum_and_shape(self):
        s.check_probs(np.array([[.1,.2,.3,.4]]))
        for p in (np.array([[np.nan,0,0,1]]),np.array([[-.1,.2,.4,.5]]),np.array([[.1,.1,.1,.1]]),np.array([[.5,.5]])):
            with self.assertRaises(ValueError):s.check_probs(p)

    def test_metric_alignment_and_flip_counts(self):
        y=np.arange(4);clean=np.eye(4)*.9+.025
        changed=clean.copy();changed[0]=clean[1]
        m=s.metrics(y,changed,clean)
        self.assertEqual(m['accuracy'],.75);self.assertEqual(m['balanced_accuracy'],.75)
        self.assertEqual(m['label_flip_count'],1);self.assertEqual(m['clean_correct_to_wrong'],1)
        self.assertEqual(m['clean_wrong_to_correct'],0);self.assertAlmostEqual(m['mean_total_variation'],.225)


if __name__=='__main__':unittest.main()
