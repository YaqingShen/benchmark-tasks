"""Synthetic negative controls for the same cut detector used by Isaac."""
import unittest
import numpy as np
from cut_checks import BatchPass, both_tools_held

class CrossingTests(unittest.TestCase):
    def sweep(self, detector=None, count=8, *, moving=True, span=0., height=0., covered=True,
              engaged=True, last=.1, ids=None):
        detector=detector or BatchPass()
        for y in np.linspace(-.05,last,76):
            p=np.array([0.,y if moving else 0.,0.])
            centres=np.array([[span,-y+i*.004,height] for i in range(count)])
            mask=[covered and (ids is None or i in ids) for i in range(count)]
            detector.observe(p,np.eye(3),centres,mask,engaged)
        return detector

    def test_full_batch(self):
        b=self.sweep(); self.assertEqual(len(b.cut_ids),8); self.assertTrue(b.batch_complete)

    def test_wrong_height_uncovered_and_unheld(self):
        for kwargs in ({'height':.004},{'covered':False},{'engaged':False}):
            with self.subTest(kwargs=kwargs): self.assertFalse(self.sweep(**kwargs).cut_ids)

    def test_face_must_exit_and_fit_on_edge(self):
        self.assertFalse(self.sweep(last=0.,count=1).cut_ids)
        self.assertFalse(self.sweep(span=.083,count=1).cut_ids)

    def test_stationary_knife(self):
        self.assertFalse(self.sweep(moving=False).cut_ids)

    def test_separate_strokes(self):
        b=self.sweep(ids=range(4))
        self.sweep(b,ids=range(4,8))
        self.assertEqual(len(b.cut_ids),8)
        self.assertFalse(b.batch_complete)
        self.assertEqual([len(ids) for ids in b.passes],[4,4])

    def test_hand_exchange(self):
        a=np.zeros(3); b=np.array([.18,0.,0.])
        self.assertTrue(both_tools_held([a,b],a,b))
        self.assertTrue(both_tools_held([b,a],a,b))
        self.assertFalse(both_tools_held([a,a],a,b))

    def test_no_double_count(self):
        b=self.sweep(count=1);self.sweep(b,count=1)
        self.assertEqual(b.cut_ids,{0});self.assertFalse(b.batch_complete)

if __name__=='__main__': unittest.main()
