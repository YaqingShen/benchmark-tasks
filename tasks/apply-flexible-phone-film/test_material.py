import numpy as np
import unittest

from material import LENGTH, TAB_X, TAB_Y, Laminate


def sweep(mode):
    m = Laminate()
    def record(p, x, loaded=True, offset=0.):
        tab = np.array([TAB_X+2*p, TAB_Y+offset, .006])
        blade = np.array([[x,-.044,.001],[x,.044,.003 if mode=="tilted" else .001]])
        if mode=="tip20":
            blade=np.array([[x,-.032,0.],[x,.032,.020]])
        m.observe(tab,0.,True,blade,loaded)
    off=.009 if mode=="misaligned" else 0.
    record(0,0,offset=off)
    if mode=="peel_before_sweeping":
        record(LENGTH,0)
    for x in np.linspace(0,LENGTH,281):
        p=0. if mode=="intact_backing" else LENGTH if mode=="peel_before_sweeping" else x
        loaded=mode!="no_pressure" and not (mode=="miss_strip" and .05 < x < .075)
        record(p, max(0,x-.004), loaded, off)
    record(0. if mode=="intact_backing" else LENGTH,LENGTH,mode!="no_pressure",off)
    return m


class MaterialChecks(unittest.TestCase):
    def test_false_bonds(self):
        for mode in ("no_pressure", "intact_backing", "misaligned"):
            with self.subTest(mode=mode):
                self.assertEqual(sweep(mode).coverage, 0.)

    def test_progressive(self):
        m = sweep("progressive")
        self.assertEqual(m.coverage, 1.)
        self.assertTrue(m.progressive.all())

    def test_omitted_strip(self):
        m = sweep("miss_strip")
        self.assertGreater(m.coverage, 0.)
        self.assertLess(m.coverage, 1.)

    def test_only_loaded_side_bonds(self):
        for mode in ("tilted","tip20"):
            with self.subTest(mode=mode):
                m = sweep(mode)
                self.assertTrue(np.any(m.bond[:,0]))
                self.assertFalse(np.any(m.bond[:,-1]))
        self.assertLessEqual(sweep("tip20").coverage, .125)

    def test_unbonded_film_follows_tab_after_card_leaves(self):
        m=Laminate()
        blade=np.array([[0.,-.045,0.],[0.,.045,0.]])
        m.observe(np.array([TAB_X,TAB_Y,.006]),0.,True,blade,True)
        self.assertTrue(m.held_edge)
        m.observe(np.array([TAB_X+.025,TAB_Y,.008]),0.,True,blade,False)
        self.assertFalse(m.held_edge)
        self.assertEqual(m.peeled,0.)
        self.assertAlmostEqual(m.origin[0],.025)

    def test_late_sweep(self):
        m = sweep("peel_before_sweeping")
        self.assertEqual(m.coverage, 1.)
        self.assertLess(m.progressive.mean(), .25)


if __name__ == "__main__":
    unittest.main()
