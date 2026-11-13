import unittest
from pipette_correction import MassPulseControl

class PulseTests(unittest.TestCase):
    def test_waits_for_stable_reading(self):
        c=MassPulseControl('remove',.016)
        self.assertEqual(c.update(4.,{'net_g':10.5,'stable':False}),.016/.0939)
        self.assertEqual(c.pulses,0)

    def test_opposite_gap_changes_allow_settling(self):
        for mode,mass,gap in [('add',9.5,.019),('remove',10.5,.016)]:
            c=MassPulseControl(mode,gap)
            c.update(0.,{'net_g':mass,'stable':True})
            if mode=='add':self.assertLess(c.gap,gap)
            else:self.assertGreater(c.gap,gap)
            old=c.gap;c.update(.5,{'net_g':mass,'stable':True});self.assertEqual(c.gap,old)

    def test_pulses_shrink_near_target(self):
        c=MassPulseControl('remove',.016)
        c.update(0.,{'net_g':10.5,'stable':True});first=c.gap-.016
        previous=c.gap;c.update(2.,{'net_g':10.01,'stable':True})
        self.assertLess(c.gap-previous,first)

    def test_stops_only_at_settled_target(self):
        c=MassPulseControl('add',.019)
        c.update(0.,{'net_g':10.,'stable':False});self.assertFalse(c.stopped)
        c.update(1.,{'net_g':10.,'stable':True});self.assertTrue(c.stopped)

if __name__=='__main__':unittest.main()
