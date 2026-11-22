import unittest
from tear_response import TensileBond,configure


class TearResponseTests(unittest.TestCase):
    def test_configure_before_isaac_starts(self):
        reset,step=[],[]
        configure({'BREAK_BUDGET':24.,'on_reset':reset.append,'on_step':step.append})
        self.assertEqual((len(reset),len(step)),(1,1))
    def test_inward_force_and_isolated_impacts_do_not_peel(self):
        b=TensileBond(24.)
        self.assertFalse(b.observe([-90.,-90.,80.,0.,30.,12.]))
    def test_two_adjacent_substeps_can_cross_control_windows(self):
        b=TensileBond(24.)
        self.assertFalse(b.observe([0.,24.]))
        self.assertTrue(b.observe([25.]))
        self.assertFalse(b.observe([100.,100.]))
        b.reset()
        self.assertFalse(b.observe([23.99,23.99]))
