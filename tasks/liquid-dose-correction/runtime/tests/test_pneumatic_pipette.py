import unittest
import numpy as np
from sim import Formulation,CONFIG
from geometry_fluid import GeometryFluid
from pneumatic_pipette import PneumaticPipette,P_ATM


def pose(x=0,z=0):
    m=np.eye(4);m[:3,3]=[x,0,z];return m


class PneumaticTests(unittest.TestCase):
    def setUp(self):
        self.e=Formulation({**CONFIG,'cylinders':['cylinder_A']});self.f=GeometryFluid(self.e)
        self.p=PneumaticPipette(self.f,'pipette_A')
        self.e._receive('cylinder_A',self.e.vessels['bottle_A'].liquid.take(12.))

    def aspirate(self):
        self.p.step(pose(x=.2,z=.3),1.,{'cylinder_A':pose()},.01)
        for c in np.linspace(1,0,201):
            self.p.step(pose(z=.02),float(c),{'cylinder_A':pose()},.01)
        for _ in range(100):self.p.step(pose(z=.02),0.,{'cylinder_A':pose()},.01)

    def test_dry_compression_vents_without_liquid_creation(self):
        for c in np.linspace(0,1,101):self.p.step(pose(z=.3),float(c),{},.01)
        self.assertEqual(self.p.tool.liquid.volume,0.)
        self.assertAlmostEqual(self.p.pressure_pa,P_ATM)
        self.e.assert_conservation()

    def test_immersed_release_aspirates_below_finite_capacity(self):
        self.aspirate()
        self.assertGreater(self.p.tool.liquid.volume,4.8)
        self.assertLessEqual(self.p.tool.liquid.volume,5.)
        self.assertLess(self.p.pressure_pa,P_ATM)
        self.e.assert_conservation()

    def test_dispense_follows_tip_aperture_geometry(self):
        self.aspirate();before=self.p.tool.liquid.volume
        for c in np.linspace(0,.3,101):
            self.p.step(pose(z=.16),float(c),{'beaker':pose()},.01)
            self.f.advance({'beaker':pose()},.01,pour=False)
        for _ in range(100):self.f.advance({'beaker':pose()},.01,pour=False)
        self.assertGreater(self.e.vessels['beaker'].liquid.volume,1.)
        self.assertLess(self.p.tool.liquid.volume,before)
        self.assertAlmostEqual(self.e.spill.volume,0.)
        self.e.assert_conservation()

    def test_offset_tip_spills_instead_of_teleporting_to_beaker(self):
        self.aspirate()
        for c in np.linspace(0,.3,101):
            self.p.step(pose(x=.2,z=.16),float(c),{'beaker':pose()},.01)
            self.f.advance({'beaker':pose()},.01,pour=False)
        for _ in range(100):self.f.advance({'beaker':pose()},.01,pour=False)
        self.assertGreater(self.e.spill.volume,1.)
        self.assertEqual(self.e.vessels['beaker'].liquid.volume,0.)
        self.e.assert_conservation()

    def test_foreign_tip_contact_marks_stock_contaminated(self):
        self.e=Formulation({**CONFIG,'targets_g':{'A':10.,'B':30.}})
        self.f=GeometryFluid(self.e);self.p=PneumaticPipette(self.f,'pipette_A')
        self.e.vessels['bottle_B'].capped=False
        self.p.step(pose(z=.02),0.,{'bottle_B':pose()},.01)
        self.assertTrue(self.e.stock_contaminated)

    def test_input_validation(self):
        for c in (-.1,1.1,float('nan')):
            with self.assertRaises(ValueError):self.p.step(pose(),c,{},.01)


if __name__=='__main__':unittest.main()
