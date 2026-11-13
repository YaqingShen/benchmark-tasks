import unittest
import numpy as np
from geometry_fluid import INTERIORS
from liquid_mass_properties import liquid_properties,combine_masses,principal_frame,pipette_liquid_properties


class MassPropertiesTests(unittest.TestCase):
    def test_upright_liquid_matches_cylinder_formula(self):
        vessel=INTERIORS['beaker'];volume=40.;mass=.04
        level=vessel.upright_height(volume);height=level-vessel.z[0];radius=vessel.radius[0]
        center,inertia=liquid_properties(vessel,np.eye(4),level,mass)
        np.testing.assert_allclose(center,[0,0,vessel.z[0]+height/2],atol=1e-9)
        expected=np.diag([mass*(3*radius**2+height**2)/12]*2+[mass*radius**2/2])
        # The native free surface has 48 facets, giving <0.5% radial error.
        np.testing.assert_allclose(np.diag(inertia),np.diag(expected),rtol=.005)

    def test_tilt_moves_liquid_center_downhill(self):
        vessel=INTERIORS['beaker'];angle=np.deg2rad(3)
        pose=np.eye(4);pose[:3,:3]=[[np.cos(angle),0,np.sin(angle)],[0,1,0],[-np.sin(angle),0,np.cos(angle)]]
        height=vessel.upright_height(40.);level=np.cos(angle)*height
        center,inertia=liquid_properties(vessel,pose,level,.04)
        self.assertGreater(center[0],.001)
        self.assertTrue(np.linalg.eigvalsh(inertia).min()>0)

    def test_composite_parallel_axis(self):
        m,c,I=combine_masses([(1.,np.array([-1.,0,0]),np.eye(3)),(1.,np.array([1.,0,0]),np.eye(3))])
        self.assertEqual(m,2.)
        np.testing.assert_allclose(c,[0,0,0])
        np.testing.assert_allclose(I,np.diag([2.,4.,4.]))

    def test_principal_frame_reconstructs_inertia(self):
        inertia=np.array([[3.,.3,-.2],[.3,2.,.1],[-.2,.1,1.]])*1e-4
        moments,rotation=principal_frame(inertia)
        np.testing.assert_allclose(rotation@np.diag(moments)@rotation.T,inertia,atol=1e-15)
        self.assertAlmostEqual(np.linalg.det(rotation),1.)

    def test_filled_pipette_moves_liquid_center_toward_bulb(self):
        small,_=pipette_liquid_properties(.1,.0001)
        large,inertia=pipette_liquid_properties(4.,.004)
        self.assertGreater(large[2],small[2]);self.assertLess(large[2],.211)
        self.assertTrue((np.linalg.eigvalsh(inertia)>0).all())


if __name__=='__main__':unittest.main()
