import unittest
import numpy as np
from sim import Formulation,CONFIG
from geometry_fluid import GeometryFluid


def pose(x=0,y=0,z=0,tilt=0):
    t=np.eye(4);a=np.deg2rad(tilt);c,s=np.cos(a),np.sin(a)
    t[:3,:3]=[[c,0,s],[0,1,0],[-s,0,c]];t[:3,3]=[x,y,z]
    return t


class GeometryTests(unittest.TestCase):
    def test_shallow_beaker_surface_resolves_tenth_ml_differences(self):
        e=Formulation();f=GeometryFluid(e)
        e._receive('beaker',e.vessels['bottle_A'].liquid.take(10.))
        height=f.surface('beaker',pose())
        self.assertAlmostEqual(height,.004+10e-6/(np.pi*.0415**2),places=10)
        e._receive('beaker',e.vessels['bottle_A'].liquid.take(.1))
        self.assertAlmostEqual(f.surface('beaker',pose())-height,.1e-6/(np.pi*.0415**2),places=10)

    def test_small_tilt_surface_is_continuous_at_upright(self):
        e=Formulation();f=GeometryFluid(e)
        e._receive('beaker',e.vessels['bottle_A'].liquid.take(10.))
        for tilt in (0.,.01,.1,.5,1.):
            h=.004+10e-6/(np.pi*.0415**2)
            self.assertAlmostEqual(f.surface('beaker',pose(z=.8,tilt=tilt)),.8+np.cos(np.deg2rad(tilt))*h,places=10)

    def test_upright_open_bottle_does_not_pour(self):
        e=Formulation();e.cap('bottle_A',-1.5);f=GeometryFluid(e)
        self.assertEqual(f.pour('bottle_A',pose(),.01),0.)

    def test_tilt_causes_geometry_derived_flow(self):
        e=Formulation();e.cap('bottle_A',-1.5);f=GeometryFluid(e)
        amount=f.pour('bottle_A',pose(z=.3,tilt=110),.01)
        self.assertGreater(amount,0.)
        self.assertGreater(sum(p.volume for p in e.inflight),0.)
        e.assert_conservation()

    def test_ballistic_capture_without_named_destination(self):
        e=Formulation();f=GeometryFluid(e)
        packet=e.vessels['bottle_A'].liquid.take(1.)
        f.emit('bottle_A',packet,[0,0,.3],[0,0,0])
        for _ in range(100):f.advance({'beaker':pose()},.01,pour=False)
        self.assertAlmostEqual(e.vessels['beaker'].liquid.volume,1.)
        self.assertAlmostEqual(e.spill.volume,0.)
        self.assertEqual(len(e.inflight),0)
        e.assert_conservation()

    def test_missed_aperture_spills(self):
        e=Formulation();f=GeometryFluid(e)
        f.emit('bottle_A',e.vessels['bottle_A'].liquid.take(1.),[.15,0,.3],[0,0,0])
        for _ in range(100):f.advance({'beaker':pose()},.01,pour=False)
        self.assertEqual(e.vessels['beaker'].liquid.volume,0.)
        self.assertAlmostEqual(e.spill.volume,1.)
        e.assert_conservation()

    def test_closed_receiving_bottle_cannot_capture(self):
        e=Formulation({**CONFIG,'targets_g':{'A':10.,'B':30.},'initial_bottle_open':False});f=GeometryFluid(e)
        f.emit('bottle_A',e.vessels['bottle_A'].liquid.take(1.),[0,0,.3],[0,0,0])
        for _ in range(100):f.advance({'bottle_B':pose()},.01,pour=False)
        self.assertEqual(e.vessels['bottle_B'].liquid.volume,200.)
        self.assertAlmostEqual(e.spill.volume,1.)

    def test_immersion_uses_fill_level_and_geometry(self):
        e=Formulation({**CONFIG,'cylinders':['cylinder_A']});f=GeometryFluid(e)
        e._receive('cylinder_A',e.vessels['bottle_A'].liquid.take(10.))
        self.assertEqual(f.tip_immersion([0,0,.02],{'cylinder_A':pose()}),'cylinder_A')
        self.assertIsNone(f.tip_immersion([0,0,.1],{'cylinder_A':pose()}))
        self.assertIsNone(f.tip_immersion([.02,0,.02],{'cylinder_A':pose()}))

    def test_nonrigid_pose_rejected(self):
        e=Formulation();f=GeometryFluid(e);p=pose();p[0,0]=2
        with self.assertRaises(ValueError):f.surface('beaker',p)

    def test_equal_packets_keep_distinct_ownership(self):
        e=Formulation();f=GeometryFluid(e)
        for height in (.30,.13):
            f.emit('bottle_A',e.vessels['bottle_A'].liquid.take(.1),[0,0,height],[0,0,0])
        for _ in range(100):f.advance({'beaker':pose()},.01,pour=False)
        self.assertAlmostEqual(e.vessels['beaker'].liquid.volume,.2)
        self.assertEqual(len(e.inflight),0)
        e.assert_conservation()


if __name__=='__main__':unittest.main()
