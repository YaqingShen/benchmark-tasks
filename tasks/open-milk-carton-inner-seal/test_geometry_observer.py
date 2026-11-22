import unittest,math
from types import SimpleNamespace
import numpy as np
from geometry_observer import GeometryObserver,ring_inside_tray,penetrates_carton_interior
class State:
 def __init__(self,p=(0,0,0),R=None):self.pos=np.array(p,dtype=float);self.R=np.eye(3) if R is None else R;self.joints={'pullring':0.}
 def to_world(self,x):return self.pos+self.R@np.asarray(x)
 def rotation(self):return self.R
 def tilt_deg(self):return math.degrees(math.acos(np.clip(self.R[2,2],-1,1)))
class GeometricEvidence(unittest.TestCase):
 def setUp(self):
  a=-math.pi/4;self.S=np.array([[math.cos(a),0,math.sin(a)],[0,1,0],[-math.sin(a),0,math.cos(a)]])
  self.ns=dict(RIM=.016,C=np.array([-.026,0,.224]),S=self.S,SEAL_Z=.0035,DISK_R=.016,FULL_CAP=True)
  self.states={x:State((.35,0,.7475)) for x in ['carton','seal','cap']};self.ob=GeometryObserver(self.ns)
 def test_untouched_mouth_is_covered(self):
  x=self.ob.read(self.states);self.assertEqual(x['sampled_open_fraction'],0.);self.assertFalse(x['seal_clear_of_mouth']);self.assertFalse(x['cap_clear_of_mouth'])
 def test_origin_rotation_is_not_seal_removal(self):
  x=self.ob.read(self.states);s=self.states['seal'];c=self.ns['C']+self.S@np.array([0,0,.0035]);old=s.to_world(c);R=np.diag([-1,-1,1]);s.R=R;s.pos=old-R@c
  y=self.ob.read(self.states);self.assertTrue(np.allclose(x['seal_center'],y['seal_center']));self.assertFalse(y['seal_clear_of_mouth'])
 def test_pushing_disk_inside_does_not_expose_mouth(self):
  self.states['seal'].pos-=self.S[:,2]*.01;x=self.ob.read(self.states);self.assertEqual(x['sampled_open_fraction'],0.);self.assertFalse(x['seal_clear_of_mouth'])
 def test_deep_inward_motion_is_not_removal(self):
  self.states['seal'].pos-=self.S[:,2]*.05;x=self.ob.read(self.states);self.assertFalse(x['seal_clear_of_mouth'])
 def test_true_translation_clears_mouth(self):
  self.states['seal'].pos+=np.array([.08,0,.1]);x=self.ob.read(self.states);self.assertEqual(x['sampled_open_fraction'],1.);self.assertTrue(x['seal_clear_of_mouth'])
 def test_speed_tracks_feature_not_origin(self):
  self.ob.read(self.states);s=self.states['seal'];c=self.ns['C']+self.S@np.array([0,0,.0035]);old=s.to_world(c);s.R=np.diag([-1,-1,1]);s.pos=old-s.R@c
  self.assertLess(self.ob.read(self.states)['seal_center_speed_mps'],1e-12)
if __name__=='__main__':unittest.main()

class DepositedRing(unittest.TestCase):
 def test_folded_ring_over_tray_edge(self):
  tray=State();seal=State((-.075,0,.015))
  hinge=np.array([-.0115,0,.004]);points=np.array([[.023,0.,0.],[0.,0.,0.]])
  self.assertTrue(ring_inside_tray(points,hinge,0.,seal,tray))
  self.assertFalse(ring_inside_tray(points,hinge,-2.4,seal,tray))

class NeckBoundary(unittest.TestCase):
 def test_inside_neck_is_distinct_from_inside_carton(self):
  center=np.array([0.,0.,.008])
  self.assertFalse(penetrates_carton_interior(center,np.array([[.010,0.,-.003]]),.016,-.0045))
  self.assertTrue(penetrates_carton_interior(center,np.array([[.010,0.,-.006]]),.016,-.0045))
 def test_inward_dropped_disk_fails(self):
  self.assertTrue(penetrates_carton_interior(np.array([0.,0.,-.049]),np.array([[.016,0.,-.049]]),.016,-.0045))
