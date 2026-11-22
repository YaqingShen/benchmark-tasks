import unittest
import numpy as np
from rules import Observation,Tracker,box_overlap

class ScoringTests(unittest.TestCase):
 def after_braced_pull(self):
  tracker=Tracker()
  tracker.update(Observation(supported_ring_load=True))
  return tracker
 def test_opening_by_bumping_carton_earns_nothing(self):
  t=Tracker()
  for _ in range(40):self.assertEqual(t.update(self.complete()),0.)
 def test_first_ring_contact_after_removal_earns_nothing(self):
  t=Tracker()
  for _ in range(40):self.assertEqual(t.update(self.complete(supported_ring_load=True)),0.)
 def complete(self,**kw):
  args=dict(cap_off=True,cap_clean_supported=True,seal_detached=True,seal_material_open=True,open_fraction=1.,seal_in_tray=True,seal_supported=True,seal_speed=0.,hands_released=True)
  args.update(kw);return Observation(**args)
 def test_untouched_entire_horizon(self):
  t=Tracker()
  for _ in range(3600):self.assertEqual(t.update(Observation()),0.)
 def test_pre_removed_clean_cap_earns_zero(self):
  self.assertEqual(Tracker().update(Observation(cap_off=True,cap_clean_supported=True)),0.)
 def test_cap_held_without_clean_support_zero(self):
  self.assertEqual(Tracker().update(Observation(cap_off=True)),0.)
 def test_ring_only_break_zero(self):
  t=Tracker();t.update(Observation(cap_off=True,ring_only_broken=True));self.assertEqual(t.update(self.complete()),0.)
 def test_held_seal_not_full(self):
  t=self.after_braced_pull()
  for _ in range(40):self.assertEqual(t.update(self.complete(hands_released=False)),.7)
 def test_floating_seal_not_full(self):
  t=self.after_braced_pull()
  for _ in range(40):self.assertEqual(t.update(self.complete(seal_supported=False)),.7)
 def test_full_requires_continuous_hold(self):
  t=self.after_braced_pull()
  for _ in range(19):self.assertEqual(t.update(self.complete()),.7)
  self.assertEqual(t.update(self.complete()),1.)
  self.assertEqual(t.update(self.complete(hands_released=False)),.7)
 def test_contamination_cannot_be_undone(self):
  t=Tracker();t.update(self.complete(contamination=True))
  for _ in range(30):self.assertEqual(t.update(self.complete()),0.)
 def test_spill_accumulation_never_resets(self):
  t=Tracker();t.update(self.complete(spill_ml=1.01));self.assertEqual(t.update(self.complete()),0.)
 def test_fragment_drop_zero(self):
  self.assertEqual(Tracker().update(self.complete(fragment_in_carton=True)),0.)
 def test_cap_dirty_face_prevents_full(self):
  t=self.after_braced_pull()
  for _ in range(30):self.assertLess(t.update(self.complete(cap_clean_supported=False)),1.)
 def test_partial_open_requires_actual_area(self):
  self.assertEqual(Tracker().update(Observation(cap_off=True,open_fraction=.249)),0.)
  self.assertEqual(self.after_braced_pull().update(Observation(cap_off=True,seal_material_open=True,open_fraction=.25)),.35)
  self.assertEqual(Tracker().update(Observation(cap_off=True,open_fraction=.25)),0.)
 def test_intact_bond_earns_zero_even_if_it_flexes(self):
  t=self.after_braced_pull()
  for _ in range(30):self.assertEqual(t.update(self.complete(seal_material_open=False)),0.)
 def test_robot_collision_cannot_be_erased_by_finishing(self):
  t=Tracker();t.update(self.complete(unsafe_robot_contact=True))
  for _ in range(30):self.assertEqual(t.update(self.complete()),0.)
 def test_unsupported_opening_has_no_credit(self):
  t=Tracker();t.update(self.complete(unsupported_opening=True));self.assertEqual(t.update(self.complete()),0.)
 def test_rotated_box_overlap(self):
  I=np.eye(3);R=np.array([[2**-.5,-2**-.5,0],[2**-.5,2**-.5,0],[0,0,1]])
  self.assertTrue(box_overlap([0,0,0],I,[1,1,1],[1,0,0],R,[.2,.2,.2]))
  self.assertFalse(box_overlap([0,0,0],I,[1,1,1],[2,0,0],R,[.2,.2,.2]))

if __name__=='__main__':unittest.main()
