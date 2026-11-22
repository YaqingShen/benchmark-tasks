"""Outcome-based draft rubric for the intact pull-ring seal, with outer cap pre-removed.
This module never moves bodies or changes constraints. The task reads this score only with complete native observations.
"""
from dataclasses import dataclass,field
import numpy as np

@dataclass
class Observation:
 cap_off:bool=False
 cap_clean_supported:bool=False
 seal_detached:bool=False
 seal_material_open:bool=False
 open_fraction:float=0.
 seal_in_tray:bool=False
 seal_supported:bool=False
 seal_speed:float=1.
 carton_supported:bool=True
 carton_tilt_deg:float=0.
 hands_released:bool=False
 contamination:bool=False
 spill_ml:float=0.
 fragment_in_carton:bool=False
 carton_damaged:bool=False
 ring_only_broken:bool=False
 unsafe_robot_contact:bool=False
 unsupported_opening:bool=False
 supported_ring_load:bool=False

@dataclass
class Tracker:
 violations:set=field(default_factory=set)
 final_hold_s:float=0.
 max_spill_ml:float=0.
 supported_pull_seen:bool=False

 def update(self,o,dt=.05):
  if not 0<dt<=.1:raise ValueError('invalid observation interval')
  if not 0<=o.open_fraction<=1:raise ValueError('opening area outside [0,1]')
  self.max_spill_ml=max(self.max_spill_ml,o.spill_ml)
  for key in ['contamination','fragment_in_carton','carton_damaged','ring_only_broken','unsafe_robot_contact','unsupported_opening']:
   if getattr(o,key):self.violations.add(key)
  # 1 ml is a stated measurement tolerance, not permission for uncounted drips.
  if self.max_spill_ml>1.:self.violations.add('spill_over_1ml')
  self.supported_pull_seen |= o.supported_ring_load and not o.seal_detached
  final=(self.supported_pull_seen and o.seal_material_open and o.cap_off and o.cap_clean_supported and o.seal_detached and o.open_fraction>=.98
   and o.seal_in_tray and o.seal_supported and o.seal_speed<=.01
   and o.carton_supported and o.carton_tilt_deg<=5 and o.hands_released)
  self.final_hold_s=self.final_hold_s+dt if final and not self.violations else 0.
  return self.score(o)

 def score(self,o):
  if self.violations or not self.supported_pull_seen or not o.seal_material_open:return 0.
  if self.final_hold_s>=1.-1e-9:return 1.
  if o.seal_detached and o.open_fraction>=.98:return .7
  if o.open_fraction>=.25:return .35
  # The cap is already removed at reset and earns no progress credit.
  return 0.


def box_vertices(center,rotation,half):
 signs=np.array([[x,y,z] for x in [-1,1] for y in [-1,1] for z in [-1,1]])
 return signs*np.asarray(half)@np.asarray(rotation).T+np.asarray(center)

def convex_overlap(a,b,axes,tolerance=0.):
 """Separating-axis test, including all face normals and edge cross products."""
 for axis in axes:
  axis=np.asarray(axis);norm=np.linalg.norm(axis)
  if norm<1e-10:continue
  axis=axis/norm;pa=np.asarray(a)@axis;pb=np.asarray(b)@axis
  if pa.max()<pb.min()-tolerance or pb.max()<pa.min()-tolerance:return False
 return True

def box_overlap(ca,ra,ha,cb,rb,hb,tolerance=0.):
 ra=np.asarray(ra);rb=np.asarray(rb)
 axes=[*ra.T,*rb.T,*[np.cross(a,b) for a in ra.T for b in rb.T]]
 return convex_overlap(box_vertices(ca,ra,ha),box_vertices(cb,rb,hb),axes,tolerance)
