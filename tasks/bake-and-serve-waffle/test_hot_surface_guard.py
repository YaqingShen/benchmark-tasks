import math
import numpy as np
import task as T
from labkit import PartState
from hot_surface_guard import contact
signs=np.array([[x,y,z] for x in (-1,1) for y in (-1,1) for z in (-1,1)])
def box(p):return signs*.002+p
for lid in (0.,math.radians(45),math.radians(95)):
 for spin in (0.,math.pi):
  w=PartState(np.asarray(T.HOME),np.array([0,0,0,1]),np.zeros(3),np.zeros(3),np.zeros(3),np.zeros(3),{'lid':lid,'spin':spin})
  axis=np.array([0,0,T.ZA]);h=np.asarray(T.HINGE)
  def world(p,upper=False):
   p=np.asarray(p)
   if upper:p=h+T._Ry(lid)@(p-h)
   return w.to_world(axis+T._Rx(spin)@(p-axis))
  assert contact(T,w,[box(world([0,0,T.ZA-.010]))]) is not None
  assert contact(T,w,[box(world([0,0,T.ZA+.010],True))]) is not None
  assert contact(T,w,[box(world([-.135,.02,(T.TAB_Z0+T.TAB_Z1)/2],True))]) is None
print('PASS: upper/lower interiors rejected and external lid handle permitted at open/closed/inverted poses')
