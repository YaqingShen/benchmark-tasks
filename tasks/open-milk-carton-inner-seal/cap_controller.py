"""Unscrew via robot contact on explicitly modeled helical ridges."""
import math
import numpy as np

def add(ns,wp,wait,clock,rg,down):
 R='right';S=ns['S'];N=ns['N'];P=ns['P'].copy();P[2]=ns['TOP'];sp=P+ns['C']
 from yam_arena.constants import FINGER_OPEN,FINGER_CLOSED
 fo,fc=FINGER_OPEN,-.020  # 40 mm commanded gap for a nominal 44.4 mm ribbed cap
 def rz(a):return ns['labkit']._rot((0,0,a))
 def grasp(turns):return sp+N*(.012+.003*turns)-.007*rg[:,2]
 wait(R,clock['left'])
 wp(R,2.,grasp(0)+.08*N,fo,rg,'cap: approach')
 for cycle in range(8):
  turns=cycle/4; p=grasp(turns)
  if cycle:
   wp(R,.65,p+.035*N,fo,rg,'cap: reset wrist')
  wp(R,.8,p,fo,rg,'cap: straddle exterior')
  wp(R,.5,p,fc,rg,'cap: grip exterior')
  for deg in range(15,91,15):
   angle=math.radians(deg);rot=S@rz(angle)@down
   wp(R,.35,grasp(turns+deg/360),fc,rot,'cap: turn')
  # Regrasp transitions reset wrist orientation only after releasing.
  if cycle<7:
   rot=S@rz(math.pi/2)@down
   wp(R,.35,grasp((cycle+1)/4),fo,rot,'cap: open jaws')
   wp(R,.5,grasp((cycle+1)/4)+.035*N,fo,rot,'cap: withdraw open jaws')
 last=grasp(2);rot=S@rz(math.pi/2)@down
 wp(R,1.2,last+.07*N,fc,rot,'cap: lift free')
 # Keep the recessed inner face clear; set only the exterior rim on a clean rest.
 centre=ns['CAP_REST_POS']+np.array([0,0,.16])
 wp(R,1.5,centre-.007*rot[:,2],fc,rot,'cap: carry above clean rest')
 for deg in range(5,46,5):
  orient=ns['labkit']._rot((0,math.radians(deg),0))@rot
  wp(R,.35,centre-.007*orient[:,2],fc,orient,'cap: align upright on clean rest')
 target=ns['CAP_REST_POS']+np.array([0,0,.016])
 wp(R,1.5,target-.007*orient[:,2],fc,orient,'cap: lower exterior rim onto clean rest')
 wp(R,.6,target-.007*orient[:,2],fo,orient,'cap: release on clean rest')
 wp(R,1.,target+[0,0,.10],fo,orient,'cap: clear rest')
