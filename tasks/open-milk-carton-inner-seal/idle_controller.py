import numpy as np

def plan(ns):
 from yam_arena import ik,demo as dm
 from yam_arena.constants import READY,FINGER_OPEN,LEFT_BASE_POS,RIGHT_BASE_POS
 class Orient:
  def __init__(self,r):self.rot=r
 old=dm.grip_rotation
 if not getattr(old,'_milk',False):
  def grip_rotation(x=0.):return x.rot if hasattr(x,'rot') else old(x)
  grip_rotation._milk=True;dm.grip_rotation=grip_rotation
 out={}
 for a,b in [('left',LEFT_BASE_POS),('right',RIGHT_BASE_POS)]:
  p,r=ik.tcp_pose(np.array(READY),np.array(b));out[a]=[dm.Waypoint(30.,tuple(p),FINGER_OPEN,'idle',Orient(r))]
 return out
