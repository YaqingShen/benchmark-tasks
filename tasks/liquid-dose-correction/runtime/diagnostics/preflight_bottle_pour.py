"""Check bottle-pour reach with an ideal grasp, never physics validation."""
import argparse,json,sys
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--source',required=True,type=Path);a=p.parse_args()
sys.path.insert(0,str(a.source))
from yam_arena import ik,station,constants
from native_coupling import transform
from bottle_pour import BottlePourProbe

def pose(T):return np.r_[T[:3,3],ik.matrix_to_quat_xyzw(T[:3,:3])]
def hand(action):
 pos,rot=ik.tcp_pose(action[7:13],station.BASES['right']);T=np.eye(4)
 T[:3,:3]=rot;T[:3,3]=pos-rot@np.array(constants.TCP_OFFSET);return T
class Robot:
 def __init__(self):self.data=NS(joint_pos=np.zeros((1,8)),body_link_pose_w=np.zeros((1,3,7)))
 def find_bodies(self,n,preserve_order=False):
  names=[n]if isinstance(n,str)else n
  return [['gripper','tip_left','tip_right'].index(x)for x in names],names
 def find_joints(self,n,preserve_order=False):return list(range(len(n))),n
bottle=np.eye(4);bottle[:3,3]=[.24,.25,.7475]
base=NS(scene={'bottle_A':NS(data=NS(root_link_pose_w=np.array([pose(bottle)]))),
 'right_robot':Robot(),'left_robot':Robot(),
 'bottle_hold_contacts':NS(data=NS(normal_force_matrix_w=np.ones((1,1,2,3))))})
beaker=np.eye(4);beaker[:3,3]=[.42,0,.7995]
coupling=NS(poses=lambda:{'beaker':beaker},balance=NS(last={'net_g':0.}))
action=np.r_[constants.READY,1.,constants.READY,1.]
action[0]=.65
action[7:13]=[.9983092546,2.3994956017,1.2475432158,1.1519438028,-.5724778771,0]
action[13]=.076/.0939
base.scene['right_robot'].data.body_link_pose_w[0,0]=pose(hand(action))
base.scene['right_robot'].data.joint_pos[0,:6]=action[7:13]
probe=BottlePourProbe(base,action,coupling)
relative=np.linalg.inv(hand(action))@bottle;previous=-1
for t in np.arange(probe.prep_duration+.05,probe.duration+.001,.05):
 base.scene['right_robot'].data.body_link_pose_w[0,0]=pose(hand(action))
 base.scene['right_robot'].data.joint_pos[0,:6]=action[7:13]
 base.scene['bottle_A'].data.root_link_pose_w[0]=pose(hand(action)@relative)
 if t>probe.prep_duration+23:coupling.balance.last['net_g']=8.6
 action,label=probe.command(float(t))
 if previous!=probe.phase_index:print(probe.phase_index,label,flush=True);previous=probe.phase_index
print('REACH_PREFLIGHT_PASSED: assumed grasp; not physical delivery validation')
