"""CPU reach check with assumed grasps; never physics or task validation."""
import argparse
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace as NS
import sys
import numpy as np

p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--candidate',type=Path);a=p.parse_args()
sys.path.insert(0,str(a.source))
from yam_arena import ik,station,constants
from yam_arena.demo import grip_rotation
from bimanual_cap import ArmTrajectory
from native_coupling import transform
if a.candidate:
 spec=importlib.util.spec_from_file_location('parking_candidate',a.candidate)
 module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);CapParkingProbe=module.CapParkingProbe
else:
 from cap_parking import CapParkingProbe

def pose(T):return np.r_[T[:3,3],ik.matrix_to_quat_xyzw(T[:3,:3])]
def obj(p):return NS(data=NS(root_link_pose_w=np.array([np.r_[p,[0,0,0,1]]])))
class Robot:
 def __init__(self):self.data=NS(joint_pos=np.zeros((1,8)),body_link_pose_w=np.zeros((1,3,7)))
 def find_bodies(self,n,preserve_order=False):
  names=[n] if isinstance(n,str) else n
  return [['gripper','tip_left','tip_right'].index(x)for x in names],names
 def find_joints(self,n,preserve_order=False):return list(range(len(n))),n
base=NS(scene={'bottle_A':obj([.24,.25,.7475]),'bottle_A_cap':obj([.24,.25,.9513]),
 'right_robot':Robot(),'left_robot':Robot(),
 'cap_right_contacts':NS(data=NS(normal_force_matrix_w=np.ones((1,1,2,3))))})
action=np.r_[constants.READY,1.,constants.READY,1.]
left=ArmTrajectory(action,'left',[(2.,np.array([.24,.25,.967]),grip_rotation(0),.042/.0939,'approach'),
 (2.,np.array([.24,.25,.967]),grip_rotation(np.pi/2),.042/.0939,'turn')])
for t in np.arange(.05,4.001,.05):action,_=left.command(t)
action[7:13]=[.9983092546463013,2.3994956016540527,1.247543215751648,1.1519438028335571,-.5724778771400452,-4.3346315692360804e-7]
action[13]=.076/.0939

def hand(value,side):
 offset=0 if side=='left' else 7
 pos,rot=ik.tcp_pose(value[offset:offset+6],station.BASES[side])
 T=np.eye(4);T[:3,:3]=rot;T[:3,3]=pos-rot@np.array(constants.TCP_OFFSET)
 return T
cap=transform(base.scene['bottle_A_cap'].data.root_link_pose_w[0]);attachment='left'
relative=np.linalg.inv(hand(action,'left'))@cap
manifest=json.loads((a.source/'assets/mechanisms/native-assets.json').read_text())
probe=CapParkingProbe(base,action,'bottle_A_cap',manifest)
# Assume a completed opening and ideal grasp only in this separate CPU check.
probe.opening_result={'passed_probe':True};previous=-1
for t in np.arange(probe.opening_duration+.05,probe.duration+.001,.05):
 for side in ['left','right']:
  base.scene[side+'_robot'].data.body_link_pose_w[0,0]=pose(hand(action,side))
 cap=hand(action,attachment)@relative;base.scene['bottle_A_cap'].data.root_link_pose_w[0]=pose(cap)
 if t>probe.opening_duration+16 and attachment=='left':
  attachment='right';relative=np.linalg.inv(hand(action,'right'))@cap
 action,label=probe.command(float(t))
 if probe.phase_index!=previous:
  print('phase',probe.phase_index,label,flush=True);previous=probe.phase_index
print('REACH_PREFLIGHT_PASSED: assumed grasps only; no contact, robot or task validation')
