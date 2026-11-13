"""CPU reach-only preflight with assumed tool grasp; not physics validation."""
import argparse,json,sys
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);args=p.parse_args();sys.path.insert(0,str(args.source))
from yam_arena import constants,ik,station
from pipette_correction import PipetteCorrectionProbe
from native_coupling import transform
from geometry_fluid import GeometryFluid
from sim import Formulation
manifest=json.loads((args.source/'assets/mechanisms/native-assets.json').read_text())
poses={n:transform(np.r_[s['position_xyz'],s.get('rotation_xyzw',[0,0,0,1])])for n,s in manifest['assets'].items()}
class Robot:
 def __init__(self):self.data=NS(body_link_pose_w=np.zeros((1,1,7)))
 def find_bodies(self,n):return [0],['gripper']
base=NS(scene={'left_robot':Robot(),'pipette_A':NS(data=NS(root_link_pose_w=np.array([np.r_[poses['pipette_A'][:3,3],ik.matrix_to_quat_xyzw(poses['pipette_A'][:3,:3])]])))})
for mode in ('remove','add'):
 e=Formulation(initial_beaker_g=10.5 if mode=='remove'else 9.5)
 if mode=='add':e.vessels['bottle_A'].capped=False
 coupling=NS(poses=lambda:poses,fluid=GeometryFluid(e),balance=NS(last={'net_g':e.balance(),'stable':True}),ledger=e)
 action=np.r_[constants.READY,1.,constants.READY,1.]
 probe=PipetteCorrectionProbe(base,action,coupling,mode)
 # Test nominal Cartesian trajectory only; no simulated feedback or grasp.
 from native_probes import PipetteProbe
 previous=''
 for t in np.arange(.05,probe.duration+.001,.05):
  action,label=PipetteProbe.command(probe,float(t))
  if label!=previous:print(mode,label,flush=True);previous=label
 print(mode,'REACH_PREFLIGHT_PASSED',flush=True)
