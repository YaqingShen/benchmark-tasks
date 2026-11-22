"""Check the effective wrapper joint moment with known hanging masses."""
from pathlib import Path
from dataclasses import replace
import json,math,os,sys
import numpy as np
sys.path.insert(0,str(Path(__file__).parent.resolve()))
import task
from task import labkit,g,at
from yam_arena.demo import Waypoint
SCENE='station'
labkit._STEP.remove(task.trial.tick)
labkit._RESET.remove(task.trial.reset)
LIMIT=task.shells[1].meta['seam_torque_Nm']
CASES=[('light',.070,LIMIT),('heavy',.225,LIMIT),('locked',.225,1.e5)]
OBJECTS=[]
for i,(name,mass,torque) in enumerate(CASES):
    center=np.array([.36,(i-1)*.19])
    L=replace(task.shells[0],name=name+'_left',fixed=True,mass=None,
              pos=tuple(map(float,at(center+[0,g.L/2],.17))),meta={'rotation_xyzw':task.q0})
    R=replace(task.shells[1],name=name+'_right',mass=mass,
              pos=tuple(map(float,at(center-[0,g.L/2],.17))),meta={'rotation_xyzw':task.q0,'seam_peer':L.name,'seam_anchor_m':g.L/2})
    # A mass with its centre 80 mm horizontally from the fracture plane.
    # Gravitational torque is mass times g times distance; no scripted impulses.
    labkit.MASS_PROPERTIES[R.name]=((0.,0.,g.L/2-.080),(.0005,.0005,.0001))
    OBJECTS.extend([L,R])
rows=[]
def reset(env):
    from isaaclab.sim.utils.stage import get_current_stage
    from pxr import UsdPhysics
    labkit.paper_events.clear();rows.clear()
    for name,mass,limit in CASES:
        joint=UsdPhysics.FixedJoint.Get(get_current_stage(),env.unwrapped.scene.env_prim_paths[0]+'/'+name+'_right/paper_seam')
        joint.GetJointEnabledAttr().Set(False)
        joint.GetBreakForceAttr().Set(10000.)
        joint.GetBreakTorqueAttr().Set(limit)
        joint.GetJointEnabledAttr().Set(True)
def tick(env):
    i=len(rows)
    row={}
    for j,(name,mass,limit) in enumerate(CASES):
        left,right=[labkit._part_state(env,s) for s in OBJECTS[2*j:2*j+2]]
        a=left.to_world([0,0,-g.L/2]);b=right.to_world([0,0,g.L/2])
        row[name]={'moment_Nm':mass*9.81*.080,'limit_Nm':limit,'gap_m':float(np.linalg.norm(a-b)),
                   'broken':any('/'+name+'_right/' in path for path in labkit.paper_events)}
    rows.append(row)
    if i%10==0:print('CALIBRATION',i,json.dumps(row),flush=True)
    if i==50:Path(os.environ['CALIBRATION_OUT']).write_text(json.dumps(rows))
labkit.on_reset(reset);labkit.on_step(tick)
def demo():
    from yam_arena import ik,station
    from yam_arena.constants import READY
    return {a:[Waypoint(4.,tuple(ik.tcp_pose(np.array(READY),station.BASES[a])[0]),-.030,'stand clear')] for a in ('left','right')}
def score(objects):return 0.
def success(objects):return False
