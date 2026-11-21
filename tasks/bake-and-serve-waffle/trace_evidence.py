"""Read-only, per-step evidence for independent demonstration verification."""
import json
import os
from pathlib import Path
import numpy as np
_stream = None
segment_start_step = None
checkpoint_source = None
feedback = None

def record(env, step, states, score, success, detail):
    global _stream
    dest=os.environ.get('TASK_TRACE_PATH')
    if not dest:
        return
    if _stream is None:
        _stream=Path(dest).open('x',buffering=1)
    from labkit import _np
    from scoring_geometry import hand_bounds
    base=env.unwrapped; origin=_np(base.scene.env_origins[0])
    objects={name:{'pos':s.pos,'quat':s.quat_xyzw,'lin_vel':s.lin_vel,'ang_vel':s.ang_vel,
                    'low':s.low,'high':s.high,'joints':s.joints} for name,s in states.items()}
    for name,s in states.items():
        a=base.scene[name]
        names=list(a.joint_names) if s.joints else []
        objects[name]['joint_vel']={k:float(_np(a.data.joint_vel)[0][names.index(k+'_joint')]) for k in s.joints}
        objects[name]['actuator_state']={}
        for key,actuator in getattr(a,'actuators',{}).items():
            fields={k:_np(getattr(actuator,k))[0] for k in ('plastic_rotation','plastic_dissipation') if hasattr(actuator,k)}
            if fields:objects[name]['actuator_state'][key]=fields
    robots={}
    for name in ('left_robot','right_robot'):
        r=base.scene[name]; poses=_np(r.data.body_link_pose_w)[0]
        i=list(r.body_names).index('gripper'); p=poses[i].copy();p[:3]-=origin
        robots[name]={'gripper_pose':p,'joint_names':list(r.joint_names),'joint_pos':_np(r.data.joint_pos)[0],'joint_vel':_np(r.data.joint_vel)[0]}
    data={'step':step,'sim_time':step*float(getattr(base,'step_dt',.05)),
          'segment_start_step':segment_start_step,'checkpoint_source':checkpoint_source,
          'feedback':feedback,'score':score,'success':bool(success),'objects':objects,'robots':robots,'hands':hand_bounds(env),'detail':detail}
    _stream.write(json.dumps(data,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item())+'\n')
