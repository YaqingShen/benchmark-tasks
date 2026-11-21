"""Explicit state-restored suffix debugging; never a complete demonstration.

Only reset writes physical state. There are no object pose writes during rollout.
Older traces estimate joint velocities and plastic rest from control-rate samples;
PhysX contact warm-start caches are not serialized. Always rerun from normal reset
before calling a task complete.
"""
import dataclasses
import hashlib
import json
from pathlib import Path
import numpy as np


def encode(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,Path):return str(x)
    raise TypeError(type(x).__name__)


def model_hash(specs):
    data=[dataclasses.asdict(x) for x in specs]
    return hashlib.sha256(json.dumps(data,default=encode,sort_keys=True).encode()).hexdigest()


def clip_plan(wps,offset):
    from yam_arena.demo import Waypoint
    result=[]
    for w in wps:
        if offset>=w.duration-1e-8:
            offset=max(0.,offset-w.duration);continue
        result.append(Waypoint(w.duration-offset,w.pos,w.finger,w.label,w.yaw))
        offset=0.
    if not result:raise ValueError('Checkpoint lies after the complete plan')
    return result


def states(row):
    from labkit import PartState
    return {n:PartState(np.array(d['pos']),np.array(d['quat']),np.array(d['lin_vel']),
             np.array(d['ang_vel']),np.array(d['low']),np.array(d['high']),d['joints'])
             for n,d in row['objects'].items()}


def configure(ns):
    path=Path(ns['__file__']).with_name('checkpoint.json')
    if not path.exists():return False
    cp=json.loads(path.read_text())
    if model_hash(ns['OBJECTS'])!=cp['model_sha256']:
        raise ValueError('Checkpoint and current physical model differ')
    import yam_arena.demo as dm
    from yam_arena.constants import LEFT_BASE_POS,RIGHT_BASE_POS,FINGER_OPEN
    original_demo=ns['demo']
    original_init=dm.ArmPlan.__init__
    identity=str(path.resolve())
    if getattr(original_init,'_segment_resume',False):
        if original_init._segment_id!=identity:raise RuntimeError('Conflicting checkpoint initializer')
        original_init=original_init._segment_original
    def start_plan(self,waypoints,base_pos,*args,**kwargs):
        arm='left_robot' if np.linalg.norm(np.asarray(base_pos)-LEFT_BASE_POS)<np.linalg.norm(np.asarray(base_pos)-RIGHT_BASE_POS) else 'right_robot'
        command=cp['robots'][arm]['command']
        original_init(self,waypoints,base_pos,q_start=command[:6],finger_start=command[6]*FINGER_OPEN)
    start_plan._segment_resume=True
    start_plan._segment_id=identity
    start_plan._segment_original=original_init
    dm.ArmPlan.__init__=start_plan
    def suffix_demo():
        return {arm:clip_plan(wps,cp['sim_time']) for arm,wps in original_demo().items()}
    ns['demo']=suffix_demo
    ns['trace_evidence'].segment_start_step=cp['step']
    ns['trace_evidence'].checkpoint_source=cp['source_run']

    def restore(environment):
        import torch
        from labkit import _np
        base=environment.unwrapped;dev=base.scene.env_origins.device
        origin=_np(base.scene.env_origins[0])
        tensor=lambda x:torch.tensor([x],dtype=torch.float32,device=dev)
        for name,row in cp['objects'].items():
            a=base.scene[name]
            pose=[*(np.asarray(row['pos'])+origin),*row['quat']]
            a.write_root_pose_to_sim_index(root_pose=tensor(pose))
            a.write_root_velocity_to_sim_index(root_velocity=tensor(row['lin_vel']+row['ang_vel']))
            if row['joints']:
                # One complete write avoids partial writes reloading stale backend joint buffers.
                names=[n.removesuffix('_joint') for n in a.joint_names]
                a.write_joint_position_to_sim_index(position=tensor([row['joints'][n] for n in names]))
                a.write_joint_velocity_to_sim_index(velocity=tensor([row['joint_vel'][n] for n in names]))
            for key,value in row.get('actuator_state',{}).items():
                actuator=a.actuators[key]
                for field,v in value.items():
                    if hasattr(actuator,field):
                        target=getattr(actuator,field)
                        target[0]=torch.tensor(v,dtype=target.dtype,device=dev)
                if hasattr(actuator,'needs_init'):actuator.needs_init[0]=False
        for name,row in cp['robots'].items():
            a=base.scene[name];names=list(a.joint_names)
            order=[row['joint_names'].index(k) for k in names]
            q=[row['joint_pos'][i] for i in order];v=[row['joint_vel'][i] for i in order]
            ids=list(range(len(names)))
            a.write_joint_position_to_sim_index(position=tensor(q),joint_ids=ids)
            a.write_joint_velocity_to_sim_index(velocity=tensor(v),joint_ids=ids)
            cmd=row['command'];target=dict(zip(['joint'+str(i) for i in range(1,7)],cmd[:6]))
            target.update(joint7=cmd[6]*FINGER_OPEN,joint8=cmd[6]*FINGER_OPEN)
            a.set_joint_position_target_index(target=tensor([target[k] for k in names]),joint_ids=ids)
        base.scene.write_data_to_sim()
        base.sim.forward()  # refresh link transforms without advancing physics
        base.scene.update(0.)
        restored={name:ns['labkit']._part_state(environment,next(o for o in ns['OBJECTS'] if o.name==name)) for name in cp['objects']}
        print('SEGMENT_STATE_CHECK '+json.dumps({name:{'root_error_mm':float(np.linalg.norm(st.pos-cp['objects'][name]['pos'])*1000),'joints':st.joints} for name,st in restored.items()}),flush=True)
        dt=cp['sim_time']/cp['step']
        for row in cp['prefix']:
            obj=states(row);hands=[(np.array(lo),np.array(hi)) for lo,hi in row['hands']]
            if 'bake_state' in cp:
                ns['phase_conversion'].STATE['solid']=row['sim_time']>cp['bake_state']['robot_time_s']
            if '_TRACK' in ns:
                ns['track_update'](ns['_TRACK'],obj,hands,ns['food_contact'].hand_hulls_from_robots(row['robots'])) if ns.get('EXACT_FOOD_GUARD') else (ns['track_update'](ns['_TRACK'],obj,hands) if ns.get('SERVING_RULES') else ns['track_update'](ns['_TRACK'],obj))
                ns['update_final_release'](obj,hands,dt)
            else:
                ns['_M'].update(n=row['step'],t=row['sim_time'],hands=hands)
                ns['update_final_hold'](obj)
        if '_S' in ns:ns['_S']['step']=cp['step']  # skips initial layout settling writes
        if '_M' in ns:ns['_M'].update(n=cp['step'],t=cp['sim_time'])
        print('SEGMENT_RESTORE '+json.dumps({'source':cp['source_run'],'step':cp['step'],
              'segment_only':True,'limitations':cp['limitations']}),flush=True)
    ns['labkit']._RESET.append(restore)
    return True
