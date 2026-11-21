"""Abstract baking state swap under a closed lid, explicitly not phase physics.

The cooked collision model is absent until the bake timer completes. The single
swap seats the canonical solid in the moving mold, then releases it to normal
rigid-body physics. Robot interactions never directly set the food pose again.
"""
import math,json,os
from pathlib import Path
import numpy as np
from cooking_state import BATTER_MASS_KG,TIME_SCALE
STATE={'solid':False,'event':None,'block_reason':'baking timer incomplete'}

def reset():
    STATE.clear();STATE.update(solid=False,event=None,block_reason='baking timer incomplete')

def mold_coordinates(ns,w,points):
    rotation=w.rotation()@ns['_Rx'](w.joints.get('spin',0.))
    axis=w.to_world([0.,0.,ns['ZA']])
    return (np.asarray(points)-axis)@rotation,axis,rotation

def eligibility(*,cooked,lid,spin,points_local):
    if not cooked:return 'baking timer incomplete'
    if abs(lid)>math.radians(3):return 'lid not closed'
    if abs(spin-math.pi)>math.radians(5):return 'waiting for inverted hold'
    p=np.asarray(points_local)
    if not len(p) or not np.isfinite(p).all():return 'invalid batter state'
    if np.any(np.linalg.norm(p[:,:2],axis=1)>.109) or np.any(np.abs(p[:,2])>.043):
        return 'visible spill outside cavity must not disappear during state swap'
    return None

def attempt(ns,env,paths,points,velocities,clock,w):
    if STATE['solid']:return
    local,axis,rot=mold_coordinates(ns,w,points)
    reason=eligibility(cooked=clock.cooked,lid=w.joints['lid'],spin=w.joints['spin'],points_local=local)
    STATE['block_reason']=reason
    if reason:return
    import torch
    from pxr import Usd,UsdGeom,UsdPhysics,PhysxSchema
    from isaaclab.sim.utils.stage import get_current_stage
    from scipy.spatial.transform import Rotation
    stage=get_current_stage();base=env.unwrapped
    food_spec=next(o for o in ns['OBJECTS'] if o.name=='belgian_waffle')
    assert abs(food_spec.mass-BATTER_MASS_KG)<1e-10
    # Canonical molded state. This is the declared abstraction, not a physical
    # reconstruction of the particle shape or conservation of its mass distribution.
    origin=axis-rot@np.array([0.,0.,ns['ZA']]);quat=Rotation.from_matrix(rot).as_quat()
    dest=Path(os.environ.get('TASK_TRACE_PATH','trace.jsonl')).parent
    np.savez_compressed(dest/'bake-swap-batter.npz',positions=points,positions_mold=local,
                        mass_per_particle=BATTER_MASS_KG/len(points))
    a=base.scene['belgian_waffle'];offset=ns['labkit']._np(base.scene.env_origins[0]);dev=base.scene.env_origins.device
    tensor=lambda v:torch.tensor([list(v)],dtype=torch.float32,device=dev)
    a.write_root_pose_to_sim_index(root_pose=tensor(np.r_[origin+offset,quat]))
    a.write_root_velocity_to_sim_index(root_velocity=tensor([0.,0.,0.,0.,0.,0.]))
    PhysxSchema.PhysxParticleSystem(stage.GetPrimAtPath(paths['system'])).CreateParticleSystemEnabledAttr(False)
    for key in ('system','particles'):
        for child in Usd.PrimRange(stage.GetPrimAtPath(paths[key])):
            if child.IsA(UsdGeom.Imageable):UsdGeom.Imageable(child).MakeInvisible()
    for p in Usd.PrimRange(stage.GetPrimAtPath(paths['cooked_path'])):
        if str(p.GetPath()) in paths['active_collider_paths']:UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Set(True)
        if p.IsA(UsdGeom.Imageable):UsdGeom.Imageable(p).MakeVisible()
        if p.HasAPI(UsdPhysics.RigidBodyAPI):PhysxSchema.PhysxRigidBodyAPI(p).CreateDisableGravityAttr(False)
    event=dict(type='user-authorized abstract bake-state swap',robot_time_s=clock.robot_time_s,
               bake_clock=clock.report(),particle_count=len(points),nominal_mass_kg=BATTER_MASS_KG,
               rigid_origin=origin.tolist(),rigid_quat_xyzw=quat.tolist(),
               physical_phase_change_validated=False,benchmark_eligible=TIME_SCALE==1.,
               limitations='Canonical solid replaces in-cavity batter at a timer trigger; internal mass distribution and cooking chemistry are not simulated.')
    (dest/'bake-state-swap.json').write_text(json.dumps(event,indent=2)+'\n')
    STATE.update(solid=True,event=event,block_reason=None)
    print('BAKE_STATE_SWAP '+json.dumps(event),flush=True)
