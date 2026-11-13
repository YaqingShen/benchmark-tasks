"""Isaac/YAM asset-loading diagnostic. This cannot validate or score the task.

Run in the existing Linux Isaac environment, with an idle GPU. Starts the
native bimanual scene, holds the arms still, steps 5 seconds and records views.
No active object poses are overwritten after reset. The current exported cap
and pipette mechanisms are incomplete, so this must precede full integration.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cpu')
    p.add_argument('--assets',default='assets/native')
    p.add_argument('--steps',type=int,default=100)
    p.add_argument('--physics-hz',type=int,default=240)
    p.add_argument('--coupled',action='store_true',help='Diagnostic liquid/mechanism coupling; not task scoring')
    p.add_argument('--probe',choices=['pipette_A','pipette_B','bottle_A_cap','bottle_B_cap','bottle_A'])
    p.add_argument('--gripper-force-n',type=float,default=5.)
    p.add_argument('--camera-width',type=int,default=640)
    p.add_argument('--antialiasing',choices=['DLAA','FXAA'],default='DLAA')
    p.add_argument('--probe-limit-s',type=float,help='Stop early for diagnosis; cannot pass the complete probe')
    p.add_argument('--balance-display-check',action='store_true',
                   help='Known-load display calibration only; not robot liquid transfer')
    p.add_argument('--cap-torque-check',action='store_true',
                   help='External torque on cap with fixed bottle; mechanism diagnostic only')
    p.add_argument('--wet-pipette-check',action='store_true',
                   help='Preload beaker for isolated wet pipette test; not full task')
    p.add_argument('--wet-cylinder-check',action='store_true',
                   help='Use 10.8 mL A or 30.8 mL B in its cylinder for isolated pipette aspiration and dispensing')
    p.add_argument('--cap-cycle-check',action='store_true',help='Include recapping and released closed-state checks in a cap probe')
    p.add_argument('--two-hand-cap-check',action='store_true',help='Hold bottle A with the right hand while the left hand unscrews its cap')
    p.add_argument('--cap-park-check',action='store_true',help='Extend two-hand opening with cap handoff and released inside-up parking')
    p.add_argument('--bottle-pour-check',action='store_true',help='Isolated bottle-A mass delivery with its cap initially parked; not a full task')
    p.add_argument('--pipette-correction-check',choices=['add','remove'],help='Isolated 9.5/10.5 g pipette correction; cap already parked for add case')
    p.add_argument('--pipette-disposal-check',action='store_true',help='Isolate pickup and bin placement; no dose correction claim')
    p.add_argument('--initial-mass-g',type=float,help='Explicit initial mass for a correction diagnostic')
    p.add_argument('--seed',type=int,help='Seed for a randomized correction reset')
    p.add_argument('--share-gpu',action='store_true',help='User-authorized concurrent run; preserve other GPU jobs')
    p.add_argument('--no-cameras',action='store_true',help='Numeric mechanism diagnostic without RTX image rendering')
    p.add_argument('--cap-free-check',action='store_true',help='Torque control test with cap initially raised and gravity disabled')
    p.add_argument('--cap-frictionless-check',action='store_true',help='Isolate cap geometry by disabling bottle/cap contact friction in this diagnostic')
    p.add_argument('--pgs',action='store_true',help='Compare PGS contact solver in a labelled diagnostic')
    p.add_argument('--cap-no-threads-check',action='store_true',help='Omit thread geometry only for an isolation test')
    p.add_argument('--cap-ignore-contact',action='append',choices=['seal','collar'],default=[],
                   help='Disable one seating contact group in a labelled cap isolation diagnostic')
    p.add_argument('--fix-bottle-for-probe',action='store_true',
                   help='Diagnostic fixture only; never a full task demonstration')
    p.add_argument('--lock',type=Path,default=Path('/tmp/robocurve-shared-gpu.lock'))
    a=p.parse_args()
    if a.cap_park_check:a.two_hand_cap_check=True
    if a.pipette_correction_check and (a.probe!='pipette_A' or not a.coupled):
        raise SystemExit('Pipette correction requires --probe pipette_A and --coupled')
    if a.pipette_disposal_check and not a.pipette_correction_check:
        raise SystemExit('Disposal isolation requires the pipette correction scene')
    if (a.initial_mass_g is not None or a.seed is not None) and not a.pipette_correction_check:
        raise SystemExit('Initial mass/seed requires the correction diagnostic')
    if a.initial_mass_g is not None and a.seed is not None:raise SystemExit('Choose initial mass or seed, not both')
    if a.bottle_pour_check and (a.probe!='bottle_A' or not a.coupled):
        raise SystemExit('Bottle delivery requires --probe bottle_A and --coupled.')
    if a.probe=='bottle_A' and not a.bottle_pour_check:
        raise SystemExit('Use --bottle-pour-check for the bottle delivery diagnostic.')
    if a.wet_cylinder_check:a.wet_pipette_check=True
    if a.balance_display_check and (not a.coupled or a.probe):
        raise SystemExit('Balance calibration requires --coupled and no robot probe.')
    if a.cap_torque_check and (a.probe or a.coupled or a.balance_display_check):
        raise SystemExit('Torque diagnostic must run alone, without liquid coupling.')
    if a.cap_free_check and not a.cap_torque_check:raise SystemExit('Free cap control requires the torque diagnostic')
    if a.cap_frictionless_check and not a.cap_torque_check:raise SystemExit('Friction isolation requires the torque diagnostic')
    if a.cap_no_threads_check and not a.cap_torque_check:raise SystemExit('Thread isolation requires the torque diagnostic')
    if a.cap_ignore_contact and not a.cap_torque_check:raise SystemExit('Contact isolation requires the torque diagnostic')
    if a.wet_pipette_check and (a.probe not in ('pipette_A','pipette_B') or not a.coupled):
        raise SystemExit('Wet test requires --probe pipette_A or pipette_B, with --coupled.')
    if a.cap_cycle_check and (not a.probe or not a.probe.endswith('_cap')):
        raise SystemExit('Cap cycle requires a cap robot probe.')
    if a.two_hand_cap_check and (a.probe!='bottle_A_cap' or a.cap_cycle_check or a.fix_bottle_for_probe):
        raise SystemExit('Two-hand check requires bottle_A_cap, without a fixture or cycle flag.')
    if a.fix_bottle_for_probe and (not a.probe or not a.probe.endswith('_cap')):
        raise SystemExit('The diagnostic bottle fixture is only for a cap probe.')
    if a.steps<1 or a.physics_hz<100 or a.physics_hz%20:
        raise SystemExit('Steps must be positive; physics Hz must be a multiple of 20 and at least 100.')
    root=Path(__file__).resolve().parent
    assets=root/a.assets
    manifest=json.loads((assets/'native-assets.json').read_text())
    if a.probe and a.probe not in manifest['assets']:raise SystemExit('The requested probe object is absent from this task revision.')
    if a.wet_cylinder_check and 'cylinder_'+a.probe[-1] not in manifest['assets']:
        raise SystemExit('This revision has no graduated cylinders; use an archived cylinder revision for that diagnostic.')
    if a.output.exists():raise SystemExit('Use a new evidence directory.')
    for name,spec in manifest['assets'].items():
        asset=assets/spec['path']
        if hashlib.sha256(asset.read_bytes()).hexdigest()!=spec['sha256']:
            raise SystemExit('Asset hash mismatch: '+name)
    a.lock.parent.mkdir(parents=True,exist_ok=True)
    lock=a.lock.open('a')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise SystemExit('Shared GPU lock is occupied; existing work is untouched.')
    jobs=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader'],capture_output=True,text=True,timeout=15)
    active=[]
    for line in jobs.stdout.splitlines():
        if not line.strip():continue
        pid=line.split(',')[0].strip()
        try:exe=str((Path('/proc')/pid/'exe').resolve(strict=True))
        except OSError:exe=''
        if exe!='/usr/libexec/gnome-remote-desktop-daemon':active.append(line)
    if jobs.returncode or (active and not a.share_gpu):
        raise SystemExit('GPU is occupied or cannot be checked; existing work is untouched.\n'+'\n'.join(active))
    if a.share_gpu:
        free=subprocess.run(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=15)
        if free.returncode or min(float(x)for x in free.stdout.splitlines())<6000:
            raise SystemExit('Concurrent test needs at least 6 GB free GPU memory.')
    a.output.mkdir(parents=True)
    os.environ['ENABLE_CAMERAS']='0' if a.no_cameras else '1'
    app=None
    env=None
    coupling=None
    probe=None
    try:
        from isaaclab.app import AppLauncher
        app=AppLauncher(headless=True,enable_cameras=not a.no_cameras,device=a.device).app
        import numpy as np
        import torch
        from PIL import Image
        from isaaclab.actuators import ImplicitActuatorCfg
        from isaaclab.sensors import ContactSensorCfg
        from isaaclab.sim import RigidBodyPropertiesCfg
        from isaaclab_arena.assets.object import Object
        from isaaclab_arena.assets.object_type import ObjectType
        from isaaclab_arena.utils.pose import Pose
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
        from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
        from yam_arena import constants,station
        from yam_arena.embodiment import BimanualYamEmbodiment
        import yam_arena.scene as scenes
        if abs(station.TABLE_TOP-manifest['table_top_m'])>1e-6:
            raise RuntimeError('Native station table height differs from exported asset poses.')
        embodiment=BimanualYamEmbodiment(enable_cameras=not a.no_cameras,rig=station.rig(),cam_height=round(a.camera_width*352/640),cam_width=a.camera_width,
                                       robot_contact_offset=.0001,contact_sensors=True)
        if not 0<a.gripper_force_n<=50:raise ValueError('Gripper force must be within the installed hardware limit')
        for side in ('left_robot','right_robot'):
            getattr(embodiment.scene_config,side).actuators['gripper'].joint_effort_limit=a.gripper_force_n
        objects=[]
        for name,spec in manifest['assets'].items():
            pose_spec=spec
            if (a.bottle_pour_check or a.pipette_correction_check=='add') and name=='bottle_A_cap':
                pose_spec=dict(spec)
                rest=manifest['assets']['station_fixtures']['cap_rest_centers']['A']
                pose_spec['position_xyz']=[rest[0],rest[1],rest[2]+.0275]
                pose_spec['rotation_xyzw']=[1.,0.,0.,0.]
            if a.cap_free_check and name=='bottle_A_cap':
                pose_spec=dict(spec);pose_spec['position_xyz']=list(spec['position_xyz'])
                pose_spec['position_xyz'][2]+=.05
            obj=Object(name=name,usd_path=str(assets/spec['path']),
                 initial_pose=Pose.from_dict(pose_spec),
                 object_type=getattr(ObjectType,spec.get('object_type','RIGID')))
            if (a.fix_bottle_for_probe and name==a.probe.removesuffix('_cap')) or (a.cap_torque_check and name=='bottle_A'):
                obj.object_cfg.spawn.rigid_props=RigidBodyPropertiesCfg(kinematic_enabled=True)
                obj.disable_reset_pose()
            if not spec.get('dynamic',True):obj.disable_reset_pose()
            if a.cap_free_check and name=='bottle_A_cap':
                obj.object_cfg.spawn.rigid_props=RigidBodyPropertiesCfg(disable_gravity=True)
            if (a.cap_frictionless_check or a.cap_no_threads_check or a.cap_ignore_contact) and name in ('bottle_A','bottle_A_cap'):
                original_spawn=obj.object_cfg.spawn.func
                def frictionless_spawn(*args,_spawn=original_spawn,**kwargs):
                    from pxr import Usd,UsdPhysics
                    prim=_spawn(*args,**kwargs)
                    for part in Usd.PrimRange(prim):
                        if a.cap_frictionless_check and part.HasAPI(UsdPhysics.MaterialAPI):
                            mat=UsdPhysics.MaterialAPI(part)
                            mat.CreateStaticFrictionAttr(0.);mat.CreateDynamicFrictionAttr(0.)
                    # Remove geometry entirely for a decisive isolation test;
                    # preserve the immutable production assets on disk.
                    suffixes=(['threads'] if a.cap_no_threads_check else [])
                    suffixes+=['inner_seal' if s=='seal' else s for s in a.cap_ignore_contact]
                    for suffix in suffixes:
                        child=prim.GetChild(suffix)
                        if child.IsValid():child.SetActive(False)
                    return prim
                obj.object_cfg.spawn.func=frictionless_spawn
            if spec.get('passive_springs'):
                obj.object_cfg.actuators={s['joint_name']+'_spring':ImplicitActuatorCfg(
                    joint_names_expr=[s['joint_name']],stiffness=s['stiffness'],damping=s['damping'],
                    joint_effort_limit=s['effort_limit'],joint_velocity_limit=s['velocity_limit'])
                    for s in spec['passive_springs']}
                obj.object_cfg.init_state.joint_pos={s['joint_name']:0. for s in spec['passive_springs']}
                obj.object_cfg.init_state.joint_vel={s['joint_name']:0. for s in spec['passive_springs']}
            objects.append(obj)
        standard_configure=scenes.env_cfg_callback(20)
        by_name={o.name:o for o in objects}
        balance_contacts=ContactSensorCfg(prim_path=by_name['beaker'].get_contact_sensor_prim_path(),
            filter_prim_paths_expr=[by_name['digital_balance'].get_contact_sensor_prim_path()],
            update_period=0.,history_length=1,track_air_time=False,max_contact_data_count_per_prim=256) if 'digital_balance' in by_name else None
        pipette_contacts={}
        if a.pipette_correction_check:
            filters=[by_name['pipette_A'].prim_path+'/'+part for part in ('stem','bulb_left','bulb_right')]
            for name in ('tip_left','tip_right'):
                pipette_contacts['pipette_'+name+'_contacts']=ContactSensorCfg(
                    prim_path=embodiment.scene_config.left_robot.prim_path+embodiment.link_paths[name],
                    filter_prim_paths_expr=filters,update_period=0.,history_length=1,
                    track_air_time=False,max_contact_data_count_per_prim=256)
        cap_name=a.probe if a.probe and a.probe.endswith('_cap') else 'bottle_A_cap'
        cap_contacts=ContactSensorCfg(prim_path=by_name[cap_name].get_contact_sensor_prim_path(),
            filter_prim_paths_expr=[by_name[cap_name.removesuffix('_cap')].get_contact_sensor_prim_path()],
            update_period=0.,history_length=1,track_air_time=False,max_contact_data_count_per_prim=1024)
        if a.two_hand_cap_check or a.bottle_pour_check:
            bottle_hold_contacts=ContactSensorCfg(prim_path=by_name['bottle_A'].get_contact_sensor_prim_path(),
                filter_prim_paths_expr=[embodiment.scene_config.right_robot.prim_path+embodiment.link_paths[n] for n in ('tip_left','tip_right')],
                update_period=0.,history_length=1,track_air_time=False,max_contact_data_count_per_prim=1024)
        if a.cap_park_check:
            cap_right_contacts=ContactSensorCfg(prim_path=by_name[cap_name].get_contact_sensor_prim_path(),
                filter_prim_paths_expr=[embodiment.scene_config.right_robot.prim_path+embodiment.link_paths[n] for n in ('tip_left','tip_right')],
                update_period=0.,history_length=1,track_air_time=False,max_contact_data_count_per_prim=1024)
            cap_rest_contacts=ContactSensorCfg(prim_path=by_name[cap_name].get_contact_sensor_prim_path(),
                filter_prim_paths_expr=[by_name['station_fixtures'].get_contact_sensor_prim_path()],
                update_period=0.,history_length=1,track_air_time=False,max_contact_data_count_per_prim=1024)
        def configure(cfg):
            cfg=standard_configure(cfg)
            cfg.sim.dt=1./a.physics_hz;cfg.decimation=a.physics_hz//20
            cfg.sim.render_interval=cfg.decimation
            if balance_contacts is not None:cfg.scene.balance_contacts=balance_contacts
            for name,sensor in pipette_contacts.items():setattr(cfg.scene,name,sensor)
            cfg.scene.cap_contacts=cap_contacts
            if a.two_hand_cap_check or a.bottle_pour_check:cfg.scene.bottle_hold_contacts=bottle_hold_contacts
            if a.cap_park_check:
                cfg.scene.cap_right_contacts=cap_right_contacts;cfg.scene.cap_rest_contacts=cap_rest_contacts
                cfg.episode_length_s=150.
            if a.pipette_correction_check:cfg.episode_length_s=600.
            if a.pgs:
                cfg.sim.physics.solver_type=0
                cfg.sim.physics.solve_articulation_contact_last=False
            return cfg
        description=IsaacLabArenaEnvironment(name='formulation_asset_check',
            scene=Scene([*scenes.task_room('station',embodiment.rig.top_cam),*objects]),
            embodiment=embodiment,env_cfg_callback=configure)
        env=ArenaEnvBuilder(description,ArenaEnvBuilderCfg(num_envs=1,solve_relations=False,device=a.device)).make_registered()
        env.reset();base=env.unwrapped
        if not a.no_cameras:
            # Render at native resolution: DLSS upscaling merged tiny lit
            # segments at 640 px. Keep the original camera poses and fields
            # of view; FXAA is available as a raw diagnostic comparison.
            import omni.replicator.core as rep
            rep.settings.set_render_rtx_realtime(antialiasing=a.antialiasing)
        def array(value):
            value=value.torch if hasattr(value,'torch') else value
            return value.detach().cpu().numpy() if hasattr(value,'detach') else np.asarray(value)
        def poses():
            return {name:{'position':array(base.scene[name].data.root_pos_w)[0].tolist(),
                          'quaternion_xyzw':array(base.scene[name].data.root_quat_w)[0].tolist()}
                    for name in manifest['assets']}
        initial=poses();arms=[]
        for name in ('left_robot','right_robot'):
            arm=base.scene[name]
            arm_ids,found=arm.find_joints(list(constants.ARM_JOINTS),preserve_order=True)
            finger_ids,_=arm.find_joints(list(constants.FINGER_JOINTS),preserve_order=True)
            if list(found)!=list(constants.ARM_JOINTS):raise RuntimeError('Unexpected robot joint order.')
            q=array(arm.data.joint_pos)[0]
            arms.append(np.r_[q[arm_ids],np.clip(q[finger_ids].mean()/constants.FINGER_OPEN,0,1)])
        command=torch.tensor([np.concatenate(arms).tolist()],device=base.device,dtype=torch.float32)
        passive={}
        for name,spec in manifest['assets'].items():
            if spec.get('passive_springs'):
                part=base.scene[name]
                ids,found=part.find_joints([s['joint_name'] for s in spec['passive_springs']],preserve_order=True)
                target=torch.zeros((1,len(ids)),device=base.device)
                part.actuators.target_command.set_position_index(value=target,joint_ids=list(ids))
                part.write_data_to_sim()
                passive[name]=(part,ids)
        if a.coupled:
            from native_coupling import NativeCoupling
            coupling=NativeCoupling(base,manifest)
            coupling.active=True
        for _ in range(a.steps):env.step(command)
        if a.pipette_correction_check:
            # Reset preparation: retain the measured EMPTY beaker tare, then
            # set up the visible starting dose before any robot task action.
            if coupling.balance.tare_g is None:raise RuntimeError('Empty-beaker tare did not settle during reset')
            initial_mass=9.5 if a.pipette_correction_check=='add' else 10.5
            if a.initial_mass_g is not None:initial_mass=a.initial_mass_g
            elif a.seed is not None:
                from reset_cases import starting_mass
                initial_mass=starting_mass(a.seed)
            if not 8.5<=initial_mass<=11.5 or abs(initial_mass-10.)<.15-1e-8:
                raise ValueError('Starting dose must be 8.50–11.50 g and at least 0.15 g from target')
            if (initial_mass<10.)!=(a.pipette_correction_check=='add'):
                raise ValueError('Diagnostic direction does not match the initial dose')
            coupling.ledger._receive('beaker',coupling.ledger.take_from('bottle_A',initial_mass/1.03))
            coupling.ledger.initial_beaker_g=initial_mass
            coupling.ledger.initial_error_g=abs(initial_mass-10.)
            for _ in range(40):
                if not a.no_cameras:coupling.render()
                env.step(command)
        camera_names=('top_cam','left_cam','right_cam')
        if not a.no_cameras:
            if coupling is not None:coupling.render()
            for _ in range(4):base.sim.render()
            rgb=array(base.scene['top_cam'].data.output['rgb'])[0,:,:,:3]
            Image.fromarray(rgb.astype(np.uint8)).save(a.output/'setup.png')
            if coupling is not None and coupling.balance is not None:
                (a.output/'setup-readout.json').write_text(json.dumps({
                    'balance':coupling.balance.last,'authored_digits':coupling.balance.screen.previous,
                    'antialiasing':a.antialiasing},indent=2))
        if a.wet_pipette_check:
            ingredient=a.probe[-1]
            source='cylinder_'+ingredient if a.wet_cylinder_check else 'beaker'
            amount=(10.8 if ingredient=='A' else 30.8) if a.wet_cylinder_check else 40.
            coupling.ledger._receive(source,coupling.ledger.take_from('bottle_'+ingredient,amount))
            for _ in range(40):env.step(command)
        if a.cap_torque_check:
            import omni.usd
            from pxr import Usd,UsdPhysics,UsdGeom,UsdUtils,PhysicsSchemaTools
            from omni.physx import get_physx_cooking_interface
            stage=omni.usd.get_context().get_stage()
            cap=base.scene['bottle_A_cap'];samples=[];schema={}
            for name in ('bottle_A','bottle_A_cap'):
                prim=stage.GetPrimAtPath('/World/envs/env_0/'+name)
                schema[name]={'kinematic':UsdPhysics.RigidBodyAPI(prim).GetKinematicEnabledAttr().Get(),
                              'mass':UsdPhysics.MassAPI(prim).GetMassAttr().Get(),
                              'native_inertia':array(base.scene[name].data.body_inertia).tolist(),
                              'native_com':array(base.scene[name].data.body_com_pose_b).tolist()}
                schema[name]['materials']={str(p.GetPath()):{'static':UsdPhysics.MaterialAPI(p).GetStaticFrictionAttr().Get(),
                    'dynamic':UsdPhysics.MaterialAPI(p).GetDynamicFrictionAttr().Get()}
                    for p in Usd.PrimRange(prim) if p.HasAPI(UsdPhysics.MaterialAPI)}
                schema[name]['runtime_colliders']=[str(p.GetPath()) for p in Usd.PrimRange(prim) if p.HasAPI(UsdPhysics.CollisionAPI)]
                schema[name]['disabled_colliders']=[str(p.GetPath()) for p in Usd.PrimRange(prim)
                    if p.HasAPI(UsdPhysics.CollisionAPI) and not UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get()]
                for getter in ('get_contact_offsets','get_rest_offsets'):
                    view=base.scene[name].root_view
                    if hasattr(view,getter):
                        values=getattr(view,getter)();values=values.numpy() if hasattr(values,'numpy') else array(values)
                        schema[name][getter]=np.unique(values).tolist()
                view=base.scene[name].root_view
                if hasattr(view,'get_material_properties'):
                    values=view.get_material_properties()
                    values=values.numpy() if hasattr(values,'numpy') else array(values)
                    schema[name]['native_material_properties']=np.unique(values.reshape(-1,3),axis=0).tolist()
            cooked={};stage_id=UsdUtils.StageCache.Get().GetId(stage).ToLongInt()
            for name,suffix in [('bottle_A','threads/segment_000'),('bottle_A','collisions/neck_0'),
                                ('bottle_A_cap','threads/segment_000'),('bottle_A_cap','collisions/wall_0')]:
                path='/World/envs/env_0/'+name+'/'+suffix
                prim=stage.GetPrimAtPath(path)
                if not prim.IsValid() or not prim.IsActive():continue
                entry={'authored_vertices':[list(p)for p in UsdGeom.Mesh(prim).GetPointsAttr().Get()]}
                def receive(result,convexes):
                    entry['result']=str(result)
                    entry['cooked_vertices']=[[[v.x,v.y,v.z]for v in c.vertices]for c in convexes]
                get_physx_cooking_interface().request_convex_collision_representation(
                    stage_id=stage_id,collision_prim_id=PhysicsSchemaTools.sdfPathToInt(path),
                    run_asynchronously=False,on_result=receive)
                cooked[path]=entry
            (a.output/'cooked-cap-geometry.json').write_text(json.dumps(cooked,indent=2))
            for torque in (0.,.001,.005,.02):
                cap.permanent_wrench_composer.set_forces_and_torques_index(
                    forces=torch.zeros((1,1,3),device=base.device),
                    torques=torch.tensor([[[0.,0.,torque]]],device=base.device),is_global=True)
                for step in range(20):
                    env.step(command)
                    samples.append({'torque_nm':torque,'step':step,
                                    'pose':array(cap.data.root_link_pose_w)[0].tolist(),
                                    'velocity':array(cap.data.root_link_vel_w)[0].tolist(),
                                    'composer_active':bool(cap.permanent_wrench_composer.active),
                                    'composed_torque_body':array(cap.permanent_wrench_composer.out_torque_b).tolist(),
                                    'bottle_contact_normal_n':array(base.scene['cap_contacts'].data.normal_force_matrix_w).tolist()})
            (a.output/'cap-torque-diagnostic.json').write_text(json.dumps({
                'type':'EXTERNALLY_LOADED_MECHANISM_DIAGNOSTIC','full_task_demo':False,
                'fixed_bottle':True,'free_cap_control':a.cap_free_check,'frictionless_control':a.cap_frictionless_check,
                'threads_removed_control':a.cap_no_threads_check,'ignored_contact_groups':a.cap_ignore_contact,
                'pgs_solver':a.pgs,'task_score':None,'schema':schema,'samples':samples},indent=2))
        if a.balance_display_check:
            if coupling.balance is None:raise RuntimeError('Native balance unavailable')
            readings=[]
            # Move ledger liquid only for a labelled instrument calibration.
            # This is never a robot demo and cannot produce a task score.
            loads=[('empty',None,0.),('A_10g','A',10.)]
            if 'B' in coupling.ledger.cfg['targets_g']:loads.append(('A10_B30g','B',30.))
            for tag,ingredient,mass_g in loads:
                if ingredient:
                    if ingredient=='B':coupling.ledger.checkpoint_A()
                    amount=mass_g/coupling.ledger.cfg['density_g_ml'][ingredient]
                    coupling.ledger._receive('beaker',coupling.ledger.take_from('bottle_'+ingredient,amount))
                for step in range(80):
                    env.step(command)
                    if not a.no_cameras:coupling.render()
                    if step%10==0:readings.append({'phase':tag,**coupling.balance.last})
                if not a.no_cameras:
                    for _ in range(3):base.sim.render()
                    rgb=array(base.scene['top_cam'].data.output['rgb'])[0,:,:,:3]
                    Image.fromarray(rgb.astype(np.uint8)).save(a.output/('balance_'+tag+'.png'))
            (a.output/'balance-calibration.json').write_text(json.dumps({
                'type':'KNOWN_LOAD_INSTRUMENT_CALIBRATION','full_task_demo':False,'task_score':None,
                'load_source':'diagnostic transfer in conserved liquid ledger; no robot delivery',
                'readings':readings},indent=2))
        if a.probe:
            from native_probes import PipetteProbe,CapProbe,PipetteWetProbe,CapCycleProbe
            if a.probe.startswith('pipette_') and not manifest['assets'][a.probe].get('passive_springs'):
                raise RuntimeError('A squeeze probe requires the articulated pipette asset')
            from bimanual_cap import TwoHandCapProbe
            from cap_parking import CapParkingProbe
            from bottle_pour import BottlePourProbe
            from pipette_correction import PipetteCorrectionProbe,PipetteDisposalProbe
            correction_class=PipetteDisposalProbe if a.pipette_disposal_check else PipetteCorrectionProbe
            probe=(correction_class(base,array(command)[0],coupling,a.pipette_correction_check) if a.pipette_correction_check else
                   BottlePourProbe(base,array(command)[0],coupling) if a.bottle_pour_check else
                   CapParkingProbe(base,array(command)[0],a.probe,manifest) if a.cap_park_check else
                   TwoHandCapProbe(base,array(command)[0],a.probe) if a.two_hand_cap_check else
                   PipetteWetProbe(base,array(command)[0],a.probe,coupling,source='cylinder_'+a.probe[-1] if a.wet_cylinder_check else 'beaker') if a.wet_pipette_check else
                   (PipetteProbe if a.probe.startswith('pipette_') else (CapCycleProbe if a.cap_cycle_check else CapProbe))(base,array(command)[0],tool=a.probe))
            frames=a.output/'camera_frames'
            if not a.no_cameras:
                for name in camera_names:(frames/name).mkdir(parents=True)
            probe_steps=round(min(probe.duration,a.probe_limit_s if a.probe_limit_s is not None else probe.duration)*20)
            for step in range(probe_steps):
                t=(step+1)/20
                action,label=probe.command(t)
                env.step(torch.tensor([action.tolist()],device=base.device,dtype=torch.float32))
                if coupling is not None and not a.no_cameras:coupling.render()
                probe.observe(t,label)
                if step%4==0 and not a.no_cameras:
                    for name in camera_names:
                        rgb=array(base.scene[name].data.output['rgb'])[0,:,:,:3]
                        Image.fromarray(rgb.astype(np.uint8)).save(frames/name/f'frame_{step//4:05d}.png')
                if step%20==0:
                    (a.output/'probe-progress.json').write_text(json.dumps(probe.rows[-1],indent=2))
                if t>=probe.duration-1e-8:
                    probe_steps=step+1
                    break
            result=probe.result()
            if a.probe_limit_s is not None and a.probe_limit_s<probe.duration:
                result.update(passed_probe=False,early_diagnostic_stop=True)
            (a.output/'mechanism-probe.json').write_text(json.dumps(result,indent=2))
            if not a.no_cameras:
                (a.output/'camera-recording.json').write_text(json.dumps({
                    'cameras':camera_names,'fps':5,'source':'original station cameras',
                    'overlays':False,'custom_cameras':False,'cover':'setup.png',
                    'full_task_demo':False,'task_score':None,
                    'asset_manifest_sha256':hashlib.sha256((assets/'native-assets.json').read_bytes()).hexdigest()
                },indent=2))
        if coupling is not None and not a.no_cameras:coupling.render()
        if not a.no_cameras:
            for _ in range(4):base.sim.render()
            for name in ('top_cam','left_cam','right_cam'):
                rgb=array(base.scene[name].data.output['rgb'])[0,:,:,:3]
                Image.fromarray(rgb.astype(np.uint8)).save(a.output/(name+'.png'))
        final=poses()
        finite=all(np.isfinite(v['position']+v['quaternion_xyzw']).all() for v in final.values())
        report={'status':'ASSET_SCENE_CHECK_COMPLETED' if finite else 'NONFINITE_STATE',
          'scope':'Asset loading and short native stepping only; not full task validation.',
          'full_task_ready':False,'task_score':None,'dream1_run':False,'post_reset_pose_writes':0,
          'control_steps':a.steps+(probe_steps if probe is not None else 0)
                          +(40 if a.wet_pipette_check else 0)+(80 if a.cap_torque_check else 0)
                          +(240 if a.balance_display_check else 0),
          'idle_control_steps':a.steps,'control_hz':20,'physics_hz':a.physics_hz,'initial_poses':initial,'final_poses':final,
          'configured_gripper_force_limit_n':a.gripper_force_n,
          'diagnostic_fixed_bottle':a.probe.removesuffix('_cap') if a.fix_bottle_for_probe else None,
          'diagnostic_balance_calibration':a.balance_display_check,
          'diagnostic_cap_torque':a.cap_torque_check,
          'diagnostic_cap_open_close_cycle':a.cap_cycle_check,
          'diagnostic_two_hand_cap_opening':a.two_hand_cap_check,
          'diagnostic_bottle_mass_delivery':a.bottle_pour_check,
          'diagnostic_pipette_mass_correction':a.pipette_correction_check,
          'initial_mass_g':coupling.ledger.initial_beaker_g if a.pipette_correction_check else None,
          'reset_seed':a.seed,
          'diagnostic_wet_pipette':a.wet_pipette_check,
          'user_authorized_gpu_sharing':a.share_gpu,
          'cameras_enabled':not a.no_cameras,
          'drift_m':{k:float(np.linalg.norm(np.array(final[k]['position'])-initial[k]['position'])) for k in final},
          'passive_joint_positions':{k:array(part.data.joint_pos)[0,ids].tolist() for k,(part,ids) in passive.items()},
          'native_device':str(base.device),'asset_manifest_sha256':hashlib.sha256((assets/'native-assets.json').read_bytes()).hexdigest(),
          'remaining':manifest['limitations']}
        if coupling is not None:
            report['coupling_diagnostic']={'physics_updates':coupling.steps,'last':coupling.last,
                'limitations':['Full task scoring is pending; liquid mass distribution is quasi-static and uncalibrated.']}
        if probe is not None:report['probe']={k:v for k,v in result.items() if k!='rows'}
        (a.output/'native-scene-check.json').write_text(json.dumps(report,indent=2))
        print(json.dumps({k:report[k] for k in ('status','scope','full_task_ready','task_score')}),flush=True)
        if not finite:raise RuntimeError('Nonfinite native object state.')
    except Exception as exc:
        details=traceback.format_exc()
        (a.output/'error.json').write_text(json.dumps({'status':'ERROR','error':type(exc).__name__+': '+str(exc),
            'traceback':details,'full_task_ready':False},indent=2))
        print(details,file=sys.stderr,flush=True)
        if probe is not None:
            partial=probe.result();partial['error']=str(exc)
            partial['passed_probe']=False
            if 'passed_lift_and_squeeze' in partial:partial['passed_lift_and_squeeze']=False
            (a.output/'mechanism-probe.json').write_text(json.dumps(partial,indent=2))
        raise
    finally:
        if coupling is not None:coupling.close()
        if env is not None:env.close()
        if app is not None:app.close()
        fcntl.flock(lock,fcntl.LOCK_UN)


if __name__=='__main__':main()
