"""Static scene capture on Isaac/5090. Not a task demonstration or charging test."""
import argparse,json,os,fcntl,subprocess
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--steps',type=int,default=60)
    p.add_argument('--airpods-fit-check',action='store_true',help='Initialize the free case above the recess, then let contacts and gravity settle it; not a robot placement demo.')
    p.add_argument('--share-gpu',action='store_true',help='Permit a bounded preview alongside existing jobs when at least 16 GB is free.')
    a=p.parse_args();root=Path(__file__).resolve().parent
    if a.output.exists():raise SystemExit('Use a fresh output directory.')
    lock=open('/tmp/robocurve-shared-gpu.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    jobs=subprocess.check_output(['nvidia-smi','--query-compute-apps=process_name','--format=csv,noheader'],text=True)
    active=any(x.strip() and 'gnome-remote-desktop' not in x for x in jobs.splitlines())
    if active:
        free=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True)
        if not a.share_gpu or min(map(float,free.splitlines()))<16000:raise SystemExit('GPU occupied; existing job untouched.')
    a.output.mkdir(parents=True);os.environ['ENABLE_CAMERAS']='1'
    from isaaclab.app import AppLauncher
    app=AppLauncher(headless=True,enable_cameras=True,device='cpu').app
    env=None
    try:
        import numpy as np, torch
        from PIL import Image
        from isaaclab.actuators import ImplicitActuatorCfg
        from isaaclab_arena.assets.object import Object
        from isaaclab_arena.assets.object_type import ObjectType
        from isaaclab_arena.utils.pose import Pose
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
        from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
        from yam_arena import station,constants
        from yam_arena.embodiment import BimanualYamEmbodiment
        import yam_arena.scene as scenes
        # Only the light size/intensity changes; all three native camera mounts remain.
        scenes.STATION_DOME_INTENSITY=250.
        # Sphere-light radiance scales with emitting area. Preserve flux when
        # enlarging the source, instead of retaining point-like lamp intensity.
        scenes.STATION_LAMP_INTENSITY=1.8e4
        scenes.STATION_LAMP_RADIUS=.55
        embodiment=BimanualYamEmbodiment(enable_cameras=True,rig=station.rig(),cam_height=720,cam_width=1280,robot_contact_offset=.0001)
        from yam_arena.ik import solve,tcp_pose,matrix_to_quat_xyzw
        from yam_arena.urdf import quat_xyzw_to_matrix as native_quat_matrix
        _,neutral=tcp_pose(np.array(constants.READY))
        for side,sgn in [('left',1),('right',-1)]:
            angle=-sgn*.75;c,s=np.cos(angle),np.sin(angle)
            rz=np.array([[c,-s,0],[s,c,0],[0,0,1.]])
            # Pitch the physical wrists up slightly; do not alter camera mounts
            # or intrinsics. This keeps the full charger heads in both views.
            mount=native_quat_matrix(np.array(station.wrist_camera(side).rot_xyzw))
            c,s=np.cos(.18),np.sin(.18);up=np.array([[1.,0,0],[0,c,-s],[0,s,c]])
            orientation=rz@neutral@mount@up@mount.T
            q,ep,er=solve([.20,sgn*.25,1.04],matrix_to_quat_xyzw(orientation),constants.READY,station.BASES[side],iters=400,damping=.02)
            if ep>.002 or er>.02:raise RuntimeError(f'Preview pose IK failed: {side} {ep} {er}')
            getattr(embodiment.scene_config,side+'_robot').init_state.joint_pos.update(dict(zip(constants.ARM_JOINTS,q.tolist())))
        manifest=json.loads((root/'assets/manifest.json').read_text());objects=[]
        if a.airpods_fit_check:
            recess=manifest['geometry']['airpods_recess'];stand=manifest['assets']['belkin']['position_xyz']
            case=manifest['assets']['airpods_pro']
            case['position_xyz']=[stand[0]+recess['center_xy_m'][0]+.003,stand[1]+recess['center_xy_m'][1]+.002,stand[2]+recess['floor_z_m']+.0217/2+.008]
            yaw=np.pi/2+.08;case['rotation_xyzw']=[0,0,float(np.sin(yaw/2)),float(np.cos(yaw/2))]
        for name,spec in manifest['assets'].items():
            obj=Object(name=name,usd_path=str(root/'assets'/spec['path']),initial_pose=Pose.from_dict(spec),object_type=getattr(ObjectType,spec['object_type']))
            if name=='apple_watch':
                obj.object_cfg.actuators={'elastic_band':ImplicitActuatorCfg(joint_names_expr=['.*bend.*','.*twist.*'],stiffness=.0025,damping=.00015,joint_effort_limit=.025,joint_velocity_limit=15.)}
                obj.object_cfg.init_state.joint_pos={'.*':0.};obj.object_cfg.init_state.joint_vel={'.*':0.}
            objects.append(obj)
        standard=scenes.env_cfg_callback(20)
        def config(cfg):
            cfg=standard(cfg);cfg.sim.dt=.001;cfg.decimation=50;cfg.sim.render_interval=50;cfg.episode_length_s=30.;return cfg
        desc=IsaacLabArenaEnvironment(name='three_device_charge_preview',scene=Scene([*scenes.task_room('station',embodiment.rig.top_cam),*objects]),embodiment=embodiment,env_cfg_callback=config)
        env=ArenaEnvBuilder(desc,ArenaEnvBuilderCfg(num_envs=1,solve_relations=False,device='cpu')).make_registered()
        env.reset();base=env.unwrapped
        import omni.replicator.core as rep
        rep.settings.set_render_rtx_realtime(antialiasing='DLAA')
        def array(v):
            v=v.torch if hasattr(v,'torch') else v
            return v.detach().cpu().numpy() if hasattr(v,'detach') else np.asarray(v)
        cmd=[]
        for side in ['left_robot','right_robot']:
            arm=base.scene[side];ids,_=arm.find_joints(constants.ARM_JOINTS,preserve_order=True)
            cmd.extend([*array(arm.data.joint_pos)[0,ids],1.])
        command=torch.tensor([cmd],device=base.device,dtype=torch.float32)
        watch=base.scene['apple_watch']
        watch.actuators.target_command.set_position_index(value=torch.zeros((1,watch.num_joints),device=base.device),joint_ids=list(range(watch.num_joints)))
        tail=[]
        for i in range(a.steps):
            env.step(command)
            if i>=a.steps-20:tail.append(array(base.scene['airpods_pro'].data.root_link_pose_w)[0].copy())
            if i%20==0:print('Preview settling step',i,flush=True)
        for _ in range(15):base.sim.render()
        frames={}
        for name in ['top_cam','left_cam','right_cam']:
            frames[name]=array(base.scene[name].data.output['rgb'])[0,:,:,:3].astype(np.uint8)
            Image.fromarray(frames[name]).save(a.output/(name+'.png'))
        panel=Image.new('RGB',(1280,1080),'white');panel.paste(Image.fromarray(frames['top_cam']),(0,0))
        for name,x in [('left_cam',0),('right_cam',640)]:panel.paste(Image.fromarray(frames[name]).resize((640,360),Image.Resampling.LANCZOS),(x,720))
        panel.save(a.output/'three-camera-setup.png')
        metadata={'status':'static_native_isaac_preview','not_a_demo':True,'not_a_charging_validation':True,
          'engine':'Isaac Sim 6.1.0.0 / Isaac Lab 17.0.2','host':'5090','seed':17,'physics_hz':1000,
          'initial_battery_percent':{'iphone17':30,'airpods_pro_case':30,'airpods_pro_earbuds':30,'apple_watch':30},
          'watch_band_native_joint_count':watch.num_joints,'cameras':['top_cam','left_cam','right_cam'],
          'object_poses':{n:array(base.scene[n].data.root_link_pose_w)[0].tolist() for n in manifest['assets']},
          'post_reset_object_pose_writes':0,'standby_screens':'black; battery percent hidden','overlays':False}
        metadata['geometry']=manifest['geometry']
        if a.airpods_fit_check:
            from pxr import Usd,UsdGeom
            def quat_xyzw_to_matrix(q):
                x,y,z,w=np.asarray(q)/np.linalg.norm(q)
                return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                                 [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                                 [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
            # Evaluate the actual case collision mesh in the settled stand frame.
            case_pose=np.array(metadata['object_poses']['airpods_pro']);stand_pose=np.array(metadata['object_poses']['belkin'])
            rc=quat_xyzw_to_matrix(case_pose[3:]);rs=quat_xyzw_to_matrix(stand_pose[3:])
            stage=Usd.Stage.Open(str(root/'assets/airpods_pro.usda'))
            vertices=np.array(UsdGeom.Mesh(stage.GetPrimAtPath('/airpods_pro/shell')).GetPointsAttr().Get())
            local=(vertices@rc.T+case_pose[:3]-stand_pose[:3])@rs
            radius=np.linalg.norm(local[:,:2]-np.array(recess['center_xy_m']),axis=1).max()
            bottom_gap=local[:,2].min()-recess['floor_z_m']
            motion=np.linalg.norm(np.array(tail)[:,:3]-case_pose[:3],axis=1).max()
            fit={'diagnostic_only':True,'initial_drop_height_mm':8,'initial_xy_offset_mm':[3,2],
                 'minimum_radial_clearance_mm':float(recess['floor_diameter_mm']/2-radius*1000),
                 'floor_gap_mm':float(bottom_gap*1000),'last_second_position_variation_mm':float(motion*1000),
                 'magnetic_or_fixed_attachment':False,'robot_placement_demonstrated':False}
            fit['passed']=bool(fit['minimum_radial_clearance_mm']>0 and abs(fit['floor_gap_mm'])<.5 and fit['last_second_position_variation_mm']<.1)
            metadata['airpods_fit_check']=fit
            (a.output/'airpods-fit-check.json').write_text(json.dumps(fit,indent=2))
            if not fit['passed']:raise RuntimeError('AirPods recess fit validation failed: '+str(fit))
        (a.output/'metadata.json').write_text(json.dumps(metadata,indent=2));print(json.dumps(metadata),flush=True)
    finally:
        if env is not None:env.close()
        app.close()

if __name__=='__main__':main()
