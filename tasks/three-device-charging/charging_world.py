"""Native Isaac environment; scripted and Dream policies share this backend."""
import atexit, hashlib, importlib.metadata, json, os, subprocess
from pathlib import Path
import numpy as np
from charging_physics import array,transform,ChargingCoupling,DEVICES,PARAMS

_APP=None

class ChargingWorld:
    def __init__(self,root,output,probe=None,power=True,magnets=True,record=True,seed=17):
        global _APP
        self.root=Path(root);self.output=Path(output);self.output.mkdir(parents=True,exist_ok=False)
        self.env=self.coupling=self.video=None;self.record=record;self.steps=0;self.power=power;self.magnets=magnets
        self.manifest=json.loads((self.root/'assets/manifest.json').read_text())
        rng=np.random.default_rng(seed);table=self.manifest['table_top_m']
        centers={'iphone17':[.32,.16,table+.010],'airpods_pro':[.29,-.025,table+.014],'apple_watch':[.37,-.18,table+.009]}
        for name,center in centers.items():
            pos=np.array(center);pos[:2]+=rng.uniform(-.018,.018,2);yaw=float(rng.uniform(-.65,.65))
            self.manifest['assets'][name]['position_xyz']=pos.tolist()
            self.manifest['assets'][name]['rotation_xyzw']=[0,0,float(np.sin(yaw/2)),float(np.cos(yaw/2))]
        self.probe=probe;self.seen=dict.fromkeys(DEVICES,False);self.hold=0.;self.max_shift=0.;self.max_rotation=0.
        self.released_dwell=dict.fromkeys(DEVICES,0.)
        os.environ['ENABLE_CAMERAS']='1'
        if _APP is None:
            from isaaclab.app import AppLauncher
            _APP=AppLauncher(headless=True,enable_cameras=True,device='cpu').app;atexit.register(_APP.close)
        import torch
        from isaaclab.actuators import ImplicitActuatorCfg
        from isaaclab.sensors import ContactSensorCfg
        from isaaclab_arena.assets.object import Object
        from isaaclab_arena.assets.object_type import ObjectType
        from isaaclab_arena.utils.pose import Pose
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
        from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
        from yam_arena import station,constants,ik
        from yam_arena.urdf import quat_xyzw_to_matrix
        from yam_arena.embodiment import BimanualYamEmbodiment
        import yam_arena.scene as scenes
        self.torch=torch;self.constants=constants;self.station=station;self.ik=ik
        scenes.STATION_DOME_INTENSITY=250.;scenes.STATION_LAMP_INTENSITY=1.8e4;scenes.STATION_LAMP_RADIUS=.55
        embodiment=BimanualYamEmbodiment(enable_cameras=True,rig=station.rig(),cam_height=720,cam_width=1280,robot_contact_offset=.0001,contact_sensors=True)
        _,neutral=ik.tcp_pose(np.array(constants.READY))
        for side,sgn in [('left',1),('right',-1)]:
            c,s=np.cos(-sgn*.75),np.sin(-sgn*.75);rz=np.array([[c,-s,0],[s,c,0],[0,0,1.]])
            mount=quat_xyzw_to_matrix(np.array(station.wrist_camera(side).rot_xyzw))
            c,s=np.cos(.18),np.sin(.18);up=np.array([[1.,0,0],[0,c,-s],[0,s,c]])
            q,ep,er=ik.solve([.20,sgn*.25,1.04],ik.matrix_to_quat_xyzw(rz@neutral@mount@up@mount.T),constants.READY,station.BASES[side],iters=400,damping=.02)
            if ep>.002 or er>.02:raise RuntimeError('Initial IK failure')
            cfg=getattr(embodiment.scene_config,side+'_robot');cfg.init_state.joint_pos.update(dict(zip(constants.ARM_JOINTS,q.tolist())))
            cfg.actuators['gripper'].joint_effort_limit=12.
        if probe:
            stand=np.array(self.manifest['assets']['belkin']['position_xyz'])
            for name in DEVICES:
                p=PARAMS[name];normal=np.array(p['normal']);center=stand+np.array(p['center'])+normal*(p['puck_half']+p['back']+.0003)
                if name=='airpods_pro':r=np.eye(3)
                else:
                    x=np.array([0,-1.,0]);r=np.column_stack([x,np.cross(normal,x),normal])
                if probe=='offset':center+=r[:,0]*.025
                if probe=='reversed':r=r@np.diag([1.,-1.,-1.])
                spec=self.manifest['assets'][name];spec['position_xyz']=center.tolist();spec['rotation_xyzw']=ik.matrix_to_quat_xyzw(r).tolist()
        self.objects={}
        for name,spec in self.manifest['assets'].items():
            obj=Object(name=name,usd_path=str(self.root/'assets'/spec['path']),initial_pose=Pose.from_dict(spec),object_type=getattr(ObjectType,spec['object_type']))
            obj.object_cfg.spawn.activate_contact_sensors=True
            if name=='apple_watch':
                obj.object_cfg.actuators={'elastic_band':ImplicitActuatorCfg(joint_names_expr=['.*bend.*','.*twist.*'],stiffness=.0025,damping=.00015,joint_effort_limit=.025,joint_velocity_limit=15.)}
                obj.object_cfg.init_state.joint_pos={'.*':0.};obj.object_cfg.init_state.joint_vel={'.*':0.}
            self.objects[name]=obj
        stand_path=self.objects['belkin'].get_contact_sensor_prim_path();hands=[]
        for side in ('left','right'):
            arm=getattr(embodiment.scene_config,side+'_robot')
            hands.extend([arm.prim_path+embodiment.link_paths[k] for k in ['tip_left','tip_right']])
        sensors={}
        for name in DEVICES:
            path=self.objects[name].prim_path+'/case' if name=='apple_watch' else self.objects[name].get_contact_sensor_prim_path()
            sensors[name+'_contacts']=ContactSensorCfg(prim_path=path,filter_prim_paths_expr=[stand_path,*hands],update_period=0.,history_length=1,max_contact_data_count_per_prim=1024)
        device_paths=[self.objects['iphone17'].get_contact_sensor_prim_path(),self.objects['airpods_pro'].get_contact_sensor_prim_path(),self.objects['apple_watch'].prim_path+'/case']
        device_paths += [self.objects['apple_watch'].prim_path+f'/{side}_band_{i}' for side in ('long','short') for i in range(8)]
        device_paths.append(stand_path)
        for i,path in enumerate(hands):
            sensors[f'hand_{i}_contacts']=ContactSensorCfg(prim_path=path,filter_prim_paths_expr=device_paths,update_period=0.,history_length=1,max_contact_data_count_per_prim=1024)
        standard=scenes.env_cfg_callback(20)
        def config(cfg):
            cfg=standard(cfg);cfg.sim.dt=.001;cfg.decimation=50;cfg.sim.render_interval=50;cfg.episode_length_s=600.
            for name,sensor in sensors.items():setattr(cfg.scene,name,sensor)
            return cfg
        desc=IsaacLabArenaEnvironment(name='three_device_charging',scene=Scene([*scenes.task_room('station',embodiment.rig.top_cam),*self.objects.values()]),embodiment=embodiment,env_cfg_callback=config)
        self.env=ArenaEnvBuilder(desc,ArenaEnvBuilderCfg(num_envs=1,solve_relations=False,device='cpu')).make_registered();self.base=self.env.unwrapped
        import omni.replicator.core as rep
        rep.settings.set_render_rtx_realtime(antialiasing='DLAA')
        self.env.reset()
        watch=self.base.scene['apple_watch'];watch.actuators.target_command.set_position_index(value=torch.zeros((1,watch.num_joints),device=self.base.device),joint_ids=list(range(watch.num_joints)))
        self.coupling=ChargingCoupling(self.base,self.objects,power,magnets)
        self.initial_stand=self.coupling.poses()['belkin'].copy();self.trace=(self.output/'trace.jsonl').open('w')
        self.metadata={'engine':'Isaac native dynamics','seed':seed,'probe':probe,'scripted':False,'dream1_run':False,'physical_pose_writes_after_reset':0,'grasp_attachments':0,
          'magnetic_model':'uncalibrated finite paired force/torque approximation','initial_battery_percent':30,'cameras':['top_cam','left_cam','right_cam'],'overlays':False,'control_hz':20,'physics_hz':1000}
        source_path=self.root/'source_manifest.json'
        if source_path.exists():
            self.metadata.update(source_manifest_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),source_revision=json.loads(source_path.read_text())['revision'])
        self.metadata['versions']={n:importlib.metadata.version(n) for n in ['isaacsim','isaaclab','inspect-robots','inspect-robots-yam']}
        (self.output/'metadata.json').write_text(json.dumps(self.metadata,indent=2))
        if record:
            import imageio_ffmpeg
            self.video_log=(self.output/'video.log').open('w')
            self.video=subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(),'-y','-loglevel','error','-f','rawvideo','-pixel_format','rgb24','-video_size','1280x1080','-framerate','10','-i','-','-an','-c:v','libx264','-preset','fast','-crf','24','-pix_fmt','yuv420p',str(self.output/'three-camera.mp4')],stdin=subprocess.PIPE,stderr=self.video_log)

    def joints(self):
        values=[]
        for side in ('left_robot','right_robot'):
            arm=self.base.scene[side];ids,_=arm.find_joints(self.constants.ARM_JOINTS,preserve_order=True);fids,_=arm.find_joints(self.constants.FINGER_JOINTS,preserve_order=True)
            q=array(arm.data.joint_pos)[0];values.extend([*q[ids],np.clip(q[fids].mean()/self.constants.FINGER_OPEN,0,1)])
        return np.array(values)

    def images(self):return {n:array(self.base.scene[n].data.output['rgb'])[0,:,:,:3].astype(np.uint8) for n in ['top_cam','left_cam','right_cam']}

    def panel(self,images):
        from PIL import Image
        panel=Image.new('RGB',(1280,1080));panel.paste(Image.fromarray(images['top_cam']),(0,0))
        for name,x in [('left_cam',0),('right_cam',640)]:panel.paste(Image.fromarray(images[name]).resize((640,360),Image.Resampling.LANCZOS),(x,720))
        return panel

    def snapshot(self,label):
        images=self.images();self.panel(images).save(self.output/(label+'.png'))
        if label=='setup':
            from PIL import Image
            Image.fromarray(images['top_cam']).save(self.output/'setup-top.png')

    def visual_feedback(self,images,poses):
        # Privileged localization of the cue, followed by actual RGB evidence.
        # This diagnostic is not a claim that a learned policy understood it.
        for name in DEVICES:
            if not self.coupling.charging[name]:continue
            z={'iphone17':.0046,'apple_watch':.0053,'airpods_pro':.01115}[name]
            point=poses[name][:3,3]+poses[name][:3,:3]@np.array([0,-.003 if name=='airpods_pro' else 0,z])
            for cam,rgb in images.items():
                data=self.base.scene[cam].data
                camera=transform(np.r_[array(data.pos_w)[0],array(data.quat_w_ros)[0]])
                local=camera[:3,:3].T@(point-camera[:3,3])
                if local[2]<=0:continue
                k=array(data.intrinsic_matrices)[0];pixel=k@local;u,v=pixel[:2]/pixel[2]
                radius={'iphone17':.026,'apple_watch':.012,'airpods_pro':.002}[name]
                half=max(2,int(k[0,0]*radius/local[2]));x,y=int(u),int(v)
                if not (0<=x<rgb.shape[1] and 0<=y<rgb.shape[0]):continue
                crop=rgb[max(0,y-half):y+half+1,max(0,x-half):x+half+1].astype(float);r,g,b=np.moveaxis(crop,-1,0)
                mask=(r>100)&(r>g*1.15)&(g>b*1.3) if name=='airpods_pro' else (g>70)&(g>r*1.4)&(g>b*1.2)
                if mask.sum()>=(2 if name=='airpods_pro' else 8):self.seen[name]=True

    def step(self,action,label=''):
        action=np.asarray(action,dtype=float)
        if action.shape!=(14,) or not np.isfinite(action).all():raise ValueError('Expected 14 finite joint targets')
        self.coupling.render();self.env.step(self.torch.tensor([action.tolist()],dtype=self.torch.float32,device=self.base.device));self.steps+=1
        poses=self.coupling.poses();stand=poses['belkin'];rel=self.initial_stand[:3,:3].T@stand[:3,:3]
        shift=float(np.linalg.norm(stand[:3,3]-self.initial_stand[:3,3]));angle=float(np.degrees(np.arccos(np.clip((np.trace(rel)-1)/2,-1,1))))
        self.max_shift=max(self.max_shift,shift);self.max_rotation=max(self.max_rotation,angle)
        hand_forces=np.zeros(20)
        for i in range(4):hand_forces+=np.linalg.norm(array(self.base.scene[f'hand_{i}_contacts'].data.normal_force_matrix_w)[0,0],axis=-1)
        contact={'iphone17':float(hand_forces[0]),'airpods_pro':float(hand_forces[1]),'apple_watch':float(hand_forces[2:19].sum())}
        if self.steps%2==0:
            images=self.images();self.visual_feedback(images,poses)
            if self.video:self.video.stdin.write(np.asarray(self.panel(images)).tobytes())
        for name in DEVICES:
            qualified=self.coupling.charging[name] and self.seen[name] and contact[name]<.05
            self.released_dwell[name]=self.released_dwell[name]+.05 if qualified else 0.
        valid=all(self.coupling.charging.values()) and all(c<.05 for c in contact.values()) and all(self.seen.values()) and self.max_shift<=.005 and self.max_rotation<=5
        self.hold=self.hold+.05 if valid else 0.
        row={'step':self.steps,'phase':label,'devices':self.coupling.last,'cue_visible_in_rgb':self.seen.copy(),'stand_shift_m':shift,'stand_angle_deg':angle,'hold_s':self.hold,'action':action.tolist(),'measured_joints':self.joints().tolist(),
             'hand_contact_force_n':contact,'released_charge_dwell_s':self.released_dwell.copy(),
             'stand_hand_support_force_n':float(hand_forces[19]),
             'poses':{n:array(self.base.scene[n].data.root_link_pose_w)[0].tolist() for n in (*DEVICES,'belkin')}}
        self.trace.write(json.dumps(row)+'\n')
        if self.steps%20==0:
            self.trace.flush();(self.output/'progress.json').write_text(json.dumps({**self.score(),'phase':label,'devices':self.coupling.last},indent=2))
        return row

    def score(self):
        valid_stand=self.max_shift<=.005 and self.max_rotation<=5
        return {'steps':self.steps,'score':float(sum(min(1.,v/5.) for v in self.released_dwell.values())/3) if valid_stand else 0.,'success':bool(self.hold>=5-1e-8),'hold_s':self.hold,
                'charging':self.coupling.charging.copy(),'cue_visible_in_rgb':self.seen.copy(),'maximum_stand_translation_m':self.max_shift,'maximum_stand_rotation_deg':self.max_rotation}

    def close(self):
        if self.coupling:
            (self.output/'result.json').write_text(json.dumps(self.score(),indent=2));self.coupling.close()
        if hasattr(self,'trace'):self.trace.close()
        if self.video:
            self.video.stdin.close();self.video.wait(timeout=30);self.video_log.close()
        if self.env:self.env.close()
