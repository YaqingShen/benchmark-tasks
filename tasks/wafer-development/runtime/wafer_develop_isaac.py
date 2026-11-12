"""Continuous physical pickup, timed dip, and return using the frozen pliers workflow."""
from pathlib import Path
import argparse
import copy
import json
import math
import hashlib
import numpy as np
import wafer_full_cycle_isaac as full
import wafer_retention_isaac_v12 as native
from wafer_develop import (SPEC, ButtonTimer, DevelopmentScore, immersion_geometry, pattern_quality,
    make_develop_return_feedback, make_develop_release_feedback, placement_geometry_ok, placement_preopen_gate)
from wafer_grasp_protocol import _pose, _angle, pillar_plane_clearances, pad_contact_geometry


class Develop(full.FullCycle):
    include_gun=False
    task_scorer_enabled=False

    def validate(self,bundle,plan,metadata):
        super().validate(bundle,plan,metadata)
        for name in ('gun','gun_stand'):
            if (Path(bundle)/'assets'/f'{name}.usda').exists():
                raise ValueError('Excluded asset remains in development bundle: '+name)
        metadata['develop']=copy.deepcopy(SPEC)
        metadata['task_qualification_scope']='Inherited mechanism-guard configuration; current development success is evaluated separately.'
        files=json.loads((Path(bundle)/'develop_files.json').read_text())
        for name,digest in files.items():
            if Path(name).is_absolute()or'..'in Path(name).parts or hashlib.sha256((Path(bundle)/name).read_bytes()).hexdigest()!=digest:raise ValueError('Development source differs: '+name)
        metadata['develop_sources']=files
        plan['right_q_rad']=list(SPEC['initial_right_arm_q_rad'])
        self.wafer_plan['right_q_rad']=list(SPEC['initial_right_arm_q_rad'])
        metadata['develop']['right_start']='Empty hand above the timer; fixed initial pose before physics starts.'
        self.develop_active=False;self.measurement_index=0;self.read_timer=False
        self.clock=ButtonTimer();self.score=DevelopmentScore(SPEC['beaker_origin_world_m'],metadata['physics_dt_s'])
        self.grip_reference=None;self.initial_wafer=None;self.last_wafer=None;self.contact_gap_s=0.

    def configure(self,objects,embodiment,Pose,Object,ObjectType,Implicit,Contact,torch):
        sensors=super().configure(objects,embodiment,Pose,Object,ObjectType,Implicit,Contact,torch)
        for name,kind in [('beaker',ObjectType.RIGID),('timer',ObjectType.ARTICULATION)]:
            a=math.radians(SPEC['timer_yaw_deg']) if name=='timer' else 0.
            pose=Pose.from_dict(dict(position_xyz=list(SPEC[name+'_origin_world_m']),rotation_xyzw=[0.,0.,math.sin(a/2),math.cos(a/2)]))
            objects[name]=Object(name=name,usd_path=str(self.root/'assets'/f'{name}.usda'),object_type=kind,initial_pose=pose)
        cfg=objects['timer'].object_cfg
        cfg.actuators={'button_spring':Implicit(joint_names_expr=['timer_button_slide'],stiffness=400.,damping=1.,joint_effort_limit=4.,joint_velocity_limit=1.)}
        cfg.init_state.joint_pos={'timer_button_slide':0.};cfg.init_state.joint_vel={'timer_button_slide':0.}
        path=objects['timer'].prim_path+'/timer_button'
        for i,tip in enumerate(self.right_tip_paths):
            sensors[f'develop_timer_tip_{i}']=Contact(prim_path=tip,filter_prim_paths_expr=[path],update_period=0.,history_length=1,track_air_time=False,max_contact_data_count_per_prim=64)
        original=objects['wafer'].object_cfg.spawn.func
        def spawn(*args,**kwargs):
            root=original(*args,**kwargs)
            child=root.GetStage().DefinePrim(str(root.GetPath())+'/coating','Xform')
            child.GetReferences().AddReference(str(self.root/'assets/wafer_coating.usda'))
            return root
        objects['wafer'].object_cfg.spawn.func=spawn
        self.develop_objects={k:objects[k]for k in ('beaker','timer')}
        return sensors

    def ready(self,base,metadata,torch):
        super().ready(base,metadata,torch)
        import omni.usd
        from wafer_benchmark_backend import _GeometryIntegrity
        self.timer=base.scene['timer'];self.beaker=base.scene['beaker']
        self.timer_joint=int(self.timer.find_joints('timer_button_slide')[0][0])
        self.timer_base=int(self.timer.find_bodies('timer_base')[0][0])
        self.timer_button=int(self.timer.find_bodies('timer_button')[0][0])
        self.timer_contacts=[base.scene.sensors[f'develop_timer_tip_{i}']for i in range(2)]
        self.timer.actuators.target_command.set_position_index(value=torch.tensor([[0.]],device=base.device,dtype=torch.float32),joint_ids=[self.timer_joint])
        self.timer.write_data_to_sim()
        roots={k:o.prim_path.replace('{ENV_REGEX_NS}','/World/envs/env_0').replace('{ENV_NS}','/World/envs/env_0').replace('/env_.*/','/env_0/')for k,o in self.develop_objects.items()}
        self.develop_integrity=_GeometryIntegrity(omni.usd.get_context().get_stage(),roots)
        self.clamp_shapes=json.loads((self.root/'assets/develop_clamp_vertices.json').read_text())
        self.body_fields={'lower_half':('lever_link_poses_world',0),'upper_half':('lever_link_poses_world',1),
            'lower_pad':('jaw_link_poses_world',0),'upper_pad':('jaw_link_poses_world',1),
            'lower_grip_paddle':('handle_paddle_poses_world',0),'upper_grip_paddle':('handle_paddle_poses_world',1)}
        metadata['develop']['initial_condition']='Both hands empty; exposed wafer on original heater pillars; pliers on original stand.'
        metadata['develop']['timer_observation']='Script reads a contact-started 30-second countdown; withdrawal begins at zero. No OCR claim.'
        metadata['initial_condition']=metadata['develop']['initial_condition']
        metadata['controller']['initial_condition']=metadata['initial_condition']
        metadata['tool_hand_retention_profile']['reference']='Last tool_grip_hold sample; retained through wafer pickup, development and return.'
        from pxr import UsdGeom
        stage=omni.usd.get_context().get_stage()
        metadata['develop_scene_objects']=sorted(self.objects)
        exclusions={}
        for name in ('gun','gun_stand'):
            path='/World/envs/env_0/'+name
            exclusions[name]=dict(configured=name in self.objects,prim_path=path,
                prim_present=stage.GetPrimAtPath(path).IsValid(),
                asset_present=(self.root/'assets'/f'{name}.usda').exists())
            if any(exclusions[name][k]for k in ('configured','prim_present','asset_present')):
                raise ValueError('Excluded object remains in development scene: '+name)
        metadata['development_scene_exclusions']=exclusions
        self.display_segments={(d,s):UsdGeom.Imageable(stage.GetPrimAtPath(roots['timer']+f'/timer_base/screen_panel/digit_{d}_{s}')).CreateVisibilityAttr()for d in range(3)for s in range(7)}
        wafer_path=self.objects['wafer'].prim_path.replace('{ENV_REGEX_NS}','/World/envs/env_0').replace('{ENV_NS}','/World/envs/env_0').replace('/env_.*/','/env_0/')
        self.pattern_views={state:UsdGeom.Imageable(stage.GetPrimAtPath(wafer_path+'/coating/'+state)).CreateVisibilityAttr()for state in ('BEFORE','UNDERDEVELOPED','GOOD','OVERDEVELOPED')}
        self.visual_key=None

    def update_display(self,elapsed,state):
        remaining=max(0.,SPEC['target_dwell_s']-elapsed)
        digits=f'{min(999,math.ceil(remaining*10-1e-6)):03d}';key=(digits,state)
        if key==self.visual_key:return
        self.visual_key=key
        segments=({0,1,2,3,4,5},{1,2},{0,1,3,4,6},{0,1,2,3,6},{1,2,5,6},{0,2,3,5,6},{0,2,3,4,5,6},{0,1,2},{0,1,2,3,4,5,6},{0,1,2,3,5,6})
        for (d,s),attr in self.display_segments.items():attr.Set('inherited'if s in segments[int(digits[d])]else'invisible')
        for name,attr in self.pattern_views.items():attr.Set('inherited'if name==state else'invisible')

    def fields(self):
        result=super().fields()
        result.extend([('develop_timer_q',self.timer.data.joint_pos,None),('develop_timer_poses',self.timer.data.body_link_pose_w,None),
            ('develop_beaker_pose',self.beaker.data.root_link_pose_w,(1,7))])
        result.extend((f'develop_timer_contact_{i}',s.data.normal_force_matrix_w,(1,1,1,3))for i,s in enumerate(self.timer_contacts))
        return result

    def unmatched_right(self,a,names):
        net=super().unmatched_right(a,names)
        for i,name in enumerate(('tip_left','tip_right')):
            net[0,names.index(name)]-=a[f'develop_timer_contact_{i}'][0,0,0]
        return net

    def measurement(self,a,row,phase):
        result=super().measurement(a,row,phase)
        combined={**row,**result};dt=self.metadata['physics_dt_s'];t=self.measurement_index*dt;self.measurement_index+=1
        forces=[float(np.linalg.norm(a[f'develop_timer_contact_{i}'][0,0,0]))for i in range(2)]
        compression=-float(a['develop_timer_q'][0,self.timer_joint])
        elapsed=self.clock.update(t,compression,sum(forces))
        wafer=_pose(row['wafer_pose_world'],'wafer')
        if self.initial_wafer is None:self.initial_wafer=wafer.copy()
        relation=np.linalg.inv(_pose(row['clamp_pose_world'],'clamp'))@wafer
        slip=rotation=0.
        if self.grip_reference is not None:
            slip=float(np.linalg.norm(relation[:3,3]-self.grip_reference[:3,3]))
            rotation=math.degrees(_angle(relation[:3,:3],self.grip_reference[:3,:3]))
        pillar=pillar_plane_clearances(row['wafer_pose_world'],self.metadata) if wafer[2,2]>.5 else dict(clearances_m=[1.]*3,all_pillars_under_wafer=False)
        gaps=pillar['clearances_m']
        support=sum(row['pillar_contact_normal_force_N'])
        on_pillars=bool(pillar['all_pillars_under_wafer'] and max(abs(x)for x in gaps)<=.0005 and support>.05)
        wafer_speed=wafer_omega=0.
        if self.last_wafer is not None:
            wafer_speed=float(np.linalg.norm(wafer[:3,3]-self.last_wafer[:3,3])/dt)
            wafer_omega=float(_angle(wafer[:3,:3],self.last_wafer[:3,:3])/dt)
        self.last_wafer=wafer.copy()
        engaged=pad_contact_geometry(row,self.metadata)['both_engaged']
        bilateral=min(row['jaw_contact_normal_force_N'])>.005 and engaged
        self.contact_gap_s=0. if bilateral or not self.develop_active else self.contact_gap_s+dt
        beaker=a['develop_beaker_pose'][0];origin=np.asarray(beaker[:3]);dry=1.
        meshes=[]
        for p in self.clamp_shapes:
            if not any(k in p['name']for k in ('handle','backing','grip')):continue
            field,index=self.body_fields[p['body']];pose=_pose(combined[field][index],p['body'])
            meshes.append(np.asarray(p['vertices'])@pose[:3,:3].T+pose[:3,3])
        poses=dict(zip(self._robot_geometry_guard.keys,combined['robot_link_poses_world']))
        for key,pieces in zip(self._robot_geometry_guard.keys,self._robot_geometry_guard.meshes):
            if key[1]not in ('gripper','tip_left','tip_right'):continue
            pose=_pose(poses[key],str(key));meshes.append(np.concatenate([m.vertices for m in pieces])@pose[:3,:3].T+pose[:3,3])
        for vertices in meshes:
            low,high=vertices.min(0),vertices.max(0)
            closest=np.clip(origin[:2],low[:2],high[:2])
            if np.linalg.norm(closest-origin[:2])<=SPEC['beaker_inner_radius_m']:
                dry=min(dry,float(low[2]-origin[2]-SPEC['liquid_height_m']))
        present,intact=self.develop_integrity.snapshot()
        tool_retained=not self.tool_hand_monitor.envelope.failed
        pose_error=float(np.linalg.norm(wafer[:3,3]-self.initial_wafer[:3,3]))
        angle_error=math.degrees(_angle(wafer[:3,:3],self.initial_wafer[:3,:3]))
        released=phase in ('release','retreat','unassisted_hold') and max(row['jaw_contact_normal_force_N'])<.005 and row['measured_jaw_gap_m']>.002
        result.update(develop_sample_time_s=t,timer_button_displacement_m=compression,timer_button_force_N=sum(forces),timer_tip_force_N=forces,
            timer_elapsed_s=elapsed,timer_remaining_s=self.clock.remaining(t),timer_reading_s=self.clock.remaining(t) if self.read_timer else None,
            timer_base_pose_world=a['develop_timer_poses'][0,self.timer_base].tolist(),timer_button_pose_world=a['develop_timer_poses'][0,self.timer_button].tolist(),
            beaker_pose_world=beaker.tolist(),pickup_verified=self.develop_active,
            develop_objects_intact=all(present.values())and all(intact.values())and all(result['objects_intact'].values()),
            tool_hand_retained=tool_retained,unexpected_contact_N=row['unexpected_robot_tool_contact_normal_force_N']+row['unexpected_robot_environment_contact_normal_force_N']+row['forbidden_wafer_contact_force_N']+result['tool_unexpected_environment_contact_normal_force_N'],
            hand_handle_liquid_clearance_m=dry,wafer_slip_m=slip,wafer_rotation_error_deg=rotation,
            wafer_bilaterally_held=bilateral or self.contact_gap_s<=SPEC['maximum_grip_contact_gap_s'],wafer_on_pillars=on_pillars,wafer_released=released,
            wafer_pillar_plane_gaps_m=list(gaps),wafer_pillar_footprints_covered=bool(pillar['all_pillars_under_wafer']),
            wafer_all_pillars_loaded=bool(min(row['pillar_contact_normal_force_N'])>SPEC['minimum_final_pillar_contact_N']),
            wafer_return_position_error_m=pose_error,wafer_return_rotation_deg=angle_error,
            wafer_return_stable=wafer_speed<.001 and wafer_omega<.02,
            wafer_pose_linear_speed_m_s=wafer_speed,wafer_pose_angular_speed_rad_s=wafer_omega)
        return result

    def guard(self,row):
        if abs(row['develop_sample_time_s']-row['time_s'])>1e-8:raise native.TrialAbort('Timer and recorded physics clocks disagree.')
        if not self.develop_active:
            super().guard(row)
        else:
            from wafer_tool_pickup import guard_handle_paddle_capacity
            guard_handle_paddle_capacity(row,self.metadata)
            if self.tool_hand_monitor.observe(row)is False:raise native.TrialAbort('Pliers retention failed during development.')
            if row['time_s']>120.:raise native.TrialAbort('Development workflow exceeded 120 seconds.')
            if row['tool_unexpected_environment_contact_normal_force_N']>.01:raise native.TrialAbort('Pliers contacted an unintended obstacle.')
            if row['phase']=='release' and not placement_geometry_ok(row,SPEC['placement_release_gap_m']):
                raise native.TrialAbort('Wafer left the bounded pillar-transfer geometry during release.')
        self.score.update({**row,'objects_intact':row['develop_objects_intact']})
        state='BEFORE'if self.score.left_at is None else pattern_quality(self.score.left_at-self.score.entered_at)
        self.update_display(row['timer_elapsed_s'],state)
        if self.score.failures:raise native.TrialAbort('Development validity: '+', '.join(sorted(self.score.failures)))
        if row['timer_button_force_N']>3.:raise native.TrialAbort('Timer press force exceeded 3 N.')

    def run(self,env,base,metadata,recorder,previous_action,preload,previous_fingers,destination,torch,prefix):
        from yam_arena import ik,station,constants
        from wafer_finger_preload_v12 import preceding_interval_features
        from wafer_cycle_protocol import plan_return,retreat_offset
        self.develop_active=True;self.workflow_stage='develop';raw=recorder.raw
        row=raw.rows[-1];self.grip_reference=np.linalg.inv(_pose(row['clamp_pose_world'],'clamp'))@_pose(row['wafer_pose_world'],'wafer')
        action=np.array(previous_action,dtype=float);records=[];origin=native._array(base.scene.env_origins,(1,3))[0]
        def tcp(q,side='left'):return ik.tcp_pose(q,np.asarray(station.BASES[side])+origin)
        def solve(p,r,q,side='left'):
            value,pe,ae=ik.solve(p,ik.matrix_to_quat_xyzw(r),q,base_pos=np.asarray(station.BASES[side])+origin,iters=300,damping=.012,tol=1e-7)
            if pe>.0001 or ae>.001:raise native.TrialAbort('Development IK did not converge.')
            return value
        def tick(phase,left=None,right=None,right_grip=-.003,left_grip=None,feedback=None,preopen=None,release_feedback=None):
            nonlocal previous_fingers
            if left is not None:
                if np.max(abs(left-action[:6]))>math.radians(2.):raise native.TrialAbort('Left development command step exceeds 2 degrees.')
                action[:6]=left
            if right is not None:
                if np.max(abs(right-action[7:13]))>math.radians(2.):raise native.TrialAbort('Right development command step exceeds 2 degrees.')
                action[7:13]=right
            if left_grip is None:
                if preload.phases[-1][0]==phase:
                    name,duration=preload.phases[-1];preload.phases[-1]=(name,duration+.05)
                else:preload.phases.append((phase,.05))
                features=preceding_interval_features(recorder.rows,preload.command_index,metadata['physics_dt_s'])
                previous_fingers,diagnostic=preload.step(phase,features,previous_fingers)
                action[6]=previous_fingers[0]/constants.FINGER_OPEN
                self.preload_reference['control_index']=diagnostic['control_index']
            else:
                action[6]=left_grip/constants.FINGER_OPEN;diagnostic=None;self.preload_reference['control_index']=None
            action[13]=right_grip/constants.FINGER_OPEN
            records.append(dict(time_s=raw.total*raw.dt,phase=phase,action=action.tolist(),preload=diagnostic,timer_reading_s=raw.rows[-1]['timer_remaining_s']if self.read_timer else None))
            if feedback is not None:records[-1]['return_feedback']=feedback
            if release_feedback is not None:records[-1]['release_feedback']=release_feedback
            if preopen is not None:records[-1]['placement_preopen_gate']=preopen
            recorder.begin(phase);env.step(torch.tensor([action.tolist()],device=base.device,dtype=torch.float32));value=recorder.end()
            if len(records)%20==0:print(f"Develop t={raw.total*raw.dt:.2f}: {phase}; remaining {value['timer_remaining_s']:.2f}; force {value['timer_button_force_N']:.2f} N",flush=True)
            return value
        def left_path(phase,p1,r1,seconds,right_target=None,right_descent=False):
            current=raw.rows[-1];w=_pose(current['wafer_pose_world'],'wafer');h=_pose(current['measured_tcp_pose_world'],'tcp');relation=np.linalg.inv(w)@h
            p0=w[:3,3];r0=w[:3,:3];q=action[:6].copy();qright=action[7:13].copy();finger0=action[13]*constants.FINGER_OPEN
            for i in range(round(seconds*20)):
                u=ik.smoothstep((i+1)/(seconds*20));p=p0+(p1-p0)*u;r=ik.slerp_matrix(r0,r1,u)
                q=solve(p+r@relation[:3,3],r@relation[:3,:3],q)
                right=qright+(right_target-qright)*u if right_target is not None else None
                if right_descent and i<40:
                    fraction=ik.smoothstep((i+1)/40)
                    right=solve(hover+[0,0,.10*(1-fraction)],down,action[7:13],'right')
                if phase=='develop_immerse' and self.clock.started_at is None and immersion_geometry(raw.rows[-1]['wafer_pose_world'],beaker)['fully_immersed']:
                    p_r,r_r=tcp(action[7:13],'right');p_r[2]-=.0003;right=solve(p_r,r_r,action[7:13],'right')
                tick(phase,q,right,right_grip=finger0+(-.003-finger0)*u)
        def right_move(phase,p,r,seconds):
            p0,r0=tcp(action[7:13],'right');q=action[7:13].copy()
            for i in range(round(seconds*20)):
                u=ik.smoothstep((i+1)/(seconds*20));q=solve(p0+(p-p0)*u,ik.slerp_matrix(r0,r,u),q,'right');tick(phase,right=q)
        def button_target(clearance):
            b=_pose(raw.rows[-1]['timer_button_pose_world'],'button')
            return b[:3,3]+np.array([0,0,.003+.0128425+clearance])
        yaw=math.radians(SPEC['timer_yaw_deg']);rz=np.array([[math.cos(yaw),-math.sin(yaw),0],[math.sin(yaw),math.cos(yaw),0],[0,0,1.]])
        down=rz@np.diag([1.,-1.,-1.]);hover=button_target(.002)
        qhover=solve(hover,down,action[7:13],'right')
        qabove=solve(hover+[0,0,.10],down,qhover,'right')
        wafer=_pose(row['wafer_pose_world'],'wafer');original_rotation=wafer[:3,:3].copy();start_xy=wafer[:2,3].copy()
        upright=np.array([[0.,1.,0.],[0.,0.,-1.],[-1.,0.,0.]])
        beaker=np.asarray(SPEC['beaker_origin_world_m']);above=beaker+np.array([0,0,.225]);inside=beaker+np.array([0,0,.07])
        try:
            left_path('develop_raise',np.r_[start_xy,above[2]],original_rotation,5.,qabove)
            left_path('develop_transfer',above,original_rotation,6.,right_descent=True)
            left_path('develop_vertical',above,upright,6.)
            left_path('develop_immerse',inside,upright,4.)
            def press(phase):
                old=self.clock.start_count
                for i in range(30):
                    row=raw.rows[-1]
                    if self.clock.start_count>old:break
                    p,r=tcp(action[7:13],'right');p[2]-=.0003
                    tick(phase,right=solve(p,r,action[7:13],'right'))
                if self.clock.start_count==old:raise native.TrialAbort('Physical timer press did not register.')
                right_move(phase+'_release',button_target(.003),down,.3)
            if self.clock.started_at is None:press('timer_start')
            else:right_move('timer_start_release',button_target(.003),down,.3)
            self.read_timer=True
            while self.clock.started_at is not None and raw.rows[-1]['timer_remaining_s']>0.:
                tick('develop_hold')
            p0=_pose(raw.rows[-1]['wafer_pose_world'],'wafer');relation=np.linalg.inv(p0)@_pose(raw.rows[-1]['measured_tcp_pose_world'],'tcp')
            q=action[:6].copy();pressed=False
            for i in range(80):
                u=ik.smoothstep((i+1)/80);p=p0[:3,3]+(above-p0[:3,3])*u
                q=solve(p+upright@relation[:3,3],upright@relation[:3,:3],q)
                right=None
                if not immersion_geometry(raw.rows[-1]['wafer_pose_world'],beaker)['fully_immersed']and self.clock.stopped_at is None:
                    p_r,r_r=tcp(action[7:13],'right');p_r[2]-=.0003;right=solve(p_r,r_r,action[7:13],'right')
                tick('develop_withdraw',q,right)
            if self.clock.stopped_at is None:raise native.TrialAbort('Timer stop was not physically registered.')
            right_move('timer_stop_release',button_target(.015),down,.5)
            for _ in range(10):
                if self.score.withdrawal_verified:break
                tick('develop_withdraw_hold')
            if not self.score.withdrawal_verified:raise native.TrialAbort('Withdrawal hold was not verified.')
            left_path('develop_horizontal',above,original_rotation,6.)
            left_path('develop_return_above',np.r_[start_xy,above[2]],original_rotation,6.)
            plan=plan_return(recorder.rows,solve,tcp,metadata)
            for phase,seconds,target,fingers in plan:
                q0=action[:6].copy();f0=action[6]*constants.FINGER_OPEN
                if phase=='lower':p0,r0=tcp(q0);p1,r1=tcp(target)
                if phase=='return_hold':
                    def residual(p,r,q):
                        value=solve(p,r,q);p2,r2=tcp(value);return value,float(np.linalg.norm(p2-p)),float(_angle(r2,r))
                    servo,nominal=make_develop_return_feedback(recorder.rows,q0,tcp,residual,ik.LOWER,ik.UPPER)
                if phase=='release':
                    preopen=placement_preopen_gate(raw.rows,raw.dt)
                    if not preopen['passed']:raise native.TrialAbort('Sustained pillar contact was not established before release.')
                    release_servo=make_develop_release_feedback(raw.rows[-1],q0,tcp,residual,ik.LOWER,ik.UPPER)
                for i in range(round(seconds*20)):
                    u=ik.smoothstep((i+1)/(seconds*20));q=q0+(target-q0)*u;feedback=None;release_feedback=None
                    if phase=='lower':q=solve(p0+(p1-p0)*u,ik.slerp_matrix(r0,r1,u),action[:6])
                    if phase=='return_hold':
                        w=_pose(raw.rows[-1]['wafer_pose_world'],'wafer');q,feedback=servo.step(nominal,w[:3,3],w[:3,:3],action[:6],integrate=True)
                    if phase=='release':q,release_feedback=release_servo.step(raw.rows[-1],action[:6])
                    tick(phase,q,left_grip=f0+(fingers[0]-f0)*u,feedback=feedback,
                        preopen=preopen if phase=='release' and i==0 else None,release_feedback=release_feedback)
                if phase=='return_hold':
                    seat_p,seat_r=tcp(action[:6]);seat_finger=float(action[6]*constants.FINGER_OPEN);descent_steps=0
                    for step in range(SPEC['placement_seating_max_commands']):
                        measured=raw.rows[-1]
                        if placement_preopen_gate(raw.rows,raw.dt)['passed']:break
                        if (measured['wafer_return_position_error_m']>SPEC['return_position_tolerance_m']
                            or measured['wafer_return_rotation_deg']>SPEC['return_orientation_tolerance_deg']):
                            raise native.TrialAbort('Wafer is not aligned for bounded pillar seating.')
                        if sum(measured['pillar_contact_normal_force_N'])>.05 and not measured['wafer_on_pillars']:
                            raise native.TrialAbort('Pillar contact occurred outside the accepted support geometry.')
                        q=action[:6].copy()
                        if not (measured['wafer_on_pillars'] and placement_geometry_ok(measured,SPEC['placement_ready_gap_m'])):
                            descent_steps+=1
                            p=seat_p-np.array([0.,0.,SPEC['placement_seating_step_m']*descent_steps])
                            q=solve(p,seat_r,action[:6])
                        tick('return_hold',q,left_grip=seat_finger)
                    if not placement_preopen_gate(raw.rows,raw.dt)['passed']:
                        raise native.TrialAbort('Bounded seating did not establish sustained pillar contact.')
                    held=action[:6].copy();p,r=tcp(held);offset=retreat_offset(recorder.rows)
                    away=solve(p+offset,r,held);raised=solve(p+offset+[0,0,.030],r,away)
                    for index,q in zip(range(2,6),(held,away,raised,raised)):
                        name,duration,_,f=plan[index];plan[index]=(name,duration,q,f)
                if phase=='release':
                    measured=raw.rows[-1]
                    if not (measured['wafer_released']and measured['wafer_on_pillars']):
                        raise native.TrialAbort('Wafer was not released onto the pillars before retreat.')
                    held=action[:6].copy();p,r=tcp(held);offset=retreat_offset(recorder.rows)
                    away=solve(p+offset,r,held);raised=solve(p+offset+[0,0,.030],r,away)
                    for index,q in zip(range(3,6),(away,raised,raised)):
                        name,duration,_,f=plan[index];plan[index]=(name,duration,q,f)
            result=self.score.result();result['result']='WAFER_DEVELOP_SUCCEEDED_IN_THIS_SIMULATION'if result['success']else'FAILED'
            result['prefix_result']=prefix['result'];result['scope']='Native physical handling; declared elapsed-time process surrogate.'
            native._json(Path(destination)/'develop_report.json',result)
            return result
        finally:
            with (Path(destination)/'develop_commands.jsonl').open('w')as stream:
                for value in records:stream.write(json.dumps(value,allow_nan=False)+'\n')
            native._json(Path(destination)/'develop_partial_report.json',self.score.result())


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--bundle',type=Path,required=True);parser.add_argument('--output',type=Path)
    parser.add_argument('--check-only',action='store_true')
    args=parser.parse_args();full.ALLOW_UNQUALIFIED_DEVELOPMENT=True
    if args.check_only:
        plan,exported,models,gate=native.validate_bundle(args.bundle)
        metadata=native.cloud_model_metadata(exported['metadata'],plan)
        Develop().validate(args.bundle,plan,metadata)
        print(json.dumps(dict(status='DEVELOP_BUNDLE_CHECKED',spec=SPEC)))
    elif args.output is None:parser.error('--output is required for a simulation.')
    else:native.run_isaac_trial(args.bundle,args.output,extension=Develop(),device='cpu')
