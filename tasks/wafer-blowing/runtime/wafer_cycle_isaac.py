"""Native extension for contact-held blowing and wafer return."""
from pathlib import Path
import argparse,hashlib,json,time,math
import numpy as np
import wafer_retention_isaac_v12 as native
from wafer_cycle_protocol import SCAN,GUN_OFF_GRIP_M,GUN_RETENTION,GunFeedback,plan_scan,plan_return,make_return_feedback,evaluate_cycle,retreat_offset,gun_motion_timing
from wafer_finger_preload_v12 import preceding_interval_features
from wafer_grasp_protocol import _pose,_angle


def evaluate_native_cycle(metadata,rows):
    def mechanical(meta,trace,require_return=False):
        report=native.evaluate_wafer_grasp_trial(meta,trace,require_return=require_return)
        peak=max(r['unexpected_robot_tool_contact_normal_force_N']+r['unexpected_robot_environment_contact_normal_force_N']for r in trace)
        report['checks']['robot_obstacle_contacts']=dict(passed=peak<=report['limits']['maximum_unwanted_contact_N'],
            maximum_combined_unwanted_normal_residual_N=peak,scope='Body-filtered force vectors; cancelling contacts can be hidden.')
        report['checks']['finger_handle_jam_guard']=native.evaluate_finger_handle_load(trace,engine='isaac')
        report['checks']['passive_pad_motion_guard']=native.evaluate_passive_pad_state(trace,meta['pad_compliance'])
        report['checks']['passive_hinge_torque_guard']=native.evaluate_passive_hinge_state(trace,meta['passive_hinge'])
        return report
    return evaluate_cycle(metadata,rows,mechanical)


class Cycle:
    def _validate_runtime_spec(self,bundle,plan,metadata,s):
        self.root=Path(bundle);self.spec=s
        if s['scan']!=SCAN:raise ValueError('Shared scan specification differs.')
        if s['gun_retention']!=GUN_RETENTION:raise ValueError('Shared gun retention criteria differ.')
        for name,digest in s['files'].items():
            native._bundle_file(self.root,name,digest)
        import importlib
        for name in ('wafer_cycle_protocol.py','wafer_cycle_isaac.py','wafer_finger_preload_v12.py'):
            path=Path(__file__)if name=='wafer_cycle_isaac.py'else Path(importlib.import_module(name[:-3]).__file__)
            if hashlib.sha256(path.read_bytes()).hexdigest()!=s['files'][name]:
                raise ValueError('Imported cycle source differs: '+name)
        self.gun=s['gun'];self.export=s['gun_export'];self.scan_row=-1
        plan['right_q_rad']=self.gun['approach_q_rad']
        metadata.update(gun=self.gun,cycle_local_gate=s['local_gate'],scan=SCAN,
            cycle_source_hashes=s['files'],right_arm_present=True,
            air_model_scope=SCAN['scope'])


    def validate(self,bundle,plan,metadata):
        self.root=Path(bundle);s=json.loads((self.root/'cycle_plan.json').read_text())
        early=s['local_gate'].get('policy')=='user_requested_early_cloud_v1'
        if early and (plan['local_validation'].get('policy')!='user_requested_early_cloud_v1' or s['local_gate'].get('qualification_complete')is not False):
            raise ValueError('Early-cloud policy disagrees with the declared incomplete qualification.')
        if not s['local_gate']['nominal_passed']or(not early and(s['local_gate']['attempted']!=20 or s['local_gate']['passed']<16)):
            raise ValueError('Complete-cycle local qualification has not passed.')
        self._validate_runtime_spec(bundle,plan,metadata,s)
        import zipfile
        with zipfile.ZipFile(self.root/'cycle_validation.zip')as archive:
            passed=0
            for case in s['local_gate']['cases']:
                values={}
                for name in ('report.json','independent_check.json'):
                    entry=case[name];data=archive.read(entry['member'])
                    if hashlib.sha256(data).hexdigest()!=entry['sha256']:raise ValueError('Cycle evidence hash differs.')
                    values[name]=json.loads(data)
                good=(values['report.json']['result']=='WAFER_AIR_PATH_AND_RETURN_SUCCEEDED_IN_THIS_SIMULATION'
                    and values['independent_check.json']['matches_report']and values['independent_check.json']['evidence']['complete'])
                if good!=case['passed']:raise ValueError('Cycle local outcome differs from saved evidence.')
                passed+=int(good)
            if len(s['local_gate']['cases'])!=s['local_gate']['attempted']or passed!=s['local_gate']['passed']:raise ValueError('Cycle batch count differs.')


    def configure(self,objects,embodiment,Pose,Object,ObjectType,Implicit,Contact,torch):
        self.objects=objects
        for name in ('gun','gun_stand'):
            p=self.export['initial_world_poses'][name]
            pose=Pose.from_dict({'position_xyz':[float(x)for x in p[:3]],'rotation_xyzw':[float(x)for x in p[3:]]})
            if pose.to_tensor(device='cpu').dtype!=torch.float32:raise ValueError('Gun initial pose must use float32.')
            obj=Object(name=name,usd_path=str(self.root/self.export['models'][name]),
                object_type=ObjectType.ARTICULATION if name=='gun'else ObjectType.RIGID,initial_pose=pose)
            objects[name]=obj
        spring=self.export['passive_spring'];cfg=objects['gun'].object_cfg
        cfg.actuators={'trigger_spring':Implicit(joint_names_expr=['gun_trigger_slide'],stiffness=1000.,damping=2.,
            joint_effort_limit=8.,joint_velocity_limit=1.)}
        cfg.init_state.joint_pos={'gun_trigger_slide':0.};cfg.init_state.joint_vel={'gun_trigger_slide':0.}
        original=cfg.spawn.func;self.spawn_check={'status':'NOT_SPAWNED','count':0}
        def spawn(*args,**kwargs):
            from pxr import Usd,UsdPhysics
            from wafer_retention_usd_v12 import verify_drive_prim
            root=original(*args,**kwargs)
            joints=[p for p in Usd.PrimRange(root,Usd.TraverseInstanceProxies())if p.IsA(UsdPhysics.Joint)]
            if len(joints)!=1 or joints[0].GetName()!='gun_trigger_slide':raise ValueError('Unexpected gun joint topology.')
            bodies=[p for p in Usd.PrimRange(root,Usd.TraverseInstanceProxies())if p.HasAPI(UsdPhysics.RigidBodyAPI)]
            if {p.GetName()for p in bodies}!={'blow_gun','gun_trigger'}:raise ValueError('Unexpected gun bodies.')
            for body in bodies:
                rb=UsdPhysics.RigidBodyAPI(body)
                if rb.GetKinematicEnabledAttr().Get()is not False or rb.GetRigidBodyEnabledAttr().Get()is not True:
                    raise ValueError('Gun and trigger must be dynamic.')
                mass=float(UsdPhysics.MassAPI(body).GetMassAttr().Get())
                if abs(mass-self.export['inertials'][body.GetName()]['mass_kg'])>1e-8:raise ValueError('Spawned gun mass differs.')
            links={str(t)for rel in (UsdPhysics.Joint(joints[0]).GetBody0Rel(),UsdPhysics.Joint(joints[0]).GetBody1Rel())for t in rel.GetTargets()}
            if links!={str(p.GetPath())for p in bodies}:raise ValueError('Gun has an unintended attachment.')
            self.spawn_check.update(verify_drive_prim(joints[0],spring));self.spawn_check['count']+=1
            if self.spawn_check['count']!=1:raise ValueError('Gun was spawned more than once.')
            return root
        cfg.spawn.func=spawn
        self.right_tip_paths=[embodiment.scene_config.right_robot.prim_path+embodiment.link_paths[n]for n in ('tip_left','tip_right')]
        gunpaths=[objects['gun'].prim_path+'/'+n for n in ('blow_gun','gun_trigger')]
        self.gunpaths=gunpaths
        def sensor(prim,filters):return Contact(prim_path=prim,filter_prim_paths_expr=filters,update_period=0.,
            history_length=1,track_air_time=False,max_contact_data_count_per_prim=256)
        result={f'right_tip_{i}_gun':sensor(path,gunpaths)for i,path in enumerate(self.right_tip_paths)}
        filters=[*self.right_tip_paths,objects['gun_stand'].get_contact_sensor_prim_path()]
        result.update({f'gun_link_{i}_contacts':sensor(path,filters)for i,path in enumerate(gunpaths)})
        return result

    def ready(self,base,metadata,torch):
        self.base=base;self.gunasset=base.scene['gun'];self.right=base.scene['right_robot']
        self.tip_sensors=[base.scene.sensors[f'right_tip_{i}_gun']for i in range(2)]
        self.gun_sensors=[base.scene.sensors[f'gun_link_{i}_contacts']for i in range(2)]
        self.right_hand=self.right.find_bodies('gripper')[0][0]
        self.gun_body=self.gunasset.find_bodies('blow_gun')[0][0]
        self.trigger_id=self.gunasset.find_joints('gun_trigger_slide')[0][0]
        self.right_arm_ids,self.right_arm_names=self.right.find_joints(['joint'+str(i)for i in range(1,7)],preserve_order=True)
        self.right_finger_ids,_=self.right.find_joints(['joint7','joint8'],preserve_order=True)
        self.left_arm_ids,_=base.scene['left_robot'].find_joints(['joint'+str(i)for i in range(1,7)],preserve_order=True)
        if self.spawn_check['status']!='COMPOSED_FORCE_DRIVE_VERIFIED':raise RuntimeError('Passive trigger drive missing.')
        metadata['verified_live_right_actuators']=native.verify_live_arm_gains(self.right,metadata['right_actuators'])
        spring=self.export['passive_spring']
        self.gunasset.actuators.target_command.set_position_index(value=torch.tensor([[0.]],device=base.device,dtype=torch.float32),joint_ids=[self.trigger_id])
        self.gunasset.write_data_to_sim()
        metadata['verified_native_trigger_spring']=native.native_mechanism_readback(self.gunasset,spring,0.,expected_dof_count=1)
        metadata['gun_contact_scope']='Filtered rigid-body vectors and residuals; no per-shape force or cancellation-proof collision guarantee.'

    def fields(self):
        fields=[('gun_poses',self.gunasset.data.body_link_pose_w,None),('gun_q',self.gunasset.data.joint_pos,None),
            ('gun_v',self.gunasset.data.joint_vel,None),('right_q',self.right.data.joint_pos,None),
            ('right_targets',self.right.data.joint_pos_target,None),('right_poses',self.right.data.body_link_pose_w,None)]
        for i in range(2):
            fields.extend([(f'right_gun_{i}',self.tip_sensors[i].data.normal_force_matrix_w,(1,1,2,3)),
                (f'gun_contacts_{i}',self.gun_sensors[i].data.normal_force_matrix_w,(1,1,3,3)),
                (f'gun_net_{i}',self.gun_sensors[i].data.net_normal_forces_w,(1,1,3))])
        return fields

    def unmatched_right(self,a,names):
        net=a['right_net'].copy()
        for i,name in enumerate(('tip_left','tip_right')):
            if names.count(name)!=1:raise ValueError('Right tip contact body is ambiguous.')
            partner=1 if i==0 else 0
            net[0,names.index(name)]-=a[f'right_gun_{i}'][0,0,partner]
        return net

    def measurement(self,a,row,phase):
        from yam_arena.constants import TCP_OFFSET
        gun=a['gun_poses'][0,self.gun_body];g=_pose(gun,'gun');hand=a['right_poses'][0,self.right_hand]
        tcp=_pose(row['measured_hand_pose_world'],'left hand');tcp[:3,3]+=tcp[:3,:3]@TCP_OFFSET
        forbidden=[]
        for i in range(2):
            matrix=a[f'gun_contacts_{i}'][0,0];net=a[f'gun_net_{i}'][0,0]
            allowed=(1,2)if i==0 else(0,)
            residual=float(np.linalg.norm(net-sum((matrix[k]for k in allowed),np.zeros(3))))
            if residual>1e-5:forbidden.append(['gun_body'if i==0 else'trigger','unmatched_partner',residual])
        q=float(a['gun_q'][0,self.trigger_id]);v=float(a['gun_v'][0,self.trigger_id])
        return dict(gun_pose_world=gun.tolist(),right_hand_pose_world=hand.tolist(),
            measured_tcp_pose_world=[*tcp[:3,3],*row['measured_hand_pose_world'][3:]],
            commanded_arm_q_rad=a['left_targets'][0,self.left_arm_ids].tolist(),
            commanded_right_arm_q_rad=a['right_targets'][0,self.right_arm_ids].tolist(),
            commanded_right_finger_q_m=a['right_targets'][0,self.right_finger_ids].tolist(),
            measured_right_arm_q_rad=a['right_q'][0,self.right_arm_ids].tolist(),
            trigger_compression_m=q,trigger_velocity_m_s=v,trigger_spring_force_N=-1000*q-2*v,
            air_on=q>=self.gun['air_on_compression_m'],scan_row=self.scan_row,
            gun_grip_normal_force_N=[float(np.linalg.norm(a['right_gun_0'][0,0,1])),float(np.linalg.norm(a['right_gun_1'][0,0,0]))],
            nozzle_tip_world_m=(g[:3,3]+g[:3,:3]@self.gun['nozzle_local_m']).tolist(),nozzle_axis_world=g[:3,0].tolist(),
            unexpected_gun_right_contacts=forbidden)

    def guard(self,row):
        if abs(row['trigger_spring_force_N'])>6 or abs(row['trigger_velocity_m_s'])>1.:
            raise native.TrialAbort('Passive trigger spring force or speed exceeded its declared diagnostic bound.')
        if any(c[2]>1. for c in row['unexpected_gun_right_contacts']):raise native.TrialAbort('Unexpected gun contact exceeded1N.')
        from wafer_tool_return import RELEASE_PHASES
        if row['phase']=='gun_hold' or (getattr(self,'_gun_hold_reference',None) is not None and row['phase']not in RELEASE_PHASES):
            relative=np.linalg.inv(_pose(row['right_hand_pose_world'],'hand'))@_pose(row['gun_pose_world'],'gun')
            center=np.asarray(self.gun['grip_local_m']).copy();center[0]-=row['trigger_compression_m']/2
            point=relative[:3,3]+relative[:3,:3]@center
            if row['phase']=='gun_hold':self._gun_hold_reference=(point.copy(),relative[:3,:3].copy())
            else:
                reference,rotation=self._gun_hold_reference
                shift=float(np.linalg.norm(point-reference));angle=math.degrees(_angle(relative[:3,:3],rotation))
                if shift>GUN_RETENTION['maximum_grip_translation_m']or angle>GUN_RETENTION['maximum_relative_rotation_deg']:
                    raise native.TrialAbort(f'Gun slipped in hand: {shift*1000:.3f} mm, {angle:.3f} deg.')


    def run(self,env,base,metadata,recorder,previous_action,preload,previous_fingers,destination,torch,prefix):
        from yam_arena import ik,station,constants
        start=time.monotonic();action=np.array(previous_action);records=[];gun_servo=None;state={}
        timing=gun_motion_timing(metadata)
        if timing:metadata['gun_motion_timing_profile']=dict(timing)
        origin=native._array(base.scene.env_origins,(1,3))[0]
        def tcp(q,side='left'):return ik.tcp_pose(q,np.asarray(station.BASES[side])+origin)
        def solve_ik(p,r,q,side='left',damping=.012):
            return ik.solve(p,ik.matrix_to_quat_xyzw(r),q,base_pos=np.asarray(station.BASES[side])+origin,iters=600,damping=damping,tol=1e-7)
        def solve_pose(p,r,q,side='left'):
            value,pe,ae=solve_ik(p,r,q,side)
            if pe>.0001 or ae>.001:raise ValueError('Cycle IK exceeds its bounds.')
            return value
        def tick(phase,diagnostic):
            if time.monotonic()-start>7200:raise RuntimeError('Cycle operational wall budget exceeded.')
            self.preload_reference['control_index']=diagnostic.get('left_preload',{}).get('control_index')
            recorder.begin(phase);env.step(torch.tensor([action.tolist()],device=base.device,dtype=torch.float32));row=recorder.end()
            records.append(dict(time_s=row['time_s']-.05,phase=phase,action=action.tolist(),**diagnostic))
            if len(records)%20==0:print(f"Cycle {row['time_s']:.1f}s: {phase}; trigger {row['trigger_compression_m']*1000:.2f}mm",flush=True)
            return row
        qgrip=np.array(self.gun['grasp_q_rad']);p,r=tcp(qgrip,'right');qlift=solve_pose(p+[0,0,.025],r,qgrip,'right')
        def motion():
            for phase,duration,q,f in [('gun_settle',2.,action[7:13].copy(),-.035),('gun_approach',4.,qgrip,-.035),
                ('gun_close',4.,qgrip,GUN_OFF_GRIP_M),('gun_hold',2.,qgrip,GUN_OFF_GRIP_M),('gun_lift',4.,qlift,GUN_OFF_GRIP_M),('gun_lift_hold',2.,qlift,GUN_OFF_GRIP_M)]:
                yield phase,(dict(timing['pickup_phases'])[phase]if timing else duration),q,f,-1
            seed=action[7:13].copy()if timing else recorder.rows[-1]['measured_right_arm_q_rad']
            commanded=None
            if timing:
                p,r=tcp(seed,'right');commanded=np.eye(4);commanded[:3,3]=p;commanded[:3,:3]=r
            yield from plan_scan(lambda:recorder.rows[-1],seed,self.gun,solve_pose,
                ik.slerp_matrix,constants.TCP_OFFSET,state,timing,commanded)
        try:
            for phase,duration,target,grip,row_id in motion():
                preload.phases.append((phase,duration));self.scan_row=row_id
                q0=action[7:13].copy();f0=action[13]*constants.FINGER_OPEN
                for i in range(round(duration*20)):
                    features=preceding_interval_features(recorder.rows,preload.command_index,metadata['physics_dt_s'])
                    previous_fingers,diagnostic=preload.step(phase,features,previous_fingers);action[6]=previous_fingers[0]/constants.FINGER_OPEN
                    s=(i+1)/(duration*20);u=s*s*(3-2*s)
                    nominal=target[i]if target.ndim==2 else target.copy()if phase in ('air_on','air_scan')else q0+(target-q0)*u
                    correction=None
                    if phase in ('air_on','air_scan'):
                        if gun_servo is None:
                            relation=state['gun_to_tcp'];inverse=np.linalg.inv(relation)
                            def fk(q):
                                p,r=tcp(q,'right');return p+r@inverse[:3,3],r@inverse[:3,:3]
                            def solve(p,r,q):return solve_ik(p+r@relation[:3,3],r@relation[:3,:3],q,'right')
                            gun_servo=GunFeedback(fk,solve,ik.LOWER,ik.UPPER)
                        g=_pose(recorder.rows[-1]['gun_pose_world'],'gun')
                        nominal,correction=gun_servo.step(nominal,g[:3,3],g[:3,:3],action[7:13],integrate=not timing or phase=='air_on')
                    if timing and np.max(np.abs(nominal-action[7:13]))>math.radians(1. if phase in ('air_on','air_scan','air_off')else timing['maximum_joint_step_deg']):
                        raise native.TrialAbort('Right-hand command jump exceeded the phase-specific bound.')
                    action[7:13]=nominal;action[13]=(f0+(grip-f0)*u)/constants.FINGER_OPEN
                    tick(phase,dict(left_preload=diagnostic,gun_feedback=correction,scan_row=row_id))
            self.scan_row=-1
            plan=plan_return(recorder.rows,solve_pose,tcp,metadata)
            for phase,duration,target,fingers in plan:
                q0=action[:6].copy();f0=action[6]*constants.FINGER_OPEN
                if phase=='return_hold':
                    servo,nominal=make_return_feedback(recorder.rows,q0,tcp,
                        lambda p,r,q:solve_ik(p,r,q,damping=.015),ik.LOWER,ik.UPPER)
                for i in range(round(duration*20)):
                    s=(i+1)/(duration*20);u=s*s*(3-2*s);correction=None
                    if phase=='return_hold':
                        w=_pose(recorder.rows[-1]['wafer_pose_world'],'wafer')
                        action[:6],correction=servo.step(nominal,w[:3,3],w[:3,:3],action[:6],integrate=True)
                    else:action[:6]=q0+(target-q0)*u
                    action[6]=(f0+(fingers[0]-f0)*u)/constants.FINGER_OPEN
                    tick(phase,dict(return_feedback=correction))
                if phase=='return_hold':
                    held=action[:6].copy();p,r=tcp(held);offset=retreat_offset(recorder.rows)
                    away=solve_pose(p+offset,r,held);raised=solve_pose(p+offset+[0,0,.030],r,away)
                    for index,q in zip(range(2,6),(held,away,raised,raised)):
                        name,duration,_,f=plan[index];plan[index]=(name,duration,q,f)
        finally:
            with (Path(destination)/'cycle_commands.jsonl').open('w')as stream:
                for record in records:stream.write(json.dumps(record,allow_nan=False)+'\n')
            metadata['cycle_completed_control_actions']=len(records)
            metadata['final_trigger_drive']=native.native_mechanism_readback(self.gunasset,self.export['passive_spring'],0.,expected_dof_count=1)
        from wafer_full_cycle_isaac import compact_evaluation_rows
        result=evaluate_native_cycle(metadata,compact_evaluation_rows(recorder.rows))
        result.update(prefix_result=prefix['result'],backend='Isaac Lab / PhysX',extension_wall_seconds=time.monotonic()-start)
        return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--bundle',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();native.run_isaac_trial(args.bundle,args.output,extension=Cycle())
