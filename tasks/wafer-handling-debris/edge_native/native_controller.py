"""Joint-target-only MCP2 workflow for the native bimanual YAM scene."""
from __future__ import annotations

import math
import numpy as np


PROFILE = dict(control_dt_s=.05, maximum_duration_s=120., maximum_joint_step_deg=1.,
    ik_position_tolerance_m=.0001, ik_angle_tolerance_rad=.001,
    carry_half_gap_m=.0185, squeeze_half_gap_m=.0155, open_half_gap_m=.035,
    tool_lift_m=.035, wafer_lift_m=.020, insertion_distance_m=.022,insertion_retraction_m=.0002,
    minimum_open_gap_m=.0018, lower_land_top_m=.002, flat_to_tool_x_m=.034,
    insertion_under_clearance_m=.0005, minimum_finger_force_N=.05,
    minimum_land_force_N=.05, maximum_tool_slip_m=.002,
    maximum_tool_rotation_deg=3., maximum_wafer_slip_m=.001,
    maximum_wafer_rotation_deg=2.)


def pose(value):
    a=np.asarray(value,dtype=float)
    if a.shape==(4,4):
        if not np.isfinite(a).all():raise ValueError('Non-finite pose.')
        return a.copy()
    if a.shape!=(7,)or not np.isfinite(a).all():raise ValueError('Expected XYZ/XYZW pose.')
    x,y,z,w=a[3:]/np.linalg.norm(a[3:]);t=np.eye(4);t[:3,3]=a[:3]
    t[:3,:3]=[[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
        [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
        [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]]
    return t


def angle(a,b):
    return math.degrees(math.acos(float(np.clip((np.trace(a.T@b)-1)/2,-1,1))))


class MechanicalPickController:
    def __init__(self,read_row,step_action,initial_action,solve_pose,fk_pose,slerp,
                 grasp_map,finger_open_q,tcp_offset,*,metadata=None,feedback_factory=None):
        self.read_row,self.step_action=read_row,step_action
        self.solve_pose,self.fk_pose,self.slerp=solve_pose,fk_pose,slerp
        self.grasp_map=grasp_map;self.finger_open_q=float(finger_open_q)
        self.tcp_offset=np.asarray(tcp_offset,dtype=float)
        self.action=np.asarray(initial_action,dtype=float).copy()
        if self.action.shape!=(14,)or not np.isfinite(self.action).all():raise ValueError('Expected14 robot targets.')
        if self.finger_open_q>=0:raise ValueError('Expected negative native finger-open position.')
        self.metadata=metadata if metadata is not None else {}
        self.rows=[dict(read_row())];self.commands=[];self.count=0
        self.half_gap=PROFILE['open_half_gap_m'];self.tool_correction=np.eye(4)
        self.tool_reference=None;self.wafer_reference=None;self.left_hold=None
        self.feedback=feedback_factory()if feedback_factory else None
        self._started=False
        self.metadata['mcp2_controller']=dict(PROFILE,alignment='Measured settled wafer flat pose; scripted privileged feedback.',
            actuation='Only14 YAM joint/gripper targets; no mechanism targets or object state writes.')

    def tcp(self,row=None,side='left'):
        row=self.rows[-1]if row is None else row
        t=pose(row['measured_hand_pose_world'if side=='left'else'right_hand_pose_world'])
        t[:3,3]+=t[:3,:3]@self.tcp_offset
        return t

    def _ik(self,t,seed,side='left'):
        q=np.asarray(self.solve_pose(t[:3,3],t[:3,:3],seed,side),dtype=float)
        if q.shape!=(6,)or not np.isfinite(q).all():raise RuntimeError('Invalid native IK result.')
        p,r=self.fk_pose(q,side)
        if np.linalg.norm(np.asarray(p)-t[:3,3])>PROFILE['ik_position_tolerance_m']or math.radians(angle(np.asarray(r),t[:3,:3]))>PROFILE['ik_angle_tolerance_rad']:
            raise RuntimeError('Native IK residual exceeds fixed bounds.')
        return q

    def _tool_tcp(self,tool,half_gap):
        tcp,finger=self.grasp_map(tool,half_gap)
        return pose(tcp)@self.tool_correction,float(finger)

    def _measure_grasp_map(self):
        tool=pose(self.rows[-1]['clamp_pose_world']);ideal,_=self.grasp_map(tool,self.half_gap)
        self.tool_correction=np.linalg.inv(pose(ideal))@self.tcp()
        self.metadata.setdefault('grasp_frame_calibrations',[]).append(dict(
            time_s=self.count*.05,half_gap_m=self.half_gap,correction=self.tool_correction.tolist()))

    def _tool_relation(self,row):
        tips=row.get('left_tip_poses_world',row.get('measured_left_tip_poses_world'))
        if tips is None:raise ValueError('Actual left fingertip poses are required.')
        fixed=tips['tip_left']if isinstance(tips,dict)else tips[0]
        return np.linalg.inv(pose(fixed))@pose(row['clamp_pose_world'])

    def _guard(self,row):
        if self.tool_reference is not None:
            t=self._tool_relation(row);ref=self.tool_reference
            if np.linalg.norm(t[:3,3]-ref[:3,3])>PROFILE['maximum_tool_slip_m']or angle(t[:3,:3],ref[:3,:3])>PROFILE['maximum_tool_rotation_deg']:
                raise RuntimeError('Measured tool retention exceeded its fixed bounds.')
        if self.wafer_reference is not None:
            t=np.linalg.inv(pose(row['clamp_pose_world']))@pose(row['wafer_pose_world']);ref=self.wafer_reference
            if np.linalg.norm(t[:3,3]-ref[:3,3])>PROFILE['maximum_wafer_slip_m']or angle(t[:3,:3],ref[:3,:3])>PROFILE['maximum_wafer_rotation_deg']:
                raise RuntimeError('Measured wafer retention exceeded its fixed bounds.')

    def _tick(self,phase):
        if (self.count+1)*.05>PROFILE['maximum_duration_s']+1e-9:raise RuntimeError('Whole workflow exceeded120seconds.')
        row=dict(self.step_action(self.action.copy(),phase));self.count+=1
        self.rows.append(row);self._guard(row)
        self.commands.append(dict(time_s=self.count*.05,phase=phase,action=self.action.tolist()))
        return row

    def _path(self,start,target,duration,mapper,seed,side):
        n=max(1,math.ceil(duration*20));limit=math.radians(.90)
        initial,_=mapper(start,0.)
        commanded_p,commanded_r=self.fk_pose(seed,side)
        commanded_p=np.asarray(commanded_p).copy();commanded_r=np.asarray(commanded_r).copy()
        if side=='left'and self.feedback is not None:
            from wafer_grasp_feedback import rotation_matrix
            commanded_p-=self.feedback.translation
            commanded_r=rotation_matrix(self.feedback.rotation).T@commanded_r
        offset=commanded_p-initial[:3,3];rotation_offset=commanded_r@initial[:3,:3].T
        for _ in range(8):
            q=np.asarray(seed).copy();command_q=q.copy();path=[];peak=0.
            for i in range(n):
                u=(i+1)/n;u=u*u*(3-2*u);t=np.eye(4)
                t[:3,3]=start[:3,3]+(target[:3,3]-start[:3,3])*u
                t[:3,:3]=self.slerp(start[:3,:3],target[:3,:3],u)
                tcp,finger=mapper(t,u)
                tcp=tcp.copy();tcp[:3,3]+=(1-u)*offset
                tcp[:3,:3]=self.slerp(rotation_offset,np.eye(3),u)@tcp[:3,:3]
                new=self._ik(tcp,q,side)
                actual=new
                if side=='left'and self.feedback is not None:
                    corrected=tcp.copy();corrected[:3,3]+=self.feedback.translation
                    corrected[:3,:3]=rotation_matrix(self.feedback.rotation)@corrected[:3,:3]
                    actual=self._ik(corrected,command_q,side)
                peak=max(peak,float(np.max(np.abs(actual-command_q))));path.append((new,finger));q=new;command_q=actual
            if peak<=limit+1e-12:return path
            n=max(n+1,math.ceil(n*peak/limit)+1)
        raise RuntimeError('Cartesian path could not satisfy native joint-step limits.')

    def _left_feedback(self,q,integrate):
        if self.feedback is None:return q
        measured=self.tcp();q,_=self.feedback.step(q,measured[:3,3],measured[:3,:3],self.action[:6],integrate=integrate)
        return q

    def move_tcp(self,phase,target,duration,half_gap):
        start=self.tcp();g0=self.half_gap
        def mapper(t,u):
            _,f=self.grasp_map(pose(self.rows[-1]['clamp_pose_world']),g0+(half_gap-g0)*u)
            return t,f
        for q,f in self._path(start,target,duration,mapper,self.action[:6],'left'):
            self.action[:6]=self._left_feedback(q,False);self.action[6]=f/self.finger_open_q;self._tick(phase)
        self.half_gap=half_gap

    def move_tool(self,phase,target,duration,half_gap,*,hold=False):
        start=pose(self.rows[-1]['clamp_pose_world']);g0=self.half_gap
        def mapper(t,u):return self._tool_tcp(t,g0+(half_gap-g0)*u)
        for q,f in self._path(start,target,duration,mapper,self.action[:6],'left'):
            self.action[:6]=self._left_feedback(q,hold);self.action[6]=f/self.finger_open_q;self._tick(phase)
        self.half_gap=half_gap

    def hold(self,phase,duration):
        for _ in range(round(duration*20)):self._tick(phase)

    def _contact_gate(self,key,minimum,label):
        tail=self.rows[-20:];values=np.asarray([r[key]for r in tail],dtype=float)
        if len(tail)<20 or values.shape!=(20,2)or not np.isfinite(values).all()or np.min(np.mean(values,axis=0))<minimum:
            raise RuntimeError(label+' contact gate failed before lifting.')

    def pickup_wafer(self,stop_after=None):
        carry=PROFILE['carry_half_gap_m'];squeeze=PROFILE['squeeze_half_gap_m']
        self.hold('tool_settle',1.)
        tool=pose(self.rows[-1]['clamp_pose_world']);self.initial_wafer=pose(self.rows[-1]['wafer_pose_world'])
        target_tcp,_=self.grasp_map(tool,carry)
        self.move_tcp('tool_approach',pose(target_tcp),3.,self.half_gap)
        self.move_tcp('tool_close',pose(target_tcp),2.,carry);self.hold('tool_grip_hold',1.)
        self._contact_gate('finger_handle_contact_normal_force_N',.05,'Tool')
        tool=pose(self.rows[-1]['clamp_pose_world']);ideal,_=self.grasp_map(tool,carry)
        self.tool_correction=np.linalg.inv(pose(ideal))@self.tcp();self.tool_reference=self._tool_relation(self.rows[-1])
        if stop_after=='grip':return self.result('STOPPED_AFTER_GRIP')
        target=tool.copy();target[2,3]+=PROFILE['tool_lift_m']
        self.move_tool('tool_lift',target,2.,carry);self.move_tool('tool_lift_hold',target,1.,carry,hold=True)
        if stop_after=='tool_lift':return self.result('STOPPED_AFTER_TOOL_LIFT')
        self.move_tool('tool_open',target,2.,squeeze)
        if float(self.rows[-1]['jaw_gap_m'])<PROFILE['minimum_open_gap_m']:raise RuntimeError('Mechanical squeeze did not open the jaws.')
        self._measure_grasp_map()
        w=pose(self.rows[-1]['wafer_pose_world']);flat=-math.sqrt(.05**2-(.0325/2)**2)
        flat_bottom=w[:3,3]+w[:3,:3]@np.array([flat,0.,-.0005])
        dock=np.eye(4);dock[:3,:3]=w[:3,:3]
        dock[:3,3]=flat_bottom+w[:3,:3]@np.array([-.034-PROFILE['insertion_retraction_m'],0.,-.002-PROFILE['insertion_under_clearance_m']])
        self.dock=dock.copy();behind=dock.copy();behind[:3,3]-=dock[:3,0]*.022
        raised=behind.copy();raised[2,3]+=.025
        self.move_tool('tool_transport',raised,3.,squeeze)
        self.move_tool('tool_lower_behind',behind,2.,squeeze)
        self.move_tool('tool_alignment_hold',behind,.5,squeeze,hold=True)
        if float(self.rows[-1]['jaw_gap_m'])<PROFILE['minimum_open_gap_m']:raise RuntimeError('Jaw opening was lost before insertion.')
        self.move_tool('tool_insert',dock,2.,squeeze)
        if np.linalg.norm(pose(self.rows[-1]['wafer_pose_world'])[:3,3]-self.initial_wafer[:3,3])>.00075:
            raise RuntimeError('Wafer moved during insertion before intended seating.')
        self._measure_grasp_map();seat=pose(self.rows[-1]['clamp_pose_world']);contact_count=0
        for _ in range(40):
            seat[2,3]+=.000025
            self.move_tool('seat_lower_pad',seat,.05,squeeze,hold=True)
            contact_count=contact_count+1 if self.rows[-1]['wafer_land_contact_normal_force_N'][0]>.03 else 0
            if contact_count>=3:break
        else:raise RuntimeError('Lower wafer pad did not seat within1mm of upward travel.')
        dock=pose(self.rows[-1]['clamp_pose_world']);self._measure_grasp_map()
        self.move_tool('wafer_close',dock,2.,carry);self.move_tool('wafer_grip_hold',dock,1.,carry,hold=True)
        self._contact_gate('wafer_land_contact_normal_force_N',.05,'Wafer')
        self.wafer_reference=np.linalg.inv(pose(self.rows[-1]['clamp_pose_world']))@pose(self.rows[-1]['wafer_pose_world'])
        target=pose(self.rows[-1]['clamp_pose_world']);target[2,3]+=.020
        self.move_tool('wafer_lift',target,2.,carry);self.move_tool('wafer_lift_hold',target,1.,carry,hold=True)
        if pose(self.rows[-1]['wafer_pose_world'])[2,3]-self.initial_wafer[2,3]<.018:raise RuntimeError('Measured wafer lift is below18mm.')
        self.left_hold=pose(self.rows[-1]['clamp_pose_world'])
        return self.result('WAFER_LIFTED')

    def return_wafer(self):
        self.left_hold=None;carry=PROFILE['carry_half_gap_m'];squeeze=PROFILE['squeeze_half_gap_m']
        w=pose(self.rows[-1]['wafer_pose_world']);tool=pose(self.rows[-1]['clamp_pose_world'])
        target=self.initial_wafer@np.linalg.inv(w)@tool
        self.move_tool('wafer_return',target,3.,carry);self.move_tool('wafer_return_hold',target,1.,carry,hold=True)
        self.wafer_reference=None
        target=pose(self.rows[-1]['clamp_pose_world']);self._measure_grasp_map()
        self.move_tool('wafer_release',target,2.,squeeze)
        if float(self.rows[-1]['jaw_gap_m'])<PROFILE['minimum_open_gap_m']:raise RuntimeError('Mechanical release opening is insufficient.')
        target=pose(self.rows[-1]['clamp_pose_world']);self._measure_grasp_map()
        for attempt in range(31):
            gaps=self._wafer_pad_clearances()
            if gaps[0]>=.0003 and min(self.rows[-1]['pillar_normal_force_N'])>.005:break
            if attempt==30 or gaps[1]<.00035:raise RuntimeError('Cannot clear the lower pad while keeping the upper pad free.')
            target[:3,3]-=pose(self.rows[-1]['wafer_pose_world'])[:3,2]*.00002
            self.move_tool('wafer_pad_clear',target,.05,squeeze,hold=True)
        self.metadata['wafer_release_clearance_m']=gaps
        away=pose(self.rows[-1]['clamp_pose_world']);self._measure_grasp_map()
        away[:3,3]-=away[:3,0]*.025
        self.move_tool('wafer_withdraw',away,2.,squeeze)
        self.move_tool('restore_carry_grip',away,2.,carry)
        away[2,3]+=.020;self.move_tool('empty_tool_retreat',away,2.,carry);self.hold('unassisted_hold',1.05)
        tail=self.rows[-21:]
        if max(np.linalg.norm(pose(r['wafer_pose_world'])[:3,3]-self.initial_wafer[:3,3])for r in tail)>.001:
            raise RuntimeError('Wafer return position exceeds1mm.')
        if max(angle(pose(r['wafer_pose_world'])[:3,:3],self.initial_wafer[:3,:3])for r in tail)>2.:
            raise RuntimeError('Wafer return orientation exceeds2degrees.')
        if any(min(r['pillar_normal_force_N'])<=.005 or max(r['wafer_land_contact_normal_force_N'])>=.01 for r in tail):
            raise RuntimeError('Wafer is not released and supported by all three pins.')
        self.left_hold=pose(self.rows[-1]['clamp_pose_world'])

    def _wafer_pad_clearances(self):
        row=self.rows[-1];wafer=pose(row['wafer_pose_world']);gaps=[]
        bottom=self.metadata['wafer_local_bottom_z_m'];top=bottom+self.metadata['wafer_thickness_m']
        for i,face in enumerate(self.metadata['contact_faces_local_m']):
            link=pose(row['jaw_link_poses_world'][i])
            local=(np.asarray(face)@link[:3,:3].T+link[:3,3]-wafer[:3,3])@wafer[:3,:3]
            gaps.append(float(bottom-local[:,2].max()if i==0 else local[:,2].min()-top))
        return gaps

    def _maintain_left(self):
        if self.left_hold is None:return
        t,f=self._tool_tcp(self.left_hold,PROFILE['carry_half_gap_m']);q=self._ik(t,self.action[:6])
        self.action[:6]=self._left_feedback(q,True);self.action[6]=f/self.finger_open_q

    def _gun_feedback(self,relation):
        from wafer_cycle_protocol import GunFeedback
        from yam_arena import ik
        inverse=np.linalg.inv(relation)
        def fk(q):
            p,r=self.fk_pose(q,'right')
            return p+r@inverse[:3,3],r@inverse[:3,:3]
        def solve(p,r,q):
            t=np.eye(4);t[:3,3]=p+r@relation[:3,3];t[:3,:3]=r@relation[:3,:3]
            value=self._ik(t,q,'right');actual_p,actual_r=self.fk_pose(value,'right')
            return value,float(np.linalg.norm(actual_p-t[:3,3])),math.radians(angle(actual_r,t[:3,:3]))
        return GunFeedback(fk,solve,ik.LOWER,ik.UPPER)

    def _gun_force_rows(self):
        rows=[]
        for row in self.rows[-20:]:
            interval=row['gun_contact_interval']
            if interval['count']!=100 or not np.isfinite(interval['duration_s']) or abs(interval['duration_s']-.05)>1e-10:
                raise RuntimeError('Gun support needs a complete50ms contact interval.')
            values=[interval['stand_vertical_mean_N'],interval['stand_normal_peak_N'],*interval['grip_mean_N']]
            if len(values)!=4 or not np.isfinite(values).all():raise RuntimeError('Invalid gun contact interval forces.')
            rows.append(dict(row,gun_stand_vertical_force_N=interval['stand_vertical_mean_N'],
                gun_stand_normal_force_N=interval['stand_normal_peak_N'],
                gun_grip_normal_force_N=interval['grip_mean_N']))
        return rows

    def execute_right(self,phase,duration,target,finger,*,servo=None,seat_load=False):
        from wafer_tool_return import gun_seat_increment,release_ready,RETURN
        target=np.asarray(target);q0=self.action[7:13].copy();f0=self.action[13]*self.finger_open_q;seat_command=None
        steps=round(duration*20)
        if steps<1 or (target.ndim==2 and target.shape!=(steps,6)):raise ValueError('Gun path duration/targets disagree.')
        for i in range(steps):
            if seat_load:
                ready=release_ready(self.metadata,self._gun_force_rows(),.05)
                if ready['passed']:
                    self.metadata['gun_return']['release_gate']=ready
                    return
            u=(i+1)/steps;u=u*u*(3-2*u)
            q=target[i].copy()if target.ndim==2 else q0+(target-q0)*u
            if servo is not None:
                if seat_command is not None:q=seat_command.copy()
                else:
                    current=self.rows[-1];g=pose(current['gun_pose_world'])
                    q,_=servo.step(q,g[:3,3],g[:3,:3],self.action[7:13],integrate=phase!='air_scan')
                    if phase=='gun_return_lower':
                        weight=(self.metadata['gun']['mass_kg']+self.metadata['gun']['trigger_mass_kg'])*9.81
                        support=current['gun_contact_interval']['stand_vertical_mean_N']
                        if support>=.20*weight and abs(current['gun_foot_gap_m'])<=RETURN['maximum_absolute_foot_gap_m']:seat_command=q.copy()
            if seat_load:
                delta,_=gun_seat_increment(self.metadata,self._gun_force_rows(),.05)
                self.gun_seat_shift+=delta
                if np.linalg.norm(self.gun_seat_shift)>RETURN['maximum_load_command_shift_m']:raise RuntimeError('Gun seating travel exceeded its limit.')
                p,r=self.fk_pose(self.action[7:13],'right');t=np.eye(4);t[:3,3]=p+delta;t[:3,:3]=r;q=self._ik(t,self.action[7:13],'right')
            transport=phase in('gun_return_raise','gun_return_transfer','gun_return_upright','gun_return_above')
            limit=1. if phase in('air_on','air_scan','air_off')or (phase.startswith('gun_return')and not transport)else 1.6
            if np.max(np.abs(q-self.action[7:13]))>math.radians(limit):raise RuntimeError('Right-arm command step exceeds phase limit.')
            self.action[7:13]=q;self.action[13]=(f0+(finger-f0)*u)/self.finger_open_q
            self._maintain_left();self._tick(phase)

    def run_gun(self):
        from wafer_cycle_protocol import FULL_GUN_TIMING,GUN_OFF_GRIP_M,plan_scan
        from wafer_tool_return import RETURN,plan_gun_return,plan_gun_release,release_ready,retreat_ready
        from wafer_finger_preload_v12 import FULL_WORKFLOW_TIMING_PROFILE
        self.metadata['full_workflow_timing_profile']=dict(FULL_WORKFLOW_TIMING_PROFILE)
        gun=self.metadata['gun'];self.metadata['gun_return']=dict(profile=dict(RETURN),reference_pose_world=list(self.rows[-1]['gun_pose_world']))
        qgrip=np.asarray(gun['grasp_q_rad']);p,r=self.fk_pose(qgrip,'right');t=np.eye(4);t[:3,3]=p+[0,0,.025];t[:3,:3]=r
        qlift=self._ik(t,qgrip,'right')
        targets=[self.action[7:13].copy(),qgrip,qgrip,qgrip,qlift,qlift]
        for (phase,duration),q in zip(FULL_GUN_TIMING['pickup_phases'],targets):
            self.execute_right(phase,duration,q,-.035 if phase in('gun_settle','gun_approach')else GUN_OFF_GRIP_M)
        state={};scan_servo=None;p,r=self.fk_pose(self.action[7:13],'right');commanded=np.eye(4);commanded[:3,3]=p;commanded[:3,:3]=r
        for phase,duration,target,finger,row_id in plan_scan(lambda:self.rows[-1],self.action[7:13],gun,self.solve_pose,self.slerp,self.tcp_offset,state,FULL_GUN_TIMING,commanded):
            self.metadata['scan_row']=row_id
            if phase in('air_on','air_scan')and scan_servo is None:scan_servo=self._gun_feedback(state['gun_to_tcp'])
            self.execute_right(phase,duration,target,finger,servo=scan_servo if phase in('air_on','air_scan')else None)
        self.metadata['scan_row']=-1
        if self.rows[-1]['air_on']:raise RuntimeError('Gun trigger remains on after the scan.')
        self.return_wafer()
        self.execute_right('gun_return_start',.25,self.action[7:13].copy(),RETURN['close_target_m'])
        p,r=self.fk_pose(self.action[7:13],'right');commanded=np.eye(4);commanded[:3,3]=p;commanded[:3,:3]=r;state={};return_servo=None
        for phase,duration,target,finger in plan_gun_return(self.metadata,lambda:self.rows[-1],self.action[7:13],self.solve_pose,self.slerp,self.tcp_offset,state,commanded):
            if phase in('gun_return_align','gun_return_lower')and return_servo is None:return_servo=self._gun_feedback(state['gun_to_tcp'])
            self.execute_right(phase,duration,target,finger,servo=return_servo)
        self.gun_seat_shift=np.zeros(3)
        self.execute_right('gun_return_seat_hold',FULL_GUN_TIMING['seat_hold_s'],self.action[7:13].copy(),RETURN['seat_target_m'],seat_load=True)
        for attempt in range(7):
            ready=release_ready(self.metadata,self._gun_force_rows(),.05)
            self.metadata['gun_return']['release_gate']=ready
            if ready['passed']:break
            if attempt==6:raise RuntimeError('Gun is not supported before release: '+str(ready))
            self.execute_right('gun_return_seat_hold',.5,self.action[7:13].copy(),RETURN['seat_target_m'],seat_load=True)
        for phase,duration,target,finger in plan_gun_release(self.action[7:13],self.solve_pose,lambda q:self.fk_pose(q,'right'),self.metadata):
            self.execute_right(phase,duration,target,finger)
            if phase=='gun_return_release_hold':
                ready=retreat_ready(self.metadata,self._gun_force_rows(),.05)
                self.metadata['gun_return']['retreat_gate']=ready
                if not ready['passed']:raise RuntimeError('Gun release did not transfer weight to the stand: '+str(ready))

    def run(self,stop_after=None,with_gun=True):
        if self._started:raise RuntimeError('Create a fresh controller for each reset episode.')
        self._started=True
        result=self.pickup_wafer(stop_after)
        if stop_after in('grip','tool_lift','wafer_lift'):return result
        if with_gun:self.run_gun()
        else:self.return_wafer()
        return self.result('COMPLETED')

    def run_tool_pickup(self):return self.run(stop_after='tool_lift')

    def run_wafer_cycle(self):return self.run(with_gun=False)

    def run_full(self,gun_spec=None,workflow_metadata=None):
        if workflow_metadata:self.metadata.update(workflow_metadata)
        if gun_spec:self.metadata['gun']=gun_spec
        return self.run(with_gun=True)

    def result(self,status):
        return dict(status=status,control_steps=self.count,simulated_seconds=self.count*.05,
            scope='Native YAM joint targets and passive contact-held tools; outcome requires independent recorded-state scoring.')


def from_yam(read_row,step_action,initial_action,grasp_map,metadata,env_origin=(0,0,0)):
    from yam_arena import ik,station,constants
    from wafer_grasp_feedback import BoundedTCPFeedback
    origin=np.asarray(env_origin)
    def fk(q,side='left'):return ik.tcp_pose(q,np.asarray(station.BASES[side])+origin)
    def raw_solve(p,r,q,side='left'):
        return ik.solve(p,ik.matrix_to_quat_xyzw(r),q,base_pos=np.asarray(station.BASES[side])+origin,iters=600,damping=.012,tol=1e-7)
    def solve(p,r,q,side='left'):
        value,pe,ae=raw_solve(p,r,q,side)
        if pe>.0001 or ae>.001:raise RuntimeError('Native IK failed.')
        return value
    return MechanicalPickController(read_row,step_action,initial_action,solve,fk,ik.slerp_matrix,grasp_map,
        constants.FINGER_OPEN,constants.TCP_OFFSET,metadata=metadata,
        feedback_factory=lambda:BoundedTCPFeedback(lambda q:fk(q),lambda p,r,q:raw_solve(p,r,q),ik.LOWER,ik.UPPER))


def build_yam_controller(adapter,read_row,step_action,initial_action,env_origin=(0,0,0),baseline_plan=None):
    metadata={}if baseline_plan is None else dict(baseline_plan)
    return from_yam(read_row,step_action,initial_action,adapter.grasp_map,metadata,env_origin)
