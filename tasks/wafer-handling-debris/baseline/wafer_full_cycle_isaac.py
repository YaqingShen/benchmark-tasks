"""Continuous native workflow and explicit stage-local controller clocks."""
from pathlib import Path
import argparse, copy, hashlib, json, math, time, sys
from collections import deque
import numpy as np
import wafer_retention_isaac_v12 as native
from wafer_cycle_isaac import Cycle
from wafer_grasp_protocol import _pose, _angle, DEFAULT_LIMITS, pillar_plane_clearances
from wafer_trace_rows import TraceRowList
from wafer_finger_preload_v12 import workflow_timing,wafer_prefix_phases

_HERE=Path(__file__).resolve().parent
_BENCHMARK=_HERE/'benchmark'if(_HERE/'benchmark').is_dir()else _HERE.parent/'benchmark'
sys.path.insert(0,str(_BENCHMARK))
from wafer_task_score import WaferTaskScorer,TASK_QUALIFICATION

SUCCESS='FULL_WORKFLOW_SUCCEEDED_IN_THIS_SIMULATION'


def workflow_duration_check(metadata,end_time_s):
    profile=workflow_timing(metadata)
    if profile is None:raise ValueError('Missing full workflow timing profile.')
    elapsed=float(end_time_s);maximum=float(profile['maximum_simulation_seconds'])
    if not math.isfinite(elapsed)or elapsed<0:raise ValueError('Invalid global simulation duration.')
    return dict(passed=elapsed<=maximum+1e-8,simulation_seconds=elapsed,maximum_simulation_seconds=maximum,
        scope='Entire measured workflow, including actions after task milestones; no deadline extension after early completion.')

TOOL_HAND_PROFILE=dict(revision='contact_retention_window_v4',maximum_translation_m=.002,maximum_rotation_deg=3.,
    window_s=.050,hard_translation_m=.003,hard_rotation_deg=5.,
    reference='Last tool_grip_hold sample; retained through docking, wafer grasp and gun return',position='Midpoint of the two measured contacting paddle centers',
    orientation='Each actual contacting paddle relative to the hand; lever rotation is diagnostic')

EVALUATION_FIELDS=frozenset('air_on clamp_pose_world commanded_arm_q_rad commanded_finger_q_m finger_handle_contact_normal_force_N finger_handle_partner_normal_force_N forbidden_wafer_contact_force_N gun_grip_normal_force_N gun_pose_world hinge_angle_rad hinge_velocity_rad_s jaw_contact_normal_force_N jaw_link_poses_world measured_jaw_gap_m mechanism_joint_velocity measured_tcp_pose_world nozzle_axis_world nozzle_tip_world_m pad_compression_m pad_compression_velocity_m_s passive_hinge_torque_Nm passive_pad_spring_force_N phase pillar_contact_normal_force_N right_hand_pose_world robot_contact_partners_N scan_row time_s trigger_compression_m trigger_spring_force_N trigger_velocity_m_s unexpected_gun_right_contacts unexpected_robot_environment_contact_normal_force_N unexpected_robot_tool_contact_normal_force_N wafer_pose_world'.split())
GUARD_FIELDS=frozenset('time_s phase handle_paddle_q_rad handle_paddle_qvel_rad_s handle_paddle_passive_torque_Nm finger_handle_contact_normal_force_N finger_handle_partner_normal_force_N pad_compression_m pad_compression_velocity_m_s passive_pad_spring_force_N hinge_angle_rad hinge_velocity_rad_s passive_hinge_torque_Nm tool_unexpected_environment_contact_normal_force_N tool_unexpected_environment_contacts_N'.split())


def compact_evaluation_rows(rows):
    return [{key:row[key]for key in EVALUATION_FIELDS if key in row}for row in rows]


def compact_guard_rows(rows):
    return [{key:row[key]for key in GUARD_FIELDS if key in row}for row in rows]


def tool_environment_contact_evidence(specs, arrays, stands, stand_allowed, particle_partners=()):
    bodies={s['body']for s in specs};particles=set(particle_partners)
    unexpected=0.;contacts=[];unmatched=[];debris=[]
    for index,spec in enumerate(specs):
        matrix=arrays[f'tool_environment_matrix_{index}'][0,0]
        net=arrays[f'tool_environment_net_{index}'][0,0]
        residual=float(np.linalg.norm(net-matrix.sum(axis=0)))
        unmatched.append(residual);unexpected+=residual
        if residual:contacts.append([spec['body'],'unmatched_environment',residual])
        for partner,force in zip(spec['partners'],np.linalg.norm(matrix,axis=1)):
            if partner in bodies or partner=='wafer'or partner.startswith('left_tip_'):continue
            if partner in stands and stand_allowed:continue
            if partner in particles:
                if force:debris.append([spec['body'],partner,float(force)])
                continue
            unexpected+=float(force)
            if force:contacts.append([spec['body'],partner,float(force)])
    return dict(tool_unexpected_environment_contact_normal_force_N=unexpected,
        tool_unexpected_environment_contacts_N=contacts,tool_environment_unmatched_normal_force_N=unmatched,
        tool_particle_contacts_N=debris,tool_particle_normal_force_N=sum(c[2]for c in debris),
        tool_environment_contact_scope='Named debris contacts are allowed after air starts. Fixtures, gun and unmatched normal-vector residuals remain forbidden.')


def evaluate_tool_environment(rows):
    peak=0.
    for row in rows:
        value=float(row['tool_unexpected_environment_contact_normal_force_N']);contacts=row['tool_unexpected_environment_contacts_N']
        if not math.isfinite(value)or value<0 or not isinstance(contacts,list):raise ValueError('Invalid tool/environment force evidence.')
        if any(len(c)!=3 or not math.isfinite(c[2])or c[2]<0 for c in contacts):raise ValueError('Invalid tool/environment contact pair.')
        if abs(sum(c[2]for c in contacts)-value)>1e-9:raise ValueError('Tool/environment aggregate differs from contact evidence.')
        peak=max(peak,value)
    return dict(passed=bool(rows)and peak<=.01,physics_sample_count=len(rows),maximum_unexpected_normal_force_N=peak,
        maximum_allowed_normal_force_N=.01,scope='All tool bodies and every stage; native vector residuals can hide cancelling unfiltered contacts.')


class RetentionEnvelope:
    def __init__(self,dt):
        self.dt=float(dt);self.width=round(TOOL_HAND_PROFILE['window_s']/dt)
        if self.width<1 or abs(self.width*dt-TOOL_HAND_PROFILE['window_s'])>1e-9:raise ValueError('Retention window requires an exact physics cadence.')
        self.values=deque();self.total=np.zeros(2);self.maximum_mean=np.zeros(2);self.failed=False;self.count=0;self.excursions=0
    def update(self,translation,rotation):
        value=np.array([translation,rotation],dtype=float)
        if not np.isfinite(value).all()or np.any(value<0):raise ValueError('Invalid retention measurement.')
        self.values.append(value);self.total+=value
        if len(self.values)>self.width:self.total-=self.values.popleft()
        mean=self.total/len(self.values);self.maximum_mean=np.maximum(self.maximum_mean,mean);self.count+=1
        nominal=np.array([TOOL_HAND_PROFILE['maximum_translation_m'],TOOL_HAND_PROFILE['maximum_rotation_deg']])
        self.excursions+=int(np.any(value>nominal))
        self.failed=self.failed or bool(np.any(mean>nominal))or translation>TOOL_HAND_PROFILE['hard_translation_m']or rotation>TOOL_HAND_PROFILE['hard_rotation_deg']
        return not self.failed
    def report(self):
        return dict(passed=self.count>=self.width and not self.failed,window_s=TOOL_HAND_PROFILE['window_s'],
            maximum_mean_translation_m=float(self.maximum_mean[0]),maximum_mean_rotation_deg=float(self.maximum_mean[1]),
            instantaneous_nominal_excursion_samples=self.excursions,scope='50 ms mean retains 2 mm/3 deg bounds; every sample must remain inside 3 mm/5 deg. Fixed reference, no recentering.')


class ToolHandMonitor:
    def __init__(self,metadata):
        self.envelope=RetentionEnvelope(metadata['physics_dt_s'])
        spec=metadata['tool_pickup'];target=_pose(spec['target_tool_pose_world'],'tool')
        grasp=target[:3,:3].T@(np.asarray(spec['target_tcp_position_world_m'])-target[:3,3])
        pivot=np.asarray(metadata['geometry']['tool_original_pivot_m']);original=grasp+pivot
        parts={p['name']:p for p in metadata['geometry']['primitive_parts_original_tool_frame']}
        self.points=[]
        for name in ('lower_robot_handle','upper_robot_handle'):
            part=parts[name];center=np.asarray(part['center']);angle=math.radians(part.get('rotate_y',0.))
            normal=np.array([math.sin(angle),0.,math.cos(angle)])
            point=original.copy();point[2]=center[2]-np.dot(normal[:2],point[:2]-center[:2])/normal[2]
            self.points.append(point-pivot)
        if metadata.get('handle_paddles'):
            specs=metadata['handle_paddles']
            if [s['parent_body']for s in specs]!=['lower_half','upper_half']:raise ValueError('Handle paddle order differs.')
            self.points=[np.asarray(s['position_parent_m'],dtype=float)for s in specs]
        if len(metadata.get('handle_paddles',[]))!=2:raise ValueError('Contact retention requires two measured passive paddles.')
        self.joint_anchors=[list(s.get('joint_anchor_local_m',[0.,0.,0.]))for s in metadata['handle_paddles']]
        self.reference=None;self.reference_lower=None;self.reference_hinge=None;self.reference_paddles=None;self.reference_lever=None
        self.maximum_translation=self.maximum_rotation=self.maximum_hinge=self.maximum_lower_rotation=0.;self.count=0
        self.maximum_paddle_rotation=np.zeros(2);self.maximum_lever_rotation=0.

    def relation(self,row):
        poses=[_pose(p,'lever')for p in row['lever_link_poses_world']];hand=_pose(row['measured_hand_pose_world'],'hand')
        paddles=[_pose(p,'contact paddle')for p in row['handle_paddle_poses_world']]
        if len(paddles)!=2:raise ValueError('Contact retention requires both measured paddle poses.')
        frame=np.eye(4);frame[:3,3]=(paddles[0][:3,3]+paddles[1][:3,3])/2
        u,_,v=np.linalg.svd(paddles[0][:3,:3]+paddles[1][:3,:3]);frame[:3,:3]=u@np.diag([1.,1.,np.linalg.det(u@v)])@v
        u,_,v=np.linalg.svd(poses[0][:3,:3]+poses[1][:3,:3]);lever=u@np.diag([1.,1.,np.linalg.det(u@v)])@v
        inverse=np.linalg.inv(hand)
        return inverse@frame,inverse@poses[0],inverse[:3,:3]@lever,[inverse@p for p in paddles]

    def observe(self,row):
        if row['phase']=='tool_grip_hold':
            self.reference,self.reference_lower,self.reference_lever,self.reference_paddles=self.relation(row)
            self.reference_hinge=float(row['hinge_angle_rad']);return None
        if self.reference is None:return None
        relation,lower,lever,paddles=self.relation(row);self.count+=1
        translation=float(np.linalg.norm(relation[:3,3]-self.reference[:3,3]))
        angles=np.array([math.degrees(_angle(p[:3,:3],r[:3,:3]))for p,r in zip(paddles,self.reference_paddles)])
        rotation=float(max(angles));self.maximum_paddle_rotation=np.maximum(self.maximum_paddle_rotation,angles)
        self.maximum_lever_rotation=max(self.maximum_lever_rotation,math.degrees(_angle(lever,self.reference_lever)))
        self.maximum_translation=max(self.maximum_translation,translation);self.maximum_rotation=max(self.maximum_rotation,rotation)
        self.maximum_hinge=max(self.maximum_hinge,abs(row['hinge_angle_rad']-self.reference_hinge))
        self.maximum_lower_rotation=max(self.maximum_lower_rotation,math.degrees(_angle(lower[:3,:3],self.reference_lower[:3,:3])))
        return self.envelope.update(translation,rotation)

    def report(self):
        return dict(passed=self.envelope.report()['passed'],tolerance=self.envelope.report(),
            profile=TOOL_HAND_PROFILE,physics_sample_count=self.count,maximum_handle_midpoint_shift_m=self.maximum_translation,
            maximum_contact_paddle_rotation_deg=self.maximum_paddle_rotation.tolist(),maximum_contact_rotation_deg=self.maximum_rotation,
            maximum_handle_bisector_rotation_deg=self.maximum_lever_rotation,lever_bisector_is_acceptance_criterion=False,maximum_hinge_change_rad=self.maximum_hinge,
            maximum_lower_lever_rotation_deg=self.maximum_lower_rotation,authored_paddle_origins_in_lever_frames_m=[p.tolist()for p in self.points],
            paddle_joint_anchors_in_body_frames_m=self.joint_anchors,
            scope='Measured contacting paddle retention; authored paddle and scissor rotations remain separate diagnostics and joint-bound checks.')


def evaluate_tool_hand_retention(metadata,rows):
    monitor=ToolHandMonitor(metadata)
    for row in rows:monitor.observe(row)
    return monitor.report()


def evaluate_handle_paddles(metadata,rows,*,operating=False):
    from wafer_tool_pickup import guard_handle_paddles,guard_handle_paddle_capacity
    specs=metadata.get('handle_paddles',[])
    if len(specs)!=2:raise ValueError('Full workflow requires two declared passive handle paddles.')
    peaks={k:np.zeros(2)for k in ('angle_rad','speed_rad_s','raw_torque_Nm')};violations=0
    for row in rows:
        for field,key in [('handle_paddle_q_rad','angle_rad'),('handle_paddle_qvel_rad_s','speed_rad_s'),('handle_paddle_passive_torque_Nm','raw_torque_Nm')]:
            values=np.asarray(row[field],dtype=float)
            if values.shape!=(2,)or not np.isfinite(values).all():raise ValueError('Invalid passive paddle observation: '+field)
            peaks[key]=np.maximum(peaks[key],abs(values))
        expected=[-s['stiffness_Nm_rad']*(q-s['rest_rad'])-s['damping_Nms_rad']*v for s,q,v in zip(specs,row['handle_paddle_q_rad'],row['handle_paddle_qvel_rad_s'])]
        if not np.allclose(row['handle_paddle_passive_torque_Nm'],expected,atol=1e-12,rtol=0):raise ValueError('Passive paddle torque does not match recorded q/qdot.')
        try:(guard_handle_paddle_capacity if operating else guard_handle_paddles)(row,metadata)
        except RuntimeError:violations+=1
    return dict(passed=bool(rows)and violations==0,physics_sample_count=len(rows),violating_samples=violations,
        maximum_absolute={k:v.tolist()for k,v in peaks.items()},joint_names=[s['joint_name']for s in specs],
        accepted_torque_limits_Nm=[min(s['force_cap_Nm'],s.get('operating_torque_guard_Nm',s['force_cap_Nm']))if operating else s['torque_guard_Nm']for s in specs],
        native_numeric_force_caps_Nm=[s['force_cap_Nm']for s in specs],
        scope=('Declared angle/speed/operating-torque envelope; numeric drive caps are reported separately.'if operating else'Legacy 25 mNm diagnostic; not a calibrated task capacity.')+' Unpowered handle hinges; raw SI spring/damper torque from actual joint state.')


def evaluate_task_score(metadata,rows):
    scorer=WaferTaskScorer(metadata,sample_dt_s=float(metadata['physics_dt_s']))
    for row in rows:scorer.update(row,report=False)
    return scorer.result()


def measured_prefix_handoff(row,fk,solve_pose):
    from wafer_grasp_feedback import PROFILE,rotation_vector
    measured=_pose(row['measured_tcp_pose_world'],'measured handoff TCP')
    nominal=np.asarray(solve_pose(measured[:3,3],measured[:3,:3],np.asarray(row['measured_arm_q_rad'])))
    previous=np.asarray(row['commanded_arm_q_rad']);p,r=fk(nominal);command_p,command_r=fk(previous)
    translation=command_p-p;rotation=rotation_vector(command_r@r.T)
    if np.linalg.norm(translation)>PROFILE['maximum_translation_correction_m']or np.linalg.norm(rotation)>PROFILE['maximum_rotation_correction_rad']:
        raise RuntimeError('Measured TCP handoff requires compensation outside original feedback bounds.')
    return nominal,previous,dict(nominal_arm_q_rad=nominal.tolist(),previous_command_arm_q_rad=previous.tolist(),
        measured_tcp_pose_world=list(row['measured_tcp_pose_world']),translation_correction_world_m=translation.tolist(),
        rotation_correction_world_rad=rotation.tolist(),scope='Initialize controller memory from measured handoff; no robot state, clock or acceptance bounds change.')


class StageRecorderView:
    def __init__(self,raw,stage='wafer_cycle'):
        if not raw.rows or raw.active is not None:raise ValueError('Stage begins only at a completed physics boundary.')
        self.raw=raw;self.origin_step=raw.total;self.origin_s=raw.total*raw.dt
        self.initial_index=len(raw.rows)-1;self.stage=stage;self._rows=TraceRowList(cache_blocks=256);self._next=self.initial_index
        self._sync();self._rows[0]=dict(self._rows[0],phase='initialization',finger_preload_command_index=None)

    def _sync(self):
        for source in self.raw.rows[self._next:]:
            row=dict(source);row['global_time_s']=source['time_s'];row['time_s']=source['time_s']-self.origin_s
            self._rows.append(row)
        self._next=len(self.raw.rows)

    @property
    def rows(self):self._sync();return self._rows

    @property
    def total(self):return self.raw.total-self.origin_step

    def begin(self,phase):self.raw.begin(phase)

    def end(self):self.raw.end();self._sync();return self._rows[-1]

    def close(self):self.raw.close()

    def __getattr__(self,name):return getattr(self.raw,name)


def check_wafer_rest(metadata,rows):
    if len(rows)<2:return dict(passed=False,reason='Missing complete gun-return wafer trace.')
    limits=DEFAULT_LIMITS;dt=float(metadata['physics_dt_s']);reference=_pose(rows[0]['wafer_pose_world'],'wafer')
    last=reference;position=angle=linear=angular=gap=jaw=forbidden=0.;minimum_pillar=minimum_weight=float('inf');geometry=True
    for index,row in enumerate(rows):
        current=_pose(row['wafer_pose_world'],'wafer');clearance=pillar_plane_clearances(row['wafer_pose_world'],metadata)
        geometry=geometry and clearance['all_pillars_under_wafer']
        position=max(position,float(np.linalg.norm(current[:3,3]-reference[:3,3])))
        angle=max(angle,math.degrees(_angle(current[:3,:3],reference[:3,:3])))
        gap=max(gap,max(abs(x)for x in clearance['clearances_m']))
        jaw=max(jaw,max(row['jaw_contact_normal_force_N']));forbidden=max(forbidden,row['forbidden_wafer_contact_force_N'])
        minimum_pillar=min(minimum_pillar,min(row['pillar_contact_normal_force_N']));minimum_weight=min(minimum_weight,sum(row['pillar_contact_normal_force_N']))
        if index:
            if abs(row['time_s']-rows[index-1]['time_s']-dt)>1e-8:raise ValueError('Gun-return wafer trace has a gap.')
            linear=max(linear,float(np.linalg.norm(current[:3,3]-last[:3,3]))/dt)
            angular=max(angular,_angle(last[:3,:3],current[:3,:3])/dt)
        last=current
    passed=(geometry and position<=limits['return_position_m']and angle<=limits['return_rotation_deg']
        and gap<=limits['support_gap_tolerance_m']and minimum_pillar>=limits['minimum_pillar_contact_N']
        and minimum_weight>=.8*metadata['wafer_mass_kg']*abs(metadata['gravity_world_m_s2'][2])
        and max(jaw,forbidden)<=limits['maximum_unwanted_contact_N']
        and linear<limits['settle_linear_speed_m_s']and angular<limits['settle_angular_speed_rad_s'])
    return dict(passed=bool(passed),sample_count=len(rows),maximum_position_error_m=position,maximum_rotation_deg=angle,
        maximum_linear_speed_m_s=linear,maximum_angular_speed_rad_s=angular,maximum_pillar_gap_m=gap,
        minimum_pillar_force_N=minimum_pillar,minimum_total_support_N=minimum_weight,
        maximum_jaw_force_N=jaw,maximum_forbidden_force_N=forbidden,
        scope='Every gun-return physics interval, using unchanged wafer support and settled-motion bounds.')


def audit_native_full_commands(metadata,rows,directory):
    directory=Path(directory);origin=next(x['global_time_s']for x in metadata['full_stage_boundaries']if x['stage']=='wafer_cycle')
    commands=[]
    for filename,clock in [('tool_commands.jsonl',0.),('commands.jsonl',origin),('cycle_commands.jsonl',origin),('gun_return_commands.jsonl',0.)]:
        with (directory/filename).open()as stream:
            for line in stream:
                item=json.loads(line);t=item['start_time_s']if filename=='commands.jsonl'else item['time_s']
                commands.append((t+clock,item['target']if filename=='commands.jsonl'else item['action'],item['phase']))
    commands.sort(key=lambda x:x[0]);dt=float(metadata['physics_dt_s']);stride=round(.05/dt)
    if len(rows)!=1+stride*len(commands):raise ValueError('Native global command/sample count differs.')
    denominator=float(metadata['controller']['native_finger_open_q_m'])
    for index,(timestamp,action,phase)in enumerate(commands):
        if abs(timestamp-index*.05)>1e-8:raise ValueError('Native global command timestamp gap.')
        action=np.asarray(action,dtype=np.float32).astype(float)
        expected={'commanded_arm_q_rad':action[:6],'commanded_finger_q_m':[action[6]*denominator]*2,
            'commanded_right_arm_q_rad':action[7:13],'commanded_right_finger_q_m':[action[13]*denominator]*2}
        for row in rows[1+index*stride:1+(index+1)*stride]:
            if row['phase']!=phase:raise ValueError('Native command and measured phase differ.')
            for key,value in expected.items():
                if not np.allclose(row[key],value,atol=1e-7,rtol=0):raise ValueError('Native actuator readback differs: '+key)
    return dict(complete=True,commands=len(commands),physics_samples=len(rows),maximum_readback_tolerance=1e-7)


def audit_native_full_provenance(metadata,rows):
    if not metadata.get('full_workflow_fresh_start')or metadata.get('post_start_pose_write_count')!=0:
        raise ValueError('Native workflow was not declared fresh and continuous.')
    initial=metadata.get('initial_empty_hand_match',{})
    if initial.get('passed')is not True:raise ValueError('Measured empty-hand initialization evidence is missing.')
    spec=metadata['full_workflow_plan'];gaps=metadata['initial_empty_hand_geometry']
    if min(gaps.values())<=.002 or gaps!=spec['initial_empty_hand_geometry']or initial['local_geometry']!=gaps:
        raise ValueError('Initial geometry differs from its qualified separation.')
    if metadata['full_workflow_sources']!=spec['files']:raise ValueError('Native full workflow source manifest differs.')
    root=Path(metadata['full_workflow_bundle_root'])
    if json.loads((root/'full_workflow_plan.json').read_text())!=spec:raise ValueError('Recorded full plan differs from the qualified bundle.')
    for name,digest in spec['files'].items():native._bundle_file(root,name,digest)
    for key in ('local_scene_sha256','local_metadata_sha256','local_contact_solver_profile','native_contact_solver_scope','full_workflow_timing_profile'):
        if metadata[key]!=spec[key]:raise ValueError('Qualified local geometry binding differs: '+key)
    if metadata['handle_paddles']!=spec['handle_paddles']:raise ValueError('Qualified paddle mechanism differs.')
    if abs(rows[0]['time_s'])>1e-10 or rows[0]['phase']!='tool_initialization':raise ValueError('Native trace does not start with empty hands.')
    if max(rows[0]['finger_handle_contact_normal_force_N']+rows[0]['gun_grip_normal_force_N'])>.005:
        raise ValueError('Initial native tool contact contradicts empty hands.')
    return dict(complete=True,bundle_files=len(spec['files']),initial_geometry=initial,
        scope='Recorded native initialization and unchanged qualified bundle; no retrospective physics step.')


def evaluate_native_full(metadata,rows,directory=None):
    from wafer_cycle_isaac import evaluate_native_cycle
    from wafer_tool_pickup import evaluate_tool_pickup
    from wafer_tool_return import evaluate_gun_return
    checks={};evidence={'complete':False};errors=[];closed={}
    if directory is not None and(Path(directory)/'trial_report.json').is_file():
        closed={name:(Path(directory)/name).read_bytes()for name in ('metadata.json','trial_report.json')}
    source_files_sha256={name:hashlib.sha256(content).hexdigest()for name,content in closed.items()}
    try:
        if closed and json.loads(closed['metadata.json'])!=metadata:raise ValueError('Passed metadata differs from its closed recorded file.')
        evidence['provenance']=audit_native_full_provenance(metadata,rows)
        dt=float(metadata['physics_dt_s']);times=np.asarray([r['time_s']for r in rows])
        if len(rows)<2 or abs(times[0])>1e-10 or not np.allclose(np.diff(times),dt,atol=1e-8,rtol=0):raise ValueError('Full physics trace is incomplete.')
        boundaries={r['stage']:r for r in metadata['full_stage_boundaries']};wafer=boundaries['wafer_cycle'];returned=boundaries['gun_return']
        start=int(wafer['initial_sample_index']);end=int(returned['initial_sample_index'])
        if not 0<start<end<len(rows)-1:raise ValueError('Invalid full stage boundaries.')
        for boundary in (wafer,returned):
            if abs(rows[boundary['initial_sample_index']]['time_s']-boundary['global_time_s'])>1e-8:raise ValueError('Stage clock and sample differ.')
        checks['tool_pickup']=evaluate_tool_pickup(metadata,rows[:start+1])
        stage=TraceRowList(dict(row,global_time_s=row['time_s'],time_s=row['time_s']-wafer['global_time_s'])for row in rows[start:end+1])
        stage[0]=dict(stage[0],phase='initialization',finger_preload_command_index=None)
        checks['wafer_cycle']=evaluate_native_cycle(metadata,compact_evaluation_rows(stage))
        checks['gun_return']=evaluate_gun_return(metadata,rows)
        checks['wafer_while_gun_returns']=check_wafer_rest(metadata,rows[end:])
        checks['tool_hand_retention']=evaluate_tool_hand_retention(metadata,rows)
        guards=compact_guard_rows(rows)
        checks['passive_handle_paddles']=evaluate_handle_paddles(metadata,guards)
        checks['passive_handle_paddle_operating_limits']=evaluate_handle_paddles(metadata,guards,operating=True)
        checks['tool_environment_contacts']=evaluate_tool_environment(guards)
        peaks=dict(unexpected_left_contact_N=max(r['unexpected_robot_tool_contact_normal_force_N']+r['unexpected_robot_environment_contact_normal_force_N']for r in rows),
            unexpected_right_contact_N=max((c[2]for r in rows for c in r['unexpected_gun_right_contacts']),default=0.),
            forbidden_wafer_contact_N=max(r['forbidden_wafer_contact_force_N']for r in rows))
        checks['all_stage_contacts']=dict(passed=max(peaks.values())<=.01,peaks=peaks)
        checks['finger_handle_jam_guard']=native.evaluate_finger_handle_load(guards,engine='isaac')
        checks['passive_pad_motion_guard']=native.evaluate_passive_pad_state(guards,metadata['pad_compliance'])
        checks['passive_hinge_torque_guard']=native.evaluate_passive_hinge_state(guards,metadata['passive_hinge'])
        del guards
        checks['task_score']=evaluate_task_score(metadata,rows)
        checks['workflow_duration']=workflow_duration_check(metadata,rows[-1]['time_s'])
        if 'dust' in metadata:
            from wafer_dust import CohesiveDustScore
            dust_score=CohesiveDustScore(metadata)
            for row in rows:dust_score.update(row)
            checks['dust_clearing']=dust_score.result()
        from wafer_robot_geometry import evaluate_native_robot_geometry
        checks['robot_geometry']=evaluate_native_robot_geometry(metadata,rows)
        if directory is not None:
            directory=Path(directory);evidence['commands']=audit_native_full_commands(metadata,rows,directory)
            records=[json.loads(line)for line in (directory/'finger_preload.jsonl').open()]
            commands=[json.loads(line)for line in (directory/'commands.jsonl').open()]
            count=1+round(sum(duration for _,duration in wafer_prefix_phases(metadata))/dt)
            evidence['preload']=native.audit_finger_preload_evidence(stage[:count],records,commands,
                metadata['controller']['native_finger_open_q_m'],dt,wafer_prefix_phases(metadata))
            controller=native.FingerPreloadController(wafer_prefix_phases(metadata))
            for record in records:controller.step(record['phase'],record['interval'],record['previous_command_q_m'])
            cycle_records=[json.loads(line)for line in (directory/'cycle_commands.jsonl').open()]
            stride=round(.05/dt)
            for index,command in enumerate(cycle_records):
                sample=count-1+index*stride
                if abs(command['time_s']-stage[sample]['time_s'])>1e-8:raise ValueError('Native cycle command clock differs.')
                if 'left_preload'not in command:continue
                record=command['left_preload'];controller.phases.append((record['phase'],.05))
                _,expected=controller.step(record['phase'],record['interval'],record['previous_command_q_m'])
                if record!=expected:raise ValueError('Native cycle preload replay differs.')
                measured=native.preceding_interval_features(stage,record['control_index'],dt,prefix_length=sample+1)
                if record['interval']!=measured:raise ValueError('Native cycle preload source samples differ.')
            evidence['cycle_preload']=dict(complete=len(stage)==count+len(cycle_records)*stride,commands=len(cycle_records))
            evidence['complete']=bool(evidence['provenance']['complete']and evidence['commands']['complete']and evidence['preload']['complete']and evidence['cycle_preload']['complete']and checks['robot_geometry']['evidence_complete'])
        else:evidence['reason']='Command files are required for a complete independent audit.'
    except (ValueError,KeyError,IndexError,TypeError,OSError)as exc:errors.append(type(exc).__name__+': '+str(exc))
    if any(hashlib.sha256((Path(directory)/name).read_bytes()).hexdigest()!=digest for name,digest in source_files_sha256.items()):
        errors.append('Closed metadata/report changed during the independent audit.');evidence['complete']=False
    physical=(checks.get('tool_pickup',{}).get('passed')is True
        and checks.get('wafer_cycle',{}).get('result')=='WAFER_AIR_PATH_AND_RETURN_SUCCEEDED_IN_THIS_SIMULATION'
        and checks.get('gun_return',{}).get('result')=='GUN_RETURNED_AND_RELEASED_IN_THIS_SIMULATION'
        and checks.get('task_score',{}).get('score')==1. and checks.get('task_score',{}).get('success')is True
        and checks.get('robot_geometry',{}).get('passed')is True
        and checks.get('robot_geometry',{}).get('result')=='CLEAR_IN_ALL_RECORDED_PHYSICS_SAMPLES'
        and all(checks.get(k,{}).get('passed')is True for k in ('workflow_duration','wafer_while_gun_returns','tool_hand_retention','passive_handle_paddle_operating_limits','tool_environment_contacts','all_stage_contacts','finger_handle_jam_guard','passive_pad_motion_guard','passive_hinge_torque_guard')))
    if 'dust' in metadata:physical=physical and checks.get('dust_clearing',{}).get('passed') is True
    result=SUCCESS if physical and evidence['complete']else'FAILED'if evidence['complete']else'INCOMPLETE'
    previous=json.loads(closed['trial_report.json'])if closed else None
    matches=None if previous is None else previous.get('result')==result and all(previous.get('checks',{}).get(k)==v for k,v in checks.items())
    return dict(result=result,matches_report=matches,checks=checks,task_score=checks.get('task_score'),
        qualification=metadata.get('task_qualification'),legacy_diagnostic_result='PASSED'if checks.get('passive_handle_paddles',{}).get('passed')else'FAILED',
        evidence=evidence,errors=errors,source_files_sha256=source_files_sha256,backend='Isaac Lab / PhysX',
        scope='Fresh empty-hand pickup, wafer scan and return, supported gun release; no evaporation validation.')


DEVELOPMENT_POLICY='user_requested_full_development_v1'
ALLOW_UNQUALIFIED_DEVELOPMENT=False


def validate_full_development(root,plan,exported):
    if not ALLOW_UNQUALIFIED_DEVELOPMENT:
        raise ValueError('Use --development explicitly for this unqualified cloud test.')
    root=Path(root);spec=json.loads((root/'full_workflow_plan.json').read_text())
    validation=plan.get('local_validation',{});gate=spec.get('local_gate',{})
    if(validation.get('policy')!=DEVELOPMENT_POLICY or validation.get('qualification_complete')is not False
       or gate.get('policy')!=DEVELOPMENT_POLICY or gate.get('qualification_complete')is not False
       or gate.get('attempted')!=0 or gate.get('passed')!=0 or gate.get('cases')!=[]
       or not validation.get('user_steering') or spec.get('task_qualification')!=TASK_QUALIFICATION):
        raise ValueError('Incomplete or misleading development provenance.')
    for name,digest in spec['files'].items():native._bundle_file(root,name,digest)
    reference=json.loads(native._bundle_file(root,'local_reference.json',spec['files']['local_reference.json']).read_text())
    local=reference['metadata']
    if local.get('full_workflow_fresh_start')is not True or local.get('post_start_pose_write_count')!=0:
        raise ValueError('Reference scene did not start empty handed.')
    for name in ('wafer_full_cycle_isaac.py','wafer_tool_pickup.py','wafer_tool_return.py','wafer_trace_rows.py'):
        if native._sha(Path(__file__).parent/name)!=spec['files'][name]:raise ValueError('Imported development source differs: '+name)
    for name in ('wafer_task_score.py','wafer_robot_geometry.py','collision_geometry.py'):
        if native._sha(_BENCHMARK/name)!=spec['files']['benchmark/'+name]:raise ValueError('Imported development scorer or collision source differs.')
    if spec['export_manifest']!=plan['export_manifest']:raise ValueError('Development assets differ.')
    for key in ('arm_gains','right_arm_gains','world_translation_m','cycle_layout'):
        if local.get(key)!=plan.get(key):raise ValueError('Development physical configuration differs: '+key)
    for key in ('full_cycle_handle_geometry','handle_paddles','pad_compliance','passive_hinge','full_workflow_no_wafer_side_cradle'):
        if local.get(key)!=exported.get(key):raise ValueError('Development mechanism differs: '+key)
    for key in ('tool_pickup','initial_empty_hand_geometry','gun','full_workflow_timing_profile'):
        if local.get(key)!=spec.get(key):raise ValueError('Development scene binding differs: '+key)
    for key in ('tcp_feedback','finger_preload'):
        if local['controller'][key]!=plan[key]:raise ValueError('Development controller differs: '+key)
    for field,key in [('initial_left_q_rad','initial_arm_q_rad'),('seated_left_q_rad','seated_arm_q_rad'),('final_left_q_rad','final_arm_q_rad')]:
        if plan[field]!=local[key]:raise ValueError('Development command binding differs: '+field)
    return spec,local,dict(status='UNQUALIFIED_DEVELOPMENT_CONFIGURATION_CHECKED',qualification_complete=False,
        count=0,passed=0,success_fraction=None,selected_scenario_passed=False,evidence=[],
        scope='Current physical configuration only; no local20 or cloud success claim.')


FULL_EVIDENCE_POLICY='full_workflow_fixed20_v1'



def _full_fixed_scenarios(root,spec,frozen):
    import random
    name='wafer_grasp_scenarios.json'
    if spec['files'].get(name)!=frozen.get(name):raise ValueError('Canonical scenario source was not frozen.')
    source=json.loads(native._bundle_file(root,name,frozen[name]).read_text());trials=copy.deepcopy(source['trials'])
    rng=random.Random(20261109)
    for index in range(15):
        trials.append(dict(name=f'combined_{index+1:02d}',overrides=dict(
            wafer_offset_xy_m=[round(rng.uniform(-.0004,.0004),8),round(rng.uniform(-.0003,.0003),8)],
            wafer_yaw_deg=round(rng.uniform(-.25,.25),6),wafer_mass_kg=round(rng.uniform(.020,.022),8),
            friction=round(rng.uniform(.25,.30),6),pad_friction=round(rng.uniform(.50,.60),6))))
    result=[]
    for trial in trials:
        cfg={**source['baseline'],**trial['overrides'],'name':trial['name']}
        for key,value in [('wafer_diameter_m',.1),('wafer_thickness_m',.001),('flat_length_m',.0325),('finger_effort_limit_N',5.)]:
            if cfg.pop(key)!=value:raise ValueError('Canonical scenario geometry or force limit differs.')
        cfg['wafer_friction']=cfg.pop('friction');cfg['tcp_feedback']=True;result.append(cfg)
    if len({json.dumps({k:v for k,v in c.items()if k!='name'},sort_keys=True)for c in result})!=20:
        raise ValueError('Twenty distinct fixed physical scenarios are required.')
    return result


def _validate_full_case_physics(local,report,audit,plan,exported,good):
    dt=exported['physics_dt_s'];duration=report.get('simulation_seconds')
    if isinstance(duration,bool)or not isinstance(duration,(int,float))or not math.isfinite(duration)or duration<0:
        raise ValueError('Full trial has no finite recorded duration.')
    steps=round(duration/dt);count=1+steps;stride=round(.05/dt)
    if abs(duration-steps*dt)>1e-7 or(good and count<1000):raise ValueError('Insufficient or off-cadence full physics evidence.')
    for key in ('arm_gains','right_arm_gains','world_translation_m','cycle_layout'):
        if local.get(key)!=plan.get(key):raise ValueError('Full case native configuration differs: '+key)
    for key in ('cycle_handle_remap','pad_compliance','passive_hinge','full_cycle_handle_geometry','handle_paddles','full_workflow_no_wafer_side_cradle'):
        if local.get(key)!=exported.get(key):raise ValueError('Full case passive mechanism differs: '+key)
    controller=local.get('controller',{})
    if controller or good:
        for key in ('tcp_feedback','finger_preload'):
            if controller.get(key)!=plan.get(key):raise ValueError('Full case controller differs: '+key)
        if controller.get('command_hz')!=20 or controller.get('physics_hz')!=1/dt or controller.get('substeps_per_command')!=stride:
            raise ValueError('Full case controller/physics cadence differs.')
    elif report.get('stopped_stage')!='tool_pickup' or any(b.get('stage')=='wafer_cycle'for b in local.get('stage_boundaries',[])):
        raise ValueError('Missing wafer controller outside an early failed tool stage.')
    expected_guards=(('finger_handle_jam_guard',native.guard_metadata('local'),'finger_handle_jam_guard',native.GUARD_PROFILE),
        ('passive_pad_guard',native.pad_guard_metadata(),'passive_pad_motion_guard',native.PAD_GUARD_PROFILE),
        ('passive_hinge_guard',native.hinge_guard_metadata(),'passive_hinge_torque_guard',native.HINGE_GUARD_PROFILE))
    for metadata_key,profile,name,guard_profile in expected_guards:
        check=report.get('stages',{}).get(name,{})
        if local.get(metadata_key)!=profile or check.get('profile')!=guard_profile or check.get('sample_count')!=count:
            raise ValueError('Full case lacks complete raw physical guard evidence: '+name)
        if good and(check.get('passed')is not True or audit.get('checks',{}).get(name)!=check):
            raise ValueError('Full case raw physical guard audit disagrees: '+name)
    jam=report['stages']['finger_handle_jam_guard']
    if jam.get('measurement_field')!=native.guard_metadata('local')['measurement_field']:
        raise ValueError('Full finger guard uses another measurement.')
    for name in ('passive_handle_paddle_operating_limits','tool_environment_contacts'):
        check=report.get('stages',{}).get(name,{})
        if check.get('physics_sample_count')!=count or(good and(check.get('passed')is not True or audit.get('checks',{}).get(name)!=check)):
            raise ValueError('Full case operating/contact evidence is incomplete: '+name)
    if not good:return
    if report.get('qualification')!=TASK_QUALIFICATION:raise ValueError('Full report task qualification differs.')
    checks=audit.get('checks',{})
    for name in ('tool_pickup','wafer_while_gun_returns','tool_hand_retention','all_stage_contacts'):
        if checks.get(name,{}).get('passed')is not True:raise ValueError('Incomplete physical task stage: '+name)
    if checks.get('wafer_cycle',{}).get('result')!='WAFER_AIR_PATH_AND_RETURN_SUCCEEDED_IN_THIS_SIMULATION' or checks.get('gun_return',{}).get('result')!='GUN_RETURNED_AND_RELEASED_IN_THIS_SIMULATION':
        raise ValueError('Wafer or gun task stage did not complete.')
    evidence=audit.get('evidence',{});commands=evidence.get('all_commands',{})
    if(commands.get('passed')is not True or commands.get('commands')!=round(duration*20)
       or commands.get('physics_rows')!=count or commands.get('physics_rows_per_command')!=stride
       or report.get('all_stage_command_audit')!=commands):
        raise ValueError('Full command replay does not cover every physics sample.')
    wafer=evidence.get('wafer_commands',{});prefix=wafer.get('prefix',{});replay=prefix.get('command_replay',{})
    prefix_s=sum(d for _,d in wafer_prefix_phases(local));prefix_count=1+round(prefix_s/dt);prefix_commands=round(prefix_s*20)
    if(wafer.get('complete')is not True or prefix.get('complete')is not True
       or prefix.get('result')!='SAVED_PRELOAD_INPUTS_OUTPUTS_AND_REPLAY_MATCH'
       or prefix.get('sample_count')!=prefix_count or prefix.get('expected_full_sample_count')!=prefix_count
       or prefix.get('command_count')!=prefix_commands or prefix.get('expected_full_command_count')!=prefix_commands
       or prefix.get('physics_dt_s')!=dt or abs(prefix.get('duration_recorded_s',-1)-prefix_s)>1e-7
       or replay.get('complete')is not True or replay.get('status')!='DETERMINISTIC_PRELOAD_COMMAND_REPLAY_MATCH'
       or replay.get('command_count')!=prefix_commands or replay.get('expected_complete_command_count')!=prefix_commands
       or replay.get('profile')!=native.FINGER_PRELOAD_PROFILE):
        raise ValueError('Full prelift/controller replay evidence is incomplete.')
    boundaries={r['stage']:r for r in local['stage_boundaries']};start=boundaries['wafer_cycle'];end=boundaries['gun_return']
    if(wafer.get('physics_samples')!=1+end['initial_sample_index']-start['initial_sample_index']
       or abs(wafer.get('global_clock_origin_s',-1)-start['global_time_s'])>1e-7
       or wafer['physics_samples']!=prefix_count+wafer.get('extension_commands',-1)*stride):
        raise ValueError('Full wafer-stage clocks or command coverage differ.')


def validate_full_workflow_evidence(bundle,plan,exported_metadata):
    """Verify the complete workflow directly; no inherited partial-run evidence."""
    if plan.get('local_validation',{}).get('policy')==DEVELOPMENT_POLICY:
        return validate_full_development(bundle,plan,exported_metadata)
    import zipfile
    root=Path(bundle).resolve();spec=json.loads((root/'full_workflow_plan.json').read_text())
    validation=plan.get('local_validation',{});gate=spec.get('local_gate',{})
    if(validation.get('policy')!=FULL_EVIDENCE_POLICY or validation.get('qualification_complete')is not True
       or spec.get('version')!=2 or spec.get('task_qualification')!=TASK_QUALIFICATION
       or gate.get('policy')!='fixed20_v1' or gate.get('attempted')!=20
       or gate.get('passed',0)<16 or gate.get('nominal_passed')is not True):
        raise ValueError('Complete empty-hand fixed20 qualification is missing.')
    if spec.get('export_manifest')!=plan.get('export_manifest'):
        raise ValueError('Full evidence and spawn assets differ.')
    for name,digest in spec['files'].items():native._bundle_file(root,name,digest)
    frozen=gate.get('source_sha256',{})
    if not frozen or any(not isinstance(v,str)or len(v)!=64 or any(c not in '0123456789abcdef'for c in v)for v in frozen.values()):
        raise ValueError('Invalid complete-workflow source manifest.')
    for name,digest in spec['files'].items():
        if name.endswith(('.py','.toml'))and frozen.get(name)!=digest:
            raise ValueError('Transferred runtime was not in the full fixed20 source freeze: '+name)
    canonical_inputs=_full_fixed_scenarios(root,spec,frozen)
    canonical=[{**exported_metadata['scenario'],**{k:v for k,v in c.items()if k!='tcp_feedback'}}for c in canonical_inputs]
    expected_names=['nominal','offset_positive','offset_negative','lower_friction','heavier_wafer']+[f'combined_{i:02d}'for i in range(1,16)]
    cases=gate.get('cases',[])
    if [x.get('name')for x in cases]!=expected_names:raise ValueError('The fixed20 scenarios differ or are reordered.')
    if [c.get('scenario')for c in cases]!=canonical:raise ValueError('The canonical fixed20 physical configurations differ.')
    archive_name=spec.get('evidence_archive')
    if archive_name!='full_workflow_validation.zip' or validation.get('archive_sha256')!=spec['files'].get(archive_name):
        raise ValueError('Full evidence archive binding differs.')
    archive_path=native._bundle_file(root,archive_name,validation['archive_sha256'])
    expected_members={case['name']+'/'+name for case in cases for name in ('report.json','independent_check.json','metadata.json')}
    passed=0;nominal=None;evidence=[];fingerprints=set()
    source_fields={'mechanism':'model_source_sha256','protocol':'protocol_sha256','settings':'settings_sha256',
        'yam_checks':'yam_checks_sha256','pad_yam_checks':'pad_yam_checks_sha256','hinge_guard':'hinge_guard_sha256',
        'parent_yam_checks':'previous_yam_checks_sha256','pad_guard':'pad_guard_sha256','base_yam_checks':'base_yam_checks_sha256',
        'jam_guard':'jam_guard_sha256','adapter':'adapter_sha256','robot_bridge':'robot_bridge_sha256',
        'tcp_feedback':'tcp_feedback_sha256','finger_preload':'finger_preload_controller_sha256'}
    with zipfile.ZipFile(archive_path)as archive:
        if len(archive.namelist())!=60 or set(archive.namelist())!=expected_members:
            raise ValueError('Expected exactly the actual metadata/report/audit for each20 full trials.')
        for case in cases:
            values={}
            for name in ('report.json','independent_check.json','metadata.json'):
                item=case[name]
                if item['member']!=case['name']+'/'+name:raise ValueError('Cross-case evidence member.')
                payload=archive.read(item['member'])
                if hashlib.sha256(payload).hexdigest()!=item['sha256']:raise ValueError('Full local evidence hash differs.')
                values[name]=json.loads(payload)
            report,audit,local=(values[n]for n in ('report.json','independent_check.json','metadata.json'))
            signature=tuple(case[n]['sha256']for n in ('report.json','metadata.json'))
            if signature in fingerprints:raise ValueError('Duplicated evidence is not another trial.')
            fingerprints.add(signature)
            if(local.get('scenario')!=case.get('scenario')or local.get('scenario',{}).get('name')!=case['name']
               or local.get('engine')!='MuJoCo'or not str(local.get('mujoco_version','')).startswith('3.')
               or local.get('full_workflow_fresh_start')is not True or local.get('post_start_pose_write_count')!=0
               or local.get('robot_collider_count')!=315 or local.get('physics_dt_s')!=exported_metadata['physics_dt_s']
               or local.get('tcp_feedback_enabled')is not True):
                raise ValueError('Full trial scenario, fresh-start or actual-robot provenance differs.')
            disturbance=local.get('retention_disturbance',{})
            if disturbance.get('enabled')is not False or disturbance.get('profile')!='none' or disturbance.get('wrench_world_N_Nm')!=[0.]*6 or disturbance.get('post_start_pose_write_count')!=0:
                raise ValueError('Disturbance probes cannot qualify the full transfer.')
            if any(local.get('source_sha256',{}).get(k)!=plan.get(v)for k,v in source_fields.items()):
                raise ValueError('A full trial used another source dependency.')
            if local.get('model_dependency_sha256')!=exported_metadata['model_dependency_sha256']:
                raise ValueError('Full trial and exported geometry dependencies differ.')
            sources=local.get('full_workflow_sources',{})
            required={'wafer_full_cycle.py','wafer_full_cycle_isaac.py','wafer_tool_pickup.py','wafer_tool_return.py','wafer_trace_rows.py',
                'wafer_finger_preload_v12.py','run_retention_experiment_v12.py','wafer_cycle.py','wafer_cycle_protocol.py','wafer_cycle_isaac.py',
                'wafer_retention_isaac_v12.py','benchmark/wafer_task_score.py','benchmark/wafer_robot_geometry.py','benchmark/collision_geometry.py'}
            if not required.issubset(sources)or any(frozen.get(k)!=v for k,v in sources.items()):raise ValueError('Full trace source freeze differs.')
            for key,name in [('finger_preload_parent','wafer_finger_preload_v10.py'),('numerical_parent','wafer_retention_candidate_v11.py'),('robot_urdf','yam_arm.urdf'),('imported_robot_xml','yam_collision.xml')]:
                if not frozen.get(name)or local['source_sha256'].get(key)!=frozen[name]:raise ValueError('Full physical parent or actual-robot source differs: '+name)
            if local.get('task_qualification')!=TASK_QUALIFICATION or local.get('full_workflow_timing_profile')!=spec['full_workflow_timing_profile']:
                raise ValueError('Full task or120-second schedule differs.')
            workflow_timing(local)
            if local.get('local_contact_solver_profile')!=spec.get('local_contact_solver_profile'):
                raise ValueError('Local solver qualification differs.')
            if not local.get('robot_collision_assets_sha256')or local['robot_collision_assets_sha256']!=spec.get('robot_collision_assets_sha256'):
                raise ValueError('Full trial actual robot collision assets differ.')
            if local.get('tool_pickup',{}).get('profile')!=spec['tool_pickup']['profile']:
                raise ValueError('Full trial tool pickup profile differs.')
            good=(report.get('result')==SUCCESS and audit.get('result')==SUCCESS
                and audit.get('matches_report')is True and audit.get('evidence',{}).get('complete')is True)
            _validate_full_case_physics(local,report,audit,plan,exported_metadata,good)
            if good:
                task=report.get('task_score',{});geometry=audit.get('checks',{}).get('robot_geometry',{})
                bindings=audit.get('source_files_sha256',{})
                if task.get('success')is not True or task.get('score')!=1. or audit.get('task_score')!=task:
                    raise ValueError('Full case lacks matching independently measured task completion.')
                if not(geometry.get('passed')is True and geometry.get('evidence_complete')is True and geometry.get('result')=='CLEAR_IN_ALL_RECORDED_PHYSICS_SAMPLES'
                    and geometry.get('physics_samples')==1+round(report['simulation_seconds']/local['physics_dt_s'])
                    and geometry.get('physics_dt_s')==local['physics_dt_s']
                    and geometry.get('geometry',{}).get('collider_counts')=={'left':315,'right':315}
                    and geometry.get('robot_collision_assets_sha256')==local['robot_collision_assets_sha256']
                    and geometry.get('audit_sources')=={n:sources['benchmark/'+n]for n in ('wafer_robot_geometry.py','collision_geometry.py')}):
                    raise ValueError('Full case lacks the all-step robot geometry audit.')
                if local.get('model_xml_sha256')!=geometry.get('scene_sha256'):raise ValueError('Full trace startup scene differs from audited geometry.')
                if any(geometry.get(k)!=case.get(k)or not isinstance(case.get(k),str)or len(case[k])!=64 for k in ('scene_sha256','replay_sha256','trace_sha256')):
                    raise ValueError('Full geometry audit describes different artifacts.')
                expected_bindings={n:case[n]['sha256']for n in ('metadata.json','report.json')}
                if bindings!=expected_bindings or bindings!=case.get('source_files_sha256'):
                    raise ValueError('Full audit metadata/report binding is stale.')
                if(report.get('simulation_seconds',float('inf'))>workflow_timing(local)['maximum_simulation_seconds']+1e-8
                   or local.get('controller',{}).get('phases')!=[list(x)for x in wafer_prefix_phases(local)]):
                    raise ValueError('Passing full trial exceeds the task deadline or uses another phase schedule.')
            if good is not case.get('passed'):raise ValueError('Full case result disagrees with batch evidence.')
            passed+=int(good);evidence.append(dict(name=case['name'],passed=good,metadata_sha256=case['metadata.json']['sha256']))
            if case['name']=='nominal':
                if not good:raise ValueError('The selected nominal full case did not pass.')
                nominal=local
                if case['scene_sha256']!=spec['local_scene_sha256']or case['metadata.json']['sha256']!=spec['local_metadata_sha256']:
                    raise ValueError('Selected nominal scene/metadata differs.')
    if passed!=gate['passed']or passed<16:raise ValueError('Full fixed20 success count differs.')
    for key in ('arm_gains','right_arm_gains','world_translation_m','cycle_layout'):
        if nominal.get(key)!=plan.get(key):raise ValueError('Native full configuration differs: '+key)
    for key in ('full_cycle_handle_geometry','handle_paddles','full_workflow_no_wafer_side_cradle'):
        if nominal.get(key)!=spec.get(key)or nominal.get(key)!=exported_metadata.get(key):raise ValueError('Full mechanism differs: '+key)
    if nominal.get('pad_compliance')!=exported_metadata.get('pad_compliance')or nominal.get('passive_hinge')!=exported_metadata.get('passive_hinge'):
        raise ValueError('Full passive core mechanism differs.')
    for key in ('tool_pickup','initial_empty_hand_geometry','gun'):
        if nominal.get(key)!=spec.get(key):raise ValueError('Nominal full configuration binding differs: '+key)
    for key in ('tcp_feedback','finger_preload'):
        if nominal.get('controller',{}).get(key)!=plan.get(key):raise ValueError('Nominal full controller differs: '+key)
    for field,key in [('initial_left_q_rad','initial_arm_q_rad'),('seated_left_q_rad','seated_arm_q_rad'),('final_left_q_rad','final_arm_q_rad')]:
        if plan.get(field)!=nominal.get(key):raise ValueError('Nominal wafer-stage command binding differs: '+field)
    return spec,nominal,dict(status='FULL_WORKFLOW_FIXED20_VERIFIED',qualification_complete=True,count=20,passed=passed,
        success_fraction=passed/20,selected_scenario_passed=True,evidence=evidence,
        scope='Actual complete local workflow evidence only; native transfer remains untested.')


class FullCycle(Cycle):
    initial_phase='tool_initialization'
    full_workflow_adapter=True

    def trace_rows(self):
        return TraceRowList(cache_blocks=1024)

    def validate(self,bundle,plan,metadata):
        self.root=Path(bundle);self.metadata=metadata
        self.full,self.local_nominal,self.full_gate=validate_full_workflow_evidence(bundle,plan,metadata)
        self._validate_runtime_spec(bundle,plan,metadata,self.full)
        metadata['task_qualification']=dict(TASK_QUALIFICATION)
        from wafer_tool_pickup import PROFILE as TOOL_PROFILE,tool_timing
        from wafer_tool_return import RETURN
        if self.full['tool_pickup']['profile']!=TOOL_PROFILE or self.full['gun_return_profile']!=RETURN:
            raise ValueError('Shared tool pickup or gun return profile differs.')
        tool_timing(dict(tool_pickup=self.full['tool_pickup']))
        gate=self.full['local_gate']
        metadata['full_workflow_timing_profile']=copy.deepcopy(self.full['full_workflow_timing_profile'])
        workflow_timing(metadata)
        plan['phases']=[list(p)for p in wafer_prefix_phases(metadata)]
        self.wafer_plan=copy.deepcopy(plan)
        spec=self.full['tool_pickup'];plan['initial_left_q_rad']=list(spec['start_arm_q_rad'])
        plan['initial_finger_state_q_m']=float(spec['profile']['open_finger_m'])
        metadata.update(full_workflow=True,full_workflow_fresh_start=True,tool_pickup=copy.deepcopy(spec),
            full_workflow_bundle_root=str(self.root.resolve()),
            full_local_gate=gate,full_workflow_sources=self.full['files'],full_workflow_plan=copy.deepcopy(self.full),
            local_scene_sha256=self.full['local_scene_sha256'],local_metadata_sha256=self.full['local_metadata_sha256'],post_start_pose_write_count=0,
            local_contact_solver_profile=copy.deepcopy(self.full['local_contact_solver_profile']),
            native_contact_solver_scope=self.full['native_contact_solver_scope'],
            initial_finger_position_m=plan['initial_finger_state_q_m'])
        metadata['initial_empty_hand_geometry']=copy.deepcopy(self.full['initial_empty_hand_geometry'])
        self.workflow_stage='tool_pickup'


    def finger_contact_links(self):
        names=[s['body_name']for s in self.metadata['handle_paddles']]
        if names!=['lower_grip_paddle','upper_grip_paddle']:raise ValueError('Paddle contact-filter order differs.')
        return names

    def wafer_contact_links(self):
        return self.finger_contact_links()

    def configure(self,objects,embodiment,Pose,Object,ObjectType,Implicit,Contact,torch):
        sensors=super().configure(objects,embodiment,Pose,Object,ObjectType,Implicit,Contact,torch)
        if 'clamp_cradle'in objects:raise ValueError('Obsolete wafer-side cradle was exported into the full workflow.')
        spec=self.full['tool_pickup'];delta=np.asarray(spec['profile']['stand_offset_world_m'])
        cfg=objects['clamp'].object_cfg
        position=[float(x)for x in np.asarray(cfg.init_state.pos)+delta]
        # Rest on the hinge support while clearing the handle support.
        pitch=.0013;c=math.cos(pitch/2);s=math.sin(pitch/2)
        x,y,z,w=[float(v)for v in cfg.init_state.rot]
        rotation=[x*c-z*s,y*c+w*s,z*c+x*s,w*c-y*s]
        self.metadata['tool_pickup']['initial_tool_pose_world'][3:]=rotation
        self.metadata['native_initial_tool_pitch_adjustment_rad']=pitch
        self._startup_contact_capture_count=0
        shifted=Pose.from_dict({'position_xyz':position,'rotation_xyzw':rotation})
        cfg.init_state.rot=tuple(rotation)
        if shifted.to_tensor(device='cpu').dtype!=torch.float32:raise ValueError('Tool initial pose must be float32.')
        objects['clamp'].initial_pose=shifted
        cfg.init_state.pos=tuple(position)
        self.metadata['full_tool_initialization']=dict(position_world_m=position,rotation_xyzw=list(shifted.rotation_xyzw),
            scope='Matching spawn default and Arena initial-reset Pose, set before simulation creation.')
        cfg.init_state.joint_pos.update(spec.get('initial_joint_positions',{'pliers_hinge':spec['initial_passive_hinge_rad']}))
        export=self.full['tool_stand_export']
        for name,path in export['models'].items():
            value=export['initial_world_poses'][name]
            pose=Pose.from_dict({'position_xyz':[float(x)for x in value[:3]],'rotation_xyzw':[float(x)for x in value[3:]]})
            if pose.to_tensor(device='cpu').dtype!=torch.float32:raise ValueError('Tool stand pose must be float32.')
            objects[name]=Object(name=name,usd_path=str(self.root/path),object_type=ObjectType.RIGID,initial_pose=pose)
        filters=[objects[n].get_contact_sensor_prim_path()for n in spec['stand_geoms']]
        self.tool_stand_sensor_names=[]
        for i,body in enumerate(('lower_half','upper_half')):
            name=f'full_tool_stand_{i}';self.tool_stand_sensor_names.append(name)
            sensors[name]=Contact(prim_path=objects['clamp'].prim_path+'/'+body,filter_prim_paths_expr=filters,
                update_period=0.,history_length=1,track_air_time=False,max_contact_data_count_per_prim=256)
        bodies=['lower_half','upper_half','lower_pad','upper_pad',*self.finger_contact_links()]
        tips=[embodiment.scene_config.left_robot.prim_path+embodiment.link_paths[n]for n in ('tip_left','tip_right')]
        self.tool_environment_filters=[];self.tool_environment_sensor_names=[]
        for index,body in enumerate(bodies):
            partners=[('wafer',objects['wafer'].get_contact_sensor_prim_path()),*zip(('left_tip_left','left_tip_right'),tips)]
            partners.extend((other,objects['clamp'].prim_path+'/'+other)for other in bodies if other!=body)
            partners.extend((name,objects[name].get_contact_sensor_prim_path())for name in spec['stand_geoms'])
            partners.extend((name,obj.get_contact_sensor_prim_path())for name,obj in objects.items()if name not in {'wafer','clamp','gun',*spec['stand_geoms']})
            partners.extend(zip(('blow_gun','gun_trigger'),self.gunpaths))
            name=f'full_tool_environment_{index}';self.tool_environment_sensor_names.append(name)
            self.tool_environment_filters.append(dict(body=body,partners=[p[0]for p in partners],paths=[p[1]for p in partners]))
            sensors[name]=Contact(prim_path=objects['clamp'].prim_path+'/'+body,filter_prim_paths_expr=[p[1]for p in partners],
                update_period=0.,history_length=1,track_air_time=False,max_contact_data_count_per_prim=256)
        self.metadata['tool_environment_filter_spec']=copy.deepcopy(self.tool_environment_filters)
        self._task_objects=dict(objects)
        return sensors

    def ready(self,base,metadata,torch):
        super().ready(base,metadata,torch)
        from yam_arena.constants import FINGER_OPEN
        metadata['controller']['native_finger_open_q_m']=float(FINGER_OPEN)
        self.tool_stand_sensors=[base.scene.sensors[n]for n in self.tool_stand_sensor_names]
        self.tool_environment_sensors=[base.scene.sensors[n]for n in self.tool_environment_sensor_names]
        for sensor,spec in zip(self.tool_environment_sensors,self.tool_environment_filters):
            if list(sensor.body_names)!=[spec['body']]:raise ValueError('Tool environment sensing body differs.')
        metadata['initial_condition']='Both hands empty; passive pliers and gun begin on separate stands.'
        metadata['controller']['initial_condition']=metadata['initial_condition']
        metadata['full_stage_boundaries']=[dict(stage='tool_pickup',global_time_s=0.,initial_sample_index=0)]
        origin=native._array(base.scene.env_origins,(1,3))[0]
        for key in ('target_tool_pose_world','initial_tool_pose_world'):
            metadata['tool_pickup'][key][:3]=(np.asarray(metadata['tool_pickup'][key][:3])+origin).tolist()
        metadata['tool_pickup']['target_tcp_position_world_m']=(np.asarray(metadata['tool_pickup']['target_tcp_position_world_m'])+origin).tolist()
        self.tool_reference=None;self.tool_handle_reference=None;self.tool_contact_gap=np.zeros(2)
        self.tool_hand_monitor=ToolHandMonitor(metadata);metadata['tool_hand_retention_profile']=dict(TOOL_HAND_PROFILE)
        clamp=base.scene.articulations['clamp'];self.paddle_body_ids=[];self.paddle_joint_ids=[]
        for spec in metadata['handle_paddles']:
            ids,names=clamp.find_bodies(spec['body_name'])
            if list(names)!=[spec['body_name']]or len(ids)!=1:raise ValueError('Paddle body mapping differs.')
            self.paddle_body_ids.append(int(ids[0]))
            ids,names=clamp.find_joints([spec['joint_name']],preserve_order=True)
            if list(names)!=[spec['joint_name']]or len(ids)!=1:raise ValueError('Paddle joint mapping differs.')
            self.paddle_joint_ids.append(int(ids[0]))
        metadata['finger_handle_contact_colliders']={s['body_name']:list(s.get('collider_names',[s['collider_name']]))for s in metadata['handle_paddles']}
        self.stand_origin=np.asarray(self.export['initial_world_poses']['gun_stand'][:3])+origin
        metadata['wafer_support_force_measurement']=dict(version='signed_wafer_normal_support_v1',frame='world',force_on='wafer',source='Isaac Lab normal_force_matrix_w on the wafer sensor, filtered by pad and pillar body.')
        self._full_integrity=None;self._task_scorer=None
        if getattr(self,'task_measurements_enabled',True):
            import omni.usd
            import wafer_benchmark_backend as benchmark_backend
            if hashlib.sha256(Path(benchmark_backend.__file__).read_bytes()).hexdigest()!=self.full['files']['benchmark/wafer_benchmark_backend.py']:
                raise ValueError('Loaded geometry-integrity implementation differs from the qualified bundle.')
            self._integrity_class=benchmark_backend._GeometryIntegrity
            from wafer_robot_geometry import RobotGeometryGuard,NATIVE_OFFLINE_EVIDENCE
            stage=omni.usd.get_context().get_stage()
            roots={side:metadata['robot_materials'][side+'_robot']['root_path']for side in ('left','right')}
            self._robot_geometry_guard=RobotGeometryGuard.from_native({side:stage.GetPrimAtPath(path)for side,path in roots.items()})
            self._robot_integrity=self._integrity_class(stage,roots)
            metadata['robot_geometry']=self._robot_geometry_guard.evidence()
            metadata['robot_geometry_capture']=dict(version=NATIVE_OFFLINE_EVIDENCE,geometry_definition_sha256=dict(self._robot_integrity.initial))
            self._robot_body_ids={}
            for side,name in self._robot_geometry_guard.keys:
                ids,names=base.scene[side+'_robot'].find_bodies(name)
                if list(names)!=[name]or len(ids)!=1:raise ValueError('Robot collision link readback is ambiguous.')
                self._robot_body_ids[(side,name)]=int(ids[0])
            self._fixture_names=tuple(sorted(set(self._task_objects)-{'wafer','clamp','gun'}))
            def planned(name):
                pose=self._task_objects[name].initial_pose
                return [*(np.asarray(pose.position_xyz,dtype=float)+origin).tolist(),*np.asarray(pose.rotation_xyzw,dtype=float).tolist()]
            metadata['task_score_reference']=dict(fixture_poses_world={n:planned(n)for n in self._fixture_names},
                initial_object_poses_world={n:planned(n)for n in ('wafer','clamp','gun')},object_names=sorted(self._task_objects))

    def fields(self):
        fields=super().fields()
        fields.extend((f'full_tool_stand_{i}',sensor.data.normal_force_matrix_w,(1,1,2,3))for i,sensor in enumerate(self.tool_stand_sensors))
        fields.extend([('gun_linear_velocity',self.gunasset.data.body_link_lin_vel_w,None),
            ('gun_angular_velocity',self.gunasset.data.body_link_ang_vel_w,None)])
        for index,(sensor,spec)in enumerate(zip(self.tool_environment_sensors,self.tool_environment_filters)):
            fields.extend([(f'tool_environment_matrix_{index}',sensor.data.normal_force_matrix_w,(1,1,len(spec['partners']),3)),
                (f'tool_environment_net_{index}',sensor.data.net_normal_forces_w,(1,1,3))])
        if getattr(self,'task_measurements_enabled',True):
            fields.extend(('fixture_'+name,self.base.scene[name].data.root_link_pose_w,(1,7))for name in self._fixture_names)
        return fields

    def allowed_tool_particle_partners(self,row):return ()

    def measurement(self,a,row,phase):
        result=super().measurement(a,row,phase)
        support=np.asarray(row['filtered_normal_force_vectors_world_N'])
        result.update(wafer_jaw_normal_force_vectors_world_N=support[:2].tolist(),wafer_pillar_normal_force_vectors_world_N=support[4:7].tolist())
        result['finger_handle_contact_normal_force_N']=list(row['finger_handle_partner_normal_force_N'])
        result['finger_handle_contact_scope']='Native tip_left/upper_grip_paddle and tip_right/lower_grip_paddle partner-filtered resultant norms, including all authored colliders per paddle body; not per-collider force sums.'
        specs=self.metadata['handle_paddles'];q=a['clamp_q'][0,self.paddle_joint_ids].tolist();v=a['clamp_v'][0,self.paddle_joint_ids].tolist()
        result.update(handle_paddle_joint_names=[s['joint_name']for s in specs],handle_paddle_q_rad=q,handle_paddle_qvel_rad_s=v,
            handle_paddle_poses_world=a['clamp_poses'][0,self.paddle_body_ids].tolist(),
            handle_paddle_passive_torque_Nm=[-s['stiffness_Nm_rad']*(p-s['rest_rad'])-s['damping_Nms_rad']*w for s,p,w in zip(specs,q,v)])
        result['measured_right_finger_q_m']=a['right_q'][0,self.right_finger_ids].tolist()
        result['gun_trigger_m']=result['trigger_compression_m']
        result['tool_stand_normal_force_N']=sum(float(np.linalg.norm(a[f'full_tool_stand_{i}'][0,0],axis=1).sum())for i in range(2))
        result['workflow_stage']=self.workflow_stage
        from wafer_tool_pickup import tool_stand_contact_allowed
        result.update(tool_environment_contact_evidence(self.tool_environment_filters,a,
            set(self.metadata['tool_pickup']['stand_geoms']),tool_stand_contact_allowed(phase),
            self.allowed_tool_particle_partners(result)))
        if self._startup_contact_capture_count<20:
            result['startup_tool_contact_forces_N']=[[s['body'],p,float(f)]for i,s in enumerate(self.tool_environment_filters)for p,f in zip(s['partners'],np.linalg.norm(a[f'tool_environment_matrix_{i}'][0,0],axis=1))if f>.0001]
            self._startup_contact_capture_count+=1
        shape=self.export['support_geometry'];connector=shape['gun_hose_connector'];stand=np.asarray(shape['gun_stand']['half_sizes_m'])
        gun=_pose(result['gun_pose_world'],'gun');center=gun[:3,3]+gun[:3,:3]@np.asarray(connector['local_position_m'])
        axis=(gun[:3,:3]@np.asarray(connector['local_rotation']))[:,2]
        radius=connector['radius_m'];extent=radius*math.sqrt(max(0.,1-axis[2]**2))+connector['halfheight_m']*abs(axis[2])
        force=a['gun_contacts_0'][0,0,2]
        result.update(gun_stand_vertical_force_N=float(force[2]),gun_stand_normal_force_N=float(np.linalg.norm(force)),
            gun_foot_gap_m=float(center[2]-extent-self.stand_origin[2]-stand[2]),
            gun_support_margin_m=float(np.min(stand[:2]-np.abs(center[:2]-self.stand_origin[:2])-radius)),
            gun_linear_speed_m_s=float(np.linalg.norm(a['gun_linear_velocity'][0,self.gun_body])),
            gun_angular_speed_rad_s=float(np.linalg.norm(a['gun_angular_velocity'][0,self.gun_body])))
        if getattr(self,'task_measurements_enabled',True):
            if self._full_integrity is None:
                import omni.usd
                roots={name:obj.prim_path.replace('{ENV_REGEX_NS}','/World/envs/env_0').replace('{ENV_NS}','/World/envs/env_0').replace('/env_.*/','/env_0/')for name,obj in self._task_objects.items()}
                self._full_integrity=self._integrity_class(omni.usd.get_context().get_stage(),roots)
            result.update(fixture_poses_world={name:a['fixture_'+name][0].tolist()for name in self._fixture_names})
            result['objects_present'],result['objects_intact']=self._full_integrity.snapshot()
            result['robot_link_poses_world']=[a[side+'_poses'][0,self._robot_body_ids[(side,name)]].tolist()for side,name in self._robot_geometry_guard.keys]
            present,intact=self._robot_integrity.snapshot()
            result['robot_geometry_intact']=all(present.values())and all(intact.values())
        return result

    def guard(self,row):
        if row['time_s']>self.metadata['full_workflow_timing_profile']['maximum_simulation_seconds']+1e-8:
            raise native.TrialAbort('Whole workflow exceeded 120 simulation seconds, including post-completion motion.')
        super().guard(row)
        from wafer_tool_pickup import guard_handle_paddle_capacity
        try:guard_handle_paddle_capacity(row,self.metadata)
        except RuntimeError as exc:raise native.TrialAbort(str(exc))from exc
        if getattr(self,'task_measurements_enabled',True):
            if self._task_scorer is None:self._task_scorer=WaferTaskScorer(self.metadata,sample_dt_s=self.metadata['physics_dt_s'])
            self._task_scorer.update(row,report=False)
            if self._task_scorer.invalid:raise native.TrialAbort('Task evidence invalid: '+str(self._task_scorer.invalid))
        if row['tool_unexpected_environment_contact_normal_force_N']>.01:raise native.TrialAbort('Unexpected tool/environment contact exceeded 0.01 N.')
        if self.tool_hand_monitor.observe(row)is False:
            self.metadata['tool_hand_retention_result']=self.tool_hand_monitor.report()
            raise native.TrialAbort('Contact-paddle midpoint or orientation exceeded hand/tool retention bounds.')
        if row['phase'].startswith('tool_'):
            from wafer_tool_pickup import PROFILE,handle_retention_error
            if max(row['jaw_contact_normal_force_N'])>.05 or row['forbidden_wafer_contact_force_N']>.01:
                raise native.TrialAbort('Tool contacted wafer before the wafer grasp stage.')
            if row['unexpected_robot_tool_contact_normal_force_N']>1 or row['unexpected_robot_environment_contact_normal_force_N']>1:
                raise native.TrialAbort('Unexpected tool-pickup robot contact.')
            if self.tool_reference is not None:
                relative=np.linalg.inv(_pose(row['measured_hand_pose_world'],'hand'))@_pose(row['clamp_pose_world'],'tool')
                ref=self.tool_reference;p=self.tool_grasp_point
                slip=np.linalg.norm(relative[:3,3]+relative[:3,:3]@p-ref[:3,3]-ref[:3,:3]@p)
                rotation=math.degrees(_angle(relative[:3,:3],ref[:3,:3]))
                slip,rotation=handle_retention_error(row,self.tool_handle_reference)
                force=np.asarray(row['finger_handle_contact_normal_force_N'])
                self.tool_contact_gap=np.where(force<=.005,self.tool_contact_gap+self.metadata['physics_dt_s'],0.)
                if max(self.tool_contact_gap)>PROFILE['maximum_contact_gap_s']+1e-10:
                    raise native.TrialAbort('Finger contact was lost during tool transport.')
                if slip>PROFILE['maximum_relative_slip_m']or rotation>PROFILE['maximum_relative_rotation_deg']:
                    raise native.TrialAbort('Empty pliers slipped relative to the hand.')
                if row['measured_jaw_gap_m']<PROFILE['minimum_transport_gap_m']:
                    raise native.TrialAbort('Transport closed the jaw opening below insertion clearance.')
        if row['phase'].startswith('gun_return_'):
            from wafer_tool_return import RETURN
            if row['air_on']:raise native.TrialAbort('Air remained on during gun return.')
            if any(c[2]>.01 for c in row['unexpected_gun_right_contacts']):raise native.TrialAbort('Unexpected gun contact during return.')
            if row['gun_stand_normal_force_N']>RETURN['maximum_support_force_N']:raise native.TrialAbort('Gun seating force exceeded the bound.')
            from wafer_tool_return import RELEASE_PHASES
            if row['phase']in RELEASE_PHASES:
                reference=_pose(self.metadata['gun_return']['reference_pose_world'],'reference');current=_pose(row['gun_pose_world'],'gun')
                if abs(row['gun_foot_gap_m'])>RETURN['maximum_absolute_foot_gap_m']or math.degrees(_angle(current[:3,:3],reference[:3,:3]))>RETURN['maximum_rotation_error_deg']or row['gun_support_margin_m']<RETURN['minimum_support_margin_m']:
                    raise native.TrialAbort('Gun lost stand support during release/retreat.')

    def before_prefix(self,env,base,metadata,recorder,plan,destination,torch):
        from yam_arena import ik,station,constants
        from wafer_tool_pickup import plan_tool_pickup,evaluate_tool_pickup,PROFILE,ToolPickupFeedback,tool_finger_target,jaw_normal_center_error
        from wafer_grasp_feedback import BoundedTCPFeedback
        origin=native._array(base.scene.env_origins,(1,3))[0];robot_base=np.asarray(station.BASES['left'])+origin
        def fk(q):return ik.tcp_pose(q,robot_base)
        def solve(p,r,q):return ik.solve(p,ik.matrix_to_quat_xyzw(r),q,base_pos=robot_base,iters=100,damping=.015,tol=1e-7)
        def solve_pose(p,r,q):
            value,pe,ae=ik.solve(p,ik.matrix_to_quat_xyzw(r),q,base_pos=robot_base,iters=600,damping=.012,tol=1e-7)
            if pe>.0001 or ae>.001:raise native.TrialAbort('Tool pickup IK exceeds the shared bounds.')
            return value
        def measured_tcp():return _pose(recorder.rows[-1]['measured_tcp_pose_world'],'tcp')
        initial=recorder.rows[0];matches={}
        for side,joint_key,pose_key,q in [('left','measured_arm_q_rad','measured_hand_pose_world',plan['initial_left_q_rad']),
                ('right','measured_right_arm_q_rad','right_hand_pose_world',plan['right_q_rad'])]:
            p,r=ik.tcp_pose(q,np.asarray(station.BASES[side])+origin);expected=np.eye(4)
            expected[:3,:3]=r;expected[:3,3]=p-r@np.asarray(constants.TCP_OFFSET);actual=_pose(initial[pose_key],side+' hand')
            pe=float(np.linalg.norm(actual[:3,3]-expected[:3,3]));ae=math.degrees(_angle(actual[:3,:3],expected[:3,:3]))
            if not np.allclose(initial[joint_key],q,atol=1e-5,rtol=0)or pe>1e-5 or ae>.001:raise ValueError('Native empty-hand initial robot pose differs.')
            matches[side]=dict(position_error_m=pe,orientation_error_deg=ae)
        for name,pose,expected in [('pliers',initial['clamp_pose_world'],metadata['tool_pickup']['initial_tool_pose_world']),
                ('gun',initial['gun_pose_world'],np.r_[np.asarray(self.export['initial_world_poses']['gun'][:3])+origin,self.export['initial_world_poses']['gun'][3:]])]:
            actual=_pose(pose,name);target_pose=_pose(expected,name+' initial');pe=float(np.linalg.norm(actual[:3,3]-target_pose[:3,3]))
            ae=math.degrees(_angle(actual[:3,:3],target_pose[:3,:3]))
            if pe>1e-5 or ae>.001:raise ValueError('Native empty-hand initial tool pose differs.')
            matches[name]=dict(position_error_m=pe,orientation_error_deg=ae)
        if abs(initial['hinge_angle_rad']-metadata['tool_pickup']['initial_passive_hinge_rad'])>1e-6:raise ValueError('Initial passive hinge differs.')
        initial_joints=metadata['tool_pickup']['initial_joint_positions']
        paddle_initial=[initial_joints[s['joint_name']]for s in metadata['handle_paddles']]
        for key,target_value,count in [('measured_finger_q_m',PROFILE['open_finger_m'],2),('measured_right_finger_q_m',-.035,2),
                ('handle_paddle_q_rad',paddle_initial,2),('pad_compression_m',0.,2),('passive_joint_velocities',0.,3),('handle_paddle_qvel_rad_s',0.,2)]:
            actual=np.asarray(initial[key],dtype=float)
            if actual.shape!=(count,)or not np.allclose(actual,target_value,atol=1e-6,rtol=0):raise ValueError('Initial measured joint state differs: '+key)
            matches[key]=dict(maximum_absolute_error=float(np.max(abs(actual-target_value))),tolerance=1e-6)
        if max(initial['finger_handle_contact_normal_force_N']+initial['gun_grip_normal_force_N'])>.005:raise native.TrialAbort('A hand initially contacts a tool.')
        if min(metadata['initial_empty_hand_geometry'].values())<=.002:raise ValueError('Local initial separation lacks the declared pose margin.')
        metadata['initial_empty_hand_match']=dict(passed=True,pose_checks=matches,
            local_geometry=metadata['initial_empty_hand_geometry'],pose_tolerance_m=1e-5,orientation_tolerance_deg=.001,
            scope='Shared collider geometry and measured empty-hand initialization; native pliers start with a 0.0013 rad resting-angle correction to remove stand penetration. No direct native distance query.')
        action=np.r_[plan['initial_left_q_rad'],PROFILE['open_finger_m']/constants.FINGER_OPEN,plan['right_q_rad'],-.035/constants.FINGER_OPEN]
        servo=BoundedTCPFeedback(fk,solve,ik.LOWER,ik.UPPER);tool_servo=None;state={};revision=0
        target=_pose(metadata['tool_pickup']['target_tool_pose_world'],'tool target')
        self.tool_grasp_point=target[:3,:3].T@(np.asarray(metadata['tool_pickup']['target_tcp_position_world_m'])-target[:3,3])
        try:
            with (Path(destination)/'tool_commands.jsonl').open('x')as stream:
                path=plan_tool_pickup(metadata,lambda:recorder.rows[-1]['clamp_pose_world'],measured_tcp,
                    lambda:action[:6].copy(),solve_pose,ik.slerp_matrix,state,read_state=lambda:recorder.rows[-1])
                for phase,duration,target_q,finger in path:
                    if state.get('relation_revision',0)!=revision:
                        revision=state['relation_revision'];relation=np.asarray(state['tool_to_tcp'])
                        tool_servo=ToolPickupFeedback(fk,solve,ik.LOWER,ik.UPPER,tool_to_tcp=relation)
                    q0=action[:6].copy();f0=action[6]*constants.FINGER_OPEN
                    print('Tool phase: '+phase,flush=True)
                    for index in range(round(duration*20)):
                        s=(index+1)/(duration*20);u=s*s*(3-2*s)
                        nominal=target_q[index]if target_q.ndim==2 else target_q if tool_servo is not None else q0+(target_q-q0)*u
                        if tool_servo is None:
                            tcp=measured_tcp();command,feedback=servo.step(nominal,tcp[:3,3],tcp[:3,:3],action[:6],integrate=phase=='tool_grip_hold')
                        else:
                            tool_pose=_pose(recorder.rows[-1]['clamp_pose_world'],'tool')
                            command,feedback=tool_servo.step(nominal,tool_pose[:3,3],tool_pose[:3,:3],action[:6],
                                integrate=phase in ('tool_lift_hold','tool_transport_hold','tool_align_hold','tool_insert','tool_dock'),
                                integrate_rotation=phase not in PROFILE['translation_only_phases']and phase!='tool_insert'and(phase not in ('tool_align_hold','tool_dock')or index/20<duration-PROFILE['alignment_orientation_settle_s']),
                                normal_axis=_pose(recorder.rows[-1]['wafer_pose_world'],'wafer')[:3,2]if phase in ('tool_align_hold','tool_insert','tool_dock')else None,
                                normal_error_m=jaw_normal_center_error(metadata,recorder.rows[-1])if phase in ('tool_align_hold','tool_insert','tool_dock')else None,normal_only=phase=='tool_insert',damped_translation=phase in ('tool_align_hold','tool_insert','tool_dock'),rotation_pivot_world=np.mean([p[:3]for p in recorder.rows[-1]['jaw_link_poses_world']],axis=0)if phase=='tool_dock'else None,settle_hold=phase=='tool_dock')
                        action[:6]=command;action[6]=float(tool_finger_target(phase,(index+1)*.05,duration,f0,finger,metadata))/constants.FINGER_OPEN
                        stream.write(json.dumps(dict(time_s=recorder.total*recorder.dt,phase=phase,action=action.tolist(),tcp_feedback=feedback))+'\n');stream.flush()
                        recorder.begin(phase);env.step(torch.tensor([action.tolist()],device=base.device,dtype=torch.float32));row=recorder.end()
                    if phase=='tool_grip_hold':
                        tail=recorder.rows[-round(.05/metadata['physics_dt_s']):]
                        forces=np.asarray([r['finger_handle_contact_normal_force_N']for r in tail]);gaps=np.zeros(2);maximum_gap=0.
                        for force in forces:
                            gaps=np.where(force<=.005,gaps+metadata['physics_dt_s'],0.);maximum_gap=max(maximum_gap,float(max(gaps)))
                        if min(forces.mean(axis=0))<PROFILE['minimum_50ms_mean_finger_force_N']or maximum_gap>PROFILE['maximum_contact_gap_s']+1e-10:
                            raise native.TrialAbort('Bilateral finger contact window did not pass before tool lift.')
                        self.tool_reference=np.linalg.inv(_pose(row['measured_hand_pose_world'],'hand'))@_pose(row['clamp_pose_world'],'tool');self.tool_handle_reference=row
        except RuntimeError as exc:
            print('Tool stage stopped: '+str(exc),flush=True)
            if 'Open jaw alignment is not ready' in str(exc):raise native.TrialAbort(str(exc))from exc
            raise
        finally:
            self.tool_report=evaluate_tool_pickup(metadata,recorder.rows);metadata['tool_pickup']['result']=self.tool_report
            if 'insertion_clearance'in state:metadata['tool_pickup']['insertion_clearance']=state['insertion_clearance']
            native._json(Path(destination)/'tool_pickup_report.json',self.tool_report)
        if not self.tool_report['passed']:raise native.TrialAbort('Measured tool pickup/docking checks did not pass.')
        plan.update(self.wafer_plan)
        try:q0,previous,handoff=measured_prefix_handoff(recorder.rows[-1],fk,solve_pose)
        except RuntimeError as exc:raise native.TrialAbort(str(exc))from exc
        metadata['tcp_feedback_handoff']=handoff;p,r=fk(q0)
        seating=float(metadata.get('fixture_seating_lift_m',metadata.get('arm_seating_lift_m',0.)))
        seated=solve_pose(p+[0,0,seating],r,q0);top=solve_pose(p+[0,0,seating+.020],r,seated)
        plan.update(initial_left_q_rad=q0.tolist(),seated_left_q_rad=seated.tolist(),final_left_q_rad=top.tolist())
        metadata.update(initial_arm_q_rad=q0.tolist(),seated_arm_q_rad=seated.tolist(),final_arm_q_rad=top.tolist())
        if abs(action[6]*constants.FINGER_OPEN-PROFILE['handoff_finger_m'])>1e-12:raise ValueError('Tool handoff target differs.')
        self.workflow_stage='wafer_cycle';self.preload_reference['control_index']=None
        view=StageRecorderView(recorder)
        metadata['controller']['clock_origin_global_s']=view.origin_s
        metadata['controller']['phases']=[list(p)for p in wafer_prefix_phases(metadata)]
        metadata['full_stage_boundaries'].append(dict(stage='wafer_cycle',global_time_s=view.origin_s,initial_sample_index=view.initial_index))
        return view

    def initialize_prefix_feedback(self,servo,plan,metadata):
        saved=metadata['tcp_feedback_handoff']
        servo.translation=np.asarray(saved['translation_correction_world_m']).copy()
        servo.rotation=np.asarray(saved['rotation_correction_world_rad']).copy()
        return np.asarray(saved['previous_command_arm_q_rad']).copy()

    def configure_prefix_feedback(self,servo,plan,metadata,initial_row,read_row):
        from wafer_grasp_feedback import JawPlaneFeedback
        return JawPlaneFeedback(servo,metadata,initial_row,plan['initial_left_q_rad'],read_row)

    def run(self,env,base,metadata,recorder,previous_action,preload,previous_fingers,destination,torch,prefix):
        from yam_arena import ik,station,constants
        from wafer_tool_return import RETURN,plan_gun_return,plan_gun_release,release_ready,retreat_ready,gun_seat_increment
        from wafer_cycle_protocol import GunFeedback,gun_motion_timing
        timing=gun_motion_timing(metadata)
        stage_result=super().run(env,base,metadata,recorder,previous_action,preload,previous_fingers,destination,torch,prefix)
        native._json(Path(destination)/'wafer_cycle_report.json',stage_result)
        if stage_result['result']!='WAFER_AIR_PATH_AND_RETURN_SUCCEEDED_IN_THIS_SIMULATION':return stage_result
        raw=recorder.raw;start_index=len(raw.rows)-1;self.workflow_stage='gun_return';self.preload_reference['control_index']=None
        metadata['full_stage_boundaries'].append(dict(stage='gun_return',global_time_s=raw.total*raw.dt,initial_sample_index=start_index))
        reference=next(r for r in reversed(raw.rows)if r['phase']=='gun_settle')
        metadata['gun_return']=dict(profile=dict(RETURN),reference_pose_world=reference['gun_pose_world'])
        row=raw.rows[-1];action=np.r_[row['commanded_arm_q_rad'],row['commanded_finger_q_m'][0]/constants.FINGER_OPEN,
            row['commanded_right_arm_q_rad'],row['commanded_right_finger_q_m'][0]/constants.FINGER_OPEN]
        origin=native._array(base.scene.env_origins,(1,3))[0]
        def tcp(q):return ik.tcp_pose(q,np.asarray(station.BASES['right'])+origin)
        def solve(p,r,q,side='right',iters=600):
            return ik.solve(p,ik.matrix_to_quat_xyzw(r),q,base_pos=np.asarray(station.BASES[side])+origin,iters=iters,damping=.012,tol=1e-7)
        def solve_pose(p,r,q,side='right'):
            value,pe,ae=solve(p,r,q,side)
            if pe>.0001 or ae>.001:raise native.TrialAbort('Gun return IK exceeds shared bounds.')
            return value
        with (Path(destination)/'gun_return_commands.jsonl').open('x')as stream:
            def execute(phase,duration,target,grip,servo=None,seat_load=False):
                start=action[7:13].copy();f0=action[13]*constants.FINGER_OPEN;seat_command=None;load_shift=np.zeros(3)
                print(phase,flush=True)
                for index in range(round(duration*20)):
                    s=(index+1)/(duration*20);u=s*s*(3-2*s);feedback=None
                    q=target[index].copy()if target.ndim==2 else start+(target-start)*u
                    if servo is not None:
                        current=raw.rows[-1]
                        if seat_command is None:
                            gun=_pose(current['gun_pose_world'],'gun');q,feedback=servo.step(q,gun[:3,3],gun[:3,:3],action[7:13],integrate=True)
                            if phase in ('gun_return_lower','gun_return_seat_hold')and current['gun_stand_vertical_force_N']>=.20*(metadata['gun']['mass_kg']+metadata['gun']['trigger_mass_kg'])*9.81 and abs(current['gun_foot_gap_m'])<=RETURN['maximum_absolute_foot_gap_m']:
                                seat_command=q.copy()
                        else:q=seat_command.copy()
                    if seat_load:
                        delta,force=gun_seat_increment(metadata,raw.rows,raw.dt);q=action[7:13].copy()
                        load_shift+=delta
                        if np.linalg.norm(load_shift)>RETURN['maximum_load_command_shift_m']:raise native.TrialAbort('Gun load-transfer travel bound exceeded.')
                        p,r=tcp(q);q=solve_pose(p+delta,r,q)
                        feedback=dict(mode='support_load_transfer',measured_support_force_N=force,command_z_shift_m=float(load_shift[2]),command_shift_world_m=load_shift.tolist(),
                            measurement_window_s=RETURN['support_force_window_s'],force_band_N=RETURN['load_force_band_N'],maximum_command_shift_m=RETURN['maximum_load_command_shift_m'])
                    limit=timing['maximum_joint_step_deg']if timing and phase in ('gun_return_raise','gun_return_transfer','gun_return_upright','gun_return_above')else 1.
                    if np.max(np.abs(q-action[7:13]))>math.radians(limit):raise native.TrialAbort('Gun return command jump exceeded the phase-specific bound.')
                    action[7:13]=q;action[13]=(f0+(grip-f0)*u)/constants.FINGER_OPEN
                    stream.write(json.dumps(dict(time_s=raw.total*raw.dt,phase=phase,action=action.tolist(),gun_return_feedback=feedback))+'\n');stream.flush()
                    raw.begin(phase);env.step(torch.tensor([action.tolist()],device=base.device,dtype=torch.float32));raw.end()
            execute('gun_return_start',timing['return_start_s']if timing else 1.,action[7:13].copy(),RETURN['close_target_m'])
            state={};servo=None
            try:
                commanded=None
                if timing:
                    p,r=tcp(action[7:13]);commanded=np.eye(4);commanded[:3,3]=p;commanded[:3,:3]=r
                path=plan_gun_return(metadata,lambda:raw.rows[-1],action[7:13].copy(),solve_pose,ik.slerp_matrix,constants.TCP_OFFSET,state,commanded)
                for phase,duration,target,grip in path:
                    if phase in ('gun_return_align','gun_return_lower')and servo is None:
                        relation=state['gun_to_tcp'];inverse=np.linalg.inv(relation)
                        def fk(q):
                            p,r=tcp(q);return p+r@inverse[:3,3],r@inverse[:3,:3]
                        def inverse_solve(p,r,q):return solve(p+r@relation[:3,3],r@relation[:3,:3],q,iters=100)
                        servo=GunFeedback(fk,inverse_solve,ik.LOWER,ik.UPPER)
                    execute(phase,duration,target,grip,servo=servo)
            except RuntimeError as exc:
                if 'exceeds1degree'in str(exc):raise native.TrialAbort(str(exc))from exc
                raise
            execute('gun_return_seat_hold',timing['seat_hold_s']if timing else 8.,action[7:13].copy(),RETURN['seat_target_m'],seat_load=True)
            gate=release_ready(metadata,raw.rows,raw.dt);metadata['gun_return']['release_gate']=gate
            if not gate['passed']:raise native.TrialAbort('Gun is not stably supported; release was not commanded.')
            for phase,duration,target,grip in plan_gun_release(action[7:13].copy(),solve_pose,tcp,metadata):
                execute(phase,duration,target,grip)
                if phase=='gun_return_release_hold':
                    gate=retreat_ready(metadata,raw.rows,raw.dt);metadata['gun_return']['retreat_gate']=gate
                    if not gate['passed']:raise native.TrialAbort('Gun is not independently supported; retreat was not commanded.')
        result=evaluate_native_full(metadata,raw.rows,destination)
        native._json(Path(destination)/'full_workflow_check.json',result)
        return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--bundle',type=Path,required=True);parser.add_argument('--output',type=Path)
    parser.add_argument('--device',choices=('cpu','cuda:0'),default='cuda:0')
    parser.add_argument('--check-only',action='store_true');parser.add_argument('--development',action='store_true');args=parser.parse_args()
    ALLOW_UNQUALIFIED_DEVELOPMENT=args.development
    import wafer_full_cycle_isaac as validation_module
    validation_module.ALLOW_UNQUALIFIED_DEVELOPMENT=args.development
    if args.check_only:
        plan,exported,models,gate=native.validate_bundle(args.bundle);metadata=native.cloud_model_metadata(exported['metadata'],plan)
        FullCycle().validate(args.bundle,plan,metadata)
        print(json.dumps(dict(status='FULL_WORKFLOW_BUNDLE_CHECKED',local_gate=metadata['full_local_gate']),indent=2))
    elif args.output is None:parser.error('--output is required for a simulation.')
    else:native.run_isaac_trial(args.bundle,args.output,extension=FullCycle(),device=args.device)
