"""Recompute task scores and command/geometry checks from the closed native trace."""
from pathlib import Path
import argparse,gzip,hashlib,json,math,sys
import numpy as np


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def validate_development_scene(metadata,bundle):
    names=metadata.get('develop_scene_objects')
    required={'clamp','wafer','heater','pillar_0','pillar_1','pillar_2','beaker','timer'}
    if (not isinstance(names,list) or not all(isinstance(n,str)for n in names)
        or names!=sorted(set(names)) or not required.issubset(names)):
        raise ValueError('Missing live development-scene object inventory.')
    evidence=metadata.get('development_scene_exclusions',{})
    if not isinstance(evidence,dict):raise ValueError('Invalid development-scene exclusion evidence.')
    for name in ('gun','gun_stand'):
        item=evidence.get(name,{})
        if (not isinstance(item,dict) or name in names or item.get('prim_path')!='/World/envs/env_0/'+name
            or any(item.get(k)is not False for k in ('configured','prim_present','asset_present'))
            or (Path(bundle)/'assets'/f'{name}.usda').exists()):
            raise ValueError('Gun-free development scene was not verified: '+name)
    return dict(passed=True,scene_objects=names,excluded_objects=['gun','gun_stand'],
        scope='Source-bound live object and USD-prim absence records; excluded asset files are absent from this bundle.')


def validate_development_row(row):
    removed=[k for k in row if k.startswith(('gun_','trigger_','nozzle_'))
        or k in ('air_on','scan_row','unexpected_gun_right_contacts')]
    fixtures=row.get('fixture_poses_world',{})
    if removed or any(n in fixtures for n in ('gun','gun_stand')):
        raise ValueError('Removed gun state remains in the development recording.')


def audit_countdown_commands(rows,commands,physics_dt_s,timer_started_s,target_s=30.):
    if timer_started_s is None:raise ValueError('Countdown never started.')
    expiry=float(timer_started_s)+target_s;read_count=0;first=None
    for command in commands:
        t=float(command['time_s'])
        if not math.isfinite(t):raise ValueError('Nonfinite countdown command time.')
        index=round(t/physics_dt_s)
        if index<0 or index>=len(rows):raise ValueError('Countdown command has no preceding physics sample.')
        row=rows[index]
        if abs(row['time_s']-t)>1e-8:raise ValueError('Countdown command is between physics samples.')
        reading=command.get('timer_reading_s')
        if reading is not None:
            if not math.isfinite(reading) or abs(reading-row['timer_remaining_s'])>1e-8:
                raise ValueError('Controller countdown differs from its preceding measured sample.')
            read_count+=1
        if command['phase'] in ('develop_hold','develop_withdraw') and reading is None:
            raise ValueError('Countdown observation missing during hold or withdrawal.')
        if command['phase']=='develop_withdraw' and first is None:
            if reading!=0. or row['timer_remaining_s']!=0. or row['timer_elapsed_s']<target_s-1e-8:
                raise ValueError('Withdrawal began before the measured countdown reached zero.')
            if not -1e-8<=t-expiry<=.05+1e-8:
                raise ValueError('Withdrawal did not begin at the first control interval after countdown expiry.')
            first=t
    if first is None:raise ValueError('No withdrawal command was recorded.')
    return dict(passed=True,reading_count=read_count,countdown_expiry_s=expiry,
        first_withdrawal_command_s=first,withdrawal_delay_after_zero_s=first-expiry,
        scope='Controller observations match preceding contact-derived timer samples; withdrawal starts at zero on the 20 Hz control clock.')


def audit_timer_start_after_immersion(rows,development,physics_dt_s,spec):
    from wafer_develop import immersion_geometry
    started=development['timer_started_s'];entered=development['immersion_start_s']
    if started is None or entered is None:
        return dict(passed=False,reason='Missing contact-derived timer start or full immersion.')
    if not all(math.isfinite(t)for t in(started,entered)):raise ValueError('Nonfinite timer/immersion start.')
    index=round(started/physics_dt_s)
    if not 0<=index<len(rows)or abs(rows[index]['time_s']-started)>1e-8:
        raise ValueError('Timer start has no matching physics sample.')
    geometry=immersion_geometry(rows[index]['wafer_pose_world'],spec['beaker_origin_world_m'],spec)
    delay=started-entered;immersed=bool(geometry['fully_immersed'])
    return dict(passed=bool(0.<=delay<=spec['timer_start_sync_s']and immersed),
        full_immersion_start_s=entered,timer_started_s=started,timer_start_delay_s=delay,
        fully_immersed_at_timer_start=immersed,maximum_start_delay_s=spec['timer_start_sync_s'],
        scope='Contact-derived timer start follows full immersion; measured wafer geometry is wholly immersed at that same physics sample.')


def audit_return_feedback(rows,commands,dt,baseline_pose,expected_count,tcp_pose=None):
    from wafer_grasp_protocol import _pose
    from wafer_grasp_feedback import PROFILE,rotation_vector,rotation_matrix
    from wafer_develop import SPEC
    records=[c for c in commands if c.get('return_feedback')is not None]
    holds=[c for c in commands if c['phase']=='return_hold']
    if len(records)!=expected_count or not records or records[0]is not holds[0]:
        raise ValueError('Return feedback is missing from the alignment interval.')
    def vector(value,shape):
        value=np.asarray(value,dtype=float)
        if value.shape!=shape or not np.isfinite(value).all():raise ValueError('Invalid return feedback array.')
        return value
    def same(actual,expected,label):
        actual=vector(actual,np.asarray(expected).shape)
        if not np.allclose(actual,expected,atol=1e-10,rtol=0):raise ValueError('Return feedback differs: '+label)
    def cap(value,limit):
        length=np.linalg.norm(value)
        return value if length<=limit else value*(limit/length)
    first=records[0];first_index=round(first['time_s']/dt);entry=rows[first_index]
    wafer=_pose(entry['wafer_pose_world'],'return entry');baseline=_pose(baseline_pose,'settled wafer')
    relation=np.linalg.inv(wafer)@_pose(entry['measured_tcp_pose_world'],'entry TCP');desired=baseline@relation
    error_p=float(np.linalg.norm(wafer[:3,3]-baseline[:3,3]))
    error_r=math.degrees(float(np.linalg.norm(rotation_vector(baseline[:3,:3]@wafer[:3,:3].T))))
    if max(error_p,entry['wafer_return_position_error_m'])>SPEC['return_position_tolerance_m']or max(error_r,entry['wafer_return_rotation_deg'])>SPEC['return_orientation_tolerance_deg']:
        raise ValueError('Return feedback began outside measured placement limits.')
    nominal=vector(first['return_feedback']['nominal_q_rad'],(6,))
    if not np.allclose(nominal.astype(np.float32),entry['commanded_arm_q_rad'],atol=1e-7,rtol=0):
        raise ValueError('Return nominal differs from the preceding actuator command.')
    anchor=vector(first['return_feedback']['nominal_tcp_position_world_m'],(3,))
    nominal_r=None
    if tcp_pose is not None:
        nominal_p,nominal_r=tcp_pose(nominal)
        if np.linalg.norm(anchor-nominal_p)>1e-6:raise ValueError('Return nominal FK differs from the recorded base frame.')
    previous_q=nominal.copy();previous_p=np.zeros(3);previous_r=np.zeros(3)
    for index,command in enumerate(records):
        rec=command['return_feedback'];t=float(command['time_s']);sample=round(t/dt)
        if (command['phase']!='return_hold'or abs(t-first['time_s']-index*.05)>1e-8
            or abs(rows[sample]['time_s']-t)>1e-8 or rec['command_index']!=index+1 or rec['integrated']is not True):
            raise ValueError('Return feedback chronology differs.')
        actual=_pose(rows[sample]['wafer_pose_world'],'measured wafer');equivalent=actual@relation
        ep=desired[:3,3]-equivalent[:3,3];er=rotation_vector(desired[:3,:3]@equivalent[:3,:3].T)
        for key,expected in [('nominal_q_rad',nominal),('nominal_tcp_position_world_m',anchor),
            ('actual_wafer_position_world_m',actual[:3,3]),('actual_wafer_rotation_world',actual[:3,:3]),
            ('measured_equivalent_tcp_position_world_m',equivalent[:3,3]),
            ('desired_equivalent_tcp_position_world_m',desired[:3,3]),('desired_equivalent_tcp_rotation_world',desired[:3,:3]),
            ('virtual_tcp_position_world_m',anchor-ep),('measured_position_error_m',ep),('measured_orientation_error_rad',er)]:
            same(rec[key],expected,key)
        same(rec['entry_check']['position_error_m'],error_p,'entry position')
        same(rec['entry_check']['rotation_error_deg'],error_r,'entry rotation')
        corrections=[]
        for error,previous,kind in ((ep,previous_p,'translation'),(er,previous_r,'rotation')):
            unit='m'if kind=='translation'else'rad'
            maximum=PROFILE[f'maximum_{kind}_correction_{unit}'];increment=PROFILE[f'maximum_{kind}_increment_{unit}']
            expected=previous.copy()
            if np.linalg.norm(error)>PROFILE[f'{kind}_deadband_{unit}']:
                expected+=cap(PROFILE['integral_gain_per_command']*error,increment)
            expected=cap(expected,maximum)
            key=f'{kind}_correction_world_{unit}';correction=vector(rec[key],(3,));same(correction,expected,key)
            if np.linalg.norm(correction)>maximum+1e-12 or np.linalg.norm(correction-previous)>increment+1e-12:
                raise ValueError('Return correction exceeded its additional-command budget.')
            corrections.append(correction)
        previous_p,previous_r=corrections;same(rec['requested_tcp_position_world_m'],anchor+previous_p,'requested TCP')
        q=vector(command['action'][:6],(6,));joint_step=float(np.max(np.abs(q-previous_q)))
        same(rec['maximum_joint_increment_rad'],joint_step,'joint increment')
        if joint_step>PROFILE['maximum_joint_increment_rad']+1e-12:raise ValueError('Return command increment exceeds its bound.')
        if tcp_pose is not None:
            p,r=tcp_pose(q);pe=np.linalg.norm(p-anchor-previous_p)
            ae=np.linalg.norm(rotation_vector(rotation_matrix(previous_r)@nominal_r@r.T))
            if pe>PROFILE['maximum_ik_position_residual_m']or ae>PROFILE['maximum_ik_rotation_residual_rad']:
                raise ValueError('Return commanded FK differs from its bounded target.')
        previous_q=q
    return dict(passed=True,command_count=len(records),fk_check='AVAILABLE'if tcp_pose is not None else'UNAVAILABLE',
        scope='Measured wafer and preceding-command binding, frozen handover, virtual error math and additional correction bounds; physical acceptance remains separate.')


def audit_placement_release(rows,commands,dt,metadata,spec):
    from wafer_grasp_protocol import pillar_plane_clearances
    releases=[c for c in commands if c['phase']=='release']
    gates=[c for c in commands if c.get('placement_preopen_gate')is not None]
    if not releases or len(gates)!=1 or gates[0]is not releases[0]:
        raise ValueError('Expected one placement gate on the first release command.')
    command=gates[0];index=round(command['time_s']/dt);count=round(spec['placement_support_window_s']/dt)
    if count<1 or abs(count*dt-spec['placement_support_window_s'])>1e-9 or index-count+1<0 or abs(rows[index]['time_s']-command['time_s'])>1e-8:
        raise ValueError('Placement gate has no complete preceding physics window.')
    window=rows[index-count+1:index+1]
    if any(abs(r['time_s']-(index-count+1+i)*dt)>1e-8 for i,r in enumerate(window)):
        raise ValueError('Placement support window is discontinuous.')
    geometry=[pillar_plane_clearances(r['wafer_pose_world'],metadata)for r in window]
    contacts=np.array([r['pillar_contact_normal_force_N']for r in window]);gaps=np.array([g['clearances_m']for g in geometry])
    if contacts.shape!=(count,3)or gaps.shape!=(count,3)or not np.isfinite(contacts).all()or np.any(contacts<0.)or not np.isfinite(gaps).all():
        raise ValueError('Invalid placement support evidence.')
    forces=contacts.sum(axis=1)
    covered=all(g['all_pillars_under_wafer']for g in geometry)
    current=bool(geometry[-1]['all_pillars_under_wafer']and max(abs(x)for x in gaps[-1])<=.0005 and forces[-1]>.05)
    expected=dict(window_s=spec['placement_support_window_s'],samples=count,start_time_s=window[0]['time_s'],end_time_s=window[-1]['time_s'],
        mean_support_force_N=float(forces.mean()),minimum_support_force_N=float(forces.min()),maximum_absolute_gap_m=float(np.abs(gaps).max()),
        all_footprints_covered=bool(covered),current_on_pillars=current,continuous=True)
    aligned=all(0.<=r['wafer_return_position_error_m']<=spec['return_position_tolerance_m']and
        0.<=r['wafer_return_rotation_deg']<=spec['return_orientation_tolerance_deg']for r in window)
    expected['passed']=bool(expected['mean_support_force_N']>spec['placement_min_mean_support_N']and expected['minimum_support_force_N']>0.
        and expected['maximum_absolute_gap_m']<=spec['placement_ready_gap_m']and covered and current and aligned)
    rec=command['placement_preopen_gate']
    for key,value in expected.items():
        if isinstance(value,bool):same=rec[key]is value
        else:same=math.isfinite(float(rec[key]))and abs(rec[key]-value)<=1e-10
        if not same:raise ValueError('Placement gate differs from physical evidence: '+key)
    if not expected['passed']:raise ValueError('Jaws opened without the required persistent support.')
    last=rows[-round(spec['return_hold_s']/dt):]
    final=bool(len(last)==round(spec['return_hold_s']/dt)and all(r['wafer_released']and r['wafer_all_pillars_loaded']and r['wafer_on_pillars']
        and r['wafer_return_stable']and r['wafer_return_position_error_m']<=spec['return_position_tolerance_m']
        and r['wafer_return_rotation_deg']<=spec['return_orientation_tolerance_deg']for r in last))
    return dict(passed=final,preopen=expected,final_unassisted_hold_s=spec['return_hold_s'],
        final_minimum_pillar_forces_N=np.min([r['pillar_contact_normal_force_N']for r in last],axis=0).tolist(),
        final_maximum_jaw_force_N=max(max(r['jaw_contact_normal_force_N'])for r in last),
        scope='Persistent measured contact before opening; every final hold sample has released jaws and all three loaded pillars.')


def audit_release_feedback(rows,commands,dt,expected_count,tcp_pose=None):
    from wafer_grasp_protocol import _pose
    from wafer_grasp_feedback import PROFILE,rotation_vector,rotation_matrix
    from wafer_develop import SPEC
    release=[c for c in commands if c['phase']=='release']
    if len(release)!=expected_count or not release or any(c.get('release_feedback')is None for c in release):
        raise ValueError('Adaptive release diagnostics are incomplete.')
    if any(c.get('release_feedback')is not None and c['phase']!='release'for c in commands):
        raise ValueError('Release feedback appeared outside jaw opening.')
    def same(value,expected,name):
        value=np.asarray(value,dtype=float);expected=np.asarray(expected)
        if value.shape!=expected.shape or not np.isfinite(value).all()or not np.allclose(value,expected,atol=1e-10,rtol=0):
            raise ValueError('Release feedback differs: '+name)
    def cap(value,limit):
        size=np.linalg.norm(value)
        return value if size<=limit else value*(limit/size)
    first=release[0];entry=rows[round(first['time_s']/dt)]
    wafer0=_pose(entry['wafer_pose_world'],'seated wafer');desired=_pose(entry['measured_tcp_pose_world'],'release TCP')
    relation=np.linalg.inv(wafer0)@desired
    nominal=np.asarray(first['release_feedback']['nominal_q_rad'],dtype=float)
    if nominal.shape!=(6,)or not np.isfinite(nominal).all()or not np.allclose(nominal.astype(np.float32),entry['commanded_arm_q_rad'],atol=1e-7,rtol=0):
        raise ValueError('Release nominal differs from the preceding actuator command.')
    anchor=np.asarray(first['release_feedback']['nominal_tcp_position_world_m'],dtype=float)
    if anchor.shape!=(3,)or not np.isfinite(anchor).all():raise ValueError('Invalid release nominal TCP.')
    nominal_r=None
    if tcp_pose is not None:
        p,nominal_r=tcp_pose(nominal)
        if np.linalg.norm(p-anchor)>1e-6:raise ValueError('Release nominal FK differs from the recorded base frame.')
    previous_q=nominal.copy();previous_p=np.zeros(3);previous_r=np.zeros(3);frozen_q=None;frozen_time=None
    for index,command in enumerate(release):
        t=float(command['time_s']);sample=round(t/dt);row=rows[sample];rec=command['release_feedback']
        if abs(t-first['time_s']-index*.05)>1e-8 or abs(row['time_s']-t)>1e-8 or rec['command_index']!=index+1:
            raise ValueError('Release feedback chronology differs.')
        if rec['controller']!='measured_wafer_release_v1'or rec['gain_per_command']!=SPEC['release_feedback_gain_per_command']or rec['integrated']is not False:
            raise ValueError('Release feedback profile differs.')
        wafer=_pose(row['wafer_pose_world'],'release wafer');measured=wafer@relation
        ep=desired[:3,3]-measured[:3,3];er=rotation_vector(desired[:3,:3]@measured[:3,:3].T)
        released=bool(row['wafer_released']and row['wafer_all_pillars_loaded'])
        if released and frozen_q is None:frozen_q=previous_q.copy();frozen_time=t
        if rec['released_and_all_pillars_loaded']is not released or rec['frozen_after_release']is not(frozen_q is not None):
            raise ValueError('Release freeze differs from measured release and support.')
        for key,expected in [('nominal_q_rad',nominal),('nominal_tcp_position_world_m',anchor),
            ('actual_wafer_position_world_m',wafer[:3,3]),('actual_wafer_rotation_world',wafer[:3,:3]),
            ('desired_wafer_pose_world',entry['wafer_pose_world']),('measured_equivalent_tcp_position_world_m',measured[:3,3]),
            ('desired_equivalent_tcp_position_world_m',desired[:3,3]),('desired_equivalent_tcp_rotation_world',desired[:3,:3]),
            ('virtual_tcp_position_world_m',anchor-ep),('measured_position_error_m',ep),('measured_orientation_error_rad',er)]:
            same(rec[key],expected,key)
        corrections=[]
        for error,previous,kind in ((ep,previous_p,'translation'),(er,previous_r,'rotation')):
            unit='m'if kind=='translation'else'rad';maximum=PROFILE[f'maximum_{kind}_correction_{unit}'];increment=PROFILE[f'maximum_{kind}_increment_{unit}']
            expected=previous.copy()
            if frozen_q is None and np.linalg.norm(error)>PROFILE[f'{kind}_deadband_{unit}']:
                expected=cap(expected+cap(SPEC['release_feedback_gain_per_command']*error,increment),maximum)
            key=f'{kind}_correction_world_{unit}';same(rec[key],expected,key)
            if np.linalg.norm(expected)>maximum+1e-12 or np.linalg.norm(expected-previous)>increment+1e-12:raise ValueError('Release correction exceeds its bounds.')
            corrections.append(expected)
        previous_p,previous_r=corrections;same(rec['requested_tcp_position_world_m'],anchor+previous_p,'requested TCP')
        q=np.asarray(command['action'][:6],dtype=float)
        if q.shape!=(6,)or not np.isfinite(q).all():raise ValueError('Invalid release arm command.')
        joint_step=float(np.max(np.abs(q-previous_q)));same(rec['maximum_joint_increment_rad'],joint_step,'joint increment')
        if joint_step>PROFILE['maximum_joint_increment_rad']+1e-12:raise ValueError('Release joint increment exceeds its bound.')
        if frozen_q is not None and not np.array_equal(q,frozen_q):raise ValueError('Arm did not freeze after measured release and support.')
        if tcp_pose is not None:
            p,r=tcp_pose(q);pe=np.linalg.norm(p-anchor-previous_p);ae=np.linalg.norm(rotation_vector(rotation_matrix(previous_r)@nominal_r@r.T))
            if pe>PROFILE['maximum_ik_position_residual_m']or ae>PROFILE['maximum_ik_rotation_residual_rad']:raise ValueError('Release commanded FK differs from its bounded target.')
        previous_q=q
    retreat=next((c for c in commands if c['phase']=='retreat'and c['time_s']>release[-1]['time_s']),None)
    if retreat is None or abs(retreat['time_s']-release[-1]['time_s']-.05)>1e-8:raise ValueError('Release was not followed by the recorded retreat.')
    ready=rows[round(retreat['time_s']/dt)]
    if abs(ready['time_s']-retreat['time_s'])>1e-8 or not(ready['wafer_released']and ready['wafer_on_pillars']):
        raise ValueError('Retreat began before measured release onto the pillars.')
    return dict(passed=True,command_count=len(release),frozen_after_release_s=frozen_time,retreat_start_s=retreat['time_s'],fk_check='AVAILABLE'if tcp_pose is not None else'UNAVAILABLE',
        scope='Measured seated-wafer target, bounded compensation while jaws open, and exact arm-command freeze after measured release and three-pillar support.')


def verify(bundle,run):
    sys.path[:0]=[str(bundle),str(bundle/'benchmark')]
    from wafer_develop import DevelopmentScore,SPEC
    from wafer_grasp_protocol import _pose,_angle,pillar_plane_clearances,pad_contact_geometry,evaluate_wafer_grasp_trial
    from wafer_trace_rows import TraceRowList
    from wafer_tool_pickup import evaluate_tool_pickup
    from wafer_full_cycle_isaac import ToolHandMonitor,compact_evaluation_rows,compact_guard_rows,evaluate_handle_paddles,evaluate_tool_environment
    from wafer_finger_preload_v12 import wafer_prefix_phases,wafer_return_phases,FingerPreloadController,preceding_interval_features
    from wafer_robot_geometry import evaluate_native_robot_geometry,RobotGeometryGuard
    import wafer_retention_isaac_v12 as native
    meta=json.loads((run/'metadata.json').read_text());report=json.loads((run/'trial_report.json').read_text())
    checks={};errors=[];trace=run/'samples.jsonl.gz';trace_sha=sha(trace)
    closed={n:sha(run/n)for n in ('metadata.json','trial_report.json','tool_commands.jsonl','commands.jsonl','develop_commands.jsonl','finger_preload.jsonl')}
    try:
        sources=dict(meta['full_workflow_sources'])
        for n,digest in meta['develop_sources'].items():
            if n in sources and sources[n]!=digest:raise ValueError('Conflicting source binding: '+n)
            sources[n]=digest
        for n,digest in sources.items():
            if Path(n).is_absolute()or'..'in Path(n).parts:raise ValueError('Source binding leaves the bundle: '+n)
            if sha(bundle/n)!=digest:raise ValueError('Changed source: '+n)
        checks['development_scene']=validate_development_scene(meta,bundle)
        if any(meta['develop'].get(k)!=v for k,v in SPEC.items()):raise ValueError('Task specification differs.')
        with gzip.open(trace,'rt')as f:rows=TraceRowList((json.loads(line)for line in f),cache_blocks=32)
        print(f'Loaded {len(rows)} physics samples for independent audit.',flush=True)
        dt=float(meta['physics_dt_s'])
        if not math.isfinite(dt)or dt<=0.:raise ValueError('Invalid physics timestep.')
        stride=round(.05/dt)
        if stride<1 or abs(stride*dt-.05)>1e-12:raise ValueError('Physics cadence does not divide the 20 Hz control interval.')
        if meta.get('post_start_pose_write_count')!=0 or not meta.get('full_workflow_fresh_start'):raise ValueError('Fresh continuous initialization not verified.')
        if max(rows[0]['finger_handle_contact_normal_force_N'])>.005:raise ValueError('Pliers already contacted at start.')
        if not np.allclose(rows[0]['measured_right_arm_q_rad'],SPEC['initial_right_arm_q_rad'],atol=1e-5,rtol=0):raise ValueError('Right starting pose differs.')
        origin=next(b for b in meta['full_stage_boundaries']if b['stage']=='wafer_cycle');start=origin['initial_sample_index']
        end=start+round(sum(d for _,d in wafer_prefix_phases(meta))/dt)
        checks['tool_pickup']=evaluate_tool_pickup(meta,rows[:start+1])
        stage=TraceRowList(dict(r,time_s=r['time_s']-origin['global_time_s'])for r in rows[start:])
        stage[0]=dict(stage[0],phase='initialization',finger_preload_command_index=None)
        checks['wafer_pickup']=evaluate_wafer_grasp_trial(meta,compact_evaluation_rows(stage[:end-start+1]))
        print('Tool and wafer pickup audits complete; checking full development trace.',flush=True)
        geometry=RobotGeometryGuard.from_evidence(meta['robot_geometry'])
        hands={key:np.concatenate([m.vertices for m in pieces])for key,pieces in zip(geometry.keys,geometry.meshes)if key[1]in('gripper','tip_left','tip_right')}
        clamp_shapes=[p for p in json.loads((bundle/'assets/develop_clamp_vertices.json').read_text())if any(k in p['name']for k in('handle','backing','grip'))]
        body_fields={'lower_half':('lever_link_poses_world',0),'upper_half':('lever_link_poses_world',1),'lower_pad':('jaw_link_poses_world',0),'upper_pad':('jaw_link_poses_world',1),'lower_grip_paddle':('handle_paddle_poses_world',0),'upper_grip_paddle':('handle_paddle_poses_world',1)}
        scorer=DevelopmentScore(SPEC['beaker_origin_world_m'],dt);monitor=ToolHandMonitor(meta)
        initial=_pose(rows[0]['wafer_pose_world'],'wafer');reference=None;previous=None;gap_s=0.;maximum_button_error=0.;contact_peak=0.;button_peak=0.;settled_pose=None
        for index,row in enumerate(rows):
            validate_development_row(row)
            if abs(row['time_s']-index*dt)>1e-8:raise ValueError('Missing or reordered physics sample.')
            if row['phase']=='settle':settled_pose=row['wafer_pose_world']
            active=index>end
            if bool(row['pickup_verified'])!=active:raise ValueError('Pickup transition differs from the measured prefix.')
            w=_pose(row['wafer_pose_world'],'wafer');c=_pose(row['clamp_pose_world'],'clamp')
            if active and reference is None:reference=np.linalg.inv(_pose(previous['clamp_pose_world'],'clamp'))@_pose(previous['wafer_pose_world'],'wafer')
            relation=np.linalg.inv(c)@w
            slip=0. if reference is None else float(np.linalg.norm(relation[:3,3]-reference[:3,3]))
            angle=0. if reference is None else math.degrees(_angle(relation[:3,:3],reference[:3,:3]))
            pillar=pillar_plane_clearances(row['wafer_pose_world'],meta)if w[2,2]>.5 else dict(clearances_m=[1.]*3,all_pillars_under_wafer=False)
            supported=bool(pillar['all_pillars_under_wafer']and max(abs(x)for x in pillar['clearances_m'])<=.0005 and sum(row['pillar_contact_normal_force_N'])>.05)
            bilateral=min(row['jaw_contact_normal_force_N'])>.005 and pad_contact_geometry(row,meta)['both_engaged']
            gap_s=0. if bilateral or not active else gap_s+dt
            speed=omega=0.
            if previous is not None:
                old=_pose(previous['wafer_pose_world'],'previous');speed=float(np.linalg.norm(w[:3,3]-old[:3,3])/dt);omega=float(_angle(w[:3,:3],old[:3,:3])/dt)
            base=_pose(row['timer_base_pose_world'],'timer');button=_pose(row['timer_button_pose_world'],'button');local=base[:3,:3].T@(button[:3,3]-base[:3,3])
            error=abs(SPEC['timer_button_local_m'][2]-local[2]-row['timer_button_displacement_m']);maximum_button_error=max(maximum_button_error,error)
            if error>5e-6:raise ValueError('Button displacement differs from measured body poses.')
            if abs(sum(row['timer_tip_force_N'])-row['timer_button_force_N'])>1e-8:raise ValueError('Button force differs from filtered tip contacts.')
            beaker=_pose(row['beaker_pose_world'],'beaker');origin_world=beaker[:3,3]
            if np.linalg.norm(origin_world-np.asarray(SPEC['beaker_origin_world_m']))>1e-6 or _angle(beaker[:3,:3],np.eye(3))>1e-6:raise ValueError('Beaker moved from the fixed scoring frame.')
            vertices=[]
            for shape in clamp_shapes:
                field,link=body_fields[shape['body']];pose=_pose(row[field][link],shape['body'])
                vertices.append(np.asarray(shape['vertices'])@pose[:3,:3].T+pose[:3,3])
            poses=dict(zip(geometry.keys,row['robot_link_poses_world']))
            for key,mesh in hands.items():
                pose=_pose(poses[key],str(key));vertices.append(mesh@pose[:3,:3].T+pose[:3,3])
            dry=1.
            for mesh in vertices:
                low,high=mesh.min(0),mesh.max(0)
                if np.linalg.norm(np.clip(origin_world[:2],low[:2],high[:2])-origin_world[:2])<=SPEC['beaker_inner_radius_m']:
                    dry=min(dry,float(low[2]-origin_world[2]-SPEC['liquid_height_m']))
            monitor.observe(row)
            unexpected=sum(row[k]for k in('unexpected_robot_tool_contact_normal_force_N','unexpected_robot_environment_contact_normal_force_N','forbidden_wafer_contact_force_N','tool_unexpected_environment_contact_normal_force_N'))
            contact_peak=max(contact_peak,unexpected);button_peak=max(button_peak,row['timer_button_force_N'])
            if row.get('timer_reading_s')is not None and not math.isfinite(row['timer_reading_s']):raise ValueError('Nonfinite controller timer reading.')
            recomputed=dict(unexpected_contact_N=unexpected,hand_handle_liquid_clearance_m=dry,wafer_slip_m=slip,wafer_rotation_error_deg=angle,wafer_on_pillars=supported,
                wafer_pillar_plane_gaps_m=pillar['clearances_m'],wafer_pillar_footprints_covered=bool(pillar['all_pillars_under_wafer']),
                wafer_all_pillars_loaded=bool(min(row['pillar_contact_normal_force_N'])>SPEC['minimum_final_pillar_contact_N']),
                wafer_bilaterally_held=bilateral or gap_s<=SPEC['maximum_grip_contact_gap_s'],
                wafer_return_position_error_m=float(np.linalg.norm(w[:3,3]-initial[:3,3])),wafer_return_rotation_deg=math.degrees(_angle(w[:3,:3],initial[:3,:3])),
                wafer_return_stable=speed<.001 and omega<.02,
                wafer_released=row['phase']in('release','retreat','unassisted_hold')and max(row['jaw_contact_normal_force_N'])<.005 and row['measured_jaw_gap_m']>.002,
                tool_hand_retained=not monitor.envelope.failed)
            for key,value in recomputed.items():
                if isinstance(value,(bool,np.bool_)):
                    if bool(row[key])!=value:raise ValueError('Derived evidence differs: '+key)
                elif isinstance(value,(list,tuple,np.ndarray)):
                    if np.asarray(row[key]).shape!=np.asarray(value).shape or not np.allclose(row[key],value,atol=1e-8,rtol=0):raise ValueError('Derived evidence differs: '+key)
                elif abs(row[key]-value)>1e-8:raise ValueError('Derived evidence differs: '+key)
            if row['phase']=='release'and (not pillar['all_pillars_under_wafer']
                or max(abs(x)for x in pillar['clearances_m'])>SPEC['placement_release_gap_m']
                or recomputed['wafer_return_position_error_m']>SPEC['return_position_tolerance_m']
                or recomputed['wafer_return_rotation_deg']>SPEC['return_orientation_tolerance_deg']):
                raise ValueError('Wafer left the allowed placement geometry while the jaws opened.')
            scorer.update({**row,**recomputed,'objects_intact':row['develop_objects_intact']})
            previous=row
            if (index+1)%25000==0:print(f'Development evidence: {index+1}/{len(rows)} samples checked.',flush=True)
        checks['development']=scorer.result();checks['tool_retention']=monitor.report()
        checks['timer_start_after_full_immersion']=audit_timer_start_after_immersion(rows,checks['development'],dt,SPEC)
        checks['timer_pose_readback']=dict(passed=True,maximum_displacement_difference_m=maximum_button_error)
        checks['all_stage_contacts']=dict(passed=contact_peak<=SPEC['maximum_unexpected_contact_N']and button_peak<=3.,maximum_unexpected_force_N=contact_peak,maximum_timer_force_N=button_peak)
        guards=compact_guard_rows(rows)
        checks['handle_capacity']=evaluate_handle_paddles(meta,guards,operating=True)
        checks['tool_obstacles']=evaluate_tool_environment(guards)
        checks['finger_load']=native.evaluate_finger_handle_load(guards,engine='isaac')
        checks['passive_pads']=native.evaluate_passive_pad_state(guards,meta['pad_compliance'])
        checks['passive_hinge']=native.evaluate_passive_hinge_state(guards,meta['passive_hinge'])
        commands=[]
        for name,clock in [('tool_commands.jsonl',0.),('commands.jsonl',origin['global_time_s']),('develop_commands.jsonl',0.)]:
            for line in (run/name).open():
                item=json.loads(line);commands.append((item['start_time_s']+clock,item['target'],item['phase'])if name=='commands.jsonl'else(item['time_s']+clock,item['action'],item['phase']))
        commands.sort(key=lambda x:x[0]);den=meta['controller']['native_finger_open_q_m']
        if len(rows)!=1+stride*len(commands):raise ValueError('Command/physics sample counts differ.')
        for index,(timestamp,action,phase)in enumerate(commands):
            if abs(timestamp-index*.05)>1e-8:raise ValueError('Global command gap.')
            action=np.asarray(action,dtype=np.float32).astype(float)
            expected=dict(commanded_arm_q_rad=action[:6],commanded_finger_q_m=[action[6]*den]*2,commanded_right_arm_q_rad=action[7:13],commanded_right_finger_q_m=[action[13]*den]*2)
            for row in rows[1+index*stride:1+(index+1)*stride]:
                if row['phase']!=phase:raise ValueError('Command phase differs.')
                if any(not np.allclose(row[k],v,atol=1e-7,rtol=0)for k,v in expected.items()):raise ValueError('Actuator readback differs from command.')
        checks['commands']=dict(passed=True,count=len(commands))
        develop_commands=[json.loads(line)for line in(run/'develop_commands.jsonl').open()]
        checks['placement_release']=audit_placement_release(rows,develop_commands,dt,meta,SPEC)
        checks['countdown_observation']=audit_countdown_commands(rows,develop_commands,dt,checks['development']['timer_started_s'],SPEC['target_dwell_s'])
        tcp_pose=None
        try:
            from yam_arena import ik
        except ImportError:pass
        else:
            base_index=[tuple(k)for k in meta['robot_geometry']['link_keys']].index(('left','base'))
            base_pose=_pose(rows[0]['robot_link_poses_world'][base_index],'left robot base')
            if np.allclose(base_pose[:3,:3],np.eye(3),atol=1e-6,rtol=0):
                tcp_pose=lambda q:ik.tcp_pose(q,base_pose[:3,3])
        hold_seconds=next(d for phase,d in wafer_return_phases(meta)if phase=='return_hold')
        checks['return_feedback']=audit_return_feedback(rows,develop_commands,dt,settled_pose,round(hold_seconds/.05),tcp_pose)
        release_seconds=next(d for phase,d in wafer_return_phases(meta)if phase=='release')
        checks['release_feedback']=audit_release_feedback(rows,develop_commands,dt,round(release_seconds/.05),tcp_pose)
        preload_records=[json.loads(line)for line in(run/'finger_preload.jsonl').open()]
        prefix_commands=[json.loads(line)for line in(run/'commands.jsonl').open()]
        checks['prefix_preload_evidence']=dict(passed=native.audit_finger_preload_evidence(stage[:end-start+1],preload_records,prefix_commands,den,dt,wafer_prefix_phases(meta))['complete'])
        controller=FingerPreloadController(wafer_prefix_phases(meta))
        for rec in preload_records:
            _,expected=controller.step(rec['phase'],rec['interval'],rec['previous_command_q_m'])
            if expected!=rec:raise ValueError('Prefix preload replay differs.')
        for command in develop_commands:
            rec=command['preload']
            if rec is None:continue
            phase=rec['phase']
            if controller.phases[-1][0]==phase:
                name,duration=controller.phases[-1];controller.phases[-1]=(name,duration+.05)
            else:controller.phases.append((phase,.05))
            sample=round((command['time_s']-origin['global_time_s'])/dt)
            measured=preceding_interval_features(stage,rec['control_index'],dt,prefix_length=sample+1)
            if rec['interval']!=measured:raise ValueError('Preload measurement interval differs.')
            _,expected=controller.step(phase,rec['interval'],rec['previous_command_q_m'])
            if expected!=rec:raise ValueError('Development preload replay differs.')
        checks['preload_replay']=dict(passed=True,command_count=controller.command_index)
        print('Task, physical contact, and command audit complete; checking robot geometry.',flush=True)
        checks['robot_geometry']=evaluate_native_robot_geometry(meta,rows)
        physical=(checks['tool_pickup']['passed']and checks['wafer_pickup']['result']==native.PASS and checks['development']['success']and all(v.get('passed')is True for k,v in checks.items()if k not in ('development','wafer_pickup')))
        matches=all(report.get(k)==v for k,v in checks['development'].items()if k not in ('scope','checks')) and all(report.get('checks',{}).get(k)==v for k,v in checks['development']['checks'].items())
        if any(sha(run/n)!=d for n,d in closed.items()):raise ValueError('Closed evidence changed during audit.')
        if sha(trace)!=trace_sha or any(sha(bundle/n)!=digest for n,digest in sources.items()):raise ValueError('Trace or bound source changed during audit.')
        passed=physical and matches and report['result']=='WAFER_DEVELOP_SUCCEEDED_IN_THIS_SIMULATION'
        if passed:
            with gzip.open(run/'video_samples.jsonl.gz','wt',compresslevel=1)as f:
                for row in rows[::stride]:f.write(json.dumps(dict(row),separators=(',',':'))+'\n')
            (run/'video_replay_manifest.json').write_text(json.dumps(dict(original_trace_sha256=trace_sha,selected_trace_sha256=sha(run/'video_samples.jsonl.gz')),indent=2))
    except (ValueError,KeyError,TypeError,IndexError,OSError)as exc:
        errors.append(type(exc).__name__+': '+str(exc));passed=False;matches=False
    result=dict(passed=bool(passed),matches_report=bool(matches),checks=checks,errors=errors,trace_sha256=trace_sha,source_files_sha256=closed,audit_source_sha256=sha(__file__),scope='Closed native physics measurements and source-bound commands; declared elapsed-time development surrogate, not calibrated chemistry.')
    (run/'independent_check.json').write_text(json.dumps(result,indent=2));print(json.dumps(dict(passed=result['passed'],errors=errors)),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);p.add_argument('--run',type=Path,required=True);a=p.parse_args();verify(a.bundle,a.run)
