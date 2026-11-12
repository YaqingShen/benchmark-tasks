"""Measured air-path coverage and unassisted return; no evaporation model."""
import math
import numpy as np
from wafer_grasp_protocol import _pose,_angle,_relative,DEFAULT_LIMITS
from wafer_grasp_feedback import rotation_vector,rotation_matrix,_bounded,BoundedTCPFeedback

SCAN=dict(rows=18,standoff_m=.080,tilt_deg=45.,speed_m_s=.025,
          footprint_radius_m=.005,minimum_exposure_s=.050,grid_pitch_m=.002,
          protected_flat_strip_m=.015,scope='Geometric air exposure only; no calibrated drying or cleaning physics.')

LAYOUT=dict(revision='left_side_180_v3',world_yaw_deg=270.,center_xy_m=[.42,-.14],
            handle_remap='Opposite-side grip; mirrored lateral guides and rear stop. Passive jaws unchanged.')
GUN_OFF_GRIP_M=-.0200

GUN_RETENTION=dict(revision='impulse_and_measured_pose_v1',window_s=.05,
    minimum_mean_finger_force_N=.75,maximum_contact_gap_s=.001,
    maximum_grip_translation_m=.0015,maximum_relative_rotation_deg=2.)

FULL_GUN_TIMING=dict(revision='complete_workflow_120s_gun_v12',scan_speed_m_s=.100,
    nominal_joint_step_deg=.90,empty_hand_joint_step_deg=.99,maximum_joint_step_deg=1.6,transport_joint_step_deg=1.3,scan_azimuth_deg=160.,scan_roll_deg=0.,
    pickup_phases=[['gun_settle',.25],['gun_approach',3.],['gun_close',1.5],['gun_hold',1.],['gun_lift',1.5],['gun_lift_hold',.75]],
    scan_minimum_s=dict(gun_raise_clear=1.,gun_rotate_clear=1.,gun_aim_above=1.,gun_aim=.75,gun_clear=.75),
    air_on_s=1.0,air_off_s=.5,turn_dwell_s=.05,row_center_extent_m=.045,
    row_transverse_calibration_m=.002,arc_overscan_m=.003,
    return_minimum_s=dict(gun_return_raise=.5,gun_return_transfer=1.,gun_return_upright=1.,gun_return_above=1.,gun_return_align=1.,gun_return_lower=3.),
    return_start_s=.25,seat_hold_s=4.,
    release_phases=[['gun_return_release',1.],['gun_return_release_hold',.2],['gun_return_retreat',1.],['gun_return_park',1.],['gun_return_free_hold',1.05]])


def gun_motion_timing(metadata):
    from wafer_finger_preload_v12 import workflow_timing
    return FULL_GUN_TIMING if workflow_timing(metadata) else None


def bounded_gun_segment(seed,p0,r0,p1,r1,relation,solve_pose,slerp,minimum_s,*,speed_m_s=None,nominal_step_deg=None):
    """Retiming changes duration only; every target retains the Cartesian path."""
    count=max(1,math.ceil(minimum_s*20-1e-10))
    if speed_m_s is not None:count=max(count,math.ceil(np.linalg.norm(p1-p0)/speed_m_s*20-1e-10))
    step=FULL_GUN_TIMING['nominal_joint_step_deg']if nominal_step_deg is None else nominal_step_deg
    if not 0<step<=FULL_GUN_TIMING['maximum_joint_step_deg']:raise ValueError('Invalid planned gun command increment.')
    limit=math.radians(step)
    for _ in range(8):
        q=np.asarray(seed).copy();commands=[];peak=0.
        for i in range(count):
            s=(i+1)/count;p=p0+(p1-p0)*s;r=slerp(r0,r1,s)
            new=np.asarray(solve_pose(p+r@relation[:3,3],r@relation[:3,:3],q,'right'))
            peak=max(peak,float(np.max(np.abs(new-q))));commands.append(new.copy());q=new
        if peak<=limit+1e-12:return count/20,np.asarray(commands)
        count=max(count+1,math.ceil(count*peak/limit)+1)
    raise RuntimeError('Gun trajectory cannot meet its declared command-step bound.')


def reverse_handle_features(tree,metadata=None):
    changes=[]
    for geom in tree.findall('.//geom'):
        name=geom.get('name','')
        if not(name.startswith(('lower_','upper_'))and('_handle_'in name or name.endswith('_robot_handle'))):continue
        old=np.fromstring(geom.get('pos','0 0 0'),sep=' ');new=old.copy();new[1]*=-1
        if not name.endswith('_robot_handle'):new[0]=2*(-.0725)-new[0]
        sign=1. if name.startswith('lower_')else -1.
        angle=sign*.04;rotation=rotation_matrix(np.array([0.,angle,0.]))
        anchor=np.array([-.0725,-.008,sign*.025])
        new=anchor+rotation@(new-anchor)
        geom.set('euler',f'0 {angle:.12g} 0')
        geom.set('pos',' '.join(f'{v:.12g}'for v in new))
        changes.append(dict(name=name,old_local_m=old.tolist(),new_local_m=new.tolist(),cant_rad=angle))
    if len(changes)!=7:raise ValueError('Expected the two handles and five passive locating features.')
    if metadata is not None:
        for change in changes:
            part=next(p for p in metadata['geometry']['primitive_parts_original_tool_frame']if p['name']==change['name'])
            part['center']=(np.array(part['center'])+np.array(change['new_local_m'])-np.array(change['old_local_m'])).tolist()
            part['rotate_y']=math.degrees(change['cant_rad'])
        metadata['cycle_handle_remap']=changes
        metadata['cycle_layout']=dict(LAYOUT)
    return changes


def retreat_offset(rows):
    baseline=_pose(next(r for r in reversed(rows)if r['phase']=='settle')['wafer_pose_world'],'wafer')
    return baseline[:3,:3]@np.array([-.025,0.,0.])


class GunFeedback:
    """Bounded gun-pose correction; arm effort limits are unchanged."""
    def __init__(self,fk,solve,lower,upper):
        self.fk,self.solve=fk,solve;self.lower,self.upper=np.array(lower),np.array(upper)
        self.translation=np.zeros(3);self.rotation=np.zeros(3);self.index=0

    def step(self,nominal,measured_p,measured_r,previous,*,integrate=True):
        p,r=self.fk(nominal);error=p-measured_p;angular=rotation_vector(r@measured_r.T)
        if integrate:
            self.translation=_bounded(self.translation+_bounded(.2*error,.00025),.010)
            self.rotation=_bounded(self.rotation+_bounded(.2*angular,math.radians(.04)),math.radians(2))
        target_p=p+self.translation;target_r=rotation_matrix(self.rotation)@r
        q,_,_=self.solve(target_p,target_r,previous);actual_p,actual_r=self.fk(q)
        pe=float(np.linalg.norm(actual_p-target_p));ae=float(np.linalg.norm(rotation_vector(target_r@actual_r.T)))
        if not np.isfinite(q).all()or np.any(q<self.lower)or np.any(q>self.upper)or pe>.00002 or ae>math.radians(.01):
            raise RuntimeError('Gun feedback IK or joint limits failed.')
        if np.max(np.abs(q-previous))>math.radians(1):raise RuntimeError('Gun command exceeds1degree per50ms.')
        self.index+=1
        return q,dict(command_index=self.index,nominal_q_rad=np.asarray(nominal).tolist(),
            measured_position_error_m=error.tolist(),measured_orientation_error_rad=angular.tolist(),
            translation_correction_world_m=self.translation.tolist(),rotation_correction_world_rad=self.rotation.tolist(),
            correction_limit_m=.010,ik_position_residual_m=pe,ik_orientation_residual_rad=ae,
            saturated_translation=bool(np.linalg.norm(self.translation)>=.010-1e-12))


def coverage_points():
    axis=np.arange(-.049,.049001,SCAN['grid_pitch_m'])
    points=np.array([(x,y)for x in axis for y in axis])
    flat=-math.sqrt(.05**2-(.0325/2)**2)
    mask=(np.linalg.norm(points,axis=1)<=.049)&(points[:,0]>=flat+.001)
    shadow=(points[:,0]<flat+SCAN['protected_flat_strip_m'])&(np.abs(points[:,1])<=.0325/2+.005)
    return points[mask&~shadow]


def air_coverage(rows,physics_dt):
    points=coverage_points();exposure=np.zeros(len(points));valid=0;invalid=0;row_ids=set();last_time=None
    for row in rows:
        t=float(row['time_s'])
        if last_time is not None and abs(t-last_time-physics_dt)>1e-8:
            raise ValueError('Coverage trace has missing physics samples.')
        last_time=t
        if row.get('phase')not in ('air_on','air_scan'):continue
        if row.get('trigger_compression_m',0)<.0022:continue
        wafer=_pose(row['wafer_pose_world'],'wafer');normal=wafer[:3,2]
        axis=np.array(row['nozzle_axis_world']);tip=np.array(row['nozzle_tip_world_m'])
        if abs(np.linalg.norm(axis)-1)>1e-6:raise ValueError('Invalid nozzle direction.')
        incidence=-float(axis@normal)
        if not math.cos(math.radians(60))<=incidence<=math.cos(math.radians(30)):
            invalid+=1;continue
        distance=float((tip-wafer[:3,3]-normal*.0005)@normal/incidence)
        if not .050<=distance<=.110:invalid+=1;continue
        hit=(tip+axis*distance-wafer[:3,3])@wafer[:3,:3]
        exposure[np.sum((points-hit[:2])**2,axis=1)<=SCAN['footprint_radius_m']**2]+=physics_dt
        valid+=1
        if row.get('scan_row',-1)>=0:row_ids.add(int(row['scan_row']))
    covered=exposure>=SCAN['minimum_exposure_s']
    return dict(passed=bool(covered.all()and row_ids==set(range(SCAN['rows']))),
        coverage_fraction=float(covered.mean()),covered_points=int(covered.sum()),eligible_points=len(points),
        minimum_exposure_s=float(exposure.min()),valid_air_samples=valid,invalid_air_samples=invalid,
        observed_rows=sorted(row_ids),missing_points_m=points[~covered].tolist(),
        excluded='1mm outer rim and the15mm flat-edge clamp shadow; not claimed dry.',
        scope=SCAN['scope'])


def held_wafer_check(rows):
    reference=next(r for r in reversed(rows)if r['phase']=='grip_hold')
    relative=_relative(_pose(reference['clamp_pose_world'],'clamp'),_pose(reference['wafer_pose_world'],'wafer'))
    translation=[];rotation=[];forces=[]
    for row in rows:
        if not(row['phase'].startswith('gun_')or row['phase']in ('air_on','air_scan','air_off')):continue
        current=_relative(_pose(row['clamp_pose_world'],'clamp'),_pose(row['wafer_pose_world'],'wafer'))
        translation.append(float(np.linalg.norm(current[:3,3]-relative[:3,3])))
        rotation.append(math.degrees(_angle(current[:3,:3],relative[:3,:3])))
        forces.append(min(row['jaw_contact_normal_force_N']))
    if not translation:return dict(passed=False,reason='No drying-stage retention measurements.')
    return dict(passed=max(translation)<=DEFAULT_LIMITS['maximum_relative_translation_m']
        and max(rotation)<=DEFAULT_LIMITS['maximum_relative_rotation_deg']and min(forces)>=DEFAULT_LIMITS['minimum_jaw_contact_N'],
        maximum_relative_translation_m=max(translation),maximum_relative_rotation_deg=max(rotation),
        minimum_bilateral_pad_force_N=min(forces),sample_count=len(translation))


def gun_retention_check(rows,dt,grip_point):
    held=[r for r in rows if 'gun_pose_world'in r and r['phase']not in
        ('initialization','settle','close','grip_hold','lift','lift_hold',
         'gun_settle','gun_approach','gun_close','gun_hold')]
    reference=next((r for r in reversed(rows)if r['phase']=='gun_hold'),None)
    if not held or reference is None:return dict(passed=False,reason='Missing gun hold evidence.')
    def relative(row):
        gun=_pose(row['gun_pose_world'],'gun');hand=_pose(row['right_hand_pose_world'],'hand')
        transform=np.linalg.inv(hand)@gun
        midpoint=np.array(grip_point,dtype=float);midpoint[0]-=row['trigger_compression_m']/2
        return transform[:3,3]+transform[:3,:3]@midpoint,transform[:3,:3]
    point,rotation=relative(reference);translation=0.;angle=0.;gap=np.zeros(2);maximum_gap=0.
    forces=np.array([r['gun_grip_normal_force_N']for r in held]);width=round(GUN_RETENTION['window_s']/dt)
    valid=forces.shape==(len(held),2)and np.isfinite(forces).all()and len(held)>=width
    if not valid:return dict(passed=False,reason='Invalid or insufficient gun contact measurements.')
    sums=np.vstack([np.zeros((1,2)),np.cumsum(forces,axis=0)])
    minimum_mean=float(np.min((sums[width:]-sums[:-width])/width))
    for row,force in zip(held,forces):
        p,r=relative(row);translation=max(translation,float(np.linalg.norm(p-point)))
        angle=max(angle,math.degrees(_angle(r,rotation)))
        gap=np.where(force<=.005,gap+dt,0.);maximum_gap=max(maximum_gap,float(gap.max()))
    return dict(passed=minimum_mean>=GUN_RETENTION['minimum_mean_finger_force_N']
        and maximum_gap<=GUN_RETENTION['maximum_contact_gap_s']+1e-10
        and translation<=GUN_RETENTION['maximum_grip_translation_m']
        and angle<=GUN_RETENTION['maximum_relative_rotation_deg'],profile=dict(GUN_RETENTION),
        minimum_bilateral_instantaneous_force_N=float(forces.min()),
        minimum_50ms_mean_finger_force_N=minimum_mean,maximum_contact_gap_s=maximum_gap,
        maximum_measured_grip_translation_m=translation,maximum_relative_rotation_deg=angle,
        force_basis='0.084kg at friction0.6 requires0.687N per finger for gravity-only friction support;0.75N draft minimum.',
        scope='Prospective impulse/contact-gap and measured retention check; earlier instantaneous-contact failures remain failures.')


def evaluate_cycle(metadata,rows,mechanical_evaluator):
    projected=[{**r,'phase':'lift_hold'}if r['phase'].startswith('gun_')or r['phase']in ('air_on','air_scan','air_off')else r for r in rows]
    result=mechanical_evaluator(metadata,projected,require_return=True)
    result['phase_projection']='All gun and air phases extend the stationary wafer hold; no physics samples are omitted.'
    result['checks']['air_path_coverage']=air_coverage(rows,float(metadata['physics_dt_s']))
    result['checks']['wafer_retained_during_air_path']=held_wafer_check(rows)
    contacts=[c[2]for r in rows for c in r.get('unexpected_gun_right_contacts',[])]
    result['checks']['gun_and_right_arm_contacts']=dict(passed=not contacts or max(contacts)<=.01,
        maximum_unexpected_contact_N=max(contacts,default=0.))
    preair=[r for r in rows if r['phase']in ('gun_settle','gun_approach','gun_close','gun_hold','gun_lift','gun_lift_hold','gun_raise_clear','gun_rotate_clear','gun_aim_above','gun_aim')]
    result['checks']['air_off_before_aim']=dict(passed=bool(preair)and all(not r.get('air_on',True)for r in preair),sample_count=len(preair))
    result['checks']['air_off_before_return']=dict(passed=any(r['phase']=='air_off'for r in rows)
        and all(not r.get('air_on',True)for r in rows if r['phase']in ('lower','return_hold','release','retreat','unassisted_hold')))
    gun_rows=[r for r in rows if 'gun_pose_world'in r]
    result['checks']['gun_bilateral_retention']=gun_retention_check(rows,float(metadata['physics_dt_s']),metadata['gun']['grip_local_m'])
    trigger_valid=bool(gun_rows)and all('trigger_spring_force_N'in r and 'trigger_velocity_m_s'in r for r in gun_rows)
    result['checks']['passive_trigger_bounds']=dict(passed=trigger_valid and
        max(abs(r['trigger_spring_force_N'])for r in gun_rows)<=6. and max(abs(r['trigger_velocity_m_s'])for r in gun_rows)<=1.,
        maximum_unconstrained_force_N=max((abs(r.get('trigger_spring_force_N',0))for r in gun_rows),default=0.),
        maximum_speed_m_s=max((abs(r.get('trigger_velocity_m_s',0))for r in gun_rows),default=0.))
    result['failed_checks']=[k for k,v in result['checks'].items()if not v['passed']]
    result['result']=('WAFER_AIR_PATH_AND_RETURN_SUCCEEDED_IN_THIS_SIMULATION'
        if not result['failed_checks']and not result['errors']else'FAILED')
    result['scope']='Contact-based handling, geometric air coverage, and unassisted return. Actual dryness, cleanliness, gas forces and wafer stress are unvalidated.'
    return result


def plan_scan(read_measurement,seed,gun_spec,solve_pose,slerp,tcp_offset,controller_state=None,timing=None,commanded_tcp_pose=None):
    measurement=read_measurement()
    gun=_pose(measurement['gun_pose_world'],'gun');hand=_pose(measurement['right_hand_pose_world'],'hand')
    hand[:3,3]+=hand[:3,:3]@np.asarray(tcp_offset)
    relation=np.linalg.inv(gun)@hand
    if controller_state is not None:controller_state['gun_to_tcp']=relation
    wafer=_pose(measurement['wafer_pose_world'],'wafer')
    a=np.deg2rad(SCAN['tilt_deg']);G=np.array([[-np.cos(a),0,-np.sin(a)],[0,-1.,0],[-np.sin(a),0,np.cos(a)]])
    if timing is not None:
        if timing!=FULL_GUN_TIMING:raise ValueError('Unexpected full-workflow gun timing profile.')
        b=math.radians(timing['scan_azimuth_deg']);G=np.array([[np.cos(a)*np.cos(b),-np.sin(b),np.sin(a)*np.cos(b)],[np.cos(a)*np.sin(b),np.cos(b),np.sin(a)*np.sin(b)],[-np.sin(a),0.,np.cos(a)]])
    if timing is not None:
        c=math.radians(timing['scan_roll_deg'])
        G=G@np.array([[1,0,0],[0,np.cos(c),-np.sin(c)],[0,np.sin(c),np.cos(c)]])
    seed=np.asarray(seed,dtype=float)
    q=seed;current_p=gun[:3,3].copy();current_r=gun[:3,:3].copy()
    if timing is not None:
        commanded=np.asarray(commanded_tcp_pose,dtype=float)
        if commanded.shape!=(4,4)or not np.isfinite(commanded).all():raise ValueError('Full gun handoff requires the last commanded TCP pose.')
        nominal_gun=commanded@np.linalg.inv(relation)
        current_p=nominal_gun[:3,3].copy();current_r=nominal_gun[:3,:3].copy()
    def segment(phase,origin,rotation,duration,grip,row=-1):
        nonlocal q,current_p,current_r
        if timing is not None:
            minimum=timing['scan_minimum_s'].get(phase,.05)
            duration,commands=bounded_gun_segment(q,current_p,current_r,origin,rotation,relation,solve_pose,slerp,minimum,
                speed_m_s=timing['scan_speed_m_s']if phase=='air_scan'else None,nominal_step_deg=timing['transport_joint_step_deg']if phase!='air_scan'else None)
            q=commands[-1].copy();current_p=origin.copy();current_r=rotation.copy()
            return phase,duration,commands,grip,row
        commands=[]
        for i in range(round(duration*20)):
            s=(i+1)/(duration*20);u=s*s*(3-2*s)
            p=current_p+(origin-current_p)*u;r=slerp(current_r,rotation,u)
            q=solve_pose(p+r@relation[:3,3],r@relation[:3,:3],q,'right');commands.append(q.copy())
        current_p=origin.copy();current_r=rotation.copy()
        return phase,duration,np.array(commands),grip,row
    lifted=current_p+[0,0,.100]
    yield segment('gun_raise_clear',lifted,current_r,5.,GUN_OFF_GRIP_M)
    yield segment('gun_rotate_clear',lifted,G,6.,GUN_OFF_GRIP_M)
    previous=None
    extent=timing['row_center_extent_m']if timing else .049
    for row,y in enumerate(np.linspace(extent,-extent,SCAN['rows'])):
        wafer=_pose(read_measurement()['wafer_pose_world'],'wafer')
        chord=np.sqrt(max(0.,.05**2-y*y));flat=-np.sqrt(.05**2-(.0325/2)**2)
        lo=max(-chord,flat);hi=chord
        if abs(y)<=.0325/2+.005:lo=max(lo,flat+SCAN['protected_flat_strip_m'])
        if timing:
            if abs(lo+chord)<1e-10:lo-=timing['arc_overscan_m']
            hi+=timing['arc_overscan_m']
        target_y=y+((-1 if row%2==0 else 1)*timing['row_transverse_calibration_m']if timing else 0.)
        points=[np.array([lo,target_y,.0005]),np.array([hi,target_y,.0005])]
        if row%2:points.reverse()
        for index,point in enumerate(points):
            tip=wafer[:3,3]+wafer[:3,:3]@point-G[:,0]*SCAN['standoff_m']
            origin=tip-G@gun_spec['nozzle_local_m']
            if previous is None:
                yield segment('gun_aim_above',origin+[0,0,.040 if timing else .060],G,6.,GUN_OFF_GRIP_M)
                yield segment('gun_aim',origin,G,4.,GUN_OFF_GRIP_M)
                yield 'air_on',timing['air_on_s']if timing else 1.,q,-.0190,-1
            else:
                duration=max(1.,np.ceil(np.linalg.norm(point-previous)/SCAN['speed_m_s']/.05)*.05)
                yield segment('air_scan',origin,G,duration,-.0190,row if index==1 else -1)
                dwell=timing['turn_dwell_s']if timing else .75
                if index==0 and dwell:yield 'air_scan',dwell,q,-.0190,-1
            previous=point
    yield 'air_off',timing['air_off_s']if timing else 1.,q,GUN_OFF_GRIP_M,-1
    yield segment('gun_clear',current_p+([0.,-.040,.030]if timing else[-.060,0,.030]),G,4.,GUN_OFF_GRIP_M)



def plan_return(rows,solve_pose,tcp_pose,metadata=None):
    seat=next(r for r in reversed(rows) if r['phase']=='grip_hold')
    end=rows[-1]
    baseline=next(r for r in reversed(rows) if r['phase']=='settle')
    wafer=_pose(end['wafer_pose_world'],'wafer')
    tcp=_pose(end['measured_tcp_pose_world'],'tcp')
    desired=_pose(baseline['wafer_pose_world'],'baseline')@np.linalg.inv(wafer)@tcp
    q0=np.array(end['commanded_arm_q_rad']);command_p,command_r=tcp_pose(q0)
    p=desired[:3,3]+command_p-tcp[:3,3]
    r=command_r@tcp[:3,:3].T@desired[:3,:3]
    qseat=solve_pose(p,r,q0)
    offset=retreat_offset(rows)
    away=solve_pose(p+offset,r,qseat)
    raised=solve_pose(p+offset+[0,0,.030],r,away)
    closed=np.array(end['commanded_finger_q_m']);opened=np.array([-.02495]*2)
    from wafer_finger_preload_v12 import wafer_return_phases
    timing=wafer_return_phases(metadata or {})
    return [(phase,duration,q,f)for (phase,duration),(q,f)in zip(timing,
        [(qseat,closed),(qseat,closed),(qseat,opened),(away,opened),(raised,opened),(raised,opened)],strict=True)]


def make_return_feedback(rows,previous,tcp_pose,solve_ik,lower,upper):
    baseline=_pose(next(r for r in reversed(rows)if r['phase']=='settle')['wafer_pose_world'],'baseline')
    current=rows[-1]
    relation=np.linalg.inv(_pose(current['wafer_pose_world'],'wafer'))@_pose(current['measured_tcp_pose_world'],'tcp')
    desired=baseline@relation
    nominal,pe,ae=solve_ik(desired[:3,3],desired[:3,:3],previous)
    if pe>.0001 or ae>.001:raise ValueError('Return IK residual exceeds the existing bounds.')
    servo=BoundedTCPFeedback(tcp_pose,solve_ik,lower,upper)
    p,r=tcp_pose(previous);servo.translation=p-desired[:3,3];servo.rotation=rotation_vector(r@desired[:3,:3].T)
    if np.linalg.norm(servo.translation)>.006 or np.linalg.norm(servo.rotation)>math.radians(2):
        raise RuntimeError('Return feedback initial compensation exceeds the existing bounds.')
    class WaferReturnFeedback:
        def step(self,nominal,wafer_p,wafer_r,previous,*,integrate=True):
            p=wafer_p+wafer_r@relation[:3,3];r=wafer_r@relation[:3,:3]
            q,record=servo.step(nominal,p,r,previous,integrate=integrate)
            record['measurement']='Measured wafer pose mapped through the measured wafer-to-TCP relation; correction limits apply at TCP.'
            return q,record
    return WaferReturnFeedback(),nominal

