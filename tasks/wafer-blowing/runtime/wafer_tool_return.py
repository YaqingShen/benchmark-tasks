"""Contact-supported gun return, release, and retreat."""
import math
import numpy as np
from wafer_grasp_protocol import _pose, _angle
from wafer_cycle_protocol import GunFeedback, gun_retention_check, gun_motion_timing, bounded_gun_segment

RETURN = dict(revision='gun_stand_return_v21', clear_height_m=.975, seat_depth_m=.00025,
    maximum_position_error_m=.002, maximum_rotation_error_deg=2.,
    maximum_linear_speed_m_s=.001, maximum_angular_speed_rad_s=.02,raw_velocity_acceptance=False,maximum_position_excursion_m=.00025,maximum_orientation_excursion_deg=.25,
    maximum_absolute_foot_gap_m=.0005, minimum_support_margin_m=.002,
    minimum_seat_weight_fraction=.20, minimum_free_weight_fraction=.80,
    maximum_support_force_N=3., maximum_released_finger_force_N=.005,
    load_force_band_N=[.45,.65], load_step_m=.000040, maximum_load_command_shift_m=.002,
    final_window_s=1., support_force_window_s=.05, release_clear_window_s=.05, minimum_hand_retreat_m=.040, close_target_m=-.0198, seat_target_m=-.020, partial_open_target_m=-.0216, open_target_m=-.035,
    release_intermediate_target_m=-.0225,release_intermediate_fraction=.75,release_clearance_drop_m=.015,release_first_drop_fraction=.3333333333333333,
    seat_center_gain=.2,seat_center_step_m=.000020,seat_center_deadband_m=.0003)
RELEASE_PHASES = ('gun_return_release','gun_return_release_hold','gun_return_retreat','gun_return_park','gun_return_free_hold')


def gun_return_measurement(model, data, base_record=None):
    import mujoco
    from wafer_cycle import gun_measurement
    row = dict(gun_measurement(model,data) if base_record is None else base_record)
    force_z = 0.; normal_force = 0.
    for i,contact in enumerate(data.contact):
        names=[model.geom(contact.geom1).name,model.geom(contact.geom2).name]
        if set(names)!={'gun_stand','gun_hose_connector'}:continue
        wrench=np.zeros(6);mujoco.mj_contactForce(model,data,i,wrench)
        sign=1. if names[1]=='gun_hose_connector' else -1.
        world=np.asarray(contact.frame).reshape(3,3).T@wrench[:3]*sign
        force_z+=float(world[2]);normal_force+=max(0.,float(wrench[0]))
    foot=model.geom('gun_hose_connector').id;stand=model.geom('gun_stand').id
    axis=data.geom_xmat[foot].reshape(3,3)[:,2]
    radius,halfheight=model.geom_size[foot,:2]
    extent=radius*math.sqrt(max(0.,1-axis[2]**2))+halfheight*abs(axis[2])
    stand_top=float(data.geom_xpos[stand,2]+model.geom_size[stand,2])
    gap=float(data.geom_xpos[foot,2]-extent-stand_top)
    margin=float(np.min(model.geom_size[stand,:2]-np.abs(data.geom_xpos[foot,:2]-data.geom_xpos[stand,:2])-radius))
    velocity=np.zeros(6);mujoco.mj_objectVelocity(model,data,mujoco.mjtObj.mjOBJ_BODY,
                                               model.body('blow_gun').id,velocity,0)
    row.update(gun_stand_vertical_force_N=force_z,gun_stand_normal_force_N=normal_force,
        gun_foot_gap_m=gap,gun_support_margin_m=margin,gun_linear_speed_m_s=float(np.linalg.norm(velocity[3:])),
        gun_angular_speed_rad_s=float(np.linalg.norm(velocity[:3])))
    return row


def plan_gun_return(metadata, read_measurement, seed, solve_pose, slerp, tcp_offset, controller_state=None,commanded_tcp_pose=None):
    timing=gun_motion_timing(metadata)
    row=read_measurement();gun=_pose(row['gun_pose_world'],'gun')
    hand=_pose(row['right_hand_pose_world'],'hand');hand[:3,3]+=hand[:3,:3]@np.asarray(tcp_offset)
    relation=np.linalg.inv(gun)@hand
    if controller_state is not None:controller_state['gun_to_tcp']=relation
    base=np.asarray(metadata['gun_return']['reference_pose_world'],dtype=float)
    target=_pose(base.tolist(),'gun reference');q=np.asarray(seed,dtype=float)
    p=gun[:3,3].copy();r=gun[:3,:3].copy();height=max(RETURN['clear_height_m'],float(p[2]))
    if timing:
        commanded=np.asarray(commanded_tcp_pose,dtype=float)
        if commanded.shape!=(4,4)or not np.isfinite(commanded).all():raise ValueError('Full gun return requires the last commanded TCP pose.')
        nominal_gun=commanded@np.linalg.inv(relation);p=nominal_gun[:3,3].copy();r=nominal_gun[:3,:3].copy()
        height=max(RETURN['clear_height_m'],float(p[2]))
    def segment(name,destination,orientation,duration):
        nonlocal q,p,r
        if timing:
            duration,commands=bounded_gun_segment(q,p,r,np.asarray(destination),orientation,relation,solve_pose,slerp,timing['return_minimum_s'][name],nominal_step_deg=timing['transport_joint_step_deg']if name in ('gun_return_raise','gun_return_transfer','gun_return_upright','gun_return_above')else None)
            q=commands[-1].copy();p=np.asarray(destination).copy();r=np.asarray(orientation).copy()
            grip=RETURN['seat_target_m']if name in('gun_return_above','gun_return_lower')else RETURN['close_target_m']
            return name,duration,commands,grip
        commands=[]
        for i in range(round(duration*20)):
            s=(i+1)/(duration*20);u=s*s*(3-2*s)
            pos=p+(destination-p)*u;rot=slerp(r,orientation,u)
            qnew=solve_pose(pos+rot@relation[:3,3],rot@relation[:3,:3],q,'right')
            if np.max(np.abs(qnew-q))>math.radians(1):raise RuntimeError('Gun return exceeds1degree per50ms.')
            q=np.asarray(qnew);commands.append(q.copy())
        p=np.asarray(destination).copy();r=np.asarray(orientation).copy()
        grip=RETURN['seat_target_m'] if name in ('gun_return_above','gun_return_lower') else RETURN['close_target_m']
        return name,duration,np.array(commands),grip
    yield segment('gun_return_raise',np.array([p[0],p[1],height]),r,4.)
    above=np.array([target[0,3],target[1,3],height])
    yield segment('gun_return_transfer',above,r,10.)
    yield segment('gun_return_upright',above,target[:3,:3],24.)
    yield segment('gun_return_above',target[:3,3]+[0,0,.010],target[:3,:3],8.)
    yield 'gun_return_align',timing['return_minimum_s']['gun_return_align']if timing else 5.,q.copy(),RETURN['seat_target_m']
    yield segment('gun_return_lower',target[:3,3]-[0,0,RETURN['seat_depth_m']],target[:3,:3],20.)


def plan_gun_release(held, solve_pose, tcp_pose,metadata=None):
    timing=gun_motion_timing(metadata)if metadata is not None else None
    durations=dict(timing['release_phases'])if timing else {}
    q=np.asarray(held,dtype=float);p,r=tcp_pose(q)
    def segment(name,target,duration,opening):
        nonlocal q,p
        if timing:
            relation=np.eye(4)
            duration,commands=bounded_gun_segment(q,p,r,target,r,relation,solve_pose,lambda a,b,t:a,durations[name],
                nominal_step_deg=timing['empty_hand_joint_step_deg']if name=='gun_return_park'else None)
            q=commands[-1].copy();p=target.copy()
            return name,duration,commands,opening
        commands=[]
        for i in range(round(duration*20)):
            a=(i+1)/(duration*20);u=a*a*(3-2*a)
            q=solve_pose(p+(target-p)*u,r,q,'right');commands.append(q.copy())
        p=target.copy()
        return name,duration,np.array(commands),opening
    backward=-r[:,2].copy()
    if timing:
        total=durations['gun_return_release'];fraction=RETURN['release_intermediate_fraction']
        yield segment('gun_return_release',p+[0,0,-RETURN['release_clearance_drop_m']*RETURN['release_first_drop_fraction']],total*fraction,RETURN['release_intermediate_target_m'])
        yield segment('gun_return_release',p+[0,0,-RETURN['release_clearance_drop_m']*(1-RETURN['release_first_drop_fraction'])],total*(1-fraction),RETURN['open_target_m'])
    else:
        yield segment('gun_return_release',p+[0,0,-.002],2.,RETURN['partial_open_target_m'])
    opening=RETURN['open_target_m']if timing else RETURN['partial_open_target_m']
    yield 'gun_return_release_hold',durations.get('gun_return_release_hold',.5),q.copy(),opening
    yield segment('gun_return_retreat',p+backward*.020 if timing else p+[0,0,-.008],4.,RETURN['open_target_m'])
    yield segment('gun_return_park',p+backward*(.040 if timing else .060)+[0,0,.020 if timing else .030],6.,RETURN['open_target_m'])
    yield 'gun_return_free_hold',durations.get('gun_return_free_hold',3.),q.copy(),RETURN['open_target_m']


def _window(rows, phases, dt, duration=None):
    selected=[r for r in rows if r.get('phase') in phases]
    count=round((RETURN['final_window_s']if duration is None else duration)/dt)
    tail=selected[-count:]
    valid=len(tail)==count and all(abs(b['time_s']-a['time_s']-dt)<1e-8 for a,b in zip(tail,tail[1:]))
    return tail,valid


def _support_check(metadata,rows,dt,phases,free=False,duration=None):
    if metadata['gun_return'].get('profile')!=RETURN:
        return dict(passed=False,complete=False,reason='Gun support profile differs from this prospective evaluator.')
    tail,complete=_window(rows,phases,dt,duration)
    if not complete:return dict(passed=False,complete=False,reason='Missing full supported hold window.')
    means=_support_means(tail,dt)
    target=_pose(metadata['gun_return']['reference_pose_world'],'gun reference')
    position=[float(np.linalg.norm(_pose(r['gun_pose_world'],'gun')[:3,3]-target[:3,3]))for r in tail]
    angle=[math.degrees(_angle(_pose(r['gun_pose_world'],'gun')[:3,:3],target[:3,:3]))for r in tail]
    weight=(metadata['gun']['mass_kg']+metadata['gun']['trigger_mass_kg'])*9.81
    fraction=RETURN['minimum_free_weight_fraction'] if free else RETURN['minimum_seat_weight_fraction']
    metrics=dict(maximum_position_error_m=max(position),maximum_rotation_error_deg=max(angle),
        maximum_linear_speed_m_s=max(r['gun_linear_speed_m_s']for r in tail),
        maximum_angular_speed_rad_s=max(r['gun_angular_speed_rad_s']for r in tail),
        maximum_absolute_foot_gap_m=max(abs(r['gun_foot_gap_m'])for r in tail),
        minimum_support_margin_m=min(r['gun_support_margin_m']for r in tail),
        minimum_support_force_N=min(r['gun_stand_vertical_force_N']for r in tail),
        minimum_mean_support_force_N=float(means.min()),support_force_window_s=RETURN['support_force_window_s'],
        maximum_support_force_N=max(r['gun_stand_normal_force_N']for r in tail),
        maximum_finger_force_N=max(max(r['gun_grip_normal_force_N'])for r in tail),
        air_off=all(not r['air_on']for r in tail))
    poses=[_pose(r['gun_pose_world'],'gun')for r in tail]
    metrics['maximum_position_excursion_m']=max(float(np.linalg.norm(p[:3,3]-poses[0][:3,3]))for p in poses)
    metrics['maximum_orientation_excursion_deg']=max(math.degrees(_angle(p[:3,:3],poses[0][:3,:3]))for p in poses)
    metrics['velocity_fields_diagnostic_only']=True
    checks=[metrics[k]<=RETURN[k]for k in ('maximum_position_error_m','maximum_rotation_error_deg',
        'maximum_position_excursion_m','maximum_orientation_excursion_deg','maximum_absolute_foot_gap_m','maximum_support_force_N')]
    checks.extend([metrics['minimum_support_margin_m']>=RETURN['minimum_support_margin_m'],
                   metrics['minimum_mean_support_force_N']>=fraction*weight,metrics['air_off']])
    if free:checks.append(metrics['maximum_finger_force_N']<=RETURN['maximum_released_finger_force_N'])
    return dict(passed=all(checks),complete=True,sample_count=len(tail),required_support_force_N=fraction*weight,**metrics)


def _support_means(rows,dt):
    width=round(RETURN['support_force_window_s']/dt)
    if width<1 or abs(width*dt-RETURN['support_force_window_s'])>1e-9 or len(rows)<width:
        raise ValueError('Missing complete gun support force interval.')
    values=np.asarray([r['gun_stand_vertical_force_N']for r in rows],dtype=float)
    times=np.asarray([r['time_s']for r in rows],dtype=float)
    if not np.isfinite(values).all()or not np.isfinite(times).all()or not np.allclose(np.diff(times),dt,atol=1e-8,rtol=0):
        raise ValueError('Gun support force evidence is nonfinite or discontinuous.')
    sums=np.r_[0.,np.cumsum(values)]
    return (sums[width:]-sums[:-width])/width


def measured_support_force_mean(rows,dt):
    return float(_support_means(rows[-round(RETURN['support_force_window_s']/dt):],dt)[-1])


def gun_seat_increment(metadata,rows,dt):
    force=measured_support_force_mean(rows,dt)
    reference=_pose(metadata['gun_return']['reference_pose_world'],'gun reference')
    current=_pose(rows[-1]['gun_pose_world'],'gun')
    error=reference[:2,3]-current[:2,3]
    xy=RETURN['seat_center_gain']*error if np.linalg.norm(error)>RETURN['seat_center_deadband_m']else np.zeros(2)
    xy*=min(1.,RETURN['seat_center_step_m']/max(float(np.linalg.norm(xy)),1e-15))
    dz=-RETURN['load_step_m']if force<RETURN['load_force_band_N'][0]else RETURN['load_step_m']if force>RETURN['load_force_band_N'][1]else 0.
    return np.r_[xy,dz],force


def release_ready(metadata,rows,dt):
    return _support_check(metadata,rows,dt,('gun_return_seat_hold',))


def retreat_ready(metadata,rows,dt):
    return _support_check(metadata,rows,dt,('gun_return_release_hold',),free=True,duration=RETURN['release_clear_window_s'])


def evaluate_gun_return(metadata,rows):
    dt=float(metadata['physics_dt_s']);stage=[r for r in rows if r.get('phase','').startswith('gun_return_')]
    if not stage:return dict(result='INCOMPLETE',checks={},failed_checks=['missing_gun_return'])
    entry={**stage[0],'phase':'gun_hold'}
    retained=[entry]+[r for r in stage[1:]if r['phase']not in RELEASE_PHASES]
    retention=gun_retention_check(retained,dt,metadata['gun']['grip_local_m'])
    final=_support_check(metadata,stage,dt,('gun_return_free_hold',),free=True)
    contacts=[float(c[2])for r in stage for c in r.get('unexpected_gun_right_contacts',[])]
    seq=[]
    for row in stage:
        if not seq or seq[-1]!=row['phase']:seq.append(row['phase'])
    wanted=['gun_return_start','gun_return_raise','gun_return_transfer','gun_return_upright','gun_return_above',
            'gun_return_align','gun_return_lower','gun_return_seat_hold','gun_return_release','gun_return_release_hold','gun_return_retreat',
            'gun_return_park','gun_return_free_hold']
    seat=next((r for r in reversed(stage)if r['phase']=='gun_return_seat_hold'),None)
    free,complete=_window(stage,('gun_return_free_hold',),dt)
    retreats=[float(np.linalg.norm(np.asarray(r['right_hand_pose_world'][:3])-np.asarray(seat['right_hand_pose_world'][:3])))for r in free]if seat else[]
    retreat=dict(passed=complete and bool(retreats)and min(retreats)>=RETURN['minimum_hand_retreat_m'],
                 minimum_measured_hand_retreat_m=min(retreats,default=0.),required_m=RETURN['minimum_hand_retreat_m'])
    transition=[r for r in stage if r['phase']in RELEASE_PHASES];zero=0.;maximum_zero=0.
    for row in transition:
        zero=zero+dt if row['gun_stand_vertical_force_N']<=.01 else 0.;maximum_zero=max(maximum_zero,zero)
    reference=_pose(metadata['gun_return']['reference_pose_world'],'reference')
    gaps=[abs(r['gun_foot_gap_m'])for r in transition]
    tilts=[math.degrees(_angle(_pose(r['gun_pose_world'],'gun')[:3,:3],reference[:3,:3]))for r in transition]
    margins=[r['gun_support_margin_m']for r in transition]
    transition_check=dict(passed=bool(transition)and max(gaps)<=RETURN['maximum_absolute_foot_gap_m']and
        max(tilts)<=RETURN['maximum_rotation_error_deg']and min(margins)>=RETURN['minimum_support_margin_m'],
        maximum_absolute_foot_gap_m=max(gaps,default=0.),maximum_rotation_error_deg=max(tilts,default=0.),
        minimum_support_margin_m=min(margins,default=0.),maximum_zero_support_duration_s=maximum_zero)
    checks=dict(sequence=dict(passed=seq==wanted,observed=seq),measured_hand_retreat=retreat,supported_release_transition=transition_check,
        retained_before_release=retention,supported_before_release=release_ready(metadata,stage,dt),
        released_before_retreat=retreat_ready(metadata,stage,dt),
        released_and_standing=final,
        air_off=dict(passed=all(not r.get('air_on',True)for r in stage)),
        no_unexpected_contact=dict(passed=max(contacts,default=0.)<=.01,maximum_force_N=max(contacts,default=0.)))
    failures=[k for k,v in checks.items()if not v['passed']]
    return dict(result='GUN_RETURNED_AND_RELEASED_IN_THIS_SIMULATION'if not failures else'FAILED',
                checks=checks,failed_checks=failures,scope='Measured support, release and retreat; no drying physics.')


def run_gun_return(model,data,metadata,rows,replay,sample_callback,command_callback):
    import mujoco
    from wafer_cycle import solve_pose,ik,station,constants
    timing=gun_motion_timing(metadata)
    arm=[model.actuator('right_drive_joint'+str(i)).id for i in range(1,7)]
    fingers=[model.actuator('right_drive_joint'+str(i)).id for i in (7,8)]
    baseline=next((r for r in reversed(rows)if r.get('phase')=='gun_settle'),None)
    if baseline is None:raise ValueError('Missing measured gun-on-stand reference.')
    metadata['gun_return']=dict(profile=dict(RETURN),reference_pose_world=baseline['gun_pose_world'])
    dt=float(model.opt.timestep);substeps=round(.05/dt)
    def measure():return gun_return_measurement(model,data)
    def record(phase):
        extra=measure();extra['phase']=phase
        sample_callback(phase,extra)
        if extra['air_on']:raise RuntimeError('Air remained on during gun return.')
        if abs(extra['trigger_spring_force_N'])>6 or abs(extra['trigger_velocity_m_s'])>1.:
            raise RuntimeError('Passive trigger bound exceeded during return.')
        if any(c[2]>.01 for c in extra['unexpected_gun_right_contacts']):
            raise RuntimeError('Unexpected gun/right-arm contact during return.')
        if phase in RELEASE_PHASES:
            reference=_pose(metadata['gun_return']['reference_pose_world'],'reference')
            current=_pose(extra['gun_pose_world'],'gun')
            if (abs(extra['gun_foot_gap_m'])>RETURN['maximum_absolute_foot_gap_m'] or
                math.degrees(_angle(current[:3,:3],reference[:3,:3]))>RETURN['maximum_rotation_error_deg'] or
                extra['gun_support_margin_m']<RETURN['minimum_support_margin_m']):
                raise RuntimeError('Gun lost stand support after release began.')
        if extra['gun_stand_normal_force_N']>RETURN['maximum_support_force_N']:
            raise RuntimeError('Gun seating force exceeded3N.')
        return extra
    def execute(phase,duration,target,grip,servo=None,seat_load=False):
        start=data.ctrl[arm].copy();finger_start=data.ctrl[fingers].copy();seat_command=None;load_shift=np.zeros(3)
        print(phase,flush=True)
        for i in range(round(duration/dt)):
            if i%substeps==0:
                s=min(1.,(i+substeps)*dt/duration);u=s*s*(3-2*s)
                q=target[i//substeps].copy()if target.ndim==2 else start+(target-start)*u
                feedback=None
                if servo is not None:
                    current=measure()
                    if seat_command is None:
                        g=_pose(current['gun_pose_world'],'gun')
                        q,feedback=servo.step(q,g[:3,3],g[:3,:3],data.ctrl[arm],integrate=True)
                        if phase in ('gun_return_lower','gun_return_seat_hold') and current['gun_stand_vertical_force_N']>=.20*(metadata['gun']['mass_kg']+metadata['gun']['trigger_mass_kg'])*9.81 and abs(current['gun_foot_gap_m'])<=RETURN['maximum_absolute_foot_gap_m']:
                            seat_command=q.copy()
                    else:q=seat_command.copy()
                if seat_load:
                    delta,force=gun_seat_increment(metadata,rows,dt);q=data.ctrl[arm].copy()
                    load_shift+=delta
                    if np.linalg.norm(load_shift)>RETURN['maximum_load_command_shift_m']:raise RuntimeError('Gun load-transfer travel bound exceeded.')
                    p,r=ik.tcp_pose(q,station.BASES['right']);q=solve_pose(p+delta,r,q,'right')
                    feedback=dict(mode='support_load_transfer',measured_support_force_N=force,force_band_N=RETURN['load_force_band_N'],
                                  measurement_window_s=RETURN['support_force_window_s'],command_z_shift_m=float(load_shift[2]),
                                  command_shift_world_m=load_shift.tolist(),maximum_command_shift_m=RETURN['maximum_load_command_shift_m'])
                if np.max(np.abs(q-data.ctrl[arm]))>math.radians(1):raise RuntimeError('Gun return command jump exceeded1degree.')
                data.ctrl[arm]=q;data.ctrl[fingers]=finger_start+(grip-finger_start)*u
                command_callback(dict(time_s=float(data.time),phase=phase,right_arm_q_rad=q.tolist(),
                    right_finger_q_m=data.ctrl[fingers].tolist(),gun_return_feedback=feedback))
            mujoco.mj_step(model,data);mujoco.mj_forward(model,data);record(phase)
            if (i+1)%substeps==0:replay.append((float(data.time),data.qpos.copy(),data.qvel.copy(),data.ctrl.copy(),phase))
    execute('gun_return_start',timing['return_start_s']if timing else 1.,data.ctrl[arm].copy(),RETURN['close_target_m'])
    controller_state={};path_servo=None
    commanded=None
    if timing:
        p,r=ik.tcp_pose(data.ctrl[arm],station.BASES['right']);commanded=np.eye(4);commanded[:3,3]=p;commanded[:3,:3]=r
    for phase,duration,target,grip in plan_gun_return(metadata,measure,data.ctrl[arm].copy(),solve_pose,ik.slerp_matrix,constants.TCP_OFFSET,controller_state,commanded):
        if phase in ('gun_return_align','gun_return_lower') and path_servo is None:
            path_relation=controller_state['gun_to_tcp'];path_inverse=np.linalg.inv(path_relation)
            def path_fk(q):
                p,r=ik.tcp_pose(q,station.BASES['right']);return p+r@path_inverse[:3,3],r@path_inverse[:3,:3]
            def path_solve(p,r,q):
                return ik.solve(p+r@path_relation[:3,3],ik.matrix_to_quat_xyzw(r@path_relation[:3,:3]),q,
                                base_pos=station.BASES['right'],iters=100,damping=.012,tol=1e-7)
            path_servo=GunFeedback(path_fk,path_solve,ik.LOWER,ik.UPPER)
        execute(phase,duration,target,grip,servo=path_servo)
    execute('gun_return_seat_hold',timing['seat_hold_s']if timing else 8.,data.ctrl[arm].copy(),RETURN['seat_target_m'],seat_load=True)
    ready=release_ready(metadata,rows,dt);metadata['gun_return']['release_gate']=ready
    if not ready['passed']:raise RuntimeError('Gun is not stably supported; release was not commanded: '+str(ready))
    held=data.ctrl[arm].copy()
    for phase,duration,target,grip in plan_gun_release(held,solve_pose,lambda q:ik.tcp_pose(q,station.BASES['right']),metadata):
        execute(phase,duration,target,grip)
        if phase=='gun_return_release_hold':
            released=retreat_ready(metadata,rows,dt)
            metadata['gun_return']['retreat_gate']=released
            if not released['passed']:raise RuntimeError('Gun release is not clear and supported; retreat was not commanded: '+str(released))
    return evaluate_gun_return(metadata,rows)


def development_return(source,destination):
    import gzip,json,time
    from pathlib import Path
    import mujoco
    import wafer_cycle
    source=Path(source).resolve();destination=Path(destination).resolve();destination.mkdir(parents=True,exist_ok=False)
    model=mujoco.MjModel.from_xml_path(str(source/'scene.xml'));data=mujoco.MjData(model)
    metadata=json.loads((source/'metadata.json').read_text());states=np.load(source/'replay_states.npz')
    def restore(index):
        data.qpos[:]=states['qpos'][index];data.qvel[:]=states['qvel'][index]
        data.ctrl[:]=states['ctrl'][index];data.time=float(states['time_s'][index]);mujoco.mj_forward(model,data)
    restore(np.where(states['phase']=='gun_settle')[0][-1]);baseline=gun_return_measurement(model,data);baseline['phase']='gun_settle'
    restore(-1);rows=[baseline];replay=[];commands=[];started=time.monotonic();error=None
    with gzip.open(destination/'samples.jsonl.gz','wt',compresslevel=1)as stream:
        def sample(phase,extra):
            row={**extra,'phase':phase};rows.append(row);stream.write(json.dumps(row,separators=(',',':'))+'\n');return row
        try:report=run_gun_return(model,data,metadata,rows,replay,sample,commands.append)
        except Exception as exc:
            error=type(exc).__name__+': '+str(exc);report=evaluate_gun_return(metadata,rows)
            physical=any(text in str(exc)for text in ('Air remained on','Passive trigger bound','Unexpected gun/right-arm contact',
                'Gun lost stand support','Gun seating force','Gun is not stably supported','Gun release is not clear and supported','Gun load-transfer travel bound'))
            report.update(result='FAILED'if physical else'INCOMPLETE',error=error,failure_kind='physical_guard'if physical else'implementation')
    report.update(development_only=True,source=str(source),wall_seconds=time.monotonic()-started,
                  gun_return_metadata=metadata.get('gun_return'),sample_count=len(rows)-1)
    (destination/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    (destination/'commands.jsonl').write_text(''.join(json.dumps(r,separators=(',',':'))+'\n'for r in commands))
    if replay:
        np.savez_compressed(destination/'replay_states.npz',time_s=np.array([r[0]for r in replay]),
            qpos=np.stack([r[1]for r in replay]),qvel=np.stack([r[2]for r in replay]),
            ctrl=np.stack([r[3]for r in replay]),phase=np.array([r[4]for r in replay]))
    print(json.dumps(report,indent=2),flush=True)
    return report


if __name__=='__main__':
    import argparse,sys
    parser=argparse.ArgumentParser();parser.add_argument('--development-from',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();result=development_return(args.development_from,args.output)
    sys.exit(0 if result['result']=='GUN_RETURNED_AND_RELEASED_IN_THIS_SIMULATION'else 2)
