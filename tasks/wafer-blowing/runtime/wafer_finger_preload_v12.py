"""Measured robot-finger preload; mechanical springs stay passive."""
import copy, hashlib, math
from pathlib import Path
from wafer_grasp_settings import (PHYSICS_DT_S, CONTROL_HZ, PAD_STIFFNESS_N_M,
    PAD_DAMPING_N_S_M, PAD_MASS_KG, PAD_TRAVEL_M)

PHASES=[('settle',2.),('close',6.),('grip_hold',20.),('lift',4.),('lift_hold',5.)]
FULL_WORKFLOW_TIMING_PROFILE=dict(version='complete_workflow_120s_v4',maximum_simulation_seconds=120.,control_hz=20,
    wafer_prefix_phases=[['settle',1.5],['close',4.5],['grip_hold',8.],['lift',2.5],['lift_hold',3.5]],
    wafer_return_phases=[['lower',3.],['return_hold',1.5],['release',2.],['retreat',1.],['retreat',1.5],['unassisted_hold',1.5]],
    stage_budgets_s=dict(tool_pickup=21.5,wafer_prefix=20.,wafer_return=10.5,right_hand=68.,reserve=0.),
    acceptance_window_s=1.,scope='Full workflow only; unchanged measured readiness and physical limits.')


def workflow_timing(metadata):
    profile=metadata.get('full_workflow_timing_profile')
    if profile is not None and profile!=FULL_WORKFLOW_TIMING_PROFILE:
        raise ValueError('Full workflow timing profile differs from the declared 120 s schedule.')
    return profile


def wafer_prefix_phases(metadata):
    profile=workflow_timing(metadata)
    return [tuple(p)for p in profile['wafer_prefix_phases']]if profile else list(PHASES)


def wafer_return_phases(metadata):
    profile=workflow_timing(metadata)
    return [tuple(p)for p in profile['wafer_return_phases']]if profile else [
        ('lower',5.),('return_hold',3.),('release',4.),('retreat',5.),('retreat',4.),('unassisted_hold',3.)]

PROFILE=dict(revision='shared_compliant_pad_preload',command_hz=CONTROL_HZ,physics_dt_s=PHYSICS_DT_S,
    pad_stiffness_N_m=PAD_STIFFNESS_N_M,pad_damping_N_s_m=PAD_DAMPING_N_S_M,target_mean_static_preload_N=1.3,preload_deadband_N=.10,
    maximum_pad_static_preload_opening_priority_N=2.5,
    maximum_dynamic_force_opening_priority_N=3.,maximum_dynamic_force_closing_inhibit_N=2.5,
    gain_m_per_N_per_command=.000015,precontact_maximum_closing_step_m=.000025,
    contacted_maximum_absolute_step_m=.000010,
    bilateral_mean_force_for_fine_closing_N=.05,fine_maximum_closing_step_m=.000005,
    maximum_tracking_lag_for_closing_m=.001,prelift_positive_closing_inhibit_s=2.,
    contact_onset_normal_force_N=.005,contact_onset_compression_m=(.04+PAD_MASS_KG*9.81)/PAD_STIFFNESS_N_M,
    target_lower_bound_m=-.0245,target_upper_bound_m=-.0215,
    finger_order=['tip_left','tip_right'],pad_order=['lower_pad','upper_pad'],
    native_gripper_action_dimension=1,symmetric_targets=True,
    contact_latch='Either pad triggers one retained latch for all subsequent active phases, including contact dropouts.',
    sensor_scope='Exact completed preceding physics interval; mean positive K*q, full-interval static/dynamic peaks, and wafer-to-pad forces. K*q is not contact-force calibration; tracking lag is not measured actuator force.',
    physical_scope='Draft measured-preload controller. No internal jaw drive, attachment, actuator-cap increase or acceptance-threshold change.')

PROTOCOL_SHA256='c7bee231bb98cbc3684efaac7b49cb7f4a6e123cdcf794860cea5f0a7ff13ea2'


class PreliftNotReady(RuntimeError):
    """Readiness was recorded as failed before any lift command was issued."""


def prelift_readiness(metadata,rows):
    """Apply the unchanged measured-grasp criteria before entering the lift.

    Missing measurements are a blocked transition, never inferred contact.
    This gate does not predict a retained lift or replace the final evaluator.
    """
    import numpy as np
    import wafer_grasp_protocol as protocol
    result=dict(passed=False,result='PRELIFT_NOT_READY',checks={},errors=[],
        protocol_sha256=PROTOCOL_SHA256,
        scope='Original bilateral-grasp and finite-pad engagement checks on saved physics measurements before the first lift command; no retained-lift prediction.')
    try:
        if hashlib.sha256(Path(protocol.__file__).read_bytes()).hexdigest()!=PROTOCOL_SHA256:
            raise ValueError('Original wafer acceptance source changed.')
        limits=copy.deepcopy(protocol.DEFAULT_LIMITS)
        result['limits']={k:limits[k] for k in ('settle_window_s','contact_window_s','minimum_jaw_contact_N',
            'minimum_closed_gap_m','maximum_closed_gap_m','minimum_measured_gap_reduction_m','pad_geometry_gap_tolerance_m')}
        dt=protocol._number(metadata['physics_dt_s'],'physics timestep')
        if metadata.get('contact_force_scope')not in ('wafer_partner_filtered','wafer_link_filtered'):
            raise ValueError('Readiness requires partner-filtered wafer force measurements.')
        if not 0<dt<=1/240+1e-12 or not isinstance(rows,(list,tuple)) or len(rows)<2:
            raise ValueError('Missing complete physics trace for prelift readiness.')
        times=np.array([protocol._number(r['time_s'],'physics time')for r in rows])
        if not np.allclose(np.diff(times),dt,atol=1e-9,rtol=0):raise ValueError('Readiness trace cadence is incomplete.')
        if rows[-1].get('phase')!='grip_hold':raise ValueError('Readiness must be evaluated at the end of grip_hold.')
        expected_end=sum(duration for phase,duration in wafer_prefix_phases(metadata)[:3])
        if abs(times[-1]-expected_end)>1e-8:raise ValueError('Readiness is not at the declared prelift transition time.')
        settled=protocol._window(rows,'settle',limits['settle_window_s'],dt)
        grasp=protocol._window(rows,'grip_hold',limits['contact_window_s'],dt)
        if settled is None or grasp is None:raise ValueError('Missing complete settled/open or grip-hold window.')
        open_gap=float(np.median([protocol._number(r['measured_jaw_gap_m'],'open gap')for r in settled]))
        closed=np.array([protocol._number(r['measured_jaw_gap_m'],'closed gap')for r in grasp[1:]])
        forces=np.array([protocol._array(r['jaw_contact_normal_force_N'],(2,),'jaw forces')for r in grasp[1:]])
        if open_gap<0 or np.any(closed<0) or np.any(forces<0):raise ValueError('Negative gap/contact-force measurement.')
        result.update(transition_time_s=float(times[-1]),physics_dt_s=dt,
            evaluated_sample_count=len(grasp)-1,window_start_s=float(grasp[0]['time_s']),window_end_s=float(grasp[-1]['time_s']))
        result['checks']['moving_jaw_bilateral_grasp']=dict(
            passed=bool(open_gap-float(closed.max())>=limits['minimum_measured_gap_reduction_m']
                and float(closed.min())>=limits['minimum_closed_gap_m']
                and float(closed.max())<=limits['maximum_closed_gap_m']
                and float(forces.min())>=limits['minimum_jaw_contact_N']),
            open_gap_m=open_gap,closed_gap_range_m=[float(closed.min()),float(closed.max())],
            minimum_contact_force_by_pad_N=forces.min(axis=0).tolist(),
            minimum_measured_gap_reduction_m=open_gap-float(closed.max()))
        if 'contact_faces_local_m' not in metadata:
            raise ValueError('The passive-pad mechanism requires measured finite pad-face geometry.')
        engaged=[protocol.pad_contact_geometry(r,metadata,limits['pad_geometry_gap_tolerance_m'])for r in grasp[1:]]
        gaps=[p['signed_gap_m']for e in engaged for p in e['pads']if p['signed_gap_m']is not None]
        result['checks']['measured_pad_face_engagement']=dict(passed=all(e['both_engaged']for e in engaged),
            evaluated_sample_count=len(engaged),signed_gap_range_m=[min(gaps),max(gaps)]if gaps else None)
        result['passed']=all(c['passed']for c in result['checks'].values())
        if result['passed']:result['result']='READY_FOR_LIFT'
    except (ValueError,KeyError,TypeError,IndexError,OSError) as exc:
        result['errors'].append(type(exc).__name__+': '+str(exc))
    return result


def _pair(value,name,nonnegative=False):
    if not isinstance(value,(list,tuple)) or len(value)!=2:raise ValueError('Expected two '+name+' values.')
    values=[]
    for v in value:
        if isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v):raise ValueError('Invalid '+name)
        if nonnegative and v<0:raise ValueError('Negative '+name)
        values.append(float(v))
    return values


def preceding_interval_features(rows,control_index,physics_dt_s,*,prefix_length=None):
    """Read the exact previous completed interval, never a padded/future window."""
    import numpy as np
    if type(control_index)is not int or control_index<0:raise ValueError('Invalid control index.')
    if isinstance(physics_dt_s,bool) or not isinstance(physics_dt_s,(int,float)) or not math.isfinite(physics_dt_s) or abs(physics_dt_s-PROFILE['physics_dt_s'])>1e-15:
        raise ValueError('Invalid physics timestep.')
    count=round(.05/physics_dt_s)
    if count<1 or abs(count*physics_dt_s-.05)>1e-12:raise ValueError('Physics interval is not exactly20Hz.')
    end=control_index*count;start=0 if control_index==0 else end-count+1
    if not isinstance(rows,(list,tuple)):raise ValueError('Expected the saved physics sequence.')
    available=len(rows) if prefix_length is None else prefix_length
    if type(available)is not int or available!=end+1 or available>len(rows):
        raise ValueError('Stale, incomplete, or future physics trace at command boundary.')
    # Audits may select a validated prefix of a complete trace. Only the exact
    # preceding interval is copied, never the entire growing history.
    selected=rows[start:end+1]
    times=np.array([r['time_s']for r in selected])
    if times.dtype.kind not in 'fiu' or not np.isfinite(times).all() or not np.allclose(times,np.arange(start,end+1)*physics_dt_s,atol=1e-8,rtol=0):
        raise ValueError('Source interval timestamps differ from exact recorded physics indices.')
    q=np.array([_pair(r['pad_compression_m'],'pad compression')for r in selected])
    v=np.array([_pair(r['pad_compression_velocity_m_s'],'pad speed')for r in selected])
    forces=np.array([_pair(r['jaw_contact_normal_force_N'],'wafer/pad normal force',True)for r in selected])
    recorded=np.array([_pair(r['passive_pad_spring_force_N'],'unconstrained passive force')for r in selected])
    expected=-PROFILE['pad_stiffness_N_m']*q-PROFILE['pad_damping_N_s_m']*v
    if not np.allclose(recorded,expected,atol=1e-7,rtol=1e-6):raise ValueError('Saved passive-force values do not match the unchanged SI spring law.')
    static=PROFILE['pad_stiffness_N_m']*np.maximum(q,0.)
    last=selected[-1]
    return dict(control_index=control_index,time_s=control_index/20.,physics_dt_s=physics_dt_s,
        mode='INITIAL_SNAPSHOT'if control_index==0 else'PRECEDING_COMPLETE_INTERVAL',
        source_start_index=start,source_end_index=end,source_sample_count=len(selected),
        source_first_time_s=float(times[0]),source_last_time_s=float(times[-1]),
        interval_start_s=max(0.,(control_index-1)/20.),interval_end_s=control_index/20.,
        mean_positive_static_preload_by_pad_N=static.mean(0).tolist(),
        maximum_positive_static_preload_by_pad_N=static.max(0).tolist(),
        maximum_absolute_spring_damper_force_by_pad_N=np.abs(expected).max(0).tolist(),
        mean_jaw_contact_normal_force_by_pad_N=forces.mean(0).tolist(),
        maximum_jaw_contact_normal_force_by_pad_N=forces.max(0).tolist(),
        minimum_jaw_contact_normal_force_by_pad_N=forces.min(0).tolist(),
        end_pad_compression_m=q[-1].tolist(),end_pad_compression_velocity_m_s=v[-1].tolist(),
        end_jaw_contact_normal_force_N=forces[-1].tolist(),
        end_measured_finger_q_m=_pair(last['measured_finger_q_m'],'measured finger position'))


def _command_phase(index,phases=PHASES):
    offset=0
    for phase,duration in phases:
        count=round(duration*20)
        if index<offset+count:return phase,(index-offset)/20.,duration
        offset+=count
    raise ValueError('Too many control commands.')


class FingerPreloadController:
    def __init__(self,phases=None):
        self.phases=list(PHASES if phases is None else phases)
        self.contact_seen=False;self.fine_closing_seen=False
        self.previous=[PROFILE['target_lower_bound_m']]*2
        self.command_index=0

    def step(self,phase,interval_features,previous_command_q_m):
        interval=copy.deepcopy(interval_features)
        if not isinstance(interval,dict)or type(interval.get('control_index'))is not int or interval['control_index']!=self.command_index:
            raise ValueError('Interval/control index mismatch.')
        expected_phase,elapsed,duration=_command_phase(self.command_index,self.phases)
        if phase!=expected_phase:raise ValueError('Control phase/cadence differs.')
        if isinstance(interval.get('time_s'),bool)or not isinstance(interval.get('time_s'),(int,float))or not math.isfinite(interval['time_s']) or abs(interval['time_s']-self.command_index/20.)>1e-10:
            raise ValueError('Interval time is stale or future.')
        dt=PROFILE['physics_dt_s'];count=round((1/PROFILE['command_hz'])/dt)
        end=self.command_index*count;start=0 if self.command_index==0 else end-count+1
        for key,expected in [('source_start_index',start),('source_end_index',end),('source_sample_count',1 if self.command_index==0 else count)]:
            if type(interval.get(key))is not int or interval[key]!=expected:raise ValueError('Wrong interval source indices/count.')
        for key,expected in [('physics_dt_s',dt),('source_first_time_s',start*dt),('source_last_time_s',end*dt),
                ('interval_start_s',max(0.,(self.command_index-1)/20.)),('interval_end_s',self.command_index/20.)]:
            v=interval.get(key)
            if isinstance(v,bool)or not isinstance(v,(int,float))or not math.isfinite(v)or abs(v-expected)>1e-8:
                raise ValueError('Wrong interval source timing.')
        expected_mode='INITIAL_SNAPSHOT'if self.command_index==0 else'PRECEDING_COMPLETE_INTERVAL'
        if interval.get('mode')!=expected_mode:raise ValueError('Wrong initial/completed interval mode.')
        mean=_pair(interval['mean_positive_static_preload_by_pad_N'],'mean static preload',True)
        peak=_pair(interval['maximum_positive_static_preload_by_pad_N'],'static peak',True)
        dynamic=_pair(interval['maximum_absolute_spring_damper_force_by_pad_N'],'dynamic peak',True)
        force_mean=_pair(interval['mean_jaw_contact_normal_force_by_pad_N'],'mean contact',True)
        force_peak=_pair(interval['maximum_jaw_contact_normal_force_by_pad_N'],'peak contact',True)
        measured=_pair(interval['end_measured_finger_q_m'],'measured finger position')
        previous=_pair(previous_command_q_m,'previous finger command')
        lo=PROFILE['target_lower_bound_m'];hi=PROFILE['target_upper_bound_m']
        if any(not lo-1e-12<=v<=hi+1e-12 for v in previous)or previous[0]!=previous[1]:raise ValueError('Native scalar gripper command is outside its reviewed bounds.')
        if any(abs(v-w)>1e-12 for v,w in zip(previous,self.previous)):raise ValueError('Previous command differs from controller history.')
        if any(a>b+1e-12 for a,b in zip(mean,peak))or any(a>b+1e-12 for a,b in zip(force_mean,force_peak)):
            raise ValueError('Interval means exceed their peaks.')
        mean_static=sum(mean)/2.;maximum_static=max(peak);maximum_dynamic=max(dynamic)
        error=PROFILE['target_mean_static_preload_N']-mean_static
        lag=previous[0]-min(measured)
        seen=self.contact_seen;fine=self.fine_closing_seen
        final_hold=phase=='grip_hold'and elapsed>=duration-PROFILE['prelift_positive_closing_inhibit_s']
        if phase=='settle':value=lo;decision='SETTLE_OPEN'
        else:
            seen=seen or max(force_peak)>PROFILE['contact_onset_normal_force_N']or maximum_static>PROFILE['contact_onset_compression_m']*PROFILE['pad_stiffness_N_m']
            fine=fine or all(f>PROFILE['bilateral_mean_force_for_fine_closing_N']for f in force_mean)
            limit=PROFILE['contacted_maximum_absolute_step_m']
            if maximum_dynamic>=PROFILE['maximum_dynamic_force_opening_priority_N']or maximum_static>=PROFILE['maximum_pad_static_preload_opening_priority_N']:
                delta=-limit;decision='INTERVAL_PEAK_OPENING_PRIORITY'
            elif not seen:delta=PROFILE['precontact_maximum_closing_step_m'];decision='PRECONTACT_CLOSE'
            elif abs(error)<=PROFILE['preload_deadband_N']:delta=0.;decision='HOLD_PRELOAD_BAND'
            else:
                closing=PROFILE['fine_maximum_closing_step_m']if fine else limit
                delta=max(-limit,min(closing,PROFILE['gain_m_per_N_per_command']*error))
                decision='INCREASE_PRELOAD'if delta>0 else'DECREASE_PRELOAD'
            if delta>0:
                if maximum_dynamic>=PROFILE['maximum_dynamic_force_closing_inhibit_N']:
                    delta=0.;decision='HOLD_DYNAMIC_MARGIN'
                elif lag>=PROFILE['maximum_tracking_lag_for_closing_m']:
                    delta=0.;decision='HOLD_TRACKING_LAG'
                elif final_hold:delta=0.;decision='HOLD_BEFORE_LIFT'
            value=max(lo,min(hi,previous[0]+delta))
            if value==previous[0]and delta!=0:decision='TARGET_BOUND_'+decision
        targets=[value,value]
        diagnostic=dict(control_index=self.command_index,time_s=self.command_index/20.,phase=phase,
            interval=interval,previous_command_q_m=previous,commanded_finger_q_m=targets,
            commanded_common_q_m=value,mean_static_preload_N=mean_static,maximum_static_preload_N=maximum_static,
            maximum_dynamic_force_N=maximum_dynamic,mean_preload_error_N=error,
            tracking_lag_proxy_m=lag,prelift_positive_closing_inhibited=final_hold,
            contact_seen=seen,fine_closing_seen=fine,decision=decision,
            applied_step_m=[a-b for a,b in zip(targets,previous)])
        self.previous=targets.copy();self.contact_seen=seen;self.fine_closing_seen=fine;self.command_index+=1
        return targets.copy(),diagnostic


def replay_commands(records,phases=None):
    """Replay logged features; an independent audit must bind them to raw rows."""
    if not isinstance(records,list)or not records:raise ValueError('Missing preload command records.')
    phases=PHASES if phases is None else phases
    controller=FingerPreloadController(phases);total=round(sum(duration for _,duration in phases)*20)
    if len(records)>total:raise ValueError('Too many preload commands.')
    for index,row in enumerate(records):
        if type(row.get('control_index'))is not int or row['control_index']!=index:raise ValueError('Invalid command index.')
        _,expected=controller.step(row['phase'],row['interval'],row['previous_command_q_m'])
        if any(row.get(key)!=value for key,value in expected.items()):raise ValueError('Preload deterministic replay differs.')
    return dict(status='DETERMINISTIC_PRELOAD_COMMAND_REPLAY_MATCH',command_count=len(records),complete=len(records)==total,
        expected_complete_command_count=total,profile=copy.deepcopy(PROFILE),
        scope='Deterministic interval-feature control replay only; all features must also be bound to the exact preceding saved physics interval.')
