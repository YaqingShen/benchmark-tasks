"""Per-step passive-pad checks; original force and speed bounds retained."""
import copy
import math
from wafer_grasp_settings import (PHYSICS_DT_S, CONTROL_HZ, PAD_STIFFNESS_N_M,
    PAD_DAMPING_N_S_M, PAD_MASS_KG, PAD_TRAVEL_M)

PAD_GUARD_PROFILE=dict(revision='passive_pad_diagnostic_v6_1',
    maximum_absolute_speed_m_s=.05,maximum_absolute_unconstrained_spring_force_N=4.,
    comparison='strictly_greater_than',
    checked_at='Every physics sample after the violating measurement has been saved',
    force_definition='-stiffness_N_m*(q_m-rest_m)-damping_N_s_m*qdot_m_s; never a clipped actuator readback',
    scope='Passive-mechanism diagnostic for the shared declared pad profile; native8N cap remains separate from4N diagnostic, not a hardware rating or change to wafer acceptance criteria.')


class PadComplianceStop(RuntimeError):
    """A persisted finite pad state exceeded the declared prospective bound."""


def pad_guard_metadata():
    return copy.deepcopy(PAD_GUARD_PROFILE)


def _number(value,name):
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):
        raise ValueError('Invalid finite passive-pad field: '+name)
    return float(value)


def _parameters(pads):
    if not isinstance(pads,list) or len(pads)!=2:raise ValueError('Two declared passive pads are required.')
    values=[]
    for half,p in zip(('lower','upper'),pads):
        if p.get('joint_name')!=half+'_pad_compression' or p.get('body_name')!=half+'_pad':
            raise ValueError('Passive pads must use the declared lower/upper joint order.')
        k=_number(p.get('stiffness_N_m'),'stiffness');d=_number(p.get('damping_N_s_m'),'damping')
        mass=_number(p.get('mass_kg'),'mass');rest=_number(p.get('rest_m'),'rest')
        bounds=p.get('limits_m')
        if (k!=PAD_STIFFNESS_N_M or mass!=PAD_MASS_KG or d!=PAD_DAMPING_N_S_M or rest!=0. or not isinstance(bounds,list) or len(bounds)!=2
                or [_number(v,'joint limit')for v in bounds]!=[0.,PAD_TRAVEL_M]):
            raise ValueError('The approved passive-pad physical profile differs.')
        speed=_number(p.get('speed_guard_m_s'),'speed guard');cap=_number(p.get('force_cap_N'),'force cap')
        force_guard=_number(p.get('force_guard_N'),'unconstrained force guard')
        native_speed=_number(p.get('velocity_limit_m_s'),'native speed ceiling')
        if speed!=.05 or cap!=8. or force_guard!=4. or native_speed!=1.:
            raise ValueError('The approved pad diagnostic/native ceiling differs.')
        if k*max(abs(v-rest)for v in bounds)+d*speed>=cap:
            raise ValueError('The native force cap can clip the spring within the declared nominal range.')
        values.append((k,d,rest))
    return values


def _state(row,parameters):
    arrays=[]
    for key in ('pad_compression_m','pad_compression_velocity_m_s','passive_pad_spring_force_N'):
        value=row.get(key)
        if not isinstance(value,(list,tuple)) or len(value)!=2:raise ValueError('Missing two '+key+' measurements.')
        arrays.append([_number(v,key)for v in value])
    q,velocity,recorded=arrays
    expected=[-k*(x-rest)-d*v for x,v,(k,d,rest) in zip(q,velocity,parameters)]
    error=max(abs(a-b)for a,b in zip(expected,recorded))
    if any(not math.isclose(a,b,rel_tol=1e-6,abs_tol=1e-7)for a,b in zip(expected,recorded)):
        raise ValueError('Recorded passive-pad force is not the unconstrained SI spring/damper force.')
    exceeded=any(abs(v)>.05 for v in velocity) or any(abs(f)>4. for f in expected)
    return q,velocity,expected,error,exceeded


def guard_passive_pad_state(row,pad_compliance):
    """Call after persisting the row; reads q/qdot and never clamps either."""
    q,velocity,forces,error,exceeded=_state(row,_parameters(pad_compliance))
    if exceeded:
        raise PadComplianceStop('Declared v6 passive-pad diagnostic exceeded at '+str(row.get('time_s'))+
            ' s ('+str(row.get('phase'))+'): qdot='+str(velocity)+' m/s, spring='+str(forces)+' N.')


def evaluate_passive_pad_state(rows,pad_compliance):
    """Reevaluate saved states; this does not claim wafer or mechanism success."""
    parameters=_parameters(pad_compliance)
    if not rows:raise ValueError('No saved passive-pad measurements.')
    speed=[0.,0.];force=[0.,0.];low=[float('inf')]*2;high=[-float('inf')]*2;first=None;error=0.
    for row in rows:
        q,v,f,e,exceeded=_state(row,parameters);error=max(error,e)
        speed=[max(a,abs(b))for a,b in zip(speed,v)];force=[max(a,abs(b))for a,b in zip(force,f)]
        low=[min(a,b)for a,b in zip(low,q)];high=[max(a,b)for a,b in zip(high,q)]
        if first is None and exceeded:
            first=dict(time_s=row.get('time_s'),phase=row.get('phase'),velocity_m_s=v,unconstrained_force_N=f)
    return dict(passed=first is None,status='WITHIN_DECLARED_PAD_BOUNDS' if first is None else 'DECLARED_PAD_BOUNDS_EXCEEDED',
        sample_count=len(rows),peak_absolute_velocity_m_s=speed,peak_absolute_unconstrained_force_N=force,
        observed_compression_range_m=[list(pair)for pair in zip(low,high)],first_violation=first,
        maximum_force_consistency_error_N=error,profile=pad_guard_metadata(),
        range_scope='Observed compression is reported; joint limits are physical constraints, not a silently changed wafer criterion.')
