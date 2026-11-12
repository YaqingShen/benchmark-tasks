"""Unclipped passive hinge torque guard; no physics steps or state changes."""
import copy
import math
from wafer_grasp_settings import HINGE_CLOSED_LIMIT_RAD

HINGE_GUARD_PROFILE=dict(revision='passive_hinge_torque_045',
    maximum_absolute_unconstrained_torque_Nm=.3,comparison='strictly_greater_than',
    checked_at='Every saved physics sample, using actual joint velocity',
    torque_definition='-0.45*(hinge_angle_rad+0.2)-0.2*hinge_velocity_rad_s',
    scope='Prospective mechanical-damper diagnostic; unchanged0.3Nm native cap, not calibrated hardware torque or a relaxed wafer criterion.')


class HingeComplianceStop(RuntimeError):
    """A persisted measured hinge state exceeded its unchanged native cap."""


def hinge_guard_metadata():return copy.deepcopy(HINGE_GUARD_PROFILE)


def _number(value,name):
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):
        raise ValueError('Invalid finite hinge field: '+name)
    return float(value)


def _parameters(hinge):
    if not isinstance(hinge,dict) or hinge.get('joint_name')!='pliers_hinge':
        raise ValueError('Missing reviewed passive hinge profile.')
    expected=dict(stiffness_Nm_rad=.45,damping_Nms_rad=.2,rest_rad=-.2,
        force_cap_Nm=.3,torque_guard_Nm=.3,velocity_limit_rad_s=10.)
    if any(_number(hinge.get(k),k)!=v for k,v in expected.items()):
        raise ValueError('Reviewed hinge SI spring/damper/native-cap profile differs.')
    if hinge.get('limits_rad')!=[-.04,HINGE_CLOSED_LIMIT_RAD]:raise ValueError('Reviewed hinge position limits differ.')
    return expected


def _state(row):
    q=_number(row.get('hinge_angle_rad'),'angle')
    velocity=_number(row.get('hinge_velocity_rad_s'),'actual joint speed')
    recorded=_number(row.get('passive_hinge_torque_Nm'),'unconstrained torque')
    raw=-.45*(q+.2)-.2*velocity
    if not math.isclose(raw,recorded,rel_tol=1e-6,abs_tol=1e-8):
        raise ValueError('Saved hinge torque does not match its unconstrained SI law.')
    return q,velocity,raw,abs(recorded-raw),abs(raw)>.3


def guard_passive_hinge_state(row,passive_hinge):
    """Call only after saving the row. Never clip the measured torque or speed."""
    _parameters(passive_hinge)
    q,v,raw,error,exceeded=_state(row)
    if exceeded:
        raise HingeComplianceStop('Passive hinge torque exceeded0.3Nm at '+str(row.get('time_s'))+
            ' s ('+str(row.get('phase'))+'): q='+str(q)+' rad, qdot='+str(v)+' rad/s, raw='+str(raw)+' Nm.')


def evaluate_passive_hinge_state(rows,passive_hinge):
    """Independent saved-measurement evaluation, including partial failed runs."""
    _parameters(passive_hinge)
    if not rows:raise ValueError('No saved actual hinge measurements.')
    peak=0.;speed=0.;error=0.;first=None
    for row in rows:
        q,v,raw,e,exceeded=_state(row);peak=max(peak,abs(raw));speed=max(speed,abs(v));error=max(error,e)
        if first is None and exceeded:
            first=dict(time_s=row.get('time_s'),phase=row.get('phase'),angle_rad=q,
                velocity_rad_s=v,unconstrained_torque_Nm=raw)
    return dict(passed=first is None,status='WITHIN_DECLARED_HINGE_TORQUE'if first is None else'DECLARED_HINGE_TORQUE_EXCEEDED',
        sample_count=len(rows),peak_absolute_torque_Nm=peak,peak_absolute_velocity_rad_s=speed,
        maximum_torque_consistency_error_Nm=error,first_violation=first,profile=hinge_guard_metadata())
