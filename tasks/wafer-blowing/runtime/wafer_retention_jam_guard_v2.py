"""Prospective v2 finger/tool jam diagnostic, independent of physics engines.

This 20 N stop is not a hardware rating, a wafer-stress limit, or a replacement
for the existing wafer criteria and actuator effort caps. Historical results
are never reclassified. The local and cloud force measurements differ.
"""
import copy
import math

MAXIMUM_PER_FINGER_N=20.0
GUARD_PROFILE=dict(revision='retention_v2_finger_load_diagnostic_v1',
    maximum_per_finger_N=MAXIMUM_PER_FINGER_N,comparison='strictly_greater_than',
    checked_at='Every physics sample after the violating measurement has been saved',
    consequence='Stop the new v2 trial immediately and retain FAILED evidence',
    scope='Prospective simulation jam diagnostic only; not a hardware load rating or change to wafer criteria.')
MEASUREMENTS={
    'local':dict(field='finger_handle_contact_normal_force_N',
        scope='Per-finger sum of individual intended-handle/rib/guide normal-force magnitudes.'),
    'isaac':dict(field='finger_handle_partner_normal_force_N',
        scope='Per-finger norm of the normal-force vector filtered to its intended lever link. '
              'Opposing contacts may cancel; not a per-contact magnitude sum or exact handle-surface force.')}


class FingerHandleJamStop(RuntimeError):
    """A saved finite force exceeded the declared prospective diagnostic bound."""


def guard_metadata(engine):
    if engine not in MEASUREMENTS:raise ValueError('Unknown jam-guard measurement engine.')
    return dict(profile=copy.deepcopy(GUARD_PROFILE),measurement_field=MEASUREMENTS[engine]['field'],
                measurement_scope=MEASUREMENTS[engine]['scope'])


def _forces(row,engine):
    if engine not in MEASUREMENTS:raise ValueError('Unknown jam-guard measurement engine.')
    value=row.get(MEASUREMENTS[engine]['field'])
    if not isinstance(value,(list,tuple)) or len(value)!=2:
        raise ValueError('Missing two measured finger/handle loads.')
    if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or v<0 for v in value):
        raise ValueError('Finger/handle load must be a finite nonnegative number in newtons.')
    return [float(v) for v in value]


def guard_finger_handle_load(row,engine='local'):
    """Call only after persisting this row; never modifies a measurement."""
    forces=_forces(row,engine)
    if max(forces)>MAXIMUM_PER_FINGER_N:
        raise FingerHandleJamStop('Declared v2 finger/handle jam diagnostic exceeded 20 N at '
            +str(row.get('time_s'))+' s ('+str(row.get('phase'))+'): '+str(forces)+' N. '
            'This is a simulation diagnostic stop, not a hardware rating.')


def evaluate_finger_handle_load(rows,engine='local'):
    """Independently reevaluate saved evidence without replacing other checks."""
    if not rows:raise ValueError('No saved force measurements for the jam diagnostic.')
    peak=[0.,0.];first=None
    for row in rows:
        forces=_forces(row,engine)
        peak=[max(a,b) for a,b in zip(peak,forces)]
        if first is None and max(forces)>MAXIMUM_PER_FINGER_N:
            first=dict(time_s=row.get('time_s'),phase=row.get('phase'),forces_N=forces)
    return dict(passed=first is None,
        status='WITHIN_DECLARED_DIAGNOSTIC_BOUND' if first is None else 'DECLARED_DIAGNOSTIC_BOUND_EXCEEDED',
        sample_count=len(rows),maximum_per_finger_N=MAXIMUM_PER_FINGER_N,
        peak_per_finger_N=peak,first_violation=first,**guard_metadata(engine))
