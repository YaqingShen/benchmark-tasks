# Development-only contact mapping for the declared physical registration rails.
# Frozen predecessor SHA256: c377a30e6de10858b3d7f855fa774478256e6e50ef9ff89250dd0549dcad7f51
"""Independent saved-trace checks for actual-YAM wafer trials; no engine import."""
import math
import re
from wafer_grasp_protocol import evaluate_wafer_grasp_trial


CONTACT_ALLOWANCES = {
    'left': ['upper_robot_handle','upper_handle_rib_0','upper_handle_rib_1','upper_handle_side_rail_0','upper_handle_side_rail_1','upper_handle_endstop','upper_handle_axial_stop','upper_handle_opposing_stop'],
    'right': ['lower_robot_handle','lower_handle_rib_0','lower_handle_rib_1','lower_handle_side_rail_0','lower_handle_side_rail_1','lower_handle_endstop','lower_handle_axial_stop'],
}


def classify_robot_contact(robot_geom, partner_geom):
    """Only the actual finger meshes and these explicitly named handle, rail and registration-stop surfaces are permitted.

    There is no robot-base/table exception. Robot self-collision is outside this
    external-contact trace and requires the separately reported geometry screen.
    """
    if not isinstance(robot_geom,str) or not robot_geom.startswith('robot_'):
        raise ValueError('Contact partner is not a named robot geometry.')
    if not isinstance(partner_geom,str) or partner_geom.startswith('robot_'):
        raise ValueError('External-contact trace has an invalid partner.')
    for side, allowed in CONTACT_ALLOWANCES.items():
        if re.fullmatch('robot_tip_'+side+r'_\d+',robot_geom) and partner_geom in allowed:
            return side+'_handle'
    return 'unexpected_tool' if partner_geom.startswith(('lower_','upper_','pliers_')) else 'unexpected_environment'


def _force(value):
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value < 0:
        raise ValueError('Contact force must be a finite nonnegative number.')
    return float(value)


def evaluate_yam_trial(metadata,rows,limits=None,require_return=False):
    """Reevaluate the generic wafer protocol plus saved robot-partner contacts.

    Every row must contain robot_contact_partners_N, a map from exact string
    "robot_geom -> partner_geom" to summed nonnegative normal force. Also
    required: finger_handle_contact_normal_force_N[left,right],
    unexpected_robot_tool_contact_normal_force_N, and
    unexpected_robot_environment_contact_normal_force_N. Aggregates are checked
    against partner records; missing fields never pass. No thresholds are added:
    the generic protocol's maximum_unwanted_contact_N bounds their combined sum.
    """
    report=evaluate_wafer_grasp_trial(metadata,rows,limits=limits,require_return=require_return)
    try:
        peak_tool=peak_environment=peak_combined=peak_handle=0.
        if not rows:raise ValueError('No recorded actual-YAM contact rows.')
        for row in rows:
            partners=row['robot_contact_partners_N']
            if not isinstance(partners,dict):raise ValueError('Robot contact partners must be a map.')
            sums={key:0. for key in ('left_handle','right_handle','unexpected_tool','unexpected_environment')}
            for key,value in partners.items():
                if not isinstance(key,str) or key.count(' -> ')!=1:raise ValueError('Invalid robot contact pair name.')
                robot,partner=key.split(' -> ')
                sums[classify_robot_contact(robot,partner)]+=_force(value)
            observed=row['finger_handle_contact_normal_force_N']
            if not isinstance(observed,list) or len(observed)!=2:raise ValueError('Missing left/right handle forces.')
            comparisons=[(observed[0],sums['left_handle']),(observed[1],sums['right_handle']),
                (row['unexpected_robot_tool_contact_normal_force_N'],sums['unexpected_tool']),
                (row['unexpected_robot_environment_contact_normal_force_N'],sums['unexpected_environment'])]
            if any(abs(_force(given)-expected)>1e-9 for given,expected in comparisons):
                raise ValueError('Robot contact aggregate differs from saved partner evidence.')
            peak_tool=max(peak_tool,sums['unexpected_tool'])
            peak_environment=max(peak_environment,sums['unexpected_environment'])
            peak_combined=max(peak_combined,sums['unexpected_tool']+sums['unexpected_environment'])
            peak_handle=max(peak_handle,sums['left_handle'],sums['right_handle'])
        limit=report['limits']['maximum_unwanted_contact_N'];passed=peak_combined<=limit
        report['checks']['robot_obstacle_contacts']=dict(passed=passed,sample_count=len(rows),
            maximum_unwanted_contact_N=limit,maximum_combined_unwanted_contact_N=peak_combined,
            allowed_finger_handle_partners=CONTACT_ALLOWANCES,allowed_base_contacts=[],
            scope='Saved external-contact partner forces at every physics step; self-collision is a separate geometry screen.')
        report['robot_contact_diagnostics']=dict(maximum_unexpected_robot_tool_contact_N=peak_tool,
            maximum_unexpected_robot_environment_contact_N=peak_environment,
            maximum_finger_handle_contact_N=peak_handle)
        if not passed:
            if report['result']!='INCOMPLETE':report['result']='FAILED'
            report['lift_result']='FAILED'
        report['failed_checks']=[key for key,value in report['checks'].items() if not value['passed']]
    except (KeyError,TypeError,ValueError) as exc:
        report['errors'].append('Robot contact evidence: '+str(exc));report['result']='INCOMPLETE'
        report['checks']['robot_obstacle_contacts']=dict(passed=False,reason='Incomplete or inconsistent robot contact evidence.')
        report['failed_checks']=[key for key,value in report['checks'].items() if not value['passed']]
    return report
