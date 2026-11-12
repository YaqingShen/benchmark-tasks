"""V11 hinge diagnostic layered over unchanged v6/wafer acceptance checks."""
from wafer_retention_checks_v6 import evaluate_yam_trial as _evaluate
from wafer_retention_checks import classify_robot_contact, CONTACT_ALLOWANCES
from wafer_hinge_guard_v11 import evaluate_passive_hinge_state,hinge_guard_metadata


def evaluate_yam_trial(metadata,rows,limits=None,require_return=False):
    report=_evaluate(metadata,rows,limits=limits,require_return=require_return)
    try:
        if metadata.get('passive_hinge_guard')!=hinge_guard_metadata():
            raise ValueError('Missing or different passive hinge diagnostic profile.')
        check=evaluate_passive_hinge_state(rows,metadata.get('passive_hinge'))
        report['checks']['passive_hinge_torque_guard']=check
        if not check['passed']:report['result']=report['lift_result']='FAILED'
    except (KeyError,TypeError,ValueError)as exc:
        report['result']='INCOMPLETE';report['errors'].append('Passive hinge diagnostic evidence: '+str(exc))
        report['checks']['passive_hinge_torque_guard']=dict(passed=False,status='INCOMPLETE',reason=str(exc))
    report['failed_checks']=[name for name,check in report['checks'].items()if not check['passed']]
    return report
