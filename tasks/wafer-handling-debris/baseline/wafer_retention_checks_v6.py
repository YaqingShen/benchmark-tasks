"""V6 passive-pad diagnostics layered over the unchanged wafer and v2 checks."""
from wafer_retention_checks_v2 import evaluate_yam_trial as _v2_evaluate_yam_trial
from wafer_retention_checks import classify_robot_contact, CONTACT_ALLOWANCES
from wafer_passive_pad_guard_v6 import evaluate_passive_pad_state, pad_guard_metadata


def evaluate_yam_trial(metadata,rows,limits=None,require_return=False):
    report=_v2_evaluate_yam_trial(metadata,rows,limits=limits,require_return=require_return)
    try:
        if metadata.get('passive_pad_guard')!=pad_guard_metadata():
            raise ValueError('Missing or different passive-pad diagnostic profile.')
        check=evaluate_passive_pad_state(rows,metadata.get('pad_compliance'))
        report['checks']['passive_pad_motion_guard']=check
        if not check['passed']:
            report['result']=report['lift_result']='FAILED'
    except (KeyError,TypeError,ValueError)as exc:
        report['result']='INCOMPLETE'
        report['errors'].append('Passive-pad diagnostic evidence: '+str(exc))
        report['checks']['passive_pad_motion_guard']=dict(passed=False,status='INCOMPLETE',reason=str(exc))
    report['failed_checks']=[name for name,check in report['checks'].items()if not check['passed']]
    return report
