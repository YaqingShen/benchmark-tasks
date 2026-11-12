"""V2-only prospective jam check; the frozen wafer/contact criteria are reused."""
from wafer_retention_checks import (
    evaluate_yam_trial as _original_evaluate_yam_trial,
    classify_robot_contact, CONTACT_ALLOWANCES,
)
from wafer_retention_jam_guard_v2 import evaluate_finger_handle_load, guard_metadata


def evaluate_yam_trial(metadata, rows, limits=None, require_return=False):
    report = _original_evaluate_yam_trial(metadata, rows, limits=limits, require_return=require_return)
    try:
        if metadata.get('finger_handle_jam_guard') != guard_metadata('local'):
            raise ValueError('Missing or different prospective finger-load diagnostic profile.')
        check = evaluate_finger_handle_load(rows, engine='local')
        report['checks']['finger_handle_jam_guard'] = check
        if not check['passed']:
            # A recorded diagnostic stop remains FAILED even when the shortened
            # trajectory cannot complete the unchanged wafer hold protocol.
            report['result'] = 'FAILED'
            report['lift_result'] = 'FAILED'
    except (KeyError, TypeError, ValueError) as exc:
        report['result'] = 'INCOMPLETE'
        report['errors'].append('Finger-load diagnostic evidence: ' + str(exc))
        report['checks']['finger_handle_jam_guard'] = dict(
            passed=False, status='INCOMPLETE', reason=str(exc))
    report['failed_checks'] = [key for key, value in report['checks'].items() if not value['passed']]
    return report
