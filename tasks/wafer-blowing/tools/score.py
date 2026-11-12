"""Independently score the complete measured evidence from a policy or native run."""
import argparse
import copy
import gzip
import json
from pathlib import Path
import sys

from check_bundle import RUNTIME, check


def native_score(run):
    from run_wafer_cycle_cloud import verify_run
    from wafer_dust import CohesiveDustScore
    from wafer_policy_score import combine_score
    audit = verify_run(run, full_cycle=True)
    if not (audit.get('passed') is True and audit.get('matches_report') is True
            and audit.get('evidence', {}).get('complete') is True):
        return dict(score=0.0, success=False, evidence_complete=False, native_audit=audit)
    metadata = json.loads((run / 'metadata.json').read_text())
    metadata['dust']['return_detection'] = 'measured_pin_recontact_v1'
    particles = CohesiveDustScore(metadata)
    count = 0
    with gzip.open(run / 'samples.jsonl.gz', 'rt') as stream:
        for line in stream:
            row = json.loads(line)
            if abs(row['time_s'] - count * metadata['physics_dt_s']) > 1e-8:
                raise ValueError('Missing or reordered physics samples.')
            particles.update(row)
            count += 1
    if count != audit['physics_samples']:
        raise ValueError('The trace count differs from the complete native audit.')
    handling = dict(audit['task_score'], evidence_complete=audit['evidence']['complete'])
    result = combine_score(handling, particles.result())
    result.update(native_audit=audit, physics_samples=count, trace_sha256=audit['trace_sha256'])
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('kind', choices=('policy', 'native'))
    p.add_argument('run', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--native-audit-output', type=Path)
    args = p.parse_args()
    if args.output.exists():
        p.error('Choose a new audit file; existing evidence is preserved.')
    if args.native_audit_output and (args.kind != 'native' or args.native_audit_output.exists()):
        p.error('--native-audit-output requires native scoring and a new file.')
    check()
    sys.path[:0] = [str(RUNTIME), str(RUNTIME / 'benchmark')]
    try:
        if args.kind == 'native':
            result = native_score(args.run.resolve())
        else:
            from wafer_robot_geometry import verify_benchmark_trial
            audit = verify_benchmark_trial(args.run.resolve())
            result = copy.deepcopy(audit['task_score'])
            result['independent_audit'] = audit
    except Exception as exc:
        result = dict(score=0.0, success=False, evidence_complete=False,
                      error=type(exc).__name__+': '+str(exc))
    with args.output.open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write('\n')
    if args.native_audit_output and 'native_audit' in result:
        with args.native_audit_output.open('x') as f:
            json.dump(result['native_audit'], f, indent=2, allow_nan=False)
            f.write('\n')
    print(json.dumps({k: result.get(k) for k in ('score', 'success', 'evidence_complete', 'error')}))
    return 0 if result.get('evidence_complete') else 2


if __name__ == '__main__':
    raise SystemExit(main())
