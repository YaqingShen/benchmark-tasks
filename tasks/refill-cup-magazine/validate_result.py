"""Check the official trial report and the final physical evidence."""
import json
import sys
from pathlib import Path

def validate(run,need_full):
    reports=[p for p in (run/'capture').glob('adhoc*.json') if not p.name.endswith('.live.json')]
    if len(reports)!=1:
        raise ValueError('No unique completed run report')
    report=json.loads(reports[0].read_text())
    if report.get('status')!='success' or report.get('error'):
        raise ValueError(str(report.get('error') or report.get('status')))
    rows=[json.loads(s) for s in (run/'trial.jsonl').read_text().splitlines()]
    r=rows[-1]
    metrics=report['results']['metrics']
    official=float(metrics['task_score'])
    if abs(r['score']-official)>1e-8:
        raise ValueError(f'Final physical score {r["score"]} differs from official {official}')
    passed=official==1 and r['wrapper_clear'] and r['wrapper_released'] and r['beam'] and r['counts']==[1] and r['retained']==7 and (r.get('takeout_gap_m') or 0)>=.020 and r['stable']>=10 and not r['ever_touched_cup']
    if need_full and not passed:
        raise ValueError(f'No complete demonstration: score {official}, step {r["step"]}')
    return {'report':reports[0].name,'score':official,'logged_step':r['step'],'stable_steps':r['stable'],'landed':r['counts'],'retained':r['retained'],'clear':r['wrapper_clear'],'released':r['wrapper_released'],'direct_contact':r['ever_touched_cup'],'takeout_gap_m':r.get('takeout_gap_m'),'policy':report['eval']['policy'],'checkpoint':report['eval'].get('policy_checkpoint'),'limit':report['eval']['max_steps'],'seed':report['eval']['seed']}
if __name__=='__main__':
    print(json.dumps(validate(Path(sys.argv[1]),'--full' in sys.argv),indent=2))
