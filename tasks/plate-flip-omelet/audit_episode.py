"""Reconcile the terminal physics trace with the official Inspect Robots result."""
import argparse
import json
from pathlib import Path

def audit(folder):
    logs=list(folder.glob('adhoc_????????.json'))
    if len(logs)!=1: raise ValueError('Expected one completed Inspect run')
    log=json.loads(logs[0].read_text());stats=log['stats'];ev=log['eval']
    rows=[json.loads(line) for line in (folder/'trace.jsonl').read_text().splitlines()]
    if log.get('error') or log['results']['errored_trials'] or len(rows)!=stats['total_steps'] or any(r['step']!=i for i,r in enumerate(rows)):
        raise ValueError('Incomplete official episode or unsynchronized trace')
    last=rows[-1]
    score=0. if last['failed'] else [0,.15,.35,.60,.75,.85,1.][last['stage']]
    if abs(score-log['results']['metrics']['task_score'])>1e-6: raise ValueError('Terminal score mismatch')
    dt=last['physics_dt_s']; hz=ev['embodiment_info']['control_hz']; expected=round(1/(dt*hz))
    if any(row['physics_samples']!=i*expected for i,row in enumerate(rows)):
        raise ValueError('Missing physics substeps after the reset observation')
    for row in rows:
        if row['stage']==6 and (row['hold']<20 or not row['released'] or not row['seated'] or not row['settled_before_hand_check'] or row['failed']):
            raise ValueError('Invalid full-credit frame')
    milestones=[{'step':r['step'],'stage':r['stage']} for i,r in enumerate(rows) if not i or r['stage']!=rows[i-1]['stage']]
    phases={}
    for j,key in enumerate(('skillet','plate')):
        phases[key]=[]
        for finger in (0,1):
            r=max(rows,key=lambda v:v['window_peak_N'][j][finger])
            phases[key].append({'N':r['window_peak_N'][j][finger], 'window_end_step':r['step']})
    return dict(log=logs[0].name,policy=ev['policy'],checkpoint=ev['policy_checkpoint'],seed=ev['seed'],official_score=score,
                actions=len(rows),action_limit=ev['max_steps'],control_hz=hz,physics_hz=1/dt,
                measured_substeps=last['physics_samples'],duration_wall_s=stats['duration_s'],milestones=milestones,
                first_full_action=next((r['step']+1 for r in rows if r['stage']==6 and not r['failed']),None),
                terminal=last,peak_per_pad=phases,max_skillet_tilt_deg=max(v['pan_deg'] for v in rows))
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('folder',type=Path)
    print(json.dumps(audit(p.parse_args().folder),indent=2))
