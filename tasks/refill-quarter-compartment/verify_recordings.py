"""Package only matched successful source / demo / model runs from this task."""
import argparse,hashlib,json,shutil,subprocess
from pathlib import Path
ap=argparse.ArgumentParser()
ap.add_argument('root',type=Path);ap.add_argument('demo');ap.add_argument('model');ap.add_argument('out',type=Path)
a=ap.parse_args()
root=a.root.resolve();dest=a.out.resolve()
def record(tag):
    run=root/'runs'/tag
    logs=[p for p in run.glob('adhoc_*.json') if not p.name.endswith('.live.json')]
    if len(logs)!=1:raise ValueError(f'{tag}: incomplete or ambiguous Inspect log')
    d=json.loads(logs[0].read_text())
    if not (d['status']=='success' and not d['error'] and d['results']['errored_trials']==0):raise ValueError(f'{tag}: failed evaluation')
    trace=[json.loads(v) for v in (run/'trace.jsonl').read_text().splitlines() if v]
    meta=d['eval']
    return (d,trace,logs[0],(run/'inputs.sha256').read_text())
D,dt,dl,dh=record(a.demo);M,mt,ml,mh=record(a.model)
if D['eval']['policy'] != 'yam_scripted' or M['eval']['policy'] != 'dream':
    raise ValueError('Expected one scripted run and one Dream run')
if 'dream-1' not in str(M['eval'].get('policy_checkpoint','')).lower():raise ValueError('Unverified model checkpoint')
for key in ('seed', 'max_steps', 'embodiment'):
    if D['eval'][key] != M['eval'][key]:
        raise ValueError(f'Mismatched {key}')
if not dt or not mt:
    raise ValueError('Missing terminal scene observations')
if dh!=mh:raise ValueError('Different source revisions')
for line in dh.splitlines():
    hashval,name=line.split(maxsplit=1)
    if hashlib.sha256((root/name).read_bytes()).hexdigest()!=hashval:raise ValueError(f'Source differs: {name}')
if D['results']['metrics']['task_score']!=1 or dt[-1]['score']!=1 or dt[-1]['delivered']!=40:
    raise ValueError('The reference run did not reach full credit for 40 loose quarters')
if dt[0]['broken'] or dt[0]['contained']!=40 or any(r['escaped_closed'] or r['spilled'] for r in dt):
    raise ValueError('Invalid cold start or spilled change')
if M['eval']['embodiment_info']['control_hz']!=D['eval']['embodiment_info']['control_hz']:raise ValueError('Control frequencies differ')
if any(abs(r['score']-M['results']['metrics']['task_score'])>1e-6 for r in mt[-1:]):raise ValueError('Model score differs from last scene state')
if D['samples'][0]['instruction']!=M['samples'][0]['instruction']:raise ValueError('Instructions differ')
dest.mkdir(parents=True,exist_ok=True)
rows={}
for name,d,log,t in [('demo',D,dl,dt),('dream1',M,ml,mt)]:
    cmd=['python',str(root/'export.py'),str(log),str(dest/(name+'.mp4'))]
    if name=='demo':cmd.append(str(dest/'setup.png'))
    subprocess.run(cmd,check=True)
    rows[name]=dict(score=d['results']['metrics']['task_score'],steps=d['stats']['total_steps'],wall_seconds=d['stats']['duration_s'],
                    terminal=t[-1],instruction=d['samples'][0]['instruction'])
(dest/'verified-runs.json').write_text(json.dumps(rows,indent=2)+'\n')
