"""Verify the native Inspect evaluation before accepting a recording."""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

def inspect_run(directory, required_policy, minimum=None):
    files=[p for p in directory.glob('adhoc_*.json') if not p.name.endswith('.live.json')]
    if len(files)!=1:
        raise ValueError(f'{directory} needs one completed log')
    log=json.loads(files[0].read_text())
    if log.get('error') or log['status']!='success' or log['results']['errored_trials'] or log['results']['total_trials']!=1:
        raise ValueError(f'{directory} evaluation failed or was incomplete')
    info=log['eval']
    if info['policy']!=required_policy or info['embodiment']!='yam_arena' or info['embodiment_info']['control_hz']!=20:
        raise ValueError('Wrong policy or embodiment')
    if required_policy=='dream' and 'dream-1' not in str(info.get('policy_checkpoint','')).lower():
        raise ValueError('The run did not identify a Dream-1 checkpoint')
    score=log['results']['metrics']['task_score']
    if not math.isfinite(score) or not 0<=score<=1 or (minimum is not None and score<minimum):
        raise ValueError(f'{directory} has task score {score}')
    if not 0<log['stats']['total_steps']<=info['max_steps']:
        raise ValueError('Invalid action count')
    hashes={}
    for line in (directory/'inputs.sha256').read_text().splitlines():
        digest,name=line.split(maxsplit=1)
        path=directory.parent.parent/name
        if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:raise ValueError(f'{name} changed after the run')
        hashes[name]=digest
    return files[0],log,hashes

if __name__=='__main__':
    args=argparse.ArgumentParser()
    args.add_argument('directory',type=Path)
    args.add_argument('--dream',type=Path)
    opts=args.parse_args()
    f,d,h=inspect_run(opts.directory,'yam_scripted',1.)
    if opts.dream:
        model,m,mh=inspect_run(opts.dream,'dream')
        if h!=mh or any(m['eval'][k]!=d['eval'][k] for k in ('seed','max_steps','embodiment')):
            sys.exit('Model and reference settings differ')
        if d['samples'][0]['instruction']!=m['samples'][0]['instruction']:
            sys.exit('Instruction mismatch')
        print(json.dumps({'demo':str(f),'dream':str(model),'demo_score':1.,'dream_score':m['results']['metrics']['task_score']},indent=2))
    else:print(str(f))
