"""Run the real Dream-1 SDK policy, or an explicitly named hold-interface check."""
import argparse,fcntl,hashlib,importlib.metadata,json,subprocess
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);p.add_argument('--policy',choices=['dream','hold'],default='dream');p.add_argument('--steps',type=int,default=6000)
    a=p.parse_args();root=Path(__file__).resolve().parent
    if not 1<=a.steps<=12000:p.error('Use an action budget between 1 and 12000')
    lock=open('/tmp/robocurve-shared-gpu.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    free=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True)
    if min(map(float,free.splitlines()))<16000:raise RuntimeError('Insufficient free VRAM; other jobs untouched')
    if a.output.exists():raise ValueError('Preserve previous evidence; use a new output path')
    source=json.loads((root/'source_manifest.json').read_text())
    for n,digest in source['files'].items():
        path=(root/n).resolve()
        if not path.is_relative_to(root) or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:raise ValueError('Frozen source mismatch: '+n)
    a.output.mkdir(parents=True);arena=None;result={'dream1_run':a.policy=='dream','exit_code':1}
    try:
        from charging_benchmark import ChargingArena,ChargingScore,HoldPolicy,INSTRUCTION
        from inspect_robots import eval
        from inspect_robots.scene import Scene
        from inspect_robots.task import Task
        if a.policy=='dream':result['dream_sdk']=importlib.metadata.version('inspect-robots-dream')
        arena=ChargingArena(root,a.output/'evidence',max_steps=a.steps,policy_name=a.policy)
        task=Task(name='three-device-charging',scenes=[Scene(id='three-device-charging',instruction=INSTRUCTION,init_seed=17)],scorer=ChargingScore(),max_steps=a.steps,epochs=1)
        logs=eval(task,'dream' if a.policy=='dream' else HoldPolicy(),arena,log_dir=str(a.output/'inspect'),seed=17,store_frames=True,fail_on_error=True)
        result['statuses']=[log.status for log in logs]
        if not logs or any(log.status=='error' for log in logs):raise RuntimeError('SDK evaluation did not complete')
        result['exit_code']=0
    except BaseException as e:
        result['error_type']=type(e).__name__;raise
    finally:
        try:
            if arena:arena.close()
        finally:
            (a.output/'process-result.json').write_text(json.dumps(result,indent=2))

if __name__=='__main__':main()
