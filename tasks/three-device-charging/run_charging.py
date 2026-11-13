"""Explicitly separated physical component probes and full scripted trials."""
import argparse,fcntl,json,subprocess,traceback
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--mode',choices=['aligned','offset','reversed','power-off','magnets-off','demo','pickup','watch-only','case-only'],default='aligned')
    p.add_argument('--steps',type=int,default=160);p.add_argument('--no-video',action='store_true');p.add_argument('--share-gpu',action='store_true');p.add_argument('--support-stand',action='store_true')
    a=p.parse_args();lock=open('/tmp/robocurve-shared-gpu.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    free=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True)
    if min(map(float,free.splitlines()))<16000:raise RuntimeError('Insufficient free VRAM; other jobs untouched')
    from charging_world import ChargingWorld
    world=None
    try:
        world=ChargingWorld(Path(__file__).resolve().parent,a.output,probe=None if a.mode in ('demo','pickup','watch-only','case-only') else a.mode,power=a.mode!='power-off',magnets=a.mode!='magnets-off',record=not a.no_video)
        if a.mode in ('demo','pickup','watch-only','case-only'):
            from scripted_charging import demonstrate
            demonstrate(world,pickup_only=a.mode=='pickup',only={'watch-only':'apple_watch','case-only':'airpods_pro'}.get(a.mode),support_stand=a.support_stand)
        else:
            action=world.joints()
            for i in range(a.steps):
                world.step(action,a.mode)
                if i%40==0:print(json.dumps(world.score()),flush=True)
            world.snapshot('final')
        print(json.dumps(world.score()),flush=True)
    except Exception as exc:
        if world:
            (world.output/'error.json').write_text(json.dumps({'error':str(exc),'traceback':traceback.format_exc()},indent=2));world.snapshot('failure')
        raise
    finally:
        if world:world.close()

if __name__=='__main__':main()
