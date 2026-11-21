"""Run the fixed task from reset with either a scripted policy or Dream-1."""
import argparse, json, os, subprocess
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('--policy',choices=['yam_scripted','dream'],required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--seed',type=int,default=0)
p.add_argument('--preview',action='store_true')
a=p.parse_args();root=Path(__file__).resolve().parent
spec=json.loads((root/'task.json').read_text());out=a.output.resolve()
out.mkdir(parents=True,exist_ok=False)
env=os.environ.copy();env['TASK_TRACE_PATH']=str(out/'trace.jsonl')
env['PYTHONDONTWRITEBYTECODE']='1'
w,h=(320,180) if a.preview else (1280,720)
cmd=['inspect-robots','run','--embodiment','yam_arena','--policy',a.policy,
     '--no-prompt','--grader','none','-E','device=cpu','-E','task='+str(root/'task.py'),
     '-E','object_jitter=0','-E','control_hz=20','-E',f'cam_width={w}','-E',f'cam_height={h}',
     '--instruction',spec['instruction'],'--scorer','task_score','--max-steps','2400',
     '--seed',str(a.seed),'--store-frames','--log-dir',str(out/'log')]
if a.policy=='dream':cmd.extend(['-P','model=dreamlabs/dream-1-beta'])
raise SystemExit(subprocess.call(cmd,cwd=root,env=env))
