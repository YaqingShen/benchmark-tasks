"""Run the frozen task with either the scripted policy or Dream-1."""
import argparse,json,os,subprocess
from pathlib import Path

ap=argparse.ArgumentParser()
ap.add_argument('--policy',choices=['yam_scripted','dream'],required=True)
ap.add_argument('--output',type=Path,required=True)
ap.add_argument('--preview',action='store_true',help='320x180 cameras; default1280x720')
ap.add_argument('--seed',type=int,default=0)
a=ap.parse_args();root=Path(__file__).resolve().parent
spec=json.loads((root/'task.json').read_text());out=a.output.resolve()
out.mkdir(parents=True,exist_ok=False)
width,height=(320,180) if a.preview else (1280,720)
env=os.environ.copy()
for key in ('WAFFLE_ANIM','WAFFLE_LAYOUT','ENV_STUFF_LAYOUT','ENV_STUFF_RENDER'):
    env.pop(key,None)
env['TASK_TRACE_PATH']=str(out/'trace.jsonl');env['PYTHONDONTWRITEBYTECODE']='1'
cmd=['inspect-robots','run','--embodiment','yam_arena','--policy',a.policy,
     '--no-prompt','--grader','none','-E','device=cpu','-E','task='+str(root/'task.py'),
     '-E','object_jitter=0','-E','control_hz=20','-E',f'cam_width={width}','-E',f'cam_height={height}',
     '--instruction',spec['instruction'],'--scorer','task_score','--max-steps','8500',
     '--seed',str(a.seed),'--store-frames','--log-dir',str(out/'log')]
if a.policy=='dream':cmd.extend(['-P','model=dreamlabs/dream-1-beta'])
raise SystemExit(subprocess.call(cmd,cwd=root,env=env))
