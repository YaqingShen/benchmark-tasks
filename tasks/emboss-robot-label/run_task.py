"""Run the fixed task from reset with either a scripted policy or Dream-1."""
import argparse, hashlib, json, os, subprocess
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('--policy',choices=['yam_scripted','dream'],required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--seed',type=int,default=0)
p.add_argument('--preview',action='store_true')
profiles=p.add_mutually_exclusive_group()
profiles.add_argument("--officemate-scenario",action="store_true",help="Run a specification-based load comparison on the prototype fixture")
profiles.add_argument("--engineering-baseline",action="store_true",help="Use the prototype dial and nominal load, without hardware calibration")
p.add_argument("--camera-width",type=int,choices=(640,1280),default=1280)
a=p.parse_args();root=Path(__file__).resolve().parent
spec=json.loads((root/'task.json').read_text());out=a.output.resolve()
out.mkdir(parents=True,exist_ok=False)
env=os.environ.copy();env['TASK_TRACE_PATH']=str(out/'trace.jsonl')
env['PYTHONDONTWRITEBYTECODE']='1'
env["EMBOSS_ENGINEERING_BASELINE"]="1" if a.engineering_baseline else "0"
env["EMBOSS_OFFICEMATE_SCENARIO"]="1" if a.officemate_scenario else "0"
profile="Office-Mate II engineering scenario" if a.officemate_scenario else "prototype assumptions" if a.engineering_baseline else "hardware measurements"
(out/"profile.json").write_text(json.dumps({"profile":profile,"profile_sha256":hashlib.sha256((root/("omega_manual.json" if a.engineering_baseline or a.officemate_scenario else "omega_measurements.json")).read_bytes()).hexdigest(),"source_sha256":{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(root.glob("*.py"))},"policy":a.policy,"seed":a.seed})+"\n")
w,h=(320,180) if a.preview else (a.camera_width,a.camera_width*9//16)
cmd=['inspect-robots','run','--embodiment','yam_arena','--policy',a.policy,
     '--no-prompt','--grader','none','-E','device=cpu','-E','task='+str(root/'task.py'),
     '-E','object_jitter=0','-E','control_hz=20','-E',f'cam_width={w}','-E',f'cam_height={h}',
     '--instruction',spec['instruction'],'--scorer','task_score','--max-steps','2400',
     '--seed',str(a.seed),'--store-frames','--log-dir',str(out/'log')]
if a.policy=='dream':cmd.extend(['-P','model=dreamlabs/dream-1-beta'])
raise SystemExit(subprocess.call(cmd,cwd=root,env=env))
