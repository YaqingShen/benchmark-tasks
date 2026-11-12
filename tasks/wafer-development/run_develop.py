"""Run and verify one fresh development trial, waiting for an idle GPU."""
from pathlib import Path
import argparse, fcntl, json, os, signal, subprocess, sys, tarfile, time

p=argparse.ArgumentParser()
p.add_argument('--bundle',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--lock',type=Path,required=True)
p.add_argument('--render',action='store_true')
a=p.parse_args();a.bundle=a.bundle.resolve();a.output=a.output.resolve()
if a.output.exists():raise SystemExit('Use a new output directory; existing evidence is never overwritten.')
sys.path.insert(0,str(a.bundle))
from wafer_retention_isaac_v12 import ensure_gpu_idle
scripts=Path(__file__).resolve().parent
for name in ('verify_develop.py','render_develop.py'):
    if not(scripts/name).is_file():raise SystemExit('Missing verification tool: '+name)
a.lock.parent.mkdir(parents=True,exist_ok=True)
lock=a.lock.open('a');a.output.parent.mkdir(parents=True,exist_ok=True)

def acquire():
    waiting=False
    while True:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:pass
        else:
            try:ensure_gpu_idle();return
            except RuntimeError:fcntl.flock(lock,fcntl.LOCK_UN)
        if not waiting:print('Waiting for an idle GPU and the shared run lock.',flush=True);waiting=True
        time.sleep(15)

def release():fcntl.flock(lock,fcntl.LOCK_UN)

def stop(child):
    for sig,wait in ((signal.SIGTERM,10),(signal.SIGKILL,10)):
        try:os.killpg(child.pid,sig)
        except ProcessLookupError:pass
        try:child.wait(timeout=wait);return
        except subprocess.TimeoutExpired:pass
    raise RuntimeError('Owned simulator process did not stop.')

def run_gpu(command,log_path,report=None,timeout_s=None,replay=None):
    acquire()
    try:
        with log_path.open('x')as log:
            child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            completed_since=None;started=time.monotonic()
            while child.poll()is None:
                time.sleep(5)
                if child.poll()is not None:break
                if timeout_s and time.monotonic()-started>timeout_s:
                    stop(child);raise RuntimeError('Rendering exceeded its bounded runtime; inspect '+str(log_path))
                finished=bool(report and report.is_file()and '[INFO]: SimulationContext cleared'in log_path.read_text(errors='replace'))
                if replay and replay.is_file()and (replay.parent/'wafer_develop_isaac.mp4').is_file():
                    try:proof=json.loads(replay.read_text())
                    except json.JSONDecodeError:proof={}
                    finished=proof.get('frame_count',0)>1 and proof.get('preview_only')is False
                if finished:
                    if completed_since is None:completed_since=time.monotonic()
                    if time.monotonic()-completed_since>=30:
                        print('Evidence is saved; closing its stalled simulator shutdown.',flush=True)
                        stop(child)
                        return 0
            return child.returncode
    finally:release()

print('Starting fresh native development trial: '+str(a.output),flush=True)
report=a.output/'trial_report.json'
log_path=Path(str(a.output)+'.log')
rc=run_gpu([sys.executable,'-u',str(a.bundle/'wafer_develop_isaac.py'),'--bundle',str(a.bundle),'--output',str(a.output)],log_path,report)
if not report.is_file():raise SystemExit('Simulation ended without a final report; see '+str(log_path))
result=json.loads(report.read_text())
with tarfile.open(a.output/'source_bundle.tar.gz','x:gz')as archive:
    for path in sorted(a.bundle.rglob('*')):
        if path.is_file()and '__pycache__'not in path.parts:
            archive.add(path,arcname=str(Path('bundle')/path.relative_to(a.bundle)))
print(json.dumps({k:result.get(k)for k in ('result','score','success','error')}),flush=True)
if result.get('result')!='WAFER_DEVELOP_SUCCEEDED_IN_THIS_SIMULATION':raise SystemExit(1)
subprocess.run([sys.executable,'-u',str(scripts/'verify_develop.py'),'--bundle',str(a.bundle),'--run',str(a.output)],check=True)
audit=json.loads((a.output/'independent_check.json').read_text())
if not audit.get('passed'):raise SystemExit('Independent verification failed.')
if a.render:
    rc=run_gpu([sys.executable,'-u',str(scripts/'render_develop.py'),'--bundle',str(a.bundle),'--run',str(a.output),'--output',str(a.output/'render')],a.output/'render.log',timeout_s=3600,replay=a.output/'render/replay.json')
    if rc:raise SystemExit('Video rendering failed; verified physical result remains saved.')
print('Development demonstration and independent verification passed.',flush=True)
