"""Read-only environment/GPU diagnostic; does not launch or stop GPU jobs."""
import importlib.util
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def probe():
    result={'host':platform.node(),'platform':platform.platform(),'python':sys.executable,
            'python_version':platform.python_version(),'modules':{},'gpu':None,
            'task_native_mechanisms_ready':False}
    for name in ('numpy','torch','isaaclab','isaaclab_arena','yam_arena','inspect_robots','inspect_robots_yam'):
        try:spec=importlib.util.find_spec(name)
        except (ImportError,ValueError,ModuleNotFoundError):spec=None
        result['modules'][name]=None if spec is None else spec.origin
    if shutil.which('nvidia-smi'):
        gpu=subprocess.run(['nvidia-smi','--query-gpu=name,memory.used,memory.total,utilization.gpu','--format=csv,noheader'],text=True,capture_output=True,timeout=15)
        jobs=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader'],text=True,capture_output=True,timeout=15)
        result['gpu']={'devices':gpu.stdout.strip(),'compute_jobs':jobs.stdout.strip(),
                       'query_ok':gpu.returncode==0 and jobs.returncode==0,
                       'available':gpu.returncode==0 and jobs.returncode==0 and not jobs.stdout.strip()}
    result['native_imports_present']=all(result['modules'][name] for name in ('numpy','torch','isaaclab','isaaclab_arena','yam_arena'))
    return result


if __name__=='__main__':
    print(json.dumps(probe(),indent=2))
