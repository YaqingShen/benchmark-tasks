"""Run direct Dream-1 actions in the frozen development scene."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

TASK = Path(__file__).resolve().parents[1]
RUNTIME = TASK/'runtime'


def check():
    for manifest_name in ('bundle_files.json', 'policy_files.json'):
        manifest = json.loads((RUNTIME/manifest_name).read_text())
        for name, expected in manifest.items():
            path = (RUNTIME/name).resolve()
            if not path.is_relative_to(RUNTIME.resolve()) or not path.is_file():
                raise ValueError('Missing or unsafe bundle path: '+name)
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError('Source checksum mismatch: '+name)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--policy', choices=('dream', 'hold'), default='dream')
    p.add_argument('--steps', type=int, default=2400)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--lock', type=Path)
    p.add_argument('--show-command', action='store_true')
    a = p.parse_args()
    if not 1 <= a.steps <= 2400:
        p.error('The budget is 1–2400 actions.')
    output = a.output.resolve()
    if output.exists():
        p.error('Choose a new output directory; existing evidence is preserved.')
    check()
    spec = json.loads((TASK/'task.json').read_text())
    policy = 'dream' if a.policy == 'dream' else 'wafer_develop_hold'
    command = ['inspect-robots', 'run', '--embodiment', 'wafer_develop_arena',
               '--policy', policy, '-E', 'bundle='+str(RUNTIME),
               '-E', 'evidence_dir='+str(output/'evidence'), '-E', 'device=cpu',
               '-E', 'development=true', '-E', 'cam_width=320', '-E', 'cam_height=176',
               '-E', 'max_steps='+str(a.steps), '--instruction', spec['instruction'],
               '--scorer', 'wafer_develop_score', '--max-steps', str(a.steps),
               '--store-frames', '--no-rerun', '--log-dir', str(output/'inspect')]
    if a.show_command:
        print(shlex.join(command));return
    if a.lock is None:
        p.error('--lock must name the shared GPU run lock.')
    env = dict(os.environ)
    env['PYTHONPATH'] = os.pathsep.join([str(TASK/'.policy-plugin'), str(RUNTIME),
        str(RUNTIME/'benchmark'), env.get('PYTHONPATH', '')])
    sys.path[:0] = [str(RUNTIME), str(RUNTIME/'benchmark')]
    from wafer_retention_isaac_v12 import ensure_gpu_idle
    a.lock.parent.mkdir(parents=True, exist_ok=True)
    with a.lock.open('a') as lock:
        waiting = False
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:
                pass
            else:
                try:
                    ensure_gpu_idle();break
                except RuntimeError:
                    fcntl.flock(lock, fcntl.LOCK_UN)
            if not waiting:
                print('Waiting for an idle GPU and the shared run lock.', flush=True);waiting = True
            time.sleep(15)
        subprocess.run(command, env=env, check=True)


if __name__ == '__main__':
    main()
