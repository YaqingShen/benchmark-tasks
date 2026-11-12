"""Run Dream-1 or a local control policy through Inspect Robots."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess

from check_bundle import TASK, RUNTIME, check


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--policy', choices=('dream', 'hold', 'scripted'), default='dream')
    p.add_argument('--steps', type=int, default=2400)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--native-run', type=Path)
    p.add_argument('--show-command', action='store_true')
    args = p.parse_args()
    if not 1 <= args.steps <= 2400:
        p.error('The action budget must be 1–2400.')
    if args.policy == 'scripted' and args.native_run is None:
        p.error('The scripted policy needs an independently verified --native-run.')
    check()
    output = args.output.resolve()
    if output.exists():
        p.error('Choose a new output directory; existing evidence is preserved.')
    spec = json.loads((TASK / 'task.json').read_text())
    policies = dict(dream='dream', hold='wafer_blow_hold', scripted='wafer_blow_scripted')
    command = ['inspect-robots', 'run', '--embodiment', 'wafer_blow_arena',
               '--policy', policies[args.policy], '-E', 'bundle='+str(RUNTIME),
               '-E', 'evidence_dir='+str(output / 'evidence'),
               '-E', 'device=cpu', '-E', 'development=true',
               '-E', 'cam_width=320', '-E', 'cam_height=176',
               '--instruction', spec['instruction'], '--scorer', 'wafer_blow_score',
               '--max-steps', str(args.steps), '--store-frames', '--no-rerun',
               '--log-dir', str(output / 'inspect')]
    if args.policy == 'scripted':
        source = args.native_run.resolve()
        command += ['-P', 'run='+str(source)]
        source_plan = json.loads((source / 'metadata.json').read_text())['full_workflow_plan']
        candidate_plan = json.loads((RUNTIME / 'full_workflow_plan.json').read_text())
        if source_plan != candidate_plan:
            command += ['-P', 'candidate_bundle='+str(RUNTIME)]
    if args.show_command:
        print(shlex.join(command))
        return
    env = dict(os.environ)
    paths = [str(TASK / '.policy-plugin'), str(RUNTIME), str(RUNTIME / 'benchmark')]
    if env.get('PYTHONPATH'):
        paths.append(env['PYTHONPATH'])
    env['PYTHONPATH'] = os.pathsep.join(paths)
    subprocess.run(command, env=env, check=True)


if __name__ == '__main__':
    main()
