"""Run the frozen task through the official Inspect Robots policy interface."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shlex
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--policy', choices=('dream', 'hold'), default='dream')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--steps', type=int, default=12000)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--epochs', type=int, default=1)
    p.add_argument('--show-command', action='store_true')
    a = p.parse_args()
    root = Path(__file__).resolve().parent
    output = a.output.resolve()
    if not 1 <= a.steps <= 12000:
        p.error('The action budget is between 1 and 12000')
    if a.epochs < 1:
        p.error('At least one epoch is required')
    manifest = json.loads((root/'source_manifest.json').read_text())
    for name, expected in manifest['files'].items():
        path = (root/name).resolve()
        if not path.is_relative_to(root) or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('Source checksum mismatch: '+name)
    instruction = json.loads((root/'submission-draft.json').read_text())['instruction']
    command = [sys.executable, str(Path(__file__).resolve()), '--policy', a.policy,
               '--output', str(output), '--steps', str(a.steps),
               '--seed', str(a.seed), '--epochs', str(a.epochs)]
    if a.show_command:
        print(shlex.join(command))
        return
    if output.exists():
        p.error('Choose a new output directory; existing evidence is preserved')
    versions = {n: importlib.metadata.version(n) for n in ('inspect-robots', 'inspect-robots-yam')}
    if a.policy == 'dream':
        # This client is provided by the prepared RoboCurve environment.
        # Never replace an unavailable Dream-1 policy with another model.
        versions['inspect-robots-dream'] = importlib.metadata.version('inspect-robots-dream')
    output.mkdir(parents=True)
    (output/'run-command.json').write_text(json.dumps({'command': command, 'versions': versions,
        'source_revision': manifest['revision'], 'policy': a.policy,
        'api': 'inspect_robots.eval', 'seed': a.seed, 'epochs': a.epochs,
        'instruction': instruction, 'max_steps': a.steps,
        'source_manifest_sha256': hashlib.sha256((root/'source_manifest.json').read_bytes()).hexdigest()}, indent=2))
    # Isaac Sim prepares this process with native libraries. Execute the verified
    # SDK API here; spawning a second Python CLI can break that initialization.
    arena = None
    result = {'exit_code': 1, 'dream1_run': a.policy == 'dream',
              'note': 'Process completion alone is not task success; inspect the native and SDK results.'}
    try:
        from formulation_benchmark import FormulationArena, HoldPolicy, LiquidCorrectionScore
        from inspect_robots import eval
        from inspect_robots.scene import Scene
        from inspect_robots.task import Task

        arena = FormulationArena(bundle=root, evidence_dir=output/'evidence',
                                 max_steps=a.steps, cam_width=1280, cam_height=704)
        scene = Scene(id='liquid-dose-correction', instruction=instruction, init_seed=a.seed)
        task = Task(name='liquid-dose-correction', scenes=[scene], scorer=LiquidCorrectionScore(),
                    max_steps=a.steps, epochs=a.epochs)
        policy = 'dream' if a.policy == 'dream' else HoldPolicy()
        logs = eval(task, policy, arena, log_dir=str(output/'inspect'), seed=a.seed,
                    store_frames=True, fail_on_error=True)
        statuses = [log.status for log in logs]
        if not logs or any(status == 'error' for status in statuses):
            raise RuntimeError('SDK rollout failed; see preserved logs')
        result.update(exit_code=0, log_count=len(logs), statuses=statuses)
    except BaseException as exc:
        result['error_type'] = type(exc).__name__
        raise
    finally:
        try:
            if arena is not None:
                arena.close()
        finally:
            (output/'process-result.json').write_text(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
