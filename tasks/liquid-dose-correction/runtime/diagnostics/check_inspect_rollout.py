"""One hold-action SDK rollout; this is not Dream-1 or task success."""
import argparse
import json
from pathlib import Path
import sys

p = argparse.ArgumentParser()
p.add_argument('--source', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--epochs', type=int, default=1)
a = p.parse_args()
sys.path.insert(0, str(a.source.resolve()))
from formulation_benchmark import FormulationArena, HoldPolicy, LiquidCorrectionScore
from inspect_robots import eval
from inspect_robots.scene import Scene
from inspect_robots.task import Task

arena = FormulationArena(bundle=a.source, evidence_dir=a.output/'evidence', max_steps=24)
try:
    scene = Scene(id='adapter-hold-check', instruction='Adjust the liquid dose to 10.000 +/- 0.050 g.', init_seed=1)
    task = Task(name='liquid-correction-adapter-check', scenes=[scene], scorer=LiquidCorrectionScore(), max_steps=24, epochs=a.epochs)
    logs = eval(task, HoldPolicy(), arena, log_dir=str(a.output/'inspect'), store_frames=True, fail_on_error=True)
    if any(log.status == 'error' for log in logs):
        raise RuntimeError('SDK rollout failed; see preserved logs')
    result = arena.world.score()
    assert result['score'] == 0. and not result.get('violations'), result
    print(json.dumps({'sdk_rollout_passed': True, 'dream1_run': False, 'full_task_demo': False,
                      'result': result}), flush=True)
finally:
    arena.close()
