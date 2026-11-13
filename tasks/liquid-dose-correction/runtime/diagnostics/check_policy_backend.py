"""Native adapter smoke test; does not run Dream-1 or a task demonstration."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
p = argparse.ArgumentParser()
p.add_argument('--output', type=Path, required=True)
p.add_argument('--seed', type=int, default=0)
p.add_argument('--steps', type=int, default=24)
a = p.parse_args()
from formulation_backend import FormulationWorld
world = None
try:
    world = FormulationWorld(ROOT, a.output)
    images, joints, _ = world.reset(seed=a.seed)
    assert set(images) == {'top_cam', 'left_cam', 'right_cam'}
    assert joints.shape == (14,)
    for _ in range(a.steps):
        world.step(joints)
    result = world.score()
    assert result['score'] == 0., 'Untouched starting dose must earn zero'
    assert not result.get('violations'), result
    print(json.dumps({'backend_smoke_passed': True, 'full_task_demo': False,
                      'dream1_run': False, 'result': result}), flush=True)
finally:
    if world is not None:
        world.close()
