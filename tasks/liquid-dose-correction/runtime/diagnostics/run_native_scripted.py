"""Physical scripted candidate scored through the direct-action backend.

Motion alignment uses simulator object poses; dosing uses the force-based
balance. This is explicitly a scripted feasibility run, never Dream-1 and
never a robot-observation-only policy result. No object states are prescribed.
"""
import argparse
import json
from pathlib import Path
import sys
import traceback
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--initial-mass-g', type=float, required=True)
a = p.parse_args()
from formulation_backend import FormulationWorld
from pipette_correction import PipetteCorrectionProbe
world = None
try:
    world = FormulationWorld(ROOT, a.output)
    _, joints, _ = world.reset(initial_mass=a.initial_mass_g)
    direction = 'add' if a.initial_mass_g < 10. else 'remove'
    probe = PipetteCorrectionProbe(world.base, joints, world.coupling, direction)
    world.metadata.update(controller='scripted simulator-pose alignment and force-balance feedback',
                          robot_observation_only=False, dream1_run=False,
                          controller_scope='physical feasibility candidate')
    (world.trial/'metadata.json').write_text(json.dumps(world.metadata, indent=2))
    for step in range(12000):
        t = (step+1)/20.
        action, label = probe.command(t)
        world.step(action)
        probe.observe(t, label)
        if step % 20 == 0:
            (world.trial/'probe-progress.json').write_text(json.dumps(probe.rows[-1], indent=2))
        if world.score().get('violations'):
            raise RuntimeError('Native task validity check failed: '+str(world.score()['violations']))
        if t >= probe.duration-1e-8:
            break
    (world.trial/'mechanism-probe.json').write_text(json.dumps(probe.result(), indent=2))
    world.finish()
    print(json.dumps({'scripted_candidate_finished': True, 'dream1_run': False,
                      'robot_observation_only': False, 'steps': world.steps, 'result': world.score()}), flush=True)
except Exception as exc:
    if world is not None and world.trial is not None:
        (world.trial/'error.json').write_text(json.dumps({'error': str(exc), 'traceback': traceback.format_exc()}, indent=2))
    raise
finally:
    if world is not None:
        world.close()
