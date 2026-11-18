"""On the station table: a countertop slush / granita machine (clear hopper with a turning auger on top, a front tap
with a pull-down handle at x 0.37-0.39 m, y 0, and a drip tray under the tap at z 0.86 m), and to its left (y +0.20 m)
a small serve stand whose platform top is at z 0.888 m. An empty 16 oz clear plastic (PET) cup, 93 mm rim, 110 mm tall,
stands on the serve stand.

Task: pick the plastic cup up off the serve stand, set it on the drip tray under the tap, pull the tap handle down so
slushie flows into the cup, let the handle spring back up once the cup is about two-thirds full, then carry the filled
cup back and set it upright on the serve stand without spilling.

Success (held 1 s): cup on the serve stand (within 15 mm), upright (< 5 deg), fill from 60% through 73% of capacity, spill < 5% of what
was dispensed, hands clear, cup at rest, tap closed. Partial credit: lifted 0.10, on tray 0.15, tap opened 0.10, fill progress
0.25, and once filled 0.40 x the fraction of the six final checks passing.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py -E object_jitter=0 -E control_hz=20 -E device=cpu \\
           --instruction "..." --scorer task_score --max-steps 2400 --store-frames --log-dir logs
"""
import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import slushie_runtime as rt  # noqa: E402  (installs the scene / embodiment hooks)
from slushie_runtime import UsdObject  # noqa: E402

SCENE = "station"
ASSETS = rt.SRC / 'assets'
OBJECTS = [
    UsdObject(name='slush_machine', pos=(0., 0., 0.), usd=str(ASSETS/'slush_machine.usda'), kind='articulation'),
    UsdObject(name='cupstand', pos=(0., 0., 0.), usd=str(ASSETS/'cupstand.usda'), kind='base'),
    UsdObject(name='cup', pos=tuple(map(float, rt.CUP_START)), usd=str(ASSETS/'cup.usda'), kind='rigid', mass=.012),
]
SUCCESS_HOLD_S = 1.0
SCORE_OVER = "end"


def success(objects):
    return bool(rt.RT.metrics.get('passed', False))


def score(objects):
    return float(rt.RT.metrics.get('score', 0.))
