"""Offline scorer check (no Isaac): synthetic trajectories of the waffle maker's joints and stand pose -> breakdown()."""
import json, math, pathlib, sys
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import task as T
from labkit import PartState
T.phase_conversion.STATE['solid']=True  # synthetic cooked food throughout these mechanical tests

PI = math.pi
D = math.radians


Q0 = np.array([0, 0, 0, 1.0])


def _ps(pos, q=Q0, vel=(0.0, 0.0, 0.0), joints=None):
    z = np.zeros(3)
    return PartState(pos=np.asarray(pos, float), quat_xyzw=np.asarray(q, float), lin_vel=np.asarray(vel, float), ang_vel=z,
                     low=z, high=z, joints=joints or {})


IN_GRID = np.asarray(T.HOME, float)  # the waffle's part origin where it lies in the lower grid (its centre on the axis)
ON_PLATE = np.asarray(T.PLATE_POS, float) + np.array([0.0, 0.0, T.PLATE_FLOOR + (T.ZA - T._W_LOW) + 0.0008 - T.ZA])


def st(spin, lid, latch=0.0, shift=(0.0, 0.0), food=None, fq=None, fvel=(0.0, 0.0, 0.0)):
    p = np.asarray(T.HOME, float) + np.array([shift[0], shift[1], 0.0])
    follow_grid = fq is None and food is None
    if fq is None:
        fq = np.array([math.sin(spin/2), 0., 0., math.cos(spin/2)]) if food is None else Q0
    food_origin = IN_GRID + np.array([shift[0],shift[1],0.])
    if follow_grid:
        food_origin += np.array([0.,T.ZA*math.sin(spin),T.ZA*(1-math.cos(spin))])
    return {"waffle": _ps(p, joints=dict(spin=spin, lid=lid, latch=latch)),
            "belgian_waffle": _ps(food_origin if food is None else food, fq, fvel),
            "dinner_plate": _ps(T.PLATE_POS),
            "kitchen_tongs": _ps(T.TONG_HOME, np.array([0.,0.,math.sin(T.TONG_YAW/2),math.cos(T.TONG_YAW/2)]), joints={'jaw':0.}),
            "tong_cup": _ps([T.TONG_HOME[0],T.TONG_HOME[1],T.TOP])}


def qx(deg):
    return np.array([math.sin(math.radians(deg) / 2), 0.0, 0.0, math.cos(math.radians(deg) / 2)])


def ramp(a, b, n=30):
    return list(np.linspace(a, b, n))


def run(states):
    tr = T.new_track()
    for s in states:
        T.track_update(tr, s)
    return T.breakdown(states[-1], tr)


OPEN = T.LID_START  # the lid starts propped at 45 deg and is opened back to it (O needs >= LID_OPEN_MIN = 35 deg)
close = [st(0, l, 0.0 if l < 0.01 else 0.0) for l in ramp(OPEN, 0.0, 20)]
turn = [st(a, 0.0) for a in ramp(0, PI, 40)] + [st(a, 0.0) for a in ramp(PI, 0, 40)]
unlatch_open = [st(0, 0.0, -0.014)] + [st(0, l, -0.014) for l in ramp(0, OPEN, 20)] + [st(0, OPEN, 0.0)] * 5
cases = {}
cases["untouched (lid open)"] = [st(0, OPEN)] * 10
cases["correct"] = close + turn + unlatch_open
cases["correct, stand nudged 0.6 cm"] = close + [dict(st(a, 0.0, shift=(0.006 * a / PI, 0.0))) for a in ramp(0, PI, 40)] \
    + [st(a, 0.0, shift=(0.006, 0.0)) for a in ramp(PI, 0, 40)] + [st(0, l, -0.014, (0.006, 0)) for l in ramp(0, OPEN, 20)]
pop = [st(a, (0.0 if a < D(60) else min(0.6, (a - D(60)) * 0.8)), -0.014 if a > D(60) else 0.0) for a in ramp(0, PI, 40)]
cases["lid pops open mid-turn (35 deg)"] = close + pop + [st(a, 0.6) for a in ramp(PI, 0, 40)] + [st(0, l) for l in ramp(0.6, OPEN, 10)]
cases["lid rattles 3 deg (latched)"] = close + [st(a, D(3) * math.sin(a)) for a in ramp(0, PI, 40)] + [st(a, 0.0) for a in ramp(PI, 0, 40)] + unlatch_open
cases["turned without latching (hand holds lid)"] = close[:-1] + [st(0, 0.0, -0.014)] + [st(a, 0.0, -0.014) for a in ramp(0, PI, 40)] \
    + [st(a, 0.0, -0.014) for a in ramp(PI, 0, 40)] + unlatch_open
cases["stand shoved 2 cm at the end stop"] = close + [st(a, 0.0, shift=(0.0, 0.02 if a > 3.0 else 0.0)) for a in ramp(0, PI, 40)] \
    + [st(a, 0.0, shift=(0.0, 0.02)) for a in ramp(PI, 0, 40)] + [st(0, l, -0.014, (0, 0.02)) for l in ramp(0, OPEN, 20)]
cases["stand shoved 5 cm"] = close + [st(a, 0.0, shift=(0.05 if a > 3.0 else 0.0, 0.0)) for a in ramp(0, PI, 40)] \
    + [st(a, 0.0, shift=(0.05, 0.0)) for a in ramp(PI, 0, 40)] + [st(0, l, -0.014, (0.05, 0)) for l in ramp(0, OPEN, 20)]
cases["only half-turn (left upside down)"] = close + [st(a, 0.0) for a in ramp(0, PI, 40)] + [st(PI, 0.0)] * 10
cases["half-turn, back only to 90 deg"] = close + [st(a, 0.0) for a in ramp(0, PI, 40)] + [st(a, 0.0) for a in ramp(PI, PI / 2, 20)]
cases["turned only to 150 deg and back"] = close + [st(a, 0.0) for a in ramp(0, D(150), 30)] + [st(a, 0.0) for a in ramp(D(150), 0, 30)] + unlatch_open
cases["both turns, lid left shut"] = close + turn + [st(0, 0.0)] * 5
cases["correct, but lid opened only 30 deg"] = close + turn + [st(0, 0.0, -0.014)] + [st(0, l, -0.014) for l in ramp(0, D(30), 10)] \
    + [st(0, D(30), 0.0)] * 5
cases["turned with lid open (never closed)"] = [st(a, OPEN) for a in ramp(0, PI, 40)] + [st(a, OPEN) for a in ramp(PI, 0, 40)]
# demolding: the waffle is carried from the grid to the plate after the cycle (straight line, held in the air)
carry = [st(0, OPEN, food=IN_GRID + (ON_PLATE + [0, 0, 0.10] - IN_GRID) * u, fvel=(0.0, 0.2, 0.0)) for u in np.linspace(0, 1, 15)]
cases["correct, waffle laid on the plate"] = close + turn + unlatch_open + carry + [st(0, OPEN, food=ON_PLATE)] * 5
cases["correct, waffle on the plate 3 cm off centre"] = close + turn + unlatch_open + carry + [st(0, OPEN, food=ON_PLATE + [0.03, 0, 0])] * 5
cases["correct, waffle dropped on the plate's rim (7 cm off, tilted 15 deg)"] = close + turn + unlatch_open + carry \
    + [st(0, OPEN, food=ON_PLATE + [0.07, 0, 0.012], fq=qx(15))] * 5
cases["correct, waffle still held 3 cm over the plate"] = close + turn + unlatch_open + carry + [st(0, OPEN, food=ON_PLATE + [0, 0, 0.03])] * 5
cases["correct, waffle still sliding on the plate"] = close + turn + unlatch_open + carry + [st(0, OPEN, food=ON_PLATE, fvel=(0.2, 0, 0))] * 3
cases["waffle put on the plate before the turn"] = [st(0, OPEN, food=ON_PLATE)] * 3 + [st(0, l, food=ON_PLATE) for l in ramp(OPEN, 0.0, 20)] \
    + [st(a, 0.0, food=ON_PLATE) for a in ramp(0, PI, 40)] + [st(a, 0.0, food=ON_PLATE) for a in ramp(PI, 0, 40)] \
    + [st(0, l, -0.014, food=ON_PLATE) for l in ramp(0, OPEN, 20)]
cases["waffle on the plate but no turn"] = close + [st(0, 0.0, -0.014)] + [st(0, l, -0.014) for l in ramp(0, OPEN, 20)] + carry \
    + [st(0, OPEN, food=ON_PLATE)] * 5
keys = ("half_turn_1", "half_turn_2", "latched_at_turn", "max_lid_deg_while_turned", "stand_max_shift_cm", "lid_open_at_end",
        "waffle_on_plate", "waffle_left_early")
# weights: half-turns 0.20 + 0.20, lid kept shut 0.20, stand 0.10, lid open 0.10, waffle on the plate 0.20
EXPECT = {"untouched (lid open)": 0.0, "correct": 0.80, "correct, stand nudged 0.6 cm": 0.80, "lid pops open mid-turn (35 deg)": 0.60,
          "lid rattles 3 deg (latched)": 0.80, "turned without latching (hand holds lid)": 0.60,
          "stand shoved 2 cm at the end stop": 0.75, "stand shoved 5 cm": 0.70, "only half-turn (left upside down)": 0.50,
          "half-turn, back only to 90 deg": 0.50, "turned only to 150 deg and back": 0.0, "both turns, lid left shut": 0.70,
          "correct, but lid opened only 30 deg": 0.70, "turned with lid open (never closed)": 0.50,
          "correct, waffle laid on the plate": 0.8, "correct, waffle on the plate 3 cm off centre": 0.8,
          "correct, waffle dropped on the plate's rim (7 cm off, tilted 15 deg)": 0.80,
          "correct, waffle still held 3 cm over the plate": 0.80, "correct, waffle still sliding on the plate": 0.80,
          "waffle put on the plate before the turn": 0.80, "waffle on the plate but no turn": 0.0}
bad = 0
# Illegal shortcuts revoke all credit; these tests intentionally omit proper
# tool pickup/carry evidence, which is tested positively in test_tool_serving.py.
for name in (
    'lid pops open mid-turn (35 deg)',
    'turned without latching (hand holds lid)',
    'turned with lid open (never closed)',
    'correct, waffle laid on the plate',
    'correct, waffle on the plate 3 cm off centre',
    'waffle put on the plate before the turn',
):
    EXPECT[name]=0.
for k, v in cases.items():
    b = run(v)
    ok = abs(b["score"] - EXPECT[k]) < 1e-6
    bad += not ok
    print(f"{k:70s} score={b['score']:.3f} full={b['full_success']!s:5s}", json.dumps({x: b[x] for x in keys}), "" if ok else f"  <-- expected {EXPECT[k]}")
print("ALL PASS" if not bad else f"{bad} FAILED")

misoriented=close+[st(a,0.,fq=Q0) for a in ramp(0,PI,40)]+[st(a,0.,fq=Q0) for a in ramp(PI,0,40)]+unlatch_open+carry+[st(0,OPEN,food=ON_PLATE)]*5
assert run(misoriented)['waffle_left_early']
assert not run(misoriented)['full_success']
print('PASS: grid retention requires both position and orientation')

sys.exit(1 if bad else 0)
