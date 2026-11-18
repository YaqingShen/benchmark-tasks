"""On the station table: a painted wall panel with a glazed ceramic hook held by ONE centre screw (M4, with a nylon
washer under its head), a pen-type cordless screwdriver lying in a stand, and a foam-lined parts tray.

Taking the hook down means: hold the hook so it cannot fall, back the screw out with the screwdriver (the screw head
sits in a recess in the hook, so fingers cannot turn it), then put the hook, the screw and the washer in the tray and
the screwdriver back in its stand. If nothing holds the hook when the screw lets go, it falls.

How the screwdriver works here: its motor runs while the bit is seated in the screw head (push to start). The screw
then backs out as the hand yields backwards, at most 2.8 mm/s (0.7 mm pitch at 240 rpm), and is free after 10 mm.
"""

import dataclasses
import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from labkit import _part_state, on_reset, on_step, part  # noqa: E402
from scoring_geometry import collision_bounds, hand_bounds, maximum_surface_speed, separation  # noqa: E402
from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

SCENE = "station"
TOP = TABLE_TOP[SCENE]

PAINT = (0.93, 0.92, 0.88)
WOOD = (0.72, 0.58, 0.40)
GLAZE = (0.06, 0.36, 0.52)  # glossy teal glaze
STEEL = (0.70, 0.71, 0.74)
NYLON = (0.94, 0.94, 0.90)
DARK = (0.16, 0.17, 0.19)
ORANGE = (0.92, 0.45, 0.10)
FOAM = (0.28, 0.29, 0.31)
TRAY = (0.80, 0.81, 0.83)


def plate_x(x0, x1, y0, y1, z0, z1, hole_c, hole, colour):
    """A plate lying in the y-z plane (thickness along x) with a square hole: four boxes (centre, size, colour)."""
    hy, hz = hole_c
    h = hole / 2
    xc, t = (x0 + x1) / 2, x1 - x0
    return [
        ((xc, (y0 + y1) / 2, (hz + h + z1) / 2), (t, y1 - y0, z1 - hz - h), colour),  # above the hole
        ((xc, (y0 + y1) / 2, (z0 + hz - h) / 2), (t, y1 - y0, hz - h - z0), colour),  # below it
        ((xc, (hy + h + y1) / 2, hz), (t, y1 - hy - h, hole), colour),  # left of it
        ((xc, (y0 + hy - h) / 2, hz), (t, hy - h - y0, hole), colour),  # right of it
    ]


# -- wall panel (fixed). Part frame: origin on the table under the hook, x = 0 is the painted front face ---------------
WALL_X = 0.30  # front face, world x
HOOK_Z = 0.110  # hook centre above the table
AXIS_Z = TOP + HOOK_Z
WALL_T, WALL_W, WALL_H = 0.018, 0.36, 0.155
SHANK_R, SHANK_HOLE = 0.002, 0.0042  # M4 shank in a snug hole (the insert's thread is modelled in code, below)
WALL = part(
    "wall", pos=(WALL_X, 0.0, TOP), fixed=True, friction=0.8,
    boxes=[
        *plate_x(0.0, WALL_T, -WALL_W / 2, WALL_W / 2, 0.0, WALL_H, (0.0, HOOK_Z), SHANK_HOLE, PAINT),
        ((WALL_T + 0.05, 0.15, 0.010), (0.10, 0.04, 0.020), WOOD),  # feet behind the panel
        ((WALL_T + 0.05, -0.15, 0.010), (0.10, 0.04, 0.020), WOOD),
    ],
)

# -- ceramic hook (loose): a round knob hook, a waisted neck against the wall flaring to a 60 mm cap; things hang
# behind the cap. Part frame: origin at the centre of its back face, which lies on the wall; the front is towards -x.
HOOK_T, BORE_T, CAP_T = 0.042, 0.006, 0.018  # depth off the wall; depth of the recess for the screw head; cap
SEAT_X = -(HOOK_T - BORE_T)  # floor of the recess
HOOK_D, NECK_D = 0.060, 0.030
HOOK_HOLE, BORE = 0.0052, 0.012
# Meridian profile (x along the screw axis, radius), a closed loop: back face, neck, cap rim, front, recess, hole.
HOOK_PROFILE = [
    (0.0, 0.0026), (0.0, 0.0165), (-0.002, 0.0170), (-0.008, 0.0150), (-0.016, 0.0138), (-0.022, 0.0145),
    (-0.025, 0.0150), (-0.028, 0.0185), (-0.030, 0.0240), (-0.0315, 0.0280), (-0.033, 0.0297), (-0.035, 0.0300),
    (-0.037, 0.0296), (-0.039, 0.0282), (-0.0405, 0.0255), (-0.0415, 0.0210), (-0.0420, 0.0150), (-0.0420, 0.0085),
    (-0.0416, 0.0068), (-0.0405, 0.0063), (-0.0360, 0.0060), (-0.0360, 0.0026),
]


def revolve_x(profile, colour, segments=96, roughness=0.08):
    """A closed profile turned about the x axis, as a quad mesh with smooth normals (visual only)."""
    prof = np.asarray(profile, dtype=float)
    ang = np.linspace(0.0, 2 * math.pi, segments, endpoint=False)
    pts = np.array([[x, r * math.cos(a), r * math.sin(a)] for x, r in prof for a in ang])
    n = len(prof)
    quads = [[i * segments + j, i * segments + (j + 1) % segments, ((i + 1) % n) * segments + (j + 1) % segments,
              ((i + 1) % n) * segments + j] for i in range(n) for j in range(segments)]
    q = np.array(quads)
    fn = np.cross(pts[q[:, 1]] - pts[q[:, 0]], pts[q[:, 2]] - pts[q[:, 0]])
    centre = pts[q].mean(axis=1)
    if np.einsum("ij,ij->i", fn, centre).sum() < 0:  # signed volume: make the faces point outwards
        q = q[:, ::-1]
        fn = -fn
    normals = np.zeros_like(pts)
    for k in range(4):
        np.add.at(normals, q[:, k], fn)
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    return {"points": pts.tolist(), "counts": [4] * len(q), "indices": q.ravel().tolist(), "normals": normals.tolist(),
            "color": colour, "roughness": roughness}


def _disc(x0, x1, hole, colour, d=HOOK_D):
    """Collision for a round plate: three overlapping rectangular plates with a square hole in the middle."""
    out = []
    for wy, wz in ((d, 0.6 * d), (0.6 * d, d), (0.83 * d, 0.83 * d)):
        out += plate_x(x0, x1, -wy / 2, wy / 2, -wz / 2, wz / 2, (0.0, 0.0), hole, colour)
    return out


HOOK_HOME = (WALL_X, 0.0, AXIS_Z)
HOOK = part(
    "hook", pos=HOOK_HOME, mass=0.13, friction=0.6,
    boxes=[
        *_disc(-(HOOK_T - CAP_T), 0.0, HOOK_HOLE, GLAZE, d=NECK_D),  # neck
        *_disc(SEAT_X, -(HOOK_T - CAP_T), HOOK_HOLE, GLAZE),  # cap, round the shank
        *_disc(-HOOK_T, SEAT_X, BORE, GLAZE),  # cap front, the recess round the head and washer
    ],
)
HOOK = dataclasses.replace(HOOK, meta={"meshes": [revolve_x(HOOK_PROFILE, GLAZE)], "hide_primitives": True})

# -- nylon cushion washer (loose) and the M4 x 50 pan-head screw (loose; held by the thread model below) ---------------
WASHER_D, WASHER_HOLE, WASHER_T = 0.010, 0.0046, 0.0015
PLAY = 0.0002  # axial play under the head, so the clamped stack is not squeezed
HEAD_R, HEAD_H, SHANK_L = 0.004, 0.003, 0.050
WASHER_HOME = (WALL_X + SEAT_X - WASHER_T / 2 - PLAY / 2, 0.0, AXIS_Z)
WASHER = part(
    "washer", pos=WASHER_HOME, mass=0.0006, friction=0.5,
    boxes=plate_x(-WASHER_T / 2, WASHER_T / 2, -WASHER_D / 2, WASHER_D / 2, -WASHER_D / 2, WASHER_D / 2, (0.0, 0.0),
                  WASHER_HOLE, NYLON),
)
# Screw frame: origin on the axis at the underside of the head; the shank runs towards +x.
SCREW_X0 = WALL_X + SEAT_X - WASHER_T - PLAY  # fully in
HEAD_TOP0 = SCREW_X0 - HEAD_H
SCREW = part(
    "screw", pos=(SCREW_X0, 0.0, AXIS_Z), mass=0.004, friction=0.5,
    cylinders=[
        ((-HEAD_H / 2, 0.0, 0.0), HEAD_R, HEAD_H, DARK, "X"),
        ((SHANK_L / 2, 0.0, 0.0), SHANK_R, SHANK_L, STEEL, "X"),
    ],
)

# -- pen screwdriver (loose) in its stand (fixed). Driver frame: origin at the middle of its body, bit towards +x -------
DRV_R, DRV_L, CHUCK_L, BIT_R, BIT_L = 0.009, 0.160, 0.012, 0.002, 0.030
TIP_LOCAL = DRV_L / 2 + CHUCK_L + BIT_L  # bit tip, along the driver's x
STAND_X, STAND_Y, CRADLE_Z = 0.20, -0.21, 0.035
CRADLES = (-0.060, 0.050)  # part-frame x of the two rests
GRIP_LOCAL = -0.015  # where the hand holds the driver, along its x: between the rests, behind the middle
DRIVER_HOME = (STAND_X, STAND_Y, TOP + CRADLE_Z + DRV_R)
DRIVER = part(
    "screwdriver", pos=DRIVER_HOME, mass=0.09, friction=0.9,
    cylinders=[
        ((0.0, 0.0, 0.0), DRV_R, DRV_L, ORANGE, "X"),
        ((DRV_L / 2 - 0.008, 0.0, 0.0), DRV_R, 0.016, DARK, "X"),  # dark nose band, same diameter
        ((DRV_L / 2 + CHUCK_L / 2, 0.0, 0.0), 0.005, CHUCK_L, STEEL, "X"),  # chuck
        ((DRV_L / 2 + CHUCK_L + BIT_L / 2, 0.0, 0.0), BIT_R, BIT_L, STEEL, "X"),  # PH2 bit
    ],
)


def _cradle(x):
    """A U-shaped rest at part-frame x: a floor block and two cheeks 1 mm clear of the driver's body."""
    return [
        ((x, 0.0, CRADLE_Z / 2), (0.014, 0.032, CRADLE_Z), DARK),
        ((x, DRV_R + 0.003, CRADLE_Z + 0.006), (0.014, 0.004, 0.012), DARK),
        ((x, -DRV_R - 0.003, CRADLE_Z + 0.006), (0.014, 0.004, 0.012), DARK),
    ]


STAND = part(
    "stand", pos=(STAND_X, STAND_Y, TOP), fixed=True, friction=0.8,
    boxes=[((0.0, 0.0, 0.003), (0.17, 0.05, 0.006), DARK), *_cradle(CRADLES[0]), *_cradle(CRADLES[1])],
)

# -- foam-lined parts tray (fixed), 7 cm walls so dropped parts cannot roll out ----------------------------------------
TRAY_X, TRAY_Y, TRAY_L, TRAY_W, TRAY_H, TRAY_WALL, FOAM_TOP = 0.185, 0.24, 0.20, 0.25, 0.070, 0.004, 0.014
PARTS_TRAY = part(
    "tray", pos=(TRAY_X, TRAY_Y, TOP), fixed=True, friction=0.9,
    boxes=[
        ((0.0, 0.0, 0.002), (TRAY_L, TRAY_W, 0.004), TRAY),
        ((0.0, 0.0, 0.009), (TRAY_L - 2 * TRAY_WALL, TRAY_W - 2 * TRAY_WALL, 0.010), FOAM),  # foam liner
        ((TRAY_L / 2 - TRAY_WALL / 2, 0.0, TRAY_H / 2), (TRAY_WALL, TRAY_W, TRAY_H), TRAY),
        ((-TRAY_L / 2 + TRAY_WALL / 2, 0.0, TRAY_H / 2), (TRAY_WALL, TRAY_W, TRAY_H), TRAY),
        ((0.0, TRAY_W / 2 - TRAY_WALL / 2, TRAY_H / 2), (TRAY_L, TRAY_WALL, TRAY_H), TRAY),
        ((0.0, -TRAY_W / 2 + TRAY_WALL / 2, TRAY_H / 2), (TRAY_L, TRAY_WALL, TRAY_H), TRAY),
    ],
)

OBJECTS = [WALL, HOOK, WASHER, SCREW, STAND, DRIVER, PARTS_TRAY]

# -- thread and screwdriver model (runs after every control step) ------------------------------------------------------
PITCH, MOTOR_REV_S, THREAD = 0.0007, 4.0, 0.010  # M4 coarse pitch; 240 rpm; thread engaged in the insert
RATE = PITCH * MOTOR_REV_S  # the fastest the screw can back out, m/s
SEAT_RADIAL, SEAT_GAP, SEAT_PUSH, SEAT_ANGLE = 0.0025, 0.0015, 0.003, math.radians(10.0)
DROP_SPEED = math.sqrt(2 * 9.81 * 0.02)  # the speed reached after 2 cm of free fall
_M = {"travel": 0.0, "threaded": True, "dropped": False, "n": 0, "hands": None}
_DEBUG = bool(__import__("os").environ.get("HOOK_DEBUG"))


def _in_tray_xy(p, margin=0.0):
    return abs(p[0] - TRAY_X) < TRAY_L / 2 - margin and abs(p[1] - TRAY_Y) < TRAY_W / 2 - margin


def _place(environment, name, pos):
    """Put a loose part at ``pos`` (task frame), upright and at rest."""
    import torch

    base = environment.unwrapped
    asset = base.scene[name]
    o = base.scene.env_origins[0]
    o = getattr(o, "torch", o).detach().cpu().numpy()
    pose = torch.tensor([[pos[0] + o[0], pos[1] + o[1], pos[2] + o[2], 0.0, 0.0, 0.0, 1.0]],
                        dtype=torch.float32, device=base.device)
    asset.write_root_pose_to_sim_index(root_pose=pose)
    asset.write_root_velocity_to_sim_index(root_velocity=torch.zeros((1, 6), dtype=torch.float32, device=base.device))


def _hold_screw(environment):
    """While threaded, the screw can only be where the thread puts it: on the axis, ``travel`` out of the wall."""
    _place(environment, "screw", (SCREW_X0 - _M["travel"], 0.0, AXIS_Z))


@on_reset
def _reset(environment):
    _M.update(travel=0.0, threaded=True, dropped=False, n=0, hands=None)
    _place(environment, "hook", HOOK_HOME)  # the mounted stack starts exactly assembled
    _place(environment, "washer", WASHER_HOME)
    _hold_screw(environment)


@on_step
def _mechanism(environment):
    hook = _part_state(environment, HOOK)
    _M["hands"] = hand_bounds(environment)
    _M["n"] += 1
    if _DEBUG and _M["n"] % 40 == 0:
        st = {s.name: _part_state(environment, s) for s in (WASHER, SCREW, DRIVER)}
        print("HOOKDBG", _M["n"], "thr", _M["threaded"], "trav_mm", round(_M["travel"] * 1e3, 2), "drop", _M["dropped"],
              "hook", np.round(hook.pos, 3).tolist(), "screw", np.round(st["screw"].pos, 3).tolist(),
              "washer", np.round(st["washer"].pos, 3).tolist(), "drv", np.round(st["screwdriver"].pos, 3).tolist(),
              flush=True)
    on_table = collision_bounds(hook, HOOK)[0][2] < TOP + 0.004 and not _in_tray_xy(hook.pos)
    if (hook.lin_vel[2] < -DROP_SPEED or on_table) and not _M["dropped"]:
        _M["dropped"] = True
        if _DEBUG:
            print("HOOKDBG dropped at", _M["n"], "vz", float(hook.lin_vel[2]), "low", float(hook.low[2] - TOP), flush=True)
    if not _M["threaded"]:
        return
    drv = _part_state(environment, DRIVER)
    tip = np.asarray(drv.to_world((TIP_LOCAL, 0.0, 0.0)))
    axis = tip - np.asarray(drv.to_world((0.0, 0.0, 0.0)))
    axis /= np.linalg.norm(axis)
    gap = (HEAD_TOP0 - _M["travel"]) - tip[0]  # > 0: the bit has not reached the head
    radial = math.hypot(tip[1], tip[2] - AXIS_Z)
    if _DEBUG and abs(gap) < 0.03 and _M["n"] % 10 == 0:
        print("HOOKDBG", _M["n"], "gap_mm", round(gap * 1e3, 2), "tip_dy_dz_mm", round(tip[1] * 1e3, 2),
              round((tip[2] - AXIS_Z) * 1e3, 2), "drv_dy_dz_mm", round(drv.pos[1] * 1e3, 2),
              round((drv.pos[2] - AXIS_Z) * 1e3, 2), "radial_mm", round(radial * 1e3, 2), "axis_x",
              round(float(axis[0]), 4), "trav_mm", round(_M["travel"] * 1e3, 2), flush=True)
    if radial < SEAT_RADIAL and -SEAT_PUSH < gap < SEAT_GAP and axis[0] > math.cos(SEAT_ANGLE):
        dt = float(getattr(environment.unwrapped, "step_dt", 0.05))
        follow = HEAD_TOP0 - tip[0]  # the travel that keeps the head on the bit
        _M["travel"] = min(max(follow, _M["travel"]), _M["travel"] + RATE * dt)
    if _M["travel"] >= THREAD:
        _M["threaded"] = False  # out of the thread: from here the screw is an ordinary loose part
        if _DEBUG:
            print("HOOKDBG released at step", _M["n"], flush=True)
    else:
        _hold_screw(environment)


# -- scoring ----------------------------------------------------------------------------------------------------------
SUCCESS_HOLD_S = 1.0
OFF_WALL = 0.02  # the hook counts as off the wall once it is 2 cm from where it was mounted
STILL = 0.01  # m/s
RELEASE_GAP = 0.005  # whole gripper/finger collision bounds at least 5 mm from each object


def _in_tray(s, spec):
    low, high = collision_bounds(s, spec)
    inner_low = np.array((TRAY_X - TRAY_L / 2 + TRAY_WALL, TRAY_Y - TRAY_W / 2 + TRAY_WALL))
    inner_high = np.array((TRAY_X + TRAY_L / 2 - TRAY_WALL, TRAY_Y + TRAY_W / 2 - TRAY_WALL))
    return bool(np.all(low[:2] >= inner_low) and np.all(high[:2] <= inner_high)
                and low[2] >= TOP + FOAM_TOP - 0.004 and high[2] <= TOP + TRAY_H + 0.020)


def _still(s, spec):
    return maximum_surface_speed(s, spec) < STILL


def _released(s, spec):
    return _M["hands"] is not None and all(
        separation(collision_bounds(s, spec), hand) >= RELEASE_GAP for hand in _M["hands"])


def _hook_resting(hook):
    low, _ = collision_bounds(hook, HOOK)
    return (_in_tray(hook, HOOK) and abs(low[2] - (TOP + FOAM_TOP)) <= 0.004
            and _still(hook, HOOK) and _released(hook, HOOK))


def _driver_home(drv):
    # Both cradle centres must lie beneath the cylindrical body, with the body at rest height.
    axis = drv.rotation()[:, 0]
    home = np.abs(np.asarray(drv.pos) - np.asarray(DRIVER_HOME))
    if not (abs(axis[0]) >= math.cos(math.radians(10)) and home[0] <= 0.020
            and _still(drv, DRIVER) and _released(drv, DRIVER)):
        return False
    for x in CRADLES:
        axial = (STAND_X + x - drv.pos[0]) / axis[0]
        centre = np.asarray(drv.pos) + axial * axis
        if not (abs(axial) <= DRV_L / 2 - 0.007 and abs(centre[1] - STAND_Y) <= 0.002
                and abs(centre[2] - DRIVER_HOME[2]) <= 0.004):
            return False
    return True


def score(objects):
    """0.25: the screw is fully out of the wall's thread. 0.5: also the hook is off the wall (2 cm or more from its
    mount) and was never dropped. 0.75: also the hook rests in the tray. 1.0: also the screw and the washer are in the
    tray and the screwdriver rests back in its stand. The conservative damage proxy caps the score at 0.25 if the
    hook is within 4 mm of the table outside the tray or its downward speed exceeds 0.6264 m/s, even if held.
    Full credit requires whole-part containment, rest, hand clearance, and a one-second success hold."""
    if _M["threaded"]:
        return 0.0
    hook, screw, washer, drv = (objects[n] for n in ("hook", "screw", "washer", "screwdriver"))
    if _M["dropped"]:
        return 0.25
    if float(np.linalg.norm(np.asarray(hook.pos) - np.asarray(HOOK_HOME))) < OFF_WALL:
        return 0.25
    if not _hook_resting(hook):
        return 0.5
    if (_in_tray(screw, SCREW) and _in_tray(washer, WASHER) and _still(screw, SCREW) and _still(washer, WASHER)
            and _released(screw, SCREW) and _released(washer, WASHER) and _driver_home(drv)):
        return 1.0
    return 0.75


def success(objects):
    return score(objects) == 1.0


@on_step
def _audit_state(environment):
    """Optional local evidence: actual state and scorer predicates, never used by the policy."""
    import json
    import os
    path = os.environ.get("HOOK_AUDIT_PATH")
    if not path or _M["n"] % 10:
        return
    objects = {s.name: _part_state(environment, s) for s in (HOOK, SCREW, WASHER, DRIVER)}
    row = {"step": _M["n"], "threaded": _M["threaded"], "travel": _M["travel"],
           "damage_proxy": _M["dropped"], "score": score(objects),
           "hook_resting": bool(_hook_resting(objects["hook"])),
           "driver_home": bool(_driver_home(objects["screwdriver"])), "objects": {}}
    for spec in (HOOK, SCREW, WASHER, DRIVER):
        s = objects[spec.name]
        low, high = collision_bounds(s, spec)
        row["objects"][spec.name] = {"pos": s.pos.tolist(), "quat_xyzw": s.quat_xyzw.tolist(),
            "low": low.tolist(), "high": high.tolist(), "lin_vel": s.lin_vel.tolist(),
            "ang_vel": s.ang_vel.tolist(), "released": bool(_released(s, spec))}
    with open(path, "a") as stream:
        stream.write(json.dumps(row) + "\n")


# -- scripted demo ----------------------------------------------------------------------------------------------------
# Left arm: straddles the hook's cap from above (fingers close along y), holds it while the right arm works, then
# draws it off the wall with the screw and washer still in its hole and sets it down in the tray.
# Right arm: takes the screwdriver from its stand, seats the bit in the screw head, yields backwards while the
# screw comes out, then puts the screwdriver back.
HOVER_Z = TOP + 0.17
CARRY_Z = TOP + 0.13  # the hook's lowest point 3 cm above the tray's walls
CAP_X = WALL_X - (HOOK_T - CAP_T / 2) - 0.003  # where the fingers hold the cap (3 mm in front of its middle)
GRIP_X = STAND_X + GRIP_LOCAL  # hand x over the stand
DRV_AT_HEAD_X = HEAD_TOP0 - (TIP_LOCAL - GRIP_LOCAL)  # hand x that puts the bit tip on the screw head
UNSCREW_BACK, UNSCREW_S = 0.013, 9.0  # yield 13 mm in 9 s: peak 2.2 mm/s, under the 2.8 mm/s the screw can follow
LEFT_PARK = (0.15, 0.25, TOP + 0.14)
RIGHT_PARK = (0.15, -0.25, TOP + 0.14)
TRAY_DROP_Z = TOP + FOAM_TOP + HOOK_D / 2 + 0.008  # cap's lowest point 8 mm above the foam
DRV_Z = TOP + CRADLE_Z + DRV_R

PLANCHECK_BOXES = [
    ("table", (0.35, 0.0, TOP - 0.02), (0.8, 1.2, 0.04)),
    ("wall", (WALL_X + WALL_T / 2, 0.0, TOP + WALL_H / 2), (WALL_T, WALL_W, WALL_H)),
    ("stand base", (STAND_X, STAND_Y, TOP + 0.003), (0.17, 0.05, 0.006)),
    ("stand cradle -", (STAND_X + CRADLES[0], STAND_Y, TOP + CRADLE_Z / 2 + 0.006), (0.014, 0.032, CRADLE_Z + 0.012)),
    ("stand cradle +", (STAND_X + CRADLES[1], STAND_Y, TOP + CRADLE_Z / 2 + 0.006), (0.014, 0.032, CRADLE_Z + 0.012)),
    ("tray wall +x", (TRAY_X + TRAY_L / 2 - TRAY_WALL / 2, TRAY_Y, TOP + TRAY_H / 2), (TRAY_WALL, TRAY_W, TRAY_H)),
    ("tray wall -x", (TRAY_X - TRAY_L / 2 + TRAY_WALL / 2, TRAY_Y, TOP + TRAY_H / 2), (TRAY_WALL, TRAY_W, TRAY_H)),
    ("tray wall +y", (TRAY_X, TRAY_Y + TRAY_W / 2 - TRAY_WALL / 2, TOP + TRAY_H / 2), (TRAY_L, TRAY_WALL, TRAY_H)),
    ("tray wall -y", (TRAY_X, TRAY_Y - TRAY_W / 2 + TRAY_WALL / 2, TOP + TRAY_H / 2), (TRAY_L, TRAY_WALL, TRAY_H)),
]


def _dur(wps):
    return sum(w.duration for w in wps)


def demo():
    hold = (CAP_X, 0.0, AXIS_Z)
    left = [
        Waypoint(2.0, (CAP_X, 0.0, HOVER_Z), FINGER_OPEN, "above the hook"),
        Waypoint(1.8, hold, FINGER_OPEN, "fingers down either side of the cap"),
        Waypoint(0.8, hold, FINGER_CLOSED, "hold the hook"),
    ]
    t_held = _dur(left)
    over_stand = (GRIP_X, STAND_Y, AXIS_Z)  # the driver is carried at the screw's height, low, clear of the left arm
    # The driver sits slightly askew in the closed fingers (bit tip about 2 mm to +y and 2.5 mm low, measured in
    # the simulator), so the hand aims off by that much to put the bit on the screw's axis.
    ay, az = -0.002, AXIS_Z + 0.0025
    pre = (DRV_AT_HEAD_X - 0.025, ay, az)
    seat = (DRV_AT_HEAD_X + 0.0005, ay, az)
    back = (seat[0] - 0.040, ay, az)
    side = (pre[0], -0.13, az)  # a corner on the way, so the right arm comes in from its own side
    right = [
        Waypoint(2.0, (GRIP_X, STAND_Y, HOVER_Z), FINGER_OPEN, "above the screwdriver"),
        Waypoint(1.6, (GRIP_X, STAND_Y, DRV_Z), FINGER_OPEN, "down round its body"),
        Waypoint(0.8, (GRIP_X, STAND_Y, DRV_Z), FINGER_CLOSED, "grip it"),
        Waypoint(1.2, over_stand, FINGER_CLOSED, "lift it to the screw's height"),
    ]
    if _dur(right) < t_held:
        right.append(Waypoint(t_held - _dur(right), over_stand, FINGER_CLOSED, "wait for the hold"))
    right += [
        Waypoint(1.6, side, FINGER_CLOSED, "towards the middle, staying back"),
        Waypoint(1.8, pre, FINGER_CLOSED, "in line with the screw, 25 mm back"),
        Waypoint(0.6, pre, FINGER_CLOSED, "settle"),
        Waypoint(2.5, seat, FINGER_CLOSED, "seat the bit in the screw head"),
        Waypoint(0.5, seat, FINGER_CLOSED, "motor on"),
        Waypoint(UNSCREW_S, (seat[0] - UNSCREW_BACK, ay, az), FINGER_CLOSED, "yield back as the screw comes out"),
        Waypoint(1.5, back, FINGER_CLOSED, "bit clear of the hook"),
        Waypoint(1.8, side, FINGER_CLOSED, "out to the side"),
        Waypoint(1.6, over_stand, FINGER_CLOSED, "back over the stand"),
    ]
    t_clear = _dur(right)
    right += [
        Waypoint(1.6, (GRIP_X, STAND_Y, DRV_Z + 0.002), FINGER_CLOSED, "into the rests"),
        Waypoint(0.8, (GRIP_X, STAND_Y, DRV_Z + 0.002), FINGER_OPEN, "let go"),
        Waypoint(1.2, (GRIP_X, STAND_Y, HOVER_Z), FINGER_OPEN, "up"),
        Waypoint(1.5, RIGHT_PARK, FINGER_OPEN, "park"),
    ]
    left += [
        Waypoint(t_clear - t_held, hold, FINGER_CLOSED, "keep holding while the screw comes out"),
        Waypoint(1.5, (CAP_X - 0.020, 0.0, AXIS_Z), FINGER_CLOSED, "draw the hook off the wall, screw still in it"),
        Waypoint(1.0, (CAP_X - 0.020, 0.0, CARRY_Z), FINGER_CLOSED, "lift"),
        Waypoint(2.5, (TRAY_X, TRAY_Y, CARRY_Z), FINGER_CLOSED, "over the tray"),
        Waypoint(2.0, (TRAY_X, TRAY_Y, TRAY_DROP_Z), FINGER_CLOSED, "down into the tray"),
        Waypoint(0.8, (TRAY_X, TRAY_Y, TRAY_DROP_Z), FINGER_OPEN, "let go"),
        Waypoint(1.5, (TRAY_X, TRAY_Y, HOVER_Z), FINGER_OPEN, "up"),
        Waypoint(1.5, LEFT_PARK, FINGER_OPEN, "park"),
    ]
    t_end = max(_dur(left), _dur(right)) + 10.0  # allow loose small parts to settle and satisfy the success hold
    left.append(Waypoint(t_end - _dur(left), LEFT_PARK, FINGER_OPEN, "hold"))
    right.append(Waypoint(t_end - _dur(right), RIGHT_PARK, FINGER_OPEN, "hold"))
    return {"left": left, "right": right}
