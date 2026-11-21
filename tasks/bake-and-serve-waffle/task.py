"""Rotating flip-maker with a rotation-operated lid lock.

Hardware reference: Hamilton Beach 26031G. See HARDWARE.md for the manufacturer's
operating sequence and the boundary between verified features and model inputs.
The former independently operated spring slider has been removed. A declared
joint-limit approximation represents the rotation lock; no appliance joint is
actuated by the controller. The robot closes the external handle, flips, waits,
returns upright, opens by the handle, and serves with separately grasped tongs.
"""

from __future__ import annotations

import json
import math
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from appliance_config import MODEL
from stand_scoring import stand_credit
import rotation_lock
import trace_evidence
import serving_rules
import food_contact
import hot_surface_guard
EXACT_FOOD_GUARD = True
SERVING_RULES = True
_RULE_CONTEXT = None


def _rule_context():
    # Isaac's task-file loader executes this module without registering its
    # generated name in sys.modules; ordinary Python imports do register it.
    global _RULE_CONTEXT
    if _RULE_CONTEXT is None:
        from types import SimpleNamespace
        _RULE_CONTEXT = SimpleNamespace(**globals())
    return _RULE_CONTEXT

import labkit  # noqa: E402
from labkit import Joint, on_reset, on_step, part  # noqa: E402
import hardware_limits
on_reset(hardware_limits.verify)
from scoring_geometry import hand_bounds, separation, articulated_bounds
from yam_arena.task_file import TABLE_TOP  # noqa: E402

labkit.ARTICULATION_SLEEP = 0.0  # keep the waffle maker awake (a sleeping articulation ignores joint writes)
SCENE = "station"
TOP = TABLE_TOP[SCENE]
LAYOUT = os.environ.get("WAFFLE_LAYOUT", "start")
ANIM = os.environ.get("WAFFLE_ANIM", "")
RUN_LIMIT_STEPS = 8500  # 425 s at 20 Hz, including the uncompressed 300 s bake.
SUCCESS_HOLD_S = 1.0

# -- colours (sRGB) -----------------------------------------------------------------------------------------------------
STEEL = (0.70, 0.71, 0.72)  # brushed stainless housing
STEEL_DARK = (0.55, 0.56, 0.57)  # trim rings
BLACK = (0.045, 0.045, 0.05)  # black plastic (stand, handle, hinge, cap)
GRID = (0.16, 0.16, 0.17)  # nonstick grid plates
RUBBER = (0.08, 0.08, 0.08)
LED_RED, LED_GREEN = (0.85, 0.10, 0.08), (0.10, 0.75, 0.25)
TRAY = (0.30, 0.31, 0.32)
labkit.MATTE_COLORS.update({BLACK: 0.55, RUBBER: 0.9, GRID: 0.45, TRAY: 0.6})
labkit.METAL_COLORS.update({STEEL: (0.32, 0.85), STEEL_DARK: (0.28, 0.85)})

# -- geometry (part frame: origin on the table top under the unit's centre, x away from the robots, z up) --------------
ZA = MODEL["rotation_axis_height_m"]  # rotation axis height above the table (housing sweep radius 0.123 m clears the 3.4 cm stand + tray)
RH = MODEL["housing_radius_m"]  # housing radius (22 cm housing; 18 cm modeled grid)
HH = MODEL["housing_half_height_m"]  # height of each half (closed unit 9.2 cm thick)
RP = MODEL["grid_radius_m"]  # grid plate radius
POCKET = MODEL["grid_pocket_depth_m"]  # grid pocket depth per plate (Belgian, ~1 in thick waffle)
HINGE = tuple(MODEL["lid_hinge_m"])  # lid hinge: at the top of the back of the unit
LID_OPEN = MODEL["lid_open_stop_rad"]  # rad (100 deg): hinge stop, the lid leaning just past vertical
LID_START = math.radians(MODEL["lid_start_deg"])  # the lid starts propped open at 45 deg (held by the stiff hinge)
LID_FRICTION = MODEL["lid_friction_nm"]  # nominal passive hinge resistance; native opening/closing and idle retention are verified separately
SPIN_MAX = math.radians(MODEL["rotation_stop_deg"])  # end stops at 0 and 180 deg
# One long exterior handle attached to the upper lid; used for turning and opening
HANDLE_X0, HANDLE_X1 = -0.235, -0.100
HANDLE_W, HANDLE_Z0, HANDLE_Z1 = 0.030, ZA + 0.006, ZA + 0.034
TAB_X0, TAB_X1, TAB_W, TAB_Z0, TAB_Z1 = -0.150, -0.100, 0.044, ZA + 0.008, ZA + 0.020
HANDOFF_DEG = 12.0
GRIP_X = -0.226  # distal grip location on the lid-attached handle

WX, WY = 0.40, 0.0  # world position of the unit's centre (robot base frame)
STAND_MASS, UNIT_MASS, LID_MASS = (MODEL[k] for k in ("stand_mass_kg", "lower_chamber_mass_kg", "lid_and_handle_mass_kg"))  # nominal mass partition; not measured hardware inertia


def _ring(r, z, h, w, colour, n=24, x0=0.0):
    """A ring of n boxes (radial width w, height h) of mean radius r about the vertical through (x0, 0)."""
    seg = 2 * r * math.tan(math.pi / n) + 0.001
    return [((x0 + r * math.cos(a), r * math.sin(a), z), (w, seg, h), colour, a)
            for a in (2 * math.pi * (k + 0.5) / n for k in range(n))]


def _grid(zc, sign):
    """A round Belgian grid plate: base disc, rim, and crossing ridges (pockets 2.5 cm, in four quarters). ``zc`` is
    the parting plane; ``sign`` -1 for the lower plate (ridges rise to the plane), +1 for the lid's (they hang down)."""
    pl = zc + sign * (POCKET + 0.003 + 0.002)  # plate disc centre
    rz = zc + sign * (0.003 + POCKET / 2)  # ridge centre
    boxes = _ring(RP + 0.003, rz, POCKET, 0.007, GRID, n=28)
    for k in range(-3, 4):
        u = 0.025 * k
        half = math.sqrt(max(RP ** 2 - u ** 2, 0.0))
        w = 0.008 if k == 0 else 0.0045
        boxes.append(((u, 0.0, rz), (w, 2 * half, POCKET), GRID))
        boxes.append(((0.0, u, rz), (2 * half, w, POCKET), GRID))
    cyls = [((0.0, 0.0, pl), RP + 0.006, 0.004, GRID, "Z")]
    return boxes, cyls


def _tube_mesh(r_out, r_in, z0, z1, colour, cap_at, n=72):
    """Visual-only (no collision) round housing shell: side wall r_out from z0 to z1, a full disc at ``cap_at`` (z0 or
    z1, the outside face) and an annulus r_in..r_out at the other end (the face round the grid)."""
    a = [2 * math.pi * k / n for k in range(n)]
    pts = []
    for z in (z0, z1):
        pts += [(r_out * math.cos(t), r_out * math.sin(t), z) for t in a]
    for z in (z0, z1):
        pts += [(r_in * math.cos(t), r_in * math.sin(t), z) for t in a]
    pts += [(0.0, 0.0, z0), (0.0, 0.0, z1)]
    tris = []
    for k in range(n):
        k1 = (k + 1) % n
        tris += [(k, k1, n + k1), (k, n + k1, n + k)]  # side wall
    ring_end, disc_end = (1, 0) if cap_at == z0 else (0, 1)
    o, i = ring_end * n, 2 * n + ring_end * n
    for k in range(n):
        k1 = (k + 1) % n
        tris += [(o + k, o + k1, i + k1), (o + k, i + k1, i + k)]  # annulus round the grid
    c = 4 * n + disc_end
    for k in range(n):
        tris.append((c, disc_end * n + k, disc_end * n + (k + 1) % n))  # outside disc
    return labkit.Mesh(np.asarray(pts, float), np.asarray(tris, np.int64), colour, "none")


# The housing halves collide as open shells (a solid base under each grid plate and a ring wall round it), so the
# grid's pockets are real space a waffle can fill; a visual-only tube keeps them drawn as smooth round steel.
GRID_FLOOR = POCKET + 0.007  # 2.0 cm: parting plane to the back of each grid plate
_WALL_R, _WALL_W = (0.0975 + 0.1085) / 2, 0.1085 - 0.0975  # wall ring, inside the drawn shell, outside the grid rim
# unit (base half, rotating): housing, grid, gasket, hinge brackets, rear hub, handle
_gb, _gc = _grid(ZA, -1)
UNIT_BOXES = (_gb + _ring(RH - 0.006, ZA - 0.002, 0.004, 0.012, BLACK, n=28)  # black rim round the plate
              + [((0.110, s * 0.060, ZA + 0.012), (0.020, 0.012, 0.064), BLACK) for s in (1, -1)]  # hinge brackets
              + [((0.112, 0.0, ZA - 0.006), (0.024, 0.050, 0.036), BLACK)]  # neck to the hub
              )
UNIT_BOXES = UNIT_BOXES + _ring(_WALL_R, ZA - (GRID_FLOOR + 0.004) / 2, GRID_FLOOR - 0.004, _WALL_W, STEEL, n=28)
UNIT_MESHES = [_tube_mesh(RH, 0.0965, ZA - HH, ZA - 0.002, STEEL, cap_at=ZA - HH)]
UNIT_CYLS = (_gc + [((MODEL["front_bearing_x_m"],0.,ZA+MODEL["front_bearing_z_offset_m"]),MODEL["front_bearing_radius_m"],MODEL["front_bearing_length_m"],BLACK,"X"),((0.0, 0.0, ZA - (HH + GRID_FLOOR) / 2), RH, HH - GRID_FLOOR, STEEL, "Z"),  # housing base
                    ((0.0, 0.0, ZA - HH - 0.002), 0.080, 0.004, BLACK, "Z"),  # bottom cap
                    ((0.0, 0.0, ZA - 0.030), RH + 0.002, 0.006, STEEL_DARK, "Z"),  # trim band
                    ((0.135, 0.0, ZA), 0.028, 0.034, BLACK, "X")]  # rear hub
             + [((HINGE[0], s * 0.060, HINGE[2]), 0.008, 0.012, BLACK, "Y") for s in (1, -1)])  # hinge knuckles

# lid: housing, grid (hanging down), top cap with lights, hinge arms, front tab
_lb, _lc = _grid(ZA, +1)
LID_BOXES = (_lb + _ring(RH - 0.006, ZA + 0.002, 0.004, 0.012, BLACK, n=28)
             + [((0.105, s * 0.043, ZA + HH - 0.004), (0.036, 0.016, 0.012), BLACK) for s in (1, -1)]  # hinge arms
             + [(((TAB_X0 + TAB_X1) / 2, 0.0, (TAB_Z0 + TAB_Z1) / 2), (TAB_X1 - TAB_X0, TAB_W, TAB_Z1 - TAB_Z0), BLACK),
                ])
LID_BOXES += [(((HANDLE_X0 + HANDLE_X1) / 2, 0.0, (HANDLE_Z0 + HANDLE_Z1) / 2),
               (HANDLE_X1-HANDLE_X0, HANDLE_W, HANDLE_Z1-HANDLE_Z0), BLACK)]
LID_CLOSED_HANDLE_CAP = ((HANDLE_X0+.004, 0., (HANDLE_Z0+HANDLE_Z1)/2), .016, .008, BLACK, "X")
LID_BOXES = LID_BOXES + _ring(_WALL_R, ZA + (GRID_FLOOR + 0.004) / 2, GRID_FLOOR - 0.004, _WALL_W, STEEL, n=28)
LID_MESHES = [_tube_mesh(RH, 0.0965, ZA + 0.002, ZA + HH, STEEL, cap_at=ZA + HH)]
LID_CYLS = (_lc + [LID_CLOSED_HANDLE_CAP,((0.0, 0.0, ZA + (HH + GRID_FLOOR) / 2), RH, HH - GRID_FLOOR, STEEL, "Z"),  # housing base
                   ((0.0, 0.0, ZA + 0.030), RH + 0.002, 0.006, STEEL_DARK, "Z"),
                   ((0.0, 0.0, ZA + HH + 0.003), 0.072, 0.006, BLACK, "Z"),  # top cap
                   ((-0.050, 0.012, ZA + HH + 0.0065), 0.004, 0.002, LED_RED, "Z"),
                   ((-0.050, -0.012, ZA + HH + 0.0065), 0.004, 0.002, LED_GREEN, "Z")]
            + [((HINGE[0], s * 0.043, HINGE[2]), 0.008, 0.016, BLACK, "Y") for s in (1, -1)])

# stand (heavy black base, drip tray, rear column with the bearing)
STAND_BOXES = [((MODEL["stand_base_center_x_m"], 0.0, 0.017), tuple(MODEL["stand_base_size_m"]), BLACK),  # base extends under both support columns
               ((-0.005, 0.0, 0.032), (0.200, 0.200, 0.004), TRAY),  # drip tray
               ((0.172, 0.0, (0.030 + ZA) / 2), (0.045, 0.090, ZA - 0.030), BLACK)]  # rear column
STAND_BOXES += [((MODEL["front_cradle_x_m"], 0., (.030+ZA-.040)/2), (.040,.090,ZA-.040-.030), BLACK),
                ((MODEL["front_cradle_x_m"], -(MODEL["front_cradle_inner_halfwidth_m"]+.006), ZA-.005), (.040,.012,.070), BLACK),
                ((MODEL["front_cradle_x_m"], MODEL["front_cradle_inner_halfwidth_m"]+.006, ZA-.005), (.040,.012,.070), BLACK)]
STAND_CYLS = ([((0.172, 0.0, ZA), 0.045, 0.045, BLACK, "X"),  # rounded column top / bearing housing
               ((0.154, 0.0, ZA), 0.034, 0.010, STEEL_DARK, "X")]  # bearing ring
              + [((x, y, 0.002), 0.012, 0.004, RUBBER, "Z") for x in (-0.115, 0.185) for y in (0.115, -0.115)])

_init = {"start": (0.0, LID_START), "closed": (0.0, 0.0), "half": (SPIN_MAX - 0.01, 0.0), "wide": (0.0, LID_OPEN - 0.02)}.get(LAYOUT, (0.0, LID_START))
SPIN = Joint("spin", kind="hinge", axis="X", anchor=(0.0, 0.0, ZA), lower=0.0, upper=SPIN_MAX,
             initial=_init[0], target=_init[0], stiffness=0.0, damping=0.05, friction=0.03, mass=UNIT_MASS,
             boxes=UNIT_BOXES, cylinders=UNIT_CYLS, meshes=UNIT_MESHES)
LID = Joint("lid", kind="hinge", axis="Y", anchor=HINGE, lower=0.0, upper=LID_OPEN, parent="spin",
            initial=_init[1], target=_init[1], stiffness=0.0, damping=0.03, friction=LID_FRICTION, mass=LID_MASS,
            boxes=LID_BOXES, cylinders=LID_CYLS, meshes=LID_MESHES)


def _waffle(pos):
    return part("waffle", pos=pos, mass=STAND_MASS, friction=0.9, boxes=STAND_BOXES, cylinders=STAND_CYLS,
                joints=[SPIN, LID], self_collide=True)


HOME = (WX, WY, TOP + 0.0005)

# -- the cooked waffle: a loose rigid body lying in the lower grid (not a joint: no links added to the articulation) --
# Round, 17.4 cm across, to the modeled 18 cm four-section grid: a 4 mm core resting on the lower grid's ridge tops, 16 mm square
# lugs hanging into the lower grid's pockets (0.5 mm clear of the 8 mm ridges that divide the quarters, 2.25 mm of the
# 4.5 mm ones, 1.7 mm above the plate when resting), and the same lugs on top, up to 1.3 cm above the parting plane:
# 2.7 cm thick overall; mass inherited from the nominal250mL batch. The bottom lugs key the waffle in the grid (+-0.5 mm of play), so it cannot slide out while
# the unit is upside down. The top lugs collide only on the front half (x < 0, away from the hinge), and there only
# 10 mm deep front-to-back (their sides, where the tong tips grip the waffle, are collidable): the lid's ridges come
# down on an arc that also carries them 3-4 mm towards the hinge over the last 12 mm, which would catch a full-size
# rigid lug (a real waffle is soft there); nearer the hinge the arc is flatter still, so the back half's top lugs are
# drawn only, and the closing lid meets the core there, 1.2 mm under its ridge tips.
WAFFLE_GOLD = (0.80, 0.56, 0.24)  # golden-brown lugs
WAFFLE_CRUST = (0.62, 0.38, 0.14)  # darker grooves (the core, where the grid's ridges pressed)
from cooking_state import BATTER_MASS_KG
WAFFLE_MASS = BATTER_MASS_KG
WAFFLE_R = 0.087
WAFFLE_LUG = 0.016
WAFFLE_LUG_GRIP_X = 0.010  # front top lugs: collision depth along x (drawn full size)
WAFFLE_GAP = 0.0008  # spawned this far above the ridge tops; it settles onto them
labkit.MATTE_COLORS.update({WAFFLE_GOLD: 0.7, WAFFLE_CRUST: 0.75})
_W_CORE0 = ZA - 0.003 + WAFFLE_GAP  # core underside (part frame, as spawned)
_W_CORE1 = _W_CORE0 + 0.004
_W_CELLS = [(cx, cy) for cx in (-0.0625, -0.0375, -0.0125, 0.0125, 0.0375, 0.0625)
            for cy in (-0.0625, -0.0375, -0.0125, 0.0125, 0.0375, 0.0625)
            if math.hypot(abs(cx) + 0.009, abs(cy) + 0.009) < 0.0885]  # the grid's 32 whole pockets
_W_LOW = ZA - 0.0135  # lug bottoms as spawned
_W_TOP = ZA + 0.0130 + WAFFLE_GAP


def _box_mesh(boxes, colour):
    """Visual-only triangle mesh of axis-aligned boxes (centre, size)."""
    pts, tris = [], []
    faces = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    for c, sz in boxes:
        b = len(pts)
        for dx in (-1, 1):
            for dy in (-1, 1):
                for dz in (-1, 1):
                    pts.append((c[0] + dx * sz[0] / 2, c[1] + dy * sz[1] / 2, c[2] + dz * sz[2] / 2))
        for a, b1, c1, d in faces:
            tris += [(b + a, b + b1, b + c1), (b + a, b + c1, b + d)]
    return labkit.Mesh(np.asarray(pts, float), np.asarray(tris, np.int64), colour, "none")


def _belgian_waffle(pos):
    lugs_low = [((cx, cy, (_W_LOW + _W_CORE0) / 2), (WAFFLE_LUG, WAFFLE_LUG, _W_CORE0 - _W_LOW), WAFFLE_GOLD)
                for cx, cy in _W_CELLS]
    top = _box_mesh([((cx, cy, (_W_CORE1 + _W_TOP) / 2), (WAFFLE_LUG, WAFFLE_LUG, _W_TOP - _W_CORE1)) for cx, cy in _W_CELLS],
                    WAFFLE_GOLD)
    # front top lugs: collision 10 mm deep along x, inside the drawn lug (a 0.5 mm inset hides it)
    grip = [((cx, cy, (_W_CORE1 + _W_TOP) / 2 - 0.0005), (WAFFLE_LUG_GRIP_X, WAFFLE_LUG - 0.001, _W_TOP - _W_CORE1 - 0.001),
             WAFFLE_GOLD) for cx, cy in _W_CELLS if cx < 0]
    return part("belgian_waffle", pos=pos, mass=WAFFLE_MASS, friction=0.8, boxes=lugs_low + grip,
                cylinders=[((0.0, 0.0, (_W_CORE0 + _W_CORE1) / 2), WAFFLE_R, _W_CORE1 - _W_CORE0, WAFFLE_CRUST, "Z")],
                meshes=[top])


# -- ordinary dinner plate (white glazed ceramic, 25 cm, 0.55 kg) on the existing drip tray --
PLATE_WHITE = (0.93, 0.92, 0.89)
PLATE_R, PLATE_WELL_R = 0.125, 0.097  # 25 cm across; flat well 19.4 cm (the waffle is 17.4 cm)
PLATE_FLOOR = 0.010  # well floor height above the table
PLATE_MASS = 0.55
# Loose dinner plate beside the appliance, supported directly by the table.
PLATE_POS = (WX + 0.030, WY + 0.290, TOP + 0.0005)
labkit.MATTE_COLORS.update({PLATE_WHITE: 0.25})
# drawn: a revolved profile (well, rim rising 20 deg to a rolled edge, foot ring); it collides as a foot disc, the
# well's floor disc and 36 tilted boxes along the rim, each inset 0.2-0.3 mm inside the drawn surfaces
_PLATE_PROFILE = [(0.0, 0.0102), (0.097, 0.0102), (0.100, 0.0120), (0.1235, 0.0205), (0.1255, 0.0195), (0.1235, 0.0175),
                  (0.098, 0.0085), (0.085, 0.0060), (0.085, 0.0), (0.079, 0.0), (0.079, 0.0045), (0.0, 0.0045)]
_RIM_A = math.atan2(0.0085, 0.0235)


def _revolve_mesh(profile, colour, n=72):
    """Visual-only solid of revolution about z from a closed (r, z) profile."""
    a = [2 * math.pi * k / n for k in range(n)]
    pts = [(r * math.cos(t), r * math.sin(t), z) for r, z in profile for t in a]
    m, tris = len(profile), []
    for i in range(m):
        i1 = (i + 1) % m
        for k in range(n):
            k1 = (k + 1) % n
            tris += [(i * n + k, i1 * n + k, i1 * n + k1), (i * n + k, i1 * n + k1, i * n + k1)]
    return labkit.Mesh(np.asarray(pts, float), np.asarray(tris, np.int64), colour, "none")


def _plate(pos):
    rm = (0.100 + 0.1235) / 2
    seg = 2 * rm * math.tan(math.pi / 36) + 0.001
    rim = [((rm * math.cos(t), rm * math.sin(t), 0.0150), (0.0245, seg, 0.0024), PLATE_WHITE, (t, -_RIM_A))
           for t in (2 * math.pi * (k + 0.5) / 36 for k in range(36))]
    cyls = [((0.0, 0.0, 0.0030), 0.0835, 0.0060, PLATE_WHITE, "Z"),  # foot
            ((0.0, 0.0, (0.0050 + PLATE_FLOOR) / 2), PLATE_WELL_R + 0.001, PLATE_FLOOR - 0.0050, PLATE_WHITE, "Z")]
    return part("dinner_plate", pos=pos, mass=PLATE_MASS, friction=0.5, boxes=rim, cylinders=cyls,
                meshes=[_revolve_mesh(_PLATE_PROFILE, PLATE_WHITE)])


# Kitchen tweezer-style tongs, adapted from wafer-coat's two-half passive
# spring-slide model. Robot pads squeeze the arms through physical contact.
# This is a rigid approximation to a bending kitchen tong, not a motorized jaw.
# Blade faces represent heat-resistant nonmetallic tips, protecting nonstick grids.
TONG_BLADE_L,TONG_BLADE_W,TONG_BLADE_T=.024,.024,.0015
TONG_ARM_L,TONG_ARM_W,TONG_ARM_T=.140,.012,.003
TONG_GAP=.038
_TC=math.sqrt(.5)
TONG_SEP=TONG_GAP+TONG_BLADE_T
TONG_AX=0.
TONG_AZ=TONG_BLADE_L
TONG_TOP=TONG_AZ+TONG_ARM_L
TONG_YAW=math.pi/2
TONG_HOME=(.30,.43,TOP+.022)
# Grip farther down the arms: a longer pad contact and shorter gravity lever
# resist tool rotation without closing the working tips on the food yet.
TONG_GRIP=np.array([TONG_AX,0.,TONG_TOP-.063])
TONG_WIDE,TONG_HOLD,TONG_PINCH=-.040,-.014,-.001
TONG_APPROACH=TONG_HOLD  # retain sufficient passive spring preload to hold the tilted tool


def _tong_half(sx):
    return [((TONG_AX+sx,0.,TONG_AZ+TONG_ARM_L/2),
             (TONG_ARM_T,TONG_ARM_W,TONG_ARM_L),STEEL),
            ((sx,0.,TONG_BLADE_L/2),
             (TONG_BLADE_T,TONG_BLADE_W,TONG_BLADE_L),GRID,(0.,0.))]


TONG_JAW=Joint("jaw",kind="slide",axis="X",anchor=(TONG_AX+TONG_SEP/2,0.,TONG_AZ+TONG_ARM_L/2),
    lower=-(TONG_SEP-TONG_BLADE_T-.0002),upper=0.,initial=0.,target=0.,
    stiffness=180.,damping=3.,mass=.025,boxes=_tong_half(TONG_SEP/2))
TONGS=part("kitchen_tongs",pos=TONG_HOME,yaw=TONG_YAW,mass=.035,friction=.9,
    boxes=_tong_half(-TONG_SEP/2),joints=[TONG_JAW],
    meshes=[_box_mesh([((TONG_AX,0.,TONG_TOP-.004),(TONG_SEP-TONG_ARM_T,TONG_ARM_W,.008))],STEEL)])
# Heavy, loose cup with a real floor and walls, no attachment to the tool.
_cup_x0,_cup_x1=-.050,.035
_cup_y=.016
_cw=.004
TONG_CUP=part("tong_cup",pos=(TONG_HOME[0],TONG_HOME[1],TOP),yaw=TONG_YAW,mass=.45,friction=.8,
    boxes=[(((_cup_x0+_cup_x1)/2,0.,.010),(_cup_x1-_cup_x0+2*_cw,2*_cup_y+2*_cw,.020),STEEL_DARK),
           *(( (x,0.,.038),(_cw,2*_cup_y+2*_cw,.036),STEEL_DARK) for x in (_cup_x0-_cw/2,_cup_x1+_cw/2)),
           *((((_cup_x0+_cup_x1)/2,y,.038),(_cup_x1-_cup_x0,_cw,.036),STEEL_DARK) for y in (-_cup_y-_cw/2,_cup_y+_cw/2))])
OBJECTS=[_waffle(HOME),_belgian_waffle(HOME),_plate(PLATE_POS),TONGS,TONG_CUP]

# -- scoring ------------------------------------------------------------------------------------------------------------
TURN_DONE = math.radians(MODEL["first_turn_credit_deg"])  # a half-turn counts from 173 deg
HOME_TOL = math.radians(MODEL["home_tolerance_deg"])  # back home: within 7 deg of 0
TURNING = HOME_TOL  # same opening zone in both turning and opening rules
LID_SHUT_MAX = math.radians(5.0)
STAND_OK, STAND_ZERO = MODEL["stand_full_credit_m"], MODEL["stand_zero_credit_m"]  # stand displacement: full credit < 1 cm, none from 3 cm
LID_OPEN_MIN = math.radians(35.0)
W_H1, W_H2, W_SHUT, W_STAND, W_OPEN, W_PLATE = 0.20, 0.20, 0.20, 0.10, 0.10, 0.20
FOOD_KEPT = 0.015  # the waffle counts as still in the grid while within 1.5 cm of where it lay (unit frame)
PLATE_OFF_MAX = 0.040  # waffle centre within 4 cm of the plate's axis (its edge then stays inside the 25 cm plate)
PLATE_TILT_MAX = math.radians(10.0)  # lying flat: waffle and plate both within 10 deg of level
PLATE_REST_GAP = 0.006  # resting: lug bottoms within 6 mm of the well floor (not held above it)
PLATE_STILL = 0.05  # m/s: and not moving


def new_track():
    return dict(phase=0, max_lid_turned=0.0, unsafe_turn=False, locked_at_turn=None, shut_at_turn=None, max_shift=0.0, max_spin=0.0, steps=0,
                food_left_early=False, inner_surface_contact=False, inner_contact_surface=None, inner_contact_step=None, **serving_rules.initial())


def food_pose(objects):
    """The cooked waffle's offset (m, unit frame: from where it lay in the lower grid) and tilt (rad) against the grid."""
    w, f = objects["waffle"], objects.get("belgian_waffle")
    if f is None:
        return np.zeros(3), 0.0
    Rs, Rf = w.rotation(), f.rotation()
    c = np.asarray(f.pos, float) + Rf @ np.array([0.0, 0.0, ZA])
    Ru = _Rx(w.joints.get("spin", 0.0))
    off = Ru.T @ (Rs.T @ (c - np.asarray(w.pos, float)) - np.array([0.0, 0.0, ZA]))
    zf = Ru.T @ Rs.T @ Rf @ np.array([0.0, 0.0, 1.0])
    return off, math.acos(max(-1.0, min(1.0, float(zf[2]))))


def on_plate(objects):
    """Waffle resting flat on the dinner plate: centre within 4 cm of the plate's axis, lug bottoms within 6 mm of the
    well floor, both level within 10 deg, waffle still."""
    f, pl = objects.get("belgian_waffle"), objects.get("dinner_plate")
    if f is None or pl is None:
        return False
    Rf, Rp = f.rotation(), pl.rotation()
    level = abs(Rf[2,2]) > math.cos(PLATE_TILT_MAX) and Rp[2,2] > math.cos(PLATE_TILT_MAX)
    c = np.asarray(f.pos) + Rf @ np.array([0.,0.,ZA])
    rel = Rp.T @ (c-np.asarray(pl.pos))
    pts=(serving_rules.food_points(_rule_context(),f)-pl.pos)@Rp
    gap=float(pts[:,2].min()-PLATE_FLOOR)
    inside=float(np.linalg.norm(pts[:,:2],axis=1).max())<PLATE_R-.001
    still=np.linalg.norm(f.lin_vel)<PLATE_STILL
    return bool(level and np.linalg.norm(rel[:2])<PLATE_OFF_MAX and inside
                and -.003<gap<PLATE_REST_GAP and still)


def track_update(tr, objects, hands=None, hulls=None):
    """Fold one observed state into the trial record ``tr`` (called every control step in the sim; offline tests feed
    synthetic trajectories)."""
    w = objects["waffle"]
    spin, lid = (w.joints.get(k, 0.0) for k in ("spin", "lid"))
    tr["steps"] += 1
    hit = hot_surface_guard.contact(_rule_context(), w, hulls)
    if hit and not tr["inner_surface_contact"]:
        tr.update(inner_surface_contact=True, inner_contact_surface=hit, inner_contact_step=tr["steps"])
    tr["max_spin"] = max(tr["max_spin"], spin)
    shift = float(np.linalg.norm(np.asarray(w.pos, float) - np.asarray(HOME, float)))
    tr["max_shift"] = max(tr["max_shift"], shift)
    food_offset, food_tilt = food_pose(objects)
    if phase_conversion.STATE["solid"] and tr["phase"] < 2 and (float(np.linalg.norm(food_offset)) > FOOD_KEPT or food_tilt > math.radians(15.0)):
        tr["food_left_early"] = True  # out of the grid before the cycle was done (it must ride the turn inside)
    if phase_conversion.STATE["solid"]:
        serving_rules.update(_rule_context(),tr,objects,hands,hulls)
    # The completed cycle explicitly permits opening within HOME_TOL (7 deg).
    # Apply that same home zone after return: a 5--7 deg wobble is not a new
    # flip. Any subsequent rotation outside the home zone still requires shut.
    shut_threshold = TURNING if tr['phase'] < 2 else HOME_TOL
    if spin > shut_threshold:
        if lid >= LID_SHUT_MAX:
            tr['unsafe_turn']=True
        if tr["locked_at_turn"] is None:
            tr["locked_at_turn"] = bool(lid < LID_SHUT_MAX)
            tr["shut_at_turn"] = bool(lid < LID_SHUT_MAX)
        tr["max_lid_turned"] = max(tr["max_lid_turned"], lid)
    if tr["phase"] == 0 and spin >= TURN_DONE:
        tr["phase"] = 1
    elif tr["phase"] == 1 and spin <= HOME_TOL:
        tr["phase"] = 2
    return tr


def breakdown(objects, tr=None) -> dict:
    tr = _TRACK if tr is None else tr
    w = objects["waffle"]
    spin, lid = w.joints.get("spin", 0.0), w.joints.get("lid", 0.0)
    h1 = 1.0 if tr["phase"] >= 1 else 0.0
    h2 = 1.0 if tr["phase"] >= 2 else 0.0
    shut = 1.0 if (h1 and tr["locked_at_turn"] and tr["max_lid_turned"] < LID_SHUT_MAX) else 0.0
    s = tr["max_shift"]
    stand = stand_credit(first_turn_complete=bool(h1), max_displacement_m=s)
    opened = 1.0 if (h2 and tr["shut_at_turn"] and spin <= HOME_TOL and lid >= LID_OPEN_MIN) else 0.0
    plated = 1.0 if (h2 and not tr["food_left_early"] and serving_rules.valid(tr) and on_plate(objects)) else 0.0
    total = W_H1 * h1 + W_H2 * h2 + W_SHUT * shut + W_STAND * stand + W_OPEN * opened + W_PLATE * plated
    full = bool(h1 and h2 and shut and s <= STAND_OK and opened and plated)
    disqualified = bool(tr['inner_surface_contact'] or tr['direct_food_contact'] or tr['invalid_serving'] or tr['unsafe_turn'] or tr['food_left_early'])
    if disqualified:
        total = 0.0
        full = False
    return dict(score=round(float(total), 4), full_success=full, disqualified=disqualified, half_turn_1=bool(h1), half_turn_2=bool(h2),
                locked_at_turn=tr["locked_at_turn"], unsafe_turn=tr['unsafe_turn'], max_lid_deg_while_turned=round(math.degrees(tr["max_lid_turned"]), 1),
                lid_kept_shut=bool(shut), stand_max_shift_cm=round(100 * s, 2), stand_credit=round(stand, 3), stand_score=0.0 if disqualified else round(W_STAND * stand, 4),
                stand_eligible=bool(h1),
                lid_open_at_end=bool(opened), spin_deg=round(math.degrees(spin), 1), lid_deg=round(math.degrees(lid), 1),
                max_spin_deg=round(math.degrees(tr["max_spin"]), 1), waffle_on_plate=bool(plated),
                waffle_left_early=bool(tr["food_left_early"]),
                inner_surface_contact=tr["inner_surface_contact"], inner_contact_surface=tr["inner_contact_surface"], inner_contact_step=tr["inner_contact_step"],
                **{k:tr[k] for k in serving_rules.initial()})


def score(objects):
    """Stage credit plus verified tong extraction and transfer history.

    See task.json and serving_rules.py for the explicit conditions. A normal
    reset, complete native trajectory is required to claim a demonstration.
    """
    b = breakdown(objects)
    if b["full_success"]:
        return b["score"] if final_release_held() else min(b["score"], 0.9)
    return min(b["score"], 0.99)  # rounded partial credit must never impersonate full success


def success(objects):
    """Both half-turns, lid locked and never open >= 5 deg while turned, stand moved <= 1 cm, lid open >= 35 deg at home,
    and waffle carried with kitchen tongs onto the plate, tools and hands released."""
    return breakdown(objects)["full_success"] and final_release_held()


_FINAL = {"since": None, "time": 0.0, "previous_joints": None, "released": False, "still": False}


def update_final_release(objects, hands, dt=0.05):
    """Require genuine hand withdrawal and continuous rest at the endpoint."""
    _FINAL["time"] += dt
    released = bool(hands) and all(separation(bounds,h) >= 0.005
        for spec in OBJECTS for bounds in articulated_bounds(objects[spec.name],spec) for h in hands)
    still = all(np.linalg.norm(o.lin_vel) < 0.01 and np.linalg.norm(o.ang_vel) < 0.1
                for o in objects.values())
    joints = np.array([objects["waffle"].joints.get(k, 0.) for k in ("spin", "lid")]
                      + [objects["kitchen_tongs"].joints.get("jaw",0.)])
    prev = _FINAL["previous_joints"]
    still = bool(still and prev is not None and np.all(np.abs(joints-prev)/dt < [.1,.1,.005]))
    _FINAL.update(previous_joints=joints, released=released, still=still)
    if breakdown(objects)["full_success"] and released and still:
        if _FINAL["since"] is None:
            _FINAL["since"] = _FINAL["time"]
    else:
        _FINAL["since"] = None


def final_release_held():
    return bool(_FINAL["since"] is not None and _FINAL["released"] and _FINAL["still"]
                and _FINAL["time"]-_FINAL["since"] >= SUCCESS_HOLD_S)


def distance(objects):
    """Remaining joint travel (rad) to finish: turn to 180, back to 0, then open the lid."""
    w = objects["waffle"]
    spin, lid = w.joints.get("spin", 0.0), w.joints.get("lid", 0.0)
    ph = _TRACK["phase"]
    rest = max(0.0, LID_OPEN_MIN - lid)
    if ph == 0:
        return float(lid + (SPIN_MAX - spin) + SPIN_MAX + LID_OPEN_MIN)
    if ph == 1:
        return float(spin + rest)
    return float(spin + rest)


# -- Robot-only demonstration: steady the base, close the lid handle, turn,
# wait for baking, return upright, open upper handle and serve with tongs.
class _Orient(float):
    def __new__(cls, R):
        v = float.__new__(cls, 0.0)
        v.rot = R
        return v


def _Rx(a):
    c, s_ = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s_], [0.0, s_, c]])


def _Ry(a):
    c, s_ = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s_], [0.0, 1.0, 0.0], [-s_, 0.0, c]])


def _Rz(a):
    c, s_ = math.cos(a), math.sin(a)
    return np.array([[c, -s_, 0.0], [s_, c, 0.0], [0.0, 0.0, 1.0]])


DEMO_T = dict(
    tab_x=-0.128,  # pressing point on the external upper handle
    turn_end=188.0,  # robot wrist overtravel compensates grip compliance; appliance stop remains 180 degrees
    turn_back=2.0,
    step_deg=10.0,
    step_s=0.30,
    ease_steps=3,  # the last steps into each end are slowed down (decelerate the heavy unit)
    press_in=0.004,
    steady_x=-0.118,  # stand press point (part frame x): the base front-left corner, clear of the unit's sweep  # stand press: pad tips commanded this far into the base top
    open_to=95.0,  # deg; larger angle reduces closing torque after release
    side=0.07,  # m: the right hand slides onto the handle end from this far to its right (-y)

)


def demo_plan(t=DEMO_T):
    from yam_arena import ik
    from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN, LEFT_BASE_POS, READY, RIGHT_BASE_POS
    from yam_arena.demo import DOWN_GRIP, Waypoint
    from yam_arena.urdf import quat_xyzw_to_matrix

    DOWN = quat_xyzw_to_matrix(np.asarray(DOWN_GRIP))
    FWD = _Ry(-math.pi / 2) @ DOWN @ _Rz(-math.pi / 2)  # approach +x, fingers closing along z (top and bottom of the
    # handle), wrist at -90 deg roll: the half-turn rolls the wrist from -90 to +90 deg (with the fingers across y it
    # would have to go from 0 to 180, past the wrist's 120 deg limit). The hand slides onto the handle from the side.
    SIDE = _Rx(-math.pi / 4) @ DOWN  # pointing down and towards -y (left arm, from outside the sweep)
    bases = {"left": LEFT_BASE_POS, "right": RIGHT_BASE_POS}
    W = np.array([HOME[0], HOME[1], TOP])
    AX = W + np.array([0.0, 0.0, ZA])
    Hh = W + np.array(HINGE)
    plans = {"left": [], "right": []}
    clock = {"left": 0.0, "right": 0.0}
    fo, fc = FINGER_OPEN, FINGER_CLOSED

    def wp(arm, d, p, f, label, R):
        plans[arm].append(Waypoint(float(d), tuple(float(v) for v in p), float(f), label, _Orient(np.asarray(R))))
        clock[arm] += d

    def wait_until(arm, tt, label="wait"):
        last = plans[arm][-1] if plans[arm] else None
        if last is None:
            pos, R = ik.tcp_pose(np.asarray(READY, float), np.asarray(bases[arm], float))
            p, f, Rm = pos, FINGER_OPEN, R
        else:
            p, f, Rm = last.pos, last.finger, last.yaw.rot
        if tt > clock[arm]:
            wp(arm, tt - clock[arm], p, f, label, Rm)

    C0 = W + np.array([-.226, .000, (HANDLE_Z0 + HANDLE_Z1) / 2])  # grip on the free span of the single handle, ahead of the front cradle

    opening_grip = W+np.array([-.135,.000,(HANDLE_Z0+HANDLE_Z1)/2])

    def tab(th_deg, opening=False):
        R = _Ry(math.radians(th_deg))
        Rg = R @ _Rx(-math.pi/2) @ DOWN @ _Rz(math.pi)  # side approach, jaws pinch the upper/lower faces
        if opening:
            tilt = -60.0 * min(1., max(0., (th_deg - 60.) / 30.))
            Rg = Rg @ _Ry(math.radians(tilt))
        return Hh + R @ ((opening_grip if opening else C0) - Hh), Rg, -(Rg @ np.array([0.0, 0.0, 1.0]))

    L, Rr = "left", "right"
    # Close using only the external handle.
    # 0. the right hand takes the handle first and steadies the base while the lid handle is lowered (the
    #    unit spins freely on its bearing, and the push on the tab would otherwise turn it outside the opening zone)
    G = W + np.array([GRIP_X, 0.0, (HANDLE_Z0 + HANDLE_Z1) / 2])
    side = np.array([0.0, -t["side"], 0.0])
    right_steady_R = _Rx(math.pi/4) @ DOWN
    right_axis = right_steady_R[:,2]
    right_steady = W+np.array([t["steady_x"],-.118,.030-t["press_in"]])-PAD_TIP*right_axis
    wp(Rr,1.5,right_steady-.10*right_axis,fc,"steady: above right base edge",right_steady_R)
    wp(Rr,1.,right_steady,fc,"steady: hold base during lid closure",right_steady_R)
    th0 = math.degrees(LID_START)
    p, Rg, n = tab(th0)
    wait_until(L,clock[Rr],"close: wait for base hold")
    wp(L,2.0,p+.07*n,-.020,"close: beside upper handle",Rg)
    wp(L,1.0,p,-.020,"close: straddle upper handle",Rg)
    wp(L,.5,p,fc,"close: pinch upper handle",Rg)
    th=th0
    while th > -1.+1e-6:
        th=max(th-3.,-1.)
        p,Rg,n=tab(th)
        wp(L,.4,p,fc,"close: lower external handle along hinge",Rg)
    wp(L,.5,p,fc,"close: hold shut",Rg)
    wp(L,.5,p,fo*.7,"close: release upper handle",Rg)
    wp(L,1.,p+.08*n,fo*.7,"close: withdraw sideways",Rg)
    # 2. steady the stand (left) --------------------------------------------------------------------------------
    a = SIDE @ np.array([0.0, 0.0, 1.0])  # approach axis
    # the front-left corner of the base: beyond the housing (x < -0.11) and 12 cm off the axis, outside the sweep
    tips = W + np.array([t["steady_x"], 0.118, 0.030 - t["press_in"]])
    P = tips - PAD_TIP * a
    wp(L, 2.0, P - 0.10 * a + np.array([0.0, 0.05, 0.05]), fc, "steady: beside the stand", SIDE)
    wp(L, 1.0, P - 0.03 * a, fc, "steady: above the base edge", SIDE)
    wp(L, 0.8, P, fc, "steady: press the base", SIDE)
    t_steady = clock[L]
    wait_until(Rr,t_steady,"steady: hold through handle closure")
    wp(Rr,.8,right_steady-.08*right_axis,fc,"steady: release base",right_steady_R)
    wp(Rr,1.,G+side+[0.,0.,.08],fo,"turn: above lid handle",FWD)
    wp(Rr,.8,G+side,fo,"turn: beside lid handle",FWD)
    wp(Rr,.8,G,fo,"turn: straddle lid handle",FWD)
    wp(Rr,.5,G,fc,"turn: grip lid handle",FWD)
    # 3. turn (right) -------------------------------------------------------------------------------------------
    # (the right hand takes the closed lid handle; from in front, x 0.12, is out of reach for this wrist
    # pose, so it came down onto the handle end from above)
    wait_until(Rr, max(clock[Rr], t_steady) + 0.3, "turn: wait for the stand hold")

    def roll(a0, a1, label):
        n_steps = max(1, int(math.ceil(abs(a1 - a0) / t["step_deg"])))
        for k in range(1, n_steps + 1):
            ang = a0 + (a1 - a0) * k / n_steps
            slow = 1.0 + 1.5 * max(0, k - (n_steps - t["ease_steps"])) / t["ease_steps"]
            A = math.radians(ang)
            wp(Rr, t["step_s"] * slow, AX + _Rx(A) @ (G - AX), fc, label, _Rx(A) @ FWD)

    roll(0.0, t["turn_end"], "turn: over (clockwise)")
    wp(Rr, 1.0, plans[Rr][-1].pos, fc, "turn: hold upside down", plans[Rr][-1].yaw.rot)
    roll(t["turn_end"], t["turn_back"], "turn: back")
    wp(Rr, 0.6, plans[Rr][-1].pos, fc, "turn: settle", plans[Rr][-1].yaw.rot)
    # Keep a continuous handle grip through the handoff. A free interval can
    # let the chamber roll and re-engage its rotation lock before opening.
    # Lift from the unobstructed distal grip before the proximal handoff.
    # With the lid closed, the lower bearing leaves only 6 mm below the
    # proximal handle: too little room for the second gripper's lower pad.
    handoff_deg = HANDOFF_DEG
    for th in (3., 6., 9., handoff_deg):
        lift_R = _Ry(math.radians(th))
        wp(Rr,.5,Hh+lift_R@(G-Hh),fc,"open: distal handle initial lift",lift_R@FWD)
    t_turned=clock[Rr]
    wait_until(L,t_turned,"steady: hold")
    wp(L,.8,P-.08*a,fc,"steady: lift off",SIDE)
    wp(L,1.,P-.14*a+np.array([0.,.03,.08]),fc,"handoff: clear housing before reorienting",SIDE)
    p,Rg,n=tab(handoff_deg,opening=True)
    wp(L,1.2,p+.12*n+np.array([0.,0.,.05]),-.020,"handoff: reorient outside housing",Rg)
    wp(L,1.2,p+.07*n,-.020,"open: approach proximal handle",Rg)
    wp(L,1.,p,-.020,"open: straddle proximal handle",Rg)
    wp(L,.5,p,fc,"open: secure handle before handoff",Rg)
    wait_until(Rr,clock[L]+.3,"open: retain distal grip until handoff")
    lift_R=_Ry(math.radians(handoff_deg))
    handoff_tip=Hh+lift_R@(G-Hh)
    wp(Rr,.5,handoff_tip,fo,"open: release distal handle",lift_R@FWD)
    wp(Rr,.8,handoff_tip+side,fo,"open: clear handle sideways",lift_R@FWD)
    wp(Rr,1.,handoff_tip+side+[0.,0.,.08],fo,"open: withdraw from lid path",FWD)
    wait_until(L,clock[Rr],"open: hold proximal grip while other hand withdraws")
    wp(Rr,1.2,right_steady-.08*right_axis,fc,"steady: approach right base",right_steady_R)
    wp(Rr,.8,right_steady,fc,"steady: hold base for opening and serving",right_steady_R)
    th = handoff_deg
    while th < t["open_to"] - 1e-6:
        th = min(th + 4.0, t["open_to"])
        p, Rg, n = tab(th, opening=True)
        wp(L, 0.45 if th < 16 else 0.35, p, fc, "open: lift along the hinge arc", Rg)
    t_open = clock[L]
    wait_until(Rr, t_open, "steady: hold base during opening")
    wait_until(L, max(clock[L], clock[Rr]) + 0.2, "open: hold")
    wp(L,.8,p,fo*.7,"open: release the tab",Rg)
    wp(L,1.,p+.10*n,fo*.7,"open: clear the lid fully sideways",Rg)
    # 6. Two-stage squeeze from wafer-coat: hold the tong arms with the
    # tips open, then squeeze further to loosen the top lugs of the waffle.
    # Tilt about the jaw-closing axis: faces remain vertical while the wrist
    # stays within reach of the central top lug.
    from yam_arena import demo as dm
    # Equivalent 180-degree jaw orientation avoids the left wrist limit.
    cup_R=_Rz(TONG_YAW)@dm.grip_rotation(math.pi/2)@_Rz(math.pi)
    tool_R=_Rz(TONG_YAW)
    tong_R=_Ry(-math.pi/4)@cup_R
    def tong_tcp(origin,tilt=False):
        A=_Ry(-math.pi/4) if tilt else np.eye(3)
        return np.asarray(origin)+A@tool_R@TONG_GRIP-.007*(A@cup_R)[:,2]
    home=np.asarray(TONG_HOME)
    wp(L,1.5,tong_tcp(home+[0.,-.080,.080]),TONG_WIDE,"tongs: reorient clear of tool",cup_R)
    wp(L,1.2,tong_tcp(home+[0.,0.,.080]),TONG_WIDE,"tongs: above entire tool",cup_R)
    wp(L,.4,tong_tcp(home+[0.,0.,.080]),TONG_WIDE,"tongs: settle above tool",cup_R)
    wp(L,2.0,tong_tcp(home),TONG_WIDE,"tongs: straddle arms",cup_R)
    wp(L,1.2,tong_tcp(home),TONG_HOLD,"tongs: hold with tips open",cup_R)
    wp(L,.4,tong_tcp(home),TONG_HOLD,"tongs: settle grip before lifting",cup_R)
    wp(L,1.5,tong_tcp(home+[0.,0.,.045]),TONG_HOLD,"tongs: lift from cup",cup_R)
    # The 24 mm square face is a diamond in the tilted XZ plane: its lowest
    # corner is 8.49 mm below the tool origin. Keep that corner 1.5 mm above core.
    pick=W+np.array([-.004,-.0125,_W_CORE1+.010])
    wp(L,4.,tong_tcp(pick+[0.,.16,.050],True),TONG_HOLD,"tongs: clear side rim",tong_R)
    wp(L,1.2,tong_tcp(pick+[0.,0.,.050],True),TONG_HOLD,"tongs: above front lugs",tong_R)
    wp(L,1.5,tong_tcp(pick+[0.,0.,.050],True),TONG_APPROACH,"tongs: align above lugs",tong_R)
    wp(L,1.5,tong_tcp(pick+[0.,0.,.020],True),TONG_APPROACH,"tongs: approach lugs",tong_R)
    wp(L,1.2,tong_tcp(pick,True),TONG_APPROACH,"tongs: tips around lugs",tong_R)
    wp(L,2.,tong_tcp(pick,True),TONG_APPROACH,"tongs: align tips",tong_R)
    wp(L,1.2,tong_tcp(pick,True),TONG_PINCH,"loosen: squeeze tong arms",tong_R)
    wp(L,2.,tong_tcp(pick+[0.,0.,.036],True),TONG_PINCH,"loosen: lift lugs free",tong_R)
    # Carry the food with the closed passive tongs to the side plate.
    drop=np.asarray(PLATE_POS)+[-.004,-.0125,PLATE_FLOOR-_W_LOW+_W_CORE1+.010]
    wp(L,1.5,tong_tcp(pick+[0.,0.,.070],True),TONG_PINCH,"transfer: clear grid",tong_R)
    wp(L,3.,tong_tcp([drop[0],drop[1],pick[2]+.070],True),TONG_PINCH,"transfer: above side plate",tong_R)
    wp(L,2.5,tong_tcp(drop+[0.,0.,.035],True),TONG_PINCH,"transfer: lower toward plate",tong_R)
    wp(L,1.5,tong_tcp(drop,True),TONG_PINCH,"transfer: set on plate",tong_R)
    wp(L,.8,tong_tcp(drop,True),TONG_APPROACH,"transfer: release food",tong_R)
    wp(L,1.5,tong_tcp(drop+[0.,0.,.060],True),TONG_HOLD,"transfer: withdraw",tong_R)
    wp(L,3.1,tong_tcp(home+[0.,0.,.045]),TONG_HOLD,"tongs: return above cup",cup_R)
    wp(L,2.,tong_tcp(home+[0.,0.,.045]),TONG_HOLD,"tongs: align over cup",cup_R)
    wp(L,1.5,tong_tcp(home),TONG_HOLD,"tongs: lower into cup",cup_R)
    wp(L,.7,tong_tcp(home),TONG_WIDE,"tongs: release into cup",cup_R)
    wp(L,1.,tong_tcp(home+[-.08,0.,.055]),TONG_WIDE,"tongs: clear cup",cup_R)
    wait_until(Rr,clock[L]+.2,"steady: wait for tool stow")
    wp(Rr,1.,right_steady-.10*right_axis,fc,"steady: release stand and withdraw",right_steady_R)
    for arm in (L, Rr):
        pr, Rp = ik.tcp_pose(np.asarray(READY, float), np.asarray(bases[arm], float))
        wp(arm, 2.0, pr, fo, "park", Rp)
    end = max(clock[L], clock[Rr])
    for arm in (L, Rr):
        wait_until(arm, end + 1.5, "done")
    return plans


PAD_TIP = 0.013  # pad tips past the TCP along the approach axis


def _park_plan():
    """Development previews (WAFFLE_LAYOUT / WAFFLE_ANIM set): both arms hold READY, clear of the waffle maker."""
    from yam_arena import ik
    from yam_arena.constants import FINGER_OPEN, LEFT_BASE_POS, READY, RIGHT_BASE_POS
    from yam_arena.demo import Waypoint
    plans = {}
    for arm, base in (("left", LEFT_BASE_POS), ("right", RIGHT_BASE_POS)):
        p, R = ik.tcp_pose(np.asarray(READY, float), np.asarray(base, float))
        plans[arm] = [Waypoint(float(RUN_LIMIT_STEPS) / 20.0, tuple(float(v) for v in p), float(FINGER_OPEN), "park",
                               _Orient(np.asarray(R)))]
    return plans


def demo():
    from yam_arena import demo as dm

    old = dm.grip_rotation
    if not getattr(old, "_wflip", False):
        def grip_rotation(x=0.0):
            return x.rot if hasattr(x, "rot") else old(x)
        grip_rotation._wflip = True
        dm.grip_rotation = grip_rotation
    if ANIM or "WAFFLE_LAYOUT" in os.environ:
        return _park_plan()
    return demo_plan()


# -- trial record and development hooks -----------------------------------------------------------------------------------
_TRACK = new_track()
_S = {"step": 0}
SETTLE_STEPS = 3


def _states(env):
    return {o.name: labkit._part_state(env, o) for o in OBJECTS}


def _write_joint(env, name, q, qd):
    import torch
    a = env.unwrapped.scene["waffle"]
    i = list(a.joint_names).index(f"{name}_joint")
    dev = env.unwrapped.scene.env_origins.device
    a.write_joint_position_to_sim_index(position=torch.tensor([[q]], device=dev), joint_ids=[i])
    a.write_joint_velocity_to_sim_index(velocity=torch.tensor([[qd]], device=dev), joint_ids=[i])


def _report(env, tag):
    st = _states(env)
    b = breakdown(st)
    b.update(score=score(st), full_success=success(st), hands_released=_FINAL["released"], final_still=_FINAL["still"])
    w = st["waffle"]
    b["stand_pos"] = np.round(w.pos, 4).tolist()
    if "belgian_waffle" in st:  # the cooked waffle's centre and tilt in the turning unit's frame (mm from where it lay)
        off, tilt = food_pose(st)
        b["food_mm"] = np.round(1000 * off, 1).tolist()
        b["food_tilt_deg"] = round(math.degrees(tilt), 2)
    if "dinner_plate" in st:  # waffle centre relative to the plate (mm, plate frame, z from the well floor to lug bottoms)
        f, pl = st["belgian_waffle"], st["dinner_plate"]
        c = np.asarray(f.pos, float) + f.rotation() @ np.array([0.0, 0.0, ZA])
        rel = pl.rotation().T @ (c - np.asarray(pl.pos, float)) - np.array([0.0, 0.0, PLATE_FLOOR + ZA - _W_LOW])
        b["plate_mm"] = np.round(1000 * rel, 1).tolist()
    print("WAFFLE " + json.dumps(dict(step=_S["step"], layout=LAYOUT, anim=ANIM, phase=tag, **b)), flush=True)


@on_reset
def _on_reset(env):
    _S["step"] = 0
    _TRACK.clear()
    _TRACK.update(new_track())
    _FINAL.update(since=None, time=0.0, previous_joints=None, released=False, still=False)


@on_step
def _on_step(env):
    _S["step"] += 1
    s = _S["step"]
    if s <= SETTLE_STEPS:  # hold the starting pose while the scene settles (the reset left the lid falling)
        _write_joint(env, "lid", _init[1], 0.0)
        _write_joint(env, "spin", _init[0], 0.0)
    states = _states(env)
    hands=hand_bounds(env)
    track_update(_TRACK, states, hands, food_contact.hand_hulls(env))
    update_final_release(states, hands, float(getattr(env.unwrapped, "step_dt", .05)))
    trace_evidence.record(env,s,states,score(states),success(states),breakdown(states))
    if s in (1, 2, 5) or s % 5 == 0:
        _report(env, f"step{s}")


rotation_lock.configure(globals())

import tool_alignment
tool_alignment.configure(globals())
import serving_alignment
serving_alignment.configure(globals())
import handle_alignment
handle_alignment.configure(globals())

# Explicit abstract material response, with a full 300 s simulation-time bake.
import phase_conversion
import batter_fluid
from cooking_state import TIME_SCALE, RECIPE_S
DEBUG_TIME_COMPRESSED = TIME_SCALE > 1
batter_fluid.configure(globals())
_cooking_base_demo=demo
def demo():
    from yam_arena.demo import Waypoint
    plans=_cooking_base_demo()
    extra_hold=RECIPE_S/TIME_SCALE  # full requested interval at the inverted hold
    for arm,plan in plans.items():
        if arm=='left':
            i=next(i for i,w in enumerate(plan) if w.label=='steady: hold')
            w=plan[i];plan[i]=Waypoint(w.duration+extra_hold,w.pos,w.finger,w.label,w.yaw)
        else:
            i=next(i for i,w in enumerate(plan) if w.label=='turn: hold upside down')
            w=plan[i];plan.insert(i+1,Waypoint(extra_hold,w.pos,w.finger,'cook: heat while rotation-locked',w.yaw))
    return plans
_manipulation_breakdown=breakdown
def breakdown(objects,tr=None):
    b=_manipulation_breakdown(objects,tr)
    completed=bool(phase_conversion.STATE['solid'] and b['full_success'])
    if not phase_conversion.STATE['solid']:
        b['score']=min(b['score'],.8)
    b.update(full_success=completed,debug_time_compressed=DEBUG_TIME_COMPRESSED,solid_material_active=phase_conversion.STATE['solid'],
             phase_block_reason=phase_conversion.STATE['block_reason'],
             debug_sequence_complete=completed,
             benchmark_eligible=not DEBUG_TIME_COMPRESSED)
    return b
def score(objects):
    if DEBUG_TIME_COMPRESSED:return 0.
    b=breakdown(objects)
    return b['score'] if b['full_success'] and final_release_held() else min(b['score'],.9)
def success(objects):
    return bool(not DEBUG_TIME_COMPRESSED and breakdown(objects)['full_success'] and final_release_held())

# Clip the complete cooking plan, including its waiting interval, only after
# all plan wrappers and reset callbacks have been installed.
import segment_resume
segment_resume.configure(globals())
