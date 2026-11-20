"""On the station table: a closed white foam (EPS) hinged takeout container, 23.5 x 16 x 7.5 cm (modelled on the Dart
205HT1 "Performer" 9 x 6 in container), with leftover fries and chicken nuggets lying on a sheet of red-and-white
checkered deli paper inside it. Two waste openings are cut into the table top, each behind a small sign: in the
overhead (top_cam) view the LEFT hole is LANDFILL (dark sign, trash-bag icon) and the RIGHT hole is COMPOST (green
sign, leaf icon); a hidden catch bin under each hole collects what goes through.

Sort the leftovers: open the container (lift the lid's front tab off the knuckle on the base's front tab, then lift the
lid), put the food through the COMPOST hole, and the deli paper and the empty container through the LANDFILL hole,
without dropping food on the table. The food is not pre-separated from the paper: it lies on it. Any method is allowed
(pick the food off, lift the paper and pour, or use both arms together).

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py -E object_jitter=0 -E control_hz=20 \\
           --instruction "<see task.json>" --scorer task_score ...
Layout variant: TRAY_SORT_VARIANT=<int> (default 0, the canonical layout in the setup image). Development only:
TRAY_SORT_LAYOUT=start|open|correct|dump|pour|half|spill spawns a scored end state (scorer sanity check);
TRAY_SORT_ANIM=open swings the lid open by joint writes (scene preview footage only, not part of the task).
"""

from __future__ import annotations

import json
import math
import os
import pathlib
import random
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import labkit  # noqa: E402
from labkit import Joint, on_reset, on_step, part  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

labkit.ARTICULATION_SLEEP = 0.0  # keep the clamshell and the light liner awake
# solver guards: a 4 g fry squeezed between paper cells and a finger pad must not be shot across the room
labkit.MAX_LIN_VEL, labkit.MAX_ANG_VEL, labkit.LINEAR_DAMPING = 4.0, 2000.0, 0.05
SCENE = "station"
TOP = TABLE_TOP[SCENE]
VARIANT = int(os.environ.get("TRAY_SORT_VARIANT", "0"))
LAYOUT = os.environ.get("TRAY_SORT_LAYOUT", "start")
RUN_LIMIT_STEPS = 3000  # 150 s at 20 Hz
SUCCESS_HOLD_S = float(os.environ.get("TRAY_SORT_HOLD", "4.0"))  # allow the placement to settle and the grippers to withdraw

# -- colours (sRGB) -----------------------------------------------------------------------------------------------------
FOAM = (0.925, 0.915, 0.865)  # matte off-white EPS
FOAM_SHADE = (0.885, 0.875, 0.825)  # bevels and ribs, a touch darker so the moulding reads
CHECK_RED = (0.78, 0.10, 0.11)  # red-and-white checkered deli paper
CHECK_WHITE = (0.97, 0.96, 0.92)
FRY = (0.96, 0.76, 0.30)
NUGGET = (0.78, 0.50, 0.20)
NUGGET_DARK = (0.66, 0.40, 0.15)
GREEN = (0.16, 0.52, 0.22)
TRASH = (0.20, 0.21, 0.22)
labkit.MATTE_COLORS.update({FOAM: 0.95, FOAM_SHADE: 0.95, CHECK_RED: 0.75, CHECK_WHITE: 0.75, FRY: 0.8, NUGGET: 0.85, NUGGET_DARK: 0.85})

# -- the container: Dart 205HT1 "Performer" foam hinged container (9.3 x 6.4 x 2.9 in) ----------------------------------
# Part frame: origin under the centre of the base, x towards the hinge (away from the robots), y along the long side.
# Closure (the 205HT1's single "knuckle on tab" lock): a flat tab sticks out of the base's front lip and carries a
# rounded knuckle; the lid's front lip has a matching tab with a slot that the knuckle pokes up through when closed.
# Lid top is recessed (raised perimeter band) for stacking; the base stands on a platform; the hinge is perforated.
D, WD = 0.162, 0.235  # outer depth (x) and width (y) at the rim
HB, HL = 0.044, 0.030  # base depth, lid height (7.4 cm closed)
T = 0.004  # foam wall
R = 0.024  # corner radius (plan view, at the rim)
TAPER = math.radians(8.0)  # walls lean out towards the rim
LIP = 0.007  # flat lip round base and lid
LIP_T = 0.003
TAB_W = 0.044  # front tabs: width (y)
TAB_LB, TAB_LL = 0.014, 0.034  # reach beyond the lip (x): base tab, and the longer lid tab that overhangs it by 2 cm
TAB_L = TAB_LB
SLOT_W, SLOT_L = 0.016, 0.011  # slot in the lid tab, over the knuckle
KNUCKLE_R, KNUCKLE_H = 0.0045, 0.006  # knuckle on the base tab (a rounded nub, 9 mm across, 6 mm tall)
rng0 = random.Random(1000 + VARIANT)
BX, BY = 0.24, 0.0
BOX_DX, BOX_DY = (0.0, 0.0) if VARIANT == 0 else (rng0.uniform(-0.01, 0.01), rng0.uniform(-0.01, 0.01))
LID_OPEN = 1.80  # rad (103 deg): fully opened, the lid leans just past vertical


def _outline(hx, hy, r, n_arc=3):
    """Segments (mid x, mid y, outward normal angle, length) round a rounded rectangle of half sizes hx, hy."""
    segs = [(hx, 0.0, 0.0, 2 * (hy - r)), (0.0, hy, math.pi / 2, 2 * (hx - r)),
            (-hx, 0.0, math.pi, 2 * (hy - r)), (0.0, -hy, -math.pi / 2, 2 * (hx - r))]
    for cx, cy, a0 in ((hx - r, hy - r, 0.0), (-hx + r, hy - r, math.pi / 2), (-hx + r, -hy + r, math.pi),
                       (hx - r, -hy + r, -math.pi / 2)):
        for k in range(n_arc):
            a = a0 + (k + 0.5) * (math.pi / 2) / n_arc
            chord = 2 * r * math.sin((math.pi / 2) / n_arc / 2)
            rm = r * math.cos((math.pi / 2) / n_arc / 2)
            segs.append((cx + rm * math.cos(a), cy + rm * math.sin(a), a, chord))
    return segs


def _walls(z0, h, hx_mid, hy_mid, r_mid, lean, colour):
    """Tapered walls between heights z0 and z0 + h; hx_mid/hy_mid are outer half sizes at mid-height; ``lean`` > 0
    leans the tops outwards."""
    out = []
    for mx, my, a, ln in _outline(hx_mid, hy_mid, r_mid):
        cx, cy = mx - T / 2 * math.cos(a), my - T / 2 * math.sin(a)
        out.append(((cx, cy, z0 + h / 2), (T, ln + 0.002, h / math.cos(lean)), colour, (a, lean)))
    return out


def _lip(z, hx, hy, r, colour, w=LIP, t=LIP_T):
    return [((mx, my, z), (w, ln + 0.002, t), colour, a) for mx, my, a, ln in _outline(hx + w / 2, hy + w / 2, r + w / 2)]


def _slab(z, hx, hy, r, t, colour):
    """A flat rounded-rectangle plate (two crossing boxes and four corner discs)."""
    out = [((0.0, 0.0, z), (2 * hx, 2 * (hy - r), t), colour), ((0.0, 0.0, z), (2 * (hx - r), 2 * hy, t), colour)]
    cyl = [((sx * (hx - r), sy * (hy - r), z), r, t, colour) for sx in (1, -1) for sy in (1, -1)]
    return out, cyl


dz = HB * math.tan(TAPER)
HX_RIM, HY_RIM = D / 2, WD / 2
FLOOR_BOX, FLOOR_CYL = _slab(T / 2, HX_RIM - dz, HY_RIM - dz, R - dz / 2, T, FOAM)
_PLAT_BOX, _PLAT_CYL = _slab(-0.001, HX_RIM - dz - 0.010, HY_RIM - dz - 0.010, R * 0.6, 0.002, FOAM_SHADE)  # platform
TAB_X0 = -HX_RIM - LIP  # front edge of the lip
TAB_ZB = HB - LIP_T / 2  # base tab mid-plane (flush with the base lip)
TAB_ZL = HB + 0.0004 + LIP_T / 2  # lid tab mid-plane (flush with the lid lip, lying on the base tab)
KNUCKLE_X = TAB_X0 - TAB_LB / 2
BASE_BOXES = (FLOOR_BOX + _PLAT_BOX
              + _walls(0.0, HB, HX_RIM - dz / 2, HY_RIM - dz / 2, R - dz / 4, TAPER, FOAM)
              + _lip(HB - LIP_T / 2, HX_RIM, HY_RIM, R, FOAM)
              + [((TAB_X0 - TAB_L / 2, 0.0, TAB_ZB), (TAB_L, TAB_W, LIP_T), FOAM)]  # base front tab
              # perforated hinge: a row of short dark slits along the back lip
              + [((HX_RIM + LIP * 0.6, y, HB - LIP_T / 2 + 0.0004), (0.0015, 0.008, LIP_T), FOAM_SHADE)
                 for y in np.arange(-HY_RIM + R + 0.006, HY_RIM - R - 0.004, 0.014)])
BASE_CYLS = (FLOOR_CYL + _PLAT_CYL
             + [((KNUCKLE_X, 0.0, HB + KNUCKLE_H / 2 - 0.0015), KNUCKLE_R, KNUCKLE_H, FOAM_SHADE)])  # the knuckle

LZ0 = HB + 0.0004
ldz = HL * math.tan(TAPER)
LID_TOP_Z = LZ0 + HL
_lt_box, _lt_cyl = _slab(LID_TOP_Z - T / 2, HX_RIM - ldz, HY_RIM - ldz, R - ldz / 2, T, FOAM)
BAND_W = 0.014  # raised perimeter band round the recessed lid top
BAND = [((mx, my, LID_TOP_Z + 0.0015), (BAND_W, ln + 0.002, 0.003), FOAM, a)
        for mx, my, a, ln in _outline(HX_RIM - ldz - BAND_W / 2, HY_RIM - ldz - BAND_W / 2, R - ldz / 2 - BAND_W / 2)]
BEVEL = [((mx, my, LID_TOP_Z + 0.0012), (0.006, ln + 0.002, 0.0022), FOAM_SHADE, (a, math.radians(35)))
         for mx, my, a, ln in _outline(HX_RIM - ldz - BAND_W - 0.002, HY_RIM - ldz - BAND_W - 0.002,
                                       R - ldz / 2 - BAND_W - 0.002, 2)]
RIBS = [((x, 0.0, LID_TOP_Z + 0.0005), (0.003, WD - 2 * ldz - 2 * BAND_W - 0.05, 0.001), FOAM_SHADE)
        for x in (-0.036, -0.012, 0.012, 0.036)]
_s0, _s1 = KNUCKLE_X - SLOT_L / 2, KNUCKLE_X + SLOT_L / 2  # lid tab: a frame round the slot over the knuckle
_t0 = TAB_X0 - TAB_LL
LID_TAB = [((TAB_X0 - TAB_LL / 2, (TAB_W + SLOT_W) / 4, TAB_ZL), (TAB_LL, (TAB_W - SLOT_W) / 2, LIP_T), FOAM),
           ((TAB_X0 - TAB_LL / 2, -(TAB_W + SLOT_W) / 4, TAB_ZL), (TAB_LL, (TAB_W - SLOT_W) / 2, LIP_T), FOAM),
           (((_s1 + TAB_X0) / 2, 0.0, TAB_ZL), (TAB_X0 - _s1, SLOT_W, LIP_T), FOAM),
           (((_t0 + _s0) / 2, 0.0, TAB_ZL), (_s0 - _t0, SLOT_W, LIP_T), FOAM)]
TAB_GRIP_X = TAB_X0 - TAB_LB - (TAB_LL - TAB_LB) / 2  # the middle of the lid tab's overhang (box frame)
LID_BOXES = (_lip(LZ0 + LIP_T / 2, HX_RIM, HY_RIM, R, FOAM)
             + _walls(LZ0, HL, HX_RIM - ldz / 2, HY_RIM - ldz / 2, R - ldz / 4, -TAPER, FOAM)
             + _lt_box + BAND + BEVEL + RIBS + LID_TAB)
LID = Joint(
    "lid", kind="hinge", axis="Y", anchor=(HX_RIM + LIP, 0.0, HB), lower=0.0, upper=LID_OPEN,
    initial=LID_OPEN - 0.05 if LAYOUT == "open" else 0.0, target=0.0,
    stiffness=0.0, damping=0.002, friction=0.012, mass=0.008,  # a stiff foam hinge: the lid stays where it is left
    boxes=LID_BOXES, cylinders=_lt_cyl,
)
BOX_MASS = 0.014  # base 14 g + lid 8 g: a foam 9 x 6 in container weighs about 20 g


def _clamshell(pos, yaw=0.0):
    return part("clamshell", pos=pos, yaw=yaw, mass=BOX_MASS, friction=0.6, boxes=BASE_BOXES, cylinders=BASE_CYLS,
                joints=[LID])


FLOOR_TOP_LOCAL = T

# -- the liner: red-and-white checkered deli paper, 24 x 17 cm, 0.6 mm, as a 9 x 7 grid of hinged cells -----------
# A spine of 9 cells (3 cm, along y) bends about x; from every spine cell a rib of 3 cells (2.7 cm) each way along x
# bends about y. So the sheet bends in both directions (a tree: 63 links, one articulation; neighbouring ribs are not
# tied to each other, so the sheet can split along the rib lines when pulled hard). Rest shape: all four edges draped up
# the box walls, plus small seeded waves (pre-creased paper).
NS, NR = 4, 3  # spine cells each side of the centre, rib cells each side of the spine
SP, RB = 0.0265, 0.0235  # cell size along y (spine) and x (ribs): crease lines stay 1 cm inside the walls
LT = 0.0006
CELL_MASS = 0.00015  # 0.15 g per cell, 9.5 g sheet (real 30 g/m2 deli paper: 1.2 g; raised for solver stability)
K_SPINE, K_RIB, D_PAPER = 0.003, 0.0006, 0.00005  # N m/rad per crease; damping N m s/rad
DRAPE = 1.2  # rad: the outer cell of every spine and rib starts folded up the wall (at the two short ends the
# corner rib cells fold inwards instead, like the crumpled corners of a real sheet)
_wave = random.Random(500 + VARIANT)


def _cell_boxes(cx, cy, sx, sy, gi, gj):
    """One cell drawn as 2 x 2 checks; global check parity from the cell's grid index."""
    out = []
    for a in (0, 1):
        for b in (0, 1):
            red = (2 * gi + a + 2 * gj + b) % 2 == 0
            out.append(((cx + (a - 0.5) * sx / 2, cy + (b - 0.5) * sy / 2, 0.0), (sx / 2 - 0.0001, sy / 2 - 0.0001, LT),
                        CHECK_RED if red else CHECK_WHITE))
    return out


def _liner_parts():
    """(main body boxes, joints): spine joints about x, rib joints about y, rest angles = drape + waves."""
    joints = []

    def ribs(spine_name, yc, gi):
        for side in (1, -1):
            prev = spine_name
            for k in range(NR):
                x0 = side * (SP / 2 + k * RB)
                q = (-side * DRAPE) if k == NR - 1 else _wave.uniform(-0.06, 0.06)  # +x end lifts with a negative angle
                name = f"r{gi:+d}{'p' if side > 0 else 'n'}{k}".replace("+", "p").replace("-", "n")
                joints.append(Joint(name, kind="hinge", axis="Y", anchor=(x0, yc, 0.0), parent=prev,
                                    boxes=_cell_boxes(x0 + side * RB / 2, yc, RB, SP, side * (k + 1), gi),
                                    mass=CELL_MASS, lower=-2.0, upper=2.0, initial=q, target=q,
                                    stiffness=K_RIB, damping=D_PAPER))
                prev = name

    ribs("", 0.0, 0)
    for side in (1, -1):
        prev = ""
        for i in range(NS):
            y0 = side * (SP / 2 + i * SP)
            q = (side * DRAPE) if i == NS - 1 else _wave.uniform(-0.05, 0.05)  # about +x, +y end lifts positive
            name = f"s{'p' if side > 0 else 'n'}{i}"
            joints.append(Joint(name, kind="hinge", axis="X", anchor=(0.0, y0, 0.0), parent=prev,
                                boxes=_cell_boxes(0.0, y0 + side * SP / 2, SP, SP, 0, side * (i + 1)),
                                mass=CELL_MASS, lower=-2.0, upper=2.0, initial=q, target=q,
                                stiffness=K_SPINE, damping=D_PAPER))
            prev = name
            ribs(name, y0 + side * SP / 2, side * (i + 1))
    return _cell_boxes(0.0, 0.0, SP, SP, 0, 0), joints


_LB, _LJ = _liner_parts()
assert 1 + len(_LJ) <= 64, "PhysX GPU articulations take at most 64 links"


def _liner(pos, yaw=0.0):
    return part("liner", pos=pos, yaw=yaw, mass=CELL_MASS, friction=0.6, boxes=_LB, joints=_LJ)


FLAT_HALF_Y = SP / 2 + (NS - 1) * SP  # the flat part of the sheet: +-9.3 cm along y
FLAT_HALF_X = SP / 2 + (NR - 1) * RB  # and +-6.0 cm along x

# -- food: 16 fries (9 x 9 x 65 mm, 4 g) and 3 nuggets (about 46 x 34 x 17 mm, 16 g): 112 g ---------------------------
FRY_SIZE, FRY_MASS = (0.065, 0.009, 0.009), 0.004
# fry friction (PhysX averages it with the other surface: 0.5 on the paper or foam). Was 0.7, which kept the last fry
# stuck in the crease by the near wall at the reachable ~92 deg tip (demo8-18); slightly oily fries slide more easily.
FRY_MU = float(os.environ.get("TRAY_SORT_FRY_MU", "0.7"))  # 0.4 tried: demo19 0.60, demo20 0.55
N_FRIES, N_NUGGETS, NUGGET_MASS = 16, 3, 0.016
NUGGET_FOOT = (0.046, 0.034)


def _obb_corners(cx, cy, yaw, lx, ly):
    c, s = math.cos(yaw), math.sin(yaw)
    return [(cx + c * a - s * b, cy + s * a + c * b) for a, b in
            ((lx / 2, ly / 2), (-lx / 2, ly / 2), (-lx / 2, -ly / 2), (lx / 2, -ly / 2))]


def _sat_overlap(p, q, gap=0.0015):
    for poly in (p, q):
        for i in range(4):
            ax, ay = poly[i]
            bx, by = poly[(i + 1) % 4]
            nx, ny = -(by - ay), bx - ax
            n = math.hypot(nx, ny)
            nx, ny = nx / n, ny / n
            pa = [nx * x + ny * y for x, y in p]
            qa = [nx * x + ny * y for x, y in q]
            if max(pa) + gap < min(qa) or max(qa) + gap < min(pa):
                return False
    return True


def _layout(seed):
    """Food on the flat part of the liner, in layers: [(kind, x, y, z above the paper, yaw)] in the box frame."""
    rng = random.Random(seed)
    hx, hy = FLAT_HALF_X - 0.005, FLAT_HALF_Y - 0.005
    layers: dict[int, list] = {k: [] for k in range(4)}
    out = []

    def inside(poly):
        return all(abs(x) <= hx and abs(y) <= hy for x, y in poly)

    for _ in range(N_NUGGETS):
        for _try in range(400):
            x, y, yaw = rng.uniform(-hx, hx), rng.uniform(-hy, hy), rng.uniform(0, math.pi)
            poly = _obb_corners(x, y, yaw, *NUGGET_FOOT)
            if inside(poly) and not any(_sat_overlap(poly, o) for o in layers[0]):
                layers[0].append(poly)
                layers[1].append(poly)
                out.append(("nugget", x, y, 0.0085, yaw))
                break
    fz = [0.0045 + k * 0.0095 for k in range(4)]
    for _ in range(N_FRIES):
        placed = False
        for k in range(4):
            for _try in range(300):
                x, y, yaw = rng.uniform(-hx, hx), rng.uniform(-hy, hy), rng.uniform(0, math.pi)
                poly = _obb_corners(x, y, yaw, FRY_SIZE[0], FRY_SIZE[1])
                if inside(poly) and not any(_sat_overlap(poly, o) for o in layers[k]):
                    layers[k].append(poly)
                    out.append(("fry", x, y, fz[k], yaw))
                    placed = True
                    break
            if placed:
                break
        if not placed:
            raise RuntimeError("could not lay out the fries")
    return out


FOOD_LAYOUT = _layout(VARIANT)
FOOD_NAMES = [f"nugget{i}" for i in range(N_NUGGETS)] + [f"fry{i:02d}" for i in range(N_FRIES)]
FOOD_MASS = {n: (NUGGET_MASS if n.startswith("nugget") else FRY_MASS) for n in FOOD_NAMES}
TOTAL_FOOD = sum(FOOD_MASS.values())


def _food_part(name, pos, yaw):
    if name.startswith("nugget"):  # a lumpy breaded piece: a slab, a rounded end and a bump on top
        return part(name, pos=pos, yaw=yaw, mass=NUGGET_MASS, friction=0.8,
                    boxes=[((0.004, 0.0, 0.0), (0.032, 0.030, 0.015), NUGGET),
                           ((0.002, 0.003, 0.0065), (0.020, 0.018, 0.004), NUGGET_DARK)],
                    cylinders=[((-0.012, -0.001, 0.0), 0.016, 0.017, NUGGET)])
    return part(name, pos=pos, yaw=yaw, mass=FRY_MASS, friction=FRY_MU, boxes=[((0.0, 0.0, 0.0), FRY_SIZE, FRY)])


# -- waste holes: two openings cut through the table top, each with a hidden catch bin under it -----------------------
# Sides are defined in the TOP_CAM IMAGE (the station's default overhead view, behind the robots looking at the table):
# image LEFT hole = LANDFILL (liner and foam container, dark sign), image RIGHT hole = COMPOST (food, green sign).
# In the robot frame: landfill at y = +0.31 (left arm's side, seen by left_cam), compost at y = -0.31 (right arm, right_cam).
# ONE-LINE SWAP: LANDFILL_Y_SIGN = -1 (or env TRAY_SORT_LANDFILL_Y_SIGN=-1) mirrors it.
LANDFILL_Y_SIGN = int(os.environ.get("TRAY_SORT_LANDFILL_Y_SIGN", "1"))
HOLE_X, HOLE_DY = 0.32, 0.31
HOLE_SX, HOLE_SY = 0.27, 0.29  # x by y: the container (24.9 cm wide; 25 cm front tab to open lid) fits through as it stands
TABLE_T = 0.06  # station table thickness
CATCH_SX, CATCH_SY, CATCH_H, CATCH_T = 0.33, 0.35, 0.25, 0.005  # catch bins: inner size, depth, wall
CATCH_TOP = TOP - TABLE_T - 0.004  # just under the table
CATCH_FLOOR = CATCH_TOP - CATCH_H
BINS = {  # name (scorer label): (centre x, centre y, catch inner x, catch inner y, colour)
    "compost_bin": (HOLE_X, -LANDFILL_Y_SIGN * HOLE_DY, CATCH_SX, CATCH_SY, GREEN),
    "trash_bin": (HOLE_X, LANDFILL_Y_SIGN * HOLE_DY, CATCH_SX, CATCH_SY, TRASH),
}
labkit.TABLE_CUTS = [(HOLE_X, HOLE_DY, HOLE_SX, HOLE_SY), (HOLE_X, -HOLE_DY, HOLE_SX, HOLE_SY)]


def _catch(name, cx, cy, sx, sy, colour):
    ox, oy, zc = sx / 2 + CATCH_T / 2, sy / 2 + CATCH_T / 2, CATCH_H / 2
    return part(name, pos=(cx, cy, CATCH_FLOOR - CATCH_T), fixed=True, friction=0.6, boxes=[
        ((0.0, 0.0, CATCH_T / 2), (sx + 2 * CATCH_T, sy + 2 * CATCH_T, CATCH_T), colour),
        ((ox, 0.0, zc), (CATCH_T, sy + 2 * CATCH_T, CATCH_H), colour),
        ((-ox, 0.0, zc), (CATCH_T, sy + 2 * CATCH_T, CATCH_H), colour),
        ((0.0, oy, zc), (sx, CATCH_T, CATCH_H), colour),
        ((0.0, -oy, zc), (sx, CATCH_T, CATCH_H), colour),
    ])


# Signs: US waste-station style, a 17 x 8.5 cm panel on a foot, standing 2.5 cm beyond the far edge of each hole (5 cm
# outboard), facing the robots and cameras, leaned back 30 deg (printed on both faces). About 9 cm tall.
SIGN_W, SIGN_H, SIGN_T, SIGN_GAP = 0.17, 0.085, 0.006, 0.025
SIGN_TILT = math.radians(30)  # leaned back (top away from the robots/top_cam) for legibility from the top camera
SIGN_X = HOLE_X + HOLE_SX / 2 + labkit.TABLE_TRIM["width"] + SIGN_GAP
SIGN_DY = HOLE_DY - 0.07       # well inboard of the READY grippers so top_cam sees the whole sign
SIGN_DARK, SIGN_GREEN, SIGN_FOOT = (0.15, 0.15, 0.16), (0.13, 0.47, 0.23), (0.25, 0.25, 0.26)
_ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")


def _sign(name, cy, colour, png):
    zc = 0.006 + 0.008 + SIGN_H / 2 * math.cos(SIGN_TILT) + SIGN_T / 2 * math.sin(SIGN_TILT)
    pv = (0.0, 0.0, zc)
    labkit.TEXTURES[name] = [((SIGN_T / 2 + 0.0004, 0.0, zc), SIGN_W - 0.004, SIGN_H - 0.004, 0.0, png, SIGN_TILT, pv),
                             ((-SIGN_T / 2 - 0.0004, 0.0, zc), SIGN_W - 0.004, SIGN_H - 0.004, math.pi, png, SIGN_TILT, pv)]
    return part(name, pos=(SIGN_X, cy, TOP), fixed=True, friction=0.6, boxes=[
        ((0.0, 0.0, 0.003), (0.05, SIGN_W * 0.8, 0.006), SIGN_FOOT),  # foot plate
        ((0.0, 0.0, 0.010), (0.012, SIGN_W * 0.5, 0.008), SIGN_FOOT),  # clamp
        ((0.0, 0.0, zc), (SIGN_T, SIGN_W, SIGN_H), colour, (0.0, SIGN_TILT)),  # panel, leaned back
        ((0.022, 0.0, 0.006 + 0.022), (0.004, SIGN_W * 0.4, 0.044), SIGN_FOOT, (0.0, math.radians(-25))),  # rear strut
    ])


BIN_PARTS = ([_catch(n, *v) for n, v in BINS.items()]
             + [_sign("sign_compost", -LANDFILL_Y_SIGN * SIGN_DY, SIGN_GREEN, os.path.join(_ASSETS, "sign_compost.png")),
                _sign("sign_landfill", LANDFILL_Y_SIGN * SIGN_DY, SIGN_DARK, os.path.join(_ASSETS, "sign_landfill.png"))])


# -- layouts: the task's start state, and scored end states for the scorer sanity check --------------------------------
def _bin_stack(bin_name, names, z0=None, dz_layer=0.022):
    """Spawn positions for food dropped through a hole: 6 per layer on a 3 x 2 grid (16 x 10 cm), starting 4.5 cm below
    the table top inside the hole, so nothing spawns near the grippers hanging above the holes."""
    cx, cy = BINS[bin_name][:2]
    z0 = TOP - 0.045 if z0 is None else z0
    out = {}
    for k, n in enumerate(names):
        layer, c = divmod(k, 6)
        out[n] = ((cx + 0.08 * (c % 3 - 1), cy + (0.05 if c < 3 else -0.05), z0 + dz_layer * layer), math.pi / 2)
    return out


BOX_HOME = (BX + BOX_DX, BY + BOX_DY, TOP + 0.0022)  # origin under the floor; the platform rests on the table
LINER_HOME_Z = BOX_HOME[2] + FLOOR_TOP_LOCAL + LT / 2 + 0.0003
LINER_TOP = LINER_HOME_Z + LT / 2


def _food_in_box():
    out = {}
    names = iter(FOOD_NAMES)
    nug = [f for f in FOOD_LAYOUT if f[0] == "nugget"]
    fry = [f for f in FOOD_LAYOUT if f[0] == "fry"]
    for kind_list in (nug, fry):
        for kind, x, y, z, yaw in kind_list:
            out[next(names)] = ((BOX_HOME[0] + x, BOX_HOME[1] + y, LINER_TOP + z + 0.0005), yaw)
    return out


def _build(layout):
    box_pos, box_yaw = BOX_HOME, 0.0
    liner_pos, liner_yaw = (BOX_HOME[0], BOX_HOME[1], LINER_HOME_Z), 0.0
    food = _food_in_box()
    cb, tb = BINS["compost_bin"], BINS["trash_bin"]
    in_trash_box = ((tb[0], tb[1], TOP - 0.005), 0.0)  # closed box in the landfill hole's mouth, long side x
    if layout == "correct":  # food through compost; the liner, then the box, through landfill
        food = _bin_stack("compost_bin", FOOD_NAMES)
        liner_pos, liner_yaw = (tb[0], tb[1], TOP - 0.03), 0.0
        box_pos, box_yaw = (tb[0], tb[1], TOP + 0.01), 0.0
    elif layout == "dump":  # everything into compost: food, then the liner, then the closed box on top
        food = _bin_stack("compost_bin", FOOD_NAMES)
        liner_pos, liner_yaw = (cb[0], cb[1], TOP + 0.05), 0.0
        box_pos, box_yaw = (cb[0], cb[1], TOP + 0.08), 0.0
    elif layout == "pour":  # liner poured into compost with the food; empty box correctly into landfill
        food = _bin_stack("compost_bin", FOOD_NAMES)
        liner_pos, liner_yaw = (cb[0], cb[1], TOP + 0.05), 0.0
        (box_pos, box_yaw) = in_trash_box
    elif layout == "half":  # first half of the food (incl. the nuggets) through compost, rest still in the box
        half = FOOD_NAMES[: len(FOOD_NAMES) // 2]
        food.update(_bin_stack("compost_bin", half))
    elif layout == "spill":  # correct, but 3 fries on the table in front of the box spot
        food = _bin_stack("compost_bin", FOOD_NAMES)
        for k, n in enumerate([n for n in FOOD_NAMES if n.startswith("fry")][:3]):
            food[n] = ((0.10, -0.06 + 0.06 * k, TOP + 0.006), math.pi / 2)
        liner_pos, liner_yaw = (tb[0], tb[1], TOP - 0.03), 0.0
        box_pos, box_yaw = (tb[0], tb[1], TOP + 0.01), 0.0
    objs = BIN_PARTS + [_clamshell(box_pos, box_yaw), _liner(liner_pos, liner_yaw)]
    objs += [_food_part(n, *food[n]) for n in FOOD_NAMES]
    return objs


OBJECTS = _build(LAYOUT)

# -- scoring ------------------------------------------------------------------------------------------------------------
W_FOOD, W_LINER, W_BOX, W_SPILL = 0.50, 0.20, 0.15, 0.15
SPILL_ZERO_G = 0.020  # 20 g of food on the table/floor (5 fries) loses all the no-spill credit
TABLE_BAND = 0.03  # resting with its centre < 3 cm above the table top, outside the box and bins = spilled
MAX_FOOD_IN_TRASH = 0.1 * 0.112  # packaging credit allows at most 11.2 g of food in landfill


def in_bin(p, name) -> bool:
    """Check that the food center is inside the catch bin below the chute."""
    cx, cy, sx, sy, _ = BINS[name]
    x, y, z = (float(v) for v in p)
    return abs(x - cx) <= sx / 2 and abs(y - cy) <= sy / 2 and CATCH_FLOOR <= z <= CATCH_TOP


def which_bin(p):
    for name in BINS:
        if in_bin(p, name):
            return name
    return None


def in_box(box_state, p) -> bool:
    """Inside the clamshell's base (lid open or shut): over its floor and below its closed top."""
    lx, ly, lz = box_state.to_local(p)
    return abs(lx) <= D / 2 and abs(ly) <= WD / 2 and 0.0 <= lz <= HB + HL


def _section_shapes(boxes, cylinders):
    """Store the collision shapes and their orientations in the zero pose."""
    box_centers, box_axes, box_half = [], [], []
    for center, size, _, (yaw, pitch) in boxes:
        cosine, sine = math.cos(yaw), math.sin(yaw)
        turn = np.array([[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]])
        box_centers.append(center)
        box_axes.append(turn @ _Ry(pitch))
        box_half.append(np.asarray(size) / 2)
    cylinder_centers = [item[0] for item in cylinders]
    cylinder_radii = [item[1] for item in cylinders]
    cylinder_half_lengths = [item[2] / 2 for item in cylinders]
    cylinder_axes = [np.eye(3)["XYZ".index(item[4])] for item in cylinders]
    return tuple(np.asarray(v, dtype=float) for v in
                 (box_centers, box_axes, box_half, cylinder_centers,
                  cylinder_radii, cylinder_half_lengths, cylinder_axes))


def _relative_packaging_shapes():
    # Called lazily once, after the rotation helpers have been defined.
    packaging = {}
    for part in OBJECTS:
        if part.name in ("clamshell", "liner"):
            packaging[part.name] = (
                _section_shapes(part.boxes, part.cylinders),
                [(j, np.asarray(j.anchor), _section_shapes(j.boxes, j.cylinders)) for j in part.joints],
            )
    return packaging


_PACKAGING = None


def _world_shape_bounds(shapes, rotation, origin):
    centers, axes, half, cylinder_centers, radii, half_lengths, cylinder_axes = shapes
    if len(centers):
        middle = centers @ rotation.T + origin
        extent = (np.abs(rotation @ axes) @ half[..., None])[..., 0]
        yield middle - extent, middle + extent
    if len(cylinder_centers):
        middle = cylinder_centers @ rotation.T + origin
        direction = np.abs(cylinder_axes @ rotation.T)
        extent = half_lengths[:, None] * direction + radii[:, None] * np.sqrt(np.maximum(0, 1 - direction**2))
        yield middle - extent, middle + extent


def _packaging_sections(name, state):
    """Use each box and cylinder, with the measured articulation angles."""
    root, links = _PACKAGING[name]
    rotation = state.rotation()
    origin = np.asarray(state.pos)
    transform = {"": (rotation, origin)}
    yield from _world_shape_bounds(root, rotation, origin)
    for joint, anchor, shapes in links:
        parent_rotation, parent_origin = transform[joint.parent]
        angle = state.joints[joint.name]
        local_rotation = _Rx(angle) if joint.axis == "X" else _Ry(angle)
        rotation = parent_rotation @ local_rotation
        origin = parent_origin + (parent_rotation - rotation) @ anchor
        transform[joint.name] = (rotation, origin)
        yield from _world_shape_bounds(shapes, rotation, origin)


def _packaging_destination(name, state):
    """Find a whole object inside a catch bin and detect compost contamination."""
    global _PACKAGING
    if _PACKAGING is None:
        _PACKAGING = _relative_packaging_shapes()
    links = _PACKAGING[name][1]
    if not np.isfinite(state.pos).all() or any(j.name not in state.joints for j, _, _ in links):
        return None, False
    lower, upper = zip(*_packaging_sections(name, state))
    low = np.concatenate(lower)
    high = np.concatenate(upper)
    if not (np.isfinite(low).all() and np.isfinite(high).all()):
        return None, False
    tolerance = 0.005  # allow catch wall and floor contact within 5 mm
    destination = None
    contaminates_compost = False
    for bin_name, (cx, cy, sx, sy, _) in BINS.items():
        bin_low = np.array([cx - sx / 2, cy - sy / 2, CATCH_FLOOR])
        bin_high = np.array([cx + sx / 2, cy + sy / 2, CATCH_TOP])
        complete = (
            np.all(low >= bin_low - tolerance)
            and np.all(high <= bin_high + tolerance)
        )
        if complete:
            destination = bin_name
        if bin_name == "compost_bin":
            # Test each section separately so the liner's folds follow its
            # measured shape. Ignore wall contact within the same tolerance.
            contaminates_compost = bool(complete) or bool(np.any(np.all(
                (high > bin_low + tolerance) & (low < bin_high - tolerance), axis=1
            )))
    return destination, contaminates_compost


def breakdown(objects) -> dict:
    box = objects["clamshell"]
    food_in, spilled, misplaced, in_box_mass, transit = 0.0, 0.0, 0.0, 0.0, 0.0
    left_in_box = 0
    for name, m in FOOD_MASS.items():
        p = np.asarray(objects[name].pos, float)
        b = which_bin(p)
        inside = in_box(box, p)
        left_in_box += inside
        if b == "compost_bin":
            food_in += m
        elif b is not None:
            misplaced += m
        elif inside:
            in_box_mass += m
        elif p[2] < TOP + TABLE_BAND:
            spilled += m
        else:
            transit += m
    liner_bin, liner_in_compost = _packaging_destination("liner", objects["liner"])
    box_bin, box_in_compost = _packaging_destination("clamshell", box)
    contaminated = liner_in_compost or box_in_compost
    box_empty = left_in_box == 0
    f = food_in / TOTAL_FOOD
    landfill_eligible = misplaced <= MAX_FOOD_IN_TRASH
    liner = 1.0 if liner_bin == "trash_bin" and landfill_eligible else 0.0
    boxc = 1.0 if box_bin == "trash_bin" and box_empty and landfill_eligible else 0.0
    cleanup = f * max(0.0, 1.0 - spilled / SPILL_ZERO_G)
    compost_credit = 0.0 if contaminated else W_FOOD * f + W_SPILL * cleanup
    total = compost_credit + W_LINER * liner + W_BOX * boxc
    full = not contaminated and left_in_box == 0 and f >= 1.0 - 1e-9 and liner == boxc == 1.0
    return dict(score=round(total, 4), full_success=bool(full), food_frac=round(f, 4),
                food_in_compost_g=round(food_in * 1000, 1), spilled_g=round(spilled * 1000, 1),
                food_in_trash_g=round(misplaced * 1000, 1), still_in_box_g=round(in_box_mass * 1000, 1),
                in_transit_g=round(transit * 1000, 1), liner_bin=liner_bin, box_bin=box_bin, box_empty=bool(box_empty),
                compost_packaging_contamination=contaminated)


def score(objects):
    """Score the clean compost stream and correctly deposited packaging."""
    return breakdown(objects)["score"]


def success(objects):
    """All food in clean compost and the whole empty box and liner in landfill."""
    return breakdown(objects)["full_success"]


def distance(objects):
    """Mean distance (m) of the food pieces from the compost hole's centre."""
    cx, cy = BINS["compost_bin"][:2]
    return float(np.mean([math.hypot(objects[n].pos[0] - cx, objects[n].pos[1] - cy) for n in FOOD_MASS]))


# -- demo: scripted bimanual solution (open loop, played by yam_scripted) ---------------------------------------------
# 1. Lid (landfill-side arm): pinch the lid tab's overhang between the finger pads, lift it along the hinge arc while
#    turning the grip with it, let go, close the fingers under the tab and push the lid on past vertical.
# 2. Pour (compost-side arm): pinch the near side wall of the base, with the deli paper's draped end trapped between
#    the inner pad and the wall, slide the box to the compost hole's edge and tip it so the food slides off the paper
#    and out over the far rim into the hole; tip back and let go. Then lift the liner's near-wall flap over the rim
#    so the crease flattens, tip once more so any fry left in that crease can slide out, and let go.
# 3. Dispose (landfill-side arm): pinch the other side wall, carry the box (paper inside) over the landfill hole, lower
#    it into the opening and let go.
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


DEMO_T = dict(
    wall_grip_h=0.026,  # pinch height above the box origin (the draped paper reaches ~2.6 cm up the wall)
    wall_open=0.42,  # finger opening for the wall pinch (fraction of fully open): pads ~3.8 cm apart
    tab_open=0.65,  # opening to straddle the 44 mm tab
    tab_pinch=0.40,  # light pinch on the tab edges (pads 3.8 cm apart, 3 mm into the 44 mm tab): the tab pivots
    lid_pitch_k=0.6,  # while lifting, the grip pitches forward by this fraction of the lid angle (free: the tab pivots)
    push_pitch=50.0,  # grip pitched forward (deg) while pushing the lid past vertical
    lift_deg=55.0,  # lid lifted by the tab pinch up to here, then pushed on to push_deg
    push_deg=float(os.environ.get("TRAY_SORT_PUSH_DEG", "102")),  # the hinge stops at 103 deg: pushing further drives the fingers over the lid top, and they drag it shut on the way back
    tip_deg=92.0,  # tip of the box over the compost hole's edge: the near wall must point downhill for the food to leave
    tip_pre=20.0,  # the front-wall grip is pre-rolled about x by this much (deg): keeps the wrist in reach to ~100 deg
    grip_dy=0.055,  # the front-wall pinch sits this far from the box centre, towards landfill (clear of the 44 mm tab)  # the box is tipped this far over the compost hole's edge (food slides off the paper from ~35 deg)
    edge_gap=0.004,  # the box's base edge is slid to this far short of the hole's trim
)


_DISPOSE = {}


def _wall_y(h):
    """Half-width of the base to the middle of its side wall at height h above the box origin."""
    return HY_RIM - (HB - h) * math.tan(TAPER) - T / 2


def demo_plan(t=DEMO_T):
    from yam_arena import ik
    from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN, LEFT_BASE_POS, READY, RIGHT_BASE_POS
    from yam_arena.demo import DOWN_GRIP, Waypoint
    from yam_arena.urdf import quat_xyzw_to_matrix

    DOWN = quat_xyzw_to_matrix(np.asarray(DOWN_GRIP))
    bases = {"left": LEFT_BASE_POS, "right": RIGHT_BASE_POS}
    s_c = 1.0 if BINS["compost_bin"][1] > 0 else -1.0  # side of the compost hole
    arm_c = "left" if s_c > 0 else "right"  # the compost-side arm tips the box over the hole edge
    arm_l = "right" if s_c > 0 else "left"  # the other one opens the lid and throws the box away
    bx, by, bz = BOX_HOME
    plans = {arm_c: [], arm_l: []}
    clock = {arm_c: 0.0, arm_l: 0.0}

    def wp(arm, d, p, f, label, R=DOWN):
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

    fo = FINGER_OPEN
    # 1. lid -------------------------------------------------------------------------------------------------------
    H = np.array([bx + HX_RIM + LIP, by, bz + HB])
    C0 = np.array([bx + TAB_X0 - TAB_LL + 0.007, by, bz + TAB_ZL])  # 7 mm in from the lid tab's front edge
    A = arm_l
    wp(A, 2.5, C0 + (0, 0, 0.08), fo * t["tab_open"], "lid: above the tab")
    wp(A, 1.5, C0, fo * t["tab_open"], "lid: straddle the tab")
    wp(A, 0.8, C0, fo * t["tab_pinch"], "lid: pinch the tab")
    th = 0.0
    while th < t["lift_deg"] - 1e-6:
        th = min(th + 5.0, t["lift_deg"])
        R = _Ry(math.radians(th))
        wp(A, 0.35, H + R @ (C0 - H), fo * t["tab_pinch"], "lid: lift along the hinge arc",
           _Ry(-math.radians(th * t["lid_pitch_k"])) @ DOWN)
    R = _Ry(math.radians(th))
    n = R @ np.array([0.0, 0.0, -1.0])  # the tab's underside normal
    Rg = _Ry(-math.radians(th * t["lid_pitch_k"])) @ DOWN
    p_rel = H + R @ (C0 - H)
    wp(A, 0.5, p_rel, fo * t["tab_open"], "lid: let go", Rg)
    wp(A, 0.6, p_rel + 0.022 * n, fo * t["tab_open"], "lid: under the tab", Rg)
    wp(A, 0.4, p_rel + 0.022 * n, FINGER_CLOSED, "lid: close under the tab", Rg)
    th = t["lift_deg"] - 12.0  # the pinch slips a little: start the push below where the lid should be
    while th < t["push_deg"] - 1e-6:
        th = min(th + 5.0, t["push_deg"])
        R = _Ry(math.radians(th))
        nn = R @ np.array([0.0, 0.0, -1.0])
        k = (th - t["lift_deg"]) / (t["push_deg"] - t["lift_deg"])
        a = (1 - k) * t["lift_deg"] * t["lid_pitch_k"] + k * t["push_pitch"]
        wp(A, 0.3, H + R @ (C0 - H) + 0.016 * nn, FINGER_CLOSED, "lid: push past vertical", _Ry(-math.radians(a)) @ DOWN)
    # back straight off the lid (towards the robots) before turning the wrist: parking from beside the lid with the
    # fingers pitched forward hooks the lid and pulls it shut again
    last = plans[A][-1]
    # (fingers stay shut: opening them here straddles the lid's free edge and drags it closed)
    # first pull the fingers straight back out along their own axis (their tips sit by the tab's slot), then back off
    ax = np.asarray(last.yaw.rot)[:, 2]  # the approach axis (TCP offset direction)
    p_out = np.asarray(last.pos) - 0.06 * ax
    wp(A, 0.8, p_out, FINGER_CLOSED, "lid: pull out", last.yaw.rot)
    wp(A, 0.8, p_out + (-0.08, 0.0, 0.0), FINGER_CLOSED, "lid: back off", last.yaw.rot)
    wp(A, 1.4, (0.12, -s_c * 0.22, TOP + 0.21), FINGER_CLOSED, "lid: clear of the lid")
    park_l = ik.tcp_pose(np.asarray(READY, float), np.asarray(bases[A], float))[0]
    wp(A, 1.6, park_l, fo, "park", ik.tcp_pose(np.asarray(READY, float), np.asarray(bases[A], float))[1])

    # 2. pour: slide the box to the compost hole's edge and tip it over that edge -------------------------------------
    # The compost-side arm pinches the front wall (with the paper) on the landfill half, pads closing along x, so its
    # forearm stays in front of the box while the box turns its open side away over the hole; the table carries the
    # box at the pivot the whole time, so it cannot swing in the one-wall pinch.
    P = arm_c
    wait_until(P, clock[A])
    hw = _wall_y(t["wall_grip_h"])
    fx = HX_RIM - (HB - t["wall_grip_h"]) * math.tan(TAPER) - T / 2  # front wall's middle at the grip height
    Rz90 = np.array([[0.0, s_c, 0.0], [-s_c, 0.0, 0.0], [0.0, 0.0, 1.0]])  # yaw -s_c*90 deg: pads close along x
    Rp = _Rx(math.radians(s_c * t["tip_pre"])) @ Rz90 @ DOWN
    G = np.array([bx - fx, by - s_c * t["grip_dy"], bz + t["wall_grip_h"]])
    # come in low and in front of the box (x < 0.15): from READY straight across, the wrist swings into the open lid
    wp(P, 1.4, (0.13, s_c * 0.20, TOP + 0.21), fo * t["wall_open"], "pour: in front of the box")
    wp(P, 1.3, (G[0] - 0.02, G[1], G[2] + 0.13), fo * t["wall_open"], "pour: above the front wall", Rz90 @ DOWN)
    wp(P, 1.2, G + (0, 0, 0.10), fo * t["wall_open"], "pour: above the front wall", Rp)
    wp(P, 1.5, G, fo * t["wall_open"], "pour: straddle the wall", Rp)
    wp(P, 0.8, G, FINGER_CLOSED, "pour: pinch wall and paper", Rp)
    base_half = HY_RIM - HB * math.tan(TAPER)  # outer half-width of the base
    stop = abs(BINS["compost_bin"][1]) - HOLE_SY / 2 - labkit.TABLE_TRIM["width"] - t["edge_gap"]
    yc = s_c * (stop - base_half)  # box centre with its base edge just short of the trim
    Gs = G + np.array([0.0, yc - by, 0.002])
    wp(P, 2.0, Gs, FINGER_CLOSED, "pour: slide to the hole's edge", Rp)
    Pv = np.array([bx, yc + s_c * base_half, bz])  # the base edge: the pivot

    def tip(a):
        R = _Rx(math.radians(-s_c * a))  # raises the gripped side about the pivot
        return Pv + R @ (Gs - Pv), R @ Rp

    a = 0.0
    while a < t["tip_deg"] - 1e-6:
        a = min(a + 6.0, t["tip_deg"])
        wp(P, 0.4, tip(a)[0], FINGER_CLOSED, "pour: tip over the edge", tip(a)[1])
    wp(P, 1.5, tip(a)[0], FINGER_CLOSED, "pour: hold", tip(a)[1])
    if os.environ.get("TRAY_SORT_SHAKE", "1") != "0":  # demo19: with slippery fries the shake slammed the lid onto the gripper
        wp(P, 0.6, tip(a - 25)[0], FINGER_CLOSED, "pour: shake", tip(a - 25)[1])
        wp(P, 0.5, tip(a)[0], FINGER_CLOSED, "pour: shake", tip(a)[1])
    # demo8 setting: 1.5 s. TRAY_SORT_HOLD2 (s) lengthens this still hold to let the last fry slide out (no extra shaking:
    # fast shakes fling food, demo9)
    wp(P, float(os.environ.get("TRAY_SORT_HOLD2", "1.5")), tip(a)[0], FINGER_CLOSED, "pour: hold", tip(a)[1])
    # TRAY_SORT_JERKS=n: n quick small rocks once only the fry caught in the crease by the near wall is left (demo8/10/11:
    # fry05 stays there at any hold time; the near wall is only ~level at the reachable 92 deg tip). Off by default.
    for _ in range(int(os.environ.get("TRAY_SORT_JERKS", "0"))):
        jd = float(os.environ.get("TRAY_SORT_JERK_DEG", "12"))
        jt = float(os.environ.get("TRAY_SORT_JERK_S", "0.25"))
        wp(P, jt, tip(a - jd)[0], FINGER_CLOSED, "pour: jerk", tip(a - jd)[1])
        wp(P, jt, tip(a)[0], FINGER_CLOSED, "pour: jerk", tip(a)[1])
    if int(os.environ.get("TRAY_SORT_JERKS", "0")):
        wp(P, 1.0, tip(a)[0], FINGER_CLOSED, "pour: hold", tip(a)[1])
    # TRAY_SORT_SHUFFLE=n: n "vibratory feeder" strokes at full tip. The box slides fast away from the hole by SHUF_D
    # (the fry left in the crease cannot follow: it needs ~0.65 g to be dragged along) and back slowly (the fry rides
    # along), so each stroke walks the fry SHUF_D towards the rim, and it drops just past the rim into the hole.
    n_sh = int(os.environ.get("TRAY_SORT_SHUFFLE", "0"))
    sh_d = float(os.environ.get("TRAY_SORT_SHUF_D", "0.025"))
    sh_f = float(os.environ.get("TRAY_SORT_SHUF_FAST", "0.15"))
    sh_s = float(os.environ.get("TRAY_SORT_SHUF_SLOW", "0.6"))
    away = np.array([0.0, -s_c * sh_d, 0.0])
    for _ in range(n_sh):
        wp(P, sh_f, tip(a)[0] + away, FINGER_CLOSED, "pour: shuffle out", tip(a)[1])
        wp(P, sh_s, tip(a)[0], FINGER_CLOSED, "pour: shuffle back", tip(a)[1])
    if n_sh:
        wp(P, 1.0, tip(a)[0], FINGER_CLOSED, "pour: hold", tip(a)[1])
    # TRAY_SORT_EXTRA_DEG=d: after the hold, tip on slowly by d deg past tip_deg (so the near wall slopes down to the rim and
    # the fry in its crease slides out without being flung), hold, come back. The arm reaches this only roughly (IK).
    xd = float(os.environ.get("TRAY_SORT_EXTRA_DEG", "0"))
    if xd > 0:
        for k in (1, 2, 3):
            wp(P, 0.6, tip(a + xd * k / 3)[0], FINGER_CLOSED, "pour: tip on", tip(a + xd * k / 3)[1])
        wp(P, 1.5, tip(a + xd)[0], FINGER_CLOSED, "pour: hold", tip(a + xd)[1])
        wp(P, 0.8, tip(a)[0], FINGER_CLOSED, "pour: hold", tip(a)[1])
    # TRAY_SORT_TAPS=n: knock the tipped box n times on the hole's edge, as a person taps a container on the bin rim:
    # lift it TAP_UP, bring it down fast onto the table edge (TAP_DOWN s); each knock makes the last fry hop down the
    # near wall (only ~10 deg downhill at the reachable tip) and off the rim into the hole.
    n_tap = int(os.environ.get("TRAY_SORT_TAPS", "0"))
    tap_up = float(os.environ.get("TRAY_SORT_TAP_UP", "0.015"))
    tap_dn = float(os.environ.get("TRAY_SORT_TAP_DOWN", "0.1"))
    for _ in range(n_tap):
        wp(P, 0.35, tip(a)[0] + (0, 0, tap_up), FINGER_CLOSED, "pour: tap up", tip(a)[1])
        wp(P, tap_dn, tip(a)[0] + (0, 0, -0.003), FINGER_CLOSED, "pour: tap down", tip(a)[1])
        wp(P, 0.15, tip(a)[0], FINGER_CLOSED, "pour: tap rest", tip(a)[1])
    if n_tap:
        wp(P, 1.0, tip(a)[0], FINGER_CLOSED, "pour: hold", tip(a)[1])
    while a > 1e-6:
        a = max(a - 10.0, 0.0)
        wp(P, 0.4, tip(a)[0], FINGER_CLOSED, "pour: tip back", tip(a)[1])
    # TRAY_SORT_REPOUR=1: second pour for the piece left in the paper's fold by the near wall (fry05, demo8-18). Tilt
    # the box the other way (about its landfill-side base edge) so it slides back to the far wall, then pour again:
    # it now slides the whole floor and carries over the fold like the rest of the food did in the first pour.
    if os.environ.get("TRAY_SORT_REPOUR", "0") != "0":
        Pv2 = np.array([bx, yc - s_c * base_half, bz])  # the landfill-side base edge

        def back(b):
            R = _Rx(math.radians(s_c * b))  # raises the compost side about the landfill-side edge
            return Pv2 + R @ (Gs - Pv2), R @ Rp

        bmax = float(os.environ.get("TRAY_SORT_REPOUR_BACK", "40"))
        b = 0.0
        while b < bmax - 1e-6:
            b = min(b + 6.0, bmax)
            wp(P, 0.3, back(b)[0], FINGER_CLOSED, "repour: tilt back", back(b)[1])
        wp(P, 1.2, back(b)[0], FINGER_CLOSED, "repour: hold back", back(b)[1])
        while b > 1e-6:
            b = max(b - 8.0, 0.0)
            wp(P, 0.25, back(b)[0], FINGER_CLOSED, "repour: level", back(b)[1])
        while a < t["tip_deg"] - 1e-6:
            a = min(a + 6.0, t["tip_deg"])
            wp(P, 0.3, tip(a)[0], FINGER_CLOSED, "repour: tip over the edge", tip(a)[1])
        wp(P, 2.0, tip(a)[0], FINGER_CLOSED, "repour: hold", tip(a)[1])
        while a > 1e-6:
            a = max(a - 10.0, 0.0)
            wp(P, 0.4, tip(a)[0], FINGER_CLOSED, "repour: tip back", tip(a)[1])
    wp(P, 0.6, Gs, fo * t["wall_open"], "pour: let go", Rp)
    wp(P, 1.0, Gs + (0, 0, 0.10), fo * t["wall_open"], "pour: up", Rp)
    wp(P, 1.0, Gs + (0, 0, 0.12), fo * t["wall_open"], "pour: up", Rz90 @ DOWN)

    # 2b. by grok: after the main pour, lift the liner's near-wall flap over the rim so the
    # crease flattens; then tip once more so the last fry slides into compost. Do not pick food.
    # The flap is the 2.65 cm end cell draped ~1.2 rad up the compost-side wall. Lift is local
    # to that rim (a few centimetres) so the sheet is not carried out of the box.
    flap_y = s_c * (FLAT_HALF_Y + 0.008)  # just toward the wall from the crease at +-9.3 cm
    flap_x = -0.02  # toward the robots, away from the open lid on the hinge side
    F0 = np.array([bx + flap_x, yc + flap_y, bz + 0.028])  # mid/top of the draped flap
    Rf = DOWN  # pads close along world y, the wall normal
    flap_open = fo * 0.55
    flap_pinch = fo * 0.10  # light pinch on the thin sheet, not a firm food grasp
    # via in front of the box so the arm does not cut through the open lid on the way to the flap
    wp(P, 0.8, np.array([0.13, yc + s_c * 0.08, TOP + 0.20]), flap_open, "liner: in front of the box", Rf)
    wp(P, 1.0, F0 + np.array([0.0, -s_c * 0.02, 0.08]), flap_open, "liner: above the near flap", Rf)
    wp(P, 0.8, F0 + np.array([0.0, -s_c * 0.01, 0.02]), flap_open, "liner: straddle the flap", Rf)
    wp(P, 0.6, F0, flap_open, "liner: straddle the flap", Rf)
    wp(P, 0.5, F0, flap_pinch, "liner: pinch the flap", Rf)
    wp(P, 0.7, F0 + (0.0, 0.0, 0.05), flap_pinch, "liner: lift the flap", Rf)
    Fd = F0 + np.array([0.0, s_c * 0.04, 0.03])  # toward the compost hole, still up
    wp(P, 0.8, Fd, flap_pinch, "liner: fold over the rim", Rf)
    wp(P, 0.6, Fd + (0.0, s_c * 0.01, -0.02), flap_pinch, "liner: drape over the rim", Rf)
    wp(P, 0.5, Fd + (0.0, s_c * 0.01, -0.02), flap_open, "liner: let go", Rf)
    # retreat up and back toward the box centre, not toward the hole, so the sheet is not dragged out
    wp(P, 0.7, Fd + np.array([0.0, -s_c * 0.03, 0.08]), flap_open, "liner: back off", Rf)
    wp(P, 0.8, Gs + (0, 0, 0.12), fo * t["wall_open"], "liner: clear", Rz90 @ DOWN)
    # crease should now be open; tip again so the last fry can slide out over the rim
    wp(P, 1.2, Gs, fo * t["wall_open"], "retip: straddle the wall", Rp)
    wp(P, 0.8, Gs, FINGER_CLOSED, "retip: pinch wall", Rp)
    a = 0.0
    while a < t["tip_deg"] - 1e-6:
        a = min(a + 6.0, t["tip_deg"])
        wp(P, 0.4, tip(a)[0], FINGER_CLOSED, "retip: tip over the edge", tip(a)[1])
    wp(P, 2.0, tip(a)[0], FINGER_CLOSED, "retip: hold", tip(a)[1])
    while a > 1e-6:
        a = max(a - 10.0, 0.0)
        wp(P, 0.4, tip(a)[0], FINGER_CLOSED, "retip: tip back", tip(a)[1])
    wp(P, 1.2, Gs, fo, "retip: let go fully", Rp)
    wp(P, 0.8, Gs, fo, "retip: release settle", Rp)
    wp(P, 1.0, Gs + (0, 0, 0.075), fo, "retip: up", Rp)
    # Withdraw in front of the box before crossing toward the right ready pose.
    # A direct diagonal park sweeps the open fingers and forearm across the lid.
    wp(P, 1.0, Gs + (-0.08, 0, 0.075), fo, "retip: clear front", DOWN)
    wp(P, 1.5, (0.10, s_c * 0.23, TOP + 0.13), fo, "retip: clear side", DOWN)
    wp(P, 1.0, (0.13, s_c * 0.30, TOP + 0.15), fo, "retip: outside box", DOWN)
    park_p = ik.tcp_pose(np.asarray(READY, float), np.asarray(bases[P], float))
    wp(P, 1.8, park_p[0], fo, "park", park_p[1])

    # 3. dispose: the lid arm pinches the same wall (paper inside) and drops the emptied box into the landfill hole ------
    wait_until(A, clock[P] + 2.0)
    _DISPOSE.clear()
    _DISPOSE.update(start=clock[A], bound=False, arm=A, plans=plans, down=DOWN, side=s_c, bases=bases)
    G2 = np.array([bx, yc - s_c * hw, bz + t["wall_grip_h"]])  # the lid arm's near wall, where the box now stands
    # straddle wider than the pour pinch is possible (TRAY_SORT_DISPOSE_OPEN); 0.42 = the pour pinch (demo13)
    do = float(os.environ.get("TRAY_SORT_DISPOSE_OPEN", "0.42"))  # 0.8 failed once in two (demo15/16)
    # TRAY_SORT_DUMP=1: over the landfill hole, turn the box over to empty it before letting it go. The last fry stays
    # in the paper's fold by the near wall after the pour (demo8-18); up to 10 % of the food may go to landfill, but the
    # container must land empty. The wrist turns ~120 deg only from a grip pre-rolled 45 deg about x.
    dump = os.environ.get("TRAY_SORT_DUMP", "0") != "0"
    pre_d = float(os.environ.get("TRAY_SORT_DUMP_PRE", "35")) if dump else 0.0
    Rd = _Rx(math.radians(-s_c * pre_d)) @ DOWN
    wp(A, 1.4, (0.13, -s_c * 0.20, TOP + 0.21), fo * do, "dispose: in front of the box")
    if dump:  # the rolled wrist only reaches low and on the landfill side of the wall: come in from there
        wp(A, 1.5, G2 + (0, -s_c * 0.12, 0.05), fo * do, "dispose: beside the wall", Rd)
        wp(A, 1.2, G2 + (0, -s_c * 0.01, 0.04), fo * do, "dispose: above the wall", Rd)
    else:
        wp(A, 1.3, G2 + (-0.02, 0, 0.10), fo * do, "dispose: above the wall", Rd)
    wp(A, 1.5, G2, fo * do, "dispose: straddle the wall", Rd)
    wp(A, 0.8, G2, FINGER_CLOSED, "dispose: pinch", Rd)
    lift_h = 0.04 if dump else 0.14  # the rolled wrist does not reach 14 cm up there
    wp(A, 1.5, G2 + (0, 0, lift_h), FINGER_CLOSED, "dispose: lift", Rd)
    ly = BINS["trash_bin"][1]
    Gd = np.array([HOLE_X - 0.006, ly - s_c * hw, TOP + (0.12 if dump else 0.15)])
    wp(A, 2.5, Gd, FINGER_CLOSED, "dispose: over the landfill hole", Rd)
    if dump:
        off = np.array([0.0, -s_c * hw, 0.0])  # grip from the box's middle
        Cd = np.array([Gd[0], ly, TOP + 0.12])
        cmax = float(os.environ.get("TRAY_SORT_DUMP_DEG", "115"))

        def turn(c):
            R = _Rx(math.radians(s_c * c))  # lowers the gripped wall: the open top turns over the hole
            return Cd + R @ off, R @ Rd

        wp(A, 1.0, turn(0)[0], FINGER_CLOSED, "dump: lower", turn(0)[1])
        c = 0.0
        while c < cmax - 1e-6:
            c = min(c + 10.0, cmax)
            wp(A, 0.3, turn(c)[0], FINGER_CLOSED, "dump: turn over", turn(c)[1])
        wp(A, 1.5, turn(c)[0], FINGER_CLOSED, "dump: hold", turn(c)[1])
        while c > 1e-6:
            c = max(c - 15.0, 0.0)
            wp(A, 0.3, turn(c)[0], FINGER_CLOSED, "dump: turn back", turn(c)[1])
    # demo8 let go with the grip 2.2 cm above the table (box floor only 4 mm into the hole): the box could catch the trim and
    # flip (demo10). TRAY_SORT_DROP_Z sets that grip height (m above the table top).
    dz = float(os.environ.get("TRAY_SORT_DROP_Z", "-0.02"))  # demo13: box floor 4.6 cm into the hole
    wp(A, 1.5, (Gd[0], Gd[1], TOP + dz), FINGER_CLOSED, "dispose: into the hole", Rd)
    wp(A, 0.6, (Gd[0], Gd[1], TOP + dz), fo, "dispose: let go", Rd)
    wp(A, 1.5, (Gd[0], Gd[1], TOP + 0.15), fo, "dispose: up", Rd)
    wp(A, 1.5, park_l, fo, "park", ik.tcp_pose(np.asarray(READY, float), np.asarray(bases[A], float))[1])
    _DISPOSE.update(nominal_grip=G2.copy(), nominal_carry=Gd.copy())
    end = max(clock.values())
    for arm in plans:
        wait_until(arm, end, "done")
    return plans


def demo():
    from yam_arena import demo as dm

    old = dm.grip_rotation
    if not getattr(old, "_tsort", False):
        def grip_rotation(x=0.0):
            return x.rot if hasattr(x, "rot") else old(x)
        grip_rotation._tsort = True
        dm.grip_rotation = grip_rotation
    return demo_plan()


# -- development hooks (off unless the env vars are set) ----------------------------------------------------------------
ANIM = os.environ.get("TRAY_SORT_ANIM", "")
ANIM_START, ANIM_STEPS = 20, 90  # TRAY_SORT_ANIM=open: lid swung open over 4.5 s by joint writes (preview only); the
# base is held where it stands meanwhile (as a hand would), since the joint writes otherwise kick the 14 g base
_S = {"step": 0, "hold": None}


def _states(env):
    return {o.name: labkit._part_state(env, o) for o in OBJECTS}


def _bind_disposal(env, seconds):
    """Sample the settled physical box pose for the scripted demo's final grip.

    Only robot waypoints are changed; scene bodies and scoring stay untouched.
    """
    if not _DISPOSE or _DISPOSE['bound'] or seconds < _DISPOSE['start'] - 1e-7:
        return
    _DISPOSE['bound'] = True
    from yam_arena import ik
    from yam_arena.constants import READY
    states = _states(env)
    box = states['clamshell']
    if not np.isfinite(box.pos).all():
        raise RuntimeError('Nonfinite box state before final disposal')
    s_c = _DISPOSE['side']
    h = DEMO_T['wall_grip_h']
    rot = box.rotation()
    yaw = math.atan2(rot[1, 0], rot[0, 0])
    c, s = math.cos(yaw), math.sin(yaw)
    rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    local = np.array([0., -s_c * _wall_y(h), h])
    grip = box.to_world(local)
    grasp_rotation = rot @ _DISPOSE['down']
    carry_rotation = _DISPOSE['down']  # square the box with the rectangular opening
    offset = np.array([0, -s_c * _wall_y(h), 0])
    carry = np.array([HOLE_X - .006 + offset[0], BINS['trash_bin'][1] + offset[1], TOP + .15])
    for w in _DISPOSE['plans'][_DISPOSE['arm']]:
        if w.label in ['dispose: above the wall', 'dispose: straddle the wall', 'dispose: pinch', 'dispose: lift']:
            w.pos = tuple(np.asarray(w.pos) + grip - _DISPOSE['nominal_grip'])
            w.yaw = _Orient(grasp_rotation)
        elif w.label in ['dispose: over the landfill hole', 'dispose: into the hole', 'dispose: let go', 'dispose: up']:
            w.pos = tuple(np.asarray(w.pos) + carry - _DISPOSE['nominal_carry'])
            w.yaw = _Orient(carry_rotation)
    _, pe, re = ik.solve(grip, ik.matrix_to_quat_xyzw(grasp_rotation), np.asarray(READY),
                         base_pos=np.asarray(_DISPOSE['bases'][_DISPOSE['arm']]), iters=150)
    print('TRAYDISPOSE ' + json.dumps(dict(seconds=seconds, box_pos=box.pos.tolist(),
          box_quat=box.quat_xyzw.tolist(), box_tilt_deg=box.tilt_deg(),
          grip=grip.tolist(), ik_position_error_m=pe, ik_rotation_error_deg=math.degrees(re),
          food=breakdown(states))), flush=True)


def _report(env, tag):
    st = _states(env)
    b = breakdown(st)
    b["liner_pos"] = np.round(st["liner"].pos, 3).tolist()
    b["box_pos"] = np.round(st["clamshell"].pos, 3).tolist()
    b["box_quat_xyzw"] = np.round(st["clamshell"].quat_xyzw, 6).tolist()
    b["box_velocity"] = np.round(st["clamshell"].lin_vel, 4).tolist()
    b["lid"] = round(st["clamshell"].joints.get("lid", 0.0), 3)
    box = st["clamshell"]
    b["in_box_food"] = {n: np.round(box.to_local(st[n].pos), 3).tolist() for n in FOOD_MASS
                        if which_bin(st[n].pos) is None and in_box(box, st[n].pos)}
    b["loose_food"] = {n: np.round(st[n].pos, 3).tolist() for n in FOOD_MASS
                       if which_bin(st[n].pos) is None and not in_box(box, st[n].pos)}
    print("TRAYSORT " + json.dumps(dict(step=_S["step"], layout=LAYOUT, variant=VARIANT, phase=tag, **b)), flush=True)


def _hold_box(env):
    import torch
    a = env.unwrapped.scene["clamshell"]
    if _S["hold"] is None:
        _S["hold"] = torch.as_tensor(labkit._np(a.data.root_link_pose_w)[:1], dtype=torch.float32,
                                     device=env.unwrapped.scene.env_origins.device)
    a.write_root_pose_to_sim_index(root_pose=_S["hold"])
    a.write_root_velocity_to_sim_index(root_velocity=torch.zeros((1, 6), device=_S["hold"].device))


def _set_lid(env, q, qd):
    import torch
    a = env.unwrapped.scene["clamshell"]
    i = list(a.joint_names).index(LID.joint_name)
    dev = env.unwrapped.scene.env_origins.device
    a.write_joint_position_to_sim_index(position=torch.tensor([[q]], device=dev), joint_ids=[i])
    a.write_joint_velocity_to_sim_index(velocity=torch.tensor([[qd]], device=dev), joint_ids=[i])


@on_reset
def _on_reset(env):
    _S["step"] = 0


@on_step
def _on_step(env):
    _S["step"] += 1
    s = _S["step"]
    _bind_disposal(env, s / 20.0)
    if ANIM == "open" and ANIM_START - 2 <= s <= ANIM_START + ANIM_STEPS + 10:
        _hold_box(env)
    if ANIM == "open" and ANIM_START <= s <= ANIM_START + ANIM_STEPS:
        u = (s - ANIM_START) / ANIM_STEPS
        ease = 0.5 - 0.5 * math.cos(math.pi * u)
        _set_lid(env, (LID_OPEN - 0.05) * ease, (LID_OPEN - 0.05) * 0.5 * math.pi * math.sin(math.pi * u) / ANIM_STEPS * 20)
    if s in (5, 18, 60) or s % 100 == 0 or (620 <= s <= 840 and s % 10 == 0) or (1280 <= s <= 1800 and s % 10 == 0):  # dense log through the pour
        _report(env, "settled" if s == 60 else f"step{s}")
