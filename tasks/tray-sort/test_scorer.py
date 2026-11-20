"""Offline scorer sanity check (no Isaac): synthetic object states for typical outcomes -> breakdown()."""
import json, pathlib, sys
from dataclasses import replace
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import task as T
from labkit import PartState

def st(p, yaw=0.0, joints=None):
    z = np.zeros(3)
    q = np.array([0, 0, np.sin(yaw / 2), np.cos(yaw / 2)])
    return PartState(pos=np.asarray(p, float), quat_xyzw=q, lin_vel=z, ang_vel=z, low=z, high=z, joints=joints or {})

part_by_name = {o.name: o for o in T.OBJECTS}

def package_state(name, position, yaw=0.0, **joint_overrides):
    joints = {joint.name: joint.initial for joint in part_by_name[name].joints}
    joints.update(joint_overrides)
    return st(position, yaw, joints)

def initial():
    return {o.name: st(o.pos, o.yaw, {joint.name: joint.initial for joint in o.joints})
            for o in T.OBJECTS}

def grid(b, names, z=0.02):
    cx, cy = T.BINS[b][:2]
    return {n: st((cx - 0.08 + 0.04 * (k % 5), cy - 0.06 + 0.04 * (k // 5 % 4), T.CATCH_FLOOR + z)) for k, n in enumerate(names)}

food = T.FOOD_NAMES
fries = [n for n in food if n.startswith("fry")]
cb, tb = T.BINS["compost_bin"], T.BINS["trash_bin"]
BOX_T = package_state("clamshell", (tb[0], tb[1], T.CATCH_FLOOR + 0.002), np.pi / 2)
LINER_T = package_state("liner", (tb[0], tb[1], T.CATCH_FLOOR + 0.08))
cases = {}
cases["untouched"] = initial()
s = initial(); s.update(grid("compost_bin", food)); s["clamshell"] = BOX_T; s["liner"] = LINER_T; cases["correct_sort"] = s
s = dict(cases["correct_sort"]); s["fry00"] = st((0.10, 0.0, T.TOP + 0.005)); cases["correct_but_1_fry_spilled"] = s
s = dict(cases["correct_sort"]); s.update({n: st((0.10, 0.012 * k, T.TOP + 0.005)) for k, n in enumerate(fries[:6])}); cases["correct_but_6_fries_spilled"] = s
s = dict(cases["correct_sort"]); s["liner"] = package_state("liner", (tb[0], tb[1] + 0.12, T.TOP + 0.005)); cases["correct_but_liner_left_on_table"] = s
s = initial(); s.update(grid("compost_bin", food)); s["liner"] = package_state("liner", (cb[0], cb[1], T.CATCH_FLOOR + 0.06)); s["clamshell"] = package_state("clamshell", (cb[0], cb[1], T.CATCH_FLOOR + 0.07), np.pi / 2); cases["dump_all_in_compost"] = s
s = dict(cases["dump_all_in_compost"]); s["clamshell"] = BOX_T; cases["pour_food+liner_in_compost_box_ok"] = s
s = initial(); s.update(grid("compost_bin", food[:8])); cases["half_food_rest_in_box"] = s
s = dict(cases["correct_sort"]); s["nugget0"] = st((tb[0], tb[1], T.CATCH_FLOOR + 0.03)); cases["box_binned_with_a_nugget_inside"] = s
s = initial(); s["clamshell"] = BOX_T; s["liner"] = package_state("liner", (tb[0], tb[1], T.CATCH_FLOOR + 0.012), np.pi / 2)
s.update({n: st((tb[0] - 0.08 + 0.01 * k, tb[1], T.CATCH_FLOOR + 0.02)) for k, n in enumerate(food)}); cases["whole_box_with_contents_in_trash"] = s
s = dict(cases["correct_sort"]); s.update({n: st((cb[0] + 0.12, cb[1], T.TOP + 0.004)) for n in food[:2]}); cases["2_food_pieces_on_trim_not_through"] = s
s = dict(cases["correct_sort"])
s["fry00"] = st((cb[0], cb[1], T.CATCH_TOP + 0.015))
cases["fry_in_chute_above_catch_bin"] = s
s = dict(cases["correct_sort"])
s["fry00"] = st((cb[0], cb[1], T.CATCH_TOP - 0.015))
cases["fry_inside_top_of_catch_bin"] = s
s = dict(cases["correct_sort"])
s["fry00"] = st((cb[0], cb[1], T.CATCH_FLOOR - 0.015))
cases["fry_below_catch_bin"] = s
s = dict(cases["correct_sort"]); s["clamshell"] = package_state("clamshell", (tb[0], tb[1], T.TOP - 0.01), np.pi / 2); cases["box_wedged_in_landfill_hole"] = s
s = dict(cases["correct_sort"]); s["liner"] = package_state("liner", (tb[0], tb[1], T.CATCH_FLOOR + 0.006)); s["clamshell"] = package_state("clamshell", (tb[0], tb[1], T.CATCH_FLOOR + 0.002)); cases["liner_inside_empty_box_in_landfill"] = s
s = dict(cases["correct_sort"])
s["clamshell"] = package_state("clamshell", (tb[0], tb[1], T.TOP - 0.045), lid=1.6)
cases["box_root_below_rim_but_lid_above_it"] = s
s = dict(cases["correct_sort"])
s["clamshell"] = package_state("clamshell", (tb[0], tb[1], T.TOP - 0.085), lid=0.0)
cases["box_below_table_but_still_in_chute"] = s
s = dict(cases["correct_sort"])
s.update(grid("trash_bin", food))
s["clamshell"] = package_state("clamshell", (tb[0], tb[1], T.CATCH_FLOOR + 0.12), lid=0.0)
cases["all_food_in_landfill_beside_empty_box"] = s
s = dict(cases["correct_sort"])
settled = package_state("clamshell", (0.306, 0.271, 0.568), lid=0.028)
s["clamshell"] = replace(settled, quat_xyzw=np.array([-0.424829, -0.859823, 0.056254, 0.277597]))
cases["tilted_rounded_box_resting_inside_landfill"] = s
expected = {
    "untouched": (0.0, False),
    "correct_sort": (1.0, True),
    "correct_but_1_fry_spilled": (0.9479, False),
    "correct_but_liner_left_on_table": (0.8, False),
    "fry_in_chute_above_catch_bin": (0.9479, False),
    "fry_inside_top_of_catch_bin": (1.0, True),
    "fry_below_catch_bin": (0.9479, False),
    "dump_all_in_compost": (0.0, False),
    "pour_food+liner_in_compost_box_ok": (0.15, False),
    "whole_box_with_contents_in_trash": (0.0, False),
    "all_food_in_landfill_beside_empty_box": (0.0, False),
    "box_wedged_in_landfill_hole": (0.85, False),
    "box_root_below_rim_but_lid_above_it": (0.85, False),
    "box_below_table_but_still_in_chute": (0.85, False),
    "liner_inside_empty_box_in_landfill": (1.0, True),
    "tilted_rounded_box_resting_inside_landfill": (1.0, True),
}
for k, v in cases.items():
    b = T.breakdown(v)
    if k in expected:
        assert (b["score"], b["full_success"]) == expected[k], (k, b)
    print(f"{k:36s} score={b['score']:.3f} full={b['full_success']!s:5s}", json.dumps({x: b[x] for x in ("food_frac", "spilled_g", "food_in_trash_g", "still_in_box_g", "liner_bin", "box_bin", "box_empty", "compost_packaging_contamination")}))
print("food total g:", round(T.TOTAL_FOOD * 1000, 1), "pieces:", len(food))
