"""Check the installed YAM environment and declared specimen against the native scene."""
import json
import math
from pathlib import Path

import task
from yam_arena.constants import GRIPPER, PHYSICS_DT

cfg=json.loads(Path(__file__).with_name("bench-config.json").read_text())
expected=cfg["simulation"]
pairs={
    "radius_m":(task.FOOD_R,cfg["diameter_mm"]/2000),
    "thickness_m":(task.FOOD_H,cfg["thickness_mm"]/1000),
    "mass_kg":(task.food.mass,cfg["mass_g"]/1000),
    "replica_friction":(task.food.friction,expected["replica_mu"]),
    "plate_friction":(task.plate.friction,expected["plate_mu"]),
    "pan_friction":(task.pan.friction,expected["skillet_mu"]),
    "gripper_effort_N":(GRIPPER["effort"],expected["driver_stall_force_setting_N"]),
    "gripper_drive_stiffness_N_m":(GRIPPER["stiffness"],expected["gripper_stiffness_N_m"]),
    "gripper_drive_damping_N_s_m":(GRIPPER["damping"],expected["gripper_damping_N_s_m"]),
    "physics_rate_hz":(1/PHYSICS_DT,240),
}
mu=task.labkit.SURFACE_FRICTION[("skillet",task.BLACK)]
pairs["handle_static_mu"]=(mu[0],expected["handle_static_mu"])
pairs["handle_dynamic_mu"]=(mu[1],expected["handle_dynamic_mu"])
for key,(actual,wanted) in pairs.items():
    if not math.isclose(actual,wanted,abs_tol=1e-9):
        raise ValueError(f"{key}: scene {actual}, configuration {wanted}")
print(json.dumps({k:v[0] for k,v in pairs.items()},indent=2))
