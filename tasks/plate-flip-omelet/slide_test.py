"""Separate Isaac diagnostic: release replica faces from rest on inclined material coupons."""
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import labkit
from yam_arena.demo import Waypoint
from yam_arena.task_file import TABLE_TOP

SCENE = "station"
OBJECTS = []
CASES = []
CFG = json.loads(Path(__file__).with_name("bench-config.json").read_text())
MAT = CFG["simulation"]
RADIUS, HEIGHT = CFG["diameter_mm"] / 2000, CFG["thickness_mm"] / 1000
ANGLES = [16, 18, 19, 20, 21, 22, 23, 24, 25, 27, 35]
for group, (surface, face) in enumerate((s, f) for s in ("plate", "skillet") for f in ("brown", "plain")):
    for j, angle in enumerate(ANGLES):
        a = math.radians(angle)
        center = np.array([1.4+j*.6, (group-1.5)*.42, TABLE_TOP[SCENE]+.26])
        tag = f"{surface}_{face}_{angle}"
        color = (.65,.65,.60)
        OBJECTS.append(labkit.part("ramp_"+tag, center, fixed=True, friction=MAT[surface+"_mu"],
            boxes=[((0.,0.,0.),(.47,.26,.006),color,(0.,a))]))
        n = np.array([math.sin(a),0.,math.cos(a)])
        # Origin is the plain end of the body coordinate system at z=0, with
        # a thin brown collider projecting 0.45 mm below it in the upright pose.
        gap = .003 + (.00045 if face=="brown" else HEIGHT) + .00003
        start = center + n*gap
        phi = a if face=="brown" else a+math.pi
        q = [0., math.sin(phi/2),0.,math.cos(phi/2)]
        name = "sample_"+tag
        OBJECTS.append(labkit.part(name, start, mass=CFG["mass_g"]/1000, friction=MAT["replica_mu"],
            cylinders=[((0.,0.,HEIGHT/2),RADIUS,HEIGHT,(.8,.6,.2)),
                       ((0.,0.,-.0002),RADIUS-.0005,.0005,(.5,.3,.1))]))
        CASES.append((name,surface,face,angle,start,q,np.array([math.cos(a),0.,-math.sin(a)])))

def demo():
    from yam_arena import ik,station
    from yam_arena.constants import READY
    return {side:[Waypoint(3.,tuple(ik.tcp_pose(np.array(READY), station.BASES[side])[0]),-.03,"hold")]
            for side in ("left","right")}

def start_test(env):
    import torch
    from isaaclab_physx.physics import PhysxManager
    scene = env.unwrapped.scene
    sim = env.unwrapped.sim
    if getattr(sim.step, "_omelet_slide_test", False):
        return
    origin = labkit._np(scene.env_origins[0])
    for name,_,_,_,xyz,q,_ in CASES:
        asset = scene[name]
        pose = asset.data.default_root_pose.torch.clone()
        pose[0] = torch.as_tensor([*(xyz+origin),*q], dtype=pose.dtype, device=pose.device)
        asset.write_root_pose_to_sim_index(root_pose=pose)
        asset.write_root_velocity_to_sim_index(root_velocity=torch.zeros_like(pose[:,:6]))
    ctx = PhysxManager.get_physics_sim_view()
    view = ctx.create_rigid_body_view(scene.env_prim_paths[0]+"/sample_*/body")
    names = [path.split("/")[-2] for path in view.prim_paths]
    indexes = [names.index(case[0]) for case in CASES]
    dt = float(env.unwrapped.physics_dt)
    state = {"index":0}
    records = []
    original = sim.step
    def sample(*args,**kwargs):
        result = original(*args,**kwargs)
        state["index"] += 1
        if state["index"] <= round(1./dt):
            pose = np.asarray(view.get_transforms().numpy(),dtype=float)[indexes,:3]-origin
            records.append([float(np.dot(row-case[4],case[6])*1000) for row,case in zip(pose,CASES)])
        if state["index"] == round(1./dt):
            report={"dt_s":dt,"travel_mm_by_physics_step":np.array(records).T.tolist(),
                    "cases":[[c[1],c[2],c[3]] for c in CASES]}
            Path(os.environ["OMELET_COUPON_RESULT"]).write_text(json.dumps(report))
        return result
    sample._omelet_slide_test=True
    sim.step=sample

labkit.on_step(start_test)
def score(objects):return 0.
def success(objects):return False
