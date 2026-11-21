"""Short scoring regression. Prior milestones are seeded; terminal contacts are simulated."""
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent))
import task as core
import labkit
from yam_arena.demo import Waypoint

SCENE=core.SCENE
OFFSET=np.array([.48,.34,0.])
stand=labkit.part("test_stand",core.at(core.PAN+OFFSET[:2],0.),fixed=True,
    cylinders=[core.disk(core.PAN_H/2,.078,core.PAN_H,(.33,.33,.33))])
OBJECTS=[*core.OBJECTS,stand]
moved=False

def write(asset,xyz,quat):
    import torch
    pose=asset.data.root_link_pose_w.torch.clone()
    pose[0]=torch.as_tensor([*xyz,*quat],dtype=pose.dtype,device=pose.device)
    asset.write_root_pose_to_sim_index(root_pose=pose)
    asset.write_root_velocity_to_sim_index(root_velocity=torch.zeros_like(pose[:,:6]))

def prepare(env):
    global moved
    moved=False
    scene=env.unwrapped.scene
    origin=labkit._np(scene.env_origins[0])
    # The browned face is at local -Z. Rotate pi and put that face upward.
    write(scene["omelet"],origin+core.at(core.PAN, core.PAN_H+.004+core.FOOD_H+.0006),[0.,1.,0.,0.])
    core.trial.stage=5

def intervene(env):
    global moved
    if moved or core.score(None)!=1.:
        return
    scene=env.unwrapped.scene
    for name in ("skillet","omelet"):
        obj=scene[name]
        pose=labkit._np(obj.data.root_link_pose_w)[0]
        write(obj,pose[:3]+OFFSET,pose[3:])
    moved=True
    print("OFF_HOB_INTERVENTION",core.trial.step,flush=True)

def demo():
    from yam_arena import ik,station
    from yam_arena.constants import READY
    return {side:[Waypoint(5.,tuple(ik.tcp_pose(np.array(READY),station.BASES[side])[0]),-.03,"remain clear")]
            for side in ("left","right")}

score=core.score
success=core.success
labkit.on_reset(prepare)
labkit.on_step(intervene)
