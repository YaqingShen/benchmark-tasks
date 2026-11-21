"""Run scoring counterexamples without starting Isaac Sim."""
import json
import os
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import numpy as np
import task


class Feed:
    def __init__(self, values):
        self.array = np.zeros((len(values), 3))
        self.array[:, 2] = values

    def get_contact_force_matrix(self, dt):
        return SimpleNamespace(numpy=lambda: self.array)


class Pose:
    def __init__(self, pos, angle):
        self.pos = np.array(pos, dtype=float)
        self.lin_vel = np.zeros(3)
        self.rot = task.rotate([0, 1, 0], angle)

    def rotation(self):
        return self.rot

    def tilt_deg(self):
        return float(np.rad2deg(np.arccos(np.clip(self.rot[2, 2], -1, 1))))

    def to_local(self, point):
        return self.rot.T @ (point - self.pos)

    def to_world(self, point):
        return self.rot @ point + self.pos


def main():
    os.environ.pop("OMELET_TRACE", None)
    scene = ModuleType("yam_arena.scene")
    runtime = ModuleType("yam_arena.runtime")
    physics = ModuleType("isaaclab_physx.physics")
    physics.PhysxManager = None
    runtime.link_poses = lambda robot: dict.fromkeys(("gripper", "tip_left", "tip_right"), np.eye(4))
    env = SimpleNamespace(unwrapped=SimpleNamespace(physics_dt=1 / 240, scene=dict(left_robot=0, right_robot=1)))
    objects = {}
    scene.object_states = lambda env, names: objects
    t = task.trial

    def tick(pan, plate, food, contacts=(1, 0), support=(0, 0), forbidden_index=None):
        objects.update(skillet=pan, plate=plate, omelet=food)
        forces = [*contacts, *[0] * 8]
        if forbidden_index is not None:
            forces[forbidden_index] = 1
        t.contact, t.pan_support = Feed(forces), Feed(support)
        t.tick(env)

    def reset():
        t.reset()

    def p(x, y, z, a=0):
        return Pose(task.at(task.PAN + np.array([x, y]), z), a)

    pan, plate, food = p(0, 0, .05), p(0, 0, .081), p(0, 0, .054)
    start = (pan, plate, food)
    result = {}
    assert task.plate_clearance([0.,0.,.100])>.040
    assert task.plate_clearance([0.,0.,.012])<.040
    assert task.plate_clearance([task.PR+.01,0.,.004])<.040
    assert task.plate_clearance([task.PR+.06,0.,.004])>.040
    modules = {"yam_arena.scene": scene, "yam_arena.runtime": runtime, "isaaclab_physx.physics": physics}
    with patch.dict(sys.modules, modules):
        reset()
        tick(*start, support=(1, 1))
        assert t.stage == 1
        result["plate_cover_only"] = task.score(None)
        for name, index in (("food_hits_hob", 2), ("food_hits_rack", 3), ("hand_contacts_food", 5)):
            reset()
            tick(*start, support=(1, 1))
            tick(*start, forbidden_index=index)
            assert t.failed and task.score(None) == 0
            result[name] = task.score(None)
        reset()
        tick(*start, support=(1, 1))
        tick(pan, plate, p(0, 0, .020, 45))
        assert t.failed and task.score(None) == 0
        result["tilted_food_edge_hits_table"] = task.score(None)
        reset()
        tick(pan, p(.23, 0, .066, 180), p(0, 0, .075, 180), support=(1, 0))
        assert task.score(None) == 0
        result["preflipped_food_without_sequence"] = task.score(None)
        reset()
        t.stage = 4
        flat = p(.105, 0, .100, 180)
        tick(pan, flat, Pose(flat.to_world([0, 0, -.02]), 180), contacts=(0, 1), support=(1, 0))
        tilt = p(.085, 0, .13, 150)
        tick(pan, tilt, Pose(tilt.to_world([0, 0, -.02]), 150), contacts=(0, 1), support=(1, 0))
        assert t.stage == 4
        result["tilt_only_without_slide_preserves_stage"] = t.stage
        reset()
        tick(*start, support=(1, 1))
        for angle, support_flags in ((90, (1, 0)), (180, (0, 1))):
            turning_pan = p(0, 0, .18, angle)
            turning_plate = Pose(turning_pan.to_world([0, 0, .031]), angle)
            turning_food = Pose(turning_plate.to_world([0, 0, -.021]), angle)
            tick(turning_pan, turning_plate, turning_food, contacts=support_flags)
        assert t.stage == 3
        for _ in range(4):
            tick(p(.16, 0, .23, 180), turning_plate, turning_food, contacts=(0, 1))
        assert t.stage == 4
        tick(pan, flat, Pose(flat.to_world([0, 0, -.021]), 180), contacts=(0, 1), support=(1, 0))
        tick(pan, tilt, Pose(tilt.to_world([.06, 0, -.021]), 150), contacts=(0, 1), support=(1, 0))
        assert t.stage == 5
        finish = (pan, p(.22, 0, .076, 180), p(0, 0, .076, 180))
        t.tool_hand_views = [Feed((0, 1, 0, 0, 0, 0)), Feed((0,) * 6)]
        for _ in range(25):
            tick(*finish, support=(1, 0))
        assert t.stage == 5
        result["still_holding_skillet"] = task.score(None)
        t.tool_hand_views = [Feed((0,) * 6), Feed((0,) * 6)]
        for _ in range(19):
            tick(*finish, support=(1, 0))
            assert task.score(None) == .85
        tick(*finish, support=(1, 0))
        assert task.score(None) == 1
        result["constructed_full_sequence"] = task.score(None)
        for name, variant, support, grip, contact in (
            ("move_off_hob", (p(.15,0,.05), finish[1], p(.15,0,.076,180)), (0,0), False, (1,0)),
            ("lose_hob_contact", finish, (0,0), False, (1,0)),
            ("return_plate_over_pan", (pan, plate, finish[2]), (1,0), False, (1,0)),
            ("plate_rim_still_contacts_pan", finish, (1,1), False, (1,0)),
            ("regrip_handle", finish, (1,0), True, (1,0)),
            ("tip_food_face", (pan,finish[1],p(0,0,.076,150)), (1,0), False, (1,0)),
            ("remove_food_support", finish, (1,0), False, (0,0)),
        ):
            t.tool_hand_views = [Feed((0,int(grip),0,0,0,0)), Feed((0,)*6)]
            tick(*variant, support=support, contacts=contact)
            assert task.score(None) == .85 and not task.success(None) and t.hold == 0
            result[name + "_after_full"] = task.score(None)
            t.tool_hand_views = [Feed((0,)*6), Feed((0,)*6)]
            for _ in range(19):
                tick(*finish,support=(1,0))
                assert not task.success(None)
            tick(*finish,support=(1,0))
            assert task.score(None) == 1
        tick(*finish, support=(1,0), forbidden_index=5)
        assert task.score(None) == 0 and not task.success(None)
        result["touch_replica_after_full"] = task.score(None)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
