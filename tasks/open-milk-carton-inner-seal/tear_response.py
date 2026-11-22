"""Material response of a tear-out membrane to observed external tensile contact."""
import json
import os
from pathlib import Path

import numpy as np


class TensileBond:
    def __init__(self, force_n):
        self.force_n = force_n
        self.reset()

    def reset(self):
        self.consecutive = 0
        self.released = False

    def observe(self, substep_forces):
        if self.released:
            return False
        for load in substep_forces:
            self.consecutive = self.consecutive + 1 if load >= self.force_n else 0
            if self.consecutive >= 2:
                self.released = True
                return True
        return False


def configure(ns):
    bond = TensileBond(ns["BREAK_BUDGET"])

    @ns["on_reset"]
    def reset(env):
        bond.reset()
        ns["TRACK"]["seal_material_open"] = False

    @ns["on_step"]
    def respond(env):
        contact = ns["TRACK"]["contact_window"]
        normal = ns["labkit"]._part_state(env, ns["CARTON"]).rotation() @ ns["N"]
        forces = np.asarray(contact["external_seal_forces_n"]) @ normal
        if not bond.observe(forces):
            return
        # Stage bindings must load after Kit has initialized its USD runtime.
        from pxr import UsdPhysics
        from isaaclab.sim.utils.stage import get_current_stage
        stage = get_current_stage()
        path = "/World/envs/env_0/seal/perimeter_bond"
        joint = UsdPhysics.Joint.Get(stage, path)
        # Only material connectivity changes. The link and seal stay dynamic.
        joint.GetJointEnabledAttr().Set(False)
        ns["TRACK"]["seal_material_open"] = True
        event = {"step": ns["TRACK"]["step"], "outward_force_peak_n": float(max(forces)),
                 "threshold_n": bond.force_n, "physics_hz": 240}
        print("MILK_TENSILE_RELEASE " + json.dumps(event), flush=True)
        trace = os.environ.get("TASK_TRACE_PATH")
        if trace:
            Path(trace).with_name("tear_event.json").write_text(json.dumps(event))
