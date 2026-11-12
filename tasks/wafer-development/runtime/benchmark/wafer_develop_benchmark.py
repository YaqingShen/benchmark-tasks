"""Inspect Robots interface for development; no scripted action substitution."""
from __future__ import annotations

import hashlib
import importlib
import gzip
import json
import math
from pathlib import Path

import numpy as np
from inspect_robots.embodiment import EmbodimentBase, EmbodimentInfo, PRIVILEGED_SUCCESS, RESETTABLE
from inspect_robots.policy import PolicyBase, PolicyConfig, PolicyInfo
from inspect_robots.scorer import Score
from inspect_robots.types import Action, ActionChunk, DONE, Observation, StepResult, Termination
from inspect_robots_yam.config import DEFAULT_CAMERAS, action_box, observation_space
from inspect_robots_yam.packing import STATE_KEY, pack, validate_dim

CONTROL_HZ = 20.0
MAX_TASK_SECONDS = 120.0
MAX_TASK_STEPS = int(CONTROL_HZ * MAX_TASK_SECONDS)
FINGER_OPEN_M = -0.04695
NATIVE_SUCCESS = "WAFER_DEVELOP_SUCCEEDED_IN_THIS_SIMULATION"
DOCS = """Two YAM arms: left j0..j5, left gripper, right j0..j5, right gripper.
Actions are absolute arm joint targets in radians; grippers are 0 closed, 1 open.
Both hands start empty. Passive pliers close only by squeezing their handles.
The scene includes a coated wafer on three heater pins, pliers on their stand,
a beaker with a geometric liquid surface, and a physical countdown timer button.
The timer display and wafer pattern are visible in the three RGB cameras.
Only measured 14-D joint_pos accompanies the cameras; timer values and object
poses are private scoring evidence. Follow the task instruction within 120 seconds.
The pattern is an elapsed-time surrogate, not calibrated development chemistry."""


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _plan_revision(plan):
    return hashlib.sha256(json.dumps(plan, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _validate_policy_files(bundle):
    bundle = Path(bundle).resolve()
    manifest = _read(bundle/'policy_files.json')
    modules = ('wafer_develop_benchmark', 'wafer_develop_backend', 'wafer_develop_score')
    required = {'benchmark/'+name+'.py' for name in modules}
    if not required.issubset(manifest):
        raise ValueError('Policy manifest does not bind every adapter and scoring module.')
    for name, expected in manifest.items():
        path = (bundle/name).resolve()
        if not path.is_relative_to(bundle) or not path.is_file() or _sha(path) != expected:
            raise ValueError('Policy source manifest mismatch: '+name)
    for name in modules:
        loaded = Path(importlib.import_module(name).__file__).resolve()
        if _sha(loaded) != manifest['benchmark/'+name+'.py']:
            raise ValueError('The loaded policy module differs from the selected bundle: '+name)


def _spaces(height, width):
    from yam_arena import YAM_URDF
    from yam_arena.constants import ARM_JOINTS, FINGER_OPEN
    from yam_arena.urdf import parse_joints
    if not math.isclose(float(FINGER_OPEN), FINGER_OPEN_M, rel_tol=0, abs_tol=1e-12):
        raise ValueError("The native YAM gripper mapping differs from this benchmark.")
    joints = parse_joints(YAM_URDF)
    low = np.r_[[joints[n].lower for n in ARM_JOINTS], 0.0]
    high = np.r_[[joints[n].upper for n in ARM_JOINTS], 1.0]
    return action_box(pack(low, low), pack(high, high)), observation_space(height, width, DEFAULT_CAMERAS)


class WaferDevelopArena(EmbodimentBase):
    def __init__(self, *, bundle, evidence_dir="wafer_evidence", cam_width=320, cam_height=176,
                 max_steps=MAX_TASK_STEPS, device="cpu", development=False):
        self.bundle = Path(bundle).expanduser().resolve()
        _validate_policy_files(self.bundle)
        self.evidence_dir = Path(evidence_dir).expanduser().resolve()
        self.width, self.height = int(cam_width), int(cam_height)
        self.max_steps = int(max_steps)
        if self.width < 16 or self.height < 16 or self.max_steps < 1:
            raise ValueError("Positive step limit and camera dimensions of at least 16 are required.")
        if self.max_steps > MAX_TASK_STEPS:
            raise ValueError("The complete task is limited to 120 seconds (2400 actions at 20 Hz).")
        self.device = str(device)
        if str(development).lower() not in ('true','false'):
            raise ValueError('development must explicitly be true or false.')
        self.development = str(development).lower()=='true' 
        action_space, observations = _spaces(self.height, self.width)
        self.info = EmbodimentInfo(
            name="wafer_develop_arena", action_space=action_space, observation_space=observations,
            control_hz=CONTROL_HZ, is_simulated=True, capabilities=frozenset({RESETTABLE, PRIVILEGED_SUCCESS}),
            docs=DOCS, environment_id="isaaclab-arena-wafer-develop-v1",
            environment_revision=_plan_revision({"full": _read(self.bundle / "full_workflow_plan.json"), "develop": _read(self.bundle / "develop_files.json"), "policy": _read(self.bundle / "policy_files.json")}), morphology="yam-bimanual")
        self._backend = None
        self._steps = 0
        self._done = True

    def reset(self, scene, *, seed=None):
        from wafer_develop_backend import NativeDevelopWorld
        if self._backend is None:
            self._backend = NativeDevelopWorld(self.bundle, self.evidence_dir, self.width, self.height, self.device, self.development)
        self._steps = 0
        self._done = True
        self._instruction = scene.instruction
        self._backend.reset(seed=seed if seed is not None else scene.init_seed, scene_id=scene.id)
        self._done = bool(self._backend.score()["invalid_reasons"])or not self._backend.score()['evidence_complete']
        if self._done:
            self._backend.finish()
            raise ValueError("The initial task state failed its measured scene validation.")
        return self._observation()

    def step(self, action):
        if self._done or self._backend is None:
            raise RuntimeError("Call reset before stepping a trial.")
        target = validate_dim(action.data)
        if not np.isfinite(target).all():
            raise ValueError("Actions must contain exactly 14 finite values.")
        target = np.clip(target, self.info.action_space.low, self.info.action_space.high)
        self._backend.step(target)
        self._steps += 1
        result = self._backend.score()
        success = result["success"] is True
        invalid = bool(result["invalid_reasons"])
        incomplete = not result['evidence_complete']
        truncated = (self._steps >= self.max_steps or incomplete) and not success and not invalid
        self._done = success or invalid or truncated
        info = {"score": float(result["score"]), "success": success, "wafer_task": result,
                "qualification_diagnostics": self._backend.diagnostics(),
                "physics_evidence": self._backend.evidence_reference()}
        observation = self._observation()
        if self._done:
            self._backend.finish()
        return StepResult(observation=observation, reward=float(result["score"]),
                          terminated=success or invalid, truncated=truncated,
                          termination_reason="success" if success else "invalid_task_state" if invalid else "incomplete_collision_evidence" if incomplete else None,
                          info=info)

    def _observation(self):
        images, state, timestamp = self._backend.observe()
        return Observation(images=images, state={STATE_KEY: state}, instruction=self._instruction,
                           image_times=dict.fromkeys(images, timestamp), state_time=timestamp)

    def close(self):
        self._done = True
        if self._backend is not None:
            self._backend.close()
            self._backend = None


class FinalWaferScore:
    name = "wafer_develop_score"

    def __call__(self, record, target):
        if getattr(record, "status", None) == "error":
            return Score(0.0, "Execution error; task evidence is incomplete.", metadata={"evidence_complete": False})
        if not record.steps:
            return Score(0.0, "No physics action was executed.")
        result = record.steps[-1].result.info.get("wafer_task")
        if not isinstance(result, dict):
            raise ValueError("The recorded trial has no measured wafer task score.")
        value = float(result["score"])
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("Recorded task score is outside [0, 1].")
        if result.get("invalid_reasons") or result.get('evidence_complete') is not True:
            value = 0.0
        return Score(value, "Measured timed immersion, physical timer operation and released pillar placement; declared process surrogate.", metadata=result)


def wafer_develop_arena(**kwargs):
    return WaferDevelopArena(**kwargs)



def wafer_develop_score(**kwargs):
    if kwargs:
        raise ValueError("The task scorer has no configurable acceptance thresholds.")
    return FinalWaferScore()


class HoldCurrentPolicy(PolicyBase):
    """Local smoke-test policy. No external inference calls."""
    config=PolicyConfig(action_horizon=1)
    def bind(self,embodiment_info):
        self.info=PolicyInfo(name='wafer_develop_hold',action_space=embodiment_info.action_space,
            observation_space=embodiment_info.observation_space,control_hz=20.)
    def reset(self,scene):pass
    def act(self,observation):
        return ActionChunk(actions=(Action(validate_dim(observation.state[STATE_KEY]).copy()),),control_hz=20.)

def wafer_develop_hold(**kwargs):
    if kwargs:raise ValueError('Hold policy takes no options.')
    return HoldCurrentPolicy()
