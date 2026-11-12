"""Inspect Robots interface; importing it never starts Isaac Sim."""
from __future__ import annotations

import hashlib
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
NATIVE_SUCCESS = "FULL_WORKFLOW_SUCCEEDED_IN_THIS_SIMULATION"
DOCS = """Two YAM arms: left j0..j5, left gripper, right j0..j5, right gripper.
Actions are absolute arm joint targets in radians; grippers are 0 closed, 1 open.
The passive pliers close only when the left fingers squeeze their handles. The right
fingers hold the white blow gun and squeeze its passive trigger to turn air on.
Start with empty hands. Pick up the pliers, engage the wafer's straight flat, lift
the wafer, then pick up the gun and scan its nozzle across the wafer while blowing.
Blow all twelve red particles clear before the lifted wafer recontacts its pins.
Return the wafer to all three pillars, release it, then return and release the gun
on its stand within 120 seconds. Keep all fixtures in place. Heating and evaporation are not modeled.
Observations contain three RGB cameras and measured 14-D joint_pos only."""


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


class WaferBlowArena(EmbodimentBase):
    def __init__(self, *, bundle, evidence_dir="wafer_evidence", cam_width=320, cam_height=176,
                 max_steps=MAX_TASK_STEPS, device="cpu", development=False):
        self.bundle = Path(bundle).expanduser().resolve()
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
            name="wafer_blow_arena", action_space=action_space, observation_space=observations,
            control_hz=CONTROL_HZ, is_simulated=True, capabilities=frozenset({RESETTABLE, PRIVILEGED_SUCCESS}),
            docs=DOCS, environment_id="isaaclab-arena-wafer-blow-v1",
            environment_revision=_plan_revision(_read(self.bundle / "full_workflow_plan.json")), morphology="yam-bimanual")
        self._backend = None
        self._steps = 0
        self._done = True

    def reset(self, scene, *, seed=None):
        from wafer_benchmark_backend import NativeWaferWorld
        if self._backend is None:
            self._backend = NativeWaferWorld(self.bundle, self.evidence_dir, self.width, self.height, self.device, self.development)
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


class RecordedNativePolicy(PolicyBase):
    """Replay issued joint targets through step; this does not replay object poses."""
    config = PolicyConfig(action_horizon=1)

    def __init__(self, *, run, candidate_bundle=None):
        self.run = Path(run).expanduser().resolve()
        metadata = _read(self.run / "metadata.json")
        report = _read(self.run / "trial_report.json")
        audit = _read(self.run / "independent_check.json")
        if report.get("result") != NATIVE_SUCCESS:
            raise ValueError("The scripted source must be a successful fresh native full-workflow run.")
        direct = audit.get("result") == NATIVE_SUCCESS
        batch = (audit.get("result") == "INDEPENDENT_FULL_WORKFLOW_TRACE_AND_COMMANDS_MATCH"
                 and audit.get("passed") is True and audit.get("native_result") == NATIVE_SUCCESS)
        task_score = report.get("task_score", {})
        if (not (direct or batch) or audit.get("matches_report") is not True
                or audit.get("evidence", {}).get("complete") is not True
                or task_score.get("success") is not True or task_score.get("score") != 1.0
                or audit.get("task_score") != task_score):
            raise ValueError("The scripted source lacks an independent complete-evidence audit.")
        if audit.get("trace_sha256") != _sha(self.run / "samples.jsonl.gz"):
            raise ValueError("The scripted source trace differs from its independent audit.")
        if metadata.get("engine") != "Isaac Lab / PhysX" or not metadata.get("full_workflow_fresh_start"):
            raise ValueError("Local or continuation traces cannot serve as the native scripted source.")
        if metadata.get("post_start_pose_write_count") != 0:
            raise ValueError("The scripted source contains post-start pose writes.")
        origin = next(x["global_time_s"] for x in metadata["full_stage_boundaries"] if x["stage"] == "wafer_cycle")
        commands = []
        hashes = {}
        for filename, clock in (("tool_commands.jsonl", 0.0), ("commands.jsonl", origin),
                                ("cycle_commands.jsonl", origin), ("gun_return_commands.jsonl", 0.0)):
            path = self.run / filename
            hashes[filename] = _sha(path)
            for line in path.read_text().splitlines():
                item = json.loads(line)
                t = item["start_time_s"] if filename == "commands.jsonl" else item["time_s"]
                action = item["target"] if filename == "commands.jsonl" else item["action"]
                commands.append((float(t) + clock, validate_dim(action)))
        commands.sort(key=lambda x: x[0])
        if not commands or any(abs(t - i / CONTROL_HZ) > 1e-8 or not np.isfinite(a).all()
                               for i, (t, a) in enumerate(commands)):
            raise ValueError("Native command records have a gap, duplicate, or invalid action.")
        self.targets = [a for _, a in commands]
        self.initial = pack(np.r_[metadata["tool_pickup"]["start_arm_q_rad"],
                                  metadata["initial_finger_position_m"] / FINGER_OPEN_M],
                            np.r_[metadata["gun"]["approach_q_rad"], -0.035 / FINGER_OPEN_M])
        self.environment_revision = _plan_revision(metadata["full_workflow_plan"])
        self.plan = metadata["full_workflow_plan"]
        self.execution_revision = self.environment_revision
        if candidate_bundle is not None:
            from wafer_policy_provenance import validate_candidate_bridge
            self.execution_revision = validate_candidate_bridge(self.plan, candidate_bundle)
        self.source_hashes = {**hashes, **{n: _sha(self.run / n) for n in
                                        ("metadata.json", "trial_report.json", "independent_check.json")}}
        count = self._verify_recorded_targets(metadata)
        if type(audit.get("physics_samples")) is not int or audit["physics_samples"] != count:
            raise ValueError("The scripted source sample count differs from its independent audit.")
        self.info = PolicyInfo(name="wafer_blow_scripted", action_space=action_box(), control_hz=CONTROL_HZ,
                               checkpoint=hashlib.sha256(json.dumps(self.source_hashes, sort_keys=True).encode()).hexdigest(),
                               terminal_actions=frozenset({DONE}))
        self.index = 0

    def bind(self, embodiment_info):
        if embodiment_info.name != "wafer_blow_arena" or embodiment_info.control_hz != CONTROL_HZ:
            raise ValueError("Recorded targets require this benchmark's exact 20 Hz joint interface.")
        if embodiment_info.environment_revision != self.execution_revision:
            raise ValueError("The scripted native source and current scene revisions differ.")
        for target in self.targets:
            if np.any(target < embodiment_info.action_space.low) or np.any(target > embodiment_info.action_space.high):
                raise ValueError("A recorded target exceeds the current native joint limits.")
        self.info = PolicyInfo(name=self.info.name, action_space=embodiment_info.action_space,
                               observation_space=embodiment_info.observation_space, control_hz=CONTROL_HZ,
                               checkpoint=self.info.checkpoint, terminal_actions=frozenset({DONE}))

    def _verify_recorded_targets(self, metadata):
        dt = float(metadata["physics_dt_s"])
        stride = round(1 / CONTROL_HZ / dt)
        if stride < 1 or abs(stride * dt - 1 / CONTROL_HZ) > 1e-12:
            raise ValueError("Source physics cadence differs from the command records.")
        count = 0
        with gzip.open(self.run / "samples.jsonl.gz", "rt") as stream:
            for index, line in enumerate(stream):
                row = json.loads(line)
                if abs(float(row["time_s"]) - index * dt) > 1e-8:
                    raise ValueError("Source physical trace has a time gap.")
                if index:
                    command_index = (index - 1) // stride
                    if command_index >= len(self.targets):
                        raise ValueError("Source trace is longer than its issued action sequence.")
                    target = self.targets[command_index].astype(np.float32).astype(float)
                    expected = {"commanded_arm_q_rad": target[:6], "commanded_right_arm_q_rad": target[7:13],
                                "commanded_finger_q_m": [target[6] * FINGER_OPEN_M] * 2,
                                "commanded_right_finger_q_m": [target[13] * FINGER_OPEN_M] * 2}
                    if any(not np.allclose(row[key], value, atol=1e-7, rtol=0) for key, value in expected.items()):
                        raise ValueError("A scripted target differs from actual native actuator readback.")
                count += 1
        if count != 1 + stride * len(self.targets):
            raise ValueError("The source physical trace is incomplete.")
        return count

    def reset(self, scene):
        self.index = 0

    def act(self, observation):
        if self.index == 0:
            actual = validate_dim(observation.state[STATE_KEY])
            if not np.allclose(actual, self.initial, atol=1e-5, rtol=0):
                raise ValueError("Recorded policy initial measured joint state differs.")
        if self.index >= len(self.targets):
            return ActionChunk(actions=(), terminal=Termination(DONE, "Recorded target sequence ended; environment owns scoring."))
        target = self.targets[self.index].copy()
        self.index += 1
        return ActionChunk(actions=(Action(target),), control_hz=CONTROL_HZ)

    def transcript(self):
        return {"source": "Measured native run's issued joint targets; no object-state replay",
                "source_hashes": self.source_hashes,
                "source_environment_revision": self.environment_revision,
                "execution_environment_revision": self.execution_revision,
                "executed_target_count": self.index,
                "target_count": len(self.targets)}


class FinalWaferScore:
    name = "wafer_blow_score"

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
        if result.get("invalid_reasons"):
            value = 0.0
        return Score(value, "Binary full handling and particle-clearance success; no water-evaporation model.", metadata=result)


def wafer_blow_arena(**kwargs):
    return WaferBlowArena(**kwargs)


def wafer_blow_scripted(**kwargs):
    return RecordedNativePolicy(**kwargs)


def wafer_blow_score(**kwargs):
    if kwargs:
        raise ValueError("The task scorer has no configurable acceptance thresholds.")
    return FinalWaferScore()


class HoldCurrentPolicy(PolicyBase):
    """Local smoke-test policy. No external inference calls."""
    config=PolicyConfig(action_horizon=1)
    def bind(self,embodiment_info):
        self.info=PolicyInfo(name='wafer_blow_hold',action_space=embodiment_info.action_space,
            observation_space=embodiment_info.observation_space,control_hz=20.)
    def reset(self,scene):pass
    def act(self,observation):
        return ActionChunk(actions=(Action(validate_dim(observation.state[STATE_KEY]).copy()),),control_hz=20.)

def wafer_blow_hold(**kwargs):
    if kwargs:raise ValueError('Hold policy takes no options.')
    return HoldCurrentPolicy()
