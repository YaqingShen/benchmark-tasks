"""Inspect Robots adapter. Actions always go directly to the native robot."""
from pathlib import Path
import hashlib
import numpy as np
from inspect_robots.embodiment import EmbodimentBase, EmbodimentInfo, RESETTABLE, PRIVILEGED_SUCCESS
from inspect_robots.policy import PolicyBase, PolicyConfig, PolicyInfo
from inspect_robots.scorer import Score
from inspect_robots.types import Action, ActionChunk, Observation, StepResult
from inspect_robots_yam.config import DEFAULT_CAMERAS, action_box, observation_space
from inspect_robots_yam.packing import STATE_KEY, pack, validate_dim

MAX_STEPS = 12000
DOCS = """Two YAM arms with parallel grippers. The 14 absolute joint targets are
left j0..j5, left gripper, right j0..j5, right gripper. Arm units are radians;
grippers are 0 closed and 1 open. Observations contain top_cam, left_cam and
right_cam RGB and measured joint_pos only. Read the balance display to determine
whether to add liquid from the open stock bottle or remove it into the waste
beaker. Keep the receiving beaker on its balance and stock bottle upright.
The original empty-beaker tare is fixed. The target is 10.000 +/- 0.050 g.
Finish with the used pipette released completely inside the bin."""


class FormulationArena(EmbodimentBase):
    def __init__(self, *, bundle, evidence_dir, cam_width=1280, cam_height=704,
                 max_steps=MAX_STEPS, device='cpu'):
        from yam_arena import YAM_URDF
        from yam_arena.constants import ARM_JOINTS
        from yam_arena.urdf import parse_joints
        self.root = Path(bundle).resolve()
        self.output = Path(evidence_dir).resolve()
        self.width, self.height = int(cam_width), int(cam_height)
        self.max_steps = int(max_steps)
        if not 1 <= self.max_steps <= MAX_STEPS:
            raise ValueError('Action budget must be between 1 and 12000')
        self.device = device
        joints = parse_joints(YAM_URDF)
        low = np.r_[[joints[n].lower for n in ARM_JOINTS], 0.]
        high = np.r_[[joints[n].upper for n in ARM_JOINTS], 1.]
        revision = hashlib.sha256((self.root/'source_manifest.json').read_bytes()).hexdigest()
        self.info = EmbodimentInfo(name='liquid_correction_arena',
            action_space=action_box(pack(low, low), pack(high, high)),
            observation_space=observation_space(self.height, self.width, DEFAULT_CAMERAS),
            control_hz=20., is_simulated=True, capabilities=frozenset({RESETTABLE, PRIVILEGED_SUCCESS}),
            docs=DOCS, environment_id='isaaclab-liquid-correction-v1',
            environment_revision=revision, morphology='yam-bimanual')
        self.world = None
        self.done = True
        self.reset_count = 0

    def reset(self, scene, *, seed=None):
        from formulation_backend import FormulationWorld
        # Liquid changes native mass, COM and inertia. Rebuild from the frozen
        # assets between trials so none of those properties leak into a reset.
        if self.world is not None:
            self.world.close()
        self.world = FormulationWorld(self.root, self.output, self.width, self.height, self.device)
        self.world.trial_count = self.reset_count
        self.instruction = scene.instruction
        seed = seed if seed is not None else scene.init_seed
        self.world.reset(seed=0 if seed is None else int(seed))
        self.reset_count += 1
        self.done = False
        return self._observation()

    def _observation(self):
        images, state, timestamp = self.world.observe()
        return Observation(images=images, state={STATE_KEY: state}, instruction=self.instruction,
                           image_times=dict.fromkeys(images, timestamp), state_time=timestamp)

    def step(self, action):
        if self.done:
            raise RuntimeError('Reset the trial before acting')
        target = validate_dim(action.data)
        if not np.isfinite(target).all():
            raise ValueError('Nonfinite action')
        target = np.clip(target, self.info.action_space.low, self.info.action_space.high)
        self.world.step(target)
        result = self.world.score()
        success = result['success']
        invalid = bool(result.get('violations'))
        truncated = self.world.steps >= self.max_steps and not (success or invalid)
        self.done = success or invalid or truncated
        observation = self._observation()
        if self.done:
            self.world.finish()
        return StepResult(observation=observation, reward=result['score'],
            terminated=success or invalid, truncated=truncated,
            termination_reason='success' if success else 'invalid_task_state' if invalid else None,
            info={'score': result['score'], 'success': success, 'liquid_correction': result})

    def close(self):
        self.done = True
        if self.world is not None:
            self.world.close()
            self.world = None


class LiquidCorrectionScore:
    name = 'liquid_correction_score'

    def __call__(self, record, target):
        if getattr(record, 'status', None) == 'error' or not record.steps:
            return Score(0., 'No completed native evidence')
        result = record.steps[-1].result.info.get('liquid_correction')
        if not isinstance(result, dict):
            raise ValueError('Native scoring evidence is absent')
        value = float(result['score'])
        if not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError('Invalid task score')
        return Score(value, 'Settled liquid mass correction and released pipette disposal.', metadata=result)


class HoldPolicy(PolicyBase):
    config = PolicyConfig(action_horizon=1)

    def bind(self, info):
        self.info = PolicyInfo(name='liquid_correction_hold', action_space=info.action_space,
                              observation_space=info.observation_space, control_hz=20.)

    def reset(self, scene):
        pass

    def act(self, observation):
        return ActionChunk(actions=(Action(validate_dim(observation.state[STATE_KEY]).copy()),), control_hz=20.)


def liquid_correction_arena(**kwargs):
    return FormulationArena(**kwargs)


def liquid_correction_score(**kwargs):
    if kwargs:
        raise ValueError('Scoring thresholds are fixed')
    return LiquidCorrectionScore()


def liquid_correction_hold(**kwargs):
    if kwargs:
        raise ValueError('Hold policy takes no arguments')
    return HoldPolicy()
