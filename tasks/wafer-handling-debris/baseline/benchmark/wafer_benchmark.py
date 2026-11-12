"""Inspect Robots interface to the shared native MCP2 visible-debris task."""
from __future__ import annotations

import hashlib
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

CONTROL_HZ=20.
MAX_TASK_STEPS=2400
FINGER_OPEN_M=-.04695
DOCS="""Other: cleanroom-inspired tool manipulation and visible-debris removal.
Actions: left arm six absolute joint targets (radians), left gripper (0 closed,
1 open), right arm six joint targets, right gripper. Both hands start empty.
Pick up the MCP2-inspired tool; squeezing its handle opens its spring-closed jaw.
Grip and lift the wafer at its straight flat. Pick up the white blow gun with the
other hand, squeeze its mechanical trigger and aim at the held wafer to remove
all twelve sparse visible cleanroom surrogates: silicon/quartz-like grains,
garment fibers, skin-like flakes and equipment-wear fragments. Return the wafer to its three pillars and the
released gun to its stand within 120 seconds. Retain the objects and fixtures.
Only three RGB cameras and measured 14-D joint_pos are policy observations.
Dream's matching camera format is 320x176 RGB at 20 control steps per second.
This tests visible-debris manipulation, not nanofabrication cleanliness, drying,
employment impact or calibrated airflow. Camera-enabled native validation of
this staged interface is pending; importing this module starts no simulator."""


def _spaces(height,width):
    from yam_arena import YAM_URDF
    from yam_arena.constants import ARM_JOINTS,FINGER_OPEN
    from yam_arena.urdf import parse_joints
    if not math.isclose(float(FINGER_OPEN),FINGER_OPEN_M,rel_tol=0.,abs_tol=1e-12):
        raise ValueError('The native YAM gripper mapping differs.')
    joints=parse_joints(YAM_URDF)
    low=np.r_[[joints[name].lower for name in ARM_JOINTS],0.]
    high=np.r_[[joints[name].upper for name in ARM_JOINTS],1.]
    return action_box(pack(low,low),pack(high,high)),observation_space(height,width,DEFAULT_CAMERAS)


class WaferBlowArena(EmbodimentBase):
    def __init__(self,*,bundle,evidence_dir='wafer_evidence',cam_width=320,cam_height=176,
                 max_steps=MAX_TASK_STEPS,device='cpu',layout_seed=None):
        from wafer_benchmark_backend import environment_revision,bundle_paths,_sha,_layout_seed
        self.bundle=Path(bundle).expanduser().resolve()
        _,baseline=bundle_paths(self.bundle)
        if _sha(__file__)!=_sha(baseline/'benchmark/wafer_benchmark.py'):
            raise ValueError('Installed embodiment differs from the selected bundle.')
        self.evidence_dir=Path(evidence_dir).expanduser().resolve()
        self.width,self.height=int(cam_width),int(cam_height)
        self.max_steps=int(max_steps);self.device=str(device);self.layout_seed=_layout_seed(layout_seed)
        if min(self.width,self.height)<16 or not 1<=self.max_steps<=MAX_TASK_STEPS:
            raise ValueError('Camera dimensions must be at least16 and step limit1..2400.')
        if self.device!='cpu':raise ValueError('This native benchmark uses CPU PhysX.')
        actions,observations=_spaces(self.height,self.width)
        self.info=EmbodimentInfo(name='wafer_blow_arena',action_space=actions,observation_space=observations,
            control_hz=CONTROL_HZ,is_simulated=True,capabilities=frozenset({RESETTABLE,PRIVILEGED_SUCCESS}),
            docs=DOCS+('\nDebris layout follows the requested trial seed.'if self.layout_seed is None else f'\nFixed debris layout seed: {self.layout_seed}.'),environment_id='isaaclab-arena-mcp2-mixed-debris-v6',
            environment_revision=environment_revision(self.bundle,self.layout_seed),morphology='yam-bimanual')
        self._backend=None;self._steps=0;self._done=True

    def reset(self,scene,*,seed=None):
        from wafer_benchmark_backend import NativeWaferWorld
        if self._backend is None:
            self._backend=NativeWaferWorld(self.bundle,self.evidence_dir,self.width,self.height,self.device,layout_seed=self.layout_seed)
        self._steps=0;self._done=True;self._instruction=scene.instruction
        self._backend.reset(seed=seed if seed is not None else getattr(scene,'init_seed',None),scene_id=scene.id)
        result=self._backend.score()
        if result['invalid_reasons']or not result['evidence_complete']:
            self._backend.finish()
            raise ValueError('The initial measured task state is invalid.')
        self._done=False
        return self._observation()

    def step(self,action):
        if self._done or self._backend is None:raise RuntimeError('Call reset before stepping.')
        target=validate_dim(action.data)
        if not np.isfinite(target).all():raise ValueError('Actions must contain14 finite joint targets.')
        target=np.clip(target,self.info.action_space.low,self.info.action_space.high)
        self._backend.step(target);self._steps+=1
        result=self._backend.score()
        success=result['success']is True;invalid=bool(result['invalid_reasons'])
        incomplete=not result['evidence_complete']
        self._done=success or invalid or incomplete or self._steps>=self.max_steps
        observation=self._observation()
        if self._done:
            self._backend.finish()
            result=self._backend.score()
            success=result['success']is True;invalid=bool(result['invalid_reasons'])
            incomplete=not result['evidence_complete']
        truncated=(self._steps>=self.max_steps or incomplete)and not success and not invalid
        reason='success'if success else'incomplete_evidence'if incomplete else'invalid_task_state'if invalid else None
        info=dict(score=float(result['score']),success=success,wafer_task=result,
                  qualification_diagnostics=self._backend.diagnostics(),physics_evidence=self._backend.evidence_reference())
        return StepResult(observation=observation,reward=float(result['score']),terminated=success or invalid,
                          truncated=truncated,termination_reason=reason,info=info)

    def _observation(self):
        images,state,timestamp=self._backend.observe()
        return Observation(images=images,state={STATE_KEY:state},instruction=self._instruction,
                           image_times=dict.fromkeys(images,timestamp),state_time=timestamp)

    def close(self):
        self._done=True
        if self._backend is not None:self._backend.close();self._backend=None


class RecordedNativePolicy(PolicyBase):
    """Replay verified joint targets through physics; never replay object poses."""
    config=PolicyConfig(action_horizon=1)

    def __init__(self,*,run,candidate_bundle):
        from wafer_benchmark_backend import load_recorded_targets
        self.run=Path(run).expanduser().resolve()
        data=load_recorded_targets(self.run,candidate_bundle)
        self.targets=data['targets'];self.initial=data['initial'];self.source_hashes=data['source_hashes']
        self.source_revision=data['source_revision'];self.execution_revision=data['execution_revision']
        self.bridge=data['bridge'];self.source_seed=data['seed'];self.index=0;self._bound=False
        self.info=PolicyInfo(name='wafer_blow_scripted',action_space=action_box(),control_hz=CONTROL_HZ,
            checkpoint=hashlib.sha256(json.dumps(self.source_hashes,sort_keys=True).encode()).hexdigest(),
            terminal_actions=frozenset({DONE}))

    def bind(self,embodiment_info):
        if embodiment_info.name!='wafer_blow_arena'or embodiment_info.control_hz!=CONTROL_HZ:
            raise ValueError('Recorded targets require this exact20 Hz joint interface.')
        if embodiment_info.environment_revision!=self.execution_revision:
            raise ValueError(f'Recorded replay requires this candidate bundle and explicit layout_seed={self.source_seed}; the active configuration differs.')
        for target in self.targets:
            if np.any(target<embodiment_info.action_space.low)or np.any(target>embodiment_info.action_space.high):
                raise ValueError('A recorded joint target exceeds native joint limits.')
        self.info=PolicyInfo(name=self.info.name,action_space=embodiment_info.action_space,
            observation_space=embodiment_info.observation_space,control_hz=CONTROL_HZ,
            checkpoint=self.info.checkpoint,terminal_actions=frozenset({DONE}))
        self._bound=True

    def reset(self,scene):
        if not self._bound:raise ValueError('Bind the recorded policy to its fixed-layout embodiment before reset.')
        self.index=0

    def act(self,observation):
        if self.index==0 and not np.allclose(validate_dim(observation.state[STATE_KEY]),self.initial,atol=1e-5,rtol=0.):
            raise ValueError('Recorded policy initial measured joints differ.')
        if self.index>=len(self.targets):
            return ActionChunk(actions=(),terminal=Termination(DONE,'Joint-target replay ended; measured environment score determines success.'))
        target=self.targets[self.index].copy();self.index+=1
        return ActionChunk(actions=(Action(target),),control_hz=CONTROL_HZ)

    def transcript(self):
        return dict(source='Verified native issued robot targets; no object-state replay.',source_hashes=self.source_hashes,
            source_environment_revision=self.source_revision,execution_environment_revision=self.execution_revision,
            source_layout_seed=self.source_seed,execution_bridge=self.bridge,executed_target_count=self.index,
            target_count=len(self.targets),scope='Source success does not validate the new camera-enabled execution.')


class FinalWaferScore:
    name='wafer_blow_score'
    def __call__(self,record,target):
        if getattr(record,'status',None)=='error':return Score(0.,'Execution error; evidence incomplete.',metadata={'evidence_complete':False})
        if not record.steps:return Score(0.,'No native physics action was executed.')
        result=record.steps[-1].result.info.get('wafer_task')
        if not isinstance(result,dict):raise ValueError('The trial has no measured task score.')
        value=float(result['score'])
        if not math.isfinite(value)or not 0.<=value<=1.:raise ValueError('Task score is outside[0,1].')
        if result.get('invalid_reasons')or result.get('evidence_complete')is not True:value=0.
        return Score(value,'Measured handling, trigger operation and visible-marker removal; partial credit, no nanocleanliness claim.',metadata=result)


class HoldCurrentPolicy(PolicyBase):
    config=PolicyConfig(action_horizon=1)
    def bind(self,embodiment_info):
        self.info=PolicyInfo(name='wafer_blow_hold',action_space=embodiment_info.action_space,
            observation_space=embodiment_info.observation_space,control_hz=CONTROL_HZ)
    def reset(self,scene):pass
    def act(self,observation):
        return ActionChunk(actions=(Action(validate_dim(observation.state[STATE_KEY]).copy()),),control_hz=CONTROL_HZ)


def wafer_blow_arena(**kwargs):return WaferBlowArena(**kwargs)
def wafer_blow_scripted(**kwargs):return RecordedNativePolicy(**kwargs)
def wafer_blow_hold(**kwargs):
    if kwargs:raise ValueError('Hold policy takes no options.')
    return HoldCurrentPolicy()
def wafer_blow_score(**kwargs):
    if kwargs:raise ValueError('Scoring thresholds are fixed by the benchmark.')
    return FinalWaferScore()
