"""Official Inspect Robots interface; observations are RGB and robot joints only."""
from pathlib import Path
import hashlib
import numpy as np
from inspect_robots.embodiment import EmbodimentBase,EmbodimentInfo,RESETTABLE,PRIVILEGED_SUCCESS
from inspect_robots.policy import PolicyBase,PolicyConfig,PolicyInfo
from inspect_robots.scorer import Score
from inspect_robots.types import Action,ActionChunk,Observation,StepResult
from inspect_robots_yam.config import DEFAULT_CAMERAS,action_box,observation_space
from inspect_robots_yam.packing import STATE_KEY,pack,validate_dim

INSTRUCTION='Place the phone, watch and closed earbuds case on their respective charging positions. All three must remain charging after release, with their charging indicators visible to a camera. Keep the charging stand within 5 mm and 5 degrees of its starting pose.'
DOCS='Two YAM arms with parallel grippers. Actions are 14 absolute targets: left six joints in radians, left gripper (0 closed, 1 open), right six joints, right gripper. Observations are the original top, left-wrist and right-wrist RGB cameras plus measured joint positions. The phone and watch attach magnetically after alignment. The watch band is flexible and must clear its rear charging surface. The earbuds case rests in the shallow base recess. Phone/watch screens show green charging cues; the case briefly shows an amber status LED. No object poses or charging-state variables are exposed to the policy.'

class ChargingArena(EmbodimentBase):
    def __init__(self,bundle,evidence_dir,max_steps=6000,policy_name='dream'):
        from yam_arena import YAM_URDF
        from yam_arena.constants import ARM_JOINTS
        from yam_arena.urdf import parse_joints
        self.root=Path(bundle).resolve();self.output=Path(evidence_dir).resolve();self.max_steps=max_steps;self.world=None;self.reset_count=0;self.done=True
        self.policy_name=policy_name
        joints=parse_joints(YAM_URDF);low=np.r_[[joints[n].lower for n in ARM_JOINTS],0.];high=np.r_[[joints[n].upper for n in ARM_JOINTS],1.]
        digest=hashlib.sha256((self.root/'source_manifest.json').read_bytes()).hexdigest()
        self.info=EmbodimentInfo(name='three_device_charging',action_space=action_box(pack(low,low),pack(high,high)),observation_space=observation_space(720,1280,DEFAULT_CAMERAS),control_hz=20.,is_simulated=True,capabilities=frozenset({RESETTABLE,PRIVILEGED_SUCCESS}),docs=DOCS,environment_id='isaaclab-three-device-charging',environment_revision=digest,morphology='yam-bimanual')

    def reset(self,scene,*,seed=None):
        from charging_world import ChargingWorld
        if self.world:self.world.close()
        selected_seed=seed if seed is not None else scene.init_seed
        self.world=ChargingWorld(self.root,self.output/f'trial-{self.reset_count:03d}',seed=17 if selected_seed is None else int(selected_seed))
        import json
        self.world.metadata.update(dream1_run=self.policy_name=='dream',controller=self.policy_name,robot_observation_only=True)
        (self.world.output/'metadata.json').write_text(json.dumps(self.world.metadata,indent=2))
        self.reset_count+=1;self.instruction=scene.instruction;self.done=False
        self.policy_steps=0
        target=self.world.joints()
        for _ in range(40):self.world.step(target,'reset settling')
        self.world.initial_stand=self.world.coupling.poses()['belkin'].copy();self.world.max_shift=self.world.max_rotation=0.
        self.world.snapshot('setup');return self._observation()

    def _observation(self):
        images=self.world.images();t=self.world.steps/20.
        return Observation(images=images,state={STATE_KEY:self.world.joints().astype(np.float32)},instruction=self.instruction,image_times=dict.fromkeys(images,t),state_time=t)

    def step(self,action):
        if self.done:raise RuntimeError('Reset before acting')
        target=validate_dim(action.data)
        if not np.isfinite(target).all():raise ValueError('Nonfinite policy action')
        self.world.step(np.clip(target,self.info.action_space.low,self.info.action_space.high),'policy')
        self.policy_steps+=1
        result=self.world.score();result['policy_steps']=self.policy_steps;success=result['success'];truncated=self.policy_steps>=self.max_steps and not success;self.done=success or truncated
        return StepResult(observation=self._observation(),reward=result['score'],terminated=success,truncated=truncated,termination_reason='success' if success else None,info={'score':result['score'],'success':success,'charging_task':result})

    def close(self):
        if self.world:self.world.snapshot('final');self.world.close();self.world=None

class ChargingScore:
    name='charging_score'
    def __call__(self,record,target):
        if getattr(record,'status',None)=='error' or not record.steps:return Score(0.,'No completed physical evidence')
        result=record.steps[-1].result.info['charging_task']
        return Score(float(result['score']),'Released and aligned charging devices with a stable stand.',metadata=result)

class HoldPolicy(PolicyBase):
    config=PolicyConfig(action_horizon=1)
    def bind(self,info):self.info=PolicyInfo(name='charging_hold',action_space=info.action_space,observation_space=info.observation_space,control_hz=20.)
    def reset(self,scene):pass
    def act(self,observation):return ActionChunk(actions=(Action(validate_dim(observation.state[STATE_KEY]).copy()),),control_hz=20.)
