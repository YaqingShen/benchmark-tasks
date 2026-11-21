"""Passive elastic-perfectly-plastic scored hinge, integrated at physics steps.

This is an uncalibrated reduced material model, not an angle-dependent latch.
Return mapping depends only on elastic strain; no preferred assembly angles.
Elastic energy is 0.5*K*(q-p)^2; plastic dissipation is My*abs(dp) >= 0.
No panel poses, velocities, root state, or grasp constraints are written.
"""
import torch
from isaaclab.actuators import ImplicitActuator, ImplicitActuatorCfg
from isaaclab.utils import configclass

class PassiveCrease(ImplicitActuator):
 def __init__(self,*args,**kwargs):
  super().__init__(*args,**kwargs)
  self.plastic_rotation=torch.zeros_like(self.stiffness)
  self.plastic_dissipation=torch.zeros_like(self.stiffness)

 def reset(self,env_ids=None):
  if hasattr(self,'plastic_rotation'):
   ids=slice(None) if env_ids is None else env_ids
   self.plastic_rotation[ids]=0.;self.plastic_dissipation[ids]=0.

 def compute(self,control_action,joint_pos,joint_vel):
  limit=self.cfg.elastic_yield_rotation
  strain=joint_pos-self.plastic_rotation
  increment=strain-torch.clamp(strain,-limit,limit)
  self.plastic_rotation+=increment
  self.plastic_dissipation+=self.stiffness*limit*torch.abs(increment)
  control_action.joint_positions=self.plastic_rotation.clone()
  control_action.joint_velocities=torch.zeros_like(joint_vel)
  control_action.joint_efforts=torch.zeros_like(joint_pos)
  return super().compute(control_action,joint_pos,joint_vel)

@configclass
class PassiveCreaseCfg(ImplicitActuatorCfg):
 class_type:type=PassiveCrease
 elastic_yield_rotation:float=.20
