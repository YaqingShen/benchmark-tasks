"""pack_grok002 (by grok): passive foam spring for the pad's prismatic joint.

Linear spring with preload: F = F0 + k*(-q) - c*qdot for compression q<=0 (joint upper limit 0).
Implemented like crease_material.PassiveCrease: an implicit drive whose target is a constant
material rest length (F0/k above the limit). No pose writes, no constraint, no robot help.
"""
import torch
from isaaclab.actuators import ImplicitActuator, ImplicitActuatorCfg
from isaaclab.utils import configclass

class FoamSpring(ImplicitActuator):
    def compute(self, control_action, joint_pos, joint_vel):
        control_action.joint_positions = torch.full_like(joint_pos, self.cfg.rest_offset_m)
        control_action.joint_velocities = torch.zeros_like(joint_vel)
        control_action.joint_efforts = torch.zeros_like(joint_pos)
        return super().compute(control_action, joint_pos, joint_vel)

@configclass
class FoamSpringCfg(ImplicitActuatorCfg):
    class_type: type = FoamSpring
    rest_offset_m: float = .0104
