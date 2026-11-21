"""Task-process-only YAM finger limit matching the supplied review checklist.

This limits simulated actuator effort. It is not a real force measurement.
The shared workstation package is never edited.
"""
import json
from yam_arena.constants import GRIPPER
FINGER_LIMIT_N=40.0
GRIPPER['effort']=min(float(GRIPPER['effort']),FINGER_LIMIT_N)


def verify(env):
    records={}
    for name in ('left_robot','right_robot'):
        robot=env.unwrapped.scene[name]
        value=float(robot.cfg.actuators['gripper'].joint_effort_limit)
        if value>FINGER_LIMIT_N+1e-6:
            raise RuntimeError(f'{name} gripper effort limit {value} exceeds {FINGER_LIMIT_N} N')
        records[name]={'configured_finger_limit_n':value,'physical_measurement':False}
    print('TASK_HARDWARE_LIMITS '+json.dumps(records),flush=True)
