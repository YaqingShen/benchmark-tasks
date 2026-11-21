"""Functional approximation of a flip maker's rotation-operated lid lock.

The manual specifies rotation to lock, and returning upright before opening.
Its internal cam profile is not published. This model uses passive joint limits
instead of inventing a separately operated spring slider. The numerical 7 deg
opening zone and 0.2 deg lid clearance are declared simulation tolerances, not
measurements of a production appliance. No joint pose or velocity is written.
"""
import math
from appliance_config import MODEL

HOME_ZONE = math.radians(MODEL["lock_release_zone_deg"])
OPEN_SPIN_STOP = math.radians(MODEL["open_body_stop_deg"])
CLOSED_CLEARANCE = math.radians(MODEL["locked_lid_clearance_deg"])
LID_MAX = MODEL["lid_open_stop_rad"]


def engaged(spin, lid, previously_locked=False):
    if spin <= HOME_ZONE:
        return False
    return previously_locked or lid <= CLOSED_CLEARANCE


def limits(spin, lid, previously_locked=False):
    # If a bad state crosses both limits, leave it visible to the scorer rather
    # than snapping it shut. The next simulation step applies joint constraints.
    lid_upper = CLOSED_CLEARANCE if engaged(spin, lid, previously_locked) else LID_MAX
    spin_upper = OPEN_SPIN_STOP if lid > CLOSED_CLEARANCE and spin <= HOME_ZONE else math.pi
    return ((0.0, spin_upper), (0.0, lid_upper))


def configure(ns):
    last = [None]
    locked = [False]

    def apply(env):
        import torch
        art = env.unwrapped.scene['waffle']
        state = ns['labkit']._part_state(env, next(o for o in ns['OBJECTS'] if o.name == 'waffle'))
        spin, lid = state.joints['spin'], state.joints['lid']
        locked[0] = engaged(spin, lid, locked[0])
        bounds = limits(spin, lid, locked[0])
        if bounds == last[0]:
            return
        ids = [list(art.joint_names).index(name + '_joint') for name in ('spin', 'lid')]
        art.write_joint_position_limit_to_sim_index(
            limits=torch.tensor([bounds], dtype=torch.float32, device=env.unwrapped.scene.env_origins.device),
            joint_ids=ids, warn_limit_violation=False)
        last[0] = bounds

    def reset(env):
        last[0] = None
        locked[0] = False
        apply(env)

    ns['on_reset'](reset)
    ns['on_step'](apply)
