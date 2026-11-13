"""Private task scoring from measured native state, never policy observations."""
from collections import deque
import math
import numpy as np
from native_coupling import array, transform
from disposal_geometry import pipette_bounds_in_bin, fully_contained
from pipette_correction import released_disposal_window


def score_measurements(initial_mass, rows, violations=()):
    """Final settled measurement; unchanged resets earn zero."""
    if not rows:
        return {'score': 0., 'success': False, 'reason': 'No task actions yet'}
    end = rows[-1]
    e0 = abs(initial_mass-10.)
    if e0 <= .05:
        raise ValueError('Reset must require a correction')
    error = abs(end['mass_g']-10.)
    quality = 1. if error <= .05 else float(np.clip((e0-error)/(e0-.05), 0., 1.))
    window = list(rows)[-20:]
    settled = len(window) == 20 and window[-1]['time_s']-window[0]['time_s'] >= .94
    settled = settled and all(r['measurement_valid'] for r in window)
    disposed = released_disposal_window(window) if settled else False
    used = end['pipette_correction_g'] >= .01
    reasons = list(violations)
    if end['spill_g'] > .10+1e-9:
        reasons.append('Spill exceeds 0.10 g')
    # A valid final reading must be free of robot/tool loading and agree with
    # conserved liquid mass. This prevents force-on-pan or re-tare shortcuts.
    value = (.9*quality + (.1 if disposed and used and quality > 0. else 0.)) if settled and used and not reasons else 0.
    return {'score': round(value, 8), 'success': bool(value >= 1.-1e-8),
            'initial_mass_g': initial_mass, 'final_mass_g': end['mass_g'],
            'balance_g': end['balance_g'], 'error_g': error,
            'mass_quality': quality, 'settled': bool(settled),
            'pipette_used': bool(used), 'pipette_disposed': bool(disposed),
            'violations': reasons, 'spill_g': end['spill_g']}


class NativeTaskMonitor:
    def __init__(self, base, coupling):
        self.base, self.coupling = base, coupling
        self.initial_mass = coupling.ledger.initial_beaker_g
        self.initial_poses = coupling.poses()
        self.tare = coupling.balance.tare_g
        self.rows = deque(maxlen=20)
        self.violations = set()
        self.steps = 0

    def advance(self):
        from yam_arena import constants
        c = self.coupling
        poses = c.poses()
        self.steps += 1
        for name in ('beaker', 'bottle_A'):
            p = poses[name]
            if p[2, 2] < math.cos(math.radians(5)):
                self.violations.add(name+' tilted over 5 degrees')
            if np.linalg.norm(p[:3, 3]-self.initial_poses[name][:3, 3]) > .005:
                self.violations.add(name+' moved over 5 mm from its support')
        if c.balance.tare_g != self.tare:
            self.violations.add('Original empty-beaker tare changed')
        tool = self.base.scene['pipette_A']
        ids, _ = tool.find_bodies(['bulb_left', 'bulb_right'], preserve_order=True)
        bulbs = [transform(p) for p in array(tool.data.body_link_pose_w)[0, ids]]
        lower, upper = pipette_bounds_in_bin(poses['pipette_A'], bulbs, poses['trash_bin'])
        contained = fully_contained(lower, upper, c.manifest['assets']['trash_bin']['interior_bounds_m'])
        forces = []
        gaps = []
        for side in ('left', 'right'):
            arm = self.base.scene[side+'_robot']
            ids, _ = arm.find_joints(list(constants.FINGER_JOINTS), preserve_order=True)
            gaps.append(float(abs(array(arm.data.joint_pos)[0, ids].mean())*2))
            for finger in ('tip_left', 'tip_right'):
                forces.append(float(np.linalg.norm(array(self.base.scene[side+'_'+finger+'_tool_contact'].data.normal_force_matrix_w), axis=-1).sum()))
        tool_free = max(forces) < .005
        mass = c.ledger.vessels['beaker'].liquid.mass(c.ledger.cfg['density_g_ml'])
        balance = c.balance.last
        external = float(np.linalg.norm(array(self.base.scene['measurement_contacts'].data.normal_force_matrix_w), axis=-1).sum())
        speed = float(np.linalg.norm(array(self.base.scene['beaker'].data.root_link_vel_w)[0, :3]))
        valid = bool(balance.get('stable') and balance.get('net_g') is not None and
                     abs(balance['net_g']-mass) <= .02 and external < .005 and speed < .005)
        row = {'time_s': self.steps/20., 'mass_g': mass, 'balance_g': balance.get('net_g'),
               'measurement_valid': valid, 'external_beaker_contact_n': external,
               'tip_position': poses['pipette_A'][:3, 3].tolist(),
               'tool_axis_world': poses['pipette_A'][:3, 2].tolist(),
               'tool_bin_bounds_min_m': lower.tolist(), 'tool_bin_bounds_max_m': upper.tolist(),
               'finger_tool_contact_n': max(forces), 'gripper_gaps_m': gaps,
               'pipette_contained_and_released': bool(contained and tool_free),
               'pipette_correction_g': c.ledger.pipettes['pipette_A'].correction_g,
               'spill_g': c.ledger.spill.mass(c.ledger.cfg['density_g_ml'])}
        self.rows.append(row)
        return row

    def score(self):
        return score_measurements(self.initial_mass, self.rows, sorted(self.violations))
