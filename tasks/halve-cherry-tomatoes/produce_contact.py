"""Passive resistance between an intact cherry tomato and the working edge (SI units)."""
import math
import numpy as np
from cut_checks import RADIUS, BLADE_HALF_LENGTH

SKIN_PEAK = 1.6
FLESH_DRAG = 0.35
SKIN_DEPTH = 0.0025
MIN_DRAW = 0.003
MIN_WORK = 0.004

def contact(local, relative_speed):
    """Return forward load and depth using the measured relative approach velocity."""
    x, y, z = local
    if abs(z) >= RADIUS or abs(x) > BLADE_HALF_LENGTH:
        return 0.0, 0.0
    chord = math.sqrt(RADIUS * RADIUS - z * z)
    depth = chord - y
    if depth <= 0.0 or depth >= 2 * chord:
        return 0.0, depth
    if depth < SKIN_DEPTH:
        load = SKIN_PEAK * depth / SKIN_DEPTH
    else:
        load = FLESH_DRAG + (SKIN_PEAK - FLESH_DRAG) * math.exp(-(depth - SKIN_DEPTH) / 0.0015)
    # Smooth the change of direction over 3 mm/s; the force cannot drive the edge forward.
    return load * math.tanh(relative_speed / 0.003), depth

class FruitContact:
    """Track physical work and the range of the drawing motion for each traversal."""
    def __init__(self):
        self.work = np.zeros(8)
        self.draw = np.zeros(8)
        self.origin = [None] * 8
        self.low = np.zeros(8)
        self.high = np.zeros(8)
        self.peak = np.zeros(8)
        self.peak_total = 0.
        self.active_max = 0
        self.physical_steps = 0

    def update(self, i, local, speed, dt):
        load, depth = contact(local, speed)
        if depth <= 0:
            self.origin[i] = None
            self.work[i] = self.draw[i] = 0.
        elif load:
            if self.origin[i] is None:
                self.origin[i] = local[0]
                self.low[i] = self.high[i] = local[0]
            self.low[i] = min(self.low[i], local[0])
            self.high[i] = max(self.high[i], local[0])
            self.draw[i] = self.high[i] - self.low[i]
            self.work[i] += max(load, 0.) * max(speed, 0.) * dt
            self.peak[i] = max(self.peak[i], abs(load))
        return load

    def ready(self):
        return (self.work >= MIN_WORK) & (self.draw >= MIN_DRAW)

def install_physics(env, trial):
    """Apply contact on the same physics steps that take the gripper commands."""
    import torch
    from scipy.spatial.transform import Rotation
    import labkit
    base = env.unwrapped
    scene = base.scene
    if getattr(scene, '_produce_contact_installed', False):
        return
    original = scene.write_data_to_sim
    bodies = [scene[f'tomato_{i}'] for i in range(8)]
    knife = scene['knife']
    dt = float(base.physics_dt)

    def apply(obj, force, point):
        def tensor(x):
            return torch.as_tensor(x, device=base.device, dtype=torch.float32).reshape(1, 1, 3)
        obj.instantaneous_wrench_composer.set_forces_and_torques_index(
            forces=tensor(force), positions=tensor(point), is_global=True)

    def advance(*args, **kwargs):
        model = trial.contact
        model.physical_steps += 1
        p = labkit._np(knife.data.root_link_pose_w)[0]
        rot = Rotation.from_quat(p[3:]).as_matrix()
        # Broad phase around the tray. Do not apply fruit loads to tools elsewhere on the worktop.
        tray = trial.world((.071,0,.094))
        if abs(p[0] - tray[0]) < .105 and abs(p[2] - tray[2]) < .03 and len(trial.cut) < 8:
            vel = labkit._np(knife.data.root_vel_w)[0]
            com = labkit._np(knife.data.root_com_pose_w)[0,:3]
            forward = rot[:,1]
            loads, points = [], []
            for i, obj in enumerate(bodies):
                if i in trial.cut:
                    continue
                center = labkit._np(obj.data.root_link_pose_w)[0,:3]
                local = (center - p[:3]) @ rot
                point = p[:3] + local[0] * rot[:,0]
                speed = 0.
                if abs(local[1]) < RADIUS and abs(local[2]) < RADIUS:
                    fruit_vel = labkit._np(obj.data.root_vel_w)[0]
                    edge_vel = vel[:3] + np.cross(vel[3:], point - com)
                    tissue_vel = fruit_vel[:3] + np.cross(fruit_vel[3:], point - center)
                    speed = float((edge_vel - tissue_vel) @ forward)
                force = model.update(i, local, speed, dt)
                if force:
                    apply(obj, force * forward, point)
                    loads.append(force)
                    points.append(point)
            if loads:
                forces = -np.array(loads)[:,None] * forward
                torque = np.cross(np.array(points) - com, forces).sum(axis=0)
                def vec(x):
                    return torch.as_tensor(x,device=base.device,dtype=torch.float32).reshape(1,1,3)
                knife.instantaneous_wrench_composer.set_forces_and_torques_index(
                    forces=vec(forces.sum(axis=0)), torques=vec(torque), is_global=True)
                model.peak_total = max(model.peak_total, abs(sum(loads)))
                model.active_max = max(model.active_max, len(loads))
        return original(*args, **kwargs)
    scene.write_data_to_sim = advance
    scene._produce_contact_installed = True
