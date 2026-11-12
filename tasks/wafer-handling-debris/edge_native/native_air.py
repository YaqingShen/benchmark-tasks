"""Existing contact-held gun and breakable visible markers for the new pick."""
from pathlib import Path
import copy
import hashlib
import math
import numpy as np
from native_adapter import _host_snapshot, pose_matrix, matrix_pose


def _convex_inertia(particle):
    tri = np.asarray(particle['vertices_local_m'])[np.asarray(particle['faces'])]
    volume = np.einsum('ij,ij->i', tri[:, 0], np.cross(tri[:, 1], tri[:, 2])) / 6.
    summed = tri.sum(axis=1)
    moment = np.einsum('f,fij->ij', volume / 20.,
        np.einsum('fki,fkj->fij', tri, tri) + np.einsum('fi,fj->fij', summed, summed))
    moment *= particle['mass_kg'] / volume.sum()
    return np.trace(moment) * np.eye(3) - moment


class GunContactInterval:
    """Time means from the 100 existing physics samples in one robot action."""
    dt = .0005
    samples = 100

    def __init__(self):
        self.clear()

    def clear(self):
        self.count = 0
        self.stand_vertical_sum = self.stand_normal_peak = 0.
        self.grip_sum = np.zeros(2)

    def add(self, stand_matrix, tip_matrices):
        stand = np.asarray(stand_matrix, dtype=float)
        tips = np.asarray(tip_matrices, dtype=float)
        if (stand.shape != (1, 1, 3, 3) or tips.shape != (2, 1, 1, 2, 3)
                or not np.isfinite(stand).all() or not np.isfinite(tips).all()):
            raise ValueError('Invalid gun contact interval sample.')
        if self.count >= self.samples:
            raise RuntimeError('Gun contact interval has more than 100 physics samples.')
        support = stand[0, 0, 2]
        grip = [np.linalg.norm(tips[0, 0, 0, 1]), np.linalg.norm(tips[1, 0, 0, 0])]
        self.stand_vertical_sum += float(support[2])
        self.stand_normal_peak = max(self.stand_normal_peak, float(np.linalg.norm(support)))
        self.grip_sum += grip
        self.count += 1

    def consume(self, *, initial=False):
        if not (self.count == self.samples or initial and self.count == 0):
            raise RuntimeError('Gun contact interval must contain exactly 100 physics samples.')
        divisor = max(1, self.count)
        result = dict(count=self.count, duration_s=self.count * self.dt,
            stand_vertical_mean_N=self.stand_vertical_sum / divisor,
            stand_normal_peak_N=self.stand_normal_peak,
            grip_mean_N=(self.grip_sum / divisor).tolist())
        self.clear()
        return result


class NativeAir:
    def __init__(self, baseline, full_plan, metadata, seed=17):
        from wafer_cycle_isaac import Cycle
        from wafer_dust import mixed_layout
        from mixed_debris_assets import export_mixed_assets
        self.gun = Cycle()
        self.gun.root = Path(baseline)
        self.gun.gun = full_plan['gun']
        self.gun.export = full_plan['gun_export']
        self.gun.scan_row = -1
        self.metadata = metadata
        metadata['gun'] = self.gun.gun
        self.root = Path(__file__).resolve().parent
        self.dust = mixed_layout(seed)
        export_mixed_assets(self.root, self.dust)
        self.names = list(self.dust['ids'])
        self.anchors = np.asarray(self.dust['initial_centers_wafer_m'])
        self.profile = copy.deepcopy(self.dust['profile'])
        metadata['dust'] = self.dust
        metadata['dust']['post_start_particle_pose_writes'] = 0
        self.first_air = None
        self.total = self.substeps = 0
        self.original_update = None
        self.integrity = None
        self.contact_interval = GunContactInterval()
        metadata['gun_contact_interval'] = dict(physics_dt_s=.0005, control_interval_s=.05,
            samples_per_interval=100, raw_fields_preserved=True,
            source='Existing scene.update contact sensors: gun body/stand; tip0/trigger and tip1/gun body.',
            scope='Engineering contact means and stand-force peak; no additional physics steps or task-score thresholds.')

    def configure(self, objects, embodiment):
        import torch
        from isaaclab.actuators import ImplicitActuatorCfg
        from isaaclab.sensors import ContactSensorCfg
        from isaaclab.sim import MassPropertiesCfg
        from isaaclab_arena.assets.object import Object
        from isaaclab_arena.assets.object_type import ObjectType
        from isaaclab_arena.utils.pose import Pose
        sensors = self.gun.configure(objects, embodiment, Pose, Object, ObjectType,
                                     ImplicitActuatorCfg, ContactSensorCfg, torch)
        initial = objects['wafer'].initial_pose
        wafer = pose_matrix([*initial.position_xyz, *initial.rotation_xyzw])
        for particle, anchor in zip(self.dust['particles'], self.anchors):
            pose = matrix_pose(wafer @ pose_matrix([*anchor, *particle['initial_rotation_xyzw']]))
            objects[particle['id']] = Object(name=particle['id'],
                usd_path=str(self.root / particle['asset_file']),
                scale=tuple(particle['scale_m']), object_type=ObjectType.RIGID,
                spawn_cfg_addon={'mass_props': MassPropertiesCfg(mass=particle['mass_kg'])},
                initial_pose=Pose.from_dict(dict(position_xyz=pose[:3], rotation_xyzw=pose[3:])))
        self.objects = objects
        return sensors

    def bind(self, base):
        import torch
        import omni.usd
        from pxr import Usd, UsdPhysics, UsdGeom
        from wafer_dust import SurfaceAdhesion
        from wafer_benchmark_backend import _GeometryIntegrity
        self.base, self.torch = base, torch
        self.gun.ready(base, self.metadata, torch)
        self.particles = [base.scene[n] for n in self.names]
        self.wafer = base.scene['wafer']
        self.left = base.scene['left_robot']
        self.bonds = SurfaceAdhesion(self.anchors, self.profile)
        stage = omni.usd.get_context().get_stage()
        roots = {n: self.objects[n].prim_path.replace('{ENV_REGEX_NS}', '/World/envs/env_0') for n in self.names}
        for particle in self.dust['particles']:
            name, path = particle['id'], roots[particle['id']]
            prim = stage.GetPrimAtPath(path)
            rb = UsdPhysics.RigidBodyAPI(prim)
            if not prim or rb.GetKinematicEnabledAttr().Get() or not rb.GetRigidBodyEnabledAttr().Get():
                raise RuntimeError('Marker must be a free dynamic body: ' + name)
            if abs(float(UsdPhysics.MassAPI(prim).GetMassAttr().Get()) - particle['mass_kg']) > max(1e-14, particle['mass_kg'] * 1e-6):
                raise RuntimeError('Marker mass differs.')
            UsdGeom.Gprim(stage.GetPrimAtPath(path + '/Particle')).GetDisplayColorAttr().Set([tuple(particle['color_rgb'])])
            if any(p.IsA(UsdPhysics.Joint) for p in Usd.PrimRange(prim)):
                raise RuntimeError('Marker attachment joint is forbidden.')
        for prim in stage.Traverse():
            if prim.IsA(UsdPhysics.Joint):
                joint = UsdPhysics.Joint(prim)
                targets = [str(p) for rel in (joint.GetBody0Rel(), joint.GetBody1Rel()) for p in rel.GetTargets()]
                if any(target == path or target.startswith(path + '/') for target in targets for path in roots.values()):
                    raise RuntimeError('An external joint attaches a marker.')
        inertial_fields = [(p['id'] + '_mass', asset.data.body_mass)
                           for p, asset in zip(self.dust['particles'], self.particles)]
        inertial_fields += [(p['id'] + '_inertia', asset.data.body_inertia)
                            for p, asset in zip(self.dust['particles'], self.particles)]
        measured = _host_snapshot(inertial_fields)
        self.dust['native_inertia'] = {}
        for particle in self.dust['particles']:
            name = particle['id']
            mass = float(measured[name + '_mass'].reshape(-1)[0])
            actual = measured[name + '_inertia'].reshape(3, 3)
            expected = _convex_inertia(particle)
            eigen_actual, eigen_expected = np.linalg.eigvalsh(actual), np.linalg.eigvalsh(expected)
            self.dust['native_inertia'][name] = dict(mass_kg=mass,
                inertia_kg_m2=actual.tolist(), expected_eigenvalues_kg_m2=eigen_expected.tolist(),
                actual_eigenvalues_kg_m2=eigen_actual.tolist())
            if (not np.isclose(mass, particle['mass_kg'], rtol=1e-6, atol=1e-15)
                    or np.any(eigen_actual <= 0)
                    or not np.allclose(eigen_actual, eigen_expected, rtol=.02, atol=1e-20)):
                raise RuntimeError('Native scaled particle mass/inertia differs: ' + name)
        self.integrity = _GeometryIntegrity(stage, roots)
        self.stand_origin = np.asarray(self.gun.export['initial_world_poses']['gun_stand'][:3])
        self.wrench = torch.zeros((13, 2, 1, 3), device=base.device, dtype=torch.float32)
        self.expected_wrench = np.zeros((13, 2, 1, 3), dtype=np.float32)
        self.force_digest = hashlib.sha256()
        self.force_audit = dict(status='ACTIVE', prepared_intervals=0, completed_intervals=0,
            readbacks=0, maximum_readback_error=0., maximum_unexpected_wrench=0.,
            maximum_particle_force_N=0., maximum_wafer_reaction_N=0., maximum_wafer_torque_Nm=0.,
            scope='Online composer readback before each existing physics update callback; source-bound force calculation, not an independent solver-force measurement or task score.')
        import wafer_dust
        self.force_audit['equation_source_sha256'] = hashlib.sha256(Path(wafer_dust.__file__).read_bytes()).hexdigest()
        self.metadata['particle_force_audit'] = self.force_audit
        self.metadata['gun_native_export'] = self.gun.export
        self.metadata['gun_link_pose_names'] = ['blow_gun', 'gun_trigger']
        self.metadata['particle_force_timing'] = 'After each existing 0.5 ms physics update, applied for next interval.'

    def verify_wrench(self):
        import warp as wp
        fields = []
        owned = [*self.particles, self.wafer]
        channels = ('global_force_at_com_w', 'global_torque_w', 'global_force_w', 'local_force_b', 'local_torque_b')
        for i, asset in enumerate([*owned, self.base.scene['clamp'], self.left,
                                   self.base.scene['right_robot'], self.gun.gunasset]):
            for channel in channels:
                value = getattr(asset.permanent_wrench_composer, channel)
                fields.append((f'{i}_{channel}', value if isinstance(value, self.torch.Tensor) else wp.to_torch(value)))
        measured = _host_snapshot(fields)
        error, unexpected = 0., 0.
        for i in range(13):
            for component, channel in enumerate(channels[:2]):
                actual = measured[f'{i}_{channel}']
                if actual.shape != (1, 1, 3):
                    raise RuntimeError('Unexpected particle/wafer wrench shape.')
                error = max(error, float(np.max(np.abs(actual - self.expected_wrench[i, component:component + 1]))))
        for name, value in measured.items():
            index, channel = name.split('_', 1)
            if int(index) >= 13 or channel not in channels[:2]:
                unexpected = max(unexpected, float(np.max(np.abs(value))))
        self.force_audit['readbacks'] += 1
        self.force_audit['maximum_readback_error'] = max(self.force_audit['maximum_readback_error'], error)
        self.force_audit['maximum_unexpected_wrench'] = max(self.force_audit['maximum_unexpected_wrench'], unexpected)
        if error > 1e-10 or unexpected != 0.:
            self.force_audit['status'] = 'FAILED'
            raise RuntimeError(f'External wrench integrity failed: readback {error:g}, unexpected {unexpected:g}.')

    def force_row(self):
        fields = [('wafer_pose', self.wafer.data.root_link_pose_w),
            ('wafer_lin', self.wafer.data.body_link_lin_vel_w),
            ('wafer_ang', self.wafer.data.body_link_ang_vel_w),
            ('wafer_com', self.wafer.data.body_com_pos_w),
            ('gun_pose', self.gun.gunasset.data.body_link_pose_w),
            ('gun_q', self.gun.gunasset.data.joint_pos)]
        for i, asset in enumerate(self.particles):
            fields.extend([(f'p{i}', asset.data.root_link_pose_w), (f'v{i}', asset.data.body_link_lin_vel_w)])
        a = _host_snapshot(fields)
        g = pose_matrix(a['gun_pose'][0, self.gun.gun_body])
        return dict(wafer_pose_world=a['wafer_pose'][0].tolist(),
            dust_position_world_m=[a[f'p{i}'][0, :3].tolist() for i in range(12)],
            dust_pose_world=[a[f'p{i}'][0].tolist() for i in range(12)],
            dust_velocity_world_m_s=[a[f'v{i}'][0, 0].tolist() for i in range(12)],
            dust_wafer_link_velocity_world_m_s=a['wafer_lin'][0, 0].tolist(),
            dust_wafer_angular_velocity_world_rad_s=a['wafer_ang'][0, 0].tolist(),
            dust_wafer_com_world_m=a['wafer_com'][0, 0].tolist(),
            nozzle_tip_world_m=(g[:3, 3] + g[:3, :3] @ self.gun.gun['nozzle_local_m']).tolist(),
            nozzle_axis_world=g[:3, 0].tolist(),
            air_on=bool(a['gun_q'][0, self.gun.trigger_id] >= self.gun.gun['air_on_compression_m']),
            dust_bonds_bound=self.bonds.bound.tolist())

    def apply_forces(self):
        from wafer_dust import cohesive_forces
        self.verify_wrench()
        row = self.force_row()
        if row['air_on']:
            self.first_air = self.total * .0005 if self.first_air is None else self.first_air
        w = pose_matrix(row['wafer_pose_world'])
        local = (np.asarray(row['dust_position_world_m']) - w[:3, 3]) @ w[:3, :3]
        if self.first_air is None and (not self.bonds.bound.all() or np.any(np.linalg.norm(local[:, :2], axis=1) > .05)
                                       or np.any(np.abs(local[:, 2] - self.anchors[:, 2]) > .002)):
            raise RuntimeError('A marker left the wafer before the gun was operated.')
        force, jet, reaction, torque = cohesive_forces(self.bonds, row, self.profile)
        values = np.zeros((13, 2, 1, 3), dtype=np.float32)
        values[:12, 0, 0] = force
        values[12, 0, 0] = reaction
        values[12, 1, 0] = torque
        if not np.isfinite(values).all():
            raise RuntimeError('Non-finite particle force calculation.')
        self.wrench.copy_(self.torch.from_numpy(values).to(self.base.device))
        for i, asset in enumerate([*self.particles, self.wafer]):
            asset.permanent_wrench_composer.reset()
            asset.permanent_wrench_composer.add_forces_and_torques_index(
                self.wrench[i, 0:1], self.wrench[i, 1:2], is_global=True)
        self.expected_wrench = values.copy()
        for name in ('wafer_pose_world', 'dust_position_world_m', 'dust_pose_world', 'dust_velocity_world_m_s',
                     'dust_wafer_link_velocity_world_m_s', 'dust_wafer_angular_velocity_world_rad_s',
                     'dust_wafer_com_world_m', 'nozzle_tip_world_m', 'nozzle_axis_world'):
            self.force_digest.update(np.asarray(row[name], dtype='<f8').tobytes())
        self.force_digest.update(bytes([row['air_on']]))
        self.force_digest.update(self.bonds.bound.tobytes())
        self.force_digest.update(values.astype('<f4').tobytes())
        self.force_audit['prepared_intervals'] += 1
        self.force_audit['maximum_particle_force_N'] = max(self.force_audit['maximum_particle_force_N'], float(np.linalg.norm(force, axis=1).max()))
        self.force_audit['maximum_wafer_reaction_N'] = max(self.force_audit['maximum_wafer_reaction_N'], float(np.linalg.norm(reaction)))
        self.force_audit['maximum_wafer_torque_Nm'] = max(self.force_audit['maximum_wafer_torque_Nm'], float(np.linalg.norm(torque)))

    def hook(self):
        self.original_update = self.base.scene.update
        self.had_instance_update = 'update' in self.base.scene.__dict__
        self.instance_update = self.base.scene.__dict__.get('update')
        def update(dt, *args, **kwargs):
            result = self.original_update(dt, *args, **kwargs)
            if abs(float(dt) - .0005) > 1e-12:
                raise RuntimeError('Unexpected particle-force update cadence.')
            self.total += 1; self.substeps += 1
            self.force_audit['completed_intervals'] = self.total
            contacts = _host_snapshot([
                ('stand', self.gun.gun_sensors[0].data.normal_force_matrix_w),
                ('tip0', self.gun.tip_sensors[0].data.normal_force_matrix_w),
                ('tip1', self.gun.tip_sensors[1].data.normal_force_matrix_w)])
            self.contact_interval.add(contacts['stand'], [contacts['tip0'], contacts['tip1']])
            self.apply_forces()
            return result
        self.base.scene.update = update
        self.apply_forces()

    def close(self):
        from mixed_debris_assets import particle_asset_bindings
        if self.original_update is not None:
            if self.had_instance_update:
                self.base.scene.update = self.instance_update
            else:
                del self.base.scene.__dict__['update']
            self.original_update = None
        try:
            particle_asset_bindings(self.root, self.dust)
            if hasattr(self, 'force_audit'):
                self.verify_wrench()
                self.force_audit['status'] = 'READBACKS_MATCHED' if self.force_audit['status'] != 'FAILED' else 'FAILED'
                self.force_audit['input_output_sha256'] = self.force_digest.hexdigest()
        finally:
            if self.integrity is not None:
                self.integrity.close()

    def measurement(self, row):
        fields = [(n, v) for n, v, _ in self.gun.fields()]
        fields.extend([('left_targets', self.left.data.joint_pos_target),
            ('gun_targets', self.gun.gunasset.data.joint_pos_target),
            ('gun_lin', self.gun.gunasset.data.body_link_lin_vel_w),
            ('gun_ang', self.gun.gunasset.data.body_link_ang_vel_w)])
        a = _host_snapshot(fields)
        row.update(self.gun.measurement(a, row, row['phase']))
        if abs(float(a['gun_targets'][0, self.gun.trigger_id])) > 1e-12:
            raise RuntimeError('Passive gun trigger spring target changed.')
        row['gun_link_poses_world'] = [a['gun_poses'][0, self.gun.gunasset.find_bodies(name)[0][0]].tolist()
                                       for name in self.metadata['gun_link_pose_names']]
        row.update(self.force_row())
        present, intact = self.integrity.snapshot()
        row['dust_intact'] = all(present.values()) and all(intact.values())
        row['measured_right_finger_q_m'] = a['right_q'][0, self.gun.right_finger_ids].tolist()
        shape = self.gun.export['support_geometry']
        connector, stand = shape['gun_hose_connector'], np.asarray(shape['gun_stand']['half_sizes_m'])
        gun = pose_matrix(row['gun_pose_world'])
        center = gun[:3, 3] + gun[:3, :3] @ np.asarray(connector['local_position_m'])
        axis = (gun[:3, :3] @ np.asarray(connector['local_rotation']))[:, 2]
        radius = connector['radius_m']
        extent = radius * math.sqrt(max(0., 1 - axis[2]**2)) + connector['halfheight_m'] * abs(axis[2])
        force = a['gun_contacts_0'][0, 0, 2]
        row.update(gun_stand_vertical_force_N=float(force[2]), gun_stand_normal_force_N=float(np.linalg.norm(force)),
            gun_foot_gap_m=float(center[2] - extent - self.stand_origin[2] - stand[2]),
            gun_support_margin_m=float(np.min(stand[:2] - np.abs(center[:2] - self.stand_origin[:2]) - radius)),
            gun_linear_speed_m_s=float(np.linalg.norm(a['gun_lin'][0, self.gun.gun_body])),
            gun_angular_speed_rad_s=float(np.linalg.norm(a['gun_ang'][0, self.gun.gun_body])))
        self.gun.guard(row)
        row['gun_contact_interval'] = self.contact_interval.consume(initial=self.total == 0)
        return row
