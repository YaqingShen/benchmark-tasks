"""Read-only, 20 Hz evidence for the normally closed native wafer pick."""
import copy
import hashlib
from pathlib import Path
import numpy as np

from wafer_benchmark_backend import _GeometryIntegrity
from wafer_debris_score import HANDS
from wafer_grasp_protocol import _pose
from wafer_robot_geometry import RobotGeometryGuard, compact_result


def _host(value):
    value = value.torch if hasattr(value, 'torch') else value
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    elif hasattr(value, 'numpy'):
        value = value.numpy()
    result = np.asarray(value, dtype=float)
    if not np.isfinite(result).all():
        raise ValueError('Non-finite native evidence.')
    return result


def _path(value):
    return str(value).replace('{ENV_REGEX_NS}', '/World/envs/env_0').replace(
        '{ENV_NS}', '/World/envs/env_0').replace('/env_.*/', '/env_0/')


def _robot_roots(materials):
    roots = {}
    for side in ('left', 'right'):
        record = materials[side + '_robot']
        if record.get('status') != 'COMPOSED_BINDINGS_VERIFIED' or record.get('callback_count') != 1:
            raise ValueError('A verified whole-robot spawn root is required: ' + side)
        path = record['root_path']
        if not isinstance(path, str) or not path.startswith('/') or any(c in path for c in '*{}'):
            raise ValueError('Unresolved actual robot spawn path: ' + side)
        roots[side] = path
    if len(set(roots.values())) != 2:
        raise ValueError('Robot spawn roots are not distinct.')
    return roots


def _contact_center(guard, side, poses, hand_pose):
    hand = _pose(hand_pose, side + ' hand')
    centers = []
    for index, tip in enumerate(('tip_left', 'tip_right')):
        key = (side, tip)
        link = _pose(poses[key], side + ' fingertip')
        vertices = np.vstack([m.vertices for m in guard.meshes[guard.keys.index(key)]])
        vertices = (vertices @ link[:3, :3].T + link[:3, 3] - hand[:3, 3]) @ hand[:3, :3]
        distal = vertices[vertices[:, 2] >= vertices[:, 2].max() - .025]
        plane = distal[:, 0].max() if index == 0 else distal[:, 0].min()
        face = distal[np.abs(distal[:, 0] - plane) <= .0001]
        if len(face) < 4 or np.ptp(face[:, 2]) < .010 or np.ptp(face[:, 1]) < .010:
            raise ValueError('No measured distal finger contact patch: ' + side + '/' + tip)
        centers.append((face.min(axis=0) + face.max(axis=0)) / 2)
    return np.mean(centers, axis=0).tolist()


class NativeEvidence:
    """Attach actual robot/fixture/integrity reads; call after each completed action.

    ``objects`` contains every real Object, including the twelve dust objects.
    The supplied row must already contain measured tool, wafer, gun and dust data.
    The first attach at t=0 completes metadata; no simulation state is changed.
    """
    def __init__(self, base, adapter, objects, dust_metadata, gun_metadata, *, robot_roots=None, robot_materials=None):
        import omni.usd
        from isaaclab_physx.physics import PhysxManager
        from pxr import Usd, UsdPhysics

        self.base, self.adapter = base, adapter
        self.stage = omni.usd.get_context().get_stage()
        self.integrity = self.robot_integrity = None
        self.roots = {name: _path(obj.prim_path) for name, obj in objects.items()}
        self.dust = copy.deepcopy(dust_metadata)
        self.dust_ids = tuple(self.dust['ids'])
        required = {'clamp', 'wafer', 'supports', 'gun', 'gun_stand', *self.dust_ids}
        if len(self.dust_ids) != 12 or len(set(self.dust_ids)) != 12 or not required.issubset(self.roots):
            raise ValueError('Incomplete actual task-object inventory.')
        self.robot_roots = _robot_roots(robot_materials) if robot_materials is not None else dict(robot_roots or {})
        if set(self.robot_roots) != {'left', 'right'} or len(set(self.robot_roots.values())) != 2:
            raise ValueError('Two exact whole-robot spawn roots are required.')
        root_binding = {}
        for side, path in self.robot_roots.items():
            configured = _path(base.scene[side + '_robot'].cfg.prim_path)
            if not isinstance(path, str) or not path.startswith('/') or any(c in path for c in '*{}'):
                raise ValueError('Invalid exact robot spawn root: ' + str(path))
            paths = None
            for attribute in ('root_view', 'root_physx_view'):
                view = getattr(base.scene[side + '_robot'], attribute, None)
                if view is not None and getattr(view, 'prim_paths', None) is not None:
                    paths = list(view.prim_paths)
                    break
            if paths is not None and (len(paths) != 1 or not all(
                    str(p) == path or str(p).startswith(path + '/') for p in paths)):
                raise ValueError('Actual articulation view lies outside its verified robot spawn root: '
                                 + repr(dict(side=side, configured=configured, spawn_root=path, view_paths=paths)))
            root_binding[side] = dict(spawn_root=path, configured_expression=configured,
                                      actual_view_paths=paths,
                                      scope='Verified spawn root plus complete cooked collider, joint and measured body inventories.')
        self.guard = RobotGeometryGuard.from_native({side: self.stage.GetPrimAtPath(path)
                                                     for side, path in self.robot_roots.items()})
        self.body_ids = {}
        for side, name in self.guard.keys:
            ids, found = base.scene[side + '_robot'].find_bodies(name, preserve_order=True)
            if list(found) != [name] or len(ids) != 1:
                raise ValueError('Ambiguous native robot link: ' + side + '/' + name)
            self.body_ids[(side, name)] = int(ids[0])
        self.joint_ids = {}
        joint_names = ['joint' + str(i) for i in range(1, 9)]
        for side in ('left', 'right'):
            ids, found = base.scene[side + '_robot'].find_joints(joint_names, preserve_order=True)
            if list(found) != joint_names or len(set(ids)) != 8:
                raise ValueError('Ambiguous native robot command mapping: ' + side)
            self.joint_ids[side] = [int(i) for i in ids]

        self.fixture_paths = {}
        for name in ('supports', 'gun_stand'):
            root = self.stage.GetPrimAtPath(self.roots[name])
            bodies = [p for p in Usd.PrimRange(root, Usd.TraverseInstanceProxies())
                      if p.HasAPI(UsdPhysics.RigidBodyAPI)]
            if not bodies or name == 'gun_stand' and len(bodies) != 1:
                raise ValueError('Unsupported fixture-body inventory: ' + name)
            for body in bodies:
                label = 'gun_stand' if name == 'gun_stand' else body.GetName()
                if label in self.fixture_paths:
                    raise ValueError('Duplicate actual fixture name: ' + label)
                if UsdPhysics.RigidBodyAPI(body).GetKinematicEnabledAttr().Get() is not True:
                    raise ValueError('Declared fixed fixture is not kinematic: ' + label)
                self.fixture_paths[label] = str(body.GetPath())
        stands = sorted(n for n in self.fixture_paths if n.startswith('tool_stand_'))
        if len(stands) != 2 or not {'heater', 'pedestal', 'pillar_0', 'pillar_1', 'pillar_2', 'gun_stand'}.issubset(self.fixture_paths):
            raise ValueError('Required fixed-fixture inventory differs.')
        self.sim_view = PhysxManager.get_physics_sim_view()
        self.fixture_views = {}
        for name, path in self.fixture_paths.items():
            view = self.sim_view.create_rigid_body_view(path)
            if view.count != 1 or list(view.prim_paths) != [path]:
                raise ValueError('Actual fixture tensor view is ambiguous: ' + name)
            self.fixture_views[name] = view
        self.initial_fixtures = self._fixtures()

        spec = adapter.spec
        self.metadata = dict(engine='Isaac Lab / PhysX', physics_dt_s=float(base.physics_dt),
            control_hz=20., observation_interval_s=.05,
            robot_command_joint_names=joint_names, native_finger_open_q_m=-.04695,
            wafer_radius_m=spec['wafer_diameter'] / 2, wafer_thickness_m=spec['wafer_thickness'],
            wafer_flat_length_m=spec['wafer_flat'], wafer_local_bottom_z_m=-spec['wafer_thickness'] / 2,
            pillar_radius_m=spec['pillar_radius'],
            contact_faces_local_m=copy.deepcopy(adapter.metadata['contact_faces_local_m']),
            pad_contact_band_depths_m=copy.deepcopy(adapter.metadata['pad_contact_band_depths_m']),
            tool_pickup=dict(stand_geoms=stands), dust=self.dust, gun=copy.deepcopy(gun_metadata),
            robot_geometry=self.guard.evidence(),
            robot_geometry_capture='sampled_20hz_native_measured_link_poses_v1',
            evidence_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            object_root_paths=copy.deepcopy(self.roots), fixture_body_paths=copy.deepcopy(self.fixture_paths),
            robot_root_binding=root_binding,
            integrity_scope='Actual composed USD geometry/topology plus live PhysX fixture and robot poses; not material-damage prediction.',
            collision_scope='Sampled robot self/inter-arm geometry at 20 Hz; not continuous or all-physics-step collision validation.')
        if abs(float(base.step_dt) - .05) > 1e-12 or abs(float(base.physics_dt) - .0005) > 1e-12:
            raise ValueError('Native evidence expects 2,000 Hz physics and 20 Hz control.')
        self.metadata['pillar_top_centers_world_m'] = [
            (_pose(self.initial_fixtures['pillar_' + str(i)], 'pillar') @
             np.array([0., 0., spec['pillar_height'] / 2, 1.]))[:3].tolist() for i in range(3)]
        self.integrity = _GeometryIntegrity(self.stage, self.roots)
        try:
            self.robot_integrity = _GeometryIntegrity(self.stage, self.robot_roots)
        except Exception:
            self.integrity.close()
            raise
        self.last_time = None

    def _fixtures(self):
        result = {}
        for name, view in self.fixture_views.items():
            value = _host(view.get_transforms())
            if value.shape != (1, 7):
                raise ValueError('Invalid live fixture pose: ' + name)
            _pose(value[0], name)
            result[name] = value[0].tolist()
        return result

    def attach(self, source_row):
        row = copy.deepcopy(source_row)
        t = float(row['time_s'])
        if not np.isfinite(t) or abs(t - (0. if self.last_time is None else self.last_time + .05)) > 1e-8:
            raise ValueError('Native evidence requires the complete 20 Hz observation sequence.')
        poses = {}
        for side in ('left', 'right'):
            values = _host(self.base.scene[side + '_robot'].data.body_link_pose_w)
            for key, index in self.body_ids.items():
                if key[0] == side:
                    _pose(values[0, index], str(key))
                    poses[key] = values[0, index].tolist()
            targets = _host(self.base.scene[side + '_robot'].data.joint_pos_target)[0, self.joint_ids[side]]
            arm_field = 'commanded_arm_q_rad' if side == 'left' else 'commanded_right_arm_q_rad'
            finger_field = 'commanded_finger_q_m' if side == 'left' else 'commanded_right_finger_q_m'
            if finger_field in row and not np.allclose(row[finger_field], targets[6:], rtol=0., atol=1e-7):
                raise ValueError('Adapter finger targets differ from the actual actuator: ' + side)
            row[arm_field] = targets[:6].tolist()
            row[finger_field] = targets[6:].tolist()
        for side, field in zip(('left', 'right'), HANDS):
            existing = _pose(row[field], field)
            observed = _pose(poses[(side, 'gripper')], field)
            if not np.allclose(existing, observed, rtol=0., atol=1e-7):
                raise ValueError('Adapter hand pose differs from the actual robot link: ' + side)
        row['robot_link_poses_world'] = [poses[key] for key in self.guard.keys]
        present, intact = self.robot_integrity.snapshot()
        row['robot_geometry_intact'] = all(present.values()) and all(intact.values())
        checked = self.guard.check(poses) if row['robot_geometry_intact'] else dict(
            status='UNSUPPORTED', complete=False, reason='Actual robot geometry changed.')
        row['robot_geometry_check'] = compact_result(checked)
        row['objects_present'], row['objects_intact'] = self.integrity.snapshot()
        row['dust_intact'] = all(row['objects_intact'][name] for name in self.dust_ids)
        row['fixture_poses_world'] = self._fixtures()
        if self.last_time is None:
            self.metadata['hand_grasp_points_local_m'] = {
                field: _contact_center(self.guard, side, poses, row[field])
                for side, field in zip(('left', 'right'), HANDS)}
            initial = {name: row[name + '_pose_world'] for name in ('wafer', 'clamp', 'gun')}
            for name, pose in initial.items():
                _pose(pose, name)
            self.metadata['task_score_reference'] = dict(
                initial_object_poses_world=copy.deepcopy(initial),
                fixture_poses_world=copy.deepcopy(self.initial_fixtures), object_names=sorted(self.roots))
        self.last_time = t
        return row

    def close(self):
        for item in (self.integrity, self.robot_integrity):
            if item is not None:
                item.close()
