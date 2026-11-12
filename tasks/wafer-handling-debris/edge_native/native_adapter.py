"""Native three-body edge pick; import Isaac modules only after app startup."""
from pathlib import Path
import hashlib
import base64
import gzip
import json
import math
import xml.etree.ElementTree as ET
import zlib

import numpy as np
from scipy.spatial.transform import Rotation


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pose_matrix(pose):
    pose = np.asarray(pose, dtype=float)
    if pose.shape != (7,) or not np.isfinite(pose).all() or abs(np.linalg.norm(pose[3:]) - 1) > .001:
        raise ValueError('Expected finite XYZ + XYZW link pose.')
    result = np.eye(4)
    result[:3, :3] = Rotation.from_quat(pose[3:]).as_matrix()
    result[:3, 3] = pose[:3]
    return result


def matrix_pose(matrix):
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError('Invalid pose matrix.')
    return np.r_[matrix[:3, 3], Rotation.from_matrix(matrix[:3, :3]).as_quat()].tolist()


def _numpy(value, *, finite=True):
    value = value.torch if hasattr(value, 'torch') else value
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    elif hasattr(value, 'numpy'):
        value = value.numpy()
    result = np.asarray(value, dtype=float)
    if (not np.isfinite(result).all()) if finite else np.isnan(result).any():
        raise ValueError('Invalid native numeric readback.')
    return result


def _host_snapshot(fields):
    import torch
    tensors = [(name, value.torch if hasattr(value, 'torch') else value) for name, value in fields]
    if all(isinstance(value, torch.Tensor) for _, value in tensors):
        sizes = [value.numel() for _, value in tensors]
        packed = torch.cat([value.reshape(-1) for _, value in tensors]).detach().cpu().numpy()
        if not np.isfinite(packed).all():
            raise ValueError('Non-finite live measurement.')
        result, offset = {}, 0
        for (name, value), size in zip(tensors, sizes):
            result[name] = packed[offset:offset + size].reshape(tuple(value.shape)).astype(float)
            offset += size
        return result
    return {name: _numpy(value) for name, value in tensors}


class NativeEdgePick:
    BODY_NAMES = ('base', 'jaw', 'squeeze_slider')
    JOINT_NAMES = ('jaw_hinge', 'squeeze_travel')
    TIP_NAMES = ('tip_left', 'tip_right')
    TCP_OFFSET = np.array([0., 0., .134])

    def __init__(self, native_dir, world_translation_m=(0., 0., 0.), world_yaw_deg=0., dark_appearance=True):
        self.directory = Path(native_dir).resolve()
        self.report = json.loads((self.directory / 'parity.json').read_text())
        self.world = np.eye(4)
        self.world[:3, :3] = Rotation.from_euler('z', world_yaw_deg, degrees=True).as_matrix()
        self.world[:3, 3] = np.asarray(world_translation_m, dtype=float)
        if not np.isfinite(self.world).all():
            raise ValueError('Invalid scene placement.')
        self.dark_appearance = bool(dark_appearance)
        self.objects, self.sensor_specs, self.spawn_checks = {}, {}, {}
        self.base = self.grasp_calibration = None
        self.left_material = None
        self._springs_initialized = False
        self._check_files()
        batch_path = self.directory.parent / 'batch_report.json'
        qualifier = self.report['local_fixture_qualification']
        if sha(batch_path) != qualifier['sha256']:
            raise ValueError('Local mechanism report changed.')
        self.batch = json.loads(batch_path.read_text())
        if not qualifier['passed'] or self.batch['successes'] < 16 or self.batch['count'] != 20:
            raise ValueError('The declared local mechanism qualification is incomplete.')
        self.spec = self.batch['spec']
        source_meta = self.batch['metadata']
        self.grip_local = np.asarray(source_meta['gripper_origin']) - np.asarray(source_meta['tool_origin'])
        xml = ET.parse(self.directory.parent / 'scene.xml').getroot()
        fingers = [g for g in xml.iter('geom') if g.get('name') in ('finger_lower', 'finger_upper')]
        sizes = [np.fromstring(g.get('size'), sep=' ') for g in fingers]
        if len(sizes) != 2 or any(x.shape != (3,) for x in sizes) or not np.array_equal(*sizes):
            raise ValueError('Ambiguous fixture finger thickness.')
        self.fixture_half_thickness = float(sizes[0][2])
        self.tool_records = {b['name']: b for b in self.report['bodies']}
        self.colliders = {g['name']: g for g in self.report['colliders']}
        self.joints = {j['name']: j for j in self.report['joints']}
        if set(self.tool_records) != set(self.BODY_NAMES) or set(self.joints) != set(self.JOINT_NAMES):
            raise ValueError('The native mechanism must have exactly three free/passive bodies and two joints.')
        if not all(p['enabled'] for p in self.report['collision_pairs']):
            raise ValueError('The accepted mechanism requires all internal body pairs enabled.')
        self.metadata = dict(adapter_sha256=sha(__file__), source_scene_sha256=self.report['source_scene_sha256'],
            parity_sha256=sha(self.directory / 'parity.json'), native_assets={
                r['asset_file']: r['asset_sha256'] for r in self._asset_records()},
            world_transform=self.world.tolist(), pose_order='XYZ + XYZW',
            tool_grasp_point_local_m=self.grip_local.tolist(), fixture_finger_half_thickness_m=self.fixture_half_thickness,
            physics_scope='New actual-YAM/PhysX test; local fixture success is not native validation.',
            force_scope='Partner-filtered rigid-body normal-vector norms, not per-shape forces. Collider geometry must independently exclude non-land wafer contact.',
            spring_scope='Two fixed passive spring-rest targets; no commanded jaw trajectory or attachment.',
            native_contact_offset_m=.0001, native_rest_offset_m=0.)
        self.initial_poses = {name: self._initial_pose(record['initial_root_pose_world_wxyz']) for name, record in (
            ('clamp', self.report), ('wafer', self.report['wafer_asset']), ('supports', self.report['support_asset']))}
        self.metadata['initial_target_poses_world'] = self.initial_poses
        self.metadata['contact_faces_local_m'] = [self._contact_face('lower_land', 1), self._contact_face('upper_land', -1)]
        self.metadata['pad_contact_band_depths_m'] = [self.spec['lower_envelope'], self.spec['upper_envelope']]
        fixed, moving = self.colliders['fixed_handle'], self.colliders['upper_handle']
        self.handle_points = [self.grip_local.copy(), self.grip_local - np.asarray(self.tool_records['squeeze_slider']['position_m'])]
        self.handle_points[0][2] = fixed['position_m'][2] - fixed['half_sizes_m'][2]
        self.handle_points[1][2] = moving['position_m'][2] + moving['half_sizes_m'][2]
        self.metadata['handle_contact_points_local_m'] = [x.tolist() for x in self.handle_points]
        self.metadata['handle_contact_frame_scope'] = 'Design contact points on the actual fixed/slider handle faces, transformed by measured body poses; not measured per-contact locations.'

    def _contact_face(self, name, sign):
        record = self.colliders[name]
        center, size = np.asarray(record['position_m']), np.asarray(record['half_sizes_m'])
        return [(center + np.array([x * size[0], y * size[1], sign * size[2]])).tolist()
                for x, y in [(-1, -1), (1, -1), (1, 1), (-1, 1)]]

    def _asset_records(self):
        return [self.report, self.report['wafer_asset'], self.report['support_asset']]

    def _check_files(self):
        if sha(self.directory.parent / 'scene.xml') != self.report['source_scene_sha256']:
            raise ValueError('Source nominal scene differs from exported assets.')
        generator = self.report['source_generator_verification']
        if sha(self.directory.parent / 'edge_pick.py') != generator['sha256']:
            raise ValueError('Source mechanical generator differs.')
        for record in self._asset_records():
            path = self.directory / record['asset_file']
            if path.parent != self.directory or sha(path) != record['asset_sha256']:
                raise ValueError('Native asset changed: ' + path.name)

    def _initial_pose(self, wxyz):
        value = list(wxyz)
        return matrix_pose(self.world @ pose_matrix(value[:3] + value[4:] + value[3:4]))

    def make_objects(self):
        from isaaclab.actuators import ImplicitActuatorCfg
        from isaaclab_arena.assets.object import Object
        from isaaclab_arena.assets.object_type import ObjectType
        from isaaclab_arena.utils.pose import Pose
        if self.objects:
            raise RuntimeError('Objects may be configured only once.')
        specs = [('clamp', self.report, ObjectType.ARTICULATION),
                 ('wafer', self.report['wafer_asset'], ObjectType.RIGID),
                 ('supports', self.report['support_asset'], ObjectType.BASE)]
        self.initial_poses = {}
        for name, record, kind in specs:
            value = self._initial_pose(record['initial_root_pose_world_wxyz'])
            pose = Pose.from_dict(dict(position_xyz=value[:3], rotation_xyzw=value[3:]))
            obj = Object(name=name, usd_path=str(self.directory / record['asset_file']), object_type=kind, initial_pose=pose)
            obj.object_cfg.spawn.activate_contact_sensors = True
            self.objects[name] = obj
            self.initial_poses[name] = value
            original = obj.object_cfg.spawn.func
            def spawn(*args, _name=name, _original=original, **kwargs):
                if _name in self.spawn_checks:
                    raise RuntimeError('Multiple native edge-pick spawns are unsupported.')
                root = _original(*args, **kwargs)
                self.spawn_checks[_name] = self.verify_composed(root, _name)
                if _name == 'clamp' and self.dark_appearance:
                    from pxr import UsdGeom
                    colors = {}
                    for g in self.report['colliders']:
                        prim = root.GetStage().GetPrimAtPath(str(root.GetPath()) + g['path'].removeprefix('/Asset'))
                        color = [.23, .24, .25] if g['name'].endswith('_land') else [.065, .070, .075]
                        UsdGeom.Gprim(prim).CreateDisplayColorAttr([tuple(color)])
                        colors[g['name']] = color
                    self.spawn_checks[_name]['appearance_only'] = colors
                    self.verify_composed(root, _name)
                return root
            obj.object_cfg.spawn.func = spawn
        cfg = self.objects['clamp'].object_cfg
        cfg.actuators = {name + '_spring': ImplicitActuatorCfg(joint_names_expr=[name],
            stiffness=self.joints[name]['stiffness_si'], damping=self.joints[name]['damping_si'],
            joint_effort_limit=float('inf'), joint_velocity_limit=float('inf')) for name in self.JOINT_NAMES}
        cfg.init_state.joint_pos = {name: 0. for name in self.JOINT_NAMES}
        cfg.init_state.joint_vel = {name: 0. for name in self.JOINT_NAMES}
        self.metadata['initial_target_poses_world'] = self.initial_poses
        return self.objects

    def verify_composed(self, root, kind):
        from pxr import Usd, UsdGeom, UsdPhysics
        if not root or not root.IsValid():
            raise ValueError('Missing spawned object: ' + kind)
        prims = list(Usd.PrimRange(root, Usd.TraverseInstanceProxies()))
        bodies = {p.GetName(): p for p in prims if p.HasAPI(UsdPhysics.RigidBodyAPI)}
        expected = (self.tool_records if kind == 'clamp' else
                    {b['name']: b for b in self.report['wafer_asset']['bodies']} if kind == 'wafer' else
                    {b['name']: b for b in self.report['support_asset']['supports']})
        if set(bodies) != set(expected):
            raise ValueError('Spawned rigid-body inventory differs: ' + kind)
        for name, prim in bodies.items():
            api = UsdPhysics.RigidBodyAPI(prim)
            if api.GetRigidBodyEnabledAttr().Get() is not True or bool(api.GetKinematicEnabledAttr().Get()) != (kind == 'supports'):
                raise ValueError('Dynamic/static body state differs: ' + name)
            if prim.GetAttribute('physxRigidBody:disableGravity').Get() is True:
                raise ValueError('Unexpected gravity-disabled body: ' + name)
            if kind != 'supports':
                mass = UsdPhysics.MassAPI(prim)
                for actual, wanted in [(mass.GetMassAttr().Get(), expected[name]['mass_kg']),
                                       (mass.GetCenterOfMassAttr().Get(), expected[name]['com_m']),
                                       (mass.GetDiagonalInertiaAttr().Get(), expected[name]['inertia_kg_m2'])]:
                    if not np.allclose(actual, wanted, rtol=2e-6, atol=1e-12):
                        raise ValueError('Spawned inertial parameters differ: ' + name)
        joints = {p.GetName(): p for p in prims if p.IsA(UsdPhysics.Joint)}
        if set(joints) != (set(self.JOINT_NAMES) if kind == 'clamp' else set()):
            raise ValueError('Unexpected attachment or joint topology.')
        for name, prim in joints.items():
            record = self.joints[name]
            joint = UsdPhysics.Joint(prim)
            actual = [str(t) for rel in (joint.GetBody0Rel(), joint.GetBody1Rel()) for t in rel.GetTargets()]
            wanted = [str(root.GetPath()) + record[k].removeprefix('/Asset') for k in ('parent', 'child')]
            if actual != wanted or joint.GetCollisionEnabledAttr().Get() is not True:
                raise ValueError('Joint attachment/collision state differs: ' + name)
            drive = UsdPhysics.DriveAPI(prim, 'angular' if record['angular'] else 'linear')
            scale = record['native_gain_scale']
            for actual, wanted in [(drive.GetStiffnessAttr().Get(), record['stiffness_si'] * scale),
                                   (drive.GetDampingAttr().Get(), record['damping_si'] * scale),
                                   (drive.GetTargetPositionAttr().Get(), record['rest_si'] / scale),
                                   (drive.GetTargetVelocityAttr().Get(), 0.)]:
                if not math.isclose(actual, wanted, rel_tol=2e-6, abs_tol=1e-9):
                    raise ValueError('Composed passive spring differs: ' + name)
            if drive.GetTypeAttr().Get() != 'force' or not math.isinf(drive.GetMaxForceAttr().Get()):
                raise ValueError('Composed spring must use the uncapped passive force law.')
        if kind == 'clamp':
            if root.GetAttribute('physxArticulation:enabledSelfCollisions').Get() is not True:
                raise ValueError('Tool self collision is disabled.')
            if any(UsdPhysics.FilteredPairsAPI(p).GetFilteredPairsRel().GetTargets() for p in prims):
                raise ValueError('Unexpected filtered tool contacts.')
        colliders = [p for p in prims if p.HasAPI(UsdPhysics.CollisionAPI)]
        expected_colliders = (self.report['colliders'] if kind == 'clamp' else self.report['wafer_asset']['colliders']
                              if kind == 'wafer' else [s['collider'] for s in self.report['support_asset']['supports']])
        if len(colliders) != len(expected_colliders):
            raise ValueError('Spawned collider inventory differs.')
        for record in expected_colliders:
            prim = root.GetStage().GetPrimAtPath(str(root.GetPath()) + record['path'].removeprefix('/Asset'))
            if not prim or UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is not True:
                raise ValueError('Missing/disabled collider: ' + record['name'])
            for field, value in [('physxCollision:contactOffset', .0001), ('physxCollision:restOffset', 0.)]:
                if not math.isclose(prim.GetAttribute(field).Get(), value, abs_tol=1e-10):
                    raise ValueError('Native contact offset differs.')
        return dict(status='COMPOSED_DYNAMIC_ASSET_VERIFIED', root_path=str(root.GetPath()),
                    bodies=sorted(bodies), joints=sorted(joints), collider_count=len(colliders))

    def configure(self, embodiment):
        from isaaclab.sensors import ContactSensorCfg
        if not self.objects or self.sensor_specs:
            raise RuntimeError('Configure objects once before contact sensors.')
        self.embodiment = embodiment
        tool = self.objects['clamp'].prim_path
        wafer = self.objects['wafer'].get_contact_sensor_prim_path()
        support = self.objects['supports'].prim_path
        self.partner_paths = {name: tool + '/' + name for name in self.BODY_NAMES}
        self.partner_paths['wafer'] = wafer
        self.partner_paths.update({s['name']: support + '/' + s['name'] for s in self.report['support_asset']['supports']})
        self.tip_paths = {side: {tip: getattr(embodiment.scene_config, side + '_robot').prim_path + embodiment.link_paths[tip]
                                 for tip in self.TIP_NAMES} for side in ('left', 'right')}
        self.partner_paths.update({side + '_' + tip: path for side, tips in self.tip_paths.items() for tip, path in tips.items()})
        def add(name, path, partners):
            self.sensor_specs[name] = dict(path=path, partners=list(partners))
            return ContactSensorCfg(prim_path=path, filter_prim_paths_expr=[self.partner_paths[p] for p in partners],
                update_period=0., history_length=1, track_air_time=False, max_contact_data_count_per_prim=256)
        sensors = {'edge_wafer_contacts': add('edge_wafer_contacts', wafer, [p for p in self.partner_paths if p != 'wafer'])}
        for side, tips in self.tip_paths.items():
            for tip, path in tips.items():
                key = 'edge_' + side + '_' + tip
                sensors[key] = add(key, path, [*self.BODY_NAMES, 'wafer', *[s['name'] for s in self.report['support_asset']['supports']]])
        for body in self.BODY_NAMES:
            key = 'edge_tool_' + body
            sensors[key] = add(key, self.partner_paths[body], [p for p in self.partner_paths if p != body])
        self.sensors = sensors
        return sensors

    def configure_env(self, cfg):
        from isaaclab_arena.terms.events import reset_articulation_pose_and_joints
        name, event = self.objects['clamp'].get_event_cfg()
        if name != 'clamp' or event.mode != 'reset':
            raise ValueError('Expected the initial articulated-clamp reset.')
        event.func = reset_articulation_pose_and_joints
        event.params['pose'] = self.objects['clamp'].initial_pose
        setattr(cfg.events, name, event)
        for name, sensor in self.sensors.items():
            setattr(cfg.scene, name, sensor)
        return cfg

    def install_left_robot_material(self, spawn_cfg, expected_count=315):
        """Match the qualified fixture's .5 sliding friction on the fresh left robot."""
        from pxr import Usd, UsdPhysics, UsdShade, Sdf
        if self.left_material is not None:
            raise RuntimeError('Left material callback is already installed.')
        state = self.left_material = dict(status='NOT_SPAWNED', callback_count=0)
        original = spawn_cfg.func
        def spawn(*args, **kwargs):
            root = original(*args, **kwargs)
            state['callback_count'] += 1
            if state['callback_count'] != 1:
                raise RuntimeError('Expected one fresh left-arm spawn.')
            colliders = [p for p in Usd.PrimRange(root, Usd.TraverseInstanceProxies()) if p.HasAPI(UsdPhysics.CollisionAPI)]
            if len(colliders) != expected_count or any(p.IsInstanceProxy() for p in colliders):
                raise ValueError('Unexpected or still-instanced left-arm collision inventory.')
            path = str(root.GetPath()) + '/EdgePickPhysicsMaterial'
            if root.GetStage().GetPrimAtPath(path):
                raise ValueError('The isolated left material already exists.')
            material = UsdShade.Material.Define(root.GetStage(), path)
            prim = material.GetPrim(); api = UsdPhysics.MaterialAPI.Apply(prim)
            api.CreateStaticFrictionAttr(.5); api.CreateDynamicFrictionAttr(.5); api.CreateRestitutionAttr(0.)
            prim.AddAppliedSchema('PhysxMaterialAPI')
            for key in ('frictionCombineMode', 'restitutionCombineMode'):
                prim.CreateAttribute('physxMaterial:' + key, Sdf.ValueTypeNames.Token).Set('max')
            for collider in colliders:
                if UsdPhysics.CollisionAPI(collider).GetCollisionEnabledAttr().Get() is not True:
                    raise ValueError('A left-arm collider is disabled.')
                UsdShade.MaterialBindingAPI.Apply(collider).Bind(material, UsdShade.Tokens.strongerThanDescendants, 'physics')
                bound, _ = UsdShade.MaterialBindingAPI(collider).ComputeBoundMaterial('physics')
                if not bound or str(bound.GetPath()) != path:
                    raise ValueError('Left-arm physics material was shadowed.')
            state.update(status='COMPOSED_BINDINGS_VERIFIED', root_path=str(root.GetPath()), collider_count=expected_count,
                static_friction=.5, dynamic_friction=.5, restitution=0., friction_combine_mode='max',
                scope='New left robot only; source files unchanged. Right gun-hand material remains separately configured.')
            return root
        spawn_cfg.func = spawn
        return state

    @staticmethod
    def _live_material(asset, count, friction):
        for name in ('root_view', 'root_physx_view'):
            view = getattr(asset, name, None)
            if view is not None and callable(getattr(view, 'get_material_properties', None)):
                break
        else:
            raise ValueError('Public native shape material readback is unavailable.')
        values = _numpy(view.get_material_properties())
        if values.shape != (1, count, 3) or not np.allclose(values, [friction, friction, 0.], rtol=0., atol=1e-6):
            raise ValueError('Live PhysX material differs from the declared contact model.')
        return dict(status='LIVE_PHYSX_SHAPES_VERIFIED', shape_count=count,
                    minimum=values[0].min(0).tolist(), maximum=values[0].max(0).tolist())

    @staticmethod
    def _ids(asset, names, bodies=False):
        ids, found = (asset.find_bodies(list(names), preserve_order=True) if bodies else asset.find_joints(list(names), preserve_order=True))
        if list(found) != list(names) or len(set(ids)) != len(names):
            raise ValueError('Ambiguous native name mapping: ' + str(names))
        return [int(i) for i in ids]

    def bind(self, base):
        import torch
        import omni.usd
        from pxr import Usd, UsdPhysics
        from yam_arena.constants import TCP_OFFSET
        if not np.array_equal(np.asarray(TCP_OFFSET), self.TCP_OFFSET):
            raise ValueError('Actual YAM TCP offset differs from the CPU planning convention.')
        if self.base is not None:
            raise RuntimeError('Bind only after the one initial reset.')
        self._check_files()
        self.base, self.torch = base, torch
        self.clamp, self.wafer = base.scene['clamp'], base.scene['wafer']
        self.robots = {side: base.scene[side + '_robot'] for side in ('left', 'right')}
        if set(self.spawn_checks) != set(self.objects):
            raise ValueError('One or more scene assets were not verified at spawn.')
        self.body_ids = self._ids(self.clamp, self.BODY_NAMES, True)
        self.joint_ids = self._ids(self.clamp, self.JOINT_NAMES)
        if len(self.clamp.joint_names) != 2:
            raise ValueError('Unexpected extra mechanism DOF.')
        self.robot_ids = {side: dict(arm=self._ids(robot, ['joint' + str(i) for i in range(1, 7)]),
            fingers=self._ids(robot, ['joint7', 'joint8']), bodies=self._ids(robot, ['gripper', *self.TIP_NAMES], True))
            for side, robot in self.robots.items()}
        self.live_sensors = {name: base.scene.sensors[name] for name in self.sensor_specs}
        stage = omni.usd.get_context().get_stage()
        actual_paths = {name: info['root_path'] for name, info in self.spawn_checks.items()}
        for name, path in actual_paths.items():
            self.verify_composed(stage.GetPrimAtPath(path), name)
        for prim in stage.Traverse():
            if prim.IsA(UsdPhysics.Joint):
                rels = (UsdPhysics.Joint(prim).GetBody0Rel(), UsdPhysics.Joint(prim).GetBody1Rel())
                targets = [str(x) for rel in rels for x in rel.GetTargets()]
                for name in ('clamp', 'wafer'):
                    root = actual_paths[name]
                    if any(x == root or x.startswith(root + '/') for x in targets) and not str(prim.GetPath()).startswith(root + '/'):
                        raise ValueError('Unexpected external tool/wafer attachment.')
        rests = [self.joints[n]['rest_si'] for n in self.JOINT_NAMES]
        self.clamp.actuators.target_command.set_position_index(
            value=torch.tensor([rests], device=base.device, dtype=torch.float32), joint_ids=self.joint_ids)
        self.clamp.write_data_to_sim()
        self._springs_initialized = True
        self.metadata['passive_drive_readback'] = self.verify_live_springs(parameters=True)
        if self.left_material is None or self.left_material['status'] != 'COMPOSED_BINDINGS_VERIFIED':
            raise ValueError('Install and verify the .5 left-arm material before simulation creation.')
        self.left_material['live_physics'] = self._live_material(self.robots['left'], self.left_material['collider_count'], .5)
        self.metadata['left_robot_material'] = self.left_material
        self.metadata['tool_material'] = self._live_material(self.clamp, self.report['collider_count'], .5)
        self.metadata['wafer_material'] = self._live_material(self.wafer, self.report['wafer_asset']['collider_count'], .5)
        self.metadata['spawn_checks'] = self.spawn_checks
        self.metadata['native_robot_mapping'] = self.robot_ids
        row = self.read_row(0., 'initialization')
        self._calibrate_grasp(stage, actual_paths, row)
        self.metadata['grasp_calibration'] = self.grasp_calibration
        self.metadata['pillar_top_centers_world_m'] = [(self.world @ np.r_[p, 1.])[:3].tolist()
            for p in self.batch['metadata']['pillar_top_centers']]
        return self.metadata

    def verify_live_springs(self, parameters=False):
        view = getattr(self.clamp, 'root_view', None)
        if view is None:
            view = getattr(self.clamp, 'root_physx_view', None)
        if view is None:
            raise ValueError('Native public articulation view unavailable.')
        keys = ['position_targets', 'velocity_targets']
        if parameters:
            keys += ['stiffnesses', 'dampings', 'max_forces', 'max_velocities']
            kinds = _numpy(view.get_drive_types())
            if kinds.shape != (1, 2) or not np.all(kinds[0, self.joint_ids] == 1):
                raise ValueError('Native passive joints require force drives.')
        result = {}
        for key in keys:
            values = _numpy(getattr(view, 'get_dof_' + key)(), finite=key not in ('max_forces', 'max_velocities'))
            if values.shape != (1, 2):
                raise ValueError('Unexpected mechanism buffer shape.')
            expected = [self.joints[n][{'position_targets': 'rest_si', 'stiffnesses': 'stiffness_si', 'dampings': 'damping_si'}[key]]
                        if key in ('position_targets', 'stiffnesses', 'dampings') else 0. if key == 'velocity_targets'
                        else float('inf') for n in self.JOINT_NAMES]
            if not np.allclose(values[0, self.joint_ids], expected, rtol=2e-6, atol=1e-7):
                raise ValueError('Native passive ' + key + ' differs from the fixed mechanical law.')
            result[key] = ['unlimited' if math.isinf(x) else float(x) for x in values[0, self.joint_ids]]
        return result

    def read_row(self, time_s, phase='measurement'):
        if not self._springs_initialized:
            raise RuntimeError('Initialize fixed mechanical springs before readout.')
        wafer_pose = getattr(self.wafer.data, 'root_link_pose_w', None)
        if wafer_pose is None:
            wafer_pose = self.wafer.data.body_link_pose_w
        fields = [('tool_poses', self.clamp.data.body_link_pose_w), ('passive_q', self.clamp.data.joint_pos),
                  ('passive_v', self.clamp.data.joint_vel), ('passive_targets', self.clamp.data.joint_pos_target), ('wafer_pose', wafer_pose)]
        for side, robot in self.robots.items():
            fields.extend([(side + '_poses', robot.data.body_link_pose_w), (side + '_q', robot.data.joint_pos),
                           (side + '_targets', robot.data.joint_pos_target)])
        for name, sensor in self.live_sensors.items():
            fields.extend([(name + '_matrix', sensor.data.normal_force_matrix_w), (name + '_net', sensor.data.net_normal_forces_w)])
        arrays = _host_snapshot(fields)
        tool_poses = {n: arrays['tool_poses'][0, i].tolist() for n, i in zip(self.BODY_NAMES, self.body_ids)}
        q = arrays['passive_q'][0, self.joint_ids]; v = arrays['passive_v'][0, self.joint_ids]
        targets = arrays['passive_targets'][0, self.joint_ids]
        if not np.allclose(targets, [self.joints[n]['rest_si'] for n in self.JOINT_NAMES], atol=1e-7, rtol=0):
            raise ValueError('Mechanical spring rest target was changed during the run.')
        contacts, residuals = {}, {}
        for name, spec in self.sensor_specs.items():
            matrix, net = arrays[name + '_matrix'], arrays[name + '_net']
            if matrix.shape != (1, 1, len(spec['partners']), 3) or net.shape != (1, 1, 3):
                raise ValueError('Native filtered contact shape differs: ' + name)
            contacts[name] = dict(zip(spec['partners'], np.linalg.norm(matrix[0, 0], axis=1).tolist()))
            residuals[name] = float(np.linalg.norm(net[0, 0] - matrix[0, 0].sum(axis=0)))
        low, high = self.colliders['lower_land'], self.colliders['upper_land']
        lower = pose_matrix(tool_poses['base']); upper = pose_matrix(tool_poses['jaw'])
        lower_point = (lower @ np.r_[np.asarray(low['position_m']) + [0., 0., low['half_sizes_m'][2]], 1.])[:3]
        upper_point = (upper @ np.r_[np.asarray(high['position_m']) - [0., 0., high['half_sizes_m'][2]], 1.])[:3]
        wafer_contacts = contacts['edge_wafer_contacts']
        row = dict(time_s=float(time_s), phase=str(phase), clamp_pose_world=tool_poses['base'], tool_pose_world=tool_poses,
            wafer_pose_world=arrays['wafer_pose'].reshape(7).tolist(), jaw_pose_world=tool_poses['jaw'],
            slider_pose_world=tool_poses['squeeze_slider'], jaw_gap_m=float((upper_point - lower_point) @ lower[:3, 2]),
            passive_joint_names=list(self.JOINT_NAMES), passive_joint_position=q.tolist(), passive_joint_velocity=v.tolist(),
            jaw_angle_rad=float(q[0]), squeeze_travel_m=float(q[1]),
            passive_spring_effort=[float(-self.joints[n]['stiffness_si'] * (a - self.joints[n]['rest_si'])
                - self.joints[n]['damping_si'] * b) for n, a, b in zip(self.JOINT_NAMES, q, v)],
            finger_handle_contact_normal_force_N=[contacts['edge_left_tip_left']['base'], contacts['edge_left_tip_right']['squeeze_slider']],
            wafer_land_contact_normal_force_N=[wafer_contacts['base'], wafer_contacts['jaw']],
            pillar_normal_force_N=[wafer_contacts['pillar_' + str(i)] for i in range(3)],
            contact_body_normal_force_N=contacts, unmatched_normal_vector_residual_N=residuals,
            force_scope=self.metadata['force_scope'])
        row['jaw_link_poses_world'] = [tool_poses['base'], tool_poses['jaw']]
        row['handle_paddle_poses_world'] = []
        for body_name, point in zip(('base', 'squeeze_slider'), self.handle_points):
            frame = pose_matrix(tool_poses[body_name])
            frame[:3, 3] += frame[:3, :3] @ point
            row['handle_paddle_poses_world'].append(matrix_pose(frame))
        row['measured_jaw_gap_m'] = row['jaw_gap_m']
        for side in ('left', 'right'):
            ids = self.robot_ids[side]; poses = arrays[side + '_poses'][0, ids['bodies']]
            row['measured_hand_pose_world' if side == 'left' else 'right_hand_pose_world'] = poses[0].tolist()
            row['measured_' + side + '_tip_poses_world'] = poses[1:].tolist()
            row['measured_arm_q_rad' if side == 'left' else 'measured_right_arm_q_rad'] = arrays[side + '_q'][0, ids['arm']].tolist()
            row['measured_finger_q_m' if side == 'left' else 'measured_right_finger_q_m'] = arrays[side + '_q'][0, ids['fingers']].tolist()
            row['commanded_finger_q_m' if side == 'left' else 'commanded_right_finger_q_m'] = arrays[side + '_targets'][0, ids['fingers']].tolist()
        for value in [*tool_poses.values(), row['wafer_pose_world'], row['measured_hand_pose_world'], row['right_hand_pose_world']]:
            pose_matrix(value)
        return row

    def _calibrate_grasp(self, stage, actual_paths, row):
        from pxr import Usd, UsdGeom, UsdPhysics, UsdUtils, PhysicsSchemaTools
        from omni.physx import get_physx_cooking_interface
        from omni.physx.bindings._physx import PhysxCollisionRepresentationResult
        cooking = get_physx_cooking_interface()
        all_vertices = []
        for i, tip in enumerate(self.TIP_NAMES):
            path = self.tip_paths['left'][tip].replace('{ENV_REGEX_NS}', '/World/envs/env_0')
            body = stage.GetPrimAtPath(path)
            if not body or body.GetName() != tip:
                raise ValueError('Actual YAM fingertip body not found.')
            points = []
            for prim in Usd.PrimRange(body, Usd.TraverseInstanceProxies()):
                if not prim.HasAPI(UsdPhysics.CollisionAPI):
                    continue
                received = []
                cooking.request_convex_collision_representation(stage_id=UsdUtils.StageCache.Get().GetId(stage).ToLongInt(),
                    collision_prim_id=PhysicsSchemaTools.sdfPathToInt(prim.GetPath()), run_asynchronously=False,
                    on_result=lambda result, values: received.append((result, values)))
                if len(received) != 1 or received[0][0] != PhysxCollisionRepresentationResult.RESULT_VALID:
                    raise ValueError('Actual fingertip cooked geometry is unavailable.')
                transform = np.asarray(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()) *
                    UsdGeom.Xformable(body).ComputeLocalToWorldTransform(Usd.TimeCode.Default()).GetInverse()).T
                for hull in received[0][1]:
                    vertices = np.array([(v.x, v.y, v.z) for v in hull.vertices])
                    points.append(vertices @ transform[:3, :3].T + transform[:3, 3])
            if not points:
                raise ValueError('Empty actual fingertip collision geometry.')
            vertices = np.vstack(points)
            all_vertices.append(vertices)
        self._grasp_geometry(all_vertices, row, 'Actual PhysX cooked fingertip convex vertices and measured link poses')

    def _grasp_geometry(self, all_vertices, row, source):
        hand_inverse = np.linalg.inv(pose_matrix(row['measured_hand_pose_world']))
        planes, centers, bounds, digest = [], [], [], hashlib.sha256()
        for i, vertices in enumerate(all_vertices):
            vertices = np.asarray(vertices, dtype=float)
            digest.update(vertices.tobytes())
            body_in_hand = hand_inverse @ pose_matrix(row['measured_left_tip_poses_world'][i])
            vertices = vertices @ body_in_hand[:3, :3].T + body_in_hand[:3, 3]
            direction = 1. if i == 0 else -1.
            # Use the distal25mm contact region; exclude the proximal mounting flange.
            distal = vertices[vertices[:, 2] >= vertices[:, 2].max() - .025]
            plane = float((distal[:, 0] * direction).max()) * direction
            face = distal[np.abs(distal[:, 0] - plane) <= .0001]
            if len(face) < 4 or np.ptp(face[:, 2]) < .010 or np.ptp(face[:, 1]) < .010:
                raise ValueError('No usable actual distal contact patch was identified.')
            q = row['measured_finger_q_m'][i]
            plane_zero = plane - direction * q
            center = (face.min(axis=0) + face.max(axis=0)) / 2
            center[0] = plane_zero
            planes.append(plane_zero); centers.append(center); bounds.append([face.min(0).tolist(), face.max(0).tolist()])
        if not np.isfinite(planes).all() or abs(planes[1] - planes[0]) > .002:
            raise ValueError('Native fingertip zero-gap geometry differs from the inspected YAM.')
        center = np.mean(centers, axis=0)
        self.grasp_calibration = dict(source=source,
            geometry_sha256=digest.hexdigest(), distal_region_m=.025, contact_plane_band_m=.0001,
            zero_q_contact_planes_hand_x_m=planes, contact_center_hand_m=center.tolist(),
            contact_patch_bounds_hand_at_calibration_m=bounds,
            hand_rotation_in_tool=[[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]],
            approach_orientation='Native side entry: hand X closes along tool Z; hand Z points along tool Y.',
            grasp_point_tool_m=self.grip_local.tolist(), fixture_half_thickness_m=self.fixture_half_thickness,
            scope='Geometric starting map only; native contact/retention must be tested.')

    def calibrate_reference(self, metadata_path, trace_path):
        """Read recorded native geometry for CPU IK planning; bind() replaces it live."""
        metadata = json.loads(Path(metadata_path).read_text())
        evidence = metadata['robot_geometry']
        if evidence['encoding'] != 'zlib_json_base64_v1':
            raise ValueError('Unknown recorded native geometry format.')
        packed = base64.b64decode(evidence['geometry'], validate=True)
        if hashlib.sha256(packed).hexdigest() != evidence['geometry_sha256']:
            raise ValueError('Recorded native collision geometry hash mismatch.')
        geometry = json.loads(zlib.decompress(packed))
        with gzip.open(trace_path, 'rt') as stream:
            row = json.loads(next(stream))
        poses = dict(zip(map(tuple, metadata['robot_geometry']['link_keys']), row['robot_link_poses_world']))
        row['measured_left_tip_poses_world'] = [poses['left', tip] for tip in self.TIP_NAMES]
        row['measured_hand_pose_world'] = poses['left', 'gripper']
        groups = {tuple(g['key']): g for g in geometry}
        vertices = [np.vstack([piece['vertices'] for piece in groups['left', tip]['pieces']]) for tip in self.TIP_NAMES]
        self._grasp_geometry(vertices, row, 'Recorded native cooked fingertip geometry; CPU planning only, replaced by live bind')
        return self.grasp_calibration

    def hand_grasp_map(self, tool_T, half_gap_m):
        if self.grasp_calibration is None:
            raise RuntimeError('Actual fingertip calibration is required before grasp targets.')
        tool_T = np.asarray(tool_T, dtype=float)
        pose_matrix(matrix_pose(tool_T))
        face_half_gap = float(half_gap_m) - self.fixture_half_thickness
        if face_half_gap <= 0:
            raise ValueError('Finger centers must leave a positive physical opening.')
        left, right = self.grasp_calibration['zero_q_contact_planes_hand_x_m']
        q = (right - left) / 2 - face_half_gap
        if not -.04695 <= q <= 0:
            raise ValueError('Requested opening is outside actual YAM finger travel.')
        rotation = tool_T[:3, :3] @ np.asarray(self.grasp_calibration['hand_rotation_in_tool'])
        local = self.grip_local.copy()
        local[2] += float(half_gap_m) - self.spec['hold_half_gap']
        grasp_point = tool_T[:3, :3] @ local + tool_T[:3, 3]
        target = np.eye(4); target[:3, :3] = rotation
        target[:3, 3] = grasp_point - rotation @ np.asarray(self.grasp_calibration['contact_center_hand_m'])
        return target, float(q)

    def grasp_map(self, tool_T, half_gap_m):
        """Return the YAM IK TCP frame, not the gripper-link origin."""
        target, q = self.hand_grasp_map(tool_T, half_gap_m)
        target[:3, 3] += target[:3, :3] @ self.TCP_OFFSET
        return target, q
