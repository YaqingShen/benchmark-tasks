"""Camera-enabled, policy-stepped wrapper around the verified native scene."""
from __future__ import annotations

import atexit
from collections import deque
import copy
from datetime import datetime, timezone
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import sys
import uuid

import numpy as np
from wafer_robot_geometry import RobotGeometryGuard,compact_result,first_issue,apply_validity

_APP = None
_LIVE_WORLD = None


def _runtime(bundle):
    root = str(bundle)
    if root not in sys.path:
        sys.path.insert(0, root)
    modules = []
    for name in ("wafer_retention_isaac_v12", "wafer_full_cycle_isaac"):
        module = importlib.import_module(name)
        if Path(module.__file__).resolve().parent != bundle:
            raise RuntimeError("An imported native helper is outside the selected immutable bundle: " + name)
        modules.append(module)
    return modules


def _json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


class _InvalidTaskState(RuntimeError):
    pass


class _GeometryIntegrity:
    """Track USD geometry/topology changes; this is not a material-damage model."""
    def __init__(self, stage, roots):
        from pxr import Tf, Usd
        self.stage, self.roots = stage, roots
        self.initial = {name: self.signature(path) for name, path in roots.items()}
        if any(not value for value in self.initial.values()):
            raise ValueError("A declared task object is absent from the composed scene.")
        self.valid = dict.fromkeys(roots, True)
        self.dirty = set()
        self.listener = Tf.Notice.Register(Usd.Notice.ObjectsChanged, self.changed, stage)

    @staticmethod
    def tracked_property(prim, name):
        from pxr import UsdPhysics
        if name in {"physics:velocity", "physics:angularVelocity"}:
            return False
        return (name.startswith(("physics:", "physx", "drive:", "material:binding")) or
                name in {"points", "faceVertexCounts", "faceVertexIndices", "size", "radius", "height", "extent"} or
                (name.startswith("xformOp") and prim and prim.HasAPI(UsdPhysics.CollisionAPI) and
                 not prim.HasAPI(UsdPhysics.RigidBodyAPI)))

    def signature(self, root):
        from pxr import Usd
        prim = self.stage.GetPrimAtPath(root)
        if not prim or not prim.IsActive():
            return None
        values = []
        for child in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()):
            properties = []
            for attr in child.GetAttributes():
                name = attr.GetName()
                if self.tracked_property(child, name):
                    properties.append((name, str(attr.Get())))
            relationships = [(r.GetName(), [str(x) for x in r.GetTargets()], str(r.GetAllMetadata()))
                             for r in child.GetRelationships() if r.GetName().startswith(("physics:", "material:binding"))]
            values.append((str(child.GetPath()), child.GetTypeName(), child.IsActive(), properties, relationships))
        return hashlib.sha256(repr(values).encode()).hexdigest()

    def changed(self, notice, sender):
        paths = []
        for path in (*notice.GetResyncedPaths(), *notice.GetChangedInfoOnlyPaths()):
            if not path.IsPropertyPath() or self.tracked_property(self.stage.GetPrimAtPath(path.GetPrimPath()), path.name):
                paths.append(str(path.GetPrimPath()))
        for name, root in self.roots.items():
            if any(p == root or p.startswith(root + "/") or root.startswith(p + "/") for p in paths):
                self.dirty.add(name)

    def snapshot(self):
        for name in self.dirty:
            self.valid[name] = self.valid[name] and self.signature(self.roots[name]) == self.initial[name]
        self.dirty.clear()
        present = {name: bool(self.stage.GetPrimAtPath(root)) and self.stage.GetPrimAtPath(root).IsActive()
                   for name, root in self.roots.items()}
        return present, {name: bool(present[name] and self.valid[name]) for name in self.roots}

    def close(self):
        self.listener.Revoke()


def _extension(full_class):
    class MeasuredWorld(full_class):
        task_measurements_enabled = False

        def fields(self):
            fields = super().fields()
            fields.extend(("fixture_" + name, self.base.scene[name].data.root_link_pose_w, (1, 7))
                          for name in self.fixture_names)
            return fields

        def measurement(self, arrays, row, phase):
            result = super().measurement(arrays, row, phase)
            bodies = {s["body"] for s in self.tool_environment_filters}
            stands = set(self.metadata["tool_pickup"]["stand_geoms"])
            contacts, raw_contacts = [], []
            for index, spec in enumerate(self.tool_environment_filters):
                matrix = arrays[f"tool_environment_matrix_{index}"][0, 0]
                net = arrays[f"tool_environment_net_{index}"][0, 0]
                residual = float(np.linalg.norm(net - matrix.sum(axis=0)))
                if residual:
                    contacts.append([spec["body"], "unmatched_environment", residual])
                for partner, force in zip(spec["partners"], np.linalg.norm(matrix, axis=1)):
                    if force:
                        raw_contacts.append([spec["body"], partner, float(force)])
                    if partner in bodies or partner == "wafer" or partner.startswith("left_tip_") or partner in stands or partner in self.allowed_tool_particle_partners(result):
                        continue
                    if force:
                        contacts.append([spec["body"], partner, float(force)])
            result.update(tool_unexpected_environment_contacts_N=contacts,
                          tool_unexpected_environment_contact_normal_force_N=sum(c[2] for c in contacts),
                          tool_environment_partner_contacts_N=raw_contacts,
                          fixture_poses_world={name: arrays["fixture_" + name][0].tolist() for name in self.fixture_names})
            result["objects_present"], result["objects_intact"] = self.integrity.snapshot()
            poses={key:arrays[key[0]+'_poses'][0,index].tolist()for key,index in self.robot_body_ids.items()}
            result['robot_link_poses_world']=[poses[key]for key in self.robot_guard.keys]
            present,intact=self.robot_integrity.snapshot()
            checked=self.robot_guard.check(poses)if all(present.values())and all(intact.values())else{
                'status':'UNSUPPORTED','complete':False,'reason':'Robot collision geometry was removed or changed.'}
            result['robot_geometry_check']=compact_result(checked)
            result.pop("scan_row", None)
            result.pop("workflow_stage", None)
            return result
    return MeasuredWorld()


class NativeWaferWorld:
    def __init__(self, bundle, evidence_dir, width, height, device, development=False):
        self.env = self.recorder = self.integrity = self.robot_integrity = None
        try:
            self._build(bundle, evidence_dir, width, height, device, development)
        except Exception:
            self.close()
            raise

    def _build(self, bundle, evidence_dir, width, height, device, development):
        global _APP, _LIVE_WORLD
        if _LIVE_WORLD is not None:
            raise RuntimeError("Only one native wafer scene may exist in a process.")
        self.bundle, self.evidence_dir = Path(bundle), Path(evidence_dir)
        self.width, self.height, self.device = width, height, device
        self.env = self.recorder = self.integrity = None
        self.destination = None
        self.native, full = _runtime(self.bundle)
        native = self.native
        full.ALLOW_UNQUALIFIED_DEVELOPMENT = bool(development)
        plan, exported, models, gate = native.validate_bundle(self.bundle)
        self.plan, self.exported = plan, exported
        metadata = native.cloud_model_metadata(exported["metadata"], plan)
        from wafer_dust_isaac import DustFullCycle
        extension = self.extension = _extension(DustFullCycle)
        extension.validate(self.bundle, plan, metadata)
        metadata['dust']['return_detection']='measured_pin_recontact_v1'
        metadata['policy_candidate']='revision036_particle_interface_v1_unqualified'
        metadata['policy_score_definition']='Binary full handling and particle acceptance; handling progress is diagnostic.'
        metadata.update(engine="Isaac Lab / PhysX", local_validation=gate, contact_force_scope="wafer_partner_filtered",
                        controller={"native_finger_open_q_m": -0.04695}, benchmark_interface_version=1,
                        benchmark_private_measurements=True, benchmark_control_hz=20.0,
                        benchmark_cameras={"names": ["top_cam", "left_cam", "right_cam"], "width": width, "height": height},
                        task_score_scope="Measured ordered handling and nozzle coverage; no evaporation or material-damage model.",
                        force_measurement_limitations=["Filtered normal-vector norms; cancelling unfiltered contacts can hide in residuals."])
        metadata["benchmark_sources"] = {name+'.py': hashlib.sha256(Path(importlib.import_module(name).__file__).read_bytes()).hexdigest()
            for name in ("wafer_benchmark", "wafer_benchmark_backend", "wafer_task_score", "wafer_robot_geometry", "collision_geometry", "wafer_policy_score")}
        for name, digest in metadata["benchmark_sources"].items():
            if extension.full["files"].get("benchmark/" + name) != digest:
                raise ValueError("The installed benchmark source differs from the selected bundle: " + name)
        self.metadata = metadata
        # A scorer import failure must occur before reserving a GPU or constructing a scene.
        import wafer_task_score
        if hashlib.sha256(Path(wafer_task_score.__file__).read_bytes()).hexdigest() != metadata["benchmark_sources"]["wafer_task_score.py"]:
            raise ValueError("The imported task scorer differs from the benchmark package.")
        self.scorer_class = wafer_task_score.WaferTaskScorer
        if _APP is None:
            metadata["gpu_snapshot"] = native.ensure_gpu_idle()
            os.environ["ENABLE_CAMERAS"] = "1"
            from isaaclab.app import AppLauncher
            _APP = AppLauncher(headless=True, enable_cameras=True, device=device).app
            atexit.register(_APP.close)
        import torch
        from isaaclab.actuators import ImplicitActuatorCfg
        from isaaclab.sensors import ContactSensorCfg
        from isaaclab_arena.assets.object import Object
        from isaaclab_arena.assets.object_type import ObjectType
        from isaaclab_arena.utils.pose import Pose
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
        from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
        from yam_arena import constants, station
        from yam_arena.embodiment import BimanualYamEmbodiment
        import yam_arena.scene as scenes
        self.torch, self.constants = torch, constants
        if abs(float(constants.FINGER_OPEN) + 0.04695) > 1e-12 or abs(plan["native_finger_open_q_m"] - constants.FINGER_OPEN) > 1e-12:
            raise ValueError("Native normalized gripper mapping differs.")
        if not math.isclose(plan["world_translation_m"][2], station.TABLE_TOP + .025, abs_tol=1e-12):
            raise ValueError("Native station height differs from the verified scene.")
        embodiment = BimanualYamEmbodiment(enable_cameras=True, rig=station.rig(), cam_height=height,
            cam_width=width, robot_contact_offset=.0001, contact_sensors=True)
        left, right = embodiment.scene_config.left_robot, embodiment.scene_config.right_robot
        left.init_state.joint_pos = {**dict(zip(constants.ARM_JOINTS, plan["initial_left_q_rad"])),
                                    "joint[7-8]": plan["initial_finger_state_q_m"]}
        right.init_state.joint_pos = {**dict(zip(constants.ARM_JOINTS, plan["right_q_rad"])), "joint[7-8]": -.035}
        metadata["left_actuators"] = native.apply_arm_gains(left, plan["arm_gains"])
        metadata["right_actuators"] = native.apply_arm_gains(right, plan["right_arm_gains"])
        metadata["robot_materials"] = {name: native.install_robot_material_match(cfg.spawn)
            for name, cfg in (("left_robot", left), ("right_robot", right))}
        translation = native._array(plan["world_translation_m"], (3,))
        poses, metadata["initial_pose_tensor_validation"] = native.prepare_native_initial_poses(
            exported["initial_local_poses"], translation, Pose, torch, plan["world_yaw_deg"])
        objects = {name: Object(name=name, usd_path=str(path), initial_pose=poses[name],
            object_type=ObjectType.ARTICULATION if name == "clamp" else ObjectType.RIGID) for name, path in models.items()}
        self.springs = springs = metadata["usd_export"]["passive_springs"]
        objects["clamp"].object_cfg.actuators = {s["joint_name"] + "_spring": ImplicitActuatorCfg(
            joint_names_expr=[s["joint_name"]], stiffness=s["stiffness"], damping=s["damping"],
            joint_effort_limit=s["effort_limit"], joint_velocity_limit=s["velocity_limit"]) for s in springs}
        objects["clamp"].object_cfg.init_state.joint_pos = {s["joint_name"]: 0.0 for s in springs}
        objects["clamp"].object_cfg.init_state.joint_vel = {s["joint_name"]: 0.0 for s in springs}
        metadata["mechanism_drive_spawn"] = native.install_mechanism_drive_check(
            objects["clamp"].object_cfg.spawn, springs, metadata["usd_export"].get("tool_self_collision"))
        partners = [*exported["contact_link_names"], *exported["lever_link_names"], "pillar_0", "pillar_1", "pillar_2", "heater", "pedestal"]
        filters = [objects["clamp"].prim_path + "/" + name for name in partners[:4]]
        filters += [objects[name].get_contact_sensor_prim_path() for name in partners[4:]]
        extra = extension.wafer_contact_links()
        partners.extend(extra)
        filters.extend(objects["clamp"].prim_path + "/" + name for name in extra)
        contact = lambda prim, targets: ContactSensorCfg(prim_path=prim, filter_prim_paths_expr=targets,
            update_period=0.0, history_length=1, track_air_time=False, max_contact_data_count_per_prim=256)
        wafer_sensor = contact(objects["wafer"].get_contact_sensor_prim_path(), filters)
        finger_filters = [objects["clamp"].prim_path + "/" + name for name in extension.finger_contact_links()]
        sensors = {name + "_tool_contacts": contact(left.prim_path + embodiment.link_paths[name], finger_filters)
                   for name in ("tip_left", "tip_right")}
        sensors.update(extension.configure(objects, embodiment, Pose, Object, ObjectType, ImplicitActuatorCfg, ContactSensorCfg, torch))
        for name,path in extension.extra_wafer_contacts():
            partners.append(name);filters.append(path)
        wafer_sensor.filter_prim_paths_expr=filters
        original = scenes.env_cfg_callback(20)
        callback_count = 0
        def configure(cfg):
            nonlocal callback_count
            callback_count += 1
            cfg = original(cfg)
            if callback_count != 1 or abs(cfg.sim.dt - 1 / 240) > 1e-12 or cfg.decimation != 12:
                raise RuntimeError("Unexpected native baseline physics configuration.")
            cfg.sim.dt = metadata["physics_dt_s"]
            cfg.decimation = cfg.sim.render_interval = gate["decimation"]
            if not hasattr(cfg.sim.physics,'solve_articulation_contact_last'):
                raise RuntimeError('Native contact ordering is unavailable.')
            cfg.sim.physics.solve_articulation_contact_last=False
            cfg.sim.physics.solver_type=0
            native.configure_mechanism_reset(cfg,objects['clamp'])
            cfg.scene.wafer_contacts = wafer_sensor
            for name, sensor in sensors.items():
                setattr(cfg.scene, name, sensor)
            return cfg
        description = IsaacLabArenaEnvironment(name="wafer_blow_benchmark_v1",
            scene=Scene([*scenes.task_room("station", embodiment.rig.top_cam), *objects.values()]),
            embodiment=embodiment, env_cfg_callback=configure)
        self.env = ArenaEnvBuilder(description, ArenaEnvBuilderCfg(num_envs=1, solve_relations=False, device=device)).make_registered()
        if callback_count != 1:
            raise RuntimeError("Native callback count differs.")
        import omni.usd
        from isaaclab_arena.terms.events import reset_articulation_pose_and_joints
        stage=omni.usd.get_context().get_stage()
        scene_prim=stage.GetPrimAtPath(self.env.unwrapped.cfg.sim.physics_prim_path)
        contact_last=scene_prim.GetAttribute('physxScene:solveArticulationContactLast').Get()
        solver_type=scene_prim.GetAttribute('physxScene:solverType').Get()
        if contact_last is not False or solver_type!='PGS':
            raise RuntimeError('Policy physics solver differs from revision036.')
        event=self.env.unwrapped.cfg.events.clamp
        if event.func is not reset_articulation_pose_and_joints or event.mode!='reset' or event.params['asset_cfg'].name!='clamp':
            raise RuntimeError('Policy passive mechanism reset differs from revision036.')
        if not np.allclose(event.params['pose'].to_tensor(device='cpu'),objects['clamp'].initial_pose.to_tensor(device='cpu'),atol=1e-9,rtol=0):
            raise RuntimeError('Policy passive mechanism reset pose differs.')
        metadata['native_solver_contact_order']=dict(solver_type=solver_type,solve_articulation_contact_last=contact_last)
        metadata['mechanism_initialization_reset']=dict(mode='reset',function='isaaclab_arena.terms.events.reset_articulation_pose_and_joints',scope='Reset before the trial only; no in-episode state replacement.')
        self.env.reset()
        self.base = base = self.env.unwrapped
        import omni.usd
        stage = omni.usd.get_context().get_stage()
        spawn = metadata["mechanism_drive_spawn"]
        if spawn["callback_count"] != 1 or spawn["status"] != "COMPOSED_FORCE_DRIVE_VERIFIED":
            raise RuntimeError("Passive mechanical springs were not verified at spawn.")
        for side, material in metadata["robot_materials"].items():
            if material["callback_count"] != 1 or material["status"] != "COMPOSED_BINDINGS_VERIFIED":
                raise RuntimeError("Robot material binding differs: " + side)
            native.verify_robot_material(stage.GetPrimAtPath(material["root_path"]))
            native.verify_live_robot_material(base.scene[side])
        metadata["spawned_assets"] = native.verify_spawned_assets(exported)
        metadata["generated_materials"] = native.verify_live_generated_materials(stage, models,
            {name: base.scene[name] for name in ("clamp", "wafer")}, metadata["spawned_assets"])
        if abs(float(base.physics_dt) - metadata["physics_dt_s"]) > 1e-12 or abs(float(base.step_dt) - .05) > 1e-12:
            raise RuntimeError("Native physics or control period differs.")
        expected_sensors = {"left_contacts", "right_contacts", "wafer_contacts", *sensors, "top_cam", "left_cam", "right_cam"}
        if set(base.scene.sensors) != expected_sensors:
            raise RuntimeError("The scene does not contain exactly the required contact and three camera sensors.")
        if base.observation_manager.active_terms != {"policy": ["joint_pos"], "camera_obs": ["top_cam_rgb", "left_cam_rgb", "right_cam_rgb"]}:
            raise RuntimeError("Native camera/joint observation terms differ.")
        metadata["verified_live_left_actuators"] = native.verify_live_arm_gains(base.scene["left_robot"], metadata["left_actuators"])
        extension.ready(base, metadata, torch)
        extension.fixture_names = tuple(sorted(set(objects) - {"wafer", "clamp", "gun", *extension.dust_names}))
        roots = {name: obj.prim_path.replace("{ENV_REGEX_NS}", "/World/envs/env_0").replace("{ENV_NS}", "/World/envs/env_0").replace("/env_.*/", "/env_0/")
                 for name, obj in objects.items()}
        self.integrity = extension.integrity = _GeometryIntegrity(stage, roots)
        robot_roots={side:metadata['robot_materials'][side+'_robot']['root_path']for side in ('left','right')}
        self.robot_guard=extension.robot_guard=RobotGeometryGuard.from_native(
            {side:stage.GetPrimAtPath(path)for side,path in robot_roots.items()})
        self.robot_integrity=extension.robot_integrity=_GeometryIntegrity(stage,robot_roots)
        metadata['robot_geometry']=self.robot_guard.evidence()
        extension.robot_body_ids={}
        for side,name in self.robot_guard.keys:
            ids,names=base.scene[side+'_robot'].find_bodies(name)
            if list(names)!=[name]or len(ids)!=1:raise ValueError('Robot collision link readback is ambiguous.')
            extension.robot_body_ids[(side,name)]=int(ids[0])
        origin = native._array(base.scene.env_origins, (1, 3))[0]
        angle = math.radians(plan["world_yaw_deg"])
        c, s = math.cos(angle), math.sin(angle)
        rotation = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        metadata["pillar_top_centers_world_m"] = (np.asarray(metadata["pillar_top_centers_world_m"]) @ rotation.T + translation + origin).tolist()
        metadata["filter_partner_names"], metadata["filter_prim_paths_expr"] = partners, filters
        metadata["passive_spring_actuations"] = springs
        metadata["mechanism_actuation"] = metadata["usd_export"]["actuation"]
        def planned_pose(name):
            pose = objects[name].initial_pose
            return [*(np.asarray(pose.position_xyz, dtype=float) + origin), *np.asarray(pose.rotation_xyzw, dtype=float)]
        metadata["task_score_reference"] = {
            "fixture_poses_world": {n: planned_pose(n) for n in extension.fixture_names},
            "initial_object_poses_world": {n: planned_pose(n) for n in ("wafer", "clamp", "gun")},
            "object_names": sorted(objects)}
        metadata["integrity_scope"] = "Hash-verified task assets, USD topology/physics/geometry notices and measured fixture poses; no material-damage model."
        self.arm_ids, self.finger_ids = {}, {}
        for side in ("left_robot", "right_robot"):
            arm_ids, names = base.scene[side].find_joints(list(constants.ARM_JOINTS), preserve_order=True)
            finger_ids, finger_names = base.scene[side].find_joints(list(constants.FINGER_JOINTS), preserve_order=True)
            if list(names) != list(constants.ARM_JOINTS) or list(finger_names) != list(constants.FINGER_JOINTS):
                raise RuntimeError("Native joint ordering differs.")
            self.arm_ids[side], self.finger_ids[side] = list(arm_ids), list(finger_ids)
        self.reader = native.make_reader(base, metadata, partners, extension=extension)
        self._restore_spring_targets()
        self._assert_initial_state()
        _LIVE_WORLD = self

    def _assert_initial_state(self):
        checks = {}
        for side, target, finger in (("left_robot", self.plan["initial_left_q_rad"], self.plan["initial_finger_state_q_m"]),
                                     ("right_robot", self.plan["right_q_rad"], -.035)):
            q = self.native._array(self.base.scene[side].data.joint_pos)[0]
            actual = np.r_[q[self.arm_ids[side]], q[self.finger_ids[side]]]
            expected = np.r_[target, [finger, finger]]
            error = float(np.max(abs(actual - expected)))
            if error > 1e-6:
                raise ValueError("Reset does not match the qualified empty-hand joint state: " + side)
            checks[side] = error
        initial = self.metadata["tool_pickup"]["initial_joint_positions"]
        names = [s["joint_name"] for s in self.springs]
        expected = np.array([float(initial[name]) for name in names])
        q = np.asarray(self.native._joint_positions(self.base.scene["clamp"], names))
        v = np.asarray(self.native._joint_velocities(self.base.scene["clamp"], names))
        if not np.allclose(q, expected, atol=1e-6, rtol=0) or np.max(abs(v)) > 1e-6:
            raise ValueError("Passive-joint reset mismatch: " + repr(dict(names=names,expected=expected.tolist(),actual=q.tolist(),velocities=v.tolist())))
        trigger = self.native._joint_positions(self.extension.gunasset, ["gun_trigger_slide"])[0]
        if abs(trigger) > 1e-6:
            raise ValueError("The gun trigger must start released.")
        self.metadata["benchmark_initial_state_checks"] = {"maximum_robot_joint_error": checks,
            "passive_joint_names": names, "expected_passive_joint_positions": expected.tolist(),
            "measured_passive_joint_positions": q.tolist(), "measured_passive_joint_velocities": v.tolist(),
            "gun_trigger_m": trigger}

    def _restore_spring_targets(self):
        clamp = self.base.scene["clamp"]
        names = [s["joint_name"] for s in self.springs]
        ids, found = clamp.find_joints(names, preserve_order=True)
        if list(found) != names or len(set(ids)) != len(names):
            raise ValueError("Passive spring mapping differs.")
        targets = [s["constant_target"] for s in self.springs]
        clamp.actuators.target_command.set_position_index(
            value=self.torch.tensor([targets], device=self.base.device, dtype=self.torch.float32), joint_ids=list(ids))
        clamp.write_data_to_sim()
        for spec in self.springs:
            self.native.native_mechanism_readback(clamp, spec, spec["constant_target"], expected_dof_count=len(names))
        trigger = self.extension.gunasset
        trigger.actuators.target_command.set_position_index(
            value=self.torch.tensor([[0.0]], device=self.base.device, dtype=self.torch.float32), joint_ids=[self.extension.trigger_id])
        trigger.write_data_to_sim()

    def reset(self, *, seed, scene_id):
        self.finish()
        self.env.reset(seed=seed)
        self._restore_spring_targets()
        self._assert_initial_state()
        if not all(self.integrity.snapshot()[1].values()):
            raise ValueError("Task geometry was changed; start a new process from the immutable bundle.")
        from wafer_dust import SurfaceAdhesion,CohesiveDustScore
        self.extension.bonds=SurfaceAdhesion(self.metadata['dust']['initial_centers_wafer_m'],self.metadata['dust']['profile'])
        self.extension.dust_score=CohesiveDustScore(self.metadata)
        for asset in [*self.extension.particles,self.extension.wafer]:
            asset.permanent_wrench_composer.reset()
        self.monitor = self.scorer_class(self.metadata, sample_dt_s=self.metadata["physics_dt_s"])
        self._diagnostics = {}
        self._runtime_error = None
        self._geometry_issue = None
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
        self.destination = self.evidence_dir / stamp
        self.destination.mkdir(parents=True, exist_ok=False)
        self.metadata["benchmark_trial"] = {"scene_id": str(scene_id), "seed": seed, "post_start_pose_write_count": 0}
        self.recorder = self.native.PhysicsRecorder(self.base.scene, self.reader, self.destination,
            self.metadata["physics_dt_s"], rows=deque(maxlen=2))
        row = self.recorder.initial("policy")
        self.monitor.reset(row,report=False)
        self.extension.apply_particle_forces(row)
        self._geometry_issue=first_issue(None,row)
        self.recorder.sample_guard = self._consume
        _json(self.destination / "metadata.json", self.metadata)
        # Renderer warm-up does not advance physics or move any object.
        for _ in range(4):
            self.base.sim.render()

    def _consume(self, row):
        from wafer_tool_pickup import guard_handle_paddles
        from wafer_cycle_isaac import Cycle
        checks = {
            "legacy_mechanism_bounds": lambda: self.native.guard_physics_loads_v12(row, self.metadata["pad_compliance"], self.metadata["passive_hinge"]),
            "legacy_handle_paddle_bounds": lambda: guard_handle_paddles(row, self.metadata),
            "legacy_gun_bounds": lambda: Cycle.guard(self.extension, row),
        }
        for name, check in checks.items():
            try:
                check()
            except RuntimeError as exc:
                entry = self._diagnostics.setdefault(name, {"count": 0, "first_time_s": row["time_s"], "message": str(exc)})
                entry["count"] += 1
        self.monitor.update(row,report=False)
        try:
            self.extension.apply_particle_forces(row)
        except RuntimeError as exc:
            self.monitor._invalidate('Particle evidence: '+str(exc))
        self._geometry_issue=first_issue(self._geometry_issue,row)
        if self.monitor.invalid or self._geometry_issue:
            raise _InvalidTaskState("Measured task state became invalid.")

    def step(self, target):
        if self.recorder is None:
            raise RuntimeError("No active trial.")
        self.recorder.begin("policy")
        try:
            self.env.step(self.torch.tensor([np.asarray(target).tolist()], device=self.base.device, dtype=self.torch.float32))
        except _InvalidTaskState:
            self.recorder.active = None
            self.recorder.stream.flush()
        except Exception as exc:
            self._runtime_error = type(exc).__name__ + ": " + str(exc)
            raise
        else:
            self.recorder.end()

    def observe(self):
        from inspect_robots_yam.packing import pack
        images = {}
        for name in ("top_cam", "left_cam", "right_cam"):
            rgb = self.base.scene[name].data.output["rgb"]
            rgb = rgb.torch if hasattr(rgb, "torch") else rgb
            rgb = rgb[0, ..., :3].detach().cpu().numpy()
            if rgb.shape != (self.height, self.width, 3) or rgb.dtype != np.uint8:
                raise RuntimeError("Actual RGB camera output has an unexpected shape or dtype: " + name)
            images[name] = np.ascontiguousarray(rgb)
        arms = []
        for side in ("left_robot", "right_robot"):
            q = self.native._array(self.base.scene[side].data.joint_pos)[0]
            arms.append(np.r_[q[self.arm_ids[side]], np.clip(q[self.finger_ids[side]].mean() / self.constants.FINGER_OPEN, 0, 1)])
        return images, pack(*arms), self.recorder.total * self.recorder.dt if self.recorder else self.last_time

    def score(self):
        from wafer_policy_score import combine_score
        return combine_score(apply_validity(self.monitor.result(),self._geometry_issue),self.extension.dust_score.result())

    def diagnostics(self):
        return {"scope": "Original qualification thresholds, separate from task score", "failures": copy.deepcopy(self._diagnostics)}

    def evidence_reference(self):
        return {"directory": str(self.destination), "trace": "samples.jsonl.gz", "cadence_hz": 1 / self.metadata["physics_dt_s"]}

    def finish(self):
        if self.recorder is None:
            return
        self.last_time = self.recorder.total * self.recorder.dt
        count = self.recorder.total + 1
        last_steps = self.recorder.count
        self.recorder.close()
        self.recorder = None
        digest = hashlib.sha256()
        with (self.destination / "samples.jsonl.gz").open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        _json(self.destination / "task_result.json", {"task": self.score(), "physics_sample_count": count,
            "duration_s": self.last_time, "qualification_diagnostics": self.diagnostics(), "post_start_pose_write_count": 0,
            "trace_sha256": digest.hexdigest(), "last_control_physics_steps": last_steps, "runtime_error": self._runtime_error,
            "end_kind": "runtime_error" if self._runtime_error else "incomplete_evidence" if not self.score()['evidence_complete']else "measured_invalid" if self.score()["invalid_reasons"] else "episode_closed"})

    def close(self):
        global _LIVE_WORLD
        self.finish()
        if self.integrity is not None:
            self.integrity.close()
            self.integrity = None
        if self.robot_integrity is not None:
            self.robot_integrity.close()
            self.robot_integrity = None
        if getattr(getattr(self,'extension',None),'dust_integrity',None) is not None:
            self.extension.dust_integrity.close();self.extension.dust_integrity=None
        if self.env is not None:
            self.env.close()
            self.env = None
        if _LIVE_WORLD is self:
            _LIVE_WORLD = None
