"""Native environment for direct 14-D policy actions; no scripted substitution.

The native correction probes remain separate development diagnostics. This
backend exposes cameras and measured joint positions only to its policy.
"""
import atexit
import hashlib
import json
import os
from pathlib import Path
import numpy as np

_APP = None


class FormulationWorld:
    def __init__(self, root, output, width=1280, height=704, device='cpu'):
        global _APP
        self.root, self.output = Path(root).resolve(), Path(output).resolve()
        self.width, self.height = int(width), int(height)
        self.env = self.coupling = None
        self.trial = None
        self.trial_count = 0
        self.manifest = json.loads((self.root/'assets/mechanisms/native-assets.json').read_text())
        source = json.loads((self.root/'source_manifest.json').read_text())
        for name, digest in source['files'].items():
            path = (self.root/name).resolve()
            if not path.is_relative_to(self.root) or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise ValueError('Frozen source mismatch: '+name)
        self.revision = source['revision']
        self.source_digest = hashlib.sha256((self.root/'source_manifest.json').read_bytes()).hexdigest()
        os.environ['ENABLE_CAMERAS'] = '1'
        if _APP is None:
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
        from yam_arena import station, constants
        from yam_arena.embodiment import BimanualYamEmbodiment
        import yam_arena.scene as scenes
        self.torch, self.constants = torch, constants
        if abs(station.TABLE_TOP-self.manifest['table_top_m']) > 1e-6:
            raise ValueError('Station height differs from the source assets')
        embodiment = BimanualYamEmbodiment(enable_cameras=True, rig=station.rig(),
            cam_height=self.height, cam_width=self.width, robot_contact_offset=.0001, contact_sensors=True)
        for side in ('left_robot', 'right_robot'):
            getattr(embodiment.scene_config, side).actuators['gripper'].joint_effort_limit = 5.
        objects = {}
        for name, spec in self.manifest['assets'].items():
            path = self.root/'assets/mechanisms'/spec['path']
            if hashlib.sha256(path.read_bytes()).hexdigest() != spec['sha256']:
                raise ValueError('Asset checksum mismatch: '+name)
            obj = Object(name=name, usd_path=str(path), initial_pose=Pose.from_dict(spec),
                         object_type=getattr(ObjectType, spec.get('object_type', 'RIGID')))
            if not spec.get('dynamic', True):
                obj.disable_reset_pose()
            springs = spec.get('passive_springs')
            if springs:
                obj.object_cfg.actuators = {s['joint_name']+'_spring': ImplicitActuatorCfg(
                    joint_names_expr=[s['joint_name']], stiffness=s['stiffness'], damping=s['damping'],
                    joint_effort_limit=s['effort_limit'], joint_velocity_limit=s['velocity_limit']) for s in springs}
                obj.object_cfg.init_state.joint_pos = {s['joint_name']: 0. for s in springs}
                obj.object_cfg.init_state.joint_vel = {s['joint_name']: 0. for s in springs}
            objects[name] = obj
        def sensor(prim, filters):
            return ContactSensorCfg(prim_path=prim, filter_prim_paths_expr=filters,
                update_period=0., history_length=1, track_air_time=False, max_contact_data_count_per_prim=1024)
        beaker = objects['beaker'].get_contact_sensor_prim_path()
        balance = objects['digital_balance'].get_contact_sensor_prim_path()
        tool_parts = [objects['pipette_A'].prim_path+'/'+p for p in ('stem', 'bulb_left', 'bulb_right')]
        sensors = {'balance_contacts': sensor(beaker, [balance])}
        robot_links = []
        for side in ('left', 'right'):
            root_path = getattr(embodiment.scene_config, side+'_robot').prim_path
            robot_links += [root_path+path for path in embodiment.link_paths.values()]
            for finger in ('tip_left', 'tip_right'):
                sensors[side+'_'+finger+'_tool_contact'] = sensor(root_path+embodiment.link_paths[finger], tool_parts)
        sensors['measurement_contacts'] = sensor(beaker, list(dict.fromkeys(robot_links+tool_parts)))
        standard = scenes.env_cfg_callback(20)
        def configure(cfg):
            cfg = standard(cfg)
            cfg.sim.dt = .001
            cfg.decimation = 50
            cfg.sim.render_interval = 50
            cfg.episode_length_s = 600.
            for name, value in sensors.items():
                setattr(cfg.scene, name, value)
            return cfg
        description = IsaacLabArenaEnvironment(name='liquid_dose_correction',
            scene=Scene([*scenes.task_room('station', embodiment.rig.top_cam), *objects.values()]),
            embodiment=embodiment, env_cfg_callback=configure)
        self.env = ArenaEnvBuilder(description, ArenaEnvBuilderCfg(num_envs=1, solve_relations=False, device=device)).make_registered()
        self.base = self.env.unwrapped
        import omni.replicator.core as rep
        rep.settings.set_render_rtx_realtime(antialiasing='DLAA')

    def joint_positions(self):
        from native_coupling import array
        values = []
        for side in ('left_robot', 'right_robot'):
            arm = self.base.scene[side]
            ids, names = arm.find_joints(list(self.constants.ARM_JOINTS), preserve_order=True)
            fingers, _ = arm.find_joints(list(self.constants.FINGER_JOINTS), preserve_order=True)
            if list(names) != list(self.constants.ARM_JOINTS):
                raise ValueError('Unexpected native joint order')
            q = array(arm.data.joint_pos)[0]
            values.extend([*q[ids], float(np.clip(q[fingers].mean()/self.constants.FINGER_OPEN, 0, 1))])
        return np.array(values, dtype=np.float32)

    def reset(self, seed=0, initial_mass=None):
        from native_coupling import NativeCoupling
        from native_task_score import NativeTaskMonitor
        from reset_cases import starting_mass
        if self.coupling is not None:
            self.coupling.close()
        self.env.reset()
        self.trial = self.output/f'trial-{self.trial_count:03d}'
        self.trial_count += 1
        self.trial.mkdir(parents=True, exist_ok=False)
        self.frames = self.trial/'camera_frames'
        for camera in ('top_cam', 'left_cam', 'right_cam'):
            (self.frames/camera).mkdir(parents=True)
        for name, spec in self.manifest['assets'].items():
            if spec.get('passive_springs'):
                part = self.base.scene[name]
                ids, _ = part.find_joints([s['joint_name'] for s in spec['passive_springs']], preserve_order=True)
                part.actuators.target_command.set_position_index(
                    value=self.torch.zeros((1, len(ids)), device=self.base.device), joint_ids=list(ids))
                part.write_data_to_sim()
        self.coupling = NativeCoupling(self.base, self.manifest)
        self.coupling.active = True
        command = self.joint_positions()
        for _ in range(40):
            self._step(command)
        if self.coupling.balance.tare_g is None:
            raise RuntimeError('Empty-beaker tare did not settle')
        mass = starting_mass(seed) if initial_mass is None else float(initial_mass)
        if not 8.5 <= mass <= 11.5 or abs(mass-10.) < .15-1e-9:
            raise ValueError('Initial mass outside the declared reset distribution')
        ledger = self.coupling.ledger
        ledger._receive('beaker', ledger.take_from('bottle_A', mass/ledger.cfg['density_g_ml']['A']))
        ledger.initial_beaker_g = mass
        ledger.initial_error_g = abs(mass-10.)
        for _ in range(40):
            self._step(command)
        self.monitor = NativeTaskMonitor(self.base, self.coupling)
        self.steps = 0
        self.trace = (self.trial/'measurements.jsonl').open('w')
        self.actions = (self.trial/'actions.jsonl').open('w')
        self.metadata = {'source_revision': self.revision, 'source_manifest_sha256': self.source_digest,
            'seed': seed, 'initial_mass_g': mass, 'cameras': ['top_cam', 'left_cam', 'right_cam'],
            'policy_inputs': ['RGB images', '14-D measured joint positions'], 'control_hz': 20,
            'physics_hz': 1000, 'frame_rate': 5, 'overlays': False,
            'liquid_model': 'uncalibrated reduced pneumatic and quasi-static flow model',
            'post_reset_object_pose_writes': 0, 'max_steps': 12000}
        (self.trial/'metadata.json').write_text(json.dumps(self.metadata, indent=2))
        images, _, _ = self.observe()
        from PIL import Image
        Image.fromarray(images['top_cam']).save(self.trial/'setup.png')
        return self.observe()

    def _step(self, target):
        if self.coupling is not None:
            self.coupling.render()
        self.env.step(self.torch.tensor([np.asarray(target).tolist()], device=self.base.device, dtype=self.torch.float32))

    def step(self, target):
        target = np.asarray(target)
        if target.shape != (14,) or not np.isfinite(target).all():
            raise ValueError('Expected 14 finite action values')
        self._step(target)
        self.steps += 1
        row = self.monitor.advance()
        self.trace.write(json.dumps(row, allow_nan=False)+'\n')
        self.actions.write(json.dumps(target.tolist(), allow_nan=False)+'\n')
        if self.steps % 4 == 0:
            from PIL import Image
            images, _, _ = self.observe()
            for camera, rgb in images.items():
                Image.fromarray(rgb).save(self.frames/camera/f'frame_{self.steps//4-1:05d}.png')
        if self.steps % 20 == 0:
            self.trace.flush()
            self.actions.flush()
            (self.trial/'progress.json').write_text(json.dumps({'steps': self.steps, **self.monitor.score()}, indent=2))

    def observe(self):
        from native_coupling import array
        images = {n: array(self.base.scene[n].data.output['rgb'])[0, :, :, :3].astype(np.uint8)
                  for n in ('top_cam', 'left_cam', 'right_cam')}
        return images, self.joint_positions(), self.steps/20.

    def score(self):
        return self.monitor.score()

    def finish(self):
        if getattr(self, 'trace', None) is not None and not self.trace.closed:
            self.trace.close()
            self.actions.close()
            (self.trial/'result.json').write_text(json.dumps(self.score(), indent=2))

    def close(self):
        self.finish()
        if self.coupling is not None:
            self.coupling.close()
            self.coupling = None
        if self.env is not None:
            self.env.close()
            self.env = None
