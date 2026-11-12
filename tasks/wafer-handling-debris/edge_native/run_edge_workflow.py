"""Native YAM world and scripted runner for the passive MCP2-inspired pick."""
from pathlib import Path
import argparse
import copy
import gzip
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
import numpy as np


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def gpu_snapshot():
    text = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory',
                                    '--format=csv,noheader,nounits'], text=True)
    active = [line for line in text.splitlines() if line.strip() and 'gnome-remote-desktop-daemon' not in line]
    minimum = os.environ.get('WAFER_GPU_MIN_FREE_MIB')
    if minimum is not None:
        minimum = int(minimum)
        if minimum < 12288:
            raise ValueError('Shared GPU mode requires at least 12288 MiB free.')
        free = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free',
                                        '--format=csv,noheader,nounits'], text=True)
        if min(int(value) for value in free.splitlines()) < minimum:
            raise RuntimeError('Insufficient GPU headroom; no process was stopped.')
        text += '\nShared GPU minimum free MiB: ' + str(minimum) + '; observed: ' + free.strip()
    elif active:
        raise RuntimeError('GPU is occupied; no process was stopped: ' + '; '.join(active))
    return text.strip()


class NativeEdgeWorld:
    """One fresh scene per process; only robot actions advance physics."""
    def __init__(self, root, baseline, output, *, stage='full', device='cpu', seed=17,
                 enable_cameras=False, width=320, height=176, defer_application_close=False):
        self.root, self.baseline, self.output = [Path(p).resolve() for p in (root, baseline, output)]
        if stage not in ('spawn', 'tool', 'wafer', 'full') or device not in ('cpu', 'cuda:0'):
            raise ValueError('Unsupported native stage or device.')
        if type(enable_cameras) is not bool or int(width) < 16 or int(height) < 16:
            raise ValueError('Invalid camera configuration.')
        if type(seed) is not int or seed < 0:
            raise ValueError('Use a nonnegative integer particle seed.')
        self.stage, self.device, self.seed = stage, device, seed
        self.enable_cameras, self.width, self.height = enable_cameras, int(width), int(height)
        self.app = self.env = self.base = self.stream = self.commands = None
        self.air = self.evidence = self.scorer = self.adapter = self.controller = None
        self.metadata, self.baseline_hashes = {}, {}
        self.last = self.action = None
        self.count = 0
        self.started = time.monotonic()
        self._attempted = self._ready = self._closed = self._output_created = False
        self.runtime_error = None
        self.report = None
        self.defer_application_close = bool(defer_application_close)

    def start(self):
        if self._attempted or self._closed:
            raise RuntimeError('Create a fresh process for each native scene reset.')
        self._attempted = True
        self.started = time.monotonic()
        try:
            self._build()
            self._ready = True
            return self
        except Exception as exc:
            self.runtime_error = type(exc).__name__ + ': ' + str(exc)
            self.close(dict(result='INCOMPLETE', error=self.runtime_error, traceback=traceback.format_exc()))
            raise

    def _build(self):
        root, baseline = self.root, self.baseline
        sys.path[:0] = [str(root), str(baseline / 'benchmark'), str(baseline)]
        self.output.mkdir(parents=True, exist_ok=False)
        self._output_created = True
        self.plan = json.loads((baseline / 'cloud_plan.json').read_text())
        self.full_plan = json.loads((baseline / 'full_workflow_plan.json').read_text())
        baseline_files = [*baseline.glob('*.py'), *baseline.glob('benchmark/*.py'),
                          *baseline.glob('assets/*.usda'), baseline / 'cloud_plan.json',
                          baseline / 'full_workflow_plan.json']
        baseline_files = [p for p in baseline_files if not p.name.startswith('._')]
        self.baseline_hashes = {str(p.relative_to(baseline)): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in baseline_files}
        self.metadata = metadata = dict(version='mcp2_native_development_v1', stage=self.stage,
            physics_dt_s=.0005, control_hz=20, physics_device=self.device,
            source_files={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in root.glob('*.py') if not p.name.startswith('._')},
            local_mechanism_report=json.loads((root / 'batch_report.json').read_text())['successes'],
            post_start_pose_write_count=0, baseline_root=str(baseline),
            baseline_source_sha256=self.baseline_hashes,
            scope='New native integration; previous-tool success is not evidence for this tool.')
        metadata['gpu_snapshot'] = gpu_snapshot()
        if self.enable_cameras:
            metadata['benchmark_cameras'] = dict(names=['top_cam', 'left_cam', 'right_cam'],
                                                width=self.width, height=self.height)
        dump(self.output / 'metadata.json', metadata)
        os.environ['ENABLE_CAMERAS'] = '1' if self.enable_cameras else '0'
        from isaaclab.app import AppLauncher
        self.app = AppLauncher(headless=True, enable_cameras=self.enable_cameras, device=self.device).app
        import torch
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
        from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
        from yam_arena import constants, station
        from yam_arena.embodiment import BimanualYamEmbodiment
        import yam_arena.scene as scenes
        import wafer_retention_isaac_v12 as old
        from native_adapter import NativeEdgePick
        from native_air import NativeAir
        self.torch, self.constants, self.native = torch, constants, old
        self.adapter = adapter = NativeEdgePick(root / 'native',
            world_translation_m=(.42, -.14, station.TABLE_TOP), world_yaw_deg=270, dark_appearance=True)
        objects = dict(adapter.make_objects())
        camera_args = dict(cam_width=self.width, cam_height=self.height) if self.enable_cameras else {}
        embodiment = BimanualYamEmbodiment(enable_cameras=self.enable_cameras, rig=station.rig(),
            robot_contact_offset=.0001, contact_sensors=True, **camera_args)
        if not self.enable_cameras:
            embodiment.camera_config = None
        q_left = np.asarray(self.full_plan['tool_pickup']['start_arm_q_rad'])
        q_right = np.asarray(self.full_plan['gun']['approach_q_rad'])
        metadata['robot_materials'] = {}
        for side, q in [('left', q_left), ('right', q_right)]:
            cfg = getattr(embodiment.scene_config, side + '_robot')
            cfg.init_state.joint_pos = {**dict(zip(constants.ARM_JOINTS, q)), 'joint[7-8]': -.035}
            gains = self.plan['arm_gains' if side == 'left' else 'right_arm_gains']
            metadata[side + '_actuators'] = old.apply_arm_gains(cfg, gains)
            if side == 'left':
                metadata['robot_materials'][side] = adapter.install_left_robot_material(cfg.spawn)
            else:
                metadata['robot_materials'][side] = old.install_robot_material_match(cfg.spawn)
        sensors = adapter.configure(embodiment)
        self.air = air = NativeAir(baseline, self.full_plan, metadata, seed=self.seed)
        sensors.update(air.configure(objects, embodiment))
        original = scenes.env_cfg_callback(20)

        def configure(cfg):
            cfg = original(cfg)
            cfg.sim.dt = .0005
            cfg.decimation = 100
            cfg.sim.render_interval = 100
            cfg.sim.physics.solve_articulation_contact_last = False
            cfg.sim.physics.solver_type = 0
            for name, sensor in sensors.items():
                setattr(cfg.scene, name, sensor)
            return adapter.configure_env(cfg)

        description = IsaacLabArenaEnvironment(name='yam_mcp2_visible_debris',
            scene=Scene([*scenes.task_room('station', embodiment.rig.top_cam), *objects.values()]),
            embodiment=embodiment, env_cfg_callback=configure)
        self.env = ArenaEnvBuilder(description, ArenaEnvBuilderCfg(num_envs=1, solve_relations=False,
                                                                  device=self.device)).make_registered()
        self.env.reset()
        self.base = base = self.env.unwrapped
        metadata['native_adapter'] = adapter.bind(base)
        metadata['robot_cfg_prim_paths'] = {s: str(base.scene[s + '_robot'].cfg.prim_path)
                                            for s in ('left', 'right')}
        air.bind(base)
        if abs(float(base.physics_dt) - .0005) > 1e-12 or abs(float(base.step_dt) - .05) > 1e-12:
            raise RuntimeError('Native cadence differs from the declared 2,000/20 Hz timing.')
        metadata['physics_device_actual'] = str(base.device)
        if (root / 'native_evidence.py').is_file():
            from native_evidence import NativeEvidence
            self.evidence = NativeEvidence(base, adapter, objects, metadata['dust'], metadata['gun'],
                robot_roots={s: m['root_path'] for s, m in metadata['robot_materials'].items()})
            metadata.update(self.evidence.metadata)
        dump(self.output / 'metadata.json', metadata)
        self.stream = gzip.open(self.output / 'observations.jsonl.gz', 'wt', compresslevel=1)
        self.commands = (self.output / 'commands.jsonl').open('w')
        self.action = np.r_[q_left, -.035 / constants.FINGER_OPEN, q_right, -.035 / constants.FINGER_OPEN]
        self.last = self._measure(0., 'initialization')
        if self.evidence is not None:
            metadata.update(self.evidence.metadata)
            dump(self.output / 'metadata.json', metadata)
            from wafer_debris_score import VisibleDebrisScorer
            self.scorer = VisibleDebrisScorer(metadata, .05)
            self.scorer.update(self.last, report=False)
        self.stream.write(json.dumps(self.last, allow_nan=False) + '\n')
        air.hook()
        if self.enable_cameras:
            for _ in range(4):
                base.sim.render()
            if air.total != 0:
                raise RuntimeError('Camera warm-up advanced physics.')

    def _measure(self, time_s, phase):
        row = self.air.measurement(self.adapter.read_row(time_s, phase))
        if self.evidence is not None:
            row = self.evidence.attach(row)
        return row

    def read_row(self):
        if self.last is None:
            raise RuntimeError('The native scene has not started.')
        return self.last

    def step_action(self, target, phase='policy'):
        if not self._ready or self._closed or self.runtime_error is not None:
            raise RuntimeError('No active native trial.')
        try:
            target = np.asarray(target, dtype=float)
            if target.shape != (14,) or not np.isfinite(target).all():
                raise ValueError('Invalid native robot action.')
            if (self.count + 1) * .05 > 120.0000001:
                raise RuntimeError('The complete workflow exceeded 120 simulation seconds.')
            self.commands.write(json.dumps(dict(time_s=self.count * .05, phase=phase, action=target.tolist())) + '\n')
            self.action = target.copy()
            self.air.substeps = 0
            self.env.step(self.torch.tensor([self.action.tolist()], device=self.base.device, dtype=self.torch.float32))
            if self.air.substeps != 100:
                raise RuntimeError('An action did not produce exactly 100 existing physics updates.')
            self.count += 1
            self.last = self._measure(self.count * .05, phase)
            if self.scorer is not None:
                self.scorer.update(self.last, report=False)
            self.stream.write(json.dumps(self.last, allow_nan=False) + '\n')
            if self.count % 20 == 0:
                self.stream.flush(); self.commands.flush()
                print(f'Native {self.count * .05:.1f} s: {phase}', flush=True)
            return self.last
        except Exception as exc:
            self.runtime_error = type(exc).__name__ + ': ' + str(exc)
            raise

    def observe(self):
        if not self._ready or self._closed or not self.enable_cameras:
            raise RuntimeError('An active camera-enabled native world is required.')
        before = self.air.total
        images = {}
        for name in ('top_cam', 'left_cam', 'right_cam'):
            rgb = self.base.scene[name].data.output['rgb']
            rgb = rgb.torch if hasattr(rgb, 'torch') else rgb
            rgb = rgb[0, ..., :3].detach().cpu().numpy()
            if rgb.shape != (self.height, self.width, 3) or rgb.dtype != np.uint8:
                raise RuntimeError('Unexpected RGB camera shape or dtype: ' + name)
            images[name] = np.ascontiguousarray(rgb)
        arms = []
        for side in ('left', 'right'):
            ids = self.adapter.robot_ids[side]
            q = self.native._array(self.base.scene[side + '_robot'].data.joint_pos)[0]
            arms.append(np.r_[q[ids['arm']], np.clip(q[ids['fingers']].mean() / self.constants.FINGER_OPEN, 0, 1)])
        if self.air.total != before:
            raise RuntimeError('Observation rendering advanced physics.')
        return images, np.r_[arms[0], arms[1]], self.count * .05

    def score(self):
        if self.scorer is None:
            return dict(score=0., success=False, evidence_complete=False, invalid_reasons=['No measured task scorer.'])
        result = copy.deepcopy(self.scorer.result())
        if self.runtime_error is not None:
            result.update(score=0., success=False, evidence_complete=False)
            result['invalid_reasons'] = [*result.get('invalid_reasons', []), 'runtime_error: ' + self.runtime_error]
        return result

    def run_scripted(self):
        if not self._ready or self._closed or self.count != 0:
            raise RuntimeError('A scripted demonstration requires a fresh native trial.')
        if self.stage == 'spawn':
            for _ in range(40):
                self.step_action(self.action, 'settle')
            return dict(result='SPAWN_AND_SETTLE_COMPLETED', stage='spawn', final=self.last,
                        scope='Environment and passive bodies only; no grasp result.')
        from native_controller import build_yam_controller
        self.controller = build_yam_controller(self.adapter, self.read_row, self.step_action, self.action,
            env_origin=self.native._array(self.base.scene.env_origins)[0], baseline_plan=self.plan)
        self.controller.metadata.update(self.metadata)
        if self.stage == 'tool':
            result = self.controller.run_tool_pickup()
        elif self.stage == 'wafer':
            result = self.controller.run_wafer_cycle()
        else:
            if self.evidence is None:
                raise RuntimeError('Full demonstration needs the actual-object evidence adapter.')
            result = self.controller.run_full(self.full_plan['gun'], self.metadata)
        self.metadata['controller_execution'] = self.controller.metadata
        result['result'] = result['status']
        return result

    def finish(self, result=None):
        return self.close(result)

    def close(self, result=None):
        if self._closed:
            return self.report
        self._closed = True
        result = dict(result) if result is not None else dict(result='COMPLETED', stage=self.stage)
        if self.runtime_error is not None:
            result.update(result='INCOMPLETE', error=self.runtime_error)
        if not self._ready and result.get('result') == 'COMPLETED':
            result.update(result='INCOMPLETE', error='The native scene never completed initialization.')
        result.update(wall_seconds=time.monotonic() - self.started,
                      simulation_seconds=self.count * .05, control_actions=self.count)
        if self.controller is not None:
            self.metadata['controller_execution'] = self.controller.metadata
        try:
            unchanged = (all(hashlib.sha256((self.baseline / n).read_bytes()).hexdigest() == digest
                             for n, digest in self.baseline_hashes.items()) and
                         all(hashlib.sha256((self.root / n).read_bytes()).hexdigest() == digest
                             for n, digest in self.metadata.get('source_files', {}).items()))
        except OSError:
            unchanged = False
        self.metadata['sources_unchanged_during_run'] = unchanged
        if not unchanged:
            result.update(result='INCOMPLETE', error='Runtime source changed during the run.')
        if self.air is not None:
            self.metadata['physics_updates'] = self.air.total
            try:
                self.air.close()
            except Exception as exc:
                result.update(result='INCOMPLETE', cleanup_error=type(exc).__name__ + ': ' + str(exc))
        if self.evidence is not None:
            try:
                self.evidence.close()
            except Exception as exc:
                result.update(result='INCOMPLETE', evidence_cleanup_error=type(exc).__name__ + ': ' + str(exc))
        if self.scorer is not None:
            result['task_score'] = self.score()
            if self.stage == 'full' and result.get('result') == 'COMPLETED':
                result['result'] = ('FULL_DEMONSTRATION_SUCCEEDED'
                    if result['task_score'].get('success') and result['task_score'].get('score') == 1.
                    else 'FAILED')
        if self._output_created:
            dump(self.output / 'metadata.json', self.metadata)
            dump(self.output / 'trial_report.json', result)
        for resource in (self.stream, self.commands, self.env):
            if resource is not None:
                try:
                    resource.close()
                except Exception as exc:
                    result.update(result='INCOMPLETE', resource_cleanup_error=type(exc).__name__ + ': ' + str(exc))
        if self._output_created and result.get('resource_cleanup_error'):
            dump(self.output / 'trial_report.json', result)
        self.report = result
        if not self.defer_application_close:
            self.shutdown_application()
        return result

    def shutdown_application(self):
        # Kit may terminate the process; owned workers send the closed evidence report first.
        app, self.app = self.app, None
        if app is None:
            return
        try:
            app.close()
        except BaseException as exc:
            if self.report is not None:
                self.report.update(result='INCOMPLETE', application_cleanup_error=type(exc).__name__ + ': ' + str(exc))
                if self._output_created:
                    dump(self.output / 'trial_report.json', self.report)
            raise


def run(args):
    world = NativeEdgeWorld(Path(__file__).resolve().parent, args.baseline, args.output,
                            stage=args.stage, device=args.device, seed=args.seed)
    result = dict(result='INCOMPLETE', stage=args.stage)
    try:
        world.start()
        result = world.run_scripted()
    except Exception as exc:
        result.update(result='INCOMPLETE', error=type(exc).__name__ + ': ' + str(exc), traceback=traceback.format_exc())
        print(result['traceback'], flush=True)
    finally:
        result = world.close(result)
    print(json.dumps({k: result.get(k) for k in ('result', 'stage', 'error', 'wall_seconds')}), flush=True)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--stage', choices=['spawn', 'tool', 'wafer', 'full'], default='spawn')
    p.add_argument('--device', choices=['cpu', 'cuda:0'], default='cpu')
    p.add_argument('--seed', type=int, default=17)
    raise SystemExit(0 if run(p.parse_args()).get('result') != 'INCOMPLETE' else 1)
