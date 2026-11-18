"""Install the rigid-body scene, metered dispenser, free-surface rendering and task scorer."""
from __future__ import annotations
import importlib.abc, importlib.util, json, os, sys, pathlib
from dataclasses import dataclass
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
SRC = HERE
for p in (str(SRC), str(HERE)):
    if p not in sys.path: sys.path.insert(0, p)
from yam_arena.task_file import Box, ObjectState  # pure python
from cup_geometry import (CUP_START, FILL_POSITION, STAND_CENTER, LEVER_PIVOT, LEVER_LENGTH, cavity_volume_ml)

PHYSICS_DT = .001
LEVER_CLOSED_DEG, LEVER_OPEN_DEG = -15., -70.
TRACE = os.environ.get('SLUSHIE_TRACE')  # optional jsonl path


@dataclass(frozen=True)
class UsdObject(Box):
    """A bundled USD asset placed as authored (kind: rigid | articulation | base)."""
    usd: str = ''
    kind: str = 'base'


def opening_from_angle(angle_deg):
    return max(0., min(1., (-float(angle_deg)-20.)/50.))


def _arr(v):
    v = v.torch if hasattr(v, 'torch') else v
    return v.detach().cpu().numpy() if hasattr(v, 'detach') else np.asarray(v)


class Runtime:
    def __init__(self):
        self.env = None; self.base = None; self.slush = None; self.surface = None
        self.ticks = 0; self.render_clock = 0.; self.steps = 0; self.metrics = {}; self.flags = {}
        self.auger = []; self.trace = None
        self.reset_milestones()

    def reset_milestones(self):
        self.flags = {'cup_lifted': False, 'cup_on_tray': False, 'tap_opened': False, 'fill_reached_ever': False,
                      'max_in_cup_ml': 0., 'max_lever_open_deg': 0.}

    # ---- world reads
    def cup_pose(self):
        from yam_arena.urdf import quat_xyzw_to_matrix
        p = _arr(self.base.scene['cup'].data.root_link_pose_w)[0]; t = np.eye(4); t[:3, 3] = p[:3]; t[:3, :3] = quat_xyzw_to_matrix(p[3:]); return t

    def lever_deg(self):
        return float(np.degrees(_arr(self.base.scene['slush_machine'].data.joint_pos)[0, 0]))

    def lever_point(self, theta):
        c, s_ = np.cos(theta), np.sin(theta); return LEVER_PIVOT+np.array([s_, 0, c])*LEVER_LENGTH

    # ---- material
    def new_material(self):
        from fill_model import Dispenser
        self.slush = Dispenser()

    def attach(self, emb):
        import torch, omni.usd
        from fill_visual import FillVisual
        self.env = emb._env; self.base = self.env.unwrapped
        self._machine_target(torch)
        self.auger = [p for p in omni.usd.get_context().get_stage().Traverse() if p.GetName() == 'Auger' and 'slush_machine' in str(p.GetPath())]
        self.surface = FillVisual(omni.usd.get_context().get_stage()); self.new_material()

    def _machine_target(self, torch):
        m = self.base.scene['slush_machine']
        try:
            m.actuators.target_command.set_position_index(value=torch.tensor([[0.]], device=self.base.device, dtype=torch.float32), joint_ids=[0])
        except AttributeError:
            pass  # run_slushie.py sets this; actuator-manager API absent -> spring keeps its default target

    def spin_auger(self):
        from pxr import UsdGeom
        for p in self.auger:
            ops = UsdGeom.Xformable(p).GetOrderedXformOps()
            if len(ops) > 1: ops[1].Set(float((self.steps*.05*180.) % 360.))

    # ---- scoring
    def update_metrics(self):
        self.steps += 1; self.spin_auger()
        cup = self.cup_pose(); lever = self.lever_deg()
        from hand_geometry import hand_bounds, separation
        self.slush.advance(cup, opening_from_angle(lever), float(self.base.step_dt))
        self.surface.draw(cup,self.slush)
        q = self.slush.metrics()
        centre=cup[:3,3]+cup[:3,2]*.055
        half=np.abs(cup[:3,:3])@np.array([.0465,.0465,.055])
        clearance=min(separation((centre-half,centre+half), hand) for hand in hand_bounds(self.env))
        motion=_arr(self.base.scene['cup'].data.root_link_vel_w)[0]
        released=clearance>=.005
        resting=np.linalg.norm(motion[:3])<.01 and np.linalg.norm(motion[3:])<.5
        q.update(hand_clearance_m=float(clearance),cup_resting=bool(resting))
        tilt = float(np.degrees(np.arccos(np.clip(cup[2, 2], -1, 1))))
        on_stand = bool(np.linalg.norm(cup[:2, 3]-STAND_CENTER[:2]) < .015 and abs(cup[2, 3]-STAND_CENTER[2]) < .004)
        on_tray = bool(np.linalg.norm(cup[:2, 3]-FILL_POSITION[:2]) < .015 and abs(cup[2, 3]-FILL_POSITION[2]) < .006)
        fill = q.get('fill_fraction', 0.); spill = q.get('spill_fraction', 0.)
        f = self.flags
        f['cup_lifted'] |= bool(cup[2, 3]-STAND_CENTER[2] > .008)
        f['cup_on_tray'] |= on_tray
        f['tap_opened'] |= bool(opening_from_angle(lever) > .3)
        f['max_in_cup_ml'] = max(f['max_in_cup_ml'], q.get('in_cup_ml', 0.))
        f['max_lever_open_deg'] = max(f['max_lever_open_deg'], -lever-(-LEVER_CLOSED_DEG))
        fill_ok = .60 <= fill <= .73
        f['fill_reached_ever'] |= bool(fill_ok)
        checks = {'cup_on_serve_stand': on_stand, 'cup_upright': tilt < 5., 'fill_reached': bool(fill_ok),
                  'spill_below_5pct': bool(spill < .05), 'released_and_resting': bool(released and resting),
                  'tap_closed': bool(abs(lever-LEVER_CLOSED_DEG) < 5)}
        milestones = .10*f['cup_lifted']+.15*f['cup_on_tray']+.10*f['tap_opened']+.25*min(1., f['max_in_cup_ml']/(.60*cavity_volume_ml()))
        final = .40*sum(checks.values())/6. if f['fill_reached_ever'] else 0.
        score = float(min(1., milestones+final))
        self.metrics = {'step': self.steps, 'cup_position': cup[:3, 3].tolist(), 'cup_tilt_deg': tilt, 'lever_deg': lever,
                        'opening': opening_from_angle(lever), 'checks': checks, 'passed': bool(all(checks.values()) and all(f[k] for k in ('cup_lifted','cup_on_tray','tap_opened'))),
                        'milestones': dict(f), 'score': score, 'hopper_empty': self.slush.emitted >= 900.,
                        **{k: v for k, v in q.items() if k != 'cup_tilt_deg'}}
        if TRACE:
            if self.trace is None: self.trace = open(TRACE, 'a')
            self.trace.write(json.dumps(self.metrics)+'\n')
            if self.steps % 20 == 0: self.trace.flush()
        if self.steps % 100 == 0:
            print('[slushie] step', self.steps, 'score', round(score, 3), 'cup', np.round(cup[:3, 3], 3).tolist(), 'tilt', round(tilt, 1),
                  'lever', round(lever, 1), 'in_cup', round(q.get('in_cup_ml', 0.), 1), 'spill', round(q.get('spill_ml', 0.), 1),
                  'checks', ''.join('1' if v else '0' for v in checks.values()), file=sys.stderr, flush=True)
        out = os.environ.get('SLUSHIE_SUMMARY')
        if out: pathlib.Path(out).write_text(json.dumps(self.metrics, indent=2))


RT = Runtime()


# ---- patches -----------------------------------------------------------------------------------------------------
def _patch_scene(scene):
    if getattr(scene, '_slushie', False): return
    scene.STATION_DOME_INTENSITY = 300.; scene.STATION_LAMP_INTENSITY = 1.8e4; scene.STATION_LAMP_RADIUS = .55
    base_object, base_states, base_cb, base_build = scene.task_object, scene.object_states, scene.env_cfg_callback, scene.build_task_environment

    def task_object(spec):
        if not isinstance(spec, UsdObject): return base_object(spec)
        from isaaclab.actuators import ImplicitActuatorCfg
        from isaaclab_arena.assets.object import Object
        from isaaclab_arena.assets.object_type import ObjectType
        from isaaclab_arena.utils.pose import Pose
        typ = {'rigid': ObjectType.RIGID, 'articulation': ObjectType.ARTICULATION, 'base': ObjectType.BASE}[spec.kind]
        obj = Object(name=spec.name, usd_path=spec.usd, initial_pose=Pose(position_xyz=tuple(map(float, spec.pos)), rotation_xyzw=(0., 0., 0., 1.)), object_type=typ)
        if spec.name == 'slush_machine':
            obj.object_cfg.actuators = {'return_spring': ImplicitActuatorCfg(joint_names_expr=['draw'], stiffness=.16, damping=.02,
                                                                             joint_effort_limit=.3, joint_velocity_limit=10.)}
            obj.object_cfg.init_state.joint_pos = {'draw': float(np.radians(LEVER_CLOSED_DEG))}
            obj.object_cfg.init_state.joint_vel = {'.*': 0.}
        return obj

    def env_cfg_callback(control_hz=None):
        inner = base_cb(control_hz)
        def configure(cfg):
            cfg = inner(cfg); hz = control_hz or 20.
            cfg.sim.dt = PHYSICS_DT; cfg.decimation = max(1, round(1./(hz*PHYSICS_DT))); cfg.sim.render_interval = cfg.decimation
            cfg.episode_length_s = 3600.; cfg.sim.use_fabric = False
            return cfg
        return configure

    def build_task_environment(embodiment, task, control_hz=None):
        for side in ['left', 'right']:  # as run_slushie.py (grasp of a 12 g thin PET cup)
            cfg = getattr(embodiment.scene_config, side+'_robot')
            cfg.actuators['gripper'].joint_effort_limit = 3.0
            if cfg.spawn.collision_props is None:  # run_slushie.py: robot_contact_offset=.0001 (yam_arm_cfg)
                from yam_arena.robot import PhysxCollisionPropertiesCfg
                cfg.spawn.collision_props = PhysxCollisionPropertiesCfg(contact_offset=.0001, rest_offset=0.)
            cfg.spawn.collision_props.torsional_patch_radius = .004
            cfg.spawn.collision_props.min_torsional_patch_radius = .002
        return base_build(embodiment, task, control_hz)

    def object_states(environment, names):
        plain = [n for n in names if n not in ('slush_machine', 'cupstand', 'cup')]
        out = base_states(environment, plain) if plain else {}
        base = environment.unwrapped
        for n in names:
            if n == 'cup':
                d = base.scene['cup'].data; p = _arr(d.root_link_pose_w)[0]; v = _arr(d.root_link_vel_w)[0]
                out[n] = ObjectState(pos=p[:3], quat_xyzw=p[3:], lin_vel=v[:3], ang_vel=v[3:], low=p[:3]-.047, high=p[:3]+.11)
            elif n in ('slush_machine', 'cupstand'):
                z = np.zeros(3); out[n] = ObjectState(pos=z, quat_xyzw=np.array([0., 0., 0., 1.]), lin_vel=z, ang_vel=z, low=z, high=z)
        return {n: out[n] for n in names}

    scene.task_object, scene.object_states, scene.env_cfg_callback, scene.build_task_environment = task_object, object_states, env_cfg_callback, build_task_environment
    scene._slushie = True


def _patch_embodiment(module):
    cls = module.YamArenaEmbodiment
    if getattr(cls, '_slushie', False): return
    base_ensure, base_reset, base_step = cls._ensure_sim, cls.reset, cls._task_step

    def _ensure_sim(self, goal):
        before = self._env
        base_ensure(self, goal)
        if self._env is not before: RT.attach(self)

    def reset(self, *a, **k):
        obs = base_reset(self, *a, **k)
        import torch
        RT._machine_target(torch)
        RT.new_material()
        RT.metrics = {}
        RT.surface.draw(RT.cup_pose(), RT.slush)
        RT.reset_milestones(); RT.steps = 0
        return obs

    def _task_step(self, *a, **k):
        RT.update_metrics()
        return base_step(self, *a, **k)

    cls._ensure_sim, cls.reset, cls._task_step = _ensure_sim, reset, _task_step
    cls._slushie = True


_TARGETS = {'yam_arena.scene': _patch_scene, 'yam_arena.inspect_embodiment': _patch_embodiment}


class _Loader(importlib.abc.Loader):
    def __init__(self, inner, patch): self.inner, self.patch = inner, patch
    def create_module(self, spec): return self.inner.create_module(spec)
    def exec_module(self, module): self.inner.exec_module(module); self.patch(module)


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname not in _TARGETS: return None
        sys.meta_path.remove(self)
        try: spec = importlib.util.find_spec(fullname)
        finally: sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None: return None
        spec.loader = _Loader(spec.loader, _TARGETS[fullname]); return spec


def install():
    for name, patch in _TARGETS.items():
        if name in sys.modules: patch(sys.modules[name])
    if not any(isinstance(f, _Finder) for f in sys.meta_path): sys.meta_path.insert(0, _Finder())


install()
