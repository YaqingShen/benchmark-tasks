"""Wrapper check: replay the robot005 scripted sequence (run_slushie.py Driver, --exit-path front) as an inspect-robots
policy on the yam_arena embodiment loaded with dream1/task.py, scored by task_score. Also: --policy hold (no-op)."""
import argparse, json, os, pathlib, sys, traceback
import numpy as np
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
INSTR = ("Pick up the plastic cup, place it on the drip tray under the slushie tap, pull the tap handle to fill it about "
         "two-thirds, then put the filled cup on the serving stand.")

p = argparse.ArgumentParser(); p.add_argument('--output', type=pathlib.Path, required=True)
p.add_argument('--policy', choices=['replay', 'hold'], default='replay'); p.add_argument('--max-steps', type=int, default=2000)
a = p.parse_args(); a.output.mkdir(parents=True, exist_ok=False)
os.environ['SLUSHIE_SUMMARY'] = str(a.output/'slushie_state.json'); os.environ['SLUSHIE_TRACE'] = str(a.output/'slushie_trace.jsonl')
import slushie_runtime as srt  # installs hooks before yam_arena.scene is imported
from inspect_robots import eval as ir_eval
from inspect_robots.policy import PolicyBase, PolicyConfig, PolicyInfo
from inspect_robots.scene import Scene
from inspect_robots.task import Task
from inspect_robots.types import Action, ActionChunk
from inspect_robots_yam.packing import STATE_KEY
from yam_arena.inspect_embodiment import YamArenaEmbodiment
from yam_arena import station, ik
from cup_geometry import FILL_POSITION, GRASP_HEIGHT, outer_radius, cavity_volume_ml

OPEN_GAP = .0939; GRASP_GAP = float(2*outer_radius(GRASP_HEIGHT)-.002); LEVER_GRIP_GAP = .021
IK_ROT_WEIGHT, IK_POS_TOL, IK_ROT_TOL = .1, .004, .25
LEVER_CLOSED_DEG, LEVER_OPEN_DEG = srt.LEVER_CLOSED_DEG, srt.LEVER_OPEN_DEG
ARGS = dict(target_fill=float(os.environ.get('SLUSH_TARGET', '.66')), max_dispense_seconds=8., settle_seconds=1., carry_seconds=8., front_exit_x=.29, filled_slowdown=1.5)
RT = srt.RT


def orientation(z):
    z = np.asarray(z, float); z /= np.linalg.norm(z); x = np.cross(z, [0, 0, 1.]); x /= np.linalg.norm(x); return np.column_stack([x, np.cross(z, x), z])


class Driver:
    def __init__(self, q):
        self.q = np.asarray(q, float).copy(); self.action = self.q.copy()
        self.rot = {'left': orientation([.8660254, -.5, 0.]), 'right': np.array([[0., 0., 1.], [-1., 0., 0.], [0., -1., 0.]])}

    def joints(self): return self.q

    def _solve(self, side, pos, q, label, rot_weight=IK_ROT_WEIGHT):
        q, ep, er = ik.solve(pos, ik.matrix_to_quat_xyzw(self.rot[side]), q, station.BASES[side], iters=200, damping=.018, rot_weight=rot_weight, tol=3e-5)
        if ep > IK_POS_TOL or er > IK_ROT_TOL: raise RuntimeError(f'IK failure {label}: pos {ep:.4f} m rot {er:.3f} rad')
        return q

    def start_pose(self, seconds=3.):
        """READY -> run_slushie.py's initial arm poses (joint-space), since the embodiment resets to READY."""
        goals = {}
        for side, pos in [('left', [.23, .23, 1.03]), ('right', [.20, -.25, 1.15])]:
            o = 0 if side == 'left' else 7
            q, ep, er = ik.solve(pos, ik.matrix_to_quat_xyzw(self.rot[side]), self.action[o:o+6], station.BASES[side], iters=700, damping=.018)
            goals[o] = q
        a0 = self.action.copy(); n = round(seconds*20)
        for i in range(n):
            t = ik.smoothstep((i+1)/n)
            for o, q in goals.items(): self.action[o:o+6] = a0[o:o+6]*(1-t)+q*t
            self.action[6] = self.action[13] = a0[6]*(1-t)+1.*t
            yield self.action.copy()

    def move(self, side, p, gap=None, seconds=2., label='move'):
        o = 0 if side == 'left' else 7; q = self.action[o:o+6].copy()
        p0, _ = ik.tcp_pose(q, station.BASES[side])
        g0 = self.action[o+6]; g1 = g0 if gap is None else gap/OPEN_GAP; n = max(1, round(seconds*20))
        print('PHASE', label, flush=True)
        for i in range(n):
            t = ik.smoothstep((i+1)/n)
            q = self._solve(side, p0*(1-t)+np.asarray(p)*t, q, label)
            self.action[o:o+6] = q; self.action[o+6] = g0*(1-t)+g1*t
            yield self.action.copy()

    def hold(self, seconds, label):
        for _ in range(round(seconds*20)): yield self.action.copy()

    def hand(self, side='left', commanded=False):
        o = 0 if side == 'left' else 7; j = self.action if commanded else self.q
        p, _ = ik.tcp_pose(j[o:o+6], station.BASES[side]); return p

    def move_cup(self, target, seconds, label):
        offset = self.hand()-RT.cup_pose()[:3, 3]; lag = self.hand(commanded=True)-self.hand()
        yield from self.move('left', np.asarray(target)+offset+lag, None, seconds, label)

    def side_grasp(self, label):
        loaded = RT.slush.inside > 0.
        grasp_height = float(os.environ.get('SLUSH_GRASP_HEIGHT', '.060')) if loaded else GRASP_HEIGHT
        tz = self.rot['left'][:, 2]; grip = RT.cup_pose()[:3, 3]+[0., 0., grasp_height]
        yield from self.move('left', grip-.09*tz+[0, 0, .04], OPEN_GAP, 2., f'{label}: pre-grasp')
        yield from self.move('left', grip-.09*tz, OPEN_GAP, 1.5, f'{label}: lower beside cup')
        yield from self.move('left', grip, OPEN_GAP, 2., f'{label}: slide fingers around cup')
        grasp_gap = 2*outer_radius(grasp_height) - float(os.environ.get('SLUSH_SQUEEZE', '.016')) if loaded else GRASP_GAP
        yield from self.move('left', grip, grasp_gap, 1.2, f'{label}: close gripper'); yield from self.hold(.4, f'{label}: settle grasp')

    def side_release(self, label, park):
        tz = self.rot['left'][:, 2]; h = self.hand(commanded=True)
        yield from self.move('left', h, OPEN_GAP, 1., f'{label}: open gripper')
        yield from self.move('left', h-.09*tz, None, 1.5, f'{label}: back away')
        yield from self.move('left', park, None, 2., f'{label}: park left arm')

    def lever(self, target_deg, seconds, label):
        initial = np.radians(RT.lever_deg()); n = round(seconds*20)
        for i in range(n):
            theta = initial+(np.radians(target_deg)-initial)*ik.smoothstep((i+1)/n)
            self.action[7:13] = self._solve('right', RT.lever_point(theta), self.action[7:13], label)
            self.action[13] = LEVER_GRIP_GAP/OPEN_GAP; yield self.action.copy()


def demonstrate(d):
    A = ARGS; target_ml = A['target_fill']*cavity_volume_ml(); park_left = np.array([.22, .26, 1.03])
    yield from d.start_pose(); yield from d.hold(1., 'settle initial state')
    stand = RT.cup_pose()[:3, 3].copy()
    yield from d.side_grasp('pick empty cup')
    yield from d.move_cup(stand+[0, 0, .012], 1.5, 'lift cup off stand')
    side = np.array([FILL_POSITION[0], .105, stand[2]+.012])
    yield from d.move_cup(side, 4., 'carry cup beside drip tray')
    yield from d.move_cup([side[0], side[1], FILL_POSITION[2]+.006], 1.5, 'lower to tray height')
    yield from d.move_cup(FILL_POSITION+[0, 0, .006], 2.5, 'slide cup under tap')
    yield from d.move_cup(FILL_POSITION+[0, 0, -.002], 1., 'set cup on drip tray')
    yield from d.side_release('leave cup on tray', park_left); yield from d.hold(.5, 'cup resting on tray')
    theta = np.radians(RT.lever_deg()); pt = RT.lever_point(theta); r = d.rot['right']
    yield from d.move('right', pt-r[:, 2]*.10, OPEN_GAP, 2.5, 'approach tap handle')
    yield from d.move('right', pt, OPEN_GAP, 2., 'fingers around tap handle'); yield from d.move('right', pt, LEVER_GRIP_GAP, 1., 'grasp tap handle')
    yield from d.lever(LEVER_OPEN_DEG, 2., 'pull tap handle down'); n = 0
    while RT.metrics.get('in_cup_ml', 0.)+RT.metrics.get('airborne_ml', 0.) < target_ml and n < A['max_dispense_seconds']*20:
        yield from d.lever(LEVER_OPEN_DEG, .05, 'hold handle down'); n += 1
    h = d.hand('right', commanded=True)
    print('RELEASE_START', RT.steps, RT.lever_deg(), flush=True)
    yield from d.move('right', h, OPEN_GAP, .45, 'open fingers at the full-open tap')
    yield from d.hold(1., 'spring returns tap without assistance')
    print('RELEASE_END', RT.steps, RT.lever_deg(), flush=True)
    yield from d.move('right', pt-r[:, 2]*.07, OPEN_GAP, 1.4, 'withdraw to reachable pre-grasp waypoint')
    yield from d.move('right', np.array([.20, -.25, 1.15]), None, 2., 'park right arm')
    yield from d.hold(A['settle_seconds'], 'slushie settles')
    yield from d.side_grasp('pick filled cup'); k = A['filled_slowdown']
    yield from d.move_cup(RT.cup_pose()[:3, 3]+[0, 0, .012], 1.2*k, 'lift cup off tray')
    for _ in range(3):
        if RT.cup_pose()[2, 3] >= FILL_POSITION[2] + .006:
            break
        yield from d.move_cup(FILL_POSITION+[0, 0, .012], .75, 'compensate for loaded grasp settling')
    clear = np.array([A['front_exit_x'], FILL_POSITION[1], FILL_POSITION[2]+.016])
    yield from d.move_cup(clear, 3.*k, 'pull cup straight out from under tap')
    yield from d.move_cup([clear[0], clear[1], stand[2]+.012], 1.5*k, 'raise to stand height')
    yield from d.move_cup(stand+[0, 0, .012], A['carry_seconds'], 'carry cup to serve stand')
    yield from d.move_cup(stand+[0, 0, -.001], 1.5*k, 'set cup on serve stand')
    yield from d.side_release('serve', np.array([.21, .26, 1.10]))
    print('PHASE scripted sequence done; holding', flush=True)


class ScriptPolicy(PolicyBase):
    config = PolicyConfig(action_horizon=1)
    def __init__(self, mode): self.mode = mode
    def bind(self, info): self.info = PolicyInfo(name=f'slushie_{self.mode}', action_space=info.action_space, observation_space=info.observation_space, control_hz=20.)
    def reset(self, scene): self.d = None; self.gen = None
    def act(self, obs):
        q = np.asarray(obs.state[STATE_KEY], float)
        if self.d is None:
            self.d = Driver(q); self.gen = demonstrate(self.d) if self.mode == 'replay' else iter(())
        self.d.q = q
        try: act = next(self.gen)
        except StopIteration: act = self.d.action.copy()
        return ActionChunk(actions=(Action(np.asarray(act, np.float32)),), control_hz=20.)


result = {'policy': a.policy}
try:
    emb = YamArenaEmbodiment(task=str(HERE/'task.py'), object_jitter=0., control_hz=20., cam_width=640, cam_height=352, device='cpu')
    task = Task(name=f'slushie-{a.policy}', scenes=[Scene(id='slushie', instruction=INSTR)], scorer='task_score', max_steps=a.max_steps)
    logs = ir_eval(task, ScriptPolicy(a.policy), emb, log_dir=str(a.output/'eval'), seed=0, store_frames=True, grader=None, fail_on_error=True)
    result['statuses'] = [getattr(l, 'status', None) for l in logs]
    try:
        r = logs[0].results; result['results'] = json.loads(json.dumps(r, default=lambda o: getattr(o, '__dict__', str(o))))
    except Exception as e: result['results_error'] = repr(e)
except BaseException:
    result['error'] = traceback.format_exc(); traceback.print_exc()
result['final_slushie_state'] = RT.metrics
(a.output/'replay_result.json').write_text(json.dumps(result, indent=2, default=str)); print('RESULT', json.dumps({k: result.get(k) for k in ['statuses', 'error']}, default=str)[:2000], flush=True)
print('FINAL', json.dumps({k: RT.metrics.get(k) for k in ['score', 'passed', 'checks', 'in_cup_ml', 'spill_fraction', 'fill_fraction']}), flush=True)
if RT.trace is not None:
    RT.trace.close()
os._exit(1 if 'error' in result or result.get('statuses') != ['success'] else 0)
