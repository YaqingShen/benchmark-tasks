"""Transfer a loaded plate from an open steamer using a spring-open rim lifter."""
import math
import os
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import labkit
from yam_arena.demo import Waypoint
from yam_arena.task_file import TABLE_TOP

labkit.CONTACT_REPORT.add('plate')

SCENE = 'station'
TOP = TABLE_TOP[SCENE]
POT = (.355, .095)
MAT = (.196, -.058)
PARK = (.174, .217)
R = .078
RIM_Z = .060
RACK_Z = .028
MAT_Z = .012
PIVOT = .061
HANDLE = .092
METAL = (.66, .69, .72)
WHITE = (.94, .93, .87)
GRIP = (.12, .25, .26)
CORK = (.66, .44, .23)
SKIN = (.84, .66, .45)
BONE = (.94, .88, .73)


def world(xy, z):
    return (*xy, TOP + z)


def bar(a, b, width, depth, color):
    a, b = np.array(a), np.array(b)
    dx, dy, dz = b - a
    return (tuple((a+b)/2), (float(np.linalg.norm(b-a)), width, depth), color,
            (math.atan2(dy, dx), -math.atan2(dz, math.hypot(dx, dy))))


def ring(r, h, z, thickness, color, n=32):
    return [((r*math.cos(a), r*math.sin(a), z), (thickness, 2*r*math.tan(math.pi/n)+.0004, h), color, a)
            for a in (2*math.pi*i/n for i in range(n))]


pot = labkit.part('steamer', world(POT, 0), fixed=True,
    boxes=ring(.100, RIM_Z, RIM_Z/2, .003, METAL),
    cylinders=[((0,0,.002),.100,.004,METAL), ((0,0,RACK_Z/2),.058,RACK_Z,METAL)])
mat = labkit.part('trivet', world(MAT, 0), fixed=True, cylinders=[((0,0,MAT_Z/2),.055,MAT_Z,CORK)])
plate = labkit.part('plate', world(POT, RACK_Z+.001), mass=.215, friction=.55,
    cylinders=[((0,0,.003),.065,.006,WHITE), ((0,0,.008),R,.004,WHITE)],
    boxes=ring(.076,.003,.0115,.003,WHITE))


def half(side, xs):
    grip_y = -side*.024
    boxes = [((0,grip_y,HANDLE),(.041,.010,.023),GRIP),
             bar((0,grip_y,HANDLE-.008),(0,0,PIVOT),.008,.006,METAL),
             ((0,grip_y-side*.004,HANDLE+.015),(.041,.018,.006),GRIP)]
    for x in xs:
        edge = math.sqrt(R**2-x**2)
        y = side*(edge+.014)
        boxes.extend([bar((0,0,PIVOT),(x,y,.042),.005,.005,METAL),
                      bar((x,y,.042),(x,y,.004),.005,.005,METAL),
                      ((x, y-side*.0047, .002), (.009,.012,.004), METAL)])
    return boxes


moving = labkit.Joint('moving_jaw', kind='hinge', axis='X', anchor=(0,0,PIVOT),
    boxes=tuple(half(1,(-.026,.026))), lower=-.54, upper=.012, target=0,
    stiffness=1.8, damping=.014, mass=.040)
lifter = labkit.part('lifter',world(PARK,.001),mass=.040,friction=.92,
    boxes=half(-1,(0,)), joints=(moving,))
OBJECTS = [pot, mat, plate, lifter]
food_names = []
def tapered_mesh(rings, color):
    # Closed elliptical sections along the length of one drumstick.
    count = 16
    vertices = [(x, ry*math.cos(t), z+rz*math.sin(t)) for x,ry,rz,z in rings
                for t in (2*math.pi*j/count for j in range(count))]
    triangles = []
    for i in range(len(rings)-1):
        for j in range(count):
            a, b = i*count+j, i*count+(j+1)%count
            triangles.extend([(a,b,b+count),(a,b+count,a+count)])
    for j in range(1,count-1):
        end = (len(rings)-1)*count
        triangles.extend([(0,j+1,j),(end,end+j,end+j+1)])
    return labkit.Mesh(np.array(vertices), np.array(triangles), color, 'hull')


meat = tapered_mesh([(-.045,.004,.003,.000),(-.040,.014,.011,.001),
    (-.030,.018,.015,.002),(-.012,.017,.014,.002),(.004,.012,.010,.001),
    (.019,.007,.006,.000),(.028,.004,.004,.001)], SKIN)
bone = tapered_mesh([(.023,.0038,.0038,.001),(.034,.003,.003,.002),
    (.039,.0055,.0045,.003),(.043,.0052,.0042,.003),(.045,.002,.002,.003)], BONE)
for i,(x,y,angle,mass) in enumerate(((.003,-.037,.05,.055),(-.003,0,math.pi,.058),(.002,.037,-.07,.060))):
    name = 'drumstick_'+str(i+1)
    food_names.append(name)
    OBJECTS.append(labkit.part(name,world((POT[0]+x,POT[1]+y),RACK_Z+.025),
        meshes=(meat,bone), yaw=angle, mass=mass, friction=.45))


class OpenHandleTarget:
    """Reference waypoints for picking up the open tool by the ends of one handle."""

    def __init__(self, waypoint, measured):
        self.waypoint = waypoint
        self.measured = measured

    def __array__(self, dtype=None, copy=None):
        from yam_arena.scene import object_states
        env = labkit.env()
        wp, m = self.waypoint, self.measured
        if env is None:
            raise RuntimeError('The reference policy needs the current plate and tool poses.')
        states = object_states(env, ['plate', 'lifter'])
        tool = states['lifter']
        if wp.label == 'aim across handle':
            fixed = tool.to_world((0., .024, HANDLE))
            q = tool.joints['moving_jaw']
            pivot = np.array((0.,0.,PIVOT))
            moving = np.array((0.,-.024,HANDLE))-pivot
            cs, sn = math.cos(q), math.sin(q)
            moving = tool.to_world(pivot + (moving[0], cs*moving[1]-sn*moving[2], sn*moving[1]+cs*moving[2]))
            d = fixed-moving
            m['yaw'] = (math.atan2(-d[0],d[1]) + math.pi/2) % math.pi - math.pi
            m['handle'] = fixed + (0.,0.,-.009)
        if wp.label == 'center hooks around rim' or wp.label.startswith('ease hooks '):
            a,b,c = hook_centers(tool)[:,:2]
            center = np.linalg.solve(2*np.array([b-a,c-a]),np.array([b@b-a@a,c@c-a@a]))
            shift = np.clip(states['plate'].pos[:2]-center, -.023, .023)
            first = wp.label == 'center hooks around rim'
            base = m['handle'] if first else m['center']
            dz = .002 if first else .004
            m['center'] = base + np.r_[np.clip(shift,-.012,.012), dz]
        target = m.get('center', m['handle']).copy()
        if wp.label == 'aim across handle':
            target[2] += .065
        if wp.label == 'withdraw tool':
            target[2] += .069
        wp.pos, wp.yaw = tuple(target), m['yaw']
        return np.asarray(target,dtype=dtype)


def demo():
    opened, carry, closed = -.04695, -.035, -.020
    def wp(sec,xy,z,finger,label):
        return Waypoint(sec,world(xy,z),finger,label)
    on_mat=(MAT[0]+.007,MAT[1]-.018)
    over_pot=(POT[0],POT[1]-.003)
    path=[
        wp(2,PARK,.147,opened,'approach lifter'),
        wp(1.3,PARK,HANDLE-.013,opened,'enter handles'),
        wp(1,PARK,HANDLE-.013,carry,'pick up open lifter'),
        wp(1.5,PARK,.184,carry,'raise lifter'),
        wp(2.3,POT,.184,carry,'align hooks'),
        wp(2.2,over_pot,HANDLE+RACK_Z-.014,carry,'straddle rim'),
        wp(1.9,over_pot,HANDLE+RACK_Z-.005,-.023,'squeeze jaws'),
        wp(2.1,POT,.187,closed,'clear pot'),
        wp(3,MAT,.187,closed,'carry plate'),
        wp(2,on_mat,HANDLE+MAT_Z-.004,closed,'lower to trivet'),
        wp(1.35,on_mat,HANDLE+MAT_Z-.034,closed,'seat plate'),
        wp(.65,on_mat,HANDLE+MAT_Z-.034,closed,'settle plate'),
        wp(1.35,on_mat,.084,opened,'unload and open hooks'),
        wp(1.1,(MAT[0]+.005,MAT[1]-.015),.160,opened,'clear handles'),
    ]
    measured = {}
    for name,seconds,finger in [('aim across handle',1.45,opened),('return to open handles',.85,opened),
            ('hold open lifter',.8,-.016),('center hooks around rim',.8,-.016),
            *((f'ease hooks {i}',.45,-.016) for i in range(1,7)),('withdraw tool',1.15,-.016)]:
        waypoint=wp(seconds,MAT,.15,finger,name)
        waypoint.pos=OpenHandleTarget(waypoint,measured)
        path.append(waypoint)
    path.extend([wp(2.5,PARK,.185,-.016,'return tool'),wp(1.5,PARK,HANDLE-.013,-.016,'lower tool'),
                 wp(.8,PARK,HANDLE-.013,opened,'release tool'),wp(1,PARK,.155,opened,'clear tool')])
    return {'left':path}


def hook_centers(tool):
    q = tool.joints['moving_jaw']
    cq, sq = math.cos(q), math.sin(q)
    hinge = np.array((0., 0., PIVOT))
    rotation = np.array(((1.,0.,0.), (0.,cq,-sq), (0.,sq,cq)))
    tips = []
    for side, x in ((-1,0.), (1,-.026), (1,.026)):
        y = side*(math.sqrt(R*R-x*x)+.014-.0047)
        point = np.array((x,y,.002))
        if side == 1:
            point = hinge+rotation@(point-hinge)
        tips.append(tool.to_world(point))
    return np.array(tips)


def finger_vertices():
    from yam_arena.analysis import link_hulls
    hulls = link_hulls()
    return {name: np.unique(np.vstack([h.vertices for h in hulls[name]]),axis=0)
            for name in ('tip_left','tip_right')}


class Trial:
    def __init__(self):
        self.pads = None
        self.contact = None
        self.reset()

    def reset(self, *args):
        self.secured = self.cleared = self.transferred = self.failed = False
        self.seated = self.early_release = False
        self.hold = self.steps = self.unsupported = self.engaged = 0
        self.credit = 0.

    def touches_dish(self, environment, plate):
        from yam_arena.runtime import link_poses
        scene = environment.unwrapped.scene
        origin = labkit._np(scene.env_origins[0])
        if self.pads is None:
            self.pads = finger_vertices()
        rotation = plate.rotation()
        for arm in ('left_robot', 'right_robot'):
            poses = link_poses(scene[arm])
            for name, vertices in self.pads.items():
                pose = poses[name]
                offset = pose[:3,3]-origin-plate.pos
                if np.linalg.norm(offset) > .22:
                    continue
                local = (vertices@pose[:3,:3].T+offset)@rotation
                radial = np.linalg.norm(local[:,:2],axis=1)
                z = local[:,2]
                base = (radial <= .066) & (z >= -.001) & (z <= .007)
                upper = (radial <= R+.001) & (z >= .005) & (z <= .011)
                lip = (radial >= .0735) & (radial <= .0785) & (z >= .009) & (z <= .014)
                if np.any(base | upper | lip):
                    return True
        return False

    def contact_view(self, environment):
        from isaaclab_physx.physics import PhysxManager
        scene = environment.unwrapped.scene
        origin = labkit._np(scene.env_origins[0])
        prefix = scene.env_prim_paths[0] + '/'
        view = PhysxManager.get_physics_sim_view().create_rigid_contact_view(
            [prefix+'plate/body'], filter_patterns=[[prefix+'lifter/body', prefix+'lifter/moving_jaw', prefix+'trivet/body']],
            max_contact_data_count=128)
        if view.sensor_count != 1 or view.filter_count != 3:
            raise RuntimeError('The jaw and trivet contacts could not be measured.')
        return view, origin

    def plate_contacts(self, environment, plate):
        if self.contact is None:
            self.contact = self.contact_view(environment)
        view, origin = self.contact
        forces, points, normals, _, counts, starts = view.get_contact_data(dt=environment.unwrapped.physics_dt)
        counts, starts = counts.numpy().ravel(), starts.numpy().ravel()
        forces, points = forces.numpy().ravel(), points.numpy().reshape(-1, 3)
        sides = []
        for count, start in zip(counts[:2], starts[:2]):
            rows = slice(int(start), int(start+count))
            loaded = np.abs(forces[rows]) > 1e-6
            local = (points[rows][loaded] - origin - plate.pos) @ plate.rotation()
            radius = np.linalg.norm(local[:,:2], axis=1)
            rim = (radius >= .064) & (radius <= .080) & (local[:,2] >= -.002) & (local[:,2] <= .015)
            sides.append(local[rim,:2] / radius[rim,None])
        opposed = bool(len(sides[0]) and len(sides[1]) and np.any(sides[0] @ sides[1].T < -.45))
        touching = any(np.any(np.abs(forces[int(start):int(start+count)]) > 1e-6)
                       for count,start in zip(counts[:2],starts[:2]))
        start, count = int(starts[2]), int(counts[2])
        rows = slice(start, start+count)
        height = points[rows,2] - origin[2] - TOP
        on_top = np.abs(height-MAT_Z) <= .002
        vertical = np.abs(normals.numpy().reshape(-1,3)[rows,2]) > .7
        on_trivet = bool(np.any((np.abs(forces[rows]) > 1e-6) & on_top & vertical))
        return opposed, touching, on_trivet

    def tick(self, environment):
        from yam_arena.scene import object_states
        states = object_states(environment,['plate','lifter',*food_names])
        opposed, touching, on_trivet = self.plate_contacts(environment, states['plate'])
        self.observe(states, direct_contact=self.touches_dish(environment,states['plate']), rim_contact=opposed, jaw_contact=touching, on_trivet=on_trivet)

    def observe(self, s, direct_contact=False, *, rim_contact, jaw_contact, on_trivet):
        plate, tool = s['plate'], s['lifter']
        xy, z = plate.pos[:2], plate.pos[2]-TOP
        from_pot = np.linalg.norm(xy-POT)
        from_mat = np.linalg.norm(xy-MAT)
        tilt = plate.tilt_deg()
        food = [plate.to_local(s[name].pos) for name in food_names]
        aboard = all(np.linalg.norm(v[:2]) <= R-.010 and .005 <= v[2] <= .032 for v in food)
        hooks = (hook_centers(tool)-plate.pos)@plate.rotation()
        radii = np.linalg.norm(hooks[:,:2],axis=1)
        supported = rim_contact and tool.joints['moving_jaw'] < -.08
        self.engaged = self.engaged+1 if supported else 0
        if (self.steps > 8 and not aboard) or direct_contact:
            self.failed = True
        if from_pot < .030 and z > RACK_Z+.009 and aboard and self.engaged >= 3:
            self.secured = True
        if self.secured and supported and from_pot < .045 and z-R*math.sin(math.radians(tilt)) > RIM_Z+.004:
            self.cleared = True
        if self.cleared and supported and from_mat < .030 and z > MAT_Z+.014:
            self.transferred = True
        if self.transferred:
            flush_with_trivet = from_mat <= .018 and abs(z-MAT_Z) <= .001 and tilt <= 1.
            if not on_trivet and not flush_with_trivet:
                self.seated = False
            if not self.seated:
                if not jaw_contact:
                    self.early_release = True
                elif on_trivet and not self.early_release:
                    self.seated = True
        airborne = self.cleared and z > max(RACK_Z,MAT_Z)+.015
        self.unsupported = self.unsupported+1 if airborne and not supported else 0
        if (self.secured and tilt >= 90) or self.unsupported > 4 or (self.cleared and z < .006 and from_mat > .020):
            self.failed = True
        placed = from_mat <= .018 and abs(z-MAT_Z) <= .004 and tilt <= 4
        released = not jaw_contact and bool(np.all((radii > R+.020) | (hooks[:,2] > .033)))
        clear_hold = self.seated and on_trivet and placed and released and aboard and np.linalg.norm(plate.lin_vel)<.015
        self.hold = self.hold+1 if clear_hold else 0
        self.credit = 0. if self.failed else (1. if self.hold>=20 else .65 if self.transferred else .4 if self.cleared else .15 if self.secured else 0.)
        if os.environ.get('STEAM_TRACE') and (self.steps%10==0 or 385 <= self.steps <= 440):
            print('TRACE',self.steps,'tool',np.round(tool.pos-[0,0,TOP],3),'plate',np.round(np.r_[xy,z],3),
                  'jaw',round(tool.joints['moving_jaw'],3),'tilt',round(tilt,1),'food',aboard,
                  'opposing rim contacts',rim_contact,'direct',direct_contact,'jaw',jaw_contact,'trivet',on_trivet,'seated',self.seated,'early',self.early_release,'score',self.credit,flush=True)
        self.steps+=1

trial = Trial()
trial.reset()
labkit.on_reset(trial.reset)
labkit.on_step(trial.tick)
def success(objects): return trial.credit==1.
def score(objects): return trial.credit
