"""Reconnect a linear-pull front brake on a stationary 12-inch wheel bench."""
import csv
import math
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import labkit
from yam_arena.constants import FINGER_OPEN
from yam_arena.demo import Waypoint
from yam_arena.task_file import TABLE_TOP
from cable import wire_guide, BowdenCable, CableVisual

labkit.CONTACT_REPORT.update({"wheel","guide","lever","cradle_arm","anchor_arm"})

SCENE = "station"
TOP = TABLE_TOP[SCENE]
CENTER = np.array((.112, .050, TOP + .151))
PIVOT_X = .079
PIVOT_Y = .036
SHOE_X = .104
TIP_X = .18
SQUEEZE_X = .169
SILVER = (.62, .65, .67)
DARK = (.15, .17, .19)
RUBBER = (.08, .085, .095)
RED = (.72, .13, .10)


def box(pos, sides, color, angle=0.0):
    return (tuple(pos), tuple(sides), color, angle)


def bar(a, b, width, depth, color):
    a, b = np.array(a), np.array(b)
    dx, dy, dz = b - a
    return box((a + b) / 2, (np.linalg.norm(b - a), width, depth), color,
               (math.atan2(dy, dx), -math.atan2(dz, math.hypot(dx, dy))))


def world(local):
    return tuple(CENTER + np.asarray(local))


def wheel_ring(radius, radial, width, color):
    pieces = []
    for i in range(32):
        a = i * math.tau / 32
        pieces.append(box((radius*math.cos(a), 0, radius*math.sin(a)),
                          (radial, width, 2*radius*math.tan(math.pi/32)+.0003), color, (0, -a)))
    return pieces


wheel_shapes = wheel_ring(.107, .013, .020, SILVER) + wheel_ring(.131, .026, .030, RUBBER)
wheel_shapes.extend(bar((0,0,0),(.102*math.cos(a),0,.102*math.sin(a)),.002,.002,SILVER)
                    for a in np.arange(12)*math.tau/12)
wheel_shapes.append(box((.077,.002,0),(.023,.002,.008),RED))
wheel = labkit.Joint("wheel", "hinge", "Y", (0,0,0), boxes=tuple(wheel_shapes),
                     cylinders=(((0,0,0),.012,.052,SILVER,"Y"),), lower=-10000, upper=10000,
                     damping=.004, mass=.52)


def make_arm(sign):
    pivot = (PIVOT_X, sign*PIVOT_Y, 0)
    tip_y = sign*.041
    arm_end = TIP_X if sign > 0 else TIP_X-.011
    shapes = [bar(pivot, (arm_end,tip_y,0),.008,.011,DARK),
              box((SHOE_X,sign*.0203,0),(.008,.015,.036),RUBBER),
              box((SHOE_X,sign*.031,0),(.013,.016,.011),SILVER)]
    if sign < 0:
        # A cable-sized open slot above a shoulder that retains the larger ferrule.
        shapes.extend([box((TIP_X+v,tip_y,0),(.004,.007,.014),SILVER) for v in (-.0052,.0052)])
        shapes.append(box((TIP_X-.004,tip_y,-.006),(.022,.007,.004),SILVER))
        for edge in (-1,1):
            shapes.append(box((TIP_X+edge*.0039,tip_y,.0048),(.0058,.007,.003),SILVER))
    else:
        shapes.append(box((TIP_X,tip_y,0),(.012,.010,.012),SILVER))
    return labkit.Joint("anchor_arm" if sign > 0 else "cradle_arm", "hinge", "Z", pivot,
                        boxes=tuple(shapes), lower=-.40 if sign>0 else -.025,
                        upper=.025 if sign>0 else .40, stiffness=.29, damping=.015, mass=.070)


fork_shapes = [box((.006,0,-.146),(.122,.156,.010),DARK)]
for sign in (-1,1):
    fork_shapes.extend([bar((.006,.052*sign,-.145),(0,.052*sign,0),.014,.014,DARK),
                        bar((0,.052*sign,0),(PIVOT_X,.052*sign,0),.013,.015,DARK)])
fork = labkit.part("fork", CENTER, fixed=True, boxes=fork_shapes,
                   joints=(wheel,make_arm(1),make_arm(-1)), self_collide=True, friction=.70)
LEVER_SHIFT = np.array((.070,-.090,0.))
LEVER_PIVOT = np.array((.165, -.150, .025))
LEVER_END = LEVER_PIVOT + np.array((-.054, -.020, 0.))
lever = labkit.Joint("lever","hinge","Z",tuple(LEVER_PIVOT),
                     boxes=(bar(LEVER_PIVOT,LEVER_END,.010,.012,DARK),),
                     lower=-.03, upper=.38, stiffness=.38, damping=.022, mass=.055)
bar_base = labkit.part("bar_stand", CENTER+LEVER_SHIFT, fixed=True, joints=(lever,), boxes=[
    box((.165,-.150,-.062),(.022,.022,.170),SILVER),
    box((.165,-.150,-.145),(.085,.070,.012),DARK),
    bar((.166,-.151,.025),(.111,-.202,.025),.021,.021,RUBBER)])
PARK = np.array((TIP_X+.028, -.042, .022))
GUIDE_GRASP = np.array((0.,-.052,.002))
WIRE_LENGTH = .065
STAGE = os.getenv("BRAKE_STAGE", "full")
if STAGE not in ("full","arms","seated-check","disconnected-check"):
    raise ValueError("Unknown brake diagnostic stage")
START_GUIDE = np.array((TIP_X,-.046,-.0003)) if STAGE=="seated-check" else PARK
path=[np.array((0,-.008,0))]
for turn in np.linspace(0.,math.pi/2,8):
    path.append(np.array((0,-.059-.020*math.sin(turn),.020*(1-math.cos(turn)))))
path.append(np.array((0,-.079,.037)))
guide = labkit.part("guide",world(START_GUIDE),mass=.012,friction=.75,
    boxes=[bar(start,end,.006,.006,SILVER) for start,end in zip(path,path[1:])],
    cylinders=[((0,-.003,0),.0055,.009,SILVER,"Y"),
               ((0,.0055,0),.0021,.008,SILVER,"Y"),
               ((0,.018,0),.00075,.018,SILVER,"Y")])
if not getattr(labkit._author,"vbrake",False):
    labkit._author=wire_guide(labkit._author,"fork","guide",tuple(PARK))
    labkit._author.vbrake=True
ASSEMBLY_ONLY = STAGE == "arms"
OBJECTS = [fork] if ASSEMBLY_ONLY else [fork,bar_base,guide]


class AngledGrip(float):
    turn=0.


def right_tilt():
    import yam_arena.demo as motion
    from yam_arena.urdf import axis_angle_matrix
    if not getattr(motion.grip_rotation,"vbrake",False):
        base=motion.grip_rotation
        def rotation(yaw=0.):
            r=base(float(yaw))
            if isinstance(yaw,AngledGrip):
                return (axis_angle_matrix(np.array((0.,0.,1.)),yaw.turn)
                        @ axis_angle_matrix(np.array((1.,0.,0.)),math.radians(35.)) @ r)
            return r
        rotation.vbrake=True
        motion.grip_rotation=rotation
    return AngledGrip(math.pi/2)


def wp(seconds, xyz, grip, label):
    return Waypoint(seconds, world(xyz), grip, label)


def lever_wp(seconds,xyz,grip,label):
    return wp(seconds,np.asarray(xyz)+LEVER_SHIFT,grip,label)


def arm_preflight():
    return {"left":[wp(2.5,(SQUEEZE_X,0,.040),FINGER_OPEN,"approach brake arms"),
                    wp(2.0,(SQUEEZE_X,0,.003),FINGER_OPEN,"straddle brake arms"),
                    wp(2.0,(SQUEEZE_X,0,.003),-.0345,"squeeze brake arms"),
                    wp(1.5,(SQUEEZE_X,0,.003),-.0345,"hold arms together"),
                    wp(1.0,(SQUEEZE_X,0,.003),FINGER_OPEN,"release arms"),
                    wp(1.5,(SQUEEZE_X,0,.040),FINGER_OPEN,"withdraw")]}


def cradle_world(angle):
    origin = np.array((PIVOT_X,-PIVOT_Y,0))
    v = np.array((TIP_X,-.041,0))-origin
    c, s = math.cos(angle), math.sin(angle)
    return CENTER + origin + (c*v[0]-s*v[1],s*v[0]+c*v[1],0)


class GuideWaypoint:
    def __init__(self, name, memo,grip):
        self.name, self.memo, self.grip = name, memo, grip
        self.cached = None

    def __array__(self, dtype=None, copy=None):
        if self.cached is None:
            from yam_arena.scene import object_states
            state = object_states(labkit.env(), ["guide","fork"])
            guide_state = state["guide"]
            point = guide_state.to_world(GUIDE_GRASP)
            cradle = cradle_world(state["fork"].joints["cradle_arm"])
            if self.name == "above":
                target = point + (0,-.021,.030)
            elif self.name == "pick":
                target = point
                self.memo["pickup"]=target.copy()
            elif self.name == "wait":
                target=self.memo["pickup"].copy()
            else:
                from yam_arena.runtime import tcp_world
                scene=labkit.env().unwrapped.scene
                tcp=tcp_world(scene["right_robot"])-labkit._np(scene.env_origins[0])
                stem=guide_state.to_world((0,.0057,0))
                phase=self.name.split("-")[0]
                axis=guide_state.rotation()[:,1]
                roll=math.atan2(-axis[0],axis[1])
                if phase in ("carry","line","insert","seat"):
                    delta=math.atan2(math.sin(state["fork"].joints["cradle_arm"]-roll),
                                     math.cos(state["fork"].joints["cradle_arm"]-roll))
                    self.memo["turn"]=float(np.clip(self.memo.get("turn",0.)+np.clip(delta,-.18,.18),-.75,.75))
                self.grip.turn=self.memo.get("turn",0.)
                clearance={"carry":(0,-.012,.017),"line":(0,-.012,.009),
                           "insert":(0,-.012,-.0005),"seat":(0,.0006,-.0005)}
                if phase=="away":
                    target=tcp+(.016,-.056,.034)
                elif phase=="hold":
                    target=self.memo["command"].copy()
                else:
                    error=cradle+clearance[phase]-stem
                    reference=self.memo.get("command",tcp)
                    target=reference+np.clip(error,-.015,.015)
                    self.memo["command"]=target.copy()
                    print("GUIDE_TARGET",self.name,"tcp",np.round(tcp-CENTER,3),"delta_mm",np.round(error*1000.,1),
                          "command",np.round(target-CENTER,3),"tube yaw",round(roll,3),"wrist adjust",round(self.grip.turn,3),flush=True)
            self.cached = np.asarray(target)
        return np.asarray(self.cached,dtype=dtype)


def demo():
    if ASSEMBLY_ONLY:
        return arm_preflight()
    if STAGE in ("seated-check","disconnected-check"):
        return {"right":[lever_wp(2.5,(.115,-.180,.057),FINGER_OPEN,"approach lever"),
                         lever_wp(1.4,(.115,-.180,.027),FINGER_OPEN,"straddle lever"),
                         lever_wp(1.4,(.115,-.180,.027),-.008,"pull lever"),
                         lever_wp(1.6,(.115,-.180,.027),-.008,"hold lever"),
                         lever_wp(1.2,(.115,-.180,.027),FINGER_OPEN,"release lever"),
                         lever_wp(2.,(.115,-.180,.066),FINGER_OPEN,"watch wheel")]
                         }
    x=SQUEEZE_X
    left=[wp(5.2,(x,.065,.035),FINGER_OPEN,"clear while the guide is grasped"),
          wp(2.5,(x,0,.040),FINGER_OPEN,"approach arms"),
          wp(2.,(x,0,.003),FINGER_OPEN,"straddle arms"),
          wp(2.,(x,0,.003),-.0345,"squeeze arms"),
          wp(13.1,(x,0,.003),-.0345,"hold during cable seating"),
          wp(1.2,(x,0,.003),FINGER_OPEN,"let the cable take tension"),
          wp(1.8,(x,.050,.065),FINGER_OPEN,"withdraw")]
    memo={}
    def guide_wp(sec,name,grip):
        angle=right_tilt()
        return Waypoint(sec,GuideWaypoint(name,memo,angle),grip,name,angle)
    right=[Waypoint(2.5,world((.212,-.160,.057)),FINGER_OPEN,"clear of brake arms",right_tilt()),
           guide_wp(1.5,"above",FINGER_OPEN),guide_wp(1.2,"pick",FINGER_OPEN),
           guide_wp(.9,"pick",-.002),guide_wp(5.9,"wait",-.002),guide_wp(1.3,"carry-1",-.002),
           guide_wp(.9,"carry-2",-.002),guide_wp(.9,"line-1",-.002),
           guide_wp(.9,"line-2",-.002),guide_wp(.8,"insert-1",-.002),
           guide_wp(.8,"insert-2",-.002),guide_wp(.8,"seat-1",-.002),
           guide_wp(.8,"seat-2",-.002),guide_wp(.8,"seat-3",-.002),
           guide_wp(5.1,"hold",-.002),guide_wp(.8,"hold",FINGER_OPEN),
           guide_wp(1.4,"away",FINGER_OPEN),
           lever_wp(2.,(.115,-.180,.057),FINGER_OPEN,"move to lever"),
           lever_wp(1.4,(.115,-.180,.027),FINGER_OPEN,"straddle grip and lever"),
           lever_wp(1.4,(.115,-.180,.027),-.008,"pull brake lever"),
           lever_wp(1.6,(.115,-.180,.027),-.008,"hold brake applied"),
           lever_wp(1.2,(.115,-.180,.027),FINGER_OPEN,"release brake lever"),
           lever_wp(1.5,(.115,-.180,.066),FINGER_OPEN,"clear lever"),
           lever_wp(2.,(.115,-.180,.066),FINGER_OPEN,"watch wheel turn")]
    return {"left":left,"right":right}


class Trial:
    def reset(self,*args):
        self.steps = 0
        self.contact = None
        self.peak = (0., 0.)
        self.squeeze = False
        self.last_squeeze = -1000
        self.handoff=self.started_disconnected=False
        self.installed = self.retained = self.spinning = self.braked = self.passed = False
        self.free_start_angle = None
        self.seated_steps = self.stop_steps = 0
        self.last_stroke = -1.
        self.grade = 0.
        self.hardware = None
        self.trace = None

    def tick(self, env):
        asset = env.unwrapped.scene["fork"]
        indices = [asset.joint_names.index(name + "_joint") for name in ("anchor_arm","cradle_arm","wheel")]
        a,b,w = labkit._np(asset.data.joint_pos)[0,indices]
        self.peak = max(self.peak[0], -a), max(self.peak[1], b)
        
        if self.contact is None:
            from isaaclab.sim.utils.stage import get_current_stage
            from isaaclab_physx.physics import PhysxManager
            from pxr import UsdPhysics
            paths = [str(p.GetPath()) for p in get_current_stage().Traverse()
                     if p.HasAPI(UsdPhysics.RigidBodyAPI) and "/LeftArm/" in str(p.GetPath())]
            tips = [next(p for p in paths if p.endswith("/" + n)) for n in ("tip_left","tip_right","gripper")]
            root = env.unwrapped.scene.env_prim_paths[0] + "/fork/"
            filters = [root+n for n in ("body","wheel","anchor_arm","cradle_arm")]
            view = PhysxManager.get_physics_sim_view().create_rigid_contact_view(tips, filter_patterns=[filters]*len(tips))
            self.contact = view
            print("BRAKE_SETUP",asset.body_names,labkit._np(asset.data.body_link_pos_w)[0]-CENTER,
                  tips,flush=True)
        left_forces=self.contact.get_contact_force_matrix(dt=env.unwrapped.physics_dt).numpy()
        opposed=np.linalg.norm(left_forces[0,2])>.05 and np.linalg.norm(left_forces[1,3])>.05
        if -a>.050 and b>.050 and opposed:
            self.squeeze=True
            self.last_squeeze=self.steps
        if not ASSEMBLY_ONLY:
            self.functional_test(env, asset, indices, a, b, w)
        if self.steps % 10 == 0:
            from yam_arena.runtime import tcp_world
            robot = env.unwrapped.scene["left_robot"]
            tcp = tcp_world(robot) - CENTER - labkit._np(env.unwrapped.scene.env_origins[0])
            gap = 2 * abs(labkit._np(robot.data.joint_pos)[0,robot.joint_names.index("joint7")])
            contact = self.contact.get_contact_force_matrix(dt=env.unwrapped.physics_dt).numpy()
            print("BRAKE_LOAD",self.steps,np.round(np.linalg.norm(contact,axis=-1),1).tolist(),flush=True)
            print("BRAKE_DIAG",self.steps,"TCP",np.round(tcp,3),"finger_gap_mm",round(1000*gap,1),"arms_rad",round(a,4),round(b,4),"wheel_rad",round(w,4),
                  "opposed_squeeze",self.squeeze,flush=True)
        self.steps += 1


    def functional_test(self,env,asset,indices,a,b,w):
        import torch
        from isaaclab.sim.utils.stage import get_current_stage
        from isaaclab_physx.physics import PhysxManager
        from pxr import UsdPhysics
        from yam_arena.runtime import tcp_world
        from yam_arena.scene import object_states
        scene=env.unwrapped.scene
        if self.hardware is None:
            root=scene.env_prim_paths[0]
            stage=get_current_stage()
            robot_bodies=[str(p.GetPath()) for p in stage.Traverse()
                          if p.HasAPI(UsdPhysics.RigidBodyAPI) and
                          any('/'+arm+'/' in str(p.GetPath()) for arm in ('LeftArm','RightArm'))]
            self.curves=CableVisual(scene)
            self.cable=BowdenCable.install(scene,(TIP_X-PIVOT_X,.041-PIVOT_Y,0.),WIRE_LENGTH)
            self.filters=[root+"/fork/anchor_arm",root+"/fork/cradle_arm"]+robot_bodies
            sim=PhysxManager.get_physics_sim_view()
            self.hardware=sim.create_rigid_contact_view([root+"/fork/wheel"],filter_patterns=[self.filters],max_contact_data_count=256)
            left_links=[p for p in robot_bodies if "/LeftArm/" in p]
            right_links=[p for p in robot_bodies if "/RightArm/" in p]
            self.right_links=right_links
            self.right_filter=left_links+[root+"/fork/"+n for n in ("body","wheel","anchor_arm","cradle_arm")]+[root+"/bar_stand/body",root+"/bar_stand/lever"]
            self.right_obstacles=sim.create_rigid_contact_view(right_links,filter_patterns=[self.right_filter]*len(right_links))
            self.hand_pair=sim.create_rigid_contact_view(left_links,filter_patterns=[right_links]*len(left_links))
            self.brake_hand_contact=sim.create_rigid_contact_view([root+"/fork/anchor_arm",root+"/fork/cradle_arm"],
                filter_patterns=[robot_bodies,robot_bodies])
            self.r_fingers=[1+next(i for i,p in enumerate(robot_bodies) if "/RightArm/" in p and p.endswith("/"+name))
                            for name in ("tip_left","tip_right")]
            self.guide_contact=sim.create_rigid_contact_view([root+"/guide/body"],
                filter_patterns=[[root+"/fork/cradle_arm"]+robot_bodies])
            self.lever_contact=sim.create_rigid_contact_view([root+"/bar_stand/lever"],
                filter_patterns=[[p for p in robot_bodies if "/RightArm/" in p]])
        origin=labkit._np(scene.env_origins[0])
        st=object_states(env,["guide","bar_stand"])
        tube=st["guide"]
        lever_q=st["bar_stand"].joints["lever"]
        torque=torch.tensor([[.016]],dtype=torch.float32,device=asset.device)
        asset.actuators.target_command.set_effort_index(value=torque,joint_ids=[indices[2]])
        speed=float(labkit._np(asset.data.joint_vel)[0,indices[2]])
        def loads(view):
            return np.linalg.norm(view.get_contact_force_matrix(dt=env.unwrapped.physics_dt).numpy()[0],axis=-1)
        rim=loads(self.hardware)
        pad_on_rim=[False,False]
        if np.all(rim[:2]>.02):
            contacts,points,_,_,counts,starts=self.hardware.get_contact_data(dt=env.unwrapped.physics_dt)
            force=contacts.numpy().reshape(-1)
            xyz=points.numpy().reshape(-1,3)-origin-CENTER
            for col,sign in enumerate((1.,-1.)):
                n=int(counts.numpy()[0,col]); start=int(starts.numpy()[0,col])
                patch=xyz[start:start+n]
                radius=np.hypot(patch[:,0],patch[:,2])
                radial=(radius>=.098)&(radius<=.115)
                side=np.abs(patch[:,1]-sign*.010)<.0025
                pad_on_rim[col]=bool(np.any(radial & side & (np.abs(force[start:start+n])>.002)))
        gload=loads(self.guide_contact)
        lever_load=loads(self.lever_contact)
        c=cradle_world(b)
        anchor=cradle_world(-a).copy()
        anchor[1]=2*CENTER[1]-anchor[1]
        self.curves.update(anchor,tube.to_world((0,.026,0)),tube.to_world((0,-.079,.037)),
                           CENTER+LEVER_SHIFT+LEVER_PIVOT,WIRE_LENGTH-.017-.047*np.clip(lever_q,0,.38))
        stem=tube.to_world((0,.006,0))-c
        x_axis=np.array((math.cos(b),math.sin(b),0.))
        cable_axis=np.array((-math.sin(b),math.cos(b),0.))
        alignment=float(cable_axis @ tube.rotation()[:,1])
        seated=(abs(x_axis@stem)<=.0022 and abs(cable_axis@stem)<=.0038 and
                -.0032<=stem[2]<=.0028 and alignment>=math.cos(math.radians(18)) and gload[0]>.003)
        touching_wheel=bool(np.any(rim[2:]>.10))
        hand_pair_load=float(np.linalg.norm(self.hand_pair.get_contact_force_matrix(
            dt=env.unwrapped.physics_dt).numpy(),axis=-1).max())
        assembly_load=float(np.linalg.norm(self.brake_hand_contact.get_contact_force_matrix(
            dt=env.unwrapped.physics_dt).numpy(),axis=-1).max())
        blocked=touching_wheel or assembly_load>.10
        clear=all(np.linalg.norm(tcp_world(scene[arm])-origin-c)>.068 for arm in ("left_robot","right_robot"))
        if self.steps==0:
            self.started_disconnected=np.linalg.norm(stem)>.018 and not seated
        gripping=bool(np.all(gload[self.r_fingers]>.04))
        if self.started_disconnected and self.last_squeeze==self.steps and gripping and gload[0]>.003 and np.linalg.norm(stem)<.014:
            self.handoff=True
        # The return springs draw the ferrule home while the mechanic lets the arms open.
        if np.linalg.norm(stem)>.016 or gload[0]<=.003:
            self.handoff=False
        if seated and self.handoff:
            self.installed=True
        if np.linalg.norm(stem)>.016:
            self.installed=False
        if not seated:
            self.seated_steps=0
            self.stop_steps=0
            self.free_start_angle=None
            self.retained=self.spinning=self.braked=self.passed=False
        elif clear and not blocked and not np.any(gload[1:]>.05):
            self.seated_steps+=1
            if self.installed and self.seated_steps>=12:
                self.retained=True
        else:
            self.seated_steps = self.stop_steps = 0
            self.free_start_angle = None
            self.retained = self.spinning = self.braked = self.passed = False
        if self.retained and clear and not blocked:
            if lever_q<.09 and speed>1.:
                if self.free_start_angle is None:
                    self.free_start_angle=w
                if w-self.free_start_angle>=1.0:
                    if self.braked:
                        self.passed=True
                    self.spinning=True
            if self.spinning and lever_q>.16 and np.max(lever_load)>.05 and all(pad_on_rim) and abs(speed)<.15:
                self.stop_steps+=1
                if self.stop_steps>=5:
                    self.braked=True
                    self.free_start_angle=None
            elif not self.braked:
                self.stop_steps=0
        self.grade=1. if self.passed else .75 if self.braked else .50 if self.retained else .25 if seated and self.installed else 0.
        if os.getenv("BRAKE_TRACE_CSV"):
            if self.trace is None:
                stream=open(os.environ["BRAKE_TRACE_CSV"],"w",newline="",buffering=1)
                self.trace=csv.writer(stream)
                self.trace.writerow(("step","axial_arm_rad","cradle_arm_rad","wheel_rad","wheel_rad_s","lever_rad",
                                     "wire_N","pad_anchor_N","pad_cradle_N","on_anchor_track","on_cradle_track","direct_wheel_N","guide_cradle_N",
                                     "guide_hand_N","brake_hand_N","arm_pair_N","lever_hand_N","stem_dx_mm","stem_dy_mm","stem_dz_mm",
                                     "simultaneous_squeeze","guide_two_fingers","handoff","installed","hands_clear","seated","retained","free","stopped","passed","score"))
            self.trace.writerow((self.steps,a,b,w,speed,lever_q,float(self.cable.load.item()) if self.cable.load is not None else 0.,
                                 *rim[:2],*pad_on_rim,float(np.max(rim[2:])),gload[0],float(np.max(gload[1:])),assembly_load,hand_pair_load,float(np.max(lever_load)),
                                 *tuple(stem*1000),self.last_squeeze==self.steps,gripping,self.handoff,self.installed,clear,seated,self.retained,self.spinning,self.braked,self.passed,self.grade))
        if self.steps%10==0:
            r=scene["right_robot"]
            rtcp=tcp_world(r)-origin
            gp=tube.to_world(GUIDE_GRASP)
            obs=np.linalg.norm(self.right_obstacles.get_contact_force_matrix(dt=env.unwrapped.physics_dt).numpy(),axis=-1)
            ii,jj=np.nonzero(obs>1.)
            if len(ii):
                print("BRAKE_OBSTACLES",self.steps,[(self.right_links[i].split("/")[-1],
                      self.right_filter[j].split("/")[-2:],round(float(obs[i,j]),1)) for i,j in zip(ii,jj)],flush=True)
            print("BRAKE_RIGHT",self.steps,"tcp",np.round(rtcp-CENTER,3),"grasp error mm",np.round((rtcp-gp)*1000,1),"seat-axis deg",round(math.degrees(math.acos(np.clip(alignment,-1,1))),1),"guide fingers N",round(float(np.max(gload[1:])),2),flush=True)
            print("BRAKE_FUNCTION",self.steps,"guide",np.round(tube.pos-CENTER,3),"stem-mm",np.round(stem*1000,1),
                  "contact",round(float(gload[0]),2),"lever",round(lever_q,3),"wheel-speed",round(speed,3),
                  "T",round(float(self.cable.load.item()),2) if self.cable.load is not None else 0.,"pad-N",np.round(rim[:2],2),"retained",self.retained,"free",self.spinning,"stopped",self.braked,
                  "score",self.grade,flush=True)


trial = Trial()
trial.reset()
labkit.on_reset(trial.reset)
labkit.on_step(trial.tick)


def success(objects):
    return STAGE=="full" and trial.passed


def score(objects):
    return trial.grade if STAGE=="full" else 0.
