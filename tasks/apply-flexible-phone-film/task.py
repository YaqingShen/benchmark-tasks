"""Apply a flexible phone screen protector with a peel tab and a squeegee."""
import math
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent))
import labkit
from visual import FilmMesh
from material import Laminate, LENGTH, WIDTH, TAB_X, TAB_Y, X, Y, ROWS, COLS
from yam_arena.demo import Waypoint
from yam_arena.task_file import TABLE_TOP

SCENE = "station"
TOP = TABLE_TOP[SCENE]
X0, SCREEN_Z = .195, TOP+.012
TAB_START = (X0+TAB_X-.022,TAB_Y+.012)
CARD_X, CARD_Y = .246,-.145
BLACK, BLUE, WHITE, AMBER = (.07,.09,.12),(.12,.46,.76),(.90,.93,.93),(.87,.53,.13)
labkit.CONTACT_REPORT.update(("card","liner_tab"))
labkit.CONTACT_OFFSET = .0002
phone = labkit.part("phone", (X0+LENGTH/2,0,TOP), fixed=True,
    boxes=[((0,0,.006),(.150,.072,.012),BLACK)])
tab = labkit.part("liner_tab",(*TAB_START,SCREEN_Z+.0015),mass=.004,friction=.8,
    boxes=[((0,0,.001),(.020,.014,.002),BLUE),((0,0,.013),(.021,.002,.024),BLUE)])
card = labkit.part("card",(CARD_X,CARD_Y,TOP+.001),mass=.035,friction=.65,
    boxes=[((0,0,.025),(.004,.106,.046),WHITE),((0,0,.002),(.007,.106,.004),AMBER)])
rack = labkit.part("card_rest",(CARD_X,CARD_Y,TOP), fixed=True,
    boxes=[((x, .024,.009),(.005,.020,.018),BLACK) for x in (-.0065,.0065)])
OBJECTS = [phone,card,tab,rack]


def wp(t,x,y,z,g,label,yaw=0.):
    return Waypoint(t,(x,y,TOP+z),g,label,yaw)


def demo():
    from yam_arena import demo as motion
    from yam_arena.urdf import axis_angle_matrix
    if not getattr(motion.grip_rotation,"film_tilt",False):
        original=motion.grip_rotation
        def grip_rotation(yaw=0.):
            if isinstance(yaw,tuple):
                azimuth,lean=yaw
                return axis_angle_matrix(np.array([1.,0.,0.]),lean) @ original(azimuth)
            return original(yaw)
        grip_rotation.film_tilt=True
        motion.grip_rotation=grip_rotation
    # The hand pinches the paper corner; the other hand grasps the right end of the card.
    l, r = [], []
    def pair(t,lp,rp,stage):
        l.append(wp(t,*lp,stage,yaw=(0.,-.65) if stage not in ("approach tools","enter grasps","pinch") else 0.))
        r.append(wp(t,*rp,stage,yaw=math.pi/2))
    ox,oy = TAB_START
    open_l,open_r=-.019,-.017
    left0=(ox,oy,.025,open_l)
    right0=(CARD_X,CARD_Y-.043,.035,open_r)
    pair(1.8,(ox,oy,.116,open_l),(CARD_X,CARD_Y-.043,.115,open_r),"approach tools")
    pair(1.8,left0,right0,"enter grasps")
    pair(1.,(ox,oy,.025,0),(CARD_X,CARD_Y-.043,.035,0),"pinch")
    pair(1.3,(ox,oy,.069,0),(CARD_X,CARD_Y-.043,.106,0),"raise")
    hold=(X0+TAB_X,TAB_Y+.009,.037,0)
    center=(X0+.001,-.043,.085,0)
    pair(1.7,(X0+TAB_X,TAB_Y+.009,.068,0),center,"align")
    pair(1.0,hold,center,"place liner")
    # The reference paper lies on the table before alignment, below the phone top.
    pair(2.0,hold,(X0+.001,-.043,.042,0),"press start")
    for i in range(1,19):
        p=i*LENGTH/18
        tx=X0+2*p+TAB_X
        left=(tx,TAB_Y+.009,.049,0)
        right=(X0+p+min(.014,.5*p)-.002,-.043,.042,0)
        pair(.42,left,right,"peel and press")
    pair(.9,left,(X0+LENGTH+.013,-.043,.042,0),"seal end")
    pair(1.2,(tx,TAB_Y,.100,0),(X0+LENGTH+.013,-.043,.100,0),"withdraw")
    return {"left":l,"right":r}


class Trial:
    def __init__(self):
        self.view = None
        self.visual = None
        self.reset()

    def reset(self,*args):
        self.material = Laminate()
        self.steps = 0
        self.credit = 0.
        self.released = 0
        self.grip_recent = {"paper":0,"card":0}

    def create_contacts(self,env):
        from isaaclab_physx.physics import PhysxManager
        prefix=env.unwrapped.scene.env_prim_paths[0]+"/"
        from isaaclab.sim.utils.stage import get_current_stage
        from pxr import Usd, UsdPhysics
        stage=get_current_stage()
        physics=PhysxManager.get_physics_sim_view()
        def tips(name):
            root=env.unwrapped.scene[name].root_physx_view.prim_paths[0]
            found={p.GetName():str(p.GetPath()) for p in Usd.PrimRange(stage.GetPrimAtPath(root))
                   if p.GetName() in ("tip_left","tip_right") and p.HasAPI(UsdPhysics.RigidBodyAPI)}
            return [found["tip_left"],found["tip_right"]]
        card=physics.create_rigid_contact_view([prefix+"card/body"],
                filter_patterns=[[prefix+"phone/body",*tips("right_robot")]],max_contact_data_count=128)
        paper=physics.create_rigid_contact_view([prefix+"liner_tab/body"],
                filter_patterns=[tips("left_robot")],max_contact_data_count=64)
        return card,paper

    def tick(self,env):
        from yam_arena.scene import object_states
        from yam_arena.runtime import tcp_world
        scene=env.unwrapped.scene
        origin=labkit._np(scene.env_origins[0])
        s=object_states(env,["card","liner_tab"])
        blade,paper=s["card"],s["liner_tab"]
        ends=np.array([[blade.to_world((x,y,0)) for x in (-.0035,.0035)] for y in (-.053,.053)])
        blade_edge=ends[np.arange(2),ends[:,:,2].argmin(axis=1)]
        base=np.array([X0,0,SCREEN_Z])
        t=paper.to_world((0,0,.007))-base
        left=scene["left_robot"]
        tcp=tcp_world(left)-origin
        left_gap=-float(labkit._np(left.data.joint_pos)[0,-2:].sum())
        pinched=bool(left_gap < .009 and np.linalg.norm(tcp-paper.to_world((0,0,.019)))<.027)
        if self.view is None:
            self.view=self.create_contacts(env)
        cv,pv=self.view
        def grip(view,last,kind):
            matrix=view.get_contact_force_matrix(dt=env.unwrapped.physics_dt).numpy().reshape((-1,3))[last:]
            active=bool(np.all(np.linalg.norm(matrix,axis=1)>.02))
            self.grip_recent[kind]=4 if active else max(0,self.grip_recent[kind]-1)
            return self.grip_recent[kind]>0
        pinched=grip(pv,0,"paper") and pinched
        card_gripped=grip(cv,1,"card")
        f,pts,norm,_,cnt,start=cv.get_contact_data(dt=env.unwrapped.physics_dt)
        n=int(cnt.numpy().ravel()[0]); a=int(start.numpy().ravel()[0])
        fs=np.abs(f.numpy().ravel()[a:a+n]); points=pts.numpy().reshape(-1,3)[a:a+n]-origin
        normal=np.abs(norm.numpy().reshape(-1,3)[a:a+n,2])
        on=(np.abs(points[:,2]-SCREEN_Z)<.002)&(normal>.65)
        force=float(fs[on].sum())
        angle=math.atan2(paper.rotation()[1,0],paper.rotation()[0,0])
        self.material.observe(t,angle,pinched,blade_edge-base,card_gripped and force>.06 and np.min(np.abs(blade_edge[:,2]-SCREEN_Z))<.005)
        m=self.material
        clear=min(blade_edge[:,2])-SCREEN_Z>.015
        self.released = self.released+1 if clear else 0
        self.credit = (.15*float(m.peeled/LENGTH)+.45*m.coverage+.30*m.progressive.mean()) if m.started else 0.
        if self.visual is None:
            from isaaclab.sim.utils.stage import get_current_stage
            self.visual = FilmMesh(get_current_stage(),scene.env_prim_paths[0]+"/FilmSurface")
        self.visual.update(m,base+origin,paper.to_world((0,0,.007))+origin)
        if self.released>=10 and m.coverage==1 and m.progressive.all() and m.peeled>=LENGTH-.0005 and m.aligned():
            self.credit=1.
        if self.steps%10==0:
            print("FILM",self.steps,"tab",np.round(t*1000,1),"pinch",pinched,"yaw",round(angle,2),
                  "card",np.round((blade_edge.mean(axis=0)-base)*1000,1),"force",round(force,2),"card_grip",card_gripped,"ends_z",np.round((blade_edge[:,2]-SCREEN_Z)*1000,1),
                  "origin",np.round(m.origin*1000,1),"anchor",m.held_edge,"peel",round(1000*m.peeled,1),
                  "coverage",round(m.coverage,3),"score",round(self.credit,3),flush=True)
        self.steps+=1


trial=Trial()
labkit.on_reset(trial.reset)
labkit.on_step(trial.tick)
def success(objects): return trial.credit == 1.
def score(objects): return trial.credit
