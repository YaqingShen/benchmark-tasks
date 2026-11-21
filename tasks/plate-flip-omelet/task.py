"""Turn a thick omelet over with a skillet and a dinner plate."""
import math
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).resolve().parent))
import labkit
from yam_arena.task_file import TABLE_TOP
from yam_arena.demo import Waypoint

SCENE = "station"
TOP = TABLE_TOP[SCENE]
PAN = np.array([.28,-.005])
FLIP_CENTER = np.array([.206,-.005])
FLIP_HEIGHT = .204
PARK = np.array([.46,-.135])
PICK_DIR = np.array([math.cos(math.radians(70)),math.sin(math.radians(70)),0.])
PAN_H = .050
PAD_H = .066
RIM_H = .031
PR = .122
RIM_GRIP = .111
FOOD_R = .065
FOOD_H = .021
METAL = (.25,.27,.28)
BLACK = (.12,.12,.13)
WHITE = (.88,.93,.93)
RIGHT = np.array([0.,-RIM_GRIP,.004])
LEFT = np.array([0.,.120,.031])

def at(xy,z):
    return np.r_[xy,TOP+z]

def disk(z,r,h,col):
    return ((0.,0.,z),r,h,col)

def ring(r,z,h,thick,color,n=36):
    return [((r*math.cos(a),r*math.sin(a),z),(thick,2*r*math.tan(math.pi/n)+.0002,h),color,a)
            for a in np.linspace(0,2*math.pi,n,endpoint=False)]

def rotate(axis, degrees):
    return Rotation.from_rotvec(np.asarray(axis)*math.radians(degrees)).as_matrix()

burner = labkit.part("hob",at(PAN,0), fixed=True,
    cylinders=(disk(PAN_H/2,.078,PAN_H,(.25,.28,.30)),))
wire=(.65,.67,.69)
rack_bars=[((x,0,PAD_H-.002),(.004,.118,.004),wire) for x in np.linspace(-.057,.057,7)]
rack_bars += [((0,y,PAD_H-.006),(.122,.004,.004),wire) for y in (-.055,.055)]
rack_bars += [((x,y,(PAD_H-.008)/2),(.005,.005,PAD_H-.008),wire)
              for x in (-.055,.055) for y in (-.055,.055)]
rest = labkit.part("cooling_rack",at(PARK,0),fixed=True,boxes=rack_bars)
labkit.SURFACE_FRICTION[("skillet",BLACK)]=(.85,.72)
pan = labkit.part("skillet",at(PAN,PAN_H+.0005),mass=.235,friction=.5,
    cylinders=(disk(.002,.087,.004,METAL),),
    boxes=ring(.086,.0165,.029,.003,METAL)+[
        ((0.,.103,.024),(.023,.041,.008),METAL),((0.,.132,.031),(.034,.060,.022),BLACK)])
plate = labkit.part("plate",at(PARK,PAD_H+.0005),mass=.16,friction=.46,
    cylinders=(disk(.003,PR,.006,WHITE),disk(.007,.072,.002,WHITE)),
    boxes=ring(PR-.007,.007,.002,.014,WHITE,48))
def cooked_marks():
    rng=np.random.default_rng(214)
    marks=[]
    for z,tones in ((FOOD_H+.00015,((.75,.48,.15),(.91,.72,.34))),
                    (-.00054,((.32,.15,.055),(.72,.43,.18)))):
        for col in tones:
            points=[]
            tris=[]
            for _ in range(16):
                a=rng.uniform(0,2*math.pi)
                radius=.053*math.sqrt(rng.uniform())
                x,y=radius*math.cos(a),radius*math.sin(a)
                rx,ry=rng.uniform(.002,.007,2)
                t0=rng.uniform(0,2*math.pi)
                i=len(points)
                points.append((x,y,z))
                for j in range(8):
                    b=j*math.pi/4
                    dx=rx*math.cos(b)*rng.uniform(.8,1.2)
                    dy=ry*math.sin(b)*rng.uniform(.8,1.2)
                    points.append((x+dx*math.cos(t0)-dy*math.sin(t0),y+dx*math.sin(t0)+dy*math.cos(t0),z))
                    tri=(i,i+1+j,i+1+(j+1)%8)
                    tris.append(tri if z>0 else tri[::-1])
            marks.append(labkit.Mesh(np.array(points),np.array(tris),col,"none"))
    return marks

food = labkit.part("omelet",at(PAN,PAN_H+.0048),mass=.12,friction=.32,meshes=cooked_marks(),
    cylinders=(disk(FOOD_H*.5,FOOD_R,FOOD_H,(.88,.66,.23)),
               disk(-.0002,FOOD_R-.0005,.0005,(.53,.28,.095))))
OBJECTS = [burner,rest,pan,plate,food]
labkit.CONTACT_REPORT.update(("omelet","plate","skillet"))
INSTRUCTION = "Use the plate to turn the whole omelet over. Slide it back into the skillet, browned side up, and set the skillet down."

def frame(direction):
    close=np.array([0.,0.,1.])
    return np.column_stack((close,np.cross(direction,close),direction))

def install_carry_control(dm):
    if getattr(dm.ArmPlan.target,"_omelet_feedback",False):
        return
    original=dm.ArmPlan.target
    def target(plan,t):
        pos,rot,g,label=original(plan,t)
        if not getattr(plan.waypoints[0].yaw,"omelet",False) or plan.base_pos[1]>0 or not trial.pose_cache:
            return pos,rot,g,label
        pan=trial.pose_cache["left"][0]
        plate=trial.pose_cache["right"][0]
        if not hasattr(plan,"rim_offset"):
            plan.rim_offset=np.zeros(3)
        if label in ("invert together","land omelet on plate") and pan.tilt_deg()>125:
            normal=pan.rotation()[:,2]
            center_error=pan.pos-plate.pos
            tangent=center_error-np.dot(center_error,normal)*normal
            delta=.052*np.clip(tangent,-.020,.020)
            plan.rim_offset=np.clip(plan.rim_offset+delta,-.023,.023)
        elif label in ("uncover","carry omelet aside"):
            plan.rim_offset*=.969
        else:
            plan.rim_offset*=.90
        if not hasattr(plan,"center_blend"):
            plan.center_blend=0.
            plan.held_center=None
        if label=="uncover":
            _,tcp,hand=trial.pose_cache["right"]
            plan.held_center=hand.T@(plate.pos-tcp)
        correction=np.zeros(3)
        if plan.held_center is not None:
            if label=="uncover":
                plan.center_blend=0.
            else:
                rate=-.04 if label=="clear" else .025
                plan.center_blend=np.clip(plan.center_blend+rate,0,1)
            ideal=frame(np.array([0.,1.,0.])).T@(-RIGHT-[0,0,-.002])
            correction=plan.center_blend*(rot@(ideal-plan.held_center))
        if not hasattr(plan,"landing_offset"):
            plan.landing_offset=np.zeros(3)
        if label=="align omelet over skillet":
            error=np.array([pan.pos[0]+.105-plate.pos[0],pan.pos[1]-trial.food_pose.pos[1],0.])
            plan.landing_offset=np.clip(plan.landing_offset+.13*np.clip(error,-.018,.018),-.075,.075)
        elif label not in ("slide omelet off plate","let omelet settle","clear skillet"):
            plan.landing_offset*=.925
        return pos+plan.rim_offset+correction+plan.landing_offset,rot,g,label
    target._omelet_feedback=True
    dm.ArmPlan.target=target

def demo():
    from yam_arena import demo as dm, ik, station
    from yam_arena.constants import READY
    if not getattr(dm.grip_rotation,"_custom_matrix",False):
        old=dm.grip_rotation
        def matrix_or_yaw(yaw):
            return yaw.matrix if hasattr(yaw,"matrix") else old(yaw)
        matrix_or_yaw._custom_matrix=True
        dm.grip_rotation=matrix_or_yaw
    install_carry_control(dm)
    names=("left","right")
    plans={a:[] for a in names}
    home={a:ik.tcp_pose(np.array(READY),station.BASES[a]) for a in names}
    last={a:(*home[a],-.032) for a in names}
    def move(sec,label,L=None,R=None):
        for a,pose in zip(names,(L,R)):
            if pose is not None:
                last[a]=pose
            p,r,g=last[a]
            plans[a].append(Waypoint(sec,tuple(p),g,label,SimpleNamespace(matrix=r,omelet=True)))
    lr=frame(np.array([0.,-1.,0.]))@np.diag([-1.,-1.,1.])
    rr=frame(np.array([0.,1.,0.]))
    fr=frame(PICK_DIR)
    left=at(PAN,PAN_H)+LEFT
    cover=at(PAN,PAN_H+RIM_H)
    edge=cover+RIGHT
    pick=at(PARK,PAD_H+.004)-RIM_GRIP*PICK_DIR
    g=-.0046
    squeeze=0.
    move(2.5,"line up with handle and rim",(left+[-.065,0,.060],lr,-.031),
        (pick-.045*PICK_DIR+[0,0,.035],fr,-.022))
    move(1.5,"approach",(left+[-.065,0,0],lr,-.030),(pick-.027*PICK_DIR,fr,-.016))
    move(1.2,"enter",(left,lr,-.030),(pick,fr,-.016))
    move(1.2,"grip handle and rim",(left,lr,g),(pick,fr,squeeze))
    move(1.8,"lift plate",R=(pick+[0,0,.065],fr,squeeze))
    move(3.2,"align plate",R=(edge+[0,0,.055],rr,squeeze))
    move(1.8,"cover",R=(edge+[0,0,-.003],rr,squeeze))
    off_l=left-cover
    off_r=RIGHT+[0,0,-.002]
    pivot=at(FLIP_CENTER,FLIP_HEIGHT)
    above_burner=at(PAN,.142)
    move(2.2,"raise covered pan",(above_burner+off_l,lr,g),(above_burner+off_r,rr,squeeze))
    move(2.3,"move clear of the hob",(pivot+off_l,lr,g),(pivot+off_r,rr,squeeze))
    for angle in range(5,181,5):
        r=rotate([0,1,0],angle)
        move(.23,"invert together",(pivot+r@off_l,r@lr,g),(pivot+r@off_r,r@rr,squeeze))
    move(1.4,"land omelet on plate")
    inv=rotate([0,1,0],180)
    pan_end=pivot+inv@off_l
    plate_end=pivot+inv@off_r
    move(1.3,"uncover",(pan_end+[0,0,.013],inv@lr,g),(plate_end+[0,0,-.031],inv@rr,squeeze))
    # The plate carries the food aside before the empty pan turns upright.
    aside=at(PARK,.155)-RIM_GRIP*PICK_DIR
    inverted_fr=fr@np.diag([-1.,-1.,1.])
    pan_center=pivot+[-.008,.005,.006]
    move(2.8,"carry omelet aside",(pan_center+inv@off_l,inv@lr,g),(aside,inverted_fr,squeeze))
    for angle in range(175,-1,-5):
        r=rotate([0,1,0],angle)
        move(.18,"restore pan",(pan_center+r@off_l,r@lr,g),None)
    move(2.,"return pan to hob",(left,lr,g),None)
    # The discharge edge is just inside the rim. The plate tips toward the pan as it withdraws.
    rack_clearance=.050
    base=at(PAN,PAN_H+RIM_H+.009+rack_clearance)+[.105,0.,0.]
    move(2.2,"place near edge above skillet",R=(base+inv@off_r,inv@rr,squeeze))
    move(2.4,"align omelet over skillet")
    axis=np.array([0.,-1.,0.])
    downhill=np.array([-1.,0.,0.])
    lip=base+.108*downhill
    for angle in range(4,37,4):
        tip=rotate(axis,angle)
        r=tip@inv
        p=lip+tip@(base+inv@off_r-lip)-[0,0,rack_clearance*min(angle/30,1.)]
        move(.25,"slide omelet off plate",R=(p,r@rr,squeeze))
    move(1.0,"let omelet settle")
    move(1.6,"clear skillet",R=(p-.040*downhill+[0,0,.015],r@rr,squeeze))
    # Rest the plate toward the outer side of the rack so its rim clears the pan.
    park_grip = pick + np.array([.031, -.024, 0.])
    move(2.5,"return plate",R=(park_grip+[0,0,.034],inverted_fr,squeeze))
    move(1.,"set plate down",R=(park_grip+[0,0,.002],inverted_fr,squeeze))
    move(1.,"release",(left,lr,-.022),(last["right"][0],last["right"][1],-.018))
    move(2.,"clear",(*home["left"],-.028),(*home["right"],-.028))
    return plans

def plate_clearance(tip):
    radial = max(0., float(np.linalg.norm(tip[:2])) - PR)
    vertical = max(0., -float(tip[2]), float(tip[2]) - .009)
    return math.hypot(radial, vertical)


class Score:
    def reset(self,*_):
        self.stage=0
        self.failed=False
        self.hold=0
        self.step=0
        self.contact=None
        self.pan_support=None
        self.plate_hob=None
        self.tool_hand_views=[]
        self.peak_finger_normals=np.zeros((2,2))
        self.physics_samples=0
        self.physics_peaks=np.zeros((2,2))
        self.window_peaks=np.zeros((2,2))
        self.separated_steps=0
        self.slide_start=None
        self.free_steps=0
        self.pose_cache={}
        self.food_pose=None

    def monitor_physics(self, env):
        """Read finger/utensil normal loads after each actual physics step."""
        sim = env.unwrapped.sim
        self.physics_dt = float(env.unwrapped.physics_dt)
        if getattr(sim.step, "_omelet_force_monitor", False):
            return
        original = sim.step
        def step(*args, **kwargs):
            result = original(*args, **kwargs)
            if len(self.tool_hand_views) == 2:
                peaks = []
                for view, tip_slice in zip(self.tool_hand_views, (slice(1, 3), slice(4, 6))):
                    forces = view.get_contact_force_matrix(dt=self.physics_dt).numpy().reshape(-1, 3)
                    peaks.append(np.linalg.norm(forces[tip_slice], axis=1))
                self.physics_peaks = np.maximum(self.physics_peaks, peaks)
                self.window_peaks = np.maximum(self.window_peaks, peaks)
                self.physics_samples += 1
            return result
        step._omelet_force_monitor = True
        sim.step = step

    def tick(self,env):
        from yam_arena.scene import object_states
        from isaaclab_physx.physics import PhysxManager
        st=object_states(env,["omelet","plate","skillet"])
        f,p,s=(st[n] for n in ("omelet","plate","skillet"))
        self.food_pose=f
        from yam_arena.runtime import link_poses
        for name,obj in (("left",s),("right",p)):
            pose=link_poses(env.unwrapped.scene[name+"_robot"])["gripper"]
            self.pose_cache[name]=(obj,pose[:3,3]+pose[:3,:3]@[0,0,.134],pose[:3,:3])
        if self.contact is None:
            prefix=env.unwrapped.scene.env_prim_paths[0]+"/"
            view=PhysxManager.get_physics_sim_view()
            paths=[prefix+"skillet/body",prefix+"plate/body",prefix+"hob/body",prefix+"cooling_rack/body"]
            for arm in ("left_robot","right_robot"):
                asset=env.unwrapped.scene[arm]
                body_paths=asset.root_physx_view.link_paths[0]
                for link in ("gripper","tip_left","tip_right"):
                    paths.append(body_paths[list(asset.body_names).index(link)])
            self.contact=view.create_rigid_contact_view([prefix+"omelet/body"],
                filter_patterns=[paths],max_contact_data_count=256)
            self.pan_support=view.create_rigid_contact_view([prefix+"skillet/body"],
                filter_patterns=[[prefix+"hob/body",prefix+"plate/body"]],max_contact_data_count=128)
            self.plate_hob=view.create_rigid_contact_view([prefix+"plate/body"],filter_patterns=[[prefix+"hob/body"]],max_contact_data_count=64)
            for tool in ("skillet","plate"):
                self.tool_hand_views.append(view.create_rigid_contact_view([prefix+tool+"/body"],filter_patterns=[paths[4:]],max_contact_data_count=128))
            if os.environ.get("OMELET_AUDIT_FILE"):
                self.monitor_physics(env)
            if self.contact.filter_count!=len(paths) or self.pan_support.filter_count!=2:
                raise RuntimeError("Could not monitor food contact with both utensils and both hands.")
        matrix=np.asarray(self.contact.get_contact_force_matrix(dt=env.unwrapped.physics_dt).numpy()).reshape(-1,3)
        contact=np.linalg.norm(matrix,axis=1)>.015
        on_pan,on_plate=contact[:2]
        if np.any(contact[2:]):
            self.failed=True
        fs=s.to_local(f.to_world([0,0,FOOD_H/2]))
        fp=p.to_local(f.to_world([0,0,FOOD_H/2]))
        tilt=s.tilt_deg()
        cover=s.to_local(p.pos)
        aligned=np.linalg.norm(cover[:2])<=PR-.087 and .022<cover[2]<.043
        same=np.dot(s.rotation()[:,2],p.rotation()[:,2])>math.cos(math.radians(15))
        fp_inside=np.linalg.norm(fp[:2])+FOOD_R < PR+.002
        fs_inside=np.linalg.norm(fs[:2])+FOOD_R < .087+.002
        pan_contact=self.pan_support.get_contact_force_matrix(dt=env.unwrapped.physics_dt).numpy().reshape(-1,3)
        on_hob,lid_contact=np.linalg.norm(pan_contact,axis=1)>.025
        before=self.stage
        if self.stage==0 and aligned and same and tilt<15 and fs_inside and on_pan and lid_contact: self.stage=1
        covers_food=fp_inside and 0<cover[2]<RIM_H+FOOD_H+.010
        if self.stage==1 and covers_food and same and (on_pan or on_plate) and 60<tilt<125 and s.pos[2]>TOP+.085:
            self.stage=2
        if self.stage==2 and tilt>150 and p.tilt_deg()>150 and f.tilt_deg()>145 and on_plate and fp_inside: self.stage=3
        released=(self.stage==3 and p.tilt_deg()>140 and on_plate and not on_pan and not lid_contact
                  and fp_inside and np.dot(s.pos-p.pos,-p.rotation()[:,2])>.050)
        self.separated_steps=self.separated_steps+1 if released else 0
        if self.separated_steps>=4: self.stage=4
        rim_drop=FOOD_R*float(np.linalg.norm(f.rotation()[2,:2]))
        hit_floor=min(f.to_world([0,0,z])[2] for z in (0,FOOD_H))-rim_drop<TOP+.001
        if hit_floor: self.failed=True
        self.free_steps=self.free_steps+1 if self.stage>=3 and not(on_plate or on_pan) else 0
        if self.free_steps>=8:
            self.failed=True
        over_hob=on_hob and np.linalg.norm(s.pos[:2]-PAN)<.022 and tilt<8
        plate_normal=-p.rotation()[:,2]
        gravity=np.array([0.,0.,-1.])
        tangent=gravity-np.dot(gravity,plate_normal)*plate_normal
        offsets=f.to_world([0,0,FOOD_H/2])-p.pos
        down=np.dot(offsets,tangent)/max(np.linalg.norm(tangent),1e-9)
        if self.stage==4 and over_hob and on_plate and p.tilt_deg()>168 and np.linalg.norm(p.pos[:2]-PAN)<.15:
            self.slide_start=fp.copy()
        if self.stage==4 and self.slide_start is not None:
            slip=p.rotation()@(fp-self.slide_start)
            slid=np.dot(slip,tangent)/max(np.linalg.norm(tangent),1e-9)>.020
            if over_hob and on_plate and 115<p.tilt_deg()<170 and down>.026 and slid and np.linalg.norm(f.pos[:2]-PAN)<.12:
                self.stage=5
        seated=over_hob and fs_inside and on_pan and f.tilt_deg()>165 and tilt<6 and abs(s.pos[2]-(TOP+PAN_H))<.006
        away=np.linalg.norm(s.pos[:2]-p.pos[:2])>.19
        end=(self.stage>=5 and seated and away and not on_plate and not lid_contact and np.linalg.norm(s.lin_vel)<.015
             and np.linalg.norm(f.lin_vel)<.015)
        released_now = False
        if end and not self.failed:
            tips = [link_poses(env.unwrapped.scene[a])[link][:3, 3]
                    for a in ("left_robot", "right_robot") for link in ("tip_left", "tip_right")]
            # Distance from the finite plate includes height. A released hand
            # above the plate can legitimately share its top-view footprint.
            clear = all(np.linalg.norm(s.to_local(q)-LEFT) > .070 and
                        plate_clearance(p.to_local(q)) > .040 for q in tips)
            if clear:
                released_now = all(np.max(np.linalg.norm(
                    v.get_contact_force_matrix(dt=env.unwrapped.physics_dt).numpy().reshape(-1, 3),
                    axis=1)) < .025 for v in self.tool_hand_views)
        self.hold = self.hold + 1 if end and released_now else 0
        if self.stage >= 5:
            self.stage = 6 if self.hold >= 20 else 5
        if self.stage!=before:
            print("OMELET_MILESTONE",self.step,self.stage,flush=True)
        if os.environ.get("OMELET_TRACE"):
            normals=[np.linalg.norm(v.get_contact_force_matrix(dt=env.unwrapped.physics_dt).numpy().reshape(-1,3),axis=1)
                     for v in self.tool_hand_views]
            self.peak_finger_normals=np.maximum(self.peak_finger_normals,np.stack((normals[0][1:3],normals[1][4:6])))
            trace_file = os.environ.get("OMELET_AUDIT_FILE")
            if trace_file:
                row = dict(step=self.step, stage=self.stage, failed=bool(self.failed), hold=self.hold,
                           seated=bool(seated), hob=bool(on_hob), away=bool(away), released=bool(released_now),
                           settled_before_hand_check=bool(end), pan_speed_m_s=float(np.linalg.norm(s.lin_vel)),
                           food_speed_m_s=float(np.linalg.norm(f.lin_vel)),
                           food_on_pan=bool(on_pan), food_on_plate=bool(on_plate),
                           pan_deg=float(tilt), plate_deg=float(p.tilt_deg()), food_deg=float(f.tilt_deg()),
                           tip_normals_N=[normals[0][1:3].tolist(), normals[1][4:6].tolist()],
                           physics_dt_s=self.physics_dt, physics_samples=self.physics_samples,
                           window_peak_N=self.window_peaks.tolist(), physics_peak_N=self.physics_peaks.tolist())
                self.window_peaks.fill(0.)
                with open(trace_file, "a") as stream:
                    print(json.dumps(row), file=stream)
            if self.step%20==0:
                print("PEAK_FINGER_NORMAL_N",self.step,np.round(self.peak_finger_normals,2).tolist(),flush=True)
        if self.step%20==0 and os.environ.get("OMELET_TRACE"):
            from yam_arena.runtime import link_poses
            tcp=[]
            for arm in ("left_robot","right_robot"):
                pose=link_poses(env.unwrapped.scene[arm])["gripper"]
                tcp.append(np.round(pose[:3,3]+pose[:3,:3]@[0,0,.134]-[0,0,TOP],3))
            print("TCP",self.step,*tcp,flush=True)
            if self.step>=20:
                hand_forces=[np.round(np.linalg.norm(v.get_contact_force_matrix(dt=env.unwrapped.physics_dt).numpy().reshape(-1,3),axis=1),1).tolist() for v in self.tool_hand_views]
                print("COOKWARE_FORCES",self.step,hand_forces,"handle error mm",round(float(1000*np.linalg.norm(s.to_local(self.pose_cache["left"][1])-LEFT)),1),flush=True)
        if self.step%10==0 and os.environ.get("OMELET_TRACE"):
            print("DISH_FRAME",self.step,"lid in skillet mm",np.round(cover*1000,1),
                  "food in plate mm",np.round(fp*1000,1),"food in skillet mm",np.round(fs*1000,1),
                  "right grasp in plate mm",np.round(p.to_local(self.pose_cache["right"][1])*1000,1),
                  "hob",bool(on_hob),"rim",bool(lid_contact),flush=True)
            print("FLIP",self.step,self.stage,self.failed,"pan",np.round(s.pos-[0,0,TOP],3),
              round(tilt),"plate",np.round(p.pos-[0,0,TOP],3),round(p.tilt_deg()),
              "food",np.round(f.pos-[0,0,TOP],3),round(f.tilt_deg()),"contact",bool(on_pan),bool(on_plate),"plate_on_hob",round(float(np.linalg.norm(self.plate_hob.get_contact_force_matrix(dt=env.unwrapped.physics_dt).numpy())),2),flush=True)
        self.step+=1

trial=Score()
trial.reset()
labkit.on_step(trial.tick)
labkit.on_reset(trial.reset)
def score(objects):
    return 0. if trial.failed else (0.,.15,.35,.60,.75,.85,1.)[trial.stage]
def success(objects):
    return score(objects)==1.
