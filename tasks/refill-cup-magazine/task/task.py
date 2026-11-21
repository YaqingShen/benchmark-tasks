"""Transfer a sleeved stack of paper cups into a single-column vending dispenser."""
import json
import math
import os
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import labkit
from yam_arena.demo import Waypoint
from yam_arena.task_file import TABLE_TOP

SCENE = "station"
TOP = TABLE_TOP[SCENE]
SOURCE, TARGET = .098, .220
SOURCE_Y=.028
from hotdrink import (BOTTOM, FOOT, PITCH, R, GATE_R, VARIANT, PANEL_MASS, WRAP_R, MASS, COUNT, GATE, FILL_LINE, RIM_TOP,
                      TRAY_TOP, BEAM_HEIGHT, WINDOW_STEPS, CYCLE_STEPS, cam_fraction, station)
DOCK=.100
FIXTURE=os.environ.get("CUP_FIXTURE","")
paper = (.93,.90,.80)
wrap = (.34,.66,.73)
metal = (.30,.34,.37)
tray = (.57,.62,.60)

def box(p,s,c,yaw=0.):
    return (p,s,c,yaw)

def part(name,x,y,z,**opts):
    return labkit.part(name,(x,y,z+TOP),**opts)

from cup_models import paper_cup, sleeve
import carry_feedback
from grip_contacts import PadForces, enable_reports
from wrapper_clearance import SleeveOutline
enable_reports(labkit)
film=(.78,.89,.91)
labkit.GLASS_COLORS.add(film)
seam=(.54,.73,.77)
labkit.CONTACT_OFFSET=.0002
# The service module mounts on a lower shelf. Its internal dimensions do not
# change. The rectangular opening leaves access to the delivery station.
labkit.TABLE_CUTS.append((TARGET,0.,.206,.230))

base=DOCK+BOTTOM
OBJECTS=[part("loading_dock",SOURCE,SOURCE_Y,base/2,fixed=True,
    boxes=[box((0,0,0),(.11,.11,base),tray)])]
folds,tab=sleeve(film,seam)
cap_mass=(math.pi*WRAP_R**2+math.pi*WRAP_R*.024*2)*.0000381*923.
OBJECTS.append(part("wrapper",SOURCE,SOURCE_Y,DOCK,mass=cap_mass,friction=1.,
                    cylinders=[((0,0,.0622),WRAP_R,.0004,film)],joints=folds,boxes=tab))
cup_walls,cup_contacts=paper_cup(paper)
for i in range(COUNT):
    x,z=(TARGET,GATE+.0010) if FIXTURE else (SOURCE,DOCK)
    OBJECTS.append(part(f"cup{i}",x,0. if FIXTURE else SOURCE_Y,z+i*PITCH,mass=MASS,friction=.58,
        cylinders=[((0,0,BOTTOM+.0006),FOOT,.0012,paper)],meshes=cup_walls,boxes=cup_contacts))
# Each pair of tabs is actuated by the dispenser, independently of the arms.
joints=[]
for level,axis in (("lower","X"),("upper","Y")):
    for s,tag in ((1,"pos"),(-1,"neg")):
        lower=level=="lower"
        radius=GATE_R+.0045 if lower else GATE_R+.0185
        z=-.001 if lower else PITCH-.001
        p=(s*radius,0,z) if lower else (0,s*radius,z)
        size=(.012,.020,.002) if lower else (.020,.012,.002)
        delta=s*(.016 if lower else -.014)
        angle=(0. if lower else math.pi/2) + (math.pi if s<0 else 0.)
        travel=0. if lower else .014
        ledges=[]
        for j in range(-3,4):
            theta=angle+j*math.pi/18
            rr=GATE_R+.0005
            px=rr*math.cos(theta)+(s*travel if lower else 0.)
            py=rr*math.sin(theta)+(s*travel if not lower else 0.)
            ledges.append(box((px,py,z),(.0048,.0074,.0015),metal,theta))
        joints.append(labkit.Joint(level+tag,axis=axis,anchor=p,boxes=tuple(ledges),
            lower=min(0.,delta),upper=max(0.,delta),mass=.006,stiffness=700,damping=3.))
OBJECTS.append(part("dispenser",TARGET,0,GATE,fixed=True,joints=joints,
    boxes=station(box)+[
        box((.086,s*.113,-GATE-.006),(.024,.032,.008),metal)
        for s in (-1,1)]))

def demo():
    # Rotate the wrists about world X so the palms stay on their respective
    # sides of the workpiece. The physical jaw closing axes remain horizontal.
    from yam_arena import demo as yam_demo
    carry_feedback.install(yam_demo)
    from yam_arena.urdf import axis_angle_matrix
    base=getattr(yam_demo.grip_rotation,"base_rotation",yam_demo.grip_rotation)
    approach={-10.:(-math.pi/2,-.11),10.:(math.pi/2,.58)}
    from scipy.spatial.transform import Rotation
    from yam_arena.station import wrist_camera
    from yam_arena.constants import TCP_OFFSET
    cam=wrist_camera("right")
    # Keep the stock calibrated camera on the wrist and move the robot to
    # look through the access opening at the tray during the dry test.
    eye=np.array([.205,-.202,TOP+.183])
    focus=np.array([TARGET,0.,TOP+TRAY_TOP+.069])
    backward=(eye-focus)/np.linalg.norm(eye-focus)
    horizontal=np.cross([0.,0.,1.],backward)
    horizontal/=np.linalg.norm(horizontal)
    view=np.column_stack((horizontal,np.cross(backward,horizontal),backward))
    inspect_rot=view@Rotation.from_quat(cam.rot_xyzw).as_matrix().T
    inspect_tcp=eye+inspect_rot@(np.array(TCP_OFFSET)-np.array(cam.pos))
    def rotation(yaw=0.):
        if yaw==20.:
            return inspect_rot
        if yaw in approach:
            jaw_yaw,tilt=approach[yaw]
            return axis_angle_matrix(np.array([1.,0.,0.]),tilt) @ base(jaw_yaw)
        return base(yaw)
    rotation.base_rotation=base
    yam_demo.grip_rotation=rotation
    if FIXTURE or os.environ.get("CUP_IDLE"):
        return {side:[Waypoint(6.,(.30,y,TOP+.24),-.032,"",0.)] for side,y in (("left",.21),("right",-.21))}
    # The lower grip and the grip on the film seam travel at the same speed.
    # Transfer weight to the magazine before pulling the sleeve free.
    left=[];right=[]
    gap=-.0465
    lower=float(os.environ.get("CUP_LOWER_GRIP","-.0397"))
    seam_grip=-.0005
    center_bias=float(os.environ.get("CUP_AIM_Y",".002"))
    def command(dt,l,r,label):
        for series,(x,y,z,grip),yaw in ((left,l,-10.),(right,r,10.)):
            fraction=np.clip((x-SOURCE)/(TARGET-SOURCE),0.,1.)
            series.append(Waypoint(dt,(x,y+center_bias*fraction,TOP+z),grip,label,yaw))
    grip_depth=float(os.environ.get("CUP_GRIP_DEPTH",".008"))
    def pose(x,y,h,lg,rg):
        return ((x,y+.003,h-grip_depth,lg),(x,y-.060,h+.070,rg))
    def together(dt,x,y,h,lg,rg,label):
        command(dt,*pose(x,y,h,lg,rg),label)
    left_home=(SOURCE,SOURCE_Y+.08,.155,gap)
    right_home=(SOURCE,-.12,.17,-.025)
    command(1.5,left_home,right_home,"approach from opposite sides")
    left_pick,right_pick=pose(SOURCE,SOURCE_Y,DOCK,gap,-.019)
    command(1.4,left_home,right_pick,"approach top corner")
    command(.9,left_home,(*right_pick[:3],seam_grip),"pinch the film first")
    command(1.4,(SOURCE,SOURCE_Y+.003,.165,gap),(*right_pick[:3],seam_grip),"approach from left")
    command(1.5,left_pick,(*right_pick[:3],seam_grip),"surround lower sleeve")
    together(1.,SOURCE,SOURCE_Y,DOCK,lower,seam_grip,"support lower taper")
    together(1.1,SOURCE,SOURCE_Y,DOCK+.020,lower,seam_grip,"lift together")
    together(1.5,SOURCE,SOURCE_Y,.160,lower,seam_grip,"raise wrapped cups")
    together(3.6,TARGET,0.,.160,lower,seam_grip,"translate together")
    handoff=float(os.environ.get("CUP_HANDOFF",".101"))
    together(2.0,TARGET,0.,handoff,lower,seam_grip,"guide bottom into magazine")
    together(1.0,TARGET,0.,handoff,lower,seam_grip,"hold at opening")
    open_dt=float(os.environ.get("CUP_OPEN_TIME",".18"))
    open_dx=float(os.environ.get("CUP_OPEN_SHIFT",".006"))
    parked_left,parked_right=pose(TARGET,0.,handoff,gap,seam_grip)
    command(open_dt,(parked_left[0]+open_dx,*parked_left[1:]),parked_right,"release lower grip over guide")
    guide=(TARGET,.090,handoff+.002,gap)
    command(.65,guide,(TARGET,-.060,handoff+.070,seam_grip),"clear cup stack")
    command(1.25,guide,(TARGET,-.060,handoff+.070,seam_grip),"settle cups in guide")
    # Once the load rests on the separator, grasp the seal at both ends.
    # Raising both ends keeps the lower film edge from catching the upper cups.
    def extract(dt,x,y,z,g,label):
        command(dt,(x,y+.055,z,g),(x,y-.055,z,g),label)
    extract(1.3,TARGET,0.,handoff+.074,-.015,"open over seal")
    extract(1.,TARGET,0.,handoff+.049,-.012,"align with both ends")
    extract(1.,TARGET,0.,handoff+.049,0.,"pinch both ends")
    extract(3.4,TARGET,0.,.266,0.,"draw sleeve off")
    extract(2.6,.115,-.095,.266,0.,"carry empty sleeve")
    extract(1.,.115,-.095,.192,0.,"set wrapper aside")
    extract(.6,.115,-.095,.192,-.018,"release empty sleeve")
    left.append(Waypoint(2.4,(SOURCE,.15,TOP+.17),gap,"inspect delivery tray",-10.))
    right.append(Waypoint(2.4,tuple(inspect_tcp),-.025,"inspect delivery tray",20.))
    left.append(Waypoint(4.,(SOURCE,.15,TOP+.17),gap,"observe dispenser",-10.))
    right.append(Waypoint(4.,tuple(inspect_tcp),-.025,"observe dispenser",20.))
    return {"left":left,"right":right}

class Trial:
    def reset(self,env):
        self.step=0
        self.loaded=False
        self.arrived=False
        self.outline=SleeveOutline()
        self.released_steps=0
        self.forces=PadForces()
        self.ever_touched_cup=False
        self.cycle_start=None
        self.beam_seen=False
        self.single=False
        self.stable=0
        self.counts=[]
        self.retained_count=0
        self.takeout_gap=None
        self.final=False
        self.credit=0.
        self.log=Path(os.environ["CUP_RUN_DIR"])/"trial.jsonl"

    def tick(self,env):
        import torch
        from yam_arena.runtime import tcp_world
        scene=env.unwrapped.scene
        def pos(name):
            return labkit._np(scene[name].data.root_link_pose_w)[0,:3]-[0,0,TOP]
        cups=np.array([pos(f"cup{i}") for i in range(COUNT)])
        pkg=pos("wrapper")
        carry_feedback.publish(scene,self.step,labkit._np(scene["wrapper"].data.root_link_pose_w)[0])
        dist=np.linalg.norm(cups[:,:2]-[TARGET,0],axis=1)
        # Check the physical rim top including tilt against the fill line.
        quat=np.array([labkit._np(scene[f"cup{i}"].data.root_link_pose_w)[0,3:] for i in range(COUNT)])
        up=1.-2.*(quat[:,0]**2+quat[:,1]**2)
        rim_z=cups[:,2]+RIM_TOP*up+R*np.sqrt(np.clip(1.-up**2,0,1))
        aligned=bool(np.all(dist<.011) and np.min(up)>.98 and np.max(rim_z)<=GATE+FILL_LINE
                     and np.max(np.abs(cups[:,2]-(GATE+PITCH*np.arange(COUNT))))<.004)
        if np.linalg.norm(pkg[:2]-[TARGET,0])<.018 and np.max(dist)<.018:
            self.arrived=True
        if aligned and (self.arrived or FIXTURE):
            self.loaded=True
        forces=self.forces.sample(float(env.unwrapped.physics_dt))
        self.ever_touched_cup |= max(forces["cup_N"])>.001
        distance=self.outline.clearance(scene["wrapper"],TARGET,labkit._np)
        clear=distance>.0565
        released=clear and max(forces["wrapper_N"])<=.001
        self.released_steps=self.released_steps+1 if released else 0
        if self.cycle_start is None and self.loaded and aligned and self.released_steps>=10 and self.step>=10:
            self.cycle_start=self.step
        age=-1 if self.cycle_start is None else self.step-self.cycle_start
        if 0<=age<CYCLE_STEPS:
            target=[]
            for name in scene["dispenser"].joint_names:
                low=name.startswith("lower")
                fraction=cam_fraction(age,low)
                sign=1 if "pos" in name else -1
                target.append(sign*(.016 if low else -.014)*fraction)
            asset=scene["dispenser"]
            desired=torch.tensor([target],dtype=torch.float32,device=env.unwrapped.device)
            if hasattr(asset,"set_joint_position_target_index"):
                asset.set_joint_position_target_index(target=desired)
            else:
                asset.set_joint_position_target(desired)
        if age>=0:
            in_beam=(np.abs(cups[:,0]-TARGET)<FOOT)&(np.abs(cups[:,1])<.058)
            if age<=CYCLE_STEPS+WINDOW_STEPS:
                self.beam_seen |= bool(np.any(in_beam&(cups[:,2]>BEAM_HEIGHT)&(cups[:,2]+BOTTOM<BEAM_HEIGHT)))
            landed=(dist<.026)&(up>.98)&(np.abs(cups[:,2]+BOTTOM-TRAY_TOP)<.003)
            retained=(dist<.013)&(up>.98)&(cups[:,2]>=GATE-.004)&(rim_z<=GATE+FILL_LINE)
            self.retained_count=int(retained.sum())
            correct_counts=landed.sum()==1 and retained.sum()==COUNT-1
            # The delivered rim must clear the lowest retained cup's base.
            # Use actual tilt and the lowest point on the base, in world z.
            base_z=cups[:,2]+BOTTOM*up-FOOT*np.sqrt(np.clip(1.-up**2,0,1))
            self.takeout_gap=float(np.min(base_z[retained])-np.max(rim_z[landed])) if correct_counts else None
            self.single=bool(self.beam_seen and released and correct_counts and self.takeout_gap>=.020)
            self.stable=self.stable+1 if self.single and age>=CYCLE_STEPS else 0
            self.counts=[int(landed.sum())]
        self.final=not FIXTURE and not self.ever_touched_cup and self.stable>=10
        self.credit=(0. if FIXTURE or self.ever_touched_cup else 1. if self.final
                     else .5 if self.cycle_start is not None else .25 if self.loaded else 0.)
        if self.step%5==0 or self.final:
            with self.log.open("a") as f:
                f.write(json.dumps(dict(step=self.step,cups=cups.round(4).tolist(),wrapper=pkg.round(4).tolist(),
                    fingers=[np.asarray(tcp_world(scene[s+"_robot"])).flatten()[:3].round(4).tolist() for s in ("left","right")],
                    forces=forces,ever_touched_cup=self.ever_touched_cup,wrapper_clear=clear,wrapper_clearance_m=distance,wrapper_released=released,retained=self.retained_count,takeout_gap_m=self.takeout_gap,
                    sleeve_joint=labkit._np(scene["wrapper"].data.joint_pos)[0].round(4).tolist(),
                    gate_mm=(labkit._np(scene["dispenser"].data.joint_pos)[0]*1000).round(2).tolist(),fixture=FIXTURE,cup_variant=VARIANT,aligned=aligned,loaded=self.loaded,age=age,beam=self.beam_seen,stable=self.stable,counts=self.counts,score=self.credit))+"\n")
        self.step+=1

trial=Trial()
labkit.on_reset(trial.reset)
labkit.on_step(trial.tick)
def score(objects):return trial.credit
def success(objects):return trial.final
