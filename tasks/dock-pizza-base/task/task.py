"""Dock a thin pizza base with a passive pin roller at the dual-YAM station."""
from pathlib import Path
import sys, os, json, math
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import labkit
from imprints import (BASE_RADIUS, CORE_RADIUS, DOUGH_THICKNESS, PINS,
                      Sheet, TIP_RADIUS, WIDTH)
from yam_arena.demo import Waypoint
from yam_arena.task_file import TABLE_TOP

SCENE = 'station'
TOP = TABLE_TOP[SCENE]
DECK = .036
CENTER = np.array([.298,-.034,TOP+DECK+DOUGH_THICKNESS])
PARK = np.array([.287,-.147,TOP+DECK+TIP_RADIUS])
HANDLE = np.array([-.065,0.,.032])
WHITE=(.85,.82,.70)
GRIP=(.13,.28,.32)
STEEL=(.50,.53,.52)
DOUGH=(.87,.70,.43)
labkit.ANGULAR_DAMPING=.012
labkit.CONTACT_OFFSET=.00015

def cap_mesh():
    n=96
    pts=[(BASE_RADIUS*math.cos(a),BASE_RADIUS*math.sin(a),z)
         for z in (-DOUGH_THICKNESS,0.) for a in np.arange(n)*2*math.pi/n]
    tris=[]
    for j in range(n):
        k=(j+1)%n
        tris.extend(((j,k,k+n),(j,k+n,j+n)))
    for j in range(1,n-1):
        tris.extend(((n,n+j,n+j+1),(0,j+1,j)))
    return labkit.Mesh(np.asarray(pts),np.asarray(tris),DOUGH,'none')

def bar(a,b,width,height,color):
    d=np.subtract(b,a)
    return (tuple(np.add(a,b)/2),(np.linalg.norm(d),width,height),color,
            (math.atan2(d[1],d[0]),-math.atan2(d[2],np.linalg.norm(d[:2]))))

def box(p,d,c):
    return (p,d,c)

bench=labkit.part('workboard',(.303,-.030,TOP+DECK/2),fixed=True,friction=1.1,
                  boxes=[box((0,0,0),(.309,.320,DECK),(.56,.43,.29))])
# Canvas for the deformable 4.5 mm layer. The rigid bottom stops the physical pins.
sheet=labkit.part('pizza',CENTER,fixed=True,meshes=[cap_mesh()],
                  cylinders=[((0,0,-DOUGH_THICKNESS+.0003),BASE_RADIUS,.0006,DOUGH)])
handle_boxes=[box(tuple(HANDLE),(.067,.018,.021),GRIP)]
for s in [-1,1]:
    handle_boxes.extend([bar((-.044,0,.032),(-.025,s*.034,.023),.004,.005,STEEL),
                         bar((-.025,s*.034,.023),(0,s*.034,0),.004,.005,STEEL)])
pins=[]
for y,phase in PINS:
    u=np.array([math.sin(phase),0,-math.cos(phase)])
    a=u*(CORE_RADIUS-.001)+[0,y,0]
    b=u*TIP_RADIUS+[0,y,0]
    pins.append(bar(a,b,.0028,.0028,WHITE))
wheel=labkit.Joint('roller',kind='hinge',axis='Y',anchor=(0,0,0),
    cylinders=[((0,0,0),CORE_RADIUS,WIDTH,WHITE,'Y')],boxes=pins,
    mass=.028,lower=-300,upper=300)
tool=labkit.part('docker',PARK,mass=.047,friction=.95,boxes=handle_boxes,joints=[wheel])
stand_height=TIP_RADIUS+HANDLE[2]-.0105
stand=labkit.part('rest',(PARK[0]+HANDLE[0],PARK[1],TOP+DECK),fixed=True,
                  boxes=[box((0,0,stand_height/2),(.024,.015,stand_height),(.42,.37,.32))])
OBJECTS=[bench,sheet,stand,tool]

def demo():
    opened=-.038
    closed=0.
    route=[]
    def move(sec,axle,grip,label):
        route.append(Waypoint(sec,tuple(axle+HANDLE),grip,label,0.))
    move(1.5,PARK+[0,0,.056],opened,'approach')
    move(1.0,PARK,opened,'surround handle')
    move(.65,PARK,closed,'grasp')
    move(1.1,PARK+[0,0,.048],closed,'lift')
    lanes=[(-.092,.092,-.048),(.092,-.092,0.),(-.092,.092,.048)]
    for start,end,y in lanes:
        air=TIP_RADIUS+.032
        z=TIP_RADIUS-DOUGH_THICKNESS+float(os.environ.get('DOCK_Z','-.0020'))
        move(1.1,CENTER+[start,y,air],closed,'align')
        move(.85,CENTER+[start,y,z],closed,'lower')
        move(3.5,CENTER+[end,y,z],closed,'roll')
        move(.8,CENTER+[end,y,air],closed,'lift clear')
    move(1.4,PARK+[0,0,.044],closed,'return')
    move(.9,PARK,closed,'seat tool')
    move(.6,PARK,opened,'release')
    move(.9,PARK+[0,-.025,.072],opened,'withdraw')
    move(1.,PARK+[0,-.025,.072],opened,'inspect')
    end=sum(w.dt for w in route) if hasattr(route[0],'dt') else 30.
    return dict(right=route,left=[Waypoint(end,(.32,.145,TOP+.19),-.03,'observe',0.)])

class Trial:
    def reset(self,env):
        self.surface=Sheet()
        self.step=0
        self.display=None
        self.done=False
        self.release_steps=0
        self.log=Path(os.environ['DOCK_OUT'])
        self.log.mkdir(exist_ok=True,parents=True)

    def draw(self):
        from pxr import UsdGeom,Gf,Vt
        import omni.usd
        if self.display is None:
            stage=omni.usd.get_context().get_stage()
            root='/World/DockerPunctures'
            inst=UsdGeom.PointInstancer.Define(stage,root)
            mark=UsdGeom.Sphere.Define(stage,root+'/mark')
            mark.CreateRadiusAttr(.00165)
            mark.CreateDisplayColorAttr([(0.27,.145,.065)])
            inst.CreatePrototypesRel().SetTargets([mark.GetPath()])
            self.display=inst
        positions=[(float(CENTER[0]+x),float(CENTER[1]+y),float(CENTER[2]+.00011)) for x,y in self.surface.holes]
        n=len(positions)
        self.display.CreateProtoIndicesAttr().Set(Vt.IntArray([0]*n))
        self.display.CreatePositionsAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*p) for p in positions]))
        self.display.CreateScalesAttr().Set(Vt.Vec3fArray([Gf.Vec3f(1,1,.13)]*n))

    def tick(self,env):
        st=labkit._part_state(env,tool)
        q=st.joints['roller']
        added=self.surface.update(st.pos-CENTER,st.rotation(),q)
        if added or self.step==0:
            self.draw()
        from yam_arena.runtime import tcp_world
        tcp=np.asarray(tcp_world(env.unwrapped.scene['right_robot'])).reshape(-1)[:3]
        clear=np.linalg.norm(tcp-st.to_world(HANDLE))>.042
        self.release_steps=self.release_steps+1 if clear else 0
        self.done=bool(self.release_steps>=6 and self.surface.fraction>=.96 and self.step>20 and np.linalg.norm(st.pos[:2]-PARK[:2])<.022
                       and abs(st.pos[2]-PARK[2])<.009 and np.linalg.norm(st.lin_vel)<.016)
        if self.step%4==0 or added:
            with (self.log/'trace.jsonl').open('a') as f:
                f.write(json.dumps(dict(step=self.step,axle=(st.pos-CENTER).round(4).tolist(),
                    q=round(q,4),grip_clear=clear.item(),holes=len(self.surface.holes),coverage=round(self.surface.fraction,4),
                    added=np.round(added,5).tolist(),returned=self.done))+'\n')
        self.step+=1

trial=Trial()
labkit.on_reset(trial.reset)
labkit.on_step(trial.tick)
def score(objects):
    return 1. if trial.done else min(.99,trial.surface.fraction)
def success(objects):
    return trial.done
