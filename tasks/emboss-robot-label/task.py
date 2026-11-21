"""Contact-driven manual embossing prototype: ROBOT, then cut.
The 49-position wheel ordering, dimensions, and forces are prototype assumptions.
No task joints or robot poses are teleported. Embossing/feed/shearing are discrete
material responses driven exclusively by measured joint stroke and wheel angle.
"""
from __future__ import annotations
import math,json,os,pathlib,sys
import numpy as np
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
import arena_assets_fix
import labkit
from labkit import part,Joint,Mesh,on_reset,on_step
import hardware_limits
from mechanism import Mechanism,SYMBOLS,PITCH,TARGET,selected
from glyph_geometry import combined
from yam_arena.task_file import TABLE_TOP
SCENE='station';TOP=TABLE_TOP[SCENE]
WX,WY=.30,0.
WH=.135;WR=.038
BLACK=(.035,.040,.045);BLUE=(.08,.25,.42);WHITE=(.89,.89,.83)
STEEL=(.55,.57,.60);RED=(.75,.06,.04)
labkit.ARTICULATION_SLEEP=0.
labkit.MATTE_COLORS.update({BLACK:.7,BLUE:.6,WHITE:.6})
labkit.METAL_COLORS[STEEL]=(.4,.6)
on_reset(hardware_limits.verify)

items=[]
for i,c in enumerate(SYMBOLS):
    a=i*PITCH
    items.append((c,(.031*math.cos(a),.031*math.sin(a),WH+.0042),.0032,a-math.pi/2))
p,f=combined(items)
wheel_mesh=Mesh(p,f,BLACK,'none')
# Each ridge is part of the wheel collider and provides frictional grip.
ridges=[((WR*math.cos(a),WR*math.sin(a),WH),(.0018,.0025,.007),WHITE,a)
        for a in np.linspace(0,2*math.pi,98,endpoint=False)]
OBJECTS=[
 part('embosser',pos=(WX,WY,TOP),fixed=True,friction=.8,
  boxes=[((0,-.013,.067),(.029,.021,.108),BLUE),
         ((0,0,.017),(.065,.065,.024),STEEL),
         ((0,-.025,.070),(.045,.012,.060),STEEL),
         ((.040,0,WH+.003),(.010,.008,.004),RED),
         ((.065,0,WH-.009),(.052,.013,.004),BLUE)],
  cylinders=[((0,0,WH-.009),.028,.010,BLUE)],
  joints=[Joint('wheel',kind='hinge',axis='Z',anchor=(0,0,WH),
                lower=-6*math.pi,upper=6*math.pi,stiffness=.22,damping=.008,
                friction=.002,mass=.025,
                cylinders=[((0,0,WH),WR,.007,WHITE)],boxes=ridges,meshes=[wheel_mesh]),
          Joint('trigger',kind='hinge',axis='X',anchor=(0,0,.112),
                lower=-.30,upper=0.,stiffness=.85,damping=.018,mass=.025,
                boxes=[((0,.027,.073),(.027,.009,.074),BLACK)])]),
 part('bench_clamp',pos=(WX,WY,TOP),fixed=True,
      boxes=[((0,0,.005),(.12,.13,.01),STEEL),
             ((-.040,0,.025),(.012,.072,.04),STEEL),
             ((.040,0,.025),(.012,.072,.04),STEEL)],
      cylinders=[((-.045,0,.027),.006,.03,BLACK,'X')]),
]
_STATE=Mechanism();_N=0;_TRACE=None

def score(objects):return _STATE.result()['score']
def success(objects):return _STATE.result()['success']
def distance(objects):return 1.-score(objects)

class Orient(float):
    def __new__(cls,R):
        x=float.__new__(cls,0);x.rot=np.asarray(R);return x

def rx(a):
    c,s=math.cos(a),math.sin(a);return np.array([[1,0,0],[0,c,-s],[0,s,c]])
def rz(a):
    c,s=math.cos(a),math.sin(a);return np.array([[c,-s,0],[s,c,0],[0,0,1]])
def ry(a):
    c,s=math.cos(a),math.sin(a);return np.array([[c,0,s],[0,1,0],[-s,0,c]])

def demo():
    from yam_arena import demo as dm,ik
    from yam_arena.constants import READY,FINGER_OPEN,FINGER_CLOSED,LEFT_BASE_POS,RIGHT_BASE_POS
    from yam_arena.urdf import quat_xyzw_to_matrix
    old=dm.grip_rotation
    if not getattr(old,'_label_orient',False):
        def rotation(v=0.):return v.rot if hasattr(v,'rot') else old(v)
        rotation._label_orient=True;dm.grip_rotation=rotation
    down=quat_xyzw_to_matrix(np.asarray(dm.DOWN_GRIP))
    # Left fingers approach the trigger from +y, pressing towards -y.
    side=rx(-math.pi/2)@down@rz(math.pi/2)
    plans={'left':[],'right':[]};times={'left':0.,'right':0.}
    def wp(arm,sec,pos,f,label,R):
        plans[arm].append(dm.Waypoint(sec,tuple(pos),f,label,Orient(R)));times[arm]+=sec
    def wait(arm,t):
        if t<=times[arm]:return
        w=plans[arm][-1];wp(arm,t-times[arm],w.pos,w.finger,'wait clear',w.yaw.rot)
    W=np.array([WX,WY,TOP]);fo=FINGER_OPEN;fc=FINGER_CLOSED
    center=W+[0,0,WH+.008];hover=center+[0,0,.025]
    left_rest=W+[0,.095,.070];left_press=W+[0,.018,.070]
    wp('left',2.,left_rest,fc,'approach trigger',side)
    wp('right',2.,hover,fo,'approach wheel',down)
    q=0.
    for n,char in enumerate(list(TARGET)+[' ',' ','CUT']):
        goal=-SYMBOLS.index(char)*PITCH
        delta=(goal-q+math.pi)%(2*math.pi)-math.pi
        remaining=int(round(delta/PITCH));turns=[]
        while remaining:
            count=int(math.copysign(min(6,abs(remaining)),remaining))
            turns.append(count*PITCH);remaining-=count
        for turn in turns:
            wp('right',.65,hover,fo,'reset grasp angle',down)
            wp('right',.65,center,fo,'lower to wheel',down)
            wp('right',.55,center,fc,'grip wheel rim',down)
            # All motion is physical finger contact; the wheel has no target-letter motor.
            for k in range(1,5):
                wp('right',.20,center,fc,f'select {char}',rz(turn*k/4)@down)
            wp('right',.40,center,fo,'release wheel',rz(turn)@down)
            wp('right',.50,hover,fo,'clear wheel',rz(turn)@down)
        q+=delta
        if char=='CUT':
            # Observe the entire output strip with the real wrist camera before
            # cutting. This is an ordinary reachable arm motion; camera
            # calibration, material events, and the scorer are unchanged.
            wp('right',2.,W+[.030,0,.190],fo,'observe complete label',ry(math.radians(-25))@down)
        wait('left',times['right']+.35)
        wp('left',.65,left_press,fc,f'press {char}',side)
        wp('left',.45,left_press,fc,'full stroke',side)
        wp('left',.65,left_rest,fc,'fully release trigger',side)
        wp('left',.40,left_rest,fc,'spring return',side)
        wait('right',times['left'])
    wp('left',1.,left_rest+[0,.025,0],fo,'withdraw',side)
    wp('right',1.,hover+[0,-.08,0],fo,'withdraw',down)
    wait('left',max(times.values())+1.);wait('right',times['left'])
    return plans

_VIS={}
def _visual_tape(env):
    """Abstract feed and embossed text, generated from observed presses, never the requested word."""
    from isaaclab.sim.utils.stage import get_current_stage
    from pxr import UsdGeom,Gf,Vt
    stage=get_current_stage()
    if 'root' not in _VIS:
        _VIS['root']=str(next(p.GetPath() for p in stage.Traverse() if p.GetName()=='embosser' and p.GetPath().pathString.endswith('/embosser')))
    root=_VIS['root']+'/body'
    text=_STATE.labels[-1] if _STATE.labels else _STATE.ribbon
    start=.046 + (.002 if _STATE.labels else 0.)
    length=.008+.004*len(text)
    tape=UsdGeom.Cube.Define(stage,root+'/emitted_tape')
    tape.CreateSizeAttr(1.)
    if not tape.GetOrderedXformOps():
        tape.AddTranslateOp();tape.AddScaleOp()
    ops=tape.GetOrderedXformOps();ops[0].Set(Gf.Vec3d(start+length/2,0,WH-.006))
    ops[1].Set(Gf.Vec3d(length,.009,.0006))
    tape.CreateDisplayColorAttr().Set([(0.02,0.02,0.02)])
    its=[(c,(start+.005+(len(text)-1-i)*.004,0,WH-.0055),.005,math.pi)
         for i,c in enumerate(text) if c!=' ']
    mesh=UsdGeom.Mesh.Define(stage,root+'/embossed_glyphs')
    p,f=combined(its)
    mesh.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*v) for v in p]))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3]*len(f)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(f.ravel().tolist()))
    mesh.CreateDoubleSidedAttr(True)
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateDisplayColorAttr().Set([(1.,1.,1.)])

@on_reset
def reset(env):
    global _STATE,_N
    _STATE=Mechanism();_N=0;_VIS.clear()

@on_step
def update(env):
    global _N,_TRACE
    import torch
    _N+=1
    st=labkit._part_state(env,OBJECTS[0]);q=st.joints['wheel'];tr=st.joints['trigger']
    # Passive nearest-detent spring: target comes only from measured wheel angle.
    a=env.unwrapped.scene['embosser'];idx=list(a.joint_names).index('wheel_joint')
    dev=env.unwrapped.scene.env_origins.device
    detent=round(q/PITCH)*PITCH
    a.set_joint_position_target_index(target=torch.tensor([[detent]],device=dev),joint_ids=[idx])
    event=_STATE.update(q,tr,float(env.unwrapped.step_dt))
    if _N==1 or event:_visual_tape(env)
    result=_STATE.result()
    row=dict(step=_N,wheel=q,trigger=tr,selected=selected(q)[0],event=event,**result)
    if event or _N%10==0 or _N==1:print('EMBOSSER '+json.dumps(row),flush=True)
    dest=os.environ.get('TASK_TRACE_PATH')
    if dest:
        if _TRACE is None:_TRACE=open(dest,'x',buffering=1)
        _TRACE.write(json.dumps(row)+'\n')
