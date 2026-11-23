"""Development scene: open a photographed milk-carton pull-ring seal cleanly.
Native rigid bodies and breakable perimeter joints; no scripted object motion.
The task starts with the outer cap removed on a clean rest and the inner seal intact.
The recorded native trials determine the reported scores.
"""
from __future__ import annotations
import json,math,os,pathlib,sys
import numpy as np
sys.path.insert(0,str(pathlib.Path(__file__).parent))
import labkit
from labkit import part,Joint,on_reset,on_step
import hardware_limits
import milk_fluid
from yam_arena.task_file import TABLE_TOP
SCENE='station'; TOP=TABLE_TOP[SCENE]; SUCCESS_HOLD_S=1.
WHITE=(.93,.93,.91); RED=(.65,.025,.025); STEEL=(.65,.67,.69); BLUE=(.06,.16,.30)
labkit.CONTACT_OFFSET=.0002
labkit.ARTICULATION_SLEEP=0.
labkit.MATTE_COLORS.update({WHITE:.55,RED:.65})
CFG=json.loads((pathlib.Path(__file__).parent/'hardware_config.json').read_text())['nominal_simulation']
BASE_FOLD=CFG['inner_floor_height_m']
H=CFG['carton_wall_height_m']; W=CFG['carton_width_m']; DEP=CFG['carton_depth_m']; PEAK=CFG['carton_roof_peak_height_m']
P=np.array([.395,.035,TOP+.004]); TILT=-math.pi/4
S=labkit._rot((0,TILT,0)); C=np.array([-.026,0.,H+.024]); N=S[:,2]
RIM=.016; DISK_R=RIM; SEAL_Z=.0035
NECK_MID_Z=.006; NECK_HEIGHT=.014

def ring_boxes(r,z,thick,width,col,n=32):
 out=[]
 for k in range(n):
  a=(k+.5)*2*math.pi/n
  out.append(((r*math.cos(a),r*math.sin(a),z),(width,2*r*math.tan(math.pi/n)+.0003,thick),col,a))
 return out

def tilt_boxes(boxes,center=C):
 # General rotated collision boxes use triangle meshes to preserve hollow geometry.
 out=[]
 signs=np.array([[x,y,z] for x in [-1,1] for y in [-1,1] for z in [-1,1]])
 tris=np.array([[0,1,3],[0,3,2],[4,6,7],[4,7,5],[0,4,5],[0,5,1],[2,3,7],[2,7,6],[0,2,6],[0,6,4],[1,5,7],[1,7,3]])
 for b in boxes:
  c,size,col=b[:3];yaw=b[3] if len(b)>3 else 0.
  pts=(signs*np.array(size)/2)@labkit._rot((0,0,yaw)).T+np.array(c)
  out.append(labkit.Mesh(pts@S.T+center,tris,col,'hull'))
 return out

def disk_mesh(radius,z,thick,col,center=C):
 n=48;pts=[]
 for zz in [z-thick/2,z+thick/2]:
  pts += [(radius*math.cos(a),radius*math.sin(a),zz) for a in np.arange(n)*2*math.pi/n]
 pts += [(0,0,z-thick/2),(0,0,z+thick/2)]
 tris=[]
 for i in range(n):
  j=(i+1)%n;tris += [(i,j,n+j),(i,n+j,n+i),(2*n,j,i),(2*n+1,n+i,n+j)]
 return labkit.Mesh(np.array(pts)@S.T+center,np.array(tris),col,'hull')

def gable_end_panels():
 # Close both triangular end folds. Subdivide thin prisms to keep their aspect
 # ratio moderate for GPU particle collisions; all faces meet geometrically.
 faces=[np.array([[-W/2,H-.001],[0,H-.001],[0,PEAK]]),np.array([[0,H-.001],[W/2,H-.001],[0,PEAK]])]
 meshes=[];tris=np.array([[0,2,1],[3,4,5],[0,1,4],[0,4,3],[1,2,5],[1,5,4],[2,0,3],[2,3,5]])
 for side in [-1,1]:
  yc=side*(DEP/2-.0007)
  for a,b,c in faces:
   ab=(a+b)/2;bc=(b+c)/2;ca=(c+a)/2
   for triangle in [[a,ab,ca],[ab,b,bc],[ca,bc,c],[ab,bc,ca]]:
    pts=np.array([[x,yc+dy,z] for dy in [-.0007,.0007] for x,z in triangle])
    meshes.append(labkit.Mesh(pts,tris,RED,'hull'))
 return meshes

import thread_mesh
from roof_fitment import flange
FULL_CAP=os.environ.get('MILK_FULL_CAP','0')=='1'
# Full-scale carton, formed roof and a genuinely hollow neck. No hidden clamp.
roofpts=np.array([[-W/2,-DEP/2,H],[-W/2,DEP/2,H],[W/2,-DEP/2,H],[W/2,DEP/2,H],[0,-DEP/2,PEAK],[0,DEP/2,PEAK]])
roofboxes=[((.0255,0,H+.0245),(.072,.102,.0015),RED,(0,math.pi/4)),
           ((-.026,-.038,H+.024),(.072,.026,.0015),RED,(0,TILT)),
           ((-.026,.038,H+.024),(.072,.026,.0015),RED,(0,TILT)),
           ((-.045,0,H+.005),(.018,.05,.0015),RED,(0,TILT)),
           ((-.007,0,H+.043),(.018,.05,.0015),RED,(0,TILT))]
# Opaque visual roof may occlude the throat; omit its front panel and retain actual shell strips.
roofvisual=labkit.Mesh(roofpts,np.array([[2,4,5],[2,5,3],[0,4,2],[1,3,5]]),RED,'none')
# Lump retained bulk inertia at the filled volume, not in the paperboard walls.
rigid_mass=CFG['filled_mass_kg']-(milk_fluid.PARTICLE_MASS if milk_fluid.ENABLED else 0.)
free_width=W-.0028
free_depth=DEP-.0028
filled_height=milk_fluid.VOLUME_ML*1e-6/(free_width*free_depth)
center=(0.,0.,BASE_FOLD+filled_height/2)
inertia=rigid_mass/12*np.array([free_depth**2+filled_height**2,free_width**2+filled_height**2,free_width**2+free_depth**2])
labkit.MASS_PROPERTIES['carton']=(center,tuple(inertia))
CARTON=part('carton',pos=P,mass=rigid_mass,friction=.6,
 boxes=[((0,0,.001),(W,DEP,.002),RED),((-W/2+.0007,0,H/2),(.0014,DEP,H),RED),((W/2-.0007,0,H/2),(.0014,DEP,H),RED),((0,-DEP/2+.0007,H/2),(W,.0014,H),RED),((0,DEP/2-.0007,H/2),(W,.0014,H),RED),((0,0,PEAK+.004),(.004,DEP,.008),WHITE)]+roofboxes,
 meshes=[roofvisual]+tilt_boxes(ring_boxes(.0178,NECK_MID_Z,NECK_HEIGHT,.0036,WHITE))+(thread_mesh.ridge(labkit,S,C,.0193,.0202,.003) if FULL_CAP else [])+gable_end_panels()+flange(labkit.Mesh,S,C,RIM,RED))
# Initial ring is flat. It must be raised by tool/hand contact.
DIRECT_GRIP_POINT=(.020,-.003)
RECEIVE_INSET=.010
HOOK_RING_Y=.010
HOOK_RING_U=.0115+math.sqrt(.0115**2-HOOK_RING_Y**2)
anchor=C+S@np.array([-.0115,0,SEAL_Z+.004])
rb=ring_boxes(.0115,SEAL_Z+.004,.003,.003,WHITE,32)
ringjoint=Joint('pullring',kind='hinge',axis='Y',anchor=tuple(anchor),lower=-2.4,upper=0.,initial=0.,target=0.,stiffness=.0004,damping=.00002,friction=.00001,mass=.001,
 meshes=tilt_boxes(rb))
SEAL=part('seal',pos=P,mass=.003,friction=.7,meshes=[disk_mesh(DISK_R,SEAL_Z,.001,WHITE)],joints=[ringjoint])
# User-selected task start: cap already removed, clean inner face upwards.
CAP_POS=np.array([.24,-.30,TOP+.004])
CAP=part('cap',pos=CAP_POS,mass=.004,friction=.6,cylinders=[((0,0,.001),.020,.002,WHITE,'Z')],boxes=ring_boxes(.019,.008,.014,.002,WHITE))
if FULL_CAP:
 CAP=part('cap',pos=P,mass=.007,friction=.6,
  meshes=tilt_boxes(ring_boxes(.0214,.012,.020,.0016,WHITE))+
         [disk_mesh(.0223,.023,.002,WHITE)]+
         thread_mesh.ridge(labkit,S,C,.0199,.02065,.0045)+
         tilt_boxes([((.02235*math.cos(a),.02235*math.sin(a),.011),(.0007,.0007,.016),WHITE,a) for a in np.arange(48)*2*math.pi/48]))

# Dimensioned side-offset stainless lifting hook with a narrow lower tongue.
# Prototype tool. It is not claimed to be a digitized retail product.
HOOK_POS=np.array([.28,.115,TOP+.008])
HOOK_GRIP=np.array([-.025,.065,.036])
HOOK_TOE=np.array([-.015,0.,.005])
HOOK=part('hook',pos=HOOK_POS,mass=.045,friction=.9,
 boxes=[(tuple(HOOK_GRIP),(.044,.016,.018),BLUE),((-.020,.052,.037),(.003,.042,.006),STEEL),
        ((-.020,.032,.027),(.003,.003,.022),STEEL),((-.020,.016,.017),(.003,.034,.002),STEEL),
        ((-.020,0,.011),(.002,.0015,.014),STEEL),(tuple(HOOK_TOE),(.009,.0015,.001),STEEL)])
TRAY_POS=np.array([.455,-.18,TOP+.004])
TRAY=part('discard',pos=TRAY_POS,mass=.12,friction=.7,
 boxes=[((0,0,.002),(.20,.12,.004),(.25,.28,.30)),((-.099,0,.012),(.002,.12,.02),WHITE),((.099,0,.012),(.002,.12,.02),WHITE),((0,-.059,.012),(.20,.002,.02),WHITE),((0,.059,.012),(.20,.002,.02),WHITE)])
TOOL_MAT=part('tool_mat',pos=(.255,.18,TOP),fixed=True,boxes=[((.025,-.030,.00025),(.140,.110,.0005),(.3,.4,.45)),((-.016,0,.01775),(.007,.018,.0355),(.3,.4,.45)),((.016,0,.01775),(.007,.018,.0355),(.3,.4,.45))])
CAP_REST_POS=np.array([.24,-.30,TOP])
CAP_REST=part('cap_rest',pos=CAP_REST_POS,mass=.10,boxes=[((0,0,.002),(.08,.08,.004),BLUE)])
OBJECTS=[CARTON,SEAL,CAP,HOOK,TRAY,TOOL_MAT,CAP_REST]
BREAK_BUDGET=float(os.environ.get('MILK_BREAK_BUDGET',CFG['native_resultant_bond_force_n']))
# Native maximum-coordinate joints may break; articulation joints cannot.
_author0=labkit._author

def author(prim_path,cfg,**kwargs):
 root=_author0(prim_path,cfg,**kwargs)
 if str(prim_path).endswith('/cap') and not FULL_CAP:
  # The removed cap is reserved on its clean rest, outside the manipulation task.
  from pxr import Usd,UsdPhysics
  from isaaclab.sim.utils.stage import get_current_stage
  globals()['FOOD_CONTACT_COLLIDERS']=[str(p.GetPath()) for p in Usd.PrimRange(get_current_stage().GetPrimAtPath(prim_path)) if p.HasAPI(UsdPhysics.CollisionAPI)]
 if str(prim_path).endswith('/cap') and FULL_CAP:
  import isaaclab.sim as sim_utils
  from isaaclab.sim.utils import bind_physics_material
  from isaaclab.sim.utils.stage import get_current_stage
  mat=str(prim_path)+'/Looks/internal_thread_material'
  cfgmat=sim_utils.RigidBodyMaterialCfg(static_friction=.15,dynamic_friction=.15,friction_combine_mode='min')
  cfgmat.func(mat,cfgmat)
  for index in range(33,93):
   bind_physics_material(str(prim_path)+f'/body/mesh_{index}',mat,stage=get_current_stage())
 if str(prim_path).endswith('/seal'):
  from pxr import Gf,Sdf,UsdPhysics
  import omni.usd
  stage=omni.usd.get_context().get_stage();parent=str(prim_path).rsplit('/',1)[0]
  # One rigid membrane has six constrained degrees of freedom. A single
  # resultant bond avoids redundant joint reactions between the same bodies.
  # The microscopic tear path is unresolved. A gross inward failure still fails scoring.
  point=C+S@np.array([0.,0.,SEAL_Z])
  bond=UsdPhysics.FixedJoint.Define(stage,str(prim_path)+'/perimeter_bond')
  bond.CreateBody0Rel().SetTargets([Sdf.Path(parent+'/carton/body')])
  bond.CreateBody1Rel().SetTargets([Sdf.Path(str(prim_path)+'/body')])
  bond.CreateLocalPos0Attr(Gf.Vec3f(*point))
  bond.CreateLocalPos1Attr(Gf.Vec3f(*point))
  bond.CreateExcludeFromArticulationAttr(True)
  # The intact bond carries the edge reaction. Resolving coincident rigid
  # seal/neck contact as well can preload the bond before the tool pulls it.
  bond.CreateCollisionEnabledAttr(False)
  bond.CreateBreakForceAttr(float('inf'))
  bond.CreateBreakTorqueAttr(float('inf'))
  bond.CreateJointEnabledAttr(True)
  print('MILK_NATIVE_SEAL '+json.dumps({'external_tensile_threshold_n':BREAK_BUDGET,'calibrated_product_measurement':False}),flush=True)
 return root
labkit._author=author

from geometry_observer import GeometryObserver
OBSERVER=GeometryObserver(globals())
TRACK={'step':0,'score':0.,'probe_only':True}
on_reset(hardware_limits.verify)
@on_reset
def reset(env):
 TRACK.update(step=0,score=0.,probe_only=True)
 OBSERVER.previous=None
@on_step
def record(env):
 TRACK['step']+=1
 if TRACK['step']%20==0:
  st={o.name:labkit._part_state(env,o) for o in OBJECTS}
  data={'step':TRACK['step'],'previous_control_step_score':score(st),
        'objects':{n:{'pos':s.pos.tolist(),'quat':s.quat_xyzw.tolist(),'joints':s.joints} for n,s in st.items()}}
  data['geometry']=OBSERVER.read(st,dt=1.)
  data['liquid']=TRACK.get('liquid')
  data['feedback']=TRACK.get('feedback')
  data['robot']={}
  from yam_arena.urdf import quat_xyzw_to_matrix
  for arm in ['left','right']:
   robot=env.unwrapped.scene[arm+'_robot'];pose=labkit._np(robot.data.body_link_pose_w)[0,list(robot.body_names).index('gripper')];rot=quat_xyzw_to_matrix(pose[3:])
   data['robot'][arm]={'tcp':(pose[:3]+.134*rot[:,2]).tolist(),'quat':pose[3:].tolist(),'q':labkit._np(robot.data.joint_pos)[0].tolist()}
  p=os.environ.get('TASK_TRACE_PATH')
  if p:
   with open(p,'a') as f:f.write(json.dumps(data)+'\n')
  print('MILK_TRACE '+json.dumps(data),flush=True)

def score(objects):
 return TRACK.get('outcome_evidence', {}).get('official_score', 0.)

def success(objects):
 return score(objects) == 1.

def demo():
 if os.environ.get('MILK_IDLE')=='1':
  from idle_controller import plan
 else:
  from controller import plan
 return plan(globals())

import tool_feedback
tool_feedback.configure(globals())

milk_fluid.configure(globals())

import contact_evidence
contact_evidence.configure(globals())

import tear_response
tear_response.configure(globals())

import outcome_evidence
outcome_evidence.configure(globals())

import physics_inspection
physics_inspection.configure(globals())

import spill_station
spill_station.configure(globals())
