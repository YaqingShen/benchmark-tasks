"""Robot TCP waypoints only; prototype inner-seal manipulation."""
import math
import numpy as np

def plan(ns):
 from yam_arena import ik,demo as dm
 from yam_arena.constants import READY,FINGER_OPEN,FINGER_CLOSED,LEFT_BASE_POS,RIGHT_BASE_POS
 from yam_arena.urdf import quat_xyzw_to_matrix
 class Orient:
  def __init__(self,r):self.rot=np.asarray(r)
 old=dm.grip_rotation
 if not getattr(old,'_milk',False):
  def grip_rotation(x=0.):return x.rot if hasattr(x,'rot') else old(x)
  grip_rotation._milk=True;dm.grip_rotation=grip_rotation
 down=quat_xyzw_to_matrix(np.asarray(dm.DOWN_GRIP));S=ns['S'];N=ns['N'];P=ns['P'].copy();P[2]=ns['TOP'];C=ns['C'];sp=P+C
 plans={'left':[],'right':[]};clock=dict(left=0.,right=0.)
 bases={'left':LEFT_BASE_POS,'right':RIGHT_BASE_POS}
 def wp(a,d,p,f,R,label):plans[a].append(dm.Waypoint(d,tuple(p),f,label,Orient(R)));clock[a]+=d
 def wait(a,t,label='hold'):
  if t<=clock[a]:return
  if plans[a]:x=plans[a][-1];wp(a,t-clock[a],x.pos,x.finger,x.yaw.rot,label)
  else:
   p,R=ik.tcp_pose(np.array(READY),np.array(bases[a]));wp(a,t,p,FINGER_OPEN,R,'wait')
 L='left';R='right';open=FINGER_OPEN;closed=FINGER_CLOSED
 rg=S@down
 # Pinch the two exterior slopes at the midpoint of the top seam.
 angle=math.radians(50)
 br=np.array([[-1.,0.,0.],[0.,math.cos(angle),-math.sin(angle)],[0.,-math.sin(angle),-math.cos(angle)]])
 brace=P+np.array([0.,0.,ns['PEAK']-.0105])-.007*br[:,2]
 # Pick up the dimensioned hook by its handle; the tip stays clear of the table.
 h=ns['HOOK_POS'].copy();g=ns['HOOK_GRIP'];pick=h+g-.007*down[:,2]
 wp(L,1.5,pick+[0,0,.075],open,down,'above hook handle')
 wp(L,1.5,pick,open,down,'straddle hook handle')
 wp(L,.7,pick,closed,down,'grip hook handle')
 wp(L,1.2,pick+[0,0,.07],closed,down,'lift hook')
 # The handle is nearly in line with the pull to minimize wrist moment.
 off=ns['HOOK_TOE']-g
 def tcp(tip):return tip-S@off-.007*rg[:,2]
 sz=ns['SEAL_Z'];ringz=sz+.004
 tip=sp+S@np.array([0.,.002,ringz-.0016])
 wp(L,2.5,tcp(tip+.07*N),closed,rg,'orient above closed ring')
 wp(L,1.6,tcp(tip+.012*N),closed,rg,'approach ring opening')
 wp(L,2.6,tcp(tip),closed,rg,'insert hook through ring')
 tip=sp+S@np.array([ns['HOOK_RING_U']-.0115,ns['HOOK_RING_Y'],ringz-.0016])
 wp(L,2.5,tcp(tip),closed,rg,'engage underside of ring')
 wp(L,1.,tcp(tip),closed,rg,'engage underside of ring')
 for deg in range(10,101,10):
  a=math.radians(deg)
  contact_y=ns['HOOK_RING_Y']
  contact_u=.0115+math.sqrt(.0115**2-contact_y**2)
  tip=sp+S@np.array([-.0115+contact_u*math.cos(a),contact_y,ringz+contact_u*math.sin(a)-.0016])
  wp(L,.35,tcp(tip),closed,rg,'raise pull ring')
 # The open hook only lifts the ring. The right fingers take its near arc.
 # Jaw commands are per finger. YAM opens towards negative joint travel.
 from inner_receiver import receive_frame
 from scipy.spatial.transform import Rotation
 a=math.radians(76)
 nominal_ring=Rotation.from_rotvec([0,-a,0]).as_matrix()@S
 take,entry,finger_tcp=receive_frame(P+ns['anchor'],nominal_ring,np.array([1.,0.,0.]))
 approach=take[:,2]
 wait(R,clock[L]+.4)
 wp(R,1.8,entry-.045*approach+.004*N,-.018,take,'approach raised ring from right')
 wp(R,1.2,entry-.012*approach,-.016,take,'align right fingers with ring')
 wp(R,2.0,entry,-.012,take,'straddle near ring wire')
 wp(R,1.4,finger_tcp,closed,take,'pinch ring directly')
 wp(R,1.6,finger_tcp,closed,take,'confirm direct finger capture')
 wp(R,.7,finger_tcp-.012*approach+.002*N,-.016,take,'retry right grip from clear')
 wp(R,1.2,finger_tcp+.002*N,-.014,take,'reseat right fingers')
 wp(R,1.2,finger_tcp+.002*N,closed,take,'close right fingers again')
 wp(R,.7,finger_tcp+.002*N,closed,take,'confirm second grip')
 wait(L,clock[R],'hold raised ring')
 # Ease off the wire, withdraw backwards, then return the hook to its posts.
 wp(L,1.25,tcp(tip-.004*N-.006*S[:,0]-.004*S[:,1]),closed,rg,'unload open hook')
 wp(L,1.5,tcp(tip-.012*S[:,0]),closed,rg,'back hook out of ring')
 wp(L,1.4,tcp(tip+.045*N+.065*S[:,1]),closed,rg,'carry clear of held ring')
 wp(L,2.,pick+[0,0,.075],closed,down,'return hook above rest')
 wp(L,1.5,pick,closed,down,'set hook on clean posts')
 wp(L,.75,pick,open,down,'release parked hook')
 wp(L,1.1,pick+[0,0,.08],open,down,'clear hook rest')
 wp(L,1.8,brace+[0,.025,.045],-.009,br,'pre-shape left hand above carton center')
 wp(L,1.3,brace+[0,0,.007],-.009,br,'straddle top seam at midpoint')
 wp(L,.8,brace+[0,0,.007],closed,br,'pinch center of sealed fin')
 wp(L,.65,brace,closed,br,'seat on both carton folds')
 wp(L,1.,brace,closed,br,'steady carton before pull')
 wait(R,clock[L],'keep direct grip')
 wp(R,2.0,finger_tcp,closed,take,'seat inside finger')
 for dist in np.arange(.0025,.0851,.0025):
  wp(R,.47,finger_tcp+dist*N,closed,take,'draw seal with direct grip')
 wp(R,.8,finger_tcp+.085*N+[0.,0.,.01],closed,take,'lift seal clear')
 target=ns['TRAY_POS']+[-.020,0.,.040]
 wp(R,2.6,target+[0.,0.,.080],closed,take,'carry seal to tray')
 wp(R,1.4,target+[0.,0.,.07],closed,down,'turn wrist above tray')
 wp(R,1.4,target,closed,down,'lower held seal')
 wp(R,.85,target,open,down,'release seal in tray')
 wp(R,1.2,target+[0,0,.08],open,down,'withdraw right hand')
 wait(L,clock[R])
 wp(L,.45,brace+[0,0,.009],-.0075,br,'relieve carton grasp')
 wp(L,.8,brace+[0,0,.045],-.009,br,'withdraw above seam')
 wp(L,.75,brace+[0,.045,.065],open,br,'clear carton top')
 for arm in [L,R]:
  p,r=ik.tcp_pose(np.array(READY),np.array(bases[arm]));wp(arm,2.,p,open,r,'hands clear')
 end=max(clock.values())+2.
 for arm in [L,R]:wait(arm,end)
 return plans
