"""Fit a physically backed flexible protector to a dummy phone."""
from pathlib import Path
import math,os,sys,json
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import labkit
from laminate import *
from measured import Measurements
from yam_arena.demo import Waypoint
from yam_arena.task_file import TABLE_TOP
SCENE='station'
TOP=TABLE_TOP[SCENE]
X0=.195
SCREEN=TOP+.012
OFFSET=np.array([-.009,0.])
O=np.array([X0,0.,SCREEN])
if os.getenv('FILM_CLOSEUP')=='1':
    from inspection_camera import register_native_camera
    register_native_camera(labkit,X0+.070,0.,SCREEN)
labkit.TRANSLUCENT_COLORS[PET]=.55
labkit.CONTACT_OFFSET=.000075
labkit.SOLVER_POS_ITERS=64
labkit.SOLVER_VEL_ITERS=8
labkit.ARTICULATION_SLEEP=0.
labkit.ANGULAR_DAMPING=.018
labkit.MAX_DEPEN_VEL=.08
labkit.MAX_LIN_VEL=2.
labkit.MAX_ANG_VEL=75.
labkit.CONTACT_REPORT.update(('card','film','liner','phone'))
def B(c,s,k):return (c,s,k)
dark=(.07,.09,.12)
phone=labkit.part('phone',(X0+.070,0,TOP),fixed=True,friction=.60,
    boxes=[B((0,0,.006),(.150,.072,.012),dark)])
# A flush surround supports the overhanging tab before pickup.
board=labkit.part('mat',(X0+.070,0,TOP),fixed=True,boxes=[
    B((0,.062,.006),(.199,.051,.012),(.41,.43,.41)),
    B((-.088,0,.006),(.025,.073,.012),(.41,.43,.41)),
    B((.061,-.040,.012+.0065),(.032,.004,.013),(.48,.50,.47))])
CARD_X,CARD_Y=.246,-.145
card=labkit.part('card',(CARD_X,CARD_Y,TOP+.001),mass=.035,friction=.65,
    boxes=[B((0,0,.025),(.004,.066,.046),(.9,.93,.93)),B((0,0,.002),(.007,.066,.004),(.87,.53,.13))])
rack=labkit.part('card_rest',(CARD_X,CARD_Y,TOP),fixed=True,boxes=[
    B((x,.024,.009),(.005,.020,.018),dark) for x in (-.0065,.0065)])
layers=sheet_parts(X0+OFFSET[0],OFFSET[1],SCREEN+.00042)
OBJECTS=[phone,board,card,rack,*layers]
labkit.CONTACT_REPORT.update(label(k,i) for k in ('film','liner') for i in range(N))

def demo():
    from yam_arena import demo as motion
    from yam_arena.urdf import axis_angle_matrix
    if not getattr(motion.grip_rotation,'film_tilt',False):
        original=motion.grip_rotation
        def grip_rotation(yaw=0.):
            if isinstance(yaw,tuple):
                return axis_angle_matrix(np.array([1.,0.,0.]),yaw[1]) @ original(yaw[0])
            return original(yaw)
        grip_rotation.film_tilt=True
        motion.grip_rotation=grip_rotation
    l=[];r=[]
    def pair(t,lp,rp,label):
        for path,(x,y,z,g),yaw in [(l,lp,(0.,-.28)),(r,rp,math.pi/2)]:
            path.append(Waypoint(t,(x,y,TOP+z),g,label,yaw))
    x=X0+PULL_X+OFFSET[0]-.021; y=PULL_Y+OFFSET[1]+.005
    grasp=.012+.042
    pair(1.6,(x,y,.116,0),(CARD_X,CARD_Y-.024,.115,-.017),'approach')
    pair(1.5,(x,y,grasp,0),(CARD_X,CARD_Y-.024,.035,-.017),'enter')
    pair(.8,(x,y,grasp,0),(CARD_X,CARD_Y-.024,.035,0),'grasp')
    pair(1.0,(x,y,grasp+.002,0),(CARD_X,CARD_Y-.024,.108,0),'lift card')
    hold=(X0+PULL_X-.009,PULL_Y+.007,grasp+.001,0)
    align=(X0+.006,-.024,.085,0)
    pair(1.5,hold,align,'slide into alignment')
    pair(1.5,hold,(X0+.006,-.024,.042,0),'tack exposed edge')
    for i in range(1,19):
        p=i*.14/18
        delay=.026-.008*np.clip((p-.065)/.025,0.,1.)
        left=(X0+PULL_X-.009+max(0.,2*(p-delay)),PULL_Y+.007,grasp+.001,0)
        right=(X0+.009+p,-.024,.040+.0015*np.clip((p-.065)/.025,0.,1.),0)
        pair(.51,left,right,'peel and press')
    pair(1.2,(left[0]+.018,PULL_Y+.007,grasp+.014,0),(X0+.145,-.024,.039,0),'finish peeling and smooth end')
    left=(left[0]+.018,PULL_Y+.007,grasp+.014,0)
    pair(1.5,(left[0],PULL_Y,.115,0),(X0+.160,-.145,.120,0),'withdraw')
    return {'left':l,'right':r}

def poses(scene,k):
    from scipy.spatial.transform import Rotation
    raw=np.stack([labkit._np(scene[label(k,i)].data.root_link_pose_w)[0] for i in range(N)])
    rot=Rotation.from_quat(raw[:,3:]).as_matrix()
    xyz=raw[:,:3]-labkit._np(scene.env_origins[0])-O
    return xyz,rot

class Trial:
    def reset(self,*_):
        self.meters=Measurements(); self.step=0; self.credit=0.; self.contacts=None; self.tacked=set()
        self.latched=False; self.front=2; self.breaks=[]; self.progress=[];self.samples=[];self.metrics={}
        self.camera=None;self.station_images=set()
        target=os.environ.get('FILM_TRACE')
        self.path=Path(target) if target else None
        if self.path:
            self.episode=getattr(self,'episode',0)+1
            if self.episode>1:
                self.path=self.path.with_name(f'{self.path.stem}-{self.episode}{self.path.suffix}')
            self.path.write_text('')
    def restore(self,env):
        from isaaclab.sim.utils.stage import get_current_stage
        from pxr import UsdPhysics
        root=env.unwrapped.scene.env_prim_paths[0]
        stage=get_current_stage()
        if not hasattr(self,'tacked'):
            return
        for i in self.tacked:
            stage.RemovePrim(f'{root}/phone/tack{i}')
        pocket=root+'/AirPockets'
        if stage.GetPrimAtPath(pocket):
            stage.RemovePrim(pocket)
        joints=f'{root}/{label("liner",N-1)}'
        UsdPhysics.PrismaticJoint.Get(stage,joints+'/slide_guide_film00').GetLowerLimitAttr().Set(0.)
        for i in range(2,N):
            path=f'{joints}/release_{label("film",i)}_{label("liner",i)}'
            UsdPhysics.Joint.Get(stage,path).GetJointEnabledAttr().Set(True)
            if i>2:
                UsdPhysics.FixedJoint.Get(stage,path+'_interior').GetJointEnabledAttr().Set(True)

    def setup(self,env):
        from isaaclab_physx.physics import PhysxManager
        prefix=env.unwrapped.scene.env_prim_paths[0]+'/'
        phys=PhysxManager.get_physics_sim_view()
        paths=[prefix+label('film',i)+'/body' for i in range(N)]
        def view(a,b):
            return phys.create_rigid_contact_view([a],filter_patterns=[b],max_contact_data_count=160)
        from pxr import Usd, UsdPhysics
        from isaaclab.sim.utils.stage import get_current_stage
        arm=get_current_stage().GetPrimAtPath(prefix+'LeftArm')
        fingers=[str(p.GetPath()) for p in Usd.PrimRange(arm)
                 if p.GetName() in ('tip_left','tip_right','gripper') and p.HasAPI(UsdPhysics.RigidBodyAPI)]
        if len(fingers)!=3:
            raise RuntimeError(f'Cannot identify all three physical gripper links: {fingers}')
        self.draw_sensor=view(prefix+label('liner',0)+'/body',fingers)
        return view(prefix+'card/body',paths+[prefix+label('liner',i)+'/body' for i in range(N)]),view(prefix+'phone/body',paths+[prefix+label('liner',i)+'/body' for i in range(N)])
    def tick(self,env):
        from yam_arena.scene import object_states
        if self.contacts is None:self.contacts=self.setup(env)
        scene=env.unwrapped.scene
        f,fr=poses(scene,'film');l,lr=poses(scene,'liner')
        phys=env.unwrapped.physics_dt
        cf,glass=[np.linalg.norm(v.get_contact_force_matrix(dt=phys).numpy().reshape(-1,3),axis=1) for v in self.contacts]
        draw_load=self.draw_sensor.get_contact_force_matrix(dt=phys).numpy().reshape(-1,3).sum(axis=0)
        s=object_states(env,['card'])['card']
        edge=np.array([s.to_world((0,y,0))-O for y in (-.033,.033)])
        # Pressure-sensitive tack only after real blade load and glass contact.
        from pxr import UsdPhysics,Gf,Sdf
        from isaaclab.sim.utils.stage import get_current_stage
        stage=get_current_stage();root=scene.env_prim_paths[0]
        # Shielded interior bonds are not peel fronts. Once a band physically
        # separates, expose the next finite-strength interface to the load.
        if not self.latched and f[0,0]>=-.00075:
            # A spring detent on the registration tray catches the leading-edge
            # carriage only after a hand has moved the film to the end stop.
            slide=UsdPhysics.PrismaticJoint.Get(stage,f'{root}/{label("liner",N-1)}/slide_guide_film00')
            slide.GetLowerLimitAttr().Set(.008)
            self.latched=True
        if self.front<N:
            i=self.front
            fc=f[i]+fr[i]@[DX/2,0,0]
            lc=l[i]+lr[i]@[DX/2,0,0]
            delta=lc-fc-fr[i]@np.array([0,0,-(T+LINER_T)/2-GAP])
            if np.linalg.norm(delta)>RELEASE_N/K:
                a,b=label('film',i),label('liner',i)
                bond=UsdPhysics.Joint.Get(stage,f'{root}/{label("liner",N-1)}/release_{a}_{b}')
                bond.GetJointEnabledAttr().Set(False)
                self.breaks.append((self.step,i,float(np.linalg.norm(delta)*K))); self.front+=1
                if self.front<N:
                    a,b=label('film',self.front),label('liner',self.front)
                    interior=UsdPhysics.FixedJoint.Get(stage,f'{root}/{label("liner",N-1)}/release_{a}_{b}_interior')
                    interior.GetJointEnabledAttr().Set(False)
        for i in range(N):
            if i in self.tacked or cf[i]<.03 or glass[i]<.004:continue
            if np.max(np.abs(fr[i,:,2]-[0,0,1]))>.13:continue
            if i and np.linalg.norm(l[i]-f[i])<.004:continue
            # The world anchor is the measured pose at the instant of bonding.
            j=UsdPhysics.FixedJoint.Define(stage,root+'/phone/tack'+str(i))
            j.CreateBody0Rel().SetTargets([Sdf.Path(root+'/phone/body')])
            j.CreateBody1Rel().SetTargets([Sdf.Path(root+'/'+label('film',i)+'/body')])
            j.CreateLocalPos0Attr(Gf.Vec3f(*(f[i]+O-[X0+.070,0,TOP])))
            from scipy.spatial.transform import Rotation
            q=Rotation.from_matrix(fr[i]).as_quat()
            j.CreateLocalRot0Attr(Gf.Quatf(q[3],Gf.Vec3f(*q[:3])))
            j.CreateExcludeFromArticulationAttr(True);j.CreateCollisionEnabledAttr(True)
            j.CreateBreakForceAttr(6.);j.CreateBreakTorqueAttr(.08)
            self.tacked.add(i)
        pts=f+fr[:,:,0]*DX/2
        lin=l+lr[:,:,0]*DX/2
        tab=l[0]+lr[0]@np.array([PULL_X,PULL_Y,.01])
        from yam_arena.runtime import tcp_world
        tcp=[(np.asarray(tcp_world(scene[k+'_robot'])).flatten()[:3]-O).tolist() for k in ('left','right')]
        report=self.meters.read(f,fr,l,lr,edge,cf,glass,stage,root+'/AirPockets',O)
        self.credit=report['score']
        if self.path and os.getenv('FILM_CLOSEUP')=='1':
            if self.camera is None and self.step>=130:
                from inspection_camera import InspectionCamera
                self.camera=InspectionCamera(scene,self.path.parent)
            if self.camera:
                for station,gap in zip((35,70,105),report['gaps_mm']):
                    if gap is not None and station not in self.station_images:
                        self.station_images.add(station)
                        self.camera.save(self.step,f'station-{station}mm-side',side=True)
                        self.camera.save(self.step,f'station-{station}mm')
            if self.camera and (self.step%15==0 or self.credit==1.) and (self.meters.clear or self.step>=450):
                self.camera.save(self.step)
        rec=dict(inspection=report,tcp=tcp,step=self.step,film=f.tolist(),film_rot=fr.tolist(),liner=l.tolist(),liner_rot=lr.tolist(),
            edge=edge.tolist(),card_force=cf.tolist(),glass_force=glass.tolist(),tack=sorted(self.tacked),front=self.front,releases=list(self.breaks),paddle_normal_load_N=draw_load.tolist())
        if self.path:
            with self.path.open('a') as o:o.write(json.dumps(rec)+'\n')
        if self.step%20==0:print('CHECK',self.step,'corners',np.round(f[[0,-1]]*1000,1).tolist(),'tab',np.round(1000*tab,1),'edge',np.round(1000*edge.mean(0),1),'peeldist',np.round(1000*np.linalg.norm(l-f,axis=1),0),'tack',sorted(self.tacked),flush=True)
        self.step+=1
trial=Trial()
labkit.on_before_reset(trial.restore)
labkit.on_reset(trial.reset);labkit.on_step(trial.tick)
def success(objects):return trial.credit==1.
def score(objects):return trial.credit
