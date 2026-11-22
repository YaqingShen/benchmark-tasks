"""Replenish the quarter compartment from one sealed paper roll."""
from pathlib import Path
from collections import deque
import sys,math,json,os
from types import SimpleNamespace
import numpy as np
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).parent.resolve()))
import labkit, geometry as g
from packed import COUNT,PAPER_MASS,INITIAL
from packed_runtime import PackedRoll
from scoring import in_compartment, terminal_score,exits_at_open_end
from yam_arena.task_file import TABLE_TOP
SCENE='station'
TOP=TABLE_TOP[SCENE]
def at(xy,z):return np.r_[xy,TOP+z]
labkit.SOLVER_POS_ITERS=96
labkit.SOLVER_VEL_ITERS=12
labkit.MAX_DEPEN_VEL=.16
labkit.CONTACT_OFFSET=.00012
q0=tuple(map(float,Rotation.from_euler('x',-90,degrees=True).as_quat()))
shells=[]
for side,sign in [('left',1),('right',-1)]:
    s=labkit.part(side+'_paper',at(g.START+[0,sign*g.L/2],g.BORE+.0011),mass=PAPER_MASS+COUNT*g.MASS,friction=.72,
                  boxes=g.tube(sign),cylinders=[((0,0,sign*(g.L/2-.00025)),g.BORE+.00065,.0005,g.PAPER)])
    s.meta['rotation_xyzw']=q0
    it=PAPER_MASS*(g.BORE**2/2+g.L**2/12)+sum(g.MASS*(g.R**2/4+g.T**2/12+v*v) for v in INITIAL)
    iz=PAPER_MASS*g.BORE**2+COUNT*g.MASS*g.R**2/2
    labkit.MASS_PROPERTIES[s.name]=((0.,0.,0.),(it,it,iz))
    s.meta['seam_anchor_m']=g.L/2
    s.meta['seam_force_N']=float(os.environ.get('BENCH_SEAM_FORCE','65'))
    s.meta['seam_torque_Nm']=float(os.environ.get('BENCH_SEAM_TORQUE','.145'))
    if sign==-1:s.meta['seam_peer']='left_paper'
    shells.append(s)
coins=[labkit.part(f'quarter_{i:02}',at(g.START+[0,(i-19.5)*(g.T+.000065)],g.BORE+.0011),
          mass=g.MASS,meshes=(g.coin_mesh(labkit),),friction=.25) for i in range(g.COUNT)]
for c in coins:c.meta['rotation_xyzw']=q0
OBJECTS=[labkit.part('till',at(g.TARGET,0),fixed=True,boxes=g.tray()),*shells,*coins]
class Trial:
    def reset(self,*args):
        if args:
            from isaaclab.sim.utils.stage import get_current_stage
            from pxr import UsdPhysics
            usd=get_current_stage()
            path=args[0].unwrapped.scene.env_prim_paths[0]+'/right_paper/paper_seam'
            joint=UsdPhysics.FixedJoint.Get(usd,path)
            assert joint and joint.GetBody1Rel().GetTargets()
            joint.GetJointEnabledAttr().Set(False)
            joint.GetBreakForceAttr().Set(shells[1].meta['seam_force_N'])
            joint.GetBreakTorqueAttr().Set(shells[1].meta['seam_torque_Nm'])
            joint.GetJointEnabledAttr().Set(True)
            self.payload=PackedRoll(args[0],coins,shells)
            labkit.paper_events.clear()
            print('PAPER_READY',str(joint.GetBody0Rel().GetTargets()),str(joint.GetBody1Rel().GetTargets()),flush=True)
        self.step=0;self.view=None;self.score=0.;self.ever_open=False
        self.earned=np.zeros(g.COUNT,bool);self.prev=None;self.hold=0;self.states=None;self.wrists=None
        self.escaped_early=np.zeros(g.COUNT,bool)
        self.spilled=np.zeros(g.COUNT,bool)
        self.rest_history=deque(maxlen=11)
        self.paper_steps=0
        self.last_log=None
        self.fracture_step=None
        self.fracture_world_mm=None
        self.free=np.zeros(g.COUNT,bool)
    def tick(self,env):
        from isaaclab_physx.physics import PhysxManager
        scene=env.unwrapped.scene
        if self.view is None:
            self.view=PhysxManager.get_physics_sim_view().create_rigid_body_view(scene.env_prim_paths[0]+'/quarter_*/body')
            self.order=np.argsort(self.view.prim_paths)
        states=[labkit._part_state(env,s) for s in shells];self.states=states
        ends=np.stack([s.to_world([0,0,-k*g.L/2]) for s,k in zip(states,[1,-1])])
        sep=np.linalg.norm(ends[0]-ends[1])
        angle=math.degrees(math.acos(np.clip(states[0].rotation()[:,2]@states[1].rotation()[:,2],-1,1)))
        if labkit.paper_events and not self.ever_open:
            self.ever_open=True
            self.fracture_step=self.step
            self.fracture_world_mm=np.round((ends.mean(axis=0)-[0,0,TOP])*1000,2).tolist()
        loose=self.payload.update(states,self.ever_open,sep)
        transforms=self.view.get_transforms().numpy()[self.order]
        pos=transforms[:,:3]-labkit._np(scene.env_origins[0])
        vel=self.view.get_velocities().numpy()[self.order]
        x,y,z,w=transforms[:,3:].T
        axes=np.stack([2*(x*z+w*y),2*(y*z-w*x),1-2*(x*x+y*y)],axis=1)
        inside=in_compartment(pos,axes,TOP)
        locs=[(pos-state.pos)@state.rotation() for state in states]
        if self.ever_open and self.prev is not None:
            for i,k in enumerate((1,-1)):
                self.free |= exits_at_open_end(self.prev[i],locs[i],-k)
        self.prev=locs
        in_paper=np.logical_or.reduce([(abs(v[:,2])<g.L/2+.002)&(np.linalg.norm(v[:,:2],axis=1)<g.BORE+.001) for v in locs])
        if not self.ever_open or self.step==0:self.escaped_early |= ~in_paper
        # A visible spill counts when a loose coin comes to rest outside the
        # receiving cup. Passing above another compartment is not a spill.
        # Position and tilt can be read from calibrated top and side recordings.
        # Eleven samples at 20 Hz span half a second. A coin spinning flat
        # in place changes neither its footprint nor the height of the stack.
        self.rest_history.append((pos.copy(),axes.copy()))
        settled=np.zeros(g.COUNT,bool)
        if len(self.rest_history)==self.rest_history.maxlen:
            coordinates=np.stack([item[0] for item in self.rest_history])
            normals=np.stack([item[1] for item in self.rest_history])
            center_motion=np.linalg.norm(np.ptp(coordinates,axis=0),axis=1)
            rim_motion=g.R*np.linalg.norm(np.ptp(normals,axis=0),axis=1)
            settled=(center_motion+rim_motion)<=.001
        # A coin leaning inside the cup at its divider is not a spill. It
        # earns no delivery credit until its whole footprint clears the rim.
        # Mark spills only when the center rests beyond the outer divider face.
        outside_center=(np.abs(pos[:,:2]-g.TARGET)>g.CELL/2+.003).any(axis=1)
        self.spilled |= (loose & ~in_paper & outside_center & settled & (pos[:,2]<TOP+g.RIM+.001))
        self.earned=(self.ever_open & loose & self.free & ~self.escaped_early & ~self.spilled & ~in_paper)
        from yam_arena import station
        from yam_arena.runtime import link_poses
        matrices=[link_poses(scene[a+'_robot'])['gripper'] for a in ('left','right')]
        tips=[m[:3,3]+m[:3,:3]@[0,0,.134] for m in matrices]
        self.wrists=list(zip(tips,[m[:3,:3] for m in matrices]))
        def paper_parked(state,side):
            low,high=g.paper_bounds(state.pos,state.rotation(),side)
            on_table=(TOP-.004<=low[2]<=TOP+.003 and high[2]<TOP+.050 and
                      np.all(low[:2]>np.array(station.TABLE_CENTER[:2])-np.array(station.TABLE_SIZE[:2])/2) and
                      np.all(high[:2]<np.array(station.TABLE_CENTER[:2])+np.array(station.TABLE_SIZE[:2])/2))
            clear_tray=(high[0]<g.TARGET[0]-g.CELL[0]/2-.003 or low[0]>g.TARGET[0]+g.CELL[0]/2+.003 or
                        high[1]<g.TARGET[1]-1.5*g.PITCH or low[1]>g.TARGET[1]+1.5*g.PITCH)
            speed=np.linalg.norm(state.lin_vel)
            hand_distance=min(np.linalg.norm(t-state.pos) for t in tips)
            if self.step%20==0 and self.step>=800:
                print('PAPER_SUPPORT',self.step,side,'base_mm',round((low[2]-TOP)*1000,2),
                      'box_base_mm',round((state.low[2]-TOP)*1000,2),'speed_mm_s',round(speed*1000,2),
                      'hand_mm',round(hand_distance*1000,1),'outside_till',clear_tray,flush=True)
            return on_table and clear_tray and speed<.02 and hand_distance>.050
        self.paper_steps=self.paper_steps+1 if all([paper_parked(state,side) for state,side in zip(states,(1,-1))]) else 0
        self.score=terminal_score(self.earned&inside,settled,self.paper_steps>=10)
        row=dict(step=self.step,native_loose=int(loose.sum()),payload_g=round(sum(b.mass-PAPER_MASS for b in self.payload.groups)*1000,2),broken=self.ever_open,fracture_step=self.fracture_step,fracture_world_mm=self.fracture_world_mm,contained=int(in_paper.sum()),free=int(self.free.sum()),escaped_closed=int(self.escaped_early.sum()),
                 spilled=int(self.spilled.sum()),in_bin=int(inside.sum()),delivered=int((self.earned&inside&settled).sum()),
                 parked_steps=self.paper_steps,score=self.score)
        if os.environ.get('QUARTER_TRACE'):
            with open(os.environ['QUARTER_TRACE'],'a') as f:print(json.dumps(row),file=f)
        if self.step%10==0:
            from yam_arena.runtime import link_poses
            t=[link_poses(scene[a+'_robot'])['gripper'] for a in ['left','right']]
            print('CONTENTS',self.step,[int(((abs(loc[:,2])<g.L/2+.0004)&(np.linalg.norm(loc[:,:2],axis=1)<.006)).sum()) for loc in locs],
                  'axis_z',[round(float(s.rotation()[2,2]),2) for s in states],'lost',int((pos[:,2]<TOP-.015).sum()),'coin-centre-range',(np.min(pos,axis=0)-[0,0,TOP]).round(3).tolist(),(np.max(pos,axis=0)-[0,0,TOP]).round(3).tolist(),flush=True)
            if self.step%100==0 or self.step==890:
                outside=np.flatnonzero(loose & ~inside)
                print('OUTSIDE_FOOTPRINT',self.step,[(int(i),((pos[i]-[0,0,TOP])*1000).round(2).tolist(),axes[i].round(3).tolist()) for i in outside],flush=True)
            print('PAD', self.step,[(m[:3,3]+m[:3,:3]@[0,0,.134]-[0,0,TOP]).round(3).tolist() for m in t],flush=True)
            print('SEAM',self.step,'broken',self.ever_open,'gap_mm',round(sep*1000,1),'angle',round(angle,1),'bin',int(inside.sum()),'eligible',int(self.earned.sum()),'held',[(s.pos-[0,0,TOP]).round(3).tolist() for s in states],'score',round(self.score,3),flush=True)
        if os.environ.get('QUARTER_TRACE') and self.step in (0,65,152,268,450,485,520,755,800):
            from PIL import Image
            folder=Path(os.environ['QUARTER_TRACE']).parent
            for camera in ('left_cam','right_cam','top_cam'):
                frame=scene[camera].data.output['rgb']
                frame=frame.torch if hasattr(frame,'torch') else frame
                Image.fromarray(frame[0,...,:3].cpu().numpy()).save(folder/f'{camera}-{self.step:03}.png')
        self.step+=1
trial=Trial();trial.reset()
labkit.on_reset(trial.reset);labkit.on_step(trial.tick)
def score(objects):return trial.score
def success(objects):return trial.score>=1.
def demo():
    from yam_arena import demo as dm,station,ik
    from yam_arena.constants import READY
    orig=dm.grip_rotation
    if not getattr(orig,'seam',False):
        def grip(yaw):return yaw.r if hasattr(yaw,'r') else orig(yaw)
        grip.seam=True;dm.grip_rotation=grip
    if not getattr(dm.ArmPlan.target,'roll_feedback',False):
        original=dm.ArmPlan.target
        def corrected(plan,t):
            pos,rot,opening,label=original(plan,t)
            arm=0 if plan.base_pos[1]>0 else 1
            if not label.startswith('drain'+str(arm)+' ') or trial.states is None or trial.wrists is None or not trial.ever_open:
                return pos,rot,opening,label
            progress=float(label.split()[1])
            sign=[1,-1][arm]
            paper=trial.states[arm]
            tcp,jaw=trial.wrists[arm]
            if np.linalg.norm(tcp-paper.pos)>.049:return pos,rot,opening,label
            edge=paper.to_world([0,0,-sign*g.L/2])
            out=sign*paper.rotation()[:,2]
            goal=np.array([0.,sign*math.cos(math.radians(progress)),math.sin(math.radians(progress))])
            axis=np.cross(out,goal)
            sine=np.linalg.norm(axis)
            angle=min(math.atan2(sine,np.clip(out@goal,-1,1)),math.radians(2.7))
            adjust=Rotation.from_rotvec(axis/max(sine,1.e-8)*angle).as_matrix()
            # Pour nearer the floor and give each stream its own landing area.
            empty=trial.payload.groups[arm].next==g.COUNT//2
            # The second sleeve swept two deposited coins off the pile during exit.
            # Give the right-hand mouth clearance above the accumulated change.
            working_height=.023 if arm==0 else .043
            height=.085 if empty else working_height+(pos[2]-TOP-.069)*.35
            # Put the second stream farther from the mound left by the first stream.
            lateral=.011 if arm==0 else .023
            desired=at([[.346,.353][arm],sign*lateral],height)
            offset=desired-edge
            shift=offset*min(1.,.0045/max(np.linalg.norm(offset),1.e-8))
            return edge+shift+adjust@(tcp-edge),adjust@jaw,opening,label
        corrected.roll_feedback=True
        dm.ArmPlan.target=corrected
    arms=['left','right'];signs=[1,-1]
    angles=[float(os.getenv('SEAM_L','-90')),float(os.getenv('SEAM_R','90'))]
    lean=float(os.getenv('SEAM_LEAN','50'))
    rr=[Rotation.from_euler('x',-k*lean,degrees=True).as_matrix()@dm.grip_rotation(math.radians(v)) for v,k in zip(angles,signs)]
    grip=float(os.environ.get('BENCH_GRIP','-.009'))
    plans={a:[] for a in arms}
    def stage(sec,name,xy,z,dy,tilt,opening):
        for a,k,r in zip(arms,signs,rr):
            q=Rotation.from_euler('x',k*tilt,degrees=True).as_matrix()
            if os.environ.get('BENCH_ONE_HAND') and a=='right':
                pos,rot=ik.tcp_pose(np.array(READY),station.BASES[a])
                plans[a].append(dm.Waypoint(sec,tuple(pos),-.032,'stand clear',SimpleNamespace(r=rot)))
                continue
            # One hand carries the closed roll. The second approaches with
            # open pads only after the paper is resting on the breaker.
            target_y=k*dy
            target_z=z
            finger=opening
            if a=='right' and name in ('approach roll','straddle ends','grip ends','lift as one roll','carry over the drawer','line up over the steel edge'):
                target_y-=.045
                target_z+=.045
                finger=-.040
            if a=='right' and name=='contact the edge':
                finger=-.032
            plans[a].append(dm.Waypoint(sec,tuple(at(xy+[0,target_y],target_z)-.005*(q@r)[:,2]),finger,name,SimpleNamespace(r=q@r)))
    stage(1.6,'approach roll',g.START,.106,.025,0,-.028)
    stage(1.1,'straddle ends',g.START,.015,.028,0,-.023)
    stage(.8,'grip ends',g.START,.015,.028,0,grip)
    stage(1.,'lift as one roll',g.START,.082,.028,0,grip)
    stage(1.7,'carry over the drawer',g.TARGET,.085,.028,0,grip)
    edge=g.TARGET+[.029,0]
    stage(.85,'line up over the steel edge',edge,.085,.028,0,grip)
    stage(1.35,'contact the edge',edge,.062,.028,0,grip)
    stage(.85,'support the second end at the edge',edge,.061,.028,0,grip)
    stage(.75,'load the seam against the edge',edge,.045,.028,-10,grip)
    stage(.5,'open the split',edge,.051,.037,-12,grip)
    def independent(sec,label,left,right):
        for a,k,r,(x,y,z,tilt,opening) in zip(arms,signs,rr,[left,right]):
            q=Rotation.from_euler('x',k*tilt,degrees=True).as_matrix()
            pos=at([x,y],z)-.005*(q@r)[:,2]
            plans[a].append(dm.Waypoint(sec,tuple(pos),opening,label,SimpleNamespace(r=q@r)))
    # Move each opened end away from the edge. The inner mouths face upward
    # while carrying so the second hand can wait with its quarters.
    left=(.336,.045,.087,-15,grip)
    right=(.382,-.046,.087,-15,grip)
    independent(1.6,'lift the separated ends',left,right)
    for i in (0,1):
        k=signs[i]
        def active(sec,label,cmd):
            nonlocal left,right
            if i==0:left=cmd
            else:right=cmd
            independent(sec,label,left,right)
        active(1.2,'position the mouth over the coin cup',(.331,k*.027,.072,-12,grip))
        for deg in (0,12,25,38,50,62,74):
            active(.65,'drain'+str(i)+' '+str(deg),(.331,k*.029,.069,deg,grip))
        for dz in (.006,-.004,.005,-.003):
            active(.33,'drain'+str(i)+' 80',(.331,k*.029,.069+dz,74,grip))
        active(1.8,'drain'+str(i)+' 80',(.331,k*.029,.069,74,grip))
        if i==1:
            # Clear the rim vertically before rotating or moving across the drawer.
            active(1.4,'raise right sleeve above rim',(.350,-.023,.123,68,grip))
        for deg in (50,30,10,0):
            active(.38,'level the empty end',(.330,k*.042,.120 if i==1 else .080,deg,grip))
        active(1.1,'carry paper to counter',(.220,k*.078,.077,0,grip))
        active(.8,'place paper on the counter',(.220,k*.078,.016,0,grip))
        active(.85,'open the fingers with the paper supported',(.220,k*.078,.016,0,-.044))
        active(1.1,'slide clear of the open paper mouth',(.220,k*.010,.018,0,-.044))
        active(.8,'clear the drawer',(.221,k*.14,.12,0,-.033))
    for a in arms:
        p,r=ik.tcp_pose(np.array(READY),station.BASES[a])
        plans[a].append(dm.Waypoint(1.2,tuple(p),-.032,'withdraw',SimpleNamespace(r=r)))
    return plans
