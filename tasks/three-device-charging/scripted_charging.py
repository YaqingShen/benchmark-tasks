"""Scripted physical feasibility controller; privileged poses, native joint actions.

This controller is never substituted for Dream-1. Object poses are read only.
"""
import json,math
import numpy as np
from charging_physics import array,transform,PARAMS

class Driver:
    def __init__(self,world):
        self.w=world;self.ik=world.ik;self.action=world.joints();self.max_ik_error=0.
        self.home=self.action.copy()
        self.events=[]

    def pose(self,side):
        o=0 if side=='left' else 7
        return self.ik.tcp_pose(self.w.joints()[o:o+6],self.w.station.BASES[side])

    def move(self,side,point,rotation=None,gap=None,duration=2.,label='move',stop_when=None):
        o=0 if side=='left' else 7;q=self.action[o:o+6].copy()
        p0,r0=self.ik.tcp_pose(q,self.w.station.BASES[side]);p1=np.array(point);r1=r0 if rotation is None else rotation
        g0=self.action[o+6];g1=g0 if gap is None else gap/.0939
        print(label,flush=True)
        for i in range(max(1,round(duration*20))):
            s=self.ik.smoothstep((i+1)/(duration*20));pos=p0*(1-s)+p1*s;r=self.ik.slerp_matrix(r0,r1,s)
            q,ep,er=self.ik.solve(pos,self.ik.matrix_to_quat_xyzw(r),q_init=q,base_pos=self.w.station.BASES[side],iters=160,damping=.018,tol=3e-5)
            self.max_ik_error=max(self.max_ik_error,ep)
            if ep>.0025 or er>.025:raise RuntimeError(f'IK infeasible during {label}: {ep:.5f} m {er:.4f} rad')
            self.action[o:o+6]=q;self.action[o+6]=g0*(1-s)+g1*s
            self.w.step(self.action,label)
            if stop_when is not None and stop_when():break
        self.w.snapshot(f'{self.w.steps:05d}-'+label.replace(' ','-'))
        self.events.append({'step':self.w.steps,'phase':label,'score':self.w.score()})

    def hold(self,seconds,label='hold'):
        for _ in range(round(seconds*20)):self.w.step(self.action,label)

    def stabilize_stand(self):
        theta=.45;z=np.array([np.sin(theta),0,-np.cos(theta)]);x=np.array([0,-1.,0]);r=np.column_stack([x,np.cross(z,x),z])
        stand=self.w.coupling.poses()['belkin'];p=stand[:3,3]+np.array([-.025,-.041,.012])-.013*z+[0,0,-.004]
        self.move('right',[.28,-.16,.92],r,0.,2.,'prepare stand support')
        self.move('right',p+[0,0,.09],r,0.,2.5,'right hand above base')
        self.move('right',p,r,0.,2.,'right hand supports base')
        self.hold(.5,'settle stand support')
        force=0.
        for i in range(40):
            force=sum(float(np.linalg.norm(array(self.w.base.scene[f'hand_{j}_contacts'].data.normal_force_matrix_w)[0,0,19])) for j in (2,3))
            if force>=9.:break
            pc,rc=self.ik.tcp_pose(self.action[7:13],self.w.station.BASES['right'])
            q,ep,er=self.ik.solve(pc+[0,0,-.0005],self.ik.matrix_to_quat_xyzw(rc),self.action[7:13],self.w.station.BASES['right'],iters=160,damping=.018)
            if ep>.002 or er>.02:raise RuntimeError('Stand stabilization IK infeasible')
            self.action[7:13]=q;self.w.step(self.action,'regulate stand support')
        print('Stand support contact force',force,flush=True)
        if force<5.:raise RuntimeError('Right hand did not establish support contact')

    def open_gripper(self,side,seconds=1.,target_gap=.0939):
        o=0 if side=='left' else 7;start=self.action[o+6];end=target_gap/.0939
        for i in range(round(seconds*20)):
            self.action[o+6]=start+(end-start)*self.ik.smoothstep((i+1)/(seconds*20))
            self.w.step(self.action,'release '+side)

    def servo_object(self,name,side,make_target,seconds=3.,stop_when_charging=False):
        # Compensate measured native tracking error with robot joint targets.
        # No body pose, body velocity or attachment state is written.
        o=0 if side=='left' else 7
        for i in range(round(seconds*20)):
            if stop_when_charging and self.w.coupling.charging[name]:
                state=self.w.coupling.last[name]
                if state['offset_m']<.0015 and state['angle_deg']<2:break
            actual=self.w.coupling.poses()[name];target=make_target()
            correction=(target[:3,3]-actual[:3,3])*.25
            correction*=min(1.,.002/max(np.linalg.norm(correction),1e-12))
            dr=self.ik.slerp_matrix(np.eye(3),target[:3,:3]@actual[:3,:3].T,.20)
            pc,rc=self.ik.tcp_pose(self.action[o:o+6],self.w.station.BASES[side])
            q,ep,er=self.ik.solve(pc+correction,self.ik.matrix_to_quat_xyzw(dr@rc),self.action[o:o+6],self.w.station.BASES[side],iters=160,damping=.018,tol=3e-5)
            if ep>.0025 or er>.025:raise RuntimeError('Feedback alignment exceeds native workspace')
            self.action[o:o+6]=q;self.w.step(self.action,'feedback align '+name)
        self.w.snapshot(f'{self.w.steps:05d}-feedback-'+name)

    def grip_rotation(self,name,side,point):
        obj=self.w.coupling.poses()[name];best=None;o=0 if side=='left' else 7
        if name=='airpods_pro':
            # Grasp the 45.2 mm short dimension from a slanted approach.
            # This keeps the wrist inside its native range at the low base pad.
            c,s=np.cos(.65),np.sin(.65)
            return obj[:3,:3]@np.array([[0,1.,0],[-1.,0,0],[0,0,1.]])@np.diag([1,-1,-1])@np.array([[1.,0,0],[0,c,-s],[0,s,c]])
        stand=self.w.coupling.poses()['belkin'];param=PARAMS[name];normal=stand[:3,:3]@param['normal']
        if name=='airpods_pro':target_r=stand[:3,:3]@np.array([[0,-1.,0],[1.,0,0],[0,0,1.]])
        else:
            axis=stand[:3,:3]@np.array([0,-1.,0]);target_r=np.column_stack([axis,np.cross(normal,axis),normal])
            if name=='iphone17':target_r=target_r@np.array([[0,-1.,0],[1.,0,0],[0,0,1.]])
        goal=stand[:3,3]+stand[:3,:3]@np.array(param['center'])+normal*(param['puck_half']+param['back']+.025)
        goal+=target_r@obj[:3,:3].T@(point-np.array([0,0,.09])-obj[:3,3])
        for diag in ([1,-1,-1],[-1,1,-1]):
            r=obj[:3,:3]@np.diag(diag)
            q,ep,er=self.ik.solve(point,self.ik.matrix_to_quat_xyzw(r),self.w.joints()[o:o+6],self.w.station.BASES[side],iters=500,damping=.02)
            end_r=target_r@obj[:3,:3].T@r
            qe,ee,re=self.ik.solve(goal,self.ik.matrix_to_quat_xyzw(end_r),q,self.w.station.BASES[side],iters=700,damping=.02)
            cost=ep*100+er+ee*100+re+np.linalg.norm(q-self.w.joints()[o:o+6])*.001
            print('Grasp candidate',name,diag,'initial errors',ep,er,'seating errors',ee,re,'end joints',qe.tolist(),flush=True)
            if best is None or cost<best[0]:best=(cost,r)
        return best[1]

    def pickup(self,name,side):
        obj=self.w.coupling.poses()[name];initial=obj[:3,3].copy()
        point=obj[:3,3].copy();point[2]=max(point[2]+.004,self.w.station.TABLE_TOP+(.020 if name=='airpods_pro' else .016))
        if name=='apple_watch':point+=obj[:3,1]*(-.009)
        r=self.grip_rotation(name,side,point+[0,0,.09])
        width={'iphone17':.0715,'apple_watch':.036,'airpods_pro':.0452}[name]
        self.move(side,point+[0,0,.09],r,.0939,2.5,'above '+name)
        self.move(side,point,r,.0939,2.,'descend '+name)
        self.move(side,point,r,width-.003,1.,'grasp '+name)
        self.hold(.4,'settle grasp '+name)
        self.move(side,point+[0,0,.10],r,width-.003,2.5,'lift '+name)
        self.hold(.5,'hold lifted '+name)
        actual=self.w.coupling.poses()[name]
        if actual[2,3]-initial[2]<.065:raise RuntimeError(name+' did not physically lift')
        p,r=self.pose(side);hand=np.eye(4);hand[:3,:3]=r;hand[:3,3]=p
        relative=np.linalg.inv(hand)@actual
        return relative,width-.003

    def place(self,name,side,relative,gap):
        stand=self.w.coupling.poses()['belkin'];param=PARAMS[name];normal=stand[:3,:3]@param['normal']
        target=np.eye(4)
        if name=='airpods_pro':target[:3,:3]=stand[:3,:3]
        else:
            x=stand[:3,:3]@np.array([0,-1.,0]);target[:3,:3]=np.column_stack([x,np.cross(normal,x),normal])
            if name=='iphone17':target[:3,:3]=target[:3,:3]@np.array([[0,-1.,0],[1.,0,0],[0,0,1.]])
        target[:3,3]=stand[:3,3]+stand[:3,:3]@np.array(param['center'])+normal*(param['puck_half']+param['back']+.001)
        inverse=np.linalg.inv(relative)
        def hand_at(offset):
            t=target.copy();t[:3,3]+=offset;h=t@inverse;return h[:3,3],h[:3,:3]
        p,r=hand_at(normal*.08+np.array([0,0,.04]));self.move(side,p,r,gap,3.,'orient '+name)
        p,r=hand_at(normal*.025);self.move(side,p,r,gap,2.,'approach pad '+name)
        def target_now(standoff):
            now=self.w.coupling.poses()['belkin'];delta=now@np.linalg.inv(stand)
            t=delta@target;t[:3,3]+=now[:3,:3]@np.array(param['normal'])*standoff
            return t
        self.servo_object(name,side,lambda:target_now(.025),2.)
        # Keep the tracking compensation learned at the clear standoff.
        # Reusing the pickup transform here would undo it at the pad surface.
        o=0 if side=='left' else 7
        cp,cr=self.ik.tcp_pose(self.action[o:o+6],self.w.station.BASES[side])
        command=np.eye(4);command[:3,3]=cp;command[:3,:3]=cr
        inverse=np.linalg.inv(np.linalg.inv(command)@self.w.coupling.poses()[name])
        def watch_captured():
            state=self.w.coupling.last[name]
            return -.0006<state['gap_m']<.0025 and state['offset_m']<.0018 and state['angle_deg']<2 and state['magnetic_force_n']>.7
        p,r=hand_at(np.zeros(3));self.move(side,p,r,gap,2.5,'seat '+name,stop_when=watch_captured if name=='apple_watch' else None)
        # Release only as far as needed, then withdraw before opening fully.
        # A wide watch release would hit the adjacent mounted phone.
        release_gap={'apple_watch':.039,'airpods_pro':.065}.get(name,.0939)
        if name=='apple_watch' and watch_captured():
            self.open_gripper(side,seconds=.35,target_gap=release_gap)
            self.hold(.6,'magnetic capture '+name)
        else:
            self.servo_object(name,side,lambda:target_now(0.),3.,stop_when_charging=True)
            self.hold(.1,'settle seating '+name)
            self.open_gripper(side,target_gap=release_gap)
        clear=stand[:3,:3]@np.array([-.10,0,.025]) if name=='airpods_pro' else normal*.10
        p,r=hand_at(clear);self.move(side,p,r,release_gap,2.,'clear '+name)
        self.hold(.6,'check '+name)
        if not self.w.coupling.charging[name]:raise RuntimeError(name+' is not charging after physical release: '+str(self.w.coupling.last[name]))

def demonstrate(world,pickup_only=False,only=None,support_stand=False):
    d=Driver(world);world.metadata.update(scripted=True,controller='privileged pose alignment, native joint targets, RGB cue verification',dream1_run=False)
    (world.output/'metadata.json').write_text(json.dumps(world.metadata,indent=2))
    d.hold(2.,'initial settling');world.initial_stand=world.coupling.poses()['belkin'].copy();world.max_shift=world.max_rotation=0.
    world.snapshot('setup')
    if support_stand and not pickup_only and only is None:d.stabilize_stand()
    # Place the low case first, before a mounted phone obstructs the wrist.
    for name,side in [('airpods_pro','right'),('iphone17','left'),('apple_watch','right')]:
        if only is not None and name!=only:continue
        relative,gap=d.pickup(name,side)
        if pickup_only:break
        d.place(name,side,relative,gap)
        # Clear the other device and restore an original wrist-camera overview.
        o=0 if side=='left' else 7
        p,r=world.ik.tcp_pose(d.home[o:o+6],world.station.BASES[side])
        d.move(side,[.20,.25 if side=='left' else -.25,1.04],r,.0939,2.,'retreat '+side)
        if support_stand and name=='iphone17' and only is None:
            p,r=d.pose('right');d.move('right',p+[0,0,.11],r,0.,2.,'release stand support')
            d.open_gripper('right')
    if not pickup_only and only is None:
        d.hold(7.,'verify all three charging with hands released')
        if not world.score()['success']:raise RuntimeError('Full task conditions not met: '+str(world.score()))
    world.snapshot('final');(world.output/'controller.json').write_text(json.dumps({'scripted_not_dream1':True,'pickup_only':pickup_only,'only_device':only,'support_stand':support_stand,'max_ik_error_m':d.max_ik_error,'events':d.events},indent=2))
