"""Finite paired magnetic wrenches and contact-gated charging approximation.

This is an uncalibrated mechanics/electrical proxy, not an electromagnetic or
battery simulator. It never changes an object's pose or creates an attachment.
"""
import math
import numpy as np

DEVICES=('iphone17','apple_watch','airpods_pro')
PARAMS={
 'iphone17':dict(center=[.023,.050,.096],normal=[-.8660254038,0,.5],puck_half=.0035,back=.00435,force=8.,decay=.004,sigma=.010,torque=.05),
 'apple_watch':dict(center=[.028,-.050,.092],normal=[-.8660254038,0,.5],puck_half=.003,back=.0057,force=2.,decay=.003,sigma=.006,torque=.012),
 'airpods_pro':dict(center=[-.012,0,.010],normal=[0,0,1],puck_half=0.,back=.01085,force=0.,decay=.003,sigma=.010,torque=0.),
}

def array(v):
    v=v.torch if hasattr(v,'torch') else v
    return v.detach().cpu().numpy() if hasattr(v,'detach') else np.asarray(v)

def transform(pose):
    p=np.asarray(pose,dtype=float);x,y,z,w=p[3:]/np.linalg.norm(p[3:]);t=np.eye(4);t[:3,3]=p[:3]
    t[:3,:3]=[[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]]
    return t

def magnetic_pair(device,stand,velocity,stand_velocity,name,device_com=None,stand_com=None):
    p=PARAMS[name];n=stand[:3,:3]@p['normal'];u=device[:3,2]
    anchor=stand[:3,3]+stand[:3,:3]@(np.array(p['center'])+np.array(p['normal'])*p['puck_half'])
    back=device[:3,3]-p['back']*u;d=back-anchor;gap=float(d@n);lateral=d-gap*n
    rho=float(np.linalg.norm(lateral));alignment=float(u@n)
    f=np.zeros(3);tau=np.zeros(3)
    if p['force'] and -.002<gap<.030 and rho<.040 and alignment>.5:
        # Smooth distance attenuation, no teleporting/snap-to-pose capture.
        range_fade=min(1.,max(0.,(.030-gap)/.006))*min(1.,max(0.,(.040-rho)/.010))
        gain=math.exp(-max(gap,0)/p['decay']-.5*(rho/p['sigma'])**2)*alignment**4*range_fade
        rel=velocity[:3]+np.cross(velocity[3:],back-device[:3,3])-stand_velocity[:3]-np.cross(stand_velocity[3:],anchor-stand[:3,3])
        f=gain*(-p['force']*n-p['force']*p['decay']/p['sigma']**2*lateral-1.2*rel)
        tau=gain*(p['torque']*np.cross(u,n)-.0015*(velocity[3:]-stand_velocity[3:]))
    device_com=device[:3,3] if device_com is None else device_com
    stand_com=stand[:3,3] if stand_com is None else stand_com
    device_torque=np.cross(back-device_com,f)+tau
    # The reaction is applied at the same world point, conserving total wrench.
    stand_torque=-np.cross(back-stand_com,f)-tau
    return f,device_torque,-f,stand_torque,{'gap_m':gap,'offset_m':rho,'angle_deg':math.degrees(math.acos(np.clip(alignment,-1,1))),'magnetic_force_n':float(np.linalg.norm(f))}


class ChargingCoupling:
    def __init__(self,base,objects,power=True,magnets=True):
        import torch
        self.base=base;self.torch=torch;self.objects=objects;self.power=power;self.magnets=magnets
        self.time=0.;self.last={};self.dwell=dict.fromkeys(DEVICES,0.);self.charging=dict.fromkeys(DEVICES,False);self.started=dict.fromkeys(DEVICES,None)
        self.body_ids={n:([0] if n!='apple_watch' else list(base.scene[n].find_bodies('case')[0])) for n in (*DEVICES,'belkin')}
        self.original=base.scene.update;self.active=True
        self.original_instance=base.scene.__dict__.get('update');self.had_instance='update' in base.scene.__dict__
        def updated(dt,*args,**kwargs):
            result=self.original(dt,*args,**kwargs)
            if self.active:self.advance(float(dt))
            return result
        self.wrapper=updated;base.scene.update=updated

    def poses(self):return {n:transform(array(self.base.scene[n].data.root_link_pose_w)[0]) for n in (*DEVICES,'belkin')}

    def apply(self,name,f,tau):
        asset=self.base.scene[name]
        tensor=lambda x:self.torch.tensor([[x.tolist()]],device=self.base.device,dtype=self.torch.float32)
        asset.permanent_wrench_composer.set_forces_and_torques_index(forces=tensor(f),torques=tensor(tau),body_ids=self.body_ids[name],is_global=True)

    def advance(self,dt):
        self.time+=dt;poses=self.poses();stand=poses['belkin'];sf=np.zeros(3);st=np.zeros(3);rows={}
        sv=array(self.base.scene['belkin'].data.root_link_vel_w)[0]
        stand_com=array(self.base.scene['belkin'].data.root_com_pose_w)[0,:3]
        for name in DEVICES:
            v=array(self.base.scene[name].data.root_link_vel_w)[0]
            device_com=array(self.base.scene[name].data.root_com_pose_w)[0,:3]
            f,t,rf,rt,row=magnetic_pair(poses[name],stand,v,sv,name,device_com,stand_com)
            if not self.magnets:
                f*=0;t*=0;rf*=0;rt*=0;row['magnetic_force_n']=0.
            self.apply(name,f,t);sf+=rf;st+=rt
            sensor=self.base.scene[name+'_contacts']
            contact=float(np.linalg.norm(array(sensor.data.normal_force_matrix_w)[:,:,0,:]))
            speed=float(np.linalg.norm(v[:3]-sv[:3]))
            row.update(contact_force_n=contact,relative_speed_m_s=speed)
            # A 2.2 mm-thick strap cannot fit the submillimetre charging gap.
            # Native colliders, face alignment and contact gate the electrical proxy.
            valid=self.power and -.0006<row['gap_m']<.0008 and row['offset_m']<(.006 if name=='airpods_pro' else .003) and row['angle_deg']<6 and contact>.04 and speed<.015
            self.dwell[name]=self.dwell[name]+dt if valid else 0.
            charging=self.dwell[name]>=.3
            if charging and not self.charging[name]:self.started[name]=self.time
            if not charging:self.started[name]=None
            self.charging[name]=charging;row.update(charging=charging,dwell_s=self.dwell[name]);rows[name]=row
        self.apply('belkin',sf,st);self.last=rows

    def render(self):
        from pxr import UsdGeom,Gf
        import omni.usd
        stage=omni.usd.get_context().get_stage()
        for name in DEVICES:
            visible=self.charging[name]
            if name=='airpods_pro':visible=visible and self.time-self.started[name]<8.
            path=self.objects[name].prim_path.replace('{ENV_REGEX_NS}','/World/envs/env_0')
            path+=('/case' if name=='apple_watch' else '')+'/charging_indicator'
            prim=stage.GetPrimAtPath(path)
            if not prim.IsValid():raise RuntimeError('Missing charging indicator '+path)
            UsdGeom.Imageable(prim).GetVisibilityAttr().Set('inherited' if visible else 'invisible')
            elapsed=self.time-self.started[name] if self.started[name] is not None else 0.
            scale=.72+.28*min(1.,elapsed/.7) if visible and name!='airpods_pro' else 1.
            attr=prim.GetAttribute('xformOp:scale')
            if attr.IsValid():attr.Set(Gf.Vec3f(scale,scale,1.))

    def close(self):
        self.active=False
        if self.had_instance:self.base.scene.update=self.original_instance
        else:del self.base.scene.__dict__['update']
