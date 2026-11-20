"""Fruit-only section transition and end-state measurement in the Isaac station."""
import json
import math
import os
from pathlib import Path
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
import labkit
from produce_contact import FruitContact, install_physics
from cut_checks import BatchPass, both_tools_held

R=.014
N=8
def sector(sign,j):
    pts=[(0,0,0),(0,0,sign*R)]
    for k in range(4):
        p=k*math.pi/8
        for a in range(3):
            t=(2*j+a)*math.pi/8
            pts.append((R*math.cos(p)*math.cos(t),R*math.cos(p)*math.sin(t),sign*R*math.sin(p)))
    return np.asarray(pts)
def segments(sign):
    out=[]
    for j in range(8):
        pts=sector(sign,j)
        hull=ConvexHull(pts); tri=hull.simplices.copy()
        for idx,(face,plane) in enumerate(zip(tri,hull.equations)):
            a,b,c=pts[face]
            if np.dot(np.cross(b-a,c-a),plane[:3])<0:tri[idx]=face[::-1]
        out.append(labkit.Mesh(pts,tri,(.86,.19,.10),'hull'))
    return out

def install_filter():
    original=labkit._author
    def author(path,cfg,*args,**kwargs):
        result=original(path,cfg,*args,**kwargs)
        if cfg.part_name=='knife':
            from isaaclab.sim.utils.stage import get_current_stage
            from pxr import Usd,UsdPhysics
            stage=get_current_stage()
            parent=stage.GetPrimAtPath(path).GetParent()
            targets=[]
            for obj in parent.GetChildren():
                name=obj.GetName().lower()
                if name.startswith(('tomato','half')):
                    targets.extend(p.GetPath() for p in Usd.PrimRange(obj) if p.HasAPI(UsdPhysics.RigidBodyAPI))
            if len(targets)!=24:raise RuntimeError(f'Expected 24 fruit bodies, got {targets}, roots {[p.GetName() for p in parent.GetChildren()]}')
            for p in Usd.PrimRange(stage.GetPrimAtPath(path)):
                if p.HasAPI(UsdPhysics.RigidBodyAPI) or p.HasAPI(UsdPhysics.CollisionAPI):
                    UsdPhysics.FilteredPairsAPI.Apply(p).CreateFilteredPairsRel().SetTargets(targets)
        return result
    labkit._author=author

class Trial:
    def __init__(self,world):
        self.world=world
        self.reset(None)
    def reset(self,env):
        self.contact=FruitContact()
        if env is not None:
            install_physics(env,self)
        self.step=0;self.batch=BatchPass();self.cut=self.batch.cut_ids
        self.final=False;self.credit=0.;self.still=0
        self.log=Path(os.environ.get('TOMATO_RUN_DIR','/tmp'))/'audit.jsonl'
    def tick(self,env):
        import torch
        from yam_arena.runtime import tcp_world
        scene=env.unwrapped.scene
        def pose(n):return labkit._np(scene[n].data.root_link_pose_w)[0]
        def velocity(n):return labkit._np(scene[n].data.root_vel_w)[0]
        def write(n,p,v):
            obj=scene[n];device=env.unwrapped.device
            obj.write_root_pose_to_sim_index(root_pose=torch.as_tensor(p[None],dtype=torch.float32,device=device))
            obj.write_root_velocity_to_sim_index(root_velocity=torch.as_tensor(v[None],dtype=torch.float32,device=device))
        knife=pose('knife');lid=pose('upper_lid');rot=Rotation.from_quat(knife[3:]).as_matrix()
        left=tcp_world(scene['left_robot']);right=tcp_world(scene['right_robot'])
        supported=np.linalg.norm(lid[:2]-self.world((.071,0,0))[:2])<.015 and abs(lid[2]-self.world((0,0,.108))[2])<.008
        lidgrip=lid[:3]+Rotation.from_quat(lid[3:]).apply([-.048,0,.029])
        knifeg=knife[:3]+rot@np.array([.134,-.011,0])
        lid_rot=Rotation.from_quat(lid[3:]).as_matrix()
        engaged=supported and lid_rot[2,2]>math.cos(math.radians(12)) and abs(rot[2,2])>math.cos(math.radians(12)) and both_tools_held([left,right],lidgrip,knifeg)
        whole=[pose('tomato_'+str(i)) for i in range(N)]
        loc=np.array([(p[:3]-knife[:3])@rot for p in whole])
        lid_local=np.array([(p[:3]-lid[:3])@lid_rot for p in whole])
        # The centre of each fruit must lie under the plate during its crossing.
        covered=(np.abs(lid_local[:,0])<=.080)&(np.abs(lid_local[:,1])<=.072)
        for i in self.batch.observe(knife[:3],rot,loc,covered,engaged,self.contact.ready()):
            p=whole[i];name='tomato_'+str(i);v=velocity(name)
            write(name,np.r_[[.10+i*.04,.16,.09],p[3:]],np.zeros(6))
            for h in (0,1):write(f'half_{2*i+h}',np.r_[p[:3],knife[3:]],v)
        retained={i:0 for i in self.cut}
        for i in self.cut:
            for h in (0,1):
                name=f'half_{2*i+h}';p=pose(name)
                if np.linalg.norm(p[:2]-self.world((.071,0,0))[:2])>.125:continue
                verts=Rotation.from_quat(p[3:]).apply(VERTS[h])+p[:3]
                if (verts.min(axis=0)>=np.array(self.world((.158,-.079,.0785)))).all() and (verts.max(axis=0)<=np.array(self.world((-.016,.079,.116)))).all() and np.linalg.norm(velocity(name)[:3])<.012:retained[i]+=1
        pieces=sum(retained.values())
        docked=np.linalg.norm(lid[:3]-self.world((-.117,.012,.110)))<.012 and np.linalg.norm(knife[:3]-self.world((.071,-.11,.094)))<.012
        start_rotation=Rotation.from_euler("z",-90,degrees=True)
        aligned=all((Rotation.from_quat(p[3:])*start_rotation.inv()).magnitude()<=math.radians(12) for p in (lid,knife))
        resting=all(np.linalg.norm(velocity(name)[:3])<.012 for name in ("knife","upper_lid"))
        clear=all(np.linalg.norm(tip-grip)>.04 for tip in (left,right) for grip in (lidgrip,knifeg))
        self.still=self.still+1 if docked and aligned and resting and clear and pieces==16 and self.batch.batch_complete else 0
        self.final=self.still>=8
        best_count,best_halves=self.batch.best_pass(retained)
        self.credit=(7*best_count+best_halves)/80 + .1*self.final
        row=dict(work_J=self.contact.work.tolist(),draw_m=self.contact.draw.tolist(),peak_N=self.contact.peak.tolist(),max_total_N=self.contact.peak_total,max_contacts=self.contact.active_max,physics_steps=self.contact.physical_steps,step=self.step,cut=sorted(self.cut),pieces=pieces,score=self.credit,complete=self.final,
                 knife=knife.tolist(),lid=lid.tolist(),tcp=[left.tolist(),right.tolist()],engaged=bool(engaged),xyz=loc.tolist(),passes=[sorted(ids) for ids in self.batch.passes],retained=retained,best_count=best_count,best_halves=best_halves,entered=sorted(self.batch.enter),single_pass=bool(self.batch.batch_complete))
        with self.log.open('a') as f:f.write(json.dumps(row)+'\n')
        self.step+=1
VERTS=[np.vstack([sector(s,j) for j in range(8)]) for s in (-1,1)]
