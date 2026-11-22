"""Couple the confined axial stack to the native paper bodies and loose quarters."""
import numpy as np
from scipy.spatial.transform import Rotation
import geometry as g
from packed import COUNT, INITIAL, PAPER_MASS, Sleeve

class PackedRoll:
    def __init__(self,env,coins,shells):
        from pxr import PhysxSchema,UsdPhysics
        from isaaclab.sim.utils.stage import get_current_stage
        self.groups=[Sleeve(),Sleeve()]
        self.scene=env.unwrapped.scene
        prefix=self.scene.env_prim_paths[0]
        stage=get_current_stage()
        self.coin_assets=[self.scene[c.name] for c in coins]
        self.shell_assets=[self.scene[s.name] for s in shells]
        self.filters=[];self.gravity=[];self.sleeve_mass=[];self.sleeve_inertia=[];self.center=[]
        self.previous_center=[None,None]
        for c in coins:
            body=stage.GetPrimAtPath(prefix+'/'+c.name+'/body')
            self.filters.append(UsdPhysics.FilteredPairsAPI(body).GetFilteredPairsRel())
            self.gravity.append(PhysxSchema.PhysxRigidBodyAPI(body).GetDisableGravityAttr())
        for s in shells:
            body=stage.GetPrimAtPath(prefix+'/'+s.name+'/body')
            props=UsdPhysics.MassAPI(body)
            self.sleeve_mass.append(props.GetMassAttr())
            self.sleeve_inertia.append(props.GetDiagonalInertiaAttr())
            self.center.append(props.GetCenterOfMassAttr())
        self.velocity=[None,None]
        self.loose=np.zeros(g.COUNT,bool)
        self.last_emission=[]
        # Restore packed collision pairs at reset, including after a previous episode.
        peers=[a.GetPrim().GetPath() for a in self.gravity]
        papers=[a.GetPrim().GetPath() for a in self.sleeve_mass]
        arms=[path for arm in ('left','right') for path in self.scene[arm+'_robot'].root_physx_view.link_paths[0]]
        for n,(rel,grav) in enumerate(zip(self.filters,self.gravity)):
            rel.SetTargets([q for j,q in enumerate(peers) if j!=n]+papers+[prefix+'/till/body']+arms)
            grav.Set(True)
        for half in range(2):self._mass(half)
    def _mass(self,half):
        from pxr import Gf
        group=self.groups[half]
        mass=group.mass
        coordinates=INITIAL[group.next:]+group.offset
        sign=(1,-1)[half]
        com=-sign*g.MASS*float(coordinates.sum())/mass
        previous=self.previous_center[half]
        if previous and previous[0]==mass and abs(previous[1]-com)<.0003:return
        self.previous_center[half]=(mass,com)
        self.sleeve_mass[half].Set(float(mass))
        self.center[half].Set(Gf.Vec3f(0.,0.,com))
        transverse=PAPER_MASS*(g.BORE**2/2+g.L**2/12)+float(np.sum(g.MASS*(g.R**2/4+g.T**2/12+coordinates**2)))-mass*com**2
        axial=PAPER_MASS*g.BORE**2+len(coordinates)*g.MASS*g.R**2/2
        self.sleeve_inertia[half].Set(Gf.Vec3f(transverse,transverse,axial))
    def update(self,states,broken,gap):
        import torch
        self.last_emission=[]
        for h,(sign,state,group) in enumerate(zip((1,-1),states,self.groups)):
            out=-sign*state.rotation()[:,2]
            linear=np.asarray(state.lin_vel,dtype=float)-np.cross(state.ang_vel,state.rotation()[:,2]*self.previous_center[h][1])
            acceleration=np.zeros(3) if self.velocity[h] is None else (linear-self.velocity[h])/.05
            self.velocity[h]=linear.copy()
            effective=np.array([0.,0.,-9.81])-acceleration
            axial=effective@out
            normal=np.linalg.norm(effective-axial*out)
            first=group.next
            emitted=group.advance(axial,normal,broken and gap>g.T*2)
            quaternion=Rotation.from_matrix(state.rotation()).as_quat()
            for u in range(first,COUNT):
                # Coin nearest the middle seam exits first. Left and right
                # packs have opposite numbering and outward directions.
                idx=COUNT+u if h==0 else COUNT-1-u
                asset=self.coin_assets[idx]
                distance=float(INITIAL[u]+group.offset)
                relative=out*distance
                location=state.pos+relative
                world_vel=linear+np.cross(state.ang_vel,relative)+out*group.speed
                pose=torch.tensor([np.r_[location,quaternion].tolist()],device=asset.device,dtype=torch.float32)
                pose[:,:3]+=self.scene.env_origins[0]
                vel=torch.tensor([np.r_[world_vel,state.ang_vel].tolist()],device=asset.device,dtype=torch.float32)
                asset.write_root_pose_to_sim(pose)
                asset.write_root_velocity_to_sim(vel)
                if u in emitted:
                    self.filters[idx].SetTargets([])
                    self.gravity[idx].Set(False)
                    self.loose[idx]=True
                    self.last_emission.append(idx)
            self._mass(h)
        return self.loose
