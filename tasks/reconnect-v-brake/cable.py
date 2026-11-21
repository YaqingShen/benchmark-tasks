"""Native cable extension limit and a compliant upstream housing mount."""
from pathlib import PurePosixPath


def wire_guide(original, fork_name, guide_name, park):
    def spawn(prim_path, cfg, translation=None, orientation=None, **kwargs):
        root = original(prim_path, cfg, translation, orientation, **kwargs)
        if cfg.part_name != guide_name:
            return root
        from isaaclab.sim.utils.stage import get_current_stage
        from pxr import Gf, Sdf, UsdPhysics
        stage = get_current_stage()
        folder = str(PurePosixPath(prim_path).parent)
        fork = f"{folder}/{fork_name}"
        body = f"{prim_path}/body"
        mount = UsdPhysics.Joint.Define(stage, f"{prim_path}/housing_compliance")
        mount.CreateBody0Rel().SetTargets([Sdf.Path(fork+"/body")])
        mount.CreateBody1Rel().SetTargets([Sdf.Path(body)])
        mount.CreateLocalPos0Attr(Gf.Vec3f(*park))
        mount.CreateLocalPos1Attr(Gf.Vec3f(0,0,0))
        mount.CreateExcludeFromArticulationAttr(True)
        for axis, stiffness, damping in (("transX",7.,.32),("transY",4.,.24),("transZ",12.,.32),
                                         ("rotX",.08,.007),("rotY",.08,.007),("rotZ",.08,.007)):
            drive = UsdPhysics.DriveAPI.Apply(mount.GetPrim(),axis)
            angular = axis.startswith("rot")
            drive.CreateTypeAttr("force")
            drive.CreateStiffnessAttr(stiffness / (57.29578 if angular else 1))
            drive.CreateDampingAttr(damping / (57.29578 if angular else 1))
            drive.CreateTargetPositionAttr(0.)
        return root
    return spawn



class BowdenCable:
    """Massless tension-only inner wire. Apply equal and opposite endpoint forces at each physics step."""

    def __init__(self, scene, wire_end, free_length):
        import torch
        self.scene=scene
        self.fork=scene["fork"]
        self.guide=scene["guide"]
        self.lever=scene["bar_stand"]
        self.idx=self.fork.body_names.index("anchor_arm")
        self.lever_idx=self.lever.joint_names.index("lever_joint")
        self.fork_local=torch.tensor(wire_end,device=self.fork.device,dtype=torch.float32)
        self.guide_local=torch.tensor((0,.009,0),device=self.fork.device,dtype=torch.float32)
        self.free_length=free_length
        self.load=None

    @staticmethod
    def endpoint(obj,idx,local):
        import torch
        pose=obj.data.body_link_pose_w.torch[:,idx:idx+1]
        q=pose[...,3:6]
        v=local.expand_as(q)
        cross=torch.linalg.cross(q,v,dim=-1)
        point=pose[...,:3]+v+2*(pose[...,6:7]*cross+torch.linalg.cross(q,cross,dim=-1))
        arm=point-obj.data.body_com_pos_w.torch[:,idx:idx+1]
        speed=obj.data.body_com_lin_vel_w.torch[:,idx:idx+1]+torch.linalg.cross(
            obj.data.body_com_ang_vel_w.torch[:,idx:idx+1],arm,dim=-1)
        return point,speed

    def update(self):
        import torch
        a,va=self.endpoint(self.fork,self.idx,self.fork_local)
        g,vg=self.endpoint(self.guide,0,self.guide_local)
        ray=a-g
        distance=torch.linalg.vector_norm(ray,dim=-1,keepdim=True).clamp_min(1e-6)
        unit=ray/distance
        q=self.lever.data.joint_pos.torch[:,self.lever_idx:self.lever_idx+1,None]
        rate=self.lever.data.joint_vel.torch[:,self.lever_idx:self.lever_idx+1,None]
        extension=distance-self.free_length+.047*q.clamp(0,.38)
        speed=((va-vg)*unit).sum(dim=-1,keepdim=True)+.047*rate
        self.load=torch.where(extension>0,(520*extension+1.65*speed).clamp(0.,30.),0.)
        force=self.load*unit
        self.fork.permanent_wrench_composer.set_forces_and_torques_index(forces=-force,positions=a,body_ids=[self.idx],is_global=True)
        self.guide.permanent_wrench_composer.set_forces_and_torques_index(forces=force,positions=g,body_ids=[0],is_global=True)
        self.lever.actuators.target_command.set_effort_index(value=-.047*self.load[:,:,0],joint_ids=[self.lever_idx])

    @classmethod
    def install(cls,scene,wire_end,free_length):
        if hasattr(scene,"vbrake_inner_wire"):
            return scene.vbrake_inner_wire
        cable=cls(scene,wire_end,free_length)
        write=scene.write_data_to_sim
        def write_with_cable():
            cable.update()
            return write()
        scene.write_data_to_sim=write_with_cable
        scene.vbrake_inner_wire=cable
        return cable


class CableVisual:
    def __init__(self,scene):
        from isaaclab.sim.utils.stage import get_current_stage
        from pxr import UsdGeom
        stage=get_current_stage()
        root=scene.env_prim_paths[0]
        self.curves=[]
        for name,width,color,count in (("wire",.0015,(.68,.70,.72),13),("housing",.005,(.055,.06,.065),13)):
            curve=UsdGeom.BasisCurves.Define(stage,root+"/brake_"+name)
            curve.CreateTypeAttr("linear")
            curve.CreateCurveVertexCountsAttr([count])
            curve.CreateWidthsAttr([width])
            curve.SetWidthsInterpolation("constant")
            curve.CreateDisplayColorAttr([color])
            self.curves.append(curve)

    def update(self,a,g,upper,lever,free_length):
        import numpy as np
        from pxr import Vt
        t=np.linspace(0.,1.,13)[:,None]
        d=np.linalg.norm(a-g)
        sag=min(.025,.50*np.sqrt(max(free_length**2-d**2,0.)))
        points=g+(a-g)*t-4*t*(1-t)*np.array([0.,0.,sag])
        self.curves[0].CreatePointsAttr().Set(Vt.Vec3fArray.FromNumpy(points.astype(np.float32)))
        c1=upper+(-.01,-.025,.045)
        c2=lever+(.025,-.020,.060)
        points=(1-t)**3*upper+3*(1-t)**2*t*c1+3*(1-t)*t**2*c2+t**3*lever
        self.curves[1].CreatePointsAttr().Set(Vt.Vec3fArray.FromNumpy(points.astype(np.float32)))
