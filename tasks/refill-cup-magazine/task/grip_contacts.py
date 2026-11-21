"""Observe native PhysX contacts between gripper pads, the wrapper and cups."""
import re

def enable_reports(labkit):
    base=labkit._author
    def author(path,cfg,*args,**kwargs):
        result=base(path,cfg,*args,**kwargs)
        if cfg.part_name=="wrapper" or re.fullmatch(r"cup\d+",cfg.part_name):
            from isaaclab.sim.utils.stage import get_current_stage
            from pxr import Usd, UsdPhysics, PhysxSchema
            for prim in Usd.PrimRange(get_current_stage().GetPrimAtPath(path)):
                if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.)
        return result
    labkit._author=author

class PadForces:
    """GPU contact-pair forces at the current physics sample (N)."""
    def __init__(self):
        from isaaclab_physx.physics import PhysxManager
        from isaaclab.sim.utils.stage import get_current_stage
        from pxr import UsdPhysics
        stage=get_current_stage()
        bodies=[str(p.GetPath()) for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
        self.pad_paths=sorted(p for p in bodies if "/env_0/" in p and p.endswith(("/tip_left","/tip_right")))
        self.sensor_paths=sorted(p for p in bodies if "/env_0/" in p and
                                  ("/wrapper/" in p or re.search(r"/cup\d+/body$",p)))
        if len(self.pad_paths)!=4 or len(self.sensor_paths)!=13:
            raise RuntimeError(f"Unexpected contact measurement paths: {self.pad_paths}, {self.sensor_paths}")
        self.view=PhysxManager.get_physics_sim_view().create_rigid_contact_view(
            self.sensor_paths,filter_patterns=[self.pad_paths for _ in self.sensor_paths],
            max_contact_data_count=256)
        if self.view.sensor_count!=len(self.sensor_paths) or self.view.filter_count!=4:
            raise RuntimeError("Wrong contact matrix dimensions")
    def sample(self,dt):
        import numpy as np
        tensor=self.view.get_contact_force_matrix(dt=dt)
        data=tensor.numpy() if not hasattr(tensor,"detach") else tensor.detach().cpu().numpy()
        values=np.linalg.norm(data,axis=-1).reshape(len(self.sensor_paths),4)
        wrapper=[i for i,p in enumerate(self.sensor_paths) if "/wrapper/" in p]
        root=next(i for i,p in enumerate(self.sensor_paths) if p.endswith("/wrapper/body"))
        cup=[i for i in range(len(self.sensor_paths)) if i not in wrapper]
        return {"pads":[p.split("/env_0/")[1].split("/")[0]+"/"+p.rsplit("/",1)[1] for p in self.pad_paths],
                "wrapper_N":values[wrapper].sum(axis=0).round(4).tolist(),
                "seam_and_cover_N":values[root].round(4).tolist(),
                "cup_N":values[cup].sum(axis=0).round(4).tolist()}
