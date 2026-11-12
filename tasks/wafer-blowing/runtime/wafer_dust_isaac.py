"""Continuous two-arm wafer workflow with breakable surface-bound test debris."""
from pathlib import Path
import argparse,copy
import numpy as np
import wafer_full_cycle_isaac as full
from wafer_dust import ADHESIVE_PROFILE,SurfaceAdhesion,CohesiveDustScore,cohesive_forces,layout,LAYOUT
from wafer_grasp_protocol import _pose


class DustFullCycle(full.FullCycle):
    def validate(self,bundle,plan,metadata):
        super().validate(bundle,plan,metadata)
        metadata['dust']=dict(profile=copy.deepcopy(ADHESIVE_PROFILE),ids=[f'dust_{i:02d}'for i in range(12)],
            initial_centers_wafer_m=layout().tolist(),layout=copy.deepcopy(LAYOUT),post_start_particle_pose_writes=0,
            force_timing='Measured after each existing physics update; forces held for the next physics interval.',
            qualification='Development only; component checks are not whole-workflow qualification.')
        self.dust_spec=metadata['dust']

    def configure(self,objects,embodiment,Pose,Object,ObjectType,Implicit,Contact,torch):
        sensors=super().configure(objects,embodiment,Pose,Object,ObjectType,Implicit,Contact,torch)
        wafer=objects['wafer'].initial_pose
        pose=_pose([*wafer.position_xyz,*wafer.rotation_xyzw],'wafer')
        self.dust_names=self.dust_spec['ids']
        for name,anchor in zip(self.dust_names,self.dust_spec['initial_centers_wafer_m']):
            center=pose[:3,3]+pose[:3,:3]@np.asarray(anchor)
            initial=Pose.from_dict(dict(position_xyz=[float(x)for x in center],rotation_xyzw=[float(x)for x in wafer.rotation_xyzw]))
            objects[name]=Object(name=name,usd_path=str(self.root/'assets/dust.usda'),object_type=ObjectType.RIGID,initial_pose=initial)
        paths=[objects[name].get_contact_sensor_prim_path()for name in self.dust_names]
        for name,spec in zip(self.tool_environment_sensor_names,self.tool_environment_filters):
            spec['partners'].extend(self.dust_names);spec['paths'].extend(paths)
            sensors[name].filter_prim_paths_expr.extend(paths)
        self.metadata['tool_environment_filter_spec']=copy.deepcopy(self.tool_environment_filters)
        self.metadata['dust']['tool_contact_scope']='Explicitly filtered test-debris impacts are allowed after air starts; unknown, fixture and gun contacts remain forbidden.'
        return sensors

    def allowed_tool_particle_partners(self,row):
        return self.dust_names if row.get('air_on') or self.dust_score.first_air is not None else ()

    def extra_wafer_contacts(self):
        return [(name,self.objects[name].get_contact_sensor_prim_path())for name in self.dust_names]

    def ready(self,base,metadata,torch):
        super().ready(base,metadata,torch)
        import omni.usd
        from wafer_benchmark_backend import _GeometryIntegrity
        from pxr import Usd,UsdPhysics
        self.torch=torch;self.particles=[base.scene[name]for name in self.dust_names];self.wafer=base.scene['wafer']
        self.bonds=SurfaceAdhesion(self.dust_spec['initial_centers_wafer_m'],self.dust_spec['profile'])
        self.dust_score=CohesiveDustScore(metadata)
        roots={name:self.objects[name].prim_path.replace('{ENV_REGEX_NS}','/World/envs/env_0').replace('{ENV_NS}','/World/envs/env_0').replace('/env_.*/','/env_0/')for name in self.dust_names}
        stage=omni.usd.get_context().get_stage();self.dust_integrity=_GeometryIntegrity(stage,roots)
        for name,path in roots.items():
            prim=stage.GetPrimAtPath(path);rb=UsdPhysics.RigidBodyAPI(prim)
            if not prim.IsValid() or rb.GetKinematicEnabledAttr().Get() or not rb.GetRigidBodyEnabledAttr().Get():
                raise ValueError('Particle is not an independent dynamic body: '+name)
            if abs(float(UsdPhysics.MassAPI(prim).GetMassAttr().Get())-self.dust_spec['profile']['mass_kg'])>1e-12:
                raise ValueError('Particle mass differs: '+name)
            if any(p.IsA(UsdPhysics.Joint)for p in Usd.PrimRange(prim)):raise ValueError('Particle attachment is forbidden.')
        self.wrench=torch.zeros((13,2,1,3),device=base.device,dtype=torch.float32)
        self.other_assets=[base.scene[n]for n in ('clamp','left_robot','right_robot','gun')]
        self.last_progress=-1

    def fields(self):
        import warp as wp
        fields=super().fields()
        for i,asset in enumerate(self.particles):
            fields.extend([(f'dust_pose_{i}',asset.data.root_link_pose_w,(1,7)),
                (f'dust_velocity_{i}',asset.data.body_link_lin_vel_w,(1,1,3)),
                (f'dust_force_{i}',wp.to_torch(asset.permanent_wrench_composer.global_force_at_com_w),(1,1,3))])
        fields.extend([('dust_wafer_link_velocity',self.wafer.data.body_link_lin_vel_w,(1,1,3)),
            ('dust_wafer_angular_velocity',self.wafer.data.body_link_ang_vel_w,(1,1,3)),
            ('dust_wafer_com',self.wafer.data.body_com_pos_w,(1,1,3)),
            ('dust_wafer_force',wp.to_torch(self.wafer.permanent_wrench_composer.global_force_at_com_w),(1,1,3)),
            ('dust_wafer_torque',wp.to_torch(self.wafer.permanent_wrench_composer.global_torque_w),(1,1,3))])
        for i,asset in enumerate(self.other_assets):
            for kind in ('global_force_w','global_torque_w','global_force_at_com_w','local_force_b','local_torque_b'):
                fields.append((f'dust_other_{i}_{kind}',wp.to_torch(getattr(asset.permanent_wrench_composer,kind)),None))
        return fields

    def measurement(self,a,row,phase):
        result=super().measurement(a,row,phase)
        present,intact=self.dust_integrity.snapshot()
        result.update(dust_position_world_m=[a[f'dust_pose_{i}'][0,:3].tolist()for i in range(12)],
            dust_pose_world=[a[f'dust_pose_{i}'][0].tolist()for i in range(12)],
            dust_velocity_world_m_s=[a[f'dust_velocity_{i}'][0,0].tolist()for i in range(12)],
            dust_applied_force_world_N=[a[f'dust_force_{i}'][0,0].tolist()for i in range(12)],
            dust_wafer_contact_N=np.linalg.norm(np.asarray(row['filtered_normal_force_vectors_world_N'])[-12:],axis=1).tolist(),
            dust_wafer_link_velocity_world_m_s=a['dust_wafer_link_velocity'][0,0].tolist(),
            dust_wafer_angular_velocity_world_rad_s=a['dust_wafer_angular_velocity'][0,0].tolist(),
            dust_wafer_com_world_m=a['dust_wafer_com'][0,0].tolist(),
            dust_applied_wafer_wrench_world_N_Nm=np.r_[a['dust_wafer_force'][0,0],a['dust_wafer_torque'][0,0]].tolist(),
            dust_bonds_bound=self.bonds.bound.tolist(),dust_intact=all(present.values())and all(intact.values()),
            dust_other_external_wrench_max_abs=max(float(np.max(abs(value)))for key,value in a.items()if key.startswith('dust_other_')))
        return result

    def guard(self,row):
        super().guard(row)
        self.apply_particle_forces(row)

    def apply_particle_forces(self,row):
        self.dust_score.update(row)
        if self.dust_score.early_loss:raise full.native.TrialAbort('A particle detached or left the wafer before blowing.')
        force,jet,reaction,torque=cohesive_forces(self.bonds,row,self.dust_spec['profile'])
        values=np.zeros((13,2,1,3),dtype=np.float32);values[:12,0,0]=force;values[12,0,0]=reaction;values[12,1,0]=torque
        self.wrench.copy_(self.torch.from_numpy(values).to(self.base.device))
        for i,asset in enumerate([*self.particles,self.wafer]):
            composer=asset.permanent_wrench_composer
            composer.reset()
            composer.add_forces_and_torques_index(self.wrench[i,0:1],self.wrench[i,1:2],is_global=True)
        second=int(row['time_s'])
        if second!=self.last_progress:
            print(f"Particles t={row['time_s']:.1f}: {self.dust_score.remaining}/12 not yet clear; bonded {int(self.bonds.bound.sum())}; phase {row['phase']}",flush=True)
            self.last_progress=second


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--bundle',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',default='cuda:0',choices=['cpu','cuda:0']);parser.add_argument('--development',action='store_true');args=parser.parse_args()
    full.ALLOW_UNQUALIFIED_DEVELOPMENT=args.development
    full.native.run_isaac_trial(args.bundle,args.output,extension=DustFullCycle(),device=args.device)
