"""Candidate bridge from native measured states to reduced liquid dynamics.

This module is deliberately not a benchmark scorer. Contact-based disposal,
balance loading, live visual fluid/display, inertia updates and independent
force readback still require validation before a full task can be evaluated.
No rigid-body pose, velocity, joint position or grasp attachment is written.
"""
import math
import numpy as np
from sim import Formulation
from geometry_fluid import GeometryFluid,INTERIORS
from pneumatic_pipette import PneumaticPipette
from pipette_geometry import STEM_DRY_MASS_KG


def array(value):
    value=value.torch if hasattr(value,'torch') else value
    if hasattr(value,'detach'):value=value.detach().cpu().numpy()
    value=np.asarray(value,dtype=float)
    if not np.isfinite(value).all():raise ValueError('Nonfinite native measurement')
    return value


def transform(pose):
    p=np.asarray(pose,dtype=float)
    if p.shape!=(7,) or not np.isfinite(p).all():raise ValueError('Expected native XYZ, XYZW pose')
    q=p[3:];norm=np.linalg.norm(q)
    if abs(norm-1)>1e-4:raise ValueError('Invalid native unit quaternion')
    x,y,z,w=q/norm
    m=np.eye(4);m[:3,3]=p[:3]
    m[:3,:3]=[[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]]
    return m


class NativeCoupling:
    """Apply paired bulb pressure forces and update masses from a liquid ledger."""
    def __init__(self,base,manifest):
        import torch
        self.base=base;self.torch=torch;self.manifest=manifest
        self.ledger=Formulation()
        self.fluid=GeometryFluid(self.ledger,table_z=manifest['table_top_m'])
        self.pipettes={n:PneumaticPipette(self.fluid,n) for n in self.ledger.pipettes}
        self.joints={};self.bodies={};self.mass_cache={}
        self.dry_masses={'bottle_A':.12,'bottle_B':.12,'cylinder_A':.045,'cylinder_B':.045,'beaker':.12,'waste':.06}
        self.dry_masses={n:m for n,m in self.dry_masses.items()if n in manifest['assets']}
        self.steps=0;self.last=None
        from native_balance import NativeBalance
        self.balance=NativeBalance(base) if 'digital_balance' in manifest['assets'] else None
        from native_liquid_visuals import LiquidVisuals
        self.visuals=LiquidVisuals(self)
        for name in self.pipettes:
            asset=base.scene[name]
            ids,names=asset.find_joints(['squeeze_left','squeeze_right'],preserve_order=True)
            if list(names)!=['squeeze_left','squeeze_right']:raise RuntimeError('Pipette squeeze joint map differs')
            self.joints[name]=list(ids)
            ids,names=asset.find_bodies(['bulb_left','bulb_right'],preserve_order=True)
            if list(names)!=['bulb_left','bulb_right']:raise RuntimeError('Pipette pressure body map differs')
            self.bodies[name]=list(ids)
        from native_mass_properties import NativeMassProperties
        self.mass_properties=NativeMassProperties(self)
        self.original=base.scene.update
        self.original_instance=base.scene.__dict__.get('update')
        self.had_instance='update' in base.scene.__dict__
        self.active=False
        def updated(dt,*args,**kwargs):
            result=self.original(dt,*args,**kwargs)
            if self.active:self.advance(float(dt))
            return result
        self.wrapper=updated;base.scene.update=updated

    def poses(self):
        return {n:transform(array(self.base.scene[n].data.root_link_pose_w)[0]) for n in self.manifest['assets']}

    def tensor(self,value):
        return self.torch.tensor(value,dtype=self.torch.float32,device=self.base.device)

    def advance(self,dt):
        if abs(dt-self.base.physics_dt)>1e-10:raise RuntimeError('Fluid hook is not at the native physics cadence')
        poses=self.poses()
        caps={}
        for name,vessel in self.ledger.vessels.items():
            pose=poses[name]
            vessel.tilt_deg=math.degrees(math.acos(float(np.clip(pose[2,2],-1,1))))
            if name.startswith('bottle_'):
                rel=np.linalg.inv(pose)@poses[name+'_cap']
                xy=float(np.linalg.norm(rel[:2,3]));z=float(rel[2,3])
                aligned=float(rel[2,2])>math.cos(math.radians(5))
                # Flow blocking follows coverage of the actual mouth, rather
                # than a controller's open/closed command or elapsed turn count.
                vessel.capped=aligned and xy<.004 and .17<z<.198
                seated=aligned and xy<.001 and abs(z-.185)<.0005
                vessel.cap_turns=self.ledger.cfg['cap_required_turns'] if seated else 0.
                home=np.array(self.manifest['assets'][name]['position_xyz'])
                vessel.home_distance_m=float(np.linalg.norm(pose[:2,3]-home[:2]))
                caps[name]={'relative_position':rel[:3,3].tolist(),'mouth_covered':vessel.capped,
                            'geometrically_seated':seated}
        liquid_poses={n:poses[n] for n in INTERIORS if n in poses}
        tools={}
        for name,pipette in self.pipettes.items():
            asset=self.base.scene[name]
            q=array(asset.data.joint_pos)[0,self.joints[name]]
            if q[0]<-.0002 or q[0]>.006 or q[1]<-.006 or q[1]>.0002:
                raise RuntimeError(f'{name} spring joint bounds exceeded at physics step {self.steps}: q={q.tolist()}')
            compression=float(np.clip((q[0]-q[1])/.0116,0,1))
            result=pipette.step(poses[name],compression,liquid_poses,dt)
            result['joint_positions_m']=q.tolist()
            force=result['outward_force_per_surface_n']
            if not math.isfinite(force) or abs(force)>100:
                raise RuntimeError('Unvalidated bulb pressure load exceeds 100 N; stop diagnostic')
            # Equal and opposite local-X forces act on the two real bodies;
            # there is no net propulsive force or prescribed bulb motion.
            asset.permanent_wrench_composer.set_forces_and_torques_index(
                forces=self.tensor([[[-force,0,0],[force,0,0]]]),
                torques=self.tensor([[[0,0,0],[0,0,0]]]),body_ids=self.bodies[name],is_global=False)
            tools[name]=result
        self.fluid.advance(liquid_poses,dt)
        for name,dry in self.dry_masses.items():
            asset=self.base.scene[name]
            mass=dry+self.ledger.vessels[name].liquid.mass(self.ledger.cfg['density_g_ml'])/1000
            if abs(mass-self.mass_cache.get(name,-1))>1e-8:
                asset.set_masses_index(masses=self.tensor([[mass]]))
                self.mass_cache[name]=mass
        for name in self.pipettes:
            asset=self.base.scene[name]
            mass=STEM_DRY_MASS_KG+self.ledger.pipettes[name].liquid.mass(self.ledger.cfg['density_g_ml'])/1000
            if abs(mass-self.mass_cache.get(name,-1))>1e-8:
                ids,found=asset.find_bodies(['stem'],preserve_order=True)
                if list(found)!=['stem']:raise RuntimeError('Pipette liquid body mapping differs')
                asset.set_masses_index(masses=self.tensor([[mass]]),body_ids=list(ids))
                self.mass_cache[name]=mass
        self.steps+=1
        self.mass_properties.advance(poses,dt)
        if self.balance is not None:self.balance.advance(dt)
        self.last={'time_s':self.ledger.time,'caps':caps,'pipettes':tools,
                   'beaker_ml':self.ledger.vessels['beaker'].liquid.snapshot(),
                   'spill_ml':self.ledger.spill.volume,'inflight_ml':sum(x.volume for x in self.ledger.inflight),
                   'applied_masses_kg':dict(self.mass_cache),'physical_pose_writes':0,
                   'mass_distribution':self.mass_properties.last,
                   'balance':self.balance.last if self.balance is not None else None}
        return self.last

    def close(self):
        self.active=False
        if self.base.scene.update is not self.wrapper:raise RuntimeError('Scene hook changed unexpectedly')
        if self.had_instance:self.base.scene.update=self.original_instance
        else:del self.base.scene.__dict__['update']

    def render(self):
        if self.balance is not None:self.balance.render()
        self.visuals.render()
