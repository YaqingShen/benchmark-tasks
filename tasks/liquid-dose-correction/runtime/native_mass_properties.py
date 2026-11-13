"""Couple conserved liquid mass and quasi-static mass distribution to PhysX."""
import numpy as np
from geometry_fluid import INTERIORS
from liquid_mass_properties import liquid_properties,combine_masses,principal_frame,pipette_liquid_properties


class NativeMassProperties:
    def __init__(self,coupling):
        from native_coupling import array,transform
        self.coupling=coupling;self.dry={};self.cache={};self.last={};self.elapsed=.02
        for name,dry_mass in coupling.dry_masses.items():
            asset=coupling.base.scene[name]
            com=array(asset.data.body_com_pose_b)[0,0]
            r=transform(com)[:3,:3]
            inertia=array(asset.data.body_inertia)[0,0].reshape(3,3)
            initial_mass=coupling.manifest['assets'][name]['mass_kg_initial']
            self.dry[name]=(dry_mass,com[:3].copy(),r@inertia@r.T*(dry_mass/initial_mass),[0])
        for name in coupling.pipettes:
            asset=coupling.base.scene[name];ids,_=asset.find_bodies(['stem'],preserve_order=True);i=ids[0]
            com=array(asset.data.body_com_pose_b)[0,i];r=transform(com)[:3,:3]
            inertia=array(asset.data.body_inertia)[0,i].reshape(3,3)
            from pipette_geometry import STEM_DRY_MASS_KG
            self.dry[name]=(STEM_DRY_MASS_KG,com[:3].copy(),r@inertia@r.T,list(ids))

    def advance(self,poses,dt):
        from yam_arena.ik import matrix_to_quat_xyzw
        self.elapsed+=dt
        # Mass itself is applied every physics step by NativeCoupling. Update
        # COM/inertia at 50 Hz; the liquid is treated as quasi-static, not CFD.
        if self.elapsed<.02-1e-10:return
        self.elapsed=0.;c=self.coupling
        for name,(dry,center,inertia,ids)in self.dry.items():
            liquid=(c.ledger.pipettes[name]if name in c.pipettes else c.ledger.vessels[name]).liquid
            mass=liquid.mass(c.ledger.cfg['density_g_ml'])/1000
            key=np.r_[mass,poses[name][2,:3]if name not in c.pipettes else [0,0,1]]
            prior=self.cache.get(name)
            if prior is not None and abs(key[0]-prior[0])<1e-8 and np.linalg.norm(key[1:]-prior[1:])<.002:continue
            parts=[(dry,center,inertia)]
            if mass>1e-12:
                if name in c.pipettes:lc,li=pipette_liquid_properties(liquid.volume,mass)
                else:lc,li=liquid_properties(INTERIORS[name],poses[name],c.fluid.surface(name,poses[name]),mass)
                parts.append((mass,lc,li))
            total,com,I=combine_masses(parts);moments,rotation=principal_frame(I)
            quaternion=matrix_to_quat_xyzw(rotation)
            asset=c.base.scene[name]
            asset.set_coms_index(coms=c.tensor([[np.r_[com,quaternion].tolist()]]),body_ids=ids)
            asset.set_inertias_index(inertias=c.tensor([[np.diag(moments).reshape(9).tolist()]]),body_ids=ids)
            self.cache[name]=key
            self.last[name]={'mass_kg':total,'com_body_m':com.tolist(),'principal_moments_kg_m2':moments.tolist()}
