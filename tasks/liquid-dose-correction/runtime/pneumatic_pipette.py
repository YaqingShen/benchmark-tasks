"""Reduced sealed-air pipette model driven by measured bulb displacement.

This is an uncalibrated engineering model, not CFD or a claim about the exact
photographed pipette. Gas pressure uses isothermal compression; liquid flow
uses a Poiseuille resistance and an exposed meniscus-pressure threshold.
The native backend must apply the returned pressure forces to the two bulb
surfaces. The model never specifies a grasp, target vessel, or robot action.
"""
import math
import numpy as np
from pipette_geometry import STEM_LENGTH_M,STEM_LIQUID_CAPACITY_ML

P_ATM=101325.
RHO_AIR=1.2
GRAVITY=9.81


class PneumaticPipette:
    def __init__(self,geometry,name,viscosity_pa_s=.001,meniscus_pa=180.):
        self.geometry=geometry
        self.env=geometry.env
        self.name=name
        self.tool=self.env.pipettes[name]
        self.dead_ml=.2+STEM_LIQUID_CAPACITY_ML-.30
        self.air_atm_ml=self.tool.capacity_ml+self.dead_ml
        self.viscosity=viscosity_pa_s
        self.meniscus_pa=float(meniscus_pa)
        self.tube_radius_m=.0008
        self.tube_length_m=STEM_LENGTH_M
        self.stroke_total_m=.0116
        self.pressure_pa=P_ATM
        self.last=None

    def _column_height(self,volume):
        # Narrow stem followed by the much wider bulb. This is a reduced
        # volume-height profile; tilt and inversion use the measured axis.
        return STEM_LENGTH_M*min(volume/STEM_LIQUID_CAPACITY_ML,1.)+.047*max(0.,volume-STEM_LIQUID_CAPACITY_ML)/(5.-STEM_LIQUID_CAPACITY_ML)

    def step(self,pose,compression,vessel_poses,dt):
        before_g=self.env.vessels['beaker'].liquid.mass(self.env.cfg['density_g_ml'])
        if not math.isfinite(compression) or not 0<=compression<=1 or not 0<dt<=.02:
            raise ValueError('Finite measured compression [0,1] and dt (0,.02] required')
        m=self.geometry.validate_pose(pose)
        tip=m[:3,3]
        immersed=self.geometry.tip_immersion(tip,vessel_poses)
        tool=self.tool
        tool.compression=float(compression)
        tool.tip_target=immersed
        tool.tip_submerged=immersed is not None
        volume=tool.liquid.volume
        bulb=self.dead_ml+tool.capacity_ml*(1-compression)
        if immersed:
            vessel=self.env.vessels[immersed]
            foreign='B' if tool.dedicated_to=='A' else 'A'
            if vessel.liquid.ingredient(foreign)>1e-9:tool.contamination=True
            if immersed.startswith('bottle_') and (tool.contamination or immersed[-1]!=tool.dedicated_to):
                self.env.stock_contaminated=True
            density=vessel.liquid.mass(self.env.cfg['density_g_ml'])/vessel.liquid.volume*1000
            source_head=max(0.,self.geometry.surface(immersed,vessel_poses[immersed])-tip[2])
            film=self.env.cfg['cylinder_unrecoverable_film_ml'] if immersed.startswith('cylinder_') else 0.
            available=max(0.,vessel.liquid.volume-film)
        else:
            density=(tool.liquid.mass(self.env.cfg['density_g_ml'])/volume*1000 if volume>1e-9 else 1000.)
            source_head=0.;available=0.
        conductance=math.pi*self.tube_radius_m**4/(8*self.viscosity*self.tube_length_m)*1e6

        def pressure(v):
            return P_ATM*self.air_atm_ml/max(1e-8,bulb-v)

        def head(v):
            return density*GRAVITY*(m[2,2]*self._column_height(v)-source_head)

        # An empty dry tip vents to atmosphere. Air entry through a wet tip
        # uses the same exposed capillary threshold; no liquid is invented.
        if immersed is None:
            if volume<=1e-9:
                self.air_atm_ml=bulb
            elif pressure(volume)-P_ATM+head(volume)<-self.meniscus_pa:
                self.air_atm_ml=max(1e-8,bulb-volume)*(P_ATM-head(volume)-self.meniscus_pa)/P_ATM

        def flow(v):
            delta=pressure(v)-P_ATM+head(v)
            if delta>self.meniscus_pa:return conductance*(delta-self.meniscus_pa)
            if delta<-self.meniscus_pa and immersed:return conductance*(delta+self.meniscus_pa)
            return 0.

        # Backward Euler for pressure-flow coupling, solved by monotone
        # bisection to avoid unstable explicit pressure updates at small air
        # volumes. Limits preserve finite capacity and available source liquid.
        lo=0.;hi=max(0.,min(tool.capacity_ml,bulb-1e-8,volume+available))
        def residual(v):return v-volume+dt*flow(v)
        if residual(lo)>=0:new=lo
        elif residual(hi)<=0:new=hi
        else:
            for _ in range(48):
                mid=(lo+hi)/2
                if residual(mid)>0:hi=mid
                else:lo=mid
            new=(lo+hi)/2
        change=new-volume
        if change>1e-12:
            if immersed is None:raise RuntimeError('Dry tip cannot aspirate liquid')
            packet=self.env.take_from(immersed,change)
            tool.liquid.add(packet)
        elif change<-1e-12:
            packet=tool.liquid.take(-change)
            if immersed:
                self.env._receive(immersed,packet)
            else:
                # A jet leaves the actual tip along the negative local stem
                # axis. Capture is resolved later against physical apertures.
                rate=(-change)/dt*1e-6
                speed=rate/(math.pi*self.tube_radius_m**2)
                direction=-m[:3,2]
                self.geometry.emit(self.name,packet,tip+direction*.0005,direction*speed)
        if tool.liquid.volume<1e-9 and immersed is None:self.air_atm_ml=bulb
        self.pressure_pa=pressure(tool.liquid.volume)
        # dV/d(total inward travel); SI conversion from mL to m3.
        area=tool.capacity_ml*1e-6/self.stroke_total_m
        force=(self.pressure_pa-P_ATM)*area
        self.env.assert_conservation()
        self.env.credit_pipette_correction(self.name,before_g)
        self.last={'tip_vessel':immersed,'compression':float(compression),
                   'liquid_ml':tool.liquid.volume,'delta_liquid_ml':change,
                   'air_pressure_pa':self.pressure_pa,'outward_force_per_surface_n':force}
        return self.last
