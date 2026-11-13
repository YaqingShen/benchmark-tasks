"""Geometry-driven reduced fluid transport for a future rigid-body backend.

This module accepts measured vessel transforms. It determines rim outflow, tip
immersion and airborne capture without a controller-specified destination.
It is not CFD: no slosh, viscosity calibration, surface tension or splash. Cap
engagement and bulb compression must come from native physical mechanisms.
"""
from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class Interior:
    z: tuple
    radius: tuple
    mouth_z: float
    mouth_radius: float

    def upright_height(self, volume_ml):
        """Invert the exact piecewise frustum volume, in metres."""
        remaining=max(0.,float(volume_ml))*1e-6
        for z0,z1,r0,r1 in zip(self.z[:-1],self.z[1:],self.radius[:-1],self.radius[1:]):
            height=z1-z0
            capacity=math.pi*height*(r0*r0+r0*r1+r1*r1)/3
            if remaining<=capacity:
                slope=(r1-r0)/height
                if abs(slope)<1e-12:return z0+remaining/(math.pi*r0*r0)
                radius=np.cbrt(r0**3+3*slope*remaining/math.pi)
                return z0+(radius-r0)/slope
            remaining-=capacity
        return self.z[-1]

    def quadrature(self, nz=96, nr=8, na=48):
        """Axisymmetric volume quadrature; coordinates in m, weights in mL."""
        z0,z1=self.z[0],self.z[-1]
        heights=z0+(np.arange(nz)+.5)*(z1-z0)/nz
        radii=np.interp(heights,self.z,self.radius)
        radial=np.sqrt((np.arange(nr)+.5)/nr)
        angles=(np.arange(na)+.5)*2*np.pi/na
        r=radii[:,None,None]*radial[None,:,None]
        x=np.broadcast_to(r*np.cos(angles)[None,None,:],(nz,nr,na))
        y=np.broadcast_to(r*np.sin(angles)[None,None,:],(nz,nr,na))
        z=np.broadcast_to(heights[:,None,None],(nz,nr,na))
        weights=np.broadcast_to((math.pi*radii*radii*(z1-z0)/nz*1e6/(nr*na))[:,None,None],(nz,nr,na))
        return np.column_stack((x.ravel(),y.ravel(),z.ravel())),weights.ravel()


INTERIORS={
 "bottle_A":Interior((.004,.119,.174,.194),(.0405,.0405,.0185,.0185),.194,.0185),
 "bottle_B":Interior((.004,.119,.174,.194),(.0405,.0405,.0185,.0185),.194,.0185),
 "cylinder_A":Interior((.013,.194),(.011,.011),.194,.011),
 "cylinder_B":Interior((.013,.194),(.011,.011),.194,.011),
 "beaker":Interior((.004,.121),(.0415,.0415),.121,.0415),
 "waste":Interior((.004,.07),(.023,.028),.07,.028),
}


@dataclass
class Packet:
    liquid: object
    position: np.ndarray
    velocity: np.ndarray
    source: str


class GeometryFluid:
    def __init__(self, env, table_z=0., discharge_coefficient=.35):
        self.env=env
        self.table_z=float(table_z)
        self.cd=float(discharge_coefficient)
        self.points={k:s.quadrature() for k,s in INTERIORS.items()if k in env.vessels}
        self.packets=[]
        self.surface_cache={}
        self.events=[]

    @staticmethod
    def validate_pose(pose):
        m=np.asarray(pose,dtype=float)
        if m.shape!=(4,4) or not np.isfinite(m).all():
            raise ValueError("A finite 4x4 measured transform is required")
        if not np.allclose(m[3],[0,0,0,1],atol=1e-7) or not np.allclose(m[:3,:3].T@m[:3,:3],np.eye(3),atol=1e-5) or np.linalg.det(m[:3,:3])<.99999:
            raise ValueError("Pose must be rigid and right-handed")
        return m

    def surface(self,name,pose):
        m=self.validate_pose(pose)
        volume=self.env.vessels[name].liquid.volume
        if volume<=1e-9:return -math.inf
        interior=INTERIORS[name]
        h=interior.upright_height(volume)
        horizontal=float(np.linalg.norm(m[2,:2]));vertical=float(m[2,2])
        if horizontal<1e-9 and vertical>0:
            return h+m[2,3]
        # For a plane wholly within a straight cylindrical section, tilt does
        # not change its centre height. This is exact, including shallow doses;
        # coarse volume quadrature otherwise introduces millimetre steps.
        if vertical>0:
            for i,(z0,z1,r0,r1) in enumerate(zip(interior.z[:-1],interior.z[1:],interior.radius[:-1],interior.radius[1:])):
                if r0!=r1 or not z0<=h<=z1:continue
                level=vertical*h
                lower=max(vertical*z+horizontal*r for z,r in zip(interior.z[:i+1],interior.radius[:i+1]))
                upper=min(vertical*z-horizontal*r for z,r in zip(interior.z[i+1:],interior.radius[i+1:]))
                if lower<level<upper:return level+m[2,3]
        points,weights=self.points[name]
        rotation_key=tuple(np.round(m[2,:3],8))
        cached=self.surface_cache.get(name)
        if cached is None or cached[0]!=rotation_key:
            z=points@m[2,:3]
            order=np.argsort(z)
            cached=(rotation_key,z[order],np.cumsum(weights[order]))
            self.surface_cache[name]=cached
        _,levels,cumulative=cached
        return float(np.interp(volume,cumulative,levels)+m[2,3])

    def outlet(self,name,pose):
        m=self.validate_pose(pose);s=INTERIORS[name]
        down=-m[2,:2]
        norm=np.linalg.norm(down)
        direction=down/norm if norm>1e-9 else np.array([1.,0.])
        local=np.r_[direction*s.mouth_radius,s.mouth_z]
        return m[:3,:3]@local+m[:3,3]

    def emit(self,source,liquid,position,velocity):
        if liquid.volume<=1e-12:
            return
        packet=Packet(liquid,np.asarray(position,dtype=float).copy(),np.asarray(velocity,dtype=float).copy(),source)
        if packet.position.shape!=(3,) or packet.velocity.shape!=(3,) or not np.isfinite(np.r_[packet.position,packet.velocity]).all():
            raise ValueError("Finite packet position/velocity required")
        self.packets.append(packet)
        self.env.inflight.append(liquid)

    def pour(self,name,pose,dt):
        vessel=self.env.vessels[name]
        if vessel.capped or vessel.broken or vessel.liquid.volume<=1e-9:
            return 0.
        m=self.validate_pose(pose)
        lip=self.outlet(name,m)
        head=max(0.,self.surface(name,m)-lip[2])
        if head<=0:
            return 0.
        s=INTERIORS[name]
        # A circular mouth approximated as a broad-crested local weir.
        width=2*math.sqrt(max(0.,s.mouth_radius**2-(s.mouth_radius-min(head,s.mouth_radius))**2))
        q=self.cd*(2/3)*math.sqrt(2*9.81)*width*head**1.5*1e6
        retained=self.env.cfg["cylinder_retained_film_ml"] if name.startswith('cylinder_') else 0.
        amount=min(q*dt,max(0.,vessel.liquid.volume-retained))
        liquid=self.env.take_from(name,amount)
        outward=m[:3,2].copy();outward[2]=min(-.05,outward[2])
        outward/=np.linalg.norm(outward)
        self.emit(name,liquid,lip+outward*.001,outward*math.sqrt(2*9.81*head))
        return amount

    def tip_immersion(self,tip_position,poses):
        tip=np.asarray(tip_position,dtype=float)
        if tip.shape!=(3,) or not np.isfinite(tip).all():
            raise ValueError("Finite measured tip position required")
        candidates=[]
        for name,m in poses.items():
            if name not in INTERIORS or name not in self.env.vessels:
                continue
            vessel=self.env.vessels[name]
            if vessel.capped or vessel.broken or vessel.liquid.volume<=1e-9:
                continue
            m=self.validate_pose(m);s=INTERIORS[name]
            local=m[:3,:3].T@(tip-m[:3,3])
            radius=np.interp(local[2],s.z,s.radius)
            if s.z[0]<=local[2]<=s.mouth_z and np.linalg.norm(local[:2])<radius-.0008 and tip[2]<self.surface(name,m)-.0003:
                candidates.append(name)
        if len(candidates)>1:
            raise ValueError("Overlapping liquid interiors: ambiguous tip immersion")
        return candidates[0] if candidates else None

    def advance(self,poses,dt,pour=True):
        if not 0<dt<=.02:
            raise ValueError("Fluid time step must lie in (0, 0.02] seconds")
        poses={name:self.validate_pose(m) for name,m in poses.items() if name in INTERIORS and name in self.env.vessels}
        if pour:
            for name,m in poses.items():
                self.pour(name,m,dt)
        live=[]
        gravity=np.array([0.,0.,-9.81])
        for packet in self.packets:
            previous=packet.position.copy()
            packet.position+=packet.velocity*dt+.5*gravity*dt*dt
            packet.velocity+=gravity*dt
            hits=[]
            for name,m in poses.items():
                if name==packet.source:
                    continue
                s=INTERIORS[name]
                if m[2,2]<.1:
                    continue
                p=m[:3,:3].T@(previous-m[:3,3]);q=m[:3,:3].T@(packet.position-m[:3,3])
                if p[2]>=s.mouth_z and q[2]<s.mouth_z:
                    fraction=(p[2]-s.mouth_z)/(p[2]-q[2])
                    xy=p[:2]+fraction*(q[:2]-p[:2])
                    if np.linalg.norm(xy)<s.mouth_radius:
                        hits.append((fraction,name))
            target=min(hits)[1] if hits else None
            landed=bool(hits) or packet.position[2]<=self.table_z
            if landed:
                volume=packet.liquid.volume
                # Remove transit ownership before depositing, preserving a
                # single ledger location for every parcel at all times.
                index=next(i for i,p in enumerate(self.env.inflight) if p is packet.liquid)
                self.env.inflight.pop(index)
                before=self.env.vessels['beaker'].liquid.mass(self.env.cfg['density_g_ml'])
                self.env._receive(target,packet.liquid)
                if packet.source in self.env.pipettes and target=='beaker':
                    self.env.credit_pipette_correction(packet.source,before)
                self.events.append({"time_s":self.env.time,"source":packet.source,"target":target or "spill","volume_ml":volume})
            else:
                live.append(packet)
        self.packets=live
        self.env.advance(dt)
