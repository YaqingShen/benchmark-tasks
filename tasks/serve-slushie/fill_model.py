"""Metered vertical discharge and quasi-static free surface in a tapered cup (SI / mL)."""
import numpy as np
from cup_geometry import FLOOR, HEIGHT, inner_radius, NOZZLE, SPOUT_EXIT_Z

N = 96
DZ = (HEIGHT-FLOOR)/N
Z = FLOOR + (np.arange(N)+.5)*DZ
R = inner_radius(Z)
AREA_ML = np.pi * R**2 * DZ * 1e6
CAP = float(AREA_ML.sum())
EDGE = float(inner_radius(HEIGHT))

def volume_below(plane, up):
    """Integrate submerged circular segments over slices in cup coordinates."""
    radial = np.hypot(*up[:2])
    if radial < 1e-6:
        heights = np.clip((plane-(Z-DZ/2)*up[2])/DZ, 0, 1) if up[2]>0 else np.clip((plane+(Z+DZ/2))/DZ,0,1)
        return float(AREA_ML@heights)
    ratio = np.clip((plane-Z*up[2])/(R*radial),-1,1)
    fraction=(np.arccos(-ratio)+ratio*np.sqrt(np.maximum(0,1-ratio**2)))/np.pi
    return float(fraction@AREA_ML)

def plane_for(ml,up):
    lo,hi=-.18,.18
    for _ in range(22):
        mid=(lo+hi)/2
        if volume_below(mid,up)<ml:lo=mid
        else:hi=mid
    return (lo+hi)/2

class Dispenser:
    def __init__(self):
        self.emitted=self.inside=self.spill=0.
        self.packets=[]
        self.time=0.

    def advance(self,cup,opening,dt):
        self.time+=dt
        ml=min(60*opening*dt, max(0.,900-self.emitted))
        if ml:
            self.emitted+=ml
            self.packets.append([0.,ml])
        up=cup[2,:3]
        origin=cup[:3,3]
        # Vertical ray through the nozzle meets the moving rim plane.
        normal=cup[:3,2]; rim=origin+HEIGHT*normal
        upright=normal[2]>.05
        x,y=NOZZLE[:2]; rim_z=rim[2]+(normal[0]*(rim[0]-x)+normal[1]*(rim[1]-y))/normal[2] if upright else -1.
        new=[]
        for age,amount in self.packets:
            z0=SPOUT_EXIT_Z-.3*age-4.905*age**2; age+=dt
            z=SPOUT_EXIT_Z-.3*age-4.905*age**2
            if upright and z<=rim_z<=z0 and np.linalg.norm(np.array([x,y,rim_z])-rim)<EDGE:
                self.inside+=amount
            elif z<.75:self.spill+=amount
            else:new.append([age,amount])
        self.packets=new
        lip=HEIGHT*up[2]-EDGE*np.hypot(*up[:2]); keep=volume_below(lip,up) if up[2]>0 else 0.
        loss=max(0.,self.inside-keep);self.spill+=loss;self.inside-=loss

    def metrics(self):
        return dict(in_cup_ml=self.inside,emitted_ml=self.emitted,spill_ml=self.spill,airborne_ml=sum(p[1] for p in self.packets),spill_fraction=self.spill/self.emitted if self.emitted else 0.,fill_fraction=self.inside/CAP,cup_capacity_ml=CAP)
