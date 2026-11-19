"""Metered vertical discharge and quasi-static free surface in a tapered cup (SI / mL)."""
import numpy as np
from trial_setup import PAN_X, PAN_Y, TRAY_X, TRAY_Y, CONNECTOR_X, CONNECTOR_Y
from cup_geometry import FLOOR, HEIGHT, inner_radius, NOZZLE, SPOUT_EXIT_Z

HOPPER_ML = 900.0
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
        self.tray_spill = self.pan_spill = self.cover_spill = 0.
        self.nozzle_ml = self.closed_drips_ml = 0.
        self.cover_bounds = []
        self.equipment_exposed = False

    def advance(self,cup,opening,dt):
        self.time+=dt
        incoming = min(60*opening*dt, max(0., HOPPER_ML-self.emitted-self.nozzle_ml))
        if opening > 0:
            trapped = min(incoming, max(0., .45-self.nozzle_ml))
            self.nozzle_ml += trapped
            ml = incoming - trapped
        else:
            ml = min(self.nozzle_ml, .90*dt)
            self.nozzle_ml -= ml
            self.closed_drips_ml += ml
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
            elif z <= .860:
                # A missed vertical stream falls onto the drip tray under the nozzle.
                self.spill += amount
                self.tray_spill += amount
            else:new.append([age,amount])
        self.packets=new
        lip=HEIGHT*up[2]-EDGE*np.hypot(*up[:2]); keep=volume_below(lip,up) if up[2]>0 else 0.
        loss=max(0.,self.inside-keep);self.spill+=loss;self.inside-=loss
        if loss:
            # Project overflow from the lowest point on the rim onto the tray.
            horizontal = np.linalg.norm(up[:2])
            edge_xy = -EDGE * up[:2] / horizontal if horizontal > 1e-8 else np.zeros(2)
            exit_point = rim + cup[:3, :2] @ edge_xy
            x, y, z = exit_point
            on_tray = TRAY_X[0] <= x <= TRAY_X[1] and TRAY_Y[0] <= y <= TRAY_Y[1] and z >= .860
            intercept_z = .860 if on_tray else .748
            if any(lo[0] <= x <= hi[0] and lo[1] <= y <= hi[1] and
                   lo[2] <= z and hi[2] >= intercept_z for lo, hi in self.cover_bounds):
                self.cover_spill += loss
            elif on_tray:
                self.tray_spill += loss
            elif PAN_X[0] <= x <= PAN_X[1] and PAN_Y[0] <= y <= PAN_Y[1] and z >= .748:
                self.pan_spill += loss
            elif CONNECTOR_X[0] <= x <= CONNECTOR_X[1] and CONNECTOR_Y[0] <= y <= CONNECTOR_Y[1] and z >= .768:
                self.equipment_exposed = True

    def metrics(self):
        return dict(in_cup_ml=self.inside,emitted_ml=self.emitted,spill_ml=self.spill,airborne_ml=sum(p[1] for p in self.packets),spill_fraction=self.spill/self.emitted if self.emitted else 0.,fill_fraction=self.inside/CAP,cup_capacity_ml=CAP)
