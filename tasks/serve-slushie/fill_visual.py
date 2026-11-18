"""Render the same free-surface plane used by the volume model and airborne parcels."""
import numpy as np
from pxr import UsdGeom,Vt,Gf
from cup_geometry import FLOOR,HEIGHT,NOZZLE,SPOUT_EXIT_Z,inner_radius
from fill_model import plane_for

class FillVisual:
    def __init__(self,stage):
        self.mesh=UsdGeom.Mesh.Define(stage,'/World/SlushFill')
        self.mesh.CreateDoubleSidedAttr(True);self.mesh.CreateSubdivisionSchemeAttr('none');self.mesh.CreateDisplayColorAttr([(0.80,.035,.12)])
        self.stream=UsdGeom.Points.Define(stage,'/World/SlushStream');self.stream.CreateDisplayColorAttr([(0.80,.035,.12)])
        n=64;angles=np.arange(n)*2*np.pi/n;self.rays=np.column_stack((np.cos(angles),np.sin(angles)))
        faces=[list(range(n,2*n)),list(reversed(range(n)))]+[[i,(i+1)%n,(i+1)%n+n,i+n] for i in range(n)]
        self.mesh.CreateFaceVertexCountsAttr([len(f) for f in faces]);self.mesh.CreateFaceVertexIndicesAttr(sum(faces,[]))
    def draw(self,cup,model):
        ml=model.inside;self.mesh.GetVisibilityAttr().Set('inherited' if ml>.01 else 'invisible')
        if ml>.01:
            up=cup[2,:3];plane=plane_for(ml,up);direction=self.rays@up[:2];rb=inner_radius(FLOOR);slope=(inner_radius(HEIGHT)-rb)/(HEIGHT-FLOOR)
            z=np.clip((plane-direction*(rb-slope*FLOOR))/np.maximum(.0001,up[2]+direction*slope),FLOOR,HEIGHT)
            local=np.vstack((np.column_stack((self.rays*rb,np.full(len(z),FLOOR))),np.column_stack((self.rays*inner_radius(z)[:,None],z))))
            pts=(local@cup[:3,:3].T+cup[:3,3]).astype('f');self.mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(pts))
            self.mesh.GetExtentAttr().Set([Gf.Vec3f(*map(float,pts.min(0))),Gf.Vec3f(*map(float,pts.max(0)))])
        pts=[(float(NOZZLE[0]),float(NOZZLE[1]),SPOUT_EXIT_Z-.3*t-4.905*t*t) for t,_ in model.packets]
        self.stream.GetPointsAttr().Set(pts);self.stream.GetWidthsAttr().Set([.013]*len(pts))
