"""Visualize the reduced liquid ledger without changing physical collisions."""
import math
import numpy as np
from geometry_fluid import INTERIORS


def clipped_interior(interior,pose,level,sides=48):
    """Clip a faceted vessel interior against the horizontal free surface."""
    rings=[np.array([[r*math.cos(a),r*math.sin(a),z] for a in np.arange(sides)*2*math.pi/sides])
           for z,r in zip(interior.z,interior.radius)]
    polygons=[rings[0][::-1],rings[-1]]
    for lo,hi in zip(rings[:-1],rings[1:]):
        for i in range(sides):
            j=(i+1)%sides;polygons.append(np.array([lo[i],lo[j],hi[j],hi[i]]))
    vertices=[];counts=[];indices=[];intersections=[]
    def height(p):return pose[2,:3]@p+pose[2,3]-level
    def append(poly):
        if len(poly)<3:return
        start=len(vertices);vertices.extend(poly);counts.append(len(poly));indices.extend(range(start,start+len(poly)))
    for poly in polygons:
        clipped=[]
        previous=poly[-1];dp=height(previous)
        for current in poly:
            dc=height(current)
            if (dc<=0)!=(dp<=0):
                point=previous+(current-previous)*(dp/(dp-dc))
                clipped.append(point);intersections.append(point)
            if dc<=0:clipped.append(current)
            previous=current;dp=dc
        append(clipped)
    if len(intersections)>=3:
        boundary=np.unique(np.round(intersections,10),axis=0)
        world=boundary@pose[:3,:3].T+pose[:3,3]
        center=world[:,:2].mean(axis=0)
        order=np.argsort(np.arctan2(world[:,1]-center[1],world[:,0]-center[0]))
        append(boundary[order])
    return vertices,counts,indices


class LiquidVisuals:
    def __init__(self,coupling):
        import omni.usd
        from pxr import Gf,Sdf,UsdGeom,UsdShade
        self.Gf=Gf;self.UsdGeom=UsdGeom;self.coupling=coupling
        self.stage=omni.usd.get_context().get_stage();self.meshes={};self.shaders={}
        for name in INTERIORS:
            if name not in coupling.manifest['assets']:continue
            root='/World/envs/env_0/'+name
            mesh=UsdGeom.Mesh.Define(self.stage,root+'/liquid_visual')
            mesh.CreateSubdivisionSchemeAttr('none');mesh.CreateDoubleSidedAttr(True)
            mesh.CreatePointsAttr([]);mesh.CreateFaceVertexCountsAttr([]);mesh.CreateFaceVertexIndicesAttr([])
            material=UsdShade.Material.Define(self.stage,root+'/Looks/Liquid')
            shader=UsdShade.Shader.Define(self.stage,str(material.GetPath())+'/Surface')
            shader.CreateIdAttr('UsdPreviewSurface')
            shader.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(.2,.55,.7))
            shader.CreateInput('opacity',Sdf.ValueTypeNames.Float).Set(.78)
            shader.CreateInput('roughness',Sdf.ValueTypeNames.Float).Set(.28)
            material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(),'surface')
            UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
            self.meshes[name]=mesh;self.shaders[name]=shader

    def render(self):
        poses=self.coupling.poses()
        for name,mesh in self.meshes.items():
            liquid=self.coupling.ledger.vessels[name].liquid
            if liquid.volume<1e-8:
                mesh.MakeInvisible();continue
            pose=poses[name];level=self.coupling.fluid.surface(name,pose)
            points,counts,indices=clipped_interior(INTERIORS[name],pose,level)
            mesh.GetPointsAttr().Set([self.Gf.Vec3f(*p) for p in points])
            mesh.GetFaceVertexCountsAttr().Set(counts);mesh.GetFaceVertexIndicesAttr().Set(indices)
            mesh.MakeVisible()
