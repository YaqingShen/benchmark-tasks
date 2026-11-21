"""Vented air exits at the perimeter. A closed contact contour traps air."""
import numpy as np
from scipy.ndimage import label, binary_fill_holes
from shapely.geometry import box
from shapely.ops import unary_union, triangulate


class AirInterface:
    def __init__(self, pitch_mm=.5):
        self.pitch = pitch_mm
        self.sealed = np.zeros((round(140/self.pitch), round(64/self.pitch)), bool)
        self.trapped = np.zeros_like(self.sealed)
        self.x, self.y = np.meshgrid((np.arange(self.sealed.shape[0])+.5)*self.pitch,
                                     (np.arange(self.sealed.shape[1])+.5)*self.pitch, indexing='ij')
        self.outlines = []

    def touch(self, mask):
        if not (mask & ~self.sealed & ~self.trapped).any():
            return False
        self.sealed |= mask & ~self.trapped
        new = binary_fill_holes(self.sealed) & ~self.sealed
        if not (new & ~self.trapped).any():
            return False
        self.trapped |= new
        components, n = label(self.trapped)
        self.outlines = []
        for k in range(1, n+1):
            runs = []
            for x in np.flatnonzero((components == k).any(axis=1)):
                edges = np.flatnonzero(np.diff(np.r_[False, components[x] == k, False]))
                runs.extend(box(x*self.pitch, start*self.pitch, (x+1)*self.pitch, end*self.pitch)
                            for start, end in edges.reshape(-1, 2))
            boundary = unary_union(runs).exterior
            self.outlines.append(np.asarray(boundary.simplify(0).coords[:-1]))
        return True

    def press(self, edge, loaded_bands, first_backed):
        edge = (edge[:, :2]+[0, .032])*1000
        a, b = edge
        v = b-a
        if np.linalg.norm(v) < 1.:
            return False
        t = np.clip(((self.x-a[0])*v[0]+(self.y-a[1])*v[1])/(v@v), 0, 1)
        mask = (self.x-a[0]-v[0]*t)**2+(self.y-a[1]-v[1]*t)**2 <= 3.5**2
        band = np.floor(self.x/14).astype(int)
        return self.touch(mask & np.asarray(loaded_bands)[band] & (band < first_backed))

    def display(self, stage, path, origin):
        from pxr import UsdGeom, Vt, Gf
        from shapely.geometry import Polygon
        mesh = UsdGeom.Mesh.Define(stage, path)
        pts = []
        for outline in self.outlines:
            region = Polygon(outline)
            for triangle in triangulate(region):
                if region.covers(triangle):
                    pts.extend([(origin[0]+x/1000, origin[1]+(y-32)/1000, origin[2]+.002)
                                for x, y in triangle.exterior.coords[:3]])
        mesh.CreatePointsAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*v) for v in pts]))
        mesh.CreateFaceVertexCountsAttr().Set(Vt.IntArray([3]*(len(pts)//3)))
        mesh.CreateFaceVertexIndicesAttr().Set(Vt.IntArray(list(range(len(pts)))))
        mesh.CreateSubdivisionSchemeAttr('none')
        mesh.CreateDoubleSidedAttr(True)
        mesh.CreateDisplayColorAttr([(.9, .85, .77)])
