"""Small CPU-only convex-polyhedron screen; NumPy and the standard library.

This module screens CLOSED CONVEX solids represented by triangles. It does not
perform continuous collision detection, collision response or distance queries.
``axis_gap_m`` is separation/overlap of projected intervals on one SAT axis,
NOT the Euclidean clearance or a measured penetration depth. A margin asks
whether the projected gaps are all at most that margin; "near" is conservative.
Invalid or oversized geometry raises GeometryError rather than being called clear.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import struct
import numpy as np


class GeometryError(ValueError):
    """Input is not a supported, closed convex three-dimensional solid."""


@dataclass(frozen=True)
class ConvexMesh:
    vertices: np.ndarray
    triangles: np.ndarray
    normals: np.ndarray
    edge_directions: np.ndarray
    bounds: np.ndarray
    name: str = ""


@dataclass(frozen=True)
class ScreenResult:
    status: str  # separated, near, touching, overlap
    axis_gap_m: float
    axis: tuple[float, float, float]
    tested_axes: int
    broad_phase_only: bool = False

    @property
    def intersects(self) -> bool:
        """Touching within the requested numerical tolerance counts as intersection."""
        return self.status in ("touching", "overlap")


def _readonly(values, dtype=float):
    result=np.array(values,dtype=dtype,copy=True)
    result.flags.writeable=False
    return result


def _unique_axes(values, *, min_norm=1e-10):
    values=np.asarray(values,dtype=float).reshape(-1,3)
    norms=np.linalg.norm(values,axis=1)
    values=values[norms>min_norm]
    if not len(values):return np.empty((0,3),dtype=float)
    values=values/np.linalg.norm(values,axis=1)[:,None]
    # Opposite axis directions have the same SAT projections. Canonicalize by
    # the largest component, which is better conditioned than a tiny first one.
    largest=np.argmax(np.abs(values),axis=1)
    signs=np.sign(values[np.arange(len(values)),largest])
    values=values*signs[:,None]
    _,indices=np.unique(np.round(values,10),axis=0,return_index=True)
    return values[np.sort(indices)]


def prepare_mesh(vertices, faces, *, name="", convex_tolerance=1e-7,
                 max_vertices=10000, max_triangles=20000) -> ConvexMesh:
    """Validate and prepare a convex solid; faces may be triangles or polygons.

    Vertices within a scale-relative welding tolerance are merged. All remaining
    triangles must have nonzero area, and every undirected edge must occur twice.
    Face winding is corrected outward before volume and support-plane checks.
    The maximum allowed support-plane convexity defect is convex_tolerance (m).
    """
    v=np.asarray(vertices,dtype=float)
    if v.ndim!=2 or v.shape[1]!=3 or len(v)<4 or len(v)>max_vertices:
        raise GeometryError(f"{name}: expected 4..{max_vertices} finite 3D vertices")
    if not np.isfinite(v).all():raise GeometryError(f"{name}: non-finite vertices")
    if convex_tolerance<0 or not np.isfinite(convex_tolerance):
        raise GeometryError("convex_tolerance must be finite and nonnegative")
    diagonal=float(np.linalg.norm(np.ptp(v,axis=0)))
    if diagonal<=1e-12:raise GeometryError(f"{name}: zero-size geometry")
    weld=max(diagonal*1e-9,1e-12)
    # Subtract before rounding, avoiding integer precision loss far from origin.
    quantized=np.round((v-v.min(axis=0))/weld)
    _,first,inverse=np.unique(quantized,axis=0,return_index=True,return_inverse=True)
    v=v[first]
    if len(v)<4:raise GeometryError(f"{name}: fewer than four unique vertices")
    tri=[]
    for face in faces:
        ids=np.asarray(face)
        if ids.ndim!=1 or len(ids)<3 or not np.issubdtype(ids.dtype,np.integer):
            raise GeometryError(f"{name}: faces must contain at least three integer indices")
        if ids.min()<0 or ids.max()>=len(inverse):raise GeometryError(f"{name}: invalid face index")
        ids=inverse[ids]
        for i in range(1,len(ids)-1):
            tri.append((ids[0],ids[i],ids[i+1]))
            if len(tri)>max_triangles:raise GeometryError(f"{name}: too many triangles")
    t=np.asarray(tri,dtype=np.int64)
    if t.shape[0]<4:raise GeometryError(f"{name}: too few triangles")
    xyz=v[t]
    raw=np.cross(xyz[:,1]-xyz[:,0],xyz[:,2]-xyz[:,0])
    norm=np.linalg.norm(raw,axis=1)
    if np.any(norm<=max(diagonal**2*1e-12,1e-20)):
        raise GeometryError(f"{name}: zero or numerically degenerate triangle area")
    center=v.mean(axis=0)
    outward=np.einsum('ij,ij->i',raw,xyz[:,0]-center)
    flip=outward<0
    t[flip]=t[flip][:,[0,2,1]]
    raw[flip]*=-1
    normals=raw/norm[:,None]
    xyz=v[t]
    relative=xyz-center
    volume=float(np.einsum('ij,ij->i',relative[:,0],np.cross(relative[:,1],relative[:,2])).sum()/6)
    if volume<=max(diagonal**3*1e-12,1e-21):
        raise GeometryError(f"{name}: zero or numerically degenerate solid volume")
    # Closed manifold and consistent directed winding after outward correction.
    edges={}
    for index,(a,b,c) in enumerate(t):
        for u,w in ((a,b),(b,c),(c,a)):
            key=(min(int(u),int(w)),max(int(u),int(w)))
            edges.setdefault(key,[]).append((index,int(u),int(w)))
    for key,uses in edges.items():
        if len(uses)!=2:raise GeometryError(f"{name}: open or nonmanifold edge {key}")
        if uses[0][1:]!=tuple(reversed(uses[1][1:])):
            raise GeometryError(f"{name}: inconsistent surface winding")
    for first_face in range(0,len(normals),128):
        ns=normals[first_face:first_face+128]
        origins=xyz[first_face:first_face+128,0]
        defect=(v-center)@ns.T-np.einsum('ij,ij->i',origins-center,ns)
        if np.max(defect)>convex_tolerance:
            raise GeometryError(f"{name}: non-convex solid (support-plane defect {np.max(defect):.3g} m)")
    # Omit triangulation diagonals between coplanar triangles. Extra axes would
    # be valid but expensive; only true polyhedron edges are needed by SAT.
    edge_vectors=[]
    for (a,b),uses in edges.items():
        if np.dot(normals[uses[0][0]],normals[uses[1][0]])<1-1e-9:
            edge_vectors.append(v[b]-v[a])
    directions=_unique_axes(edge_vectors,min_norm=1e-12)
    if len(directions)<3:raise GeometryError(f"{name}: insufficient three-dimensional edge directions")
    return ConvexMesh(_readonly(v),_readonly(t,np.int64),_readonly(_unique_axes(normals)),
                      _readonly(directions),_readonly([v.min(axis=0),v.max(axis=0)]),str(name))


def read_stl(path, *, name=None, scale=(1.,1.,1.), max_triangles=20000,
             convex_tolerance=1e-7) -> ConvexMesh:
    """Read binary or ASCII STL, including binary STL whose header starts 'solid'."""
    path=Path(path)
    if path.stat().st_size>max_triangles*1000+1024:
        raise GeometryError(f"{path.name}: STL exceeds the bounded reader size")
    data=path.read_bytes()
    count=struct.unpack_from('<I',data,80)[0] if len(data)>=84 else None
    binary=count is not None and len(data)==84+50*count
    if binary:
        if not 4<=count<=max_triangles:raise GeometryError(f"{path.name}: invalid STL triangle count")
        dtype=np.dtype([('normal','<f4',(3,)),('vertices','<f4',(3,3)),('attribute','<u2')])
        vertices=np.frombuffer(data,offset=84,count=count,dtype=dtype)['vertices'].astype(float).reshape(-1,3)
    else:
        try:text=data.decode('ascii')
        except UnicodeDecodeError as exc:raise GeometryError(f"{path.name}: truncated or invalid binary STL") from exc
        if not re.match(r'^\s*solid\b',text,re.I) or not re.search(r'^\s*endsolid\b',text,re.I|re.M):
            raise GeometryError(f"{path.name}: not a complete ASCII STL")
        rows=re.findall(r'^\s*vertex\s+(\S+)\s+(\S+)\s+(\S+)\s*$',text,re.I|re.M)
        count=len(rows)//3
        if len(rows)%3 or not 4<=count<=max_triangles:
            raise GeometryError(f"{path.name}: invalid ASCII STL vertex count")
        if any(len(re.findall(pattern,text,re.I|re.M))!=count for pattern in
               (r'^\s*facet\s+normal\b',r'^\s*outer\s+loop\s*$',r'^\s*endloop\s*$',r'^\s*endfacet\s*$')):
            raise GeometryError(f"{path.name}: malformed ASCII STL facets")
        try:vertices=np.asarray(rows,dtype=float)
        except ValueError as exc:raise GeometryError(f"{path.name}: invalid ASCII STL coordinate") from exc
    scale=np.asarray(scale,dtype=float)
    if scale.shape!=(3,) or not np.isfinite(scale).all() or np.any(scale==0):
        raise GeometryError("scale must contain three finite nonzero values")
    vertices=vertices*scale
    faces=np.arange(len(vertices),dtype=np.int64).reshape(-1,3)
    # STL duplicates each triangle's vertices, so allow that count before welding.
    return prepare_mesh(vertices,faces,name=name or path.name,
                        max_vertices=3*max_triangles,max_triangles=max_triangles,
                        convex_tolerance=convex_tolerance)


def transform_mesh(mesh: ConvexMesh, transform) -> ConvexMesh:
    """Apply a finite rigid 4x4 transform; scaling belongs in read/prepare instead."""
    matrix=np.asarray(transform,dtype=float)
    if matrix.shape!=(4,4) or not np.isfinite(matrix).all():raise GeometryError("Invalid rigid transform")
    r=matrix[:3,:3]
    if not np.allclose(matrix[3],[0,0,0,1],atol=1e-10,rtol=0) or not np.allclose(r.T@r,np.eye(3),atol=1e-7,rtol=0) or abs(np.linalg.det(r)-1)>1e-7:
        raise GeometryError("Transform must be a proper rigid transform, without scaling or reflection")
    vertices=mesh.vertices@r.T+matrix[:3,3]
    return ConvexMesh(_readonly(vertices),mesh.triangles,_readonly(mesh.normals@r.T),
                      _readonly(mesh.edge_directions@r.T),
                      _readonly([vertices.min(axis=0),vertices.max(axis=0)]),mesh.name)


def convex_screen(a: ConvexMesh, b: ConvexMesh, *, margin=0., tolerance=1e-8,
                  max_cross_axes=200000, batch_axes=128) -> ScreenResult:
    """Screen convex solids with AABB rejection and complete polyhedral SAT.

    'separated': at least one projected gap exceeds margin + tolerance.
    'near': disjoint solids, but no tested axis exceeds the requested margin.
    'touching': the largest projected gap is within +/- numerical tolerance.
    'overlap': strict overlap on every candidate SAT axis beyond tolerance.
    Margin does not inflate the solids into Euclidean offset surfaces. 'near'
    can include false-positive proximity; axis_gap_m is never a distance claim.
    """
    if not np.isfinite([margin,tolerance]).all() or margin<0 or tolerance<0:
        raise ValueError("margin and tolerance must be finite and nonnegative")
    if batch_axes<1 or max_cross_axes<1:raise ValueError("axis limits must be positive")
    gaps=np.maximum(b.bounds[0]-a.bounds[1],a.bounds[0]-b.bounds[1])
    index=int(np.argmax(gaps))
    if gaps[index]>margin+tolerance:
        axis=np.eye(3)[index]
        return ScreenResult('separated',float(gaps[index]),tuple(axis),3,True)
    # Centering keeps projections precise for millimeter parts at meter coordinates.
    origin=(a.bounds.mean(axis=0)+b.bounds.mean(axis=0))/2
    va=a.vertices-origin;vb=b.vertices-origin
    best_gap=-np.inf;best_axis=None;tested=0
    # Face normals often prove separation before the more expensive edge pairs.
    # Only a completed face+edge test can return overlap, touching, or near.
    for phase in (0,1):
        if phase==0:
            axes=_unique_axes(np.vstack([a.normals,b.normals]))
        else:
            n_cross=len(a.edge_directions)*len(b.edge_directions)
            if n_cross>max_cross_axes:
                raise GeometryError(f"SAT axis budget exceeded ({n_cross}>{max_cross_axes}); result is unknown")
            cross=np.cross(a.edge_directions[:,None,:],b.edge_directions[None,:,:]).reshape(-1,3)
            axes=_unique_axes(cross)
        for start in range(0,len(axes),batch_axes):
            block=axes[start:start+batch_axes]
            pa=va@block.T;pb=vb@block.T
            current=np.maximum(pb.min(axis=0)-pa.max(axis=0),pa.min(axis=0)-pb.max(axis=0))
            j=int(np.argmax(current));tested+=len(block)
            if current[j]>best_gap:best_gap=float(current[j]);best_axis=block[j]
            if best_gap>margin+tolerance:
                return ScreenResult('separated',best_gap,tuple(float(x) for x in best_axis),tested)
    if best_axis is None:raise GeometryError("No usable SAT axes; result is unknown")
    status='near' if best_gap>tolerance else 'touching' if best_gap>=-tolerance else 'overlap'
    return ScreenResult(status,best_gap,tuple(float(x) for x in best_axis),tested)
