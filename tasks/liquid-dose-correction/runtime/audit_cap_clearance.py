"""Independent convex separating-axis check of the initial cap fit."""
import itertools,json,math
from pathlib import Path
import numpy as np
from pxr import Usd,UsdGeom,UsdPhysics,Gf

ROOT=Path(__file__).resolve().parent


def colliders(name,z):
    stage=Usd.Stage.Open(str(ROOT/'assets/mechanisms'/f'{name}.usda'))
    result=[]
    for prim in stage.Traverse():
        if not prim.HasAPI(UsdPhysics.CollisionAPI):continue
        if prim.IsA(UsdGeom.Sphere):
            sphere=UsdGeom.Sphere(prim)
            transform=UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            center=np.array(transform.Transform(Gf.Vec3d(0,0,0)))+[0,0,z]
            scales=[np.linalg.norm(transform.TransformDir(Gf.Vec3d(*(1 if j==i else 0 for j in range(3)))))for i in range(3)]
            if np.ptp(scales)>1e-8:raise ValueError('Nonuniformly scaled sphere is unsupported')
            radius=sphere.GetRadiusAttr().Get()*scales[0]
            result.append((str(prim.GetPath()),{'center':center,'radius':radius},center-radius,center+radius))
            continue
        if prim.IsA(UsdGeom.Mesh):points=UsdGeom.Mesh(prim).GetPointsAttr().Get()
        elif prim.IsA(UsdGeom.Cylinder):
            c=UsdGeom.Cylinder(prim);r=c.GetRadiusAttr().Get();h=c.GetHeightAttr().Get()
            points=[Gf.Vec3f(r*math.cos(a),r*math.sin(a),height)for height in(-h/2,h/2)for a in np.arange(32)*2*math.pi/32]
        else:raise ValueError(prim.GetTypeName())
        transform=UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        p=np.array([list(transform.Transform(Gf.Vec3d(v)))for v in points]);p[:,2]+=z
        result.append((str(prim.GetPath()),p,p.min(axis=0),p.max(axis=0)))
    return result


def unit_unique(vectors):
    a=np.asarray(vectors).reshape(-1,3);length=np.linalg.norm(a,axis=1);a=a[length>1e-10]/length[length>1e-10,None]
    return np.unique(np.round(a,8),axis=0)


def faces_and_edges(points):
    edges=unit_unique([b-a for a,b in itertools.combinations(points,2)])
    normals=[]
    for a,b,c in itertools.combinations(points,3):
        n=np.cross(b-a,c-a);norm=np.linalg.norm(n)
        if norm<1e-12:continue
        n/=norm;distance=(points-a)@n
        if distance.min()>-1e-9 or distance.max()<1e-9:normals.append(n)
    return unit_unique(normals),edges


def point_triangle_distance(point,a,b,c):
    """Closest Euclidean distance, including face, edge and vertex regions."""
    ab=b-a;ac=c-a;ap=point-a;d1=ab@ap;d2=ac@ap
    if d1<=0 and d2<=0:return np.linalg.norm(ap)
    bp=point-b;d3=ab@bp;d4=ac@bp
    if d3>=0 and d4<=d3:return np.linalg.norm(bp)
    vc=d1*d4-d3*d2
    if vc<=0 and d1>=0 and d3<=0:return np.linalg.norm(point-(a+ab*d1/(d1-d3)))
    cp=point-c;d5=ab@cp;d6=ac@cp
    if d6>=0 and d5<=d6:return np.linalg.norm(cp)
    vb=d5*d2-d1*d6
    if vb<=0 and d2>=0 and d6<=0:return np.linalg.norm(point-(a+ac*d2/(d2-d6)))
    va=d3*d6-d5*d4
    if va<=0 and d4-d3>=0 and d5-d6>=0:
        return np.linalg.norm(point-(b+(c-b)*(d4-d3)/((d4-d3)+(d5-d6))))
    denom=va+vb+vc
    return np.linalg.norm(point-(a+ab*vb/denom+ac*vc/denom))


def sphere_penetration(sphere,points):
    center=sphere['center'];faces=[];inside=True
    for a,b,c in itertools.combinations(points,3):
        n=np.cross(b-a,c-a);length=np.linalg.norm(n)
        if length<1e-12:continue
        n/=length;d=(points-a)@n
        if d.min()<-1e-9 and d.max()>1e-9:continue
        faces.append((a,b,c))
        projection=(center-a)@n
        if projection<d.min()-1e-9 or projection>d.max()+1e-9:inside=False
    distance=0. if inside else min(point_triangle_distance(center,*f)for f in faces)
    return sphere['radius']-distance


def main():
    left=colliders('bottle_A',0.);right=colliders('bottle_A_cap',.185)
    cache={};hits=[];candidates=0
    for na,a,alo,ahi in left:
        for nb,b,blo,bhi in right:
            if np.min(np.minimum(ahi,bhi)-np.maximum(alo,blo))<=1e-7:continue
            candidates+=1
            if isinstance(a,dict) or isinstance(b,dict):
                if isinstance(a,dict) and isinstance(b,dict):
                    depth=a['radius']+b['radius']-np.linalg.norm(a['center']-b['center'])
                else:depth=sphere_penetration(a,b) if isinstance(a,dict) else sphere_penetration(b,a)
                if depth>1e-7:hits.append({'bottle':na,'cap':nb,'penetration_lower_bound_m':float(depth)})
                continue
            for name,pts in((na,a),(nb,b)):
                if name not in cache:cache[name]=faces_and_edges(pts)
            an,ae=cache[na];bn,be=cache[nb]
            axes=unit_unique(np.vstack([an,bn,np.cross(ae[:,None,:],be[None,:,:]).reshape(-1,3)]))
            pa=a@axes.T;pb=b@axes.T
            overlap=np.minimum(pa.max(axis=0),pb.max(axis=0))-np.maximum(pa.min(axis=0),pb.min(axis=0))
            if overlap.min()>1e-7:hits.append({'bottle':na,'cap':nb,'penetration_lower_bound_m':float(overlap.min())})
    report={'type':'STATIC_INITIAL_CONVEX_FIT_AUDIT','candidate_pairs':candidates,'penetrating_pairs':hits}
    output=ROOT.parent.parent/'outputs/formulation-prototype/cap-clearance-audit.json'
    output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
