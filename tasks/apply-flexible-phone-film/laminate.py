"""Load-bearing PET ribs and backing in PhysX. All lengths are in metres."""
import math
import numpy as np
import labkit

N=10
DX=.014
W=.064
T=.00065
LINER_T=.00048
GAP=.00015
LEADING_Z=0.
BLUE=(.24,.64,.83)
PET=(.30,.56,.59)
EDGE=(.12,.28,.31)
PULL_X=.055
PULL_Y=.084
RELEASE_N=.25
K=145.
ANGLE=.11
COS=math.cos(ANGLE)
SIN=math.sin(ANGLE)


def name(i):return 'body' if i==0 else f's{i}'

def label(k,i):return f'{k}{i:02}'
LINKS=[]
def sheet_parts(x,y,z):
    parts=[]
    def body(k,i,pos,boxes,mass,mu):
        p=labkit.part(label(k,i),pos,mass=mass,friction=mu,boxes=boxes)
        parts.append(p)
        return p
    guide=labkit.part('guide',(x,y,z),fixed=True,boxes=[
        ((-.003,s*.037,-.002),(.028,.003,.002),EDGE) for s in (-1,1)])
    parts.append(guide)
    for i in range(N):
        p=(x+min(i,1)*DX+max(i-1,0)*DX*COS,y,z+max(i-1,0)*DX*SIN)
        b=[((DX/2,0,LEADING_Z if i==0 else 0.),(DX-.00015,W,T),PET)]
        if i==0:b.extend([((0,s*.034,LEADING_Z),(.008,.009,.0006),EDGE) for s in (-1,1)])
        spec=body('film',i,p,b,.0012,.6)
        if i:spec.meta['pitch']=-ANGLE
        if i:LINKS.append(('bend',label('film',i-1),label('film',i),(DX,0,0),(0,0,0)))
    LINKS.append(('slide','guide',label('film',0),(0,0,0),(0,0,0)))
    # The returned end lies below the sloping laminate, clear of the phone.
    sep=(T+LINER_T)/2+GAP
    z+=.0018
    b=[((PULL_X-.002,0,-.00135),(.024,W,LINER_T),BLUE),
       ((PULL_X,.063,-.0007),(.016,.067,.0012),BLUE),
       ((PULL_X,PULL_Y,.028),(.007,.040,.056),BLUE)]
    body('liner',0,(x,y,z),b,.004,.55)
    rail=labkit.part('pullrail',(x,y,z),fixed=True,boxes=[
        ((.157,.117,.001),(.32,.005,.008),EDGE)])
    parts.append(rail)
    LINKS.append(('draw','pullrail',label('liner',0),(0,0,0),(0,0,0)))
    for i in range(1,N):
        if i==1:
            body('liner',i,(x+3*DX,y,z),[((-DX/2,0,0),(DX-.0002,W,LINER_T),BLUE)],.0009,.20)
            LINKS.append(('fold',label('liner',0),label('liner',1),(3*DX,0,0),(0,0,0)))
            continue
        b=[((DX/2,0,0),(DX-.00015,W,LINER_T),BLUE)]
        if i==N-1:b.append(((DX/2,0,0),(DX-.00015,.078,LINER_T),BLUE))
        spec=body('liner',i,(x+DX+(i-1)*DX*COS+sep*SIN,y,z-.0018+(i-1)*DX*SIN-sep*COS),b,.0009,.20)
        spec.meta['pitch']=-ANGLE
        LINKS.append(('fold',label('liner',i-1),label('liner',i),
                      (-DX,0,0) if i==2 else (DX,0,0),(0,0,0)))
        LINKS.append(('release',label('film',i),label('liner',i),
                      (DX/2,0,-(T+LINER_T)/2-GAP),(DX/2,0,0)))
    return parts

def adhesive_links(stage,path,part):
    from pxr import UsdPhysics,Gf,Sdf
    if part.name!=label('liner',N-1):return
    parent=path.rsplit('/',1)[0]
    for k,a,b,p0,p1 in LINKS:
        dest=f'{path}/{k}_{a}_{b}'
        if k=='release':
            j=UsdPhysics.Joint.Define(stage,dest)
            # Three native linear drives form a finite traction-separation law.
            # The intact interface can shear slightly as the sheet bends.
            for axis in ('transX','transY','transZ'):
                drive=UsdPhysics.DriveAPI.Apply(j.GetPrim(),axis)
                drive.CreateTypeAttr('force');drive.CreateStiffnessAttr(K)
                drive.CreateDampingAttr(.07);drive.CreateTargetPositionAttr(0.)
                drive.CreateMaxForceAttr(RELEASE_N)
            if a != label('film',2):
                # A covered interior patch cannot peel before the fold reaches it.
                # A rigid interlayer tie transmits shear into the intact laminate.
                tie=UsdPhysics.FixedJoint.Define(stage,dest+'_interior')
                tie.CreateBody0Rel().SetTargets([Sdf.Path(f'{parent}/{a}/body')])
                tie.CreateBody1Rel().SetTargets([Sdf.Path(f'{parent}/{b}/body')])
                tie.CreateLocalPos0Attr(Gf.Vec3f(*p0));tie.CreateLocalPos1Attr(Gf.Vec3f(*p1))
                tie.CreateExcludeFromArticulationAttr(True)
        elif k in ('slide','draw'):
            j=UsdPhysics.PrismaticJoint.Define(stage,dest)
            j.CreateAxisAttr('X');j.CreateLowerLimitAttr(0.);j.CreateUpperLimitAttr(.009 if k=='slide' else .255)
        else:
            j=UsdPhysics.RevoluteJoint.Define(stage,dest)
            j.CreateAxisAttr('Y')
            limit=175 if k=='fold' else 35
            j.CreateLowerLimitAttr(-limit);j.CreateUpperLimitAttr(limit)
            d=UsdPhysics.DriveAPI.Apply(j.GetPrim(),'angular')
            d.CreateStiffnessAttr((.000012 if k=='fold' else .002)/57.2958)
            d.CreateDampingAttr(.000025/57.2958)
        j.CreateBody0Rel().SetTargets([Sdf.Path(f'{parent}/{a}/body')])
        j.CreateBody1Rel().SetTargets([Sdf.Path(f'{parent}/{b}/body')])
        j.CreateLocalPos0Attr(Gf.Vec3f(*p0));j.CreateLocalPos1Attr(Gf.Vec3f(*p1))
        j.CreateExcludeFromArticulationAttr(True)
        j.CreateCollisionEnabledAttr(k=='release')
