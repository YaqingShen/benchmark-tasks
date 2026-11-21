"""Closed convex wall sectors leave the paper cup and the sleeve hollow in PhysX."""
import math
import numpy as np
import labkit
from hotdrink import BOTTOM, FOOT, PITCH, R, RIM_TOP, WRAP_R, PANEL_MASS

def sector(a,b,lo,hi,z0,z1,color,collision="hull"):
    # Each end section has four corners, wound around its hollow centre.
    verts=[]
    for angle in (a,b):
        for r,z in ((lo[0],z0),(lo[1],z0),(hi[1],z1),(hi[0],z1)):
            verts.append((r*math.cos(angle),r*math.sin(angle),z))
    triangles=[]
    pts=np.array(verts)
    for face in ((0,1,2,3),(4,7,6,5),(0,4,5,1),(1,5,6,2),(2,6,7,3),(3,7,4,0)):
        for i,j,k in ((face[0],face[1],face[2]),(face[0],face[2],face[3])):
            if np.dot(np.cross(pts[j]-pts[i],pts[k]-pts[i]),pts[i]-pts.mean(axis=0))<0:
                j,k=k,j
            triangles.append((i,j,k))
    return labkit.Mesh(pts,np.array(triangles),color,collision)

def paper_cup(paper):
    walls=[]
    contacts=[]
    sides=32
    upper=R-.0017
    lower=FOOT-.0002
    height=-BOTTOM-.0006
    slope=(upper-lower)/height
    for i in range(sides):
        a,b=i*math.tau/sides,(i+1)*math.tau/sides
        mid=(a+b)/2
        walls.append(sector(a,b,(lower-.00023,lower+.00017),(upper-.00023,upper+.00017),-height,0.,paper,"none"))
        def tab(radius,z,size,pitch=0.):
            return ((radius*math.cos(mid),radius*math.sin(mid),z),size,paper,(mid,pitch))
        contacts.append(tab((upper+lower)/2,-height/2,(.00038,.0053,math.hypot(height,upper-lower)),math.atan(slope)))
        # Outer face endpoints lie on the manufacturer's 80 mm brim circle.
        outer=R*math.cos(math.pi/sides)
        width=2*R*math.sin(math.pi/sides)
        contacts.append(tab(outer-.0008,.0005,(.0016,width,.0020)))
        # Approximate the internal base profile that sets nesting pitch.
        # Each upper cup rests on the lower cup's internal raised annulus.
        # The outer wall at the magazine separator stays smooth below the lip.
        ledge=FOOT-.0012
        span=2*ledge*math.tan(math.pi/sides)
        contacts.append(tab(ledge,BOTTOM+PITCH/2,(.0018,span,PITCH)))
    return walls,contacts

def sleeve(film,seam):
    # Four hinged groups approximate the bending of the sleeve below a joined
    # top. Circumferential creasing between the strips is not resolved.
    r=WRAP_R
    top=.062
    bottom=BOTTOM
    h=top-bottom
    joints=[]
    faces=24
    half=math.pi/faces
    for q in range(4):
        angle=q*math.pi/2
        panels=[]
        strips=[]
        for k in range(6):
            a=angle-math.pi/4+k*2*half
            t=a+half
            panels.append(sector(a,a+2*half,(r,r+.00042),(r,r+.00042),bottom,top,film,"none"))
            radial=r*math.cos(half)+.00021
            strips.append(((radial*math.cos(t),radial*math.sin(t),(top+bottom)/2),
                           (.00042,2*r*math.sin(half),h),film,t))
        joints.append(labkit.Joint(f"fold{q}",kind="hinge",axis="Y" if q%2==0 else "X",
            anchor=(r*math.cos(angle),r*math.sin(angle),top),boxes=tuple(strips),meshes=tuple(panels),
            lower=-.17,upper=.17,stiffness=.0012,damping=.00007,mass=3*PANEL_MASS))
    return joints,[((0,0,.067),(.002,math.pi*r,.024),seam)]
