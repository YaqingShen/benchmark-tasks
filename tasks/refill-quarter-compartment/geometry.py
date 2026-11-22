"""Nominal quarter and drawer dimensions (metres, kilograms)."""
import math
import numpy as np
COUNT=40
R=.02426/2
T=.00175
MASS=.00567
L=.0378
BORE=.0130
START=np.array([.224,.0])
TARGET=np.array([.352,.0])
CELL=np.array([.108,.116])
RIM=.041
FLOOR=.004
PITCH=CELL[1]+.003
PAPER=(.70,.46,.26)
INK=(.85,.33,.075)
COIN=(.67,.67,.65)
TRAY=(.20,.23,.24)
def tube(sign):
    shapes=[]
    for i in range(20):
        a=math.tau*i/20
        for z,h,col in ((0,L-.007,PAPER),(-(L-.0035)/2,.0035,INK),((L-.0035)/2,.0035,INK)):
            shapes.append((((BORE+.00045)*math.cos(a),(BORE+.00045)*math.sin(a),z),(.0009,2*(BORE+.00045)*math.tan(math.pi/20)+.00015,h),col,a))
    return shapes
def tray():
    x,y=CELL
    b=[((0,0,FLOOR/2),(x+.006,3*PITCH+.003,FLOOR),TRAY)]
    for i in (-1.5,-.5,.5,1.5):
        b.append(((0,i*PITCH,(RIM+FLOOR)/2),(x+.006,.003,RIM-FLOOR),TRAY))
    for s in (-1,1):
        b.append(((s*(x+.003)/2,0,(RIM+FLOOR)/2),(.003,3*PITCH+.003,RIM-FLOOR),TRAY))
    # Steel counter-edge fixed to the far wall, over the receiving cash cup.
    # A flat steel strip on edge forms the working lip, with a rounded top.
    b.append(((.037,0,.036),(.043,.003,.031),(.46,.48,.47)))
    return b

def coin_mesh(labkit):
    """A chamfered quarter with planar faces and nominal Mint dimensions."""
    points=[]
    n=32
    edge=.00012
    for z,r in [(-T/2,R-edge),(-T/2+edge,R),(T/2-edge,R),(T/2,R-edge)]:
        for i in range(n):
            t=math.tau*i/n
            points.append((r*math.cos(t),r*math.sin(t),z))
    triangles=[(0,i+1,i) for i in range(1,n-1)]
    triangles.extend((3*n,3*n+i,3*n+i+1) for i in range(1,n-1))
    for layer in range(3):
        for i in range(n):
            a=layer*n+i;b=layer*n+(i+1)%n
            triangles.extend(((a,b,b+n),(a,b+n,a+n)))
    return labkit.Mesh(np.array(points),np.array(triangles),COIN,'hull')

def paper_vertices():
    """Local-space corners of all wall colliders of the paper tube."""
    corners=np.array([[x,y,z] for x in (-.5,.5) for y in (-.5,.5) for z in (-.5,.5)])
    vertices=[]
    for center,size,_,angle in tube(1):
        c,s=math.cos(angle),math.sin(angle)
        rotation=np.array([[c,-s,0],[s,c,0],[0,0,1]])
        vertices.extend((corners*size)@rotation.T+center)
    return np.array(vertices)

PAPER_VERTICES=paper_vertices()

def paper_bounds(center,rotation,side):
    """World bounds of the wall facets and closed circular end."""
    walls=PAPER_VERTICES@rotation.T+center
    axis=rotation[:,2]
    cap_center=center+side*(L/2-.00025)*axis
    cap_extent=(BORE+.00065)*np.sqrt(np.maximum(0.,1.-axis**2))+.00025*abs(axis)
    return np.minimum(walls.min(axis=0),cap_center-cap_extent),np.maximum(walls.max(axis=0),cap_center+cap_extent)
