"""Sweep the physical lid through release, retraction and open travel."""
from math import cos, sin
import numpy as np
from check_geometry import overlaps
import mechanism as m

class Model:
    Mesh=staticmethod(lambda pts,tris,col,kind:(pts,kind))
    Joint=staticmethod(lambda name,**params:dict(name=name,**params))
    part=staticmethod(lambda *args,**params:params)

def projected(box,theta=0.,delta=0.):
    (x,y,z),(w,h,d),_=box
    pivot=np.array([m.HINGE_Y,.014])
    corners=np.array([[y+a*h/2,z+b*d/2] for a,b in ((-1,-1),(1,-1),(1,1),(-1,1))])
    rot=np.array([[cos(theta),-sin(theta)],[sin(theta),cos(theta)]])
    return x-w/2,x+w/2,(corners-pivot)@rot.T+pivot+[delta,0]

def conflict(a,b):
    if a[1]<=b[0]+1e-8 or b[1]<=a[0]+1e-8:return False
    if any(a[2][:,i].max()<=b[2][:,i].min()+1.e-8 or b[2][:,i].max()<=a[2][:,i].min()+1.e-8 for i in range(2)):return False
    return overlaps(a[2],b[2])

def hits(theta,slide,parts):
    fixed=[projected(box) for box in parts['boxes']]
    moving=[projected(box,delta=slide) for box in parts['joints'][3]['boxes']]
    solids=[projected(box,theta) for box in parts['joints'][4]['boxes']]
    return [(i,name,j) for i,a in enumerate(solids) for name,items in [('shell',fixed),('slide',moving)] for j,b in enumerate(items) if conflict(a,b)]

if __name__=='__main__':
    case=m.mechanism(Model,'spec',(0,0,0),m.CODE)
    assert hits(.025,0.,case),'The closed slide should retain the lid'
    for angle,travel in ([(0.,s) for s in np.linspace(0,.0057,12)]
                         +[(v,.0057) for v in np.linspace(0,.48,60)]
                         +[(.15,s) for s in np.linspace(0,.0057,12)]):
        collisions=hits(angle,travel,case)
        assert not collisions,(angle,travel,collisions)
    print('Closed slide retains lid. Released lid and returning slide clear the shell.')
