"""Reject bare robot contact with either cooking cavity, from the first step.
Uses recorded robot collision-mesh hulls and the current lid/spin transforms.
The guard is a conservative envelope of the grid and its recessed pockets;
external handles, latch and housing are outside this envelope.
"""
from functools import lru_cache
import math
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import linprog
from scoring_geometry import separation

@lru_cache(maxsize=4)
def cavity(radius,z0,z1):
    n=48;r=radius/math.cos(math.pi/n)
    return np.array([[r*math.cos(a),r*math.sin(a),z]
                     for a in np.arange(n)*2*math.pi/n for z in (z0,z1)])

def zones(T,w):
    axis=np.array([0.,0.,T.ZA]);h=np.asarray(T.HINGE)
    spin=T._Rx(w.joints.get('spin',0.));lid=T._Ry(w.joints.get('lid',0.))
    for name,z0,z1 in [('lower',T.ZA-T.GRID_FLOOR-.003,T.ZA+.0005),
                       ('upper',T.ZA-.0005,T.ZA+T.GRID_FLOOR+.003)]:
        p=cavity(T.RP+.007,z0,z1)
        if name=='upper':p=(p-h)@lid.T+h
        p=(p-axis)@spin.T+axis
        yield name,p@w.rotation().T+w.pos

def contact(T,w,hulls):
    if hulls is None:return None
    bounds=[(h.min(0),h.max(0)) for h in hulls];planes={}
    for name,p in zones(T,w):
        pb=(p.min(0),p.max(0));cp=None
        for i,(h,hb) in enumerate(zip(hulls,bounds)):
            if separation(pb,hb)>.0005:continue
            if cp is None:cp=ConvexHull(p).equations
            if i not in planes:planes[i]=ConvexHull(h).equations
            eq=np.vstack([cp,planes[i]])
            res=linprog([0.,0.,0.,-1.],A_ub=np.column_stack([eq[:,:3],np.ones(len(eq))]),
                        b_ub=-eq[:,3],bounds=[(None,None)]*4,method='highs')
            if not res.success:raise RuntimeError('Cooking-surface guard failed: '+res.message)
            if res.x[3]>=-.00025:return name
    return None
