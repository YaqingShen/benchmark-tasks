"""Top-view check that the whole package is outside the magazine opening."""
import math
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
from hotdrink import BOTTOM,WRAP_R

class SleeveOutline:
    def __init__(self):
        r=WRAP_R
        self.points={}
        # Enclose the circular top sheet, plus both faces of the folded seal.
        a=np.arange(32)*math.tau/32
        top=np.column_stack((r*np.cos(a)/math.cos(math.pi/32),r*np.sin(a)/math.cos(math.pi/32),np.full(32,.0622)))
        tab=np.array([[x,y,z] for x in (-.001,.001) for y in (-math.pi*r/2,math.pi*r/2) for z in (.055,.079)])
        self.points['body']=np.vstack((top,tab))
        for i in range(4):
            theta=i*math.pi/2
            local=[]
            for angle in np.linspace(theta-math.pi/4,theta+math.pi/4,7):
                for z in (BOTTOM,.062):
                    local.append([(r+.00042)*math.cos(angle)-r*math.cos(theta),
                                  (r+.00042)*math.sin(angle)-r*math.sin(theta),z-.062])
            self.points['fold'+str(i)]=np.array(local)

    def clearance(self,wrapper,x_origin,to_numpy):
        poses=to_numpy(wrapper.data.body_link_pose_w)[0]
        points=np.concatenate([Rotation.from_quat(p[3:]).apply(self.points[name])+p[:3]
                               for name,p in zip(wrapper.body_names,poses)])
        xy=points[:,:2]-[x_origin,0.]
        hull=ConvexHull(xy)
        if np.max(hull.equations[:,2])<=0.:
            return 0.
        vertices=xy[hull.vertices]
        edge=np.roll(vertices,-1,axis=0)-vertices
        t=np.clip(-np.sum(vertices*edge,axis=1)/np.sum(edge*edge,axis=1),0.,1.)
        return float(np.linalg.norm(vertices+t[:,None]*edge,axis=1).min())
