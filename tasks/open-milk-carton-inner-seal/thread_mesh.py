"""Explicit helical contact meshes; no pose-coupling or automatic unscrewing.
Dimensions are nominal engineering inputs pending product measurements.
"""
import math
import numpy as np

def ridge(labkit,S,C,r0,r1,z0,pitch=.003,turns=1.25,thickness=.00065,color=(.93,.93,.91),segments_per_turn=48):
 out=[];n=int(math.ceil(turns*segments_per_turn))
 tris=np.array([[0,1,3],[0,3,2],[4,6,7],[4,7,5],[0,4,5],[0,5,1],[2,3,7],[2,7,6],[0,2,6],[0,6,4],[1,5,7],[1,7,3]])
 for i in range(n):
  pts=[]
  for a in [i*2*math.pi/segments_per_turn,(i+1)*2*math.pi/segments_per_turn]:
   z=z0+pitch*a/(2*math.pi)
   for r in [r0,r1]:
    for dz in [-thickness/2,thickness/2]:pts.append([r*math.cos(a),r*math.sin(a),z+dz])
  out.append(labkit.Mesh(np.asarray(pts)@S.T+C,tris,color,'hull'))
 return out
