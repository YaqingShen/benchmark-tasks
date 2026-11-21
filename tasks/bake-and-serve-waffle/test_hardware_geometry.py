"""The real-style front cradle must not block closing or a complete turn."""
import math
import numpy as np
import task as T
from appliance_config import MODEL
from labkit import _rot

# The body shells stay entirely behind the cradle, including the metal trim.
cradle_back=MODEL['front_cradle_x_m']+.020
assert cradle_back < -(T.RH+.002)-.005
signs=np.array([[x,y,z] for x in (-1,1) for y in (-1,1) for z in (-1,1)])
cradle_front=MODEL['front_cradle_x_m']-.020
clearance=min(MODEL['front_cradle_inner_halfwidth_m'],.040)
base_left=MODEL['stand_base_center_x_m']-MODEL['stand_base_size_m'][0]/2
base_right=MODEL['stand_base_center_x_m']+MODEL['stand_base_size_m'][0]/2
assert base_left <= cradle_front and base_right >= cradle_back, 'Front support must sit on the base'

def overlaps_cradle_x(low,high):
    return low < cradle_back and high > cradle_front

checked=[]
for name,body in [('lower',T.SPIN),('lid',T.LID)]:
    for box in body.boxes:
        c,size=box[:2]
        turn=box[3] if len(box)>3 else (0.,0.)
        yaw,pitch=turn if isinstance(turn,(tuple,list)) else (turn,0.)
        pts=(signs*np.array(size)/2)@_rot((0.,pitch,yaw)).T+c
        if overlaps_cradle_x(pts[:,0].min(),pts[:,0].max()):
            radius=np.linalg.norm(pts[:,1:]-[0.,T.ZA],axis=1).max()
            assert radius < clearance-.001,(name,c,radius)
            checked.append((name,'box',float(radius)))
    for c,r,h,color,axis in body.cylinders:
        xhalf=h/2 if axis=='X' else r
        if overlaps_cradle_x(c[0]-xhalf,c[0]+xhalf):
            assert axis=='X','Unexpected cylinder intersects front cradle'
            radius=math.hypot(c[1],c[2]-T.ZA)+r
            assert radius < clearance-.001,(name,c,radius)
            checked.append((name,'cylinder',float(radius)))

handle_centre=np.array([(T.HANDLE_X0+T.HANDLE_X1)/2,0.,(T.HANDLE_Z0+T.HANDLE_Z1)/2])
assert any(np.allclose(b[0],handle_centre) for b in T.LID.boxes)
assert not any(np.allclose(b[0],handle_centre) for b in T.SPIN.boxes)
assert len(next(o for o in T.OBJECTS if o.name=='waffle').joints)==2,'No independent spring-slider joint'
assert checked
print('PASS: front-cradle shell clearance, full rotational swept clearance, single lid-attached handle and no slider')
