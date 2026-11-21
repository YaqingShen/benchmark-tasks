"""Read-only geometric evidence for kitchen-tong extraction and transfer to a side plate.

Tong-tip/food proximity is a geometric contact proxy, not a force sensor. Loosening
also requires observed food displacement, so touching a stationary waffle is
insufficient. Native hand collision-mesh hulls disqualify direct food contact; synthetic
tests without meshes use conservative per-piece AABBs.
"""
import math
import numpy as np
from scoring_geometry import articulated_vertices, separation


def initial():
    return dict(hand_evidence=True, direct_food_contact=False, tool_picked=False,
                tip_contact_steps=0, max_loosen_mm=0., loosened=False,
                tool_stowed=False, tool_carried=False, carry_steps=0,
                invalid_serving=False)


def food_points(T, state):
    # Circumscribed 32-sided cylinder bounds the round core without the large
    # diagonal overestimate of an axis-aligned cylinder box.
    spec=next(s for s in T.OBJECTS if s.name=='belgian_waffle')
    pts=[]; signs=np.array([[x,y,z] for x in (-1,1) for y in (-1,1) for z in (-1,1)])
    for c,size,*_ in spec.boxes:
        pts.extend(signs*np.asarray(size)/2+c)
    for c,r,h,_,axis in spec.cylinders:
        assert axis=='Z'
        for a in np.arange(32)*2*math.pi/32:
            for z in (-h/2,h/2):
                pts.append(np.asarray(c)+[r/math.cos(math.pi/32)*math.cos(a),r/math.cos(math.pi/32)*math.sin(a),z])
    return np.asarray(pts)@state.rotation().T+state.pos


def food_piece_bounds(T,state):
    """Separate food collision pieces: empty space above the round core is
    not food merely because taller lugs occur elsewhere on the waffle."""
    spec=next(s for s in T.OBJECTS if s.name=='belgian_waffle')
    signs=np.array([[x,y,z] for x in (-1,1) for y in (-1,1) for z in (-1,1)])
    pieces=[]
    for c,size,*_ in spec.boxes:
        v=(signs*np.asarray(size)/2+c)@state.rotation().T+state.pos
        pieces.append((v.min(0),v.max(0)))
    for c,r,h,_,axis in spec.cylinders:
        half=np.full(3,r);half['XYZ'.index(axis)]=h/2
        v=(signs*half+c)@state.rotation().T+state.pos
        pieces.append((v.min(0),v.max(0)))
    return pieces


def update(T,tr,objects,hands,hulls=None):
    if hands is None:
        tr['hand_evidence']=False
    f=objects.get('belgian_waffle'); tool=objects.get('kitchen_tongs')
    if f is None or tool is None:
        return
    fp=food_points(T,f); fb=(fp.min(0),fp.max(0))
    if (T.food_contact.contacts_food(T,f,hulls) if hulls is not None else
        hands and any(separation(piece,h)<.001 for piece in food_piece_bounds(T,f) for h in hands)):
        tr['direct_food_contact']=True
    # The handle has left the cup while a hand is near the actual grip point.
    grip=tool.to_world(T.TONG_GRIP)
    handle=(grip-.025,grip+.025)
    if (tool.pos[2]>T.TONG_HOME[2]+.030 and tool.rotation()[2,2]>math.cos(math.radians(15.))
        and tool.joints.get('jaw',0.)<-.001 and hands and any(separation(handle,h)<.005 for h in hands)):
        tr['tool_picked']=True
    # Both terminal blades must approach the food and the passive jaw must
    # have visibly closed. A lifted waffle then establishes actual loosening.
    from labkit import _rot
    signs=np.array([[x,y,z] for x in (-1,1) for y in (-1,1) for z in (-1,1)])
    near=[]
    for sx,shift in ((-T.TONG_SEP/2,0.),(T.TONG_SEP/2,tool.joints.get('jaw',0.))):
        c,size,_,angles=T._tong_half(sx)[1]
        v=(signs*np.asarray(size)/2)@_rot((0.,angles[1],angles[0])).T+np.asarray(c)+[shift,0.,0.]
        v=v@tool.rotation().T+tool.pos
        near.append(separation((v.min(0),v.max(0)),fb)<.002)
    contact=all(near) and tool.joints.get('jaw',0.)<-.006
    tr['tip_contact_steps']=tr['tip_contact_steps']+1 if contact else 0
    w=objects['waffle'];spin=w.joints.get('spin',0.);lid=w.joints.get('lid',0.)
    if tr['phase']>=2 and spin<T.TURNING and lid>=T.LID_OPEN_MIN and contact and tr['tool_picked']:
        off,_=T.food_pose(objects)
        lift=float(off[2])
        tr['max_loosen_mm']=max(tr['max_loosen_mm'],1000*lift)
        if tr['tip_contact_steps']>=3 and lift>=.005 and not tr['direct_food_contact']:
            tr['loosened']=True
    # Sustained blade contact while the lifted food leaves the grid laterally.
    off,_=T.food_pose(objects)
    carrying=bool(tr['loosened'] and contact and spin<=T.HOME_TOL
                  and np.linalg.norm(off[:2])>.100 and off[2]>.015)
    tr['carry_steps']=tr['carry_steps']+1 if carrying else 0
    if tr['carry_steps']>=3:
        tr['tool_carried']=True
    if T.on_plate(objects) and not tr['tool_carried']:
        tr['invalid_serving']=True
    tool_bounds=np.concatenate([v for _,v in articulated_vertices(tool,next(s for s in T.OBJECTS if s.name=='kitchen_tongs'))])
    tb=(tool_bounds.min(0),tool_bounds.max(0))
    tr['tool_stowed']=bool(tr['loosened'] and np.linalg.norm(tool.pos-np.asarray(T.TONG_HOME))<.025
        and tool.joints.get('jaw',0.)>-.006
        and tool.rotation()[2,2]>math.cos(math.radians(10.))
        and np.linalg.norm(tool.lin_vel)<.03 and np.linalg.norm(tool.ang_vel)<.1
        and hands and all(separation(tb,h)>=.005 for h in hands))


def valid(tr):
    return bool(tr['hand_evidence'] and not tr['direct_food_contact'] and not tr['invalid_serving']
                and tr['loosened'] and tr['tool_carried'] and tr['tool_stowed'])
