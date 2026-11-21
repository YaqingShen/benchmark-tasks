"""Read-only convex collision-mesh guard for direct robot/food contact.

Native and offline checks reconstruct the same hand geometry from recorded
joint positions and gripper poses. No contact or object state is modified.
"""
from functools import lru_cache
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import linprog
from scoring_geometry import separation


@lru_cache(maxsize=1)
def hand_meshes():
    from yam_arena import YAM_URDF
    from yam_arena.urdf import rpy_matrix
    root=ET.parse(YAM_URDF).getroot();result={}
    for name in ('gripper','tip_left','tip_right'):
        pieces=[]
        for col in root.find(f"link[@name='{name}']").findall('collision'):
            m=col.find('geometry/mesh')
            blob=(Path(YAM_URDF).parent/m.get('filename')).read_bytes()
            n=int(np.frombuffer(blob,dtype='<u4',count=1,offset=80)[0])
            dtype=np.dtype([('normal','<f4',3),('v','<f4',(3,3)),('attr','<u2')])
            v=np.frombuffer(blob,dtype=dtype,count=n,offset=84)['v'].reshape(-1,3).astype(float)
            v*=np.fromstring(m.get('scale','1 1 1'),sep=' ')
            origin=col.find('origin')
            if origin is not None:
                v=v@rpy_matrix(np.fromstring(origin.get('rpy','0 0 0'),sep=' ')).T+np.fromstring(origin.get('xyz','0 0 0'),sep=' ')
            v=np.unique(v,axis=0)
            pieces.append(v[ConvexHull(v).vertices])
        result[name]=pieces
    return result


def hand_hulls_from_robots(robots):
    from yam_arena import YAM_URDF
    from yam_arena.urdf import forward_kinematics,quat_xyzw_to_matrix
    result=[]
    for robot in robots.values():
        poses=forward_kinematics(YAM_URDF,dict(zip(robot['joint_names'],robot['joint_pos'])),root='gripper')
        g=np.asarray(robot['gripper_pose']);R=quat_xyzw_to_matrix(g[3:])
        for name,pieces in hand_meshes().items():
            tf=poses[name]
            for pts in pieces:
                result.append((pts@tf[:3,:3].T+tf[:3,3])@R.T+g[:3])
    return result


def hand_hulls(env):
    from labkit import _np
    base=env.unwrapped;origin=_np(base.scene.env_origins[0]);robots={}
    for name in ('left_robot','right_robot'):
        r=base.scene[name];pose=_np(r.data.body_link_pose_w)[0][list(r.body_names).index('gripper')].copy()
        pose[:3]-=origin
        robots[name]={'gripper_pose':pose,'joint_pos':_np(r.data.joint_pos)[0],'joint_names':list(r.joint_names)}
    return hand_hulls_from_robots(robots)


@lru_cache(maxsize=256)
def _box_points(centre,size,turn):
    from labkit import _rot
    signs=np.array([[x,y,z] for x in (-1,1) for y in (-1,1) for z in (-1,1)])
    return (signs*np.array(size)/2)@_rot((0.,turn[1],turn[0])).T+centre


def food_pieces(T,food):
    spec=next(s for s in T.OBJECTS if s.name=='belgian_waffle');pieces=[]
    for c,size,_,turn in spec.boxes:
        pieces.append(_box_points(tuple(c),tuple(size),tuple(turn)))
    for c,r,h,_,axis in spec.cylinders:
        assert axis=='Z'
        # Circumscribed polygon is conservative by <0.5 mm radially.
        n=32;r=r/np.cos(np.pi/n)
        pieces.append(np.array([np.array(c)+[r*np.cos(a),r*np.sin(a),z]
                               for a in np.arange(n)*2*np.pi/n for z in (-h/2,h/2)]))
    return [p@food.rotation().T+food.pos for p in pieces]


def contacts_food(T,food,hulls):
    boxes=[(h.min(0),h.max(0)) for h in hulls];planes={}
    for p in food_pieces(T,food):
        pb=(p.min(0),p.max(0));fp=None
        for i,(h,hb) in enumerate(zip(hulls,boxes)):
            if separation(pb,hb)>.0005:continue
            if fp is None:fp=ConvexHull(p).equations
            if i not in planes:planes[i]=ConvexHull(h).equations
            eq=np.vstack([fp,planes[i]])
            # Largest sphere common to the two convex polyhedra. A small
            # margin covers native contact offsets without a whole-hand AABB.
            res=linprog([0.,0.,0.,-1.],A_ub=np.column_stack([eq[:,:3],np.ones(len(eq))]),
                        b_ub=-eq[:,3],bounds=[(None,None)]*4,method='highs')
            if not res.success:raise RuntimeError('Food contact geometry failed: '+res.message)
            if res.x[3]>=-.00025:return True
    return False
