"""Read the installed gripper collision meshes; do not change the robot."""
from pathlib import Path
from functools import lru_cache
import xml.etree.ElementTree as ET
import numpy as np
from yam_arena import YAM_URDF,station,constants,ik
from yam_arena.urdf import forward_kinematics, rpy_matrix


@lru_cache(maxsize=8)
def hand_collision_bounds(gap):
    """Per-mesh boxes in the TCP frame at the requested finger opening."""
    joint_positions = dict(zip(constants.ARM_JOINTS, constants.READY))
    joint_positions.update(joint7=-gap/2, joint8=-gap/2)
    poses = forward_kinematics(YAM_URDF, joint_positions)
    inverse = np.linalg.inv(poses['gripper'])
    tree = ET.parse(YAM_URDF).getroot()
    lo, hi = [], []
    for name in ('gripper', 'tip_left', 'tip_right'):
        transform = inverse @ poses[name]
        for collision in tree.find(f"link[@name='{name}']").findall('collision'):
            origin = collision.find('origin')
            offset = np.fromstring(origin.get('xyz', '0 0 0'), sep=' ')
            rotation = rpy_matrix(np.fromstring(origin.get('rpy', '0 0 0'), sep=' '))
            mesh = collision.find('geometry/mesh')
            vertices = stl_vertices(YAM_URDF.parent/mesh.get('filename'))
            vertices *= np.fromstring(mesh.get('scale', '1 1 1'), sep=' ')
            vertices = (vertices @ rotation.T + offset) @ transform[:3, :3].T
            vertices += transform[:3, 3] - constants.TCP_OFFSET
            lo.append(vertices.min(axis=0))
            hi.append(vertices.max(axis=0))
    return np.array(lo), np.array(hi)

def parcel_clearances(joints, base):
    """Conservative mesh-to-closed-parcel gaps (m), one per hand."""
    closed, opened = hand_collision_bounds(0.), hand_collision_bounds(.0939)
    gaps = []
    for side, start in (("left", 0), ("right", 7)):
        point, rotation = ik.tcp_pose(joints[start:start+6], station.BASES[side])
        bounds = [low + (high-low) * joints[start+6] for low, high in zip(closed, opened)]
        center, radius = (bounds[0] + bounds[1])/2, (bounds[1] - bounds[0])/2
        matrix = base[:3, :3].T @ rotation
        center = center @ matrix.T + (point-base[:3, 3]) @ base[:3, :3]
        radius = radius @ abs(matrix).T
        delta = np.maximum(np.maximum([-0.110, -0.080, -0.0006]-(center+radius),
                                       (center-radius)-[0.110, 0.080, 0.053]), 0)
        gaps.append(float(np.min(np.linalg.norm(delta, axis=1))))
    return gaps


def stl_vertices(path):
 b=Path(path).read_bytes();n=int(np.frombuffer(b,dtype='<u4',count=1,offset=80)[0]);a=np.frombuffer(b,dtype=np.dtype([('n','<f4',3),('v','<f4',(3,3)),('a','<u2')]),count=n,offset=84);return a['v'].reshape(-1,3).astype(float)

def finger_vertices():
 qs=dict(zip(constants.ARM_JOINTS,constants.READY));qs.update(joint7=0.,joint8=0.);fk=forward_kinematics(YAM_URDF,qs);inv=np.linalg.inv(fk['gripper']);pts=[]
 tree=ET.parse(YAM_URDF).getroot()
 for side in ['tip_left','tip_right']:
  link=tree.find(f"link[@name='{side}']");t=inv@fk[side]
  for coll in link.findall('collision'):
   ori=coll.find('origin');xyz=np.array([float(v) for v in ori.get('xyz','0 0 0').split()]);assert max(abs(float(v)) for v in ori.get('rpy','0 0 0').split())<1e-8
   m=coll.find('geometry/mesh');v=stl_vertices(YAM_URDF.parent/m.get('filename'));v*=np.array([float(x) for x in m.get('scale','1 1 1').split()]);v+=xyz
   pts.extend(v@t[:3,:3].T+t[:3,3]-np.array(constants.TCP_OFFSET))
 return np.array(pts)

VERTICES=None

def lowest_point(rotation):
 global VERTICES
 if VERTICES is None:VERTICES=finger_vertices()
 world=VERTICES@rotation.T;z=world[:,2].min();return world[world[:,2]<z+.0001].mean(0)

def press_rotation(theta=.45,side='right'):
 z=np.array([np.sin(theta),0.,-np.cos(theta)]);x=np.array([0.,1. if side=='right' else -1.,0.]);return np.column_stack([x,np.cross(z,x),z])

if __name__=='__main__':
 import json
 verts=finger_vertices();rows=[]
 for side,y,x in [('left',.045,.36),('right',-.040,.20)]:
  for theta in [0.,.25,.45,.65,.85]:
   r=press_rotation(theta,side);lowest=lowest_point(r);tcp=np.array([x,y,station.TABLE_TOP+.001])-lowest
   q,ep,er=ik.solve(tcp,ik.matrix_to_quat_xyzw(r),constants.READY,station.BASES[side],iters=800,damping=.018)
   rows.append({'side':side,'theta':theta,'tcp':tcp.tolist(),'ep':ep,'er':er,'lowest':lowest.tolist(),'q':q.tolist()})
 print(json.dumps({'finger_bounds_TCP':[verts.min(0).tolist(),verts.max(0).tolist()],'candidates':rows},indent=2))
