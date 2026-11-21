"""Read-only geometric checks; never changes robot or object poses."""
from functools import lru_cache
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


def collision_bounds(state, spec):
    """Exact world AABB of the task's box/cylinder collision primitives."""
    rotation = state.rotation()
    lows, highs = [], []
    for centre, size, _ in spec.boxes:
        centre = state.to_world(centre)
        half = np.abs(rotation) @ (np.asarray(size) / 2)
        lows.append(centre - half)
        highs.append(centre + half)
    for centre, radius, height, _, axis in spec.cylinders:
        centre = state.to_world(centre)
        direction = rotation[:, 'XYZ'.index(axis)]
        half = np.abs(direction) * height / 2 + radius * np.sqrt(np.maximum(0, 1 - direction ** 2))
        lows.append(centre - half)
        highs.append(centre + half)
    return np.min(lows, axis=0), np.max(highs, axis=0)


def maximum_surface_speed(state, spec):
    """Conservative speed bound over every collision primitive, including rotation.

    Rigid-body velocity is affine in position; its norm is convex, so the box
    corners bound the entire primitive. Cylinder bounding boxes are conservative.
    """
    corners = []
    primitives = [(np.asarray(c), np.asarray(size) / 2) for c, size, _ in spec.boxes]
    for centre, radius, height, _, axis in spec.cylinders:
        half = np.full(3, radius)
        half['XYZ'.index(axis)] = height / 2
        primitives.append((np.asarray(centre), half))
    signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
    for centre, half in primitives:
        corners.extend(centre + signs * half)
    offsets = np.asarray(corners) @ state.rotation().T
    velocities = np.asarray(state.lin_vel) + np.cross(np.asarray(state.ang_vel), offsets)
    return float(max(np.linalg.norm(state.lin_vel), np.linalg.norm(velocities, axis=1).max()))


@lru_cache(maxsize=1)
def _hand_corners():
    from yam_arena import YAM_URDF
    from yam_arena.urdf import rpy_matrix

    root = ET.parse(YAM_URDF).getroot()
    result = {}
    for name in ('gripper', 'tip_left', 'tip_right'):
        vertices = []
        for collision in root.find(f"link[@name='{name}']").findall('collision'):
            mesh = collision.find('geometry/mesh')
            blob = (Path(YAM_URDF).parent / mesh.get('filename')).read_bytes()
            count = int(np.frombuffer(blob, dtype='<u4', count=1, offset=80)[0])
            dtype = np.dtype([('normal', '<f4', 3), ('v', '<f4', (3, 3)), ('attr', '<u2')])
            v = np.frombuffer(blob, dtype=dtype, count=count, offset=84)['v'].reshape(-1, 3).astype(float)
            v *= np.fromstring(mesh.get('scale', '1 1 1'), sep=' ')
            origin = collision.find('origin')
            rotation = rpy_matrix(np.fromstring(origin.get('rpy', '0 0 0'), sep=' '))
            vertices.append(v @ rotation.T + np.fromstring(origin.get('xyz', '0 0 0'), sep=' '))
        vertices = np.concatenate(vertices)
        low, high = vertices.min(0), vertices.max(0)
        result[name] = np.array([[x, y, z] for x in (low[0], high[0])
                                 for y in (low[1], high[1]) for z in (low[2], high[2])])
    return result


def hand_bounds(environment):
    """Conservative AABBs from native simulated link poses and installed URDF meshes."""
    from yam_arena.urdf import quat_xyzw_to_matrix
    from labkit import _np

    base = environment.unwrapped
    origin = _np(base.scene.env_origins[0])
    boxes = []
    for arm in ('left_robot', 'right_robot'):
        robot = base.scene[arm]
        poses = _np(robot.data.body_link_pose_w)[0]
        for name, corners in _hand_corners().items():
            pose = poses[list(robot.body_names).index(name)]
            pts = corners @ quat_xyzw_to_matrix(pose[3:7]).T + pose[:3] - origin
            boxes.append((pts.min(0), pts.max(0)))
    return boxes


def separation(a, b):
    """Minimum Euclidean separation of two axis-aligned bounding boxes."""
    return float(np.linalg.norm(np.maximum(np.maximum(a[0] - b[1], b[0] - a[1]), 0)))


def articulated_vertices(state, spec):
    """Collision vertices in world space, including every articulated child.

    Box corners are exact; cylinders use conservative bounding boxes. Joint
    shapes and anchors are authored in the root's zero-pose frame by labkit.
    """
    from labkit import _rot
    signs = np.array([[x,y,z] for x in (-1,1) for y in (-1,1) for z in (-1,1)])
    joints = {j.name:j for j in spec.joints}
    transforms = {'': np.eye(4)}
    def transform(name):
        if name in transforms:
            return transforms[name]
        j=joints[name]; q=state.joints.get(name,j.initial); t=np.eye(4)
        axis='XYZ'.index(j.axis)
        if j.kind=='slide':
            t[axis,3]=q
        else:
            angles=np.zeros(3); angles[axis]=q
            t[:3,:3]=_rot(angles); a=np.asarray(j.anchor)
            t[:3,3]=a-t[:3,:3]@a
        transforms[name]=transform(j.parent)@t
        return transforms[name]
    result=[]
    for name,body in [('',spec),*joints.items()]:
        points=[]
        for box in body.boxes:
            c,size=box[:2]; turn=box[3] if len(box)>3 else (0.,0.)
            yaw,pitch=turn if isinstance(turn,(tuple,list)) else (turn,0.)
            points.extend((signs*np.asarray(size)/2)@_rot((0.,pitch,yaw)).T+np.asarray(c))
        for c,r,h,colour,axis in body.cylinders:
            half=np.full(3,r); half['XYZ'.index(axis)]=h/2
            points.extend(signs*half+c)
        for mesh in getattr(body,'meshes',()):
            if mesh.collide!='none': points.extend(mesh.points)
        if points:
            t=transform(name); p=np.asarray(points)@t[:3,:3].T+t[:3,3]
            result.append((name,p@state.rotation().T+state.pos))
    return result


def articulated_bounds(state,spec):
    return [(v.min(0),v.max(0)) for _,v in articulated_vertices(state,spec)]
