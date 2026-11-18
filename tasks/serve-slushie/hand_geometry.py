"""Read-only geometric checks; never changes robot or object poses."""
from functools import lru_cache
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


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
    def _np(value):
        value = getattr(value, "torch", value)
        return value.detach().cpu().numpy()

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
