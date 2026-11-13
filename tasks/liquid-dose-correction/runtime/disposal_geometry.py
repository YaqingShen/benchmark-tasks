"""Whole-pipette containment from its actual collider geometry."""
import itertools
import numpy as np
from pipette_geometry import STEM_LENGTH_M


def pipette_bounds_in_bin(stem,bulbs,bin_pose):
    inverse=np.linalg.inv(bin_pose);stem=inverse@stem;limits=[]
    # Exact coordinate extrema of the two circular stem/connector colliders.
    axis=stem[:3,2]
    for radius,height,z in [(.0025,STEM_LENGTH_M,STEM_LENGTH_M/2),
                            (.004,.007,STEM_LENGTH_M+.0035)]:
        center=stem[:3,3]+axis*z
        extent=np.abs(axis)*height/2+radius*np.sqrt(np.maximum(0.,1-axis*axis))
        limits.extend([center-extent,center+extent])
    corners=np.array(list(itertools.product(*[[-v,v]for v in (.003,.0105,.024)])))
    for bulb in bulbs:
        transform=inverse@bulb
        limits.extend(corners@transform[:3,:3].T+transform[:3,3])
    points=np.array(limits)
    return points.min(axis=0),points.max(axis=0)


def fully_contained(lower,upper,bounds):
    # Sub-millimetre tolerance is limited to native contact-solver overlap at
    # the walls/floor. The entire tool must remain below the open rim.
    lo=np.array([-bounds['half_x'],-bounds['half_y'],bounds['floor_z']])
    hi=np.array([bounds['half_x'],bounds['half_y'],bounds['rim_z']])
    return bool(np.all(lower>=lo-.0005) and np.all(upper[:2]<=hi[:2]+.0005)
                and upper[2]<hi[2])
