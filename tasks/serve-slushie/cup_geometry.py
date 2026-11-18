"""Geometry of the slush machine, the 16 oz clear PET cup and the serve stand (world frame, metres).
Cup profiles are (height, radius) in the cup frame, origin at the centre of the cup bottom."""
import numpy as np

# 16 oz (473 mL brim-full) clear PET cold cup: 93 mm rim, 58 mm base, 110 mm tall, 0.6 mm wall.
HEIGHT = .110
WALL = .0006
FLOOR = .0015
OUTER_PROFILE = np.array([[0., .0290], [HEIGHT, .0465]])
INNER_PROFILE = np.array([[FLOOR, .0290-WALL], [HEIGHT, .0465-WALL]])
GRASP_HEIGHT = .045
CUP_MASS_KG = .012

TABLE = .7475
# Slush machine: cabinet + clear bowl with auger on top, front tap with pull-down handle, drip tray.
DRIP_TRAY_TOP = .860
TAP_XY = np.array([.385, 0.])
SPOUT_EXIT_Z = .995
NOZZLE = np.array([TAP_XY[0], TAP_XY[1], SPOUT_EXIT_Z-.003])   # MPM inlet convention: exit = nozzle z + 3 mm
LEVER_PIVOT = np.array([.37, 0., 1.05])
LEVER_LENGTH = .125
FILL_POSITION = np.array([TAP_XY[0], TAP_XY[1], DRIP_TRAY_TOP])  # cup stands on the drip tray under the tap
# Serve / pickup stand (flat platform, soft-serve stand without its ring), moved to stay reachable for a side grasp.
STAND_CENTER = np.array([.30, .20, TABLE + .140])
CUP_START = STAND_CENTER + [0., 0., .0005]


def inner_radius(z):
    return np.interp(z, INNER_PROFILE[:, 0], INNER_PROFILE[:, 1])


def outer_radius(z):
    return np.interp(z, OUTER_PROFILE[:, 0], OUTER_PROFILE[:, 1])


def cavity_volume_ml(top=HEIGHT):
    z = np.linspace(FLOOR, top, 400); r = inner_radius(z)
    return float(np.trapezoid(np.pi * r * r, z) * 1e6)


def level_for_volume(ml):
    lo, hi = FLOOR, HEIGHT
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if cavity_volume_ml(mid) < ml else (lo, mid)
    return (lo + hi) / 2
