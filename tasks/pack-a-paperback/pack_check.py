"""Geometric packing checks and partial credit from measured scene state."""
import numpy as np
import pack_params as pp

def cuboid_corners(pose, size, center_offset=(0., 0., 0.)):
    s = np.asarray(size)/2; c = np.asarray(center_offset)
    local = np.array([[x, y, z] for x in (-s[0], s[0]) for y in (-s[1], s[1]) for z in (-s[2], s[2])])+c
    return local@pose[:3, :3].T+pose[:3, 3]

def panel_vertices(pose, manifest, name, T=.0012):
    c = manifest['panels'][name]['center']
    local = np.array([[x-c[0], y-c[1], z] for poly in manifest['panels'][name]['polygons'] for x, y in poly for z in (-T/2, T/2)])
    return local@pose[:3, :3].T+pose[:3, 3]

def to_box(inverse, pts):
    return pts@inverse[:3, :3].T+inverse[:3, 3]

def contents(base, pad_base, pad_top, book):
    """Pad/book state in the box frame. Poses are 4x4 world body poses (body origin = cuboid centre)."""
    inv = np.linalg.inv(base); P, B = pp.PAD, pp.BOOK
    L, W, H = P['size']; hb = P['base_thickness']
    # pad body origins are at each plate's bottom centre; pack_grok002: pad_top collider starts PAD_TOP_GAP above its origin
    pb = to_box(inv, cuboid_corners(pad_base, (L, W, hb), (0, 0, hb/2)))
    pt = to_box(inv, cuboid_corners(pad_top, (L, W, H-hb-pp.PAD_TOP_GAP), (0, 0, pp.PAD_TOP_GAP+(H-hb-pp.PAD_TOP_GAP)/2)))
    bk = to_box(inv, cuboid_corners(book, B['size']))
    tilt = lambda pose: float(abs((inv[:3, :3]@pose[:3, :3])[2, 2]))
    inside = lambda p, tol=.0005: bool(np.all(np.abs(p[:, 0]) <= pp.INTERIOR_U+tol) and np.all(np.abs(p[:, 1]) <= pp.INTERIOR_V+tol) and p[:, 2].min() > pp.BASE_TOP_Z-.001)
    pad_bottom = float(pb[:, 2].min()); pad_top_z = float(pt[:, 2].max()); book_bottom = float(bk[:, 2].min()); book_top = float(bk[:, 2].max())
    pad_in_box = bool(inside(pb) and inside(pt) and abs(pad_bottom-pp.BASE_TOP_Z) < .002 and tilt(pad_base) > .995 and tilt(pad_top) > .995)
    bc = (to_box(inv, book[None, :3, 3]))[0]; pc = pt.mean(0)
    over_pad = bool(abs(bc[0]-pc[0]) < L/2 and abs(bc[1]-pc[1]) < W/2)
    book_on_pad = bool(pad_in_box and inside(bk) and abs(book_bottom-pad_top_z) < .003 and tilt(book) > .995 and over_pad)
    compression = float(np.linalg.norm(pad_top[:3, 3]-pad_base[:3, 3])-hb)
    rb = inv[:3, :3]@book[:3, :3]; yaw = float(np.degrees(np.arctan2(rb[1, 0], rb[0, 0])))
    yaw_err = abs((yaw+90.) % 180.-90.)
    book_square = bool(yaw_err <= pp.BOOK_SQUARE_TOL_DEG)
    compressed_now = bool(book_on_pad and book_top <= pp.LID_UNDERSIDE_Z+pp.CLOSE_TOL_M)
    return {'pad_in_box': pad_in_box, 'book_on_pad': book_on_pad, 'book_square': book_square, 'book_yaw_err_deg': yaw_err,
            'compressed_to_height_now': compressed_now, 'book_over_pad_footprint': over_pad,
            'pad_bottom_z_m': pad_bottom, 'pad_top_z_m': pad_top_z, 'book_bottom_z_m': book_bottom, 'book_top_z_m': book_top,
            'pad_compression_m': -compression, 'book_tilt_cos': tilt(book), 'pad_tilt_cos': tilt(pad_top),
            'book_center_box_m': bc.tolist(), 'pad_center_box_m': pc.tolist(),
            'points_world': np.vstack([cuboid_corners(pad_base, (L, W, hb), (0, 0, hb/2)), cuboid_corners(pad_top, (L, W, H-hb-pp.PAD_TOP_GAP), (0, 0, pp.PAD_TOP_GAP+(H-hb-pp.PAD_TOP_GAP)/2)), cuboid_corners(book, B['size'])])}

def closure(base, poses, manifest):
    """Lid / tuck state in the box frame from the mailer body poses."""
    inv = np.linalg.inv(base)
    tuck = to_box(inv, panel_vertices(poses['front_tuck'], manifest, 'front_tuck'))
    tuck_n = (inv[:3, :3]@poses['front_tuck'][:3, :3])[:, 2]
    tuck_inside = bool(tuck[:, 1].min() >= -pp.INTERIOR_V-.0003 and tuck[:, 1].max() <= -.055 and tuck[:, 2].max() <= .0515
                       and tuck[:, 2].min() <= .032 and abs(tuck_n[1]) > .94)
    lid = to_box(inv, panel_vertices(poses['lid'], manifest, 'lid'))
    lid_n = (inv[:3, :3]@poses['lid'][:3, :3])[:, 2]
    lid_flush = bool(np.all(np.abs(lid[:, 2]-.050) < .003) and lid_n[2] < -.985)
    flaps = {}
    for side in ['left', 'right']:
        v = to_box(inv, panel_vertices(poses['lid_'+side+'_dust'], manifest, 'lid_'+side+'_dust'))
        flaps[side] = bool(np.all(np.abs(v[:, 0]) < pp.INTERIOR_U+.0006) and np.all(np.abs(v[:, 1]) < pp.INTERIOR_V) and v[:, 2].min() < .035)
    ears = {}
    for side in ['left', 'right']:
        v = to_box(inv, panel_vertices(poses['tuck_'+side+'_ear'], manifest, 'tuck_'+side+'_ear'))
        ears[side] = bool(np.all(np.abs(v[:, 0]) < pp.INTERIOR_U+.0006) and v[:, 1].min() > -pp.INTERIOR_V-.0003 and v[:, 2].max() < .0515)
    return {'tuck_inside_front_wall': tuck_inside, 'lid_flush': lid_flush, 'dust_flaps_inside': flaps, 'tuck_ears_inside': ears,
            'tuck_v_range_m': [float(tuck[:, 1].min()), float(tuck[:, 1].max())], 'tuck_z_range_m': [float(tuck[:, 2].min()), float(tuck[:, 2].max())],
            'tuck_normal_v': float(tuck_n[1]), 'lid_z_range_m': [float(lid[:, 2].min()), float(lid[:, 2].max())], 'lid_normal_z': float(lid_n[2])}

HAND_CLEARANCE_M = .020
FINAL_HOLD_S = 3.
REQUIRED = (
    'pad_in_box', 'book_on_pad', 'book_square', 'compressed_to_height',
    'tuck_inside_front_wall', 'folded_flaps_seated', 'lid_flush', 'box_intact',
    'hands_clear', 'stable',
)

def score(flags, hold_s):
    """Points for loaded contents, valid closure, and released stability."""
    if not flags["box_intact"]:
        return 0.0
    pad = flags['pad_in_box']
    loaded = pad and flags['book_on_pad']
    squared = loaded and flags['book_square']
    compressed = loaded and flags['compressed_to_height']
    packable = loaded and flags['box_intact']
    tucked = packable and flags['tuck_inside_front_wall']
    flaps = packable and flags['folded_flaps_seated']
    flush = packable and flags['lid_flush']
    released = all(flags[name] for name in REQUIRED)
    points = 10*pad+10*loaded+5*squared+10*compressed+25*tucked+10*flaps+15*flush+10*released
    dwell = min(1., max(0., hold_s)/FINAL_HOLD_S) if released else 0.
    return float((points+5*dwell)/100)
