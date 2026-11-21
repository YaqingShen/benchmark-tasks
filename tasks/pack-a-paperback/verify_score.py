"""Offline checks for packing geometry, component credit and recorded trial scores."""
import argparse
import csv
import gzip
import itertools
import math
from pathlib import Path

import numpy as np

import pack_check as checker
import pack_params as pp


def require(value, message):
    if not value:
        raise ValueError(message)


def pose(x, y, z, yaw=0.0):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0, x], [s, c, 0, y], [0, 0, 1, z], [0, 0, 0, 1.]])


def check_geometry():
    base = np.eye(4)
    top = pp.LID_UNDERSIDE_Z - pp.BOOK['size'][2]
    pad = pose(0, 0, pp.BASE_TOP_Z)
    pt = pose(0, 0, top - pp.PAD['size'][2] + pp.PAD['base_thickness'])
    book = pose(0, 0, top + pp.BOOK['size'][2]/2)
    packed = checker.contents(base, pad, pt, book)
    require(all(packed[k] for k in ('pad_in_box', 'book_on_pad', 'book_square',
                                     'compressed_to_height_now')), 'reference packed geometry')
    for displacement, key in [((0.12, 0, 0), 'book_on_pad'), ((0, 0, .005), 'book_on_pad'),
                              ((0, 0, 0, np.deg2rad(4)), 'book_square')]:
        require(not checker.contents(base, pad, pt, pose(*displacement) @ book)[key], key)
    free = pose(0, 0, pp.OVERSTUFF_M)
    require(not checker.contents(base, pad, free @ pt, free @ book)['compressed_to_height_now'],
            'uncompressed stack must stand above the lid')
    # In the formed carton the hinge distinguishes an outward and an inward horizontal lid.
    import json
    manifest = json.loads((Path(__file__).parent / "assets/manifest.json").read_text())
    poses = {name: np.eye(4) for name in manifest["panels"]}
    poses["front_tuck"] = pose(0, -.0775, .034)
    poses["front_tuck"][:3, :3] = [[1., 0., 0.], [0., 0., 1.], [0., -1., 0.]]
    require(checker.closure(base, poses, manifest)["tuck_inside_front_wall"], "tuck inside")
    poses["front_tuck"][1, 3] -= .004
    require(not checker.closure(base, poses, manifest)["tuck_inside_front_wall"], "tuck outside")
    for side, sign in (("left", -1), ("right", 1)):
        part = "lid_" + side + "_dust"
        poses[part] = pose(sign*.102, 0, .034)
        poses[part][:3, :3] = [[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]]
        require(checker.closure(base, poses, manifest)["dust_flaps_inside"][side], part)
        poses[part][0, 3] += sign*.006
        require(not checker.closure(base, poses, manifest)["dust_flaps_inside"][side], part + " outside")
        ear = "tuck_" + side + "_ear"
        poses[ear] = pose(sign*.09, -.055, .034)
        require(checker.closure(base, poses, manifest)["tuck_ears_inside"][side], ear)
        poses[ear][1, 3] -= .02
        require(not checker.closure(base, poses, manifest)["tuck_ears_inside"][side], ear + " outside")
    poses["lid"] = pose(0, .16, .050)
    require(not checker.closure(base, poses, manifest)["lid_flush"], "lid lying outside")
    poses["lid"] = pose(0, 0, .050)
    poses["lid"][:3, :3] = np.diag([1., -1., -1.])
    require(checker.closure(base, poses, manifest)["lid_flush"], "closed face at rim height")
    poses["lid"][2, 3] += .004
    require(not checker.closure(base, poses, manifest)["lid_flush"], "closed face above rim")
    for shift in [pose(.12, -.08, 0, .45), pose(-.16, .22, .03, -.28)]:
        moved = checker.contents(shift, shift @ pad, shift @ pt, shift @ book)
        require(all(moved[k] == packed[k] for k in ('pad_in_box', 'book_on_pad', 'book_square',
                                                      'compressed_to_height_now')), 'box-frame invariance')


def check_scores():
    keys = checker.REQUIRED
    for values in itertools.product((False, True), repeat=len(keys)):
        flags = dict(zip(keys, values))
        for duration in (0., 1.5, 2.95, 3., 9.):
            score = checker.score(flags, duration)
            require(0 <= score <= 1, 'score bounds')
            require((score == 1) == (all(values) and duration >= 3), 'full-credit condition')
            if not flags['box_intact'] or not flags['pad_in_box']:
                require(score == 0, 'unusable or unloaded carton')
    full = dict.fromkeys(keys, True)
    for change, expected in [({'hands_clear':False}, .85), ({'stable':False}, .85),
                             ({'tuck_inside_front_wall':False}, .60),
                             ({'folded_flaps_seated':False}, .75), ({'lid_flush':False}, .70),
                             ({'book_on_pad':False}, .10)]:
        require(math.isclose(checker.score(full | change, 3.), expected), str(change))
    require(math.isclose(checker.score(full, 1.5), .975), 'partial dwell')


def check_trace(path):
    counts, dwell, last, changed, release_changes = {}, {}, {}, {}, {}
    with gzip.open(path, 'rt') as fh:
        for row in csv.DictReader(fh):
            run = row['run']
            counts[run] = counts.get(run, 0) + 1
            require(counts[run] == int(row['step']), 'missing or unordered steps')
            flags = {k: bool(int(row[k])) for k in checker.REQUIRED}
            recorded = float(row["original_score"])
            old_flush = flags["lid_flush"]
            flags["lid_flush"] = (max(abs(float(row[k])-50) for k in ("lid_min_mm","lid_max_mm")) < 3
                                  and float(row["lid_normal_z"]) < -.985)
            published_lid = flags["lid_flush"]
            previous_clear = flags["hands_clear"]
            flags["hands_clear"] = min(float(row["left_clearance_mm"]), float(row["right_clearance_mm"])) >= checker.HAND_CLEARANCE_M*1000
            historical = dict(flags, lid_flush=old_flush, hands_clear=previous_clear)
            require(math.isclose(checker.score(historical, float(row['hold_s'])), recorded), 'historical score mismatch')
            dwell[run] = dwell.get(run, 0) + 1 if all(flags.values()) else 0
            secs = dwell[run]/20
            last[run] = checker.score(flags, secs)
            if not math.isclose(last[run], recorded):
                if old_flush != published_lid:
                    changed[run] = changed.get(run, 0) + 1
                else:
                    require(previous_clear != flags["hands_clear"] or secs != float(row["hold_s"]),
                            "unexplained score change")
                    release_changes[run] = release_changes.get(run, 0) + 1
    require(counts == {'scripted':7791, 'dream':9000}, 'incomplete evidence')
    require(last == {'scripted':1., 'dream':0.}, 'final score mismatch')
    print('Final scores reproduced:', counts, last, 'outward-lid adjustments:', changed, 'release adjustments:', release_changes, 'terminal hold seconds:', {k:v/20 for k,v in dwell.items()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trace', nargs='?', type=Path)
    args = parser.parse_args()
    check_geometry()
    check_scores()
    if args.trace:
        check_trace(args.trace)
    print('Packing geometry and 5,120 score combinations passed.')
