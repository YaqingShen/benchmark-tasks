"""Measure the physical geometry in the same coordinates as the inspection photos."""
import numpy as np
from shapely.geometry import Polygon, box
from shapely.ops import unary_union

from inspection import score, STATIONS, CFG
from pockets import AirInterface
from laminate import N, DX, W, LEADING_Z, PULL_X, PULL_Y

TARGET = box(0, -W / 2, N * DX, W / 2)
GLASS = box(-.005, -.036, .145, .036)


def rectangle(position, axes, x1, x2, y1, y2, z=0.):
    local = np.array([(x1, y1, z), (x2, y1, z), (x2, y2, z), (x1, y2, z)])
    return position + local @ axes.T


def liner_footprints(l, lr):
    # Include the folded starter flap, pull extension, and the far support ears.
    strips = [rectangle(l[0], lr[0], PULL_X-.014, PULL_X+.010, -W/2, W/2, -.00135),
              rectangle(l[0], lr[0], PULL_X-.008, PULL_X+.008, .0295, .0965, -.0007),
              rectangle(l[0], lr[0], PULL_X-.0185, PULL_X+.0185, PULL_Y-.020, PULL_Y+.020, .028),
              rectangle(l[1], lr[1], -DX, 0, -W/2, W/2)]
    for i in range(2, N):
        half_width = .039 if i == N-1 else W/2
        strips.append(rectangle(l[i], lr[i], 0, DX, -half_width, half_width))
    return strips


def visible_fold_x(l, lr, first_backed):
    """X at the crease nearest the covered liner: its tangent turns back toward +X."""
    if first_backed == N:
        return None
    # Band 1 is built with its long axis along local -X. The other bands
    # run along +X from the returned flap to the far end of the laminate.
    directions = lr[:, 0, 0].copy()
    directions[1] *= -1
    for i in range(min(first_backed+1, N-1), max(1, first_backed-2), -1):
        if directions[i-1] < 0 < directions[i]:
            return float(l[i, 0] * 1000)
    return None


class Measurements:
    def __init__(self):
        self.air = AirInterface()
        self.stations = [None] * len(STATIONS)
        self.first_press = False
        self.clear = 0
        self.passed = [False] * len(STATIONS)

    def read(self, f, fr, l, lr, edge, cf, glass, stage, path, origin):
        film = [rectangle(f[i], fr[i], 0, DX, -W/2, W/2, LEADING_Z if i == 0 else 0)
                for i in range(N)]
        centers = np.array([ring.mean(axis=0) for ring in film])
        backing = liner_footprints(l, lr)
        still_backed = [i for i in range(2, N)
                        if np.linalg.norm(backing[i+2].mean(axis=0)-centers[i]) <= .006]
        front = still_backed[0] if still_backed else N
        fold_x = visible_fold_x(l, lr, front)
        removed = front == N and all(not Polygon(ring[:, :2]).intersects(GLASS)
                                      for ring in backing)
        good = cf[:N] > .025
        if (good[:front] & (glass[:front] > .004)).any():
            self.first_press = True
        x = edge[:, 0].mean()*1000
        for j, station in enumerate(STATIONS):
            if self.passed[j]:
                continue
            if (x > station + CFG['checkpoint_window_mm'] and abs(edge[:, 1].mean()) < W/2
                    and (good & (glass[:N] > .004)).any()):
                self.passed[j] = True
            if self.passed[j] or self.stations[j] is not None or front == N or abs(x-station) > CFG['checkpoint_window_mm']:
                continue
            # An overhead scale measures the fold relative to the actual blade.
            # Check physical pressure on the strip under the scale, on bare glass.
            for i in range(front):
                if not good[i] or glass[i] <= .004:
                    continue
                if film[i][:, 0].min()*1000-1 <= x <= film[i][:, 0].max()*1000+1:
                    self.passed[j] = True
                    if fold_x is not None:
                        self.stations[j] = float(fold_x-x)
                    break

        if self.air.press(edge, good & (glass[:N] > .004), front):
            self.air.display(stage, path, origin)
        corners = np.array([film[0][0], film[-1][1], film[-1][2], film[0][3]])
        corners = (corners[:, :2]+[0, W/2])*1000
        flat = [Polygon(ring[:, :2]) for i, ring in enumerate(film[:front])
                if ring[:, 2].max() <= .0017 and fr[i, 2, 2] > .98]
        coverage = unary_union(flat).intersection(TARGET).area/TARGET.area if flat else 0.
        self.clear = self.clear+1 if edge[:, 2].min() >= CFG['card_clearance_mm']/1000 else 0
        report = score(corners, coverage, removed,
                       [g is not None and 0 <= g <= CFG['max_fold_gap_mm'] for g in self.stations],
                       self.air.outlines, self.clear >= round(20 * CFG['clear_duration_s']), self.first_press)
        report.update(gaps_mm=self.stations.copy(), front_mm=fold_x)
        return report
