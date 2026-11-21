"""Summarize the in-line paddle load from solver contacts; force unit N, travel mm."""
import argparse
import json
from pathlib import Path
import numpy as np
from inspection import CFG


def summarize(rows, hz=20):
    travel = np.array([r['liner'][0][0] for r in rows])
    travel = (travel-travel[0])*1000
    loads = np.array([r['paddle_normal_load_N'][0] for r in rows])
    steps = np.array([r['step'] for r in rows])
    lo, hi = CFG['force_check']['middle_stroke_mm']
    if np.max(travel) < hi:
        raise ValueError('The carriage did not complete the measurement interval.')
    force, distance, duration = [], [], 0.
    for i, dx in enumerate(np.diff(travel)):
        if dx <= 0:
            continue
        x1, x2 = max(lo, travel[i]), min(hi, travel[i+1])
        if x2 <= x1:
            continue
        ds = x2-x1
        mid = ((x2+x1)/2-travel[i])/dx
        force.append(loads[i]+mid*(loads[i+1]-loads[i]))
        distance.append(ds)
        duration += ds/dx*(steps[i+1]-steps[i])/hz
    if not distance:
        raise ValueError('No forward travel samples.')
    work_mJ = float(np.dot(force, distance))
    return dict(source='simulated gripper-paddle normal contact, component along slide',
                stroke_window_mm=[lo, hi], forward_travel_mm=sum(distance),
                forward_work_mJ=work_mJ, work_per_travel_N=work_mJ/sum(distance),
                moving_sample_median_N=float(np.median(force)), moving_time_s=duration,
                mean_moving_speed_mm_s=sum(distance)/duration,
                samples=len(force), sample_rate_hz=hz)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trace', type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize([json.loads(line) for line in args.trace.read_text().splitlines()]), indent=2))
