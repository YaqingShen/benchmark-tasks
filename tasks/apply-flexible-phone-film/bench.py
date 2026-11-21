"""Score a manually annotated real trial. All evidence paths are relative to its JSON."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from inspection import CFG, PHONE, TARGET, STATIONS, apply_h, from_photo, transform
from shapely.geometry import Polygon
from shapely.ops import unary_union


def file_evidence(folder, name):
    if not isinstance(name, str) or not name.strip():
        raise ValueError('A trial evidence filename is missing.')
    path = folder / name
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f'Missing or empty evidence file: {name}')
    return path


def measure_force(path):
    with path.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    cols = CFG['force_check']['sample_columns']
    data = np.array([[float(row[k]) for k in cols] for row in rows])
    if len(data) < 3 or not np.isfinite(data).all():
        raise ValueError(f'{path.name}: insufficient valid force samples.')
    t, x, force = data.T
    lo, hi = CFG['force_check']['middle_stroke_mm']
    if np.any(np.diff(t) <= 0) or np.any(np.diff(x) < 0) or x[0] > lo or x[-1] < hi:
        raise ValueError(f'{path.name}: check sample order and stroke length.')
    if (force < 0).any() or (force > CFG['force_check']['gauge_range_N'][1]).any():
        raise ValueError(f'{path.name}: force outside configured gauge range.')
    mask = (x >= lo) & (x <= hi)
    if mask.sum() < 3:
        raise ValueError(f'{path.name}: too few middle-stroke samples.')
    return dict(file=path.name, median_force_N=float(np.median(force[mask])),
                stroke_speed_mm_s=float((hi-lo)/np.diff(np.interp([lo, hi], x, t))[0]))


def inspect_photo(d):
    """Recover only the final photograph measurements, independent of motion logs."""
    h = transform(d['phone_corners_px'], PHONE)
    corners_px = d.get('film_corners_px')
    error = None
    if corners_px is not None:
        corners = apply_h(h, corners_px)
        error = float(np.linalg.norm(corners-TARGET, axis=1).max())
    flat = unary_union([Polygon(apply_h(h, ring)) for ring in d['flat_outlines_px']])
    bubbles = [apply_h(h, ring) for ring in d['bubble_outlines_px']]
    return dict(max_corner_error_mm=error,
                flat_coverage_fraction=flat.intersection(Polygon(TARGET)).area/Polygon(TARGET).area,
                max_bubble_mm=max((float(np.linalg.norm(r[:, None]-r, axis=2).max()) for r in bubbles), default=0.),
                bubble_area_mm2=sum(Polygon(r).area for r in bubbles))


def usable_frames(folder, d):
    from PIL import Image
    import imageio_ffmpeg
    for field in ('final_photo_left_light','final_photo_right_light'):
        with Image.open(folder / d[field]) as photo:
            if photo.size != (d['photo_width_px'],d['photo_height_px']):
                raise ValueError('Annotation dimensions differ from the final photograph.')
            photo.load()
    counts = []
    for field in ('overhead_video','side_video'):
        video = str(folder / d[field])
        reader = imageio_ffmpeg.read_frames(video)
        try:
            header = next(reader)
            next(reader)
        finally:
            reader.close()
        if abs(header['fps']-d['video_hz']) > .05:
            raise ValueError(f'{field} frame rate differs from the declared rate.')
        counts.append(imageio_ffmpeg.count_frames_and_secs(video)[0])
    return min(counts)


def measure_trial(d, folder):
    if d.get('task') != CFG['task'] or not d.get('trial_id'):
        raise ValueError('Set the task and the unique real trial ID.')
    for field in ('overhead_video', 'side_video', 'final_photo_left_light', 'final_photo_right_light'):
        file_evidence(folder, d.get(field))
    if len({(folder / d[key]).resolve() for key in ('overhead_video','side_video',
            'final_photo_left_light','final_photo_right_light')}) != 4:
        raise ValueError('Use the two distinct camera recordings and two distinct illumination photographs.')
    if not np.isfinite(d['video_hz']) or d['video_hz'] < CFG['camera']['minimum_video_hz']:
        raise ValueError('Insufficient video rate.')
    nframes = usable_frames(folder, d)
    if type(d['clear_frames']) is not int or not 0 <= d['clear_frames'] <= nframes:
        raise ValueError('Clear-frame count exceeds the synchronized recordings.')
    width, height = d['photo_width_px'], d['photo_height_px']
    if min(width, height) < 1:
        raise ValueError('Record the inspection photo dimensions in pixels.')
    outlines = d['flat_outlines_px'] if 'flat_outlines_px' in d else [d['film_outline_px']]
    groups = [d['phone_corners_px'], d['ruler_check_px'], *outlines, *d['bubble_outlines_px']]
    corners_px = d['film_corners_px']
    if corners_px is not None:
        groups.append(corners_px)
    for coordinates in groups:
        points = np.asarray(coordinates, float)
        if (points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all()
                or (points < 0).any() or (points >= [width, height]).any()):
            raise ValueError('An annotation is outside the inspection photo.')
    phone = np.asarray(d['phone_corners_px'], float)
    if phone.shape != (4,2) or (corners_px is not None and np.shape(corners_px) != (4,2)):
        raise ValueError('Trace four glass corners. Enter four film corners or null if they are unobservable.')
    scale = np.linalg.norm(phone-np.roll(phone, 1, axis=0), axis=1) / np.linalg.norm(PHONE-np.roll(PHONE, 1, axis=0), axis=1)
    if np.min(scale) < CFG['camera']['minimum_pixels_per_mm']:
        raise ValueError('The glass occupies too few pixels for this inspection.')
    h = transform(phone, PHONE)
    marks = apply_h(h, d['ruler_check_px'])
    if len(marks) != 2 or abs(np.linalg.norm(marks[1]-marks[0])-CFG['camera']['horizontal_ruler_check_mm']) > CFG['camera']['ruler_error_limit_mm']:
        raise ValueError('The independent ruler marks do not confirm the calibration.')
    if len(d['first_pass']) != len(STATIONS) or type(d['liner_clear']) is not bool:
        raise ValueError('Record all three checkpoints and inspect the removed liner.')
    gaps = []
    last_frame = -1
    for station, obs in zip(STATIONS, d['first_pass']):
        gap = None
        if obs is not None:
            frame = obs['frame']
            if type(frame) is not int or frame <= last_frame or frame >= nframes:
                raise ValueError('Checkpoint frames must be ordered.')
            last_frame = frame
            if (obs['card_on_exposed_film'] is True and obs['fold_visible'] is True
                    and abs(obs['blade_mm']-station) <= CFG['checkpoint_window_mm']):
                gap = obs['fold_mm'] - obs['blade_mm']
        gaps.append(gap)
    clear = (d['clear_frames']/d['video_hz'] >= CFG['clear_duration_s']
             and d['minimum_card_clearance_mm'] >= CFG['card_clearance_mm'])
    first_press = d['first_press_frame']
    if first_press is not None and (type(first_press) is not int or not 0 <= first_press < nframes):
        raise ValueError('Record the first visible card press as a nonnegative frame index or null.')
    if first_press is not None and any(obs is not None and obs['frame'] < first_press
                                       and obs['card_on_exposed_film'] is True for obs in d['first_pass']):
        raise ValueError('A loaded checkpoint cannot precede first press.')
    for ring in [*outlines, *d['bubble_outlines_px']]:
        if len(ring) < 3 or not Polygon(ring).is_valid or Polygon(ring).area <= 0:
            raise ValueError('Trace each region as a nondegenerate, noncrossing polygon.')
    if first_press is None and any(g is not None for g in gaps):
        raise ValueError('A loaded checkpoint requires a first visible press.')
    # Missing film corners cannot earn alignment. The remaining observations
    # still reach the shared scorer, including a fully missed installation.
    measured_corners = np.full((4, 2), np.nan) if corners_px is None else corners_px
    result = from_photo(dict(d, film_corners_px=measured_corners, video_gaps_mm=gaps,
                             card_clear=clear, first_press=first_press is not None))
    if corners_px is None:
        result['corner_error_mm'] = None
    force_files = d.get('force_sample_csv', [])
    if force_files and (len(set(force_files)) != CFG['force_check']['replicates']
                        or len(force_files) != len(set(force_files))):
        raise ValueError('Supply one independent force CSV for each of three fresh samples.')
    if force_files:
        setup=d.get('force_setup',{})
        if not all(isinstance(setup.get(k),str) and setup[k].strip() for k in ('laminate_and_lot','gauge_model')):
            raise ValueError('Record the tested laminate, lot and gauge.')
        strip_width = float(setup['strip_width_mm'])
        angle = float(setup['angle_deg'])
        if not np.isfinite(strip_width) or strip_width <= 0 or not 0 < angle <= 180:
            raise ValueError('Record actual strip width and pull angle.')
    return dict(trial_id=d['trial_id'], photographic_score=result,
                force_setup=d.get('force_setup') if force_files else None,
                force_trials=[measure_force(file_evidence(folder, f)) for f in force_files],
                ruler_measured_mm=float(np.linalg.norm(marks[1]-marks[0])),
                config_sha256=hashlib.sha256(Path(__file__).with_name('bench-config.json').read_bytes()).hexdigest())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trial', type=Path)
    parser.add_argument('--photo-only', action='store_true', help='Measure the final annotated image only.')
    args = parser.parse_args()
    try:
        record = json.loads(args.trial.read_text())
        if args.photo_only:
            photo = file_evidence(args.trial.parent, record['source'])
            if hashlib.sha256(photo.read_bytes()).hexdigest() != record['source_sha256']:
                raise ValueError('The photograph differs from the annotated original.')
            result = inspect_photo(record)
        else:
            result = measure_trial(record, args.trial.parent)
        print(json.dumps(result, indent=2))
    except (ValueError, KeyError, TypeError, OSError, RuntimeError, StopIteration, np.linalg.LinAlgError) as e:
        parser.exit(2, f'Incomplete or invalid real trial: {e}\n')
