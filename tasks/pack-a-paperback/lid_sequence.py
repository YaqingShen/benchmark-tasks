"""Contact-only lid prototype with finite-mesh-preflighted handoffs.

No object pose writes, grasp constraints, latches, or material changes.
The continuous demonstration uses live joint and contact measurements.
"""
import json
import numpy as np
from gripper_geometry import hand_collision_bounds

WORK_ANGLE = np.deg2rad(88.)


def tuck_ready(d):
    return 88. < np.rad2deg(d.w.angles()['front_tuck']) < 97.


def rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1., 0., 0.], [0., c, -s], [0., s, c]])


def ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0., s], [0., 1., 0.], [-s, 0., c]])


def frame(x, z):
    return np.column_stack([x, np.cross(z, x), z])


def absolute_lid(d):
    r = d.w.poses()['base'][:3, :3].T @ d.w.poses()['lid'][:3, :3]
    return np.arctan2(r[2, 1], r[1, 1])


def reference(d):
    return d.w.point('back', [0., .130, 0.]).copy(), d.w.poses()['base'][:3, :3].copy()


def lid_frame(ref, angle):
    h, base = ref
    return h, base @ rx(angle)


def support_rotation(side):
    sign = 1 if side == 'left' else -1
    beta = 1.4 if side == 'left' else 1.5
    return frame(np.array([-sign, 0., 0.]), np.array([0., -sign*np.sin(beta), -np.cos(beta)]))


def push_pose(d, ref, angle, side, u=None, v=.240):
    h, r = lid_frame(ref, angle)
    if u is None:
        u = -.070 if side == 'left' else .090
    normal = -r[:, 2]
    point = h+r@np.array([u, v-.130, 0.])
    rotation = support_rotation(side)
    tool = d.face_tool_pose(point, normal, rotation, .0003)
    if not getattr(d, '_raw_push', False):
        if not hasattr(d, '_last_push'): d._last_push = {}
        d._last_push[side] = (u, v)
        # by grok r11: anchor to the MEASURED contact pose. establish_support stores the actual TCP error vs this
        # nominal pose in lid-local coordinates; it is carried along with the lid for all later targets.
        err = getattr(d, 'lid_err_local', {}).get(side)
        if err is not None: tool = tool+r@err
    return tool, rotation, normal


def home_with_gap(d, side, gap=.030):
    """Keep the checked clearance opening during transport; open at home."""
    o = 0 if side == 'left' else 7
    start = d.action[o:o+6].copy()
    finish = d.home[o:o+6]
    minimum = min(d.ik.tcp_pose(start*(1-f)+finish*f, d.w.station.BASES[side])[0][2]
                  for f in np.linspace(0, 1, 100))
    if minimum < d.w.support_top+.045:
        raise RuntimeError('Home transport has insufficient cardboard clearance')
    for k in range(50):
        f = d.ik.smoothstep((k+1)/50)
        d.action[o:o+6] = start*(1-f)+finish*f
        d.action[o+6] = gap/.0939
        d.step('return '+side+' hand with clearance jaw opening', side)
    p, r = d.pose(side)
    d.move(side, p, r, .040, .5, 'open cleared '+side+' hand at home')


def pinch(d, side, p, r, gap, name, floor=0.):
    for _ in range(25):
        d.feedback_step(side, p, r, gap, 'align '+name)
    floor = max(floor, .0004)
    held_gap = gap
    for k in range(40):
        held_gap = floor+(gap-floor)*(1-d.ik.smoothstep((k+1)/40))
        d.feedback_step(side, p, r, held_gap, 'pinch '+name)
        force_guard(d, side, 'pinch '+name)
        if d.panel_force(side, name) >= 4.:
            break
    if not hasattr(d, 'fold_gaps'):
        d.fold_gaps = {}
        d.fold_gap_max = {}
    d.fold_gaps[side, name] = held_gap
    d.fold_gap_max[side, name] = max(.004, held_gap+.002)
    for _ in range(12):
        d.feedback_step(side, p, r, holding_gap(d, side, name), 'settle pinch '+name)
        force_guard(d, side, 'settle pinch '+name)
    force = d.panel_force(side, name)
    print('LID_GRIP', side, name, force, flush=True)
    if force < .1:
        raise RuntimeError('No measured physical grip on '+name)


def holding_gap(d, side, name):
    """Regulate actual contact through finger commands; force guards stay unchanged."""
    gap = d.fold_gaps[side, name]
    if d.hand_force(side) > 8.:
        gap += .00005
    elif d.panel_force(side, name) < 2.:
        gap -= .000025
    gap = float(np.clip(gap, .0004, d.fold_gap_max[side, name]))
    d.fold_gaps[side, name] = gap
    return gap


def panel_clearance(d, intended):
    poses = d.w.poses()
    panels = []
    for name, panel in d.w.manifest['panels'].items():
        pose = poses[name]
        for polygon in panel['polygons']:
            xy = np.asarray(polygon)-np.asarray(panel['center'][:2])
            lower = np.r_[xy.min(axis=0), -.0006]
            upper = np.r_[xy.max(axis=0), .0006]
            panels.append((pose, (lower+upper)/2, (upper-lower)/2, name))
    panel_rotations = np.array([p[0][:3, :3] for p in panels])
    panel_centres = np.array([p[0][:3, 3]+p[0][:3, :3]@p[1] for p in panels])
    panel_extents = np.array([p[2] for p in panels])
    adjacent = np.array([p[3] not in intended for p in panels])

    def clear_poses(samples, low, high, allow_target=False):
        for tcp, rotation in samples:
            centres = ((low+high)/2) @ rotation.T + tcp
            extents = (high-low)/2
            if np.min(centres[:, 2]-extents @ np.abs(rotation[2])) < d.w.support_top+.002:
                return False
            orientation = panel_rotations.transpose(0, 2, 1)@rotation
            delta = centres[None, :, :]-panel_centres[:, None, :]
            rel_panel = np.abs(np.einsum('pmi,pij->pmj', delta, panel_rotations))
            projected_hand = np.einsum('pij,mj->pmi', np.abs(orientation), extents)
            overlap_panel = np.all(rel_panel < projected_hand+panel_extents[:, None, :]+.001, axis=-1)
            if not np.any(overlap_panel):
                continue
            projected_panel = np.einsum('pij,pi->pj', np.abs(orientation), panel_extents)
            overlap_hand = np.all(np.abs(delta @ rotation) < projected_panel[:, None, :]+extents+.001, axis=-1)
            overlap = overlap_panel & overlap_hand
            if allow_target:
                overlap &= adjacent[:, None]
            if np.any(overlap):
                return False
        return True

    return clear_poses


def approach_pinch(d, side, panel_name, candidates, gaps, minimum_gap, adjacent_contact=()):
    """Check open transport and jaw closure against measured cardboard geometry."""
    offset = 0 if side == 'left' else 7
    base = d.w.station.BASES[side]
    start = d.w.joints()[offset:offset+6].copy()
    clear_poses = panel_clearance(d, (panel_name, *adjacent_contact))

    def clear(joints, low, high):
        return clear_poses((d.ik.tcp_pose(q, base) for q in joints), low, high)

    def path(start_q, target, rotation):
        q, ep, er = d.ik.solve(target, d.ik.matrix_to_quat_xyzw(rotation), start_q, base,
                               iters=350, damping=.018)
        if ep > .001 or er > .01:
            return None
        return q

    start_point, start_rotation = d.ik.tcp_pose(start, base)
    plan = None
    for target, rotation in candidates:
        grasp_q = path(start, target, rotation)
        if grasp_q is None:
            continue
        for gap in gaps:
            low, high = hand_collision_bounds(gap)
            if not clear([start, grasp_q], low, high):
                continue
            if not all(clear_poses([(target, rotation)], *hand_collision_bounds(float(g)), allow_target=True)
                       for g in np.linspace(gap, minimum_gap, 12)):
                continue
            for distance, z in ((.035, 0.), (.055, 0.), (.085, .04), (.065, .05), (.09, 0.), (.06, 0.), (.045, 0.), (.04, .08)):
                outside = target-distance*rotation[:, 2]
                entry = outside+np.array([0., 0., z])
                entry_q = path(grasp_q, entry, rotation)
                if entry_q is None or not clear(np.linspace(start, entry_q, 45), low, high):
                    continue
                waypoints = [entry_q]
                seed = entry_q
                for end in ([outside, target] if z else [target]):
                    start_segment = d.ik.tcp_pose(seed, base)[0]
                    for sample in np.linspace(start_segment, end, 13)[1:]:
                        seed = path(seed, sample, rotation)
                        if seed is None:
                            break
                        waypoints.append(seed)
                    if seed is None:
                        break
                if seed is not None and clear(waypoints[1:], low, high):
                    plan = (target, rotation, gap, waypoints)
                    break
            if plan is not None:
                break
        if plan is not None:
            break
    if plan is None:
        raise RuntimeError(f'No sampled approach and jaw closure cleared the panels beside {panel_name}')
    target, rotation, gap, waypoints = plan
    print('PANEL_APPROACH', side, panel_name, target.tolist(), 'gap_m', gap, 'points', len(waypoints), flush=True)
    d.move(side, start_point, start_rotation, gap, .7, 'open jaw clear of box')
    previous = start
    for i, q in enumerate(waypoints):
        for t in np.linspace(0., 1., 51 if i == 0 else 4)[1:]:
            d.action[offset:offset+6] = previous+(q-previous)*d.ik.smoothstep(t)
            d.action[offset+6] = gap/.0939
            d.step(f'approach {panel_name} with open jaw', side)
            if d.hand_force(side) > 1.:
                raise RuntimeError(f'Unexpected panel contact during open-jaw approach to {panel_name}')
        previous = q
    return target, rotation, gap


def approach_tuck(d, side, point, flap_frame, sign, preferred):
    candidates = []
    if side == 'right':
        # Approach the free edge from the right, with the wrist tilted clear of the lid.
        for shift, height, depth in ((.105, .008, .010), (.085, .005, .006),
                                     (.085, .003, .010), (.085, .002, .002)):
            phi = np.deg2rad(-125.)
            tilt = -.46
            normal = -flap_frame[:, 2]
            along = flap_frame[:, 0]*np.sin(phi)+flap_frame[:, 1]*np.cos(phi)
            x = normal*np.cos(tilt)+along*np.sin(tilt)
            z = along*np.cos(tilt)-normal*np.sin(tilt)
            rotation = frame(x, z)
            target = point+flap_frame[:, 0]*shift+flap_frame[:, 1]*height-z*depth
            candidates.append((target, rotation))
    else:
        phi = np.deg2rad(125.)
        normal = flap_frame[:, 2]
        along = flap_frame[:, 0]*np.sin(phi)+flap_frame[:, 1]*np.cos(phi)
        if all(abs(d.w.angles()[f'lid_{s}_dust']) > np.deg2rad(80.) for s in ('left', 'right')):
            edge = np.deg2rad(130.)
            direction = flap_frame[:, 0]*np.sin(edge)+flap_frame[:, 1]*np.cos(edge)
            x = normal*np.cos(.1)+direction*np.sin(.1)
            z = direction*np.cos(.1)-normal*np.sin(.1)
            candidates.append((point+.010*flap_frame[:, 0]-z*.004, frame(x, z)))
            rotation = frame(normal, along)
            for offset, height, depth in ((.010, 0., .004), (.010, .002, .005), (0., .008, .006)):
                target = point+flap_frame[:, 0]*offset+flap_frame[:, 1]*height-along*depth
                candidates.append((target, rotation))
        flat = abs(d.w.angles()['front_tuck']) < np.deg2rad(30.)
        tilt = .1 if flat else 0.
        x = normal*np.cos(tilt)+along*np.sin(tilt)
        z = along*np.cos(tilt)-normal*np.sin(tilt)
        rotation = frame(x, z)
        offsets = ((-.025, .006, .010), (-.025, .008, .006), (-.010, .003, .005), (-.010, .006, .010)) if flat else (
            (0., .003, .005), (0., .006, .010), (0., .008, .006), (-.025, .008, .006))
        for shift, height, depth in offsets:
            target = point+flap_frame[:, 0]*shift+flap_frame[:, 1]*height-z*depth
            candidates.append((target, rotation))
    for shift in (0., .015, .030, -.015):
        for angle in (preferred, 1.46, 1.74, 1.32, 1.88):
            rotation = flap_frame@frame(np.array([0., 0., sign]),
                                        np.array([sign*np.sin(angle), -np.cos(angle), 0.]))
            candidates.append((point+flap_frame[:, 0]*shift, rotation))
    gaps = (.060, .080, .045) if side == 'right' else (.045, .060, .080)
    return approach_pinch(d, side, 'front_tuck', candidates, gaps, .0009)


def force_guard(d, side, label):
    if d.hand_force(side) > 15.:
        raise RuntimeError('Excessive mean contact force during '+label)


SUPPORT_SEARCH_M = .050   # by grok r7/r8: 6 and 20 mm gave no finger contact; still force-guarded and contact-gated


def lid_contact(d, side):
    # by grok r9: the coplanar (still flat) dust flap on that side is part of the same lid plate at this point
    return d.panel_force(side, 'lid')+d.panel_force(side, 'lid_left_dust' if side == 'left' else 'lid_right_dust')


def establish_support(d, side, p, r, normal, prepare):
    print('LID_SUPPORT_START', side, 'lid_abs_deg', float(np.rad2deg(absolute_lid(d))), flush=True)
    d.move_joints_safe(side, prepare, r, 2.5, 'prepare '+side+' main-lid support')
    d.move(side, p, r, 0., 1.5, 'contact main lid with '+side+' hand')
    p = np.array(p, dtype=float).copy()
    # Near vertical, gravity and the passive crease need only a small support
    # force. Do not drive the free lid forward while chasing a large preload.
    stable_contact = 0
    preload = 0.
    # by grok q5: the 1 mm search (25 um/step) never reached the raised lid (q5: zero force on the
    # left fingers); approach along the measured lid normal up to 6 mm, still force-guarded.
    for _ in range(int(SUPPORT_SEARCH_M/.00005)+40):
        d.feedback_step(side, p, r, 0., 'establish measured '+side+' lid support')
        force_guard(d, side, 'lid-support contact')
        force = lid_contact(d, side)
        stable_contact = stable_contact+1 if force >= .010 else 0
        if stable_contact >= 3:
            break
        if force < .010:
            p -= normal*.00005
            preload += .00005
        if _ % 20 == 0:
            # by grok (pack_grok002 r7): diagnostics, commanded vs actual TCP along the lid normal
            act, _r = d.pose(side); ref_ = reference(d); h_, rr_ = lid_frame(ref_, absolute_lid(d))
            print('LID_SUPPORT_DIAG', side, 'preload_mm', round(preload*1e3, 2), 'cmd_minus_act_mm', np.round((p-act)*1e3, 2).tolist(),
                  'act_above_lid_plane_mm', round(float((act-h_)@rr_[:, 2])*-1e3, 2), 'cmd_above_lid_plane_mm', round(float((p-h_)@rr_[:, 2])*-1e3, 2), 'lid_abs_deg', round(float(np.rad2deg(absolute_lid(d))), 2),
                  'all_finger_forces', round(sum(float(np.sum(d.w.hand_panel_forces(i))) for i in ([0, 1] if side == 'left' else [2, 3])), 3), flush=True)
        if preload > SUPPORT_SEARCH_M:
            raise RuntimeError(f'Lid support was not contacted within {SUPPORT_SEARCH_M*1e3:.0f} mm of approach')
    if stable_contact >= 3:
        u_, v_ = getattr(d, '_last_push', {}).get(side, (None, .240))
        ref_ = reference(d); ang_ = absolute_lid(d)
        d._raw_push = True
        try: nominal, _rr, _n = push_pose(d, ref_, ang_, side, u=u_, v=v_)
        finally: d._raw_push = False
        _h, r_lid = lid_frame(ref_, ang_)
        act, _ra = d.pose(side)
        if not hasattr(d, 'lid_err_local'): d.lid_err_local = {}
        d.lid_err_local[side] = r_lid.T@(act-nominal)
        print('LID_ERR_STORED', side, 'world_mm', np.round((act-nominal)*1e3, 1).tolist(), 'u,v', u_, v_, 'lid_deg', float(np.rad2deg(ang_)), flush=True)
    print('MAIN_LID_SUPPORT', side, d.panel_force(side, 'lid'), 'with_dust', lid_contact(d, side), 'preload_m', preload, flush=True)
    if stable_contact < 3:
        raise RuntimeError('Missing sustained measured main-lid support')
    d.support[side] = (*d.pose(side), 0.)


def clamp_lid(d, side):
    axes = d.w.poses()['lid'][:3, :3]
    sign = 1 if side == 'left' else -1
    locations = ((-.082, .235, 0.), (-.094, .185, 0.), (-.103, .185, .12)) if side == 'left' else (
        (.103, .185, -.25), (.103, .210, -.25), (.103, .235, -.25))
    dust = f'lid_{side}_dust'
    if abs(d.w.angles()[dust]) > np.deg2rad(70.):
        inside = ((-.070, .240, 0.), (-.068, .238, 0.), (-.072, .245, .12), (-.070, .250, 0.)) if side == 'left' else (
            (.082, .235, -.25), (.090, .235, -.25), (.082, .210, -.25))
        locations = (*inside, *locations)
    candidates = []
    for u, v, tilt in locations:
        normal = sign*axes[:, 2]
        along = sign*axes[:, 0]
        x = normal*np.cos(tilt)+along*np.sin(tilt)
        z = along*np.cos(tilt)-normal*np.sin(tilt)
        candidates.append((d.w.point('lid', [u, v, 0.])-z*.006, frame(x, z)))
    p, r, gap = approach_pinch(d, side, 'lid', candidates, (.045, .060, .080), .001, (dust,))
    pinch(d, side, p, r, gap, 'lid', floor=.001)
    settle_measured_grip(d, side, 'lid')
    remember_lid_clamp(d, side)


def brace_tuck(d, side):
    axes = d.w.poses()['front_tuck'][:3, :3]
    sign = 1 if side == 'left' else -1
    locations = ((-.035, .312, .004, 125., 1, 0.), (-.020, .312, .004, 125., 1, 0.),
                 (-.035, .316, .006, 125., 1, .12)) if side == 'left' else (
        (.065, .312, .004, 145., 1, 0.), (.020, .312, .004, 125., 1, 0.),
        (.035, .312, .004, 125., -1, .12), (.035, .316, .006, 125., -1, -.12))
    candidates = []
    for u, v, depth, azimuth, normal_sign, tilt in locations:
        phi = sign*np.deg2rad(azimuth)
        normal = normal_sign*axes[:, 2]
        along = axes[:, 0]*np.sin(phi)+axes[:, 1]*np.cos(phi)
        x = normal*np.cos(tilt)+along*np.sin(tilt)
        z = along*np.cos(tilt)-normal*np.sin(tilt)
        point = d.w.point('front_tuck', [u, v, 0.])-z*depth
        candidates.append((point, frame(x, z)))
    p, r, gap = approach_pinch(d, side, 'front_tuck', candidates, (.045, .060, .080), .001)
    pinch(d, side, p, r, gap, 'front_tuck', floor=.001)
    settle_measured_grip(d, side, 'front_tuck')
    d.support[side] = (*d.pose(side), d.fold_gaps[side, 'front_tuck'])


def release_panel_support(d, side):
    panel = getattr(d, 'lid_hold_panels', {}).pop(side, 'front_tuck')
    intended = ('lid', f'lid_{side}_dust') if panel == 'lid' else (panel,)
    d.support.pop(side)
    d.lid_clamps.pop(side, None)
    p, r = d.pose(side)
    gap = .030 if panel.endswith('_dust') else .045
    d.move(side, p, r, gap, .6, 'release outgoing panel support')
    withdraw_open_hand(d, side, intended, gap)


def withdraw_open_hand(d, side, intended, gap):
    offset = 0 if side == 'left' else 7
    base = d.w.station.BASES[side]
    start = d.action[offset:offset+6].copy()
    p, r = d.ik.tcp_pose(start, base)
    clear = panel_clearance(d, intended)
    bounds = hand_collision_bounds(gap)
    home = d.home[offset:offset+6]
    quat = d.ik.matrix_to_quat_xyzw(r)
    outward = 1. if side == 'left' else -1.
    dust = len(intended) == 1 and intended[0].endswith('_dust')
    clearance_moves = ((.035, 0., 0.), (.030, 0., 0.), (.025, 0., 0.), (0., 0., .040),
                       (0., 0., .055), (0., 0., .065), (0., .020, .040), (.025, .015, 0.),
                       (.030, .025, 0.), (.035, .035, 0.), (.020, .040, 0.),
                       (.020, .045, .025), (.025, .055, .040), (.015, .055, .050),
                       (.020, .035, .070), (.040, .065, .035))
    paths = [p-distance*r[:, 2]+[0., outward*lateral, rise] for distance, rise, lateral in clearance_moves]
    tuck = intended == ('front_tuck',)
    if dust:
        paths = [p+[-.025, outward*.020, 0.], p+[-.035, outward*.030, .015], *paths]
    elif tuck:
        paths.insert(0, p+[-.025, outward*.065, .080])
    for end in paths:
        positions = np.linspace(p, end, 15)
        if not clear(((x, r) for x in positions[1:]), *bounds, allow_target=not (dust or tuck)):
            continue
        q = start.copy()
        for point in positions[1:]:
            q, ep, er = d.ik.solve(point, quat, q, base, iters=200, damping=.018)
            if ep > .001 or er > .01:
                break
        else:
            path = (d.ik.tcp_pose(j, base) for j in np.linspace(q, home, 48))
            if clear(path, *bounds):
                d.move(side, end, r, gap, 1.5, 'withdraw outgoing panel support')
                force_guard(d, side, 'withdraw outgoing panel support')
                home_with_gap(d, side, gap)
                return
    raise RuntimeError('No sampled withdrawal cleared the carton and reached home')


def remember_lid_clamp(d, side, panel='lid'):
    p, r = d.pose(side)
    h, axes = lid_frame(reference(d), absolute_lid(d))
    if not hasattr(d, 'lid_clamps'):
        d.lid_clamps = {}
    if not hasattr(d, 'lid_hold_panels'):
        d.lid_hold_panels = {}
    d.lid_clamps[side] = (axes.T@(p-h), axes.T@r)
    d.lid_hold_panels[side] = panel
    d.support[side] = (p, r, d.fold_gaps[side, panel])


def lid_clamp_target(d, side, angle):
    h, axes = lid_frame(reference(d), angle)
    point, rotation = d.lid_clamps[side]
    return h+axes@point, axes@rotation


def move_lid_to_work_angle(d, side, degrees, label):
    goal = np.deg2rad(degrees)
    absolute = absolute_lid(d)
    held_panel = d.lid_hold_panels[side]
    d.support.pop(side)
    if held_panel.endswith('_dust'):
        # The back wall can flex while the lid is carried by one flap.
        measure = lambda: absolute_lid(d)
        fold_to(d, side, 'lid', absolute, goal, lambda angle: lid_clamp_target(d, side, angle),
                label, grip_panel=held_panel, tracking_lead_deg=2., measure=measure)
    else:
        start = d.w.angles()['lid']
        fold_to(d, side, 'lid', start, start+goal-absolute,
                lambda angle: lid_clamp_target(d, side, absolute+angle-start), label, grip_panel=held_panel)
    remember_lid_clamp(d, side, held_panel)
    d.hold(.5, 'settle lid at fold angle')
    if abs(absolute_lid(d)-goal) > np.deg2rad(4.):
        raise RuntimeError(f'Lid did not retain its {degrees:.0f}-degree working angle')


def lift_flat_lid(d, clamp_support=False):
    ref = reference(d)
    start = absolute_lid(d)
    local = frame(np.array([0., 0., 1.]), np.array([-np.sin(1.4), np.cos(1.4), 0.]))
    def target(angle):
        h, r = lid_frame(ref, angle)
        return h+r@np.array([.107, .090, 0.]), r@local
    p, r = target(start)
    # by grok: open the jaw high and further out, then descend (offline: search_flat_lid_entry_grok.py,
    # h=.060 out=.060 collision-free with lid_right_dust at 0 and -15 deg; revision035 path hit it).
    outside = p-.060*r[:, 2]
    d.move_joints_safe('right', outside+[0., 0., .060], r, 2.5, 'prepare flat lid side-edge grasp')
    d.move('right', outside+[0., 0., .060], r, .026, .75, 'open jaw high above flat lid edge')
    d.move('right', outside, r, .026, 1.5, 'descend outside flat lid edge with open jaw')
    d.move('right', p, r, .026, 2., 'straddle flat lid side edge')
    for _ in range(25):
        d.feedback_step('right', p, r, .026, 'align flat lid side-edge grip')
    for k in range(40):
        d.feedback_step('right', p, r, .026*(1-d.ik.smoothstep((k+1)/40)), 'pinch flat lid side edge')
    if d.panel_force('right', 'lid')+d.panel_force('right', 'lid_right_dust') < .1:
        raise RuntimeError('Missing physical flat-lid side grip')
    for k in range(220):
        f = d.ik.smoothstep((k+1)/220)
        p, r = target(start*(1-f)+np.deg2rad(58)*f)
        d.feedback_step('right', p, r, 0., 'lift main lid while dust flaps remain flat')
        force_guard(d, 'right', 'main-lid lift')
    d.support['right'] = (*d.pose('right'), 0.)
    d.hold(.5, 'hold raised lid for pusher transfer')
    d.w.snapshot('lid-flat-flaps-raised')
    if clamp_support:
        clamp_lid(d, 'left')
        d.support.pop('right')
        p, r = d.pose('right')
        d.move('right', p, r, .030, 1., 'release flat-lid side grip')
        d.move('right', p+[0., 0., .050], r, .030, 1.5, 'lift clear of flat-lid side grip')
        d.return_home('right')
        d.support.pop('left')
        start = d.w.angles()['lid']
        absolute = absolute_lid(d)
        def target(angle):
            return lid_clamp_target(d, 'left', absolute+angle-start)
        fold_to(d, 'left', 'lid', start, start+WORK_ANGLE-absolute, target, 'raise clamped lid')
        remember_lid_clamp(d, 'left')
        d.hold(.5, 'settle clamped lid at working pose')
        if abs(np.rad2deg(absolute_lid(d))-88) > 4.:
            raise RuntimeError('Clamped lid did not reach the flap working pose')
        d.w.snapshot('lid-clamped-at-working-pose')
        return
    angle = absolute_lid(d)
    ref = reference(d)
    p, r, normal = push_pose(d, ref, angle, 'left', v=.210)
    establish_support(d, 'left', p, r, normal, p+[0., .070, -.025])
    d.support.pop('right')
    p, r = d.pose('right')
    d.move('right', p, r, .030, 1., 'release flat-lid side grip')
    d.move('right', p+[0., 0., .050], r, .030, 1.5, 'lift clear of flat-lid side grip')
    d.return_home('right')
    d.support.pop('left')
    start = absolute_lid(d)
    ref = reference(d)
    for k in range(160):
        f = d.ik.smoothstep((k+1)/160)
        angle = start*(1-f)+WORK_ANGLE*f
        v = .210+.030*np.clip((np.rad2deg(angle)-58)/22, 0., 1.)
        p, r, _ = push_pose(d, ref, angle, 'left', v=v)
        d.feedback_step('left', p, r, 0., 'raise supported lid to flap working pose')
        force_guard(d, 'left', 'supported lid raise')
    d.support['left'] = (*d.pose('left'), 0.)
    d.hold(.5, 'settle supported lid at working pose')
    d.w.snapshot('lid-supported-at-working-pose')
    if abs(np.rad2deg(absolute_lid(d))-88) > 4.:
        raise RuntimeError('Main lid did not reach the flap working pose')


def fold_tuck(d, partial, already_gripped=False):
    side = 'right' if partial else 'left'
    sign = -1 if partial else 1
    u = -.050 if partial else -.025
    peak = 56.5 if partial else 107.5
    hinge = d.w.point('front_tuck', [u, d.w.manifest.get('tuck_hinge_v', .288), 0.]).copy()
    base_fr = d.w.poses()['front_tuck'][:3, :3].copy()
    point = hinge+base_fr[:, 1]*.025
    if already_gripped:
        if d.panel_force(side, 'front_tuck') < .8:
            raise RuntimeError('Restored front tuck has no measured left-hand grip')
        gap = d.action[6]*.0939
        d.fold_gaps[side, 'front_tuck'] = gap
        d.fold_gap_max[side, 'front_tuck'] = max(.004, gap+.002)
    else:
        p, r, gap = approach_tuck(d, side, point, base_fr, sign, 1.6 if partial else 1.4)
        pinch(d, side, p, r, gap, 'front_tuck', floor=.0009)
        settle_measured_grip(d, side, 'front_tuck')
    if not partial:
        from pack_checkpoint import save_checkpoint
        save_checkpoint(d, 'tuck_grip', 'right')
        if abs(np.rad2deg(d.w.angles()['lid_right_dust'])) > 89.5:
            d.support[side] = (*d.pose(side), holding_gap(d, side, 'front_tuck'))
            relax_held_dust(d, 'right', angle_deg=88.3)
            d.support.pop(side)
    p, r = d.pose(side)
    start = d.w.angles()['front_tuck']
    hinge = d.w.point('front_tuck', [u, d.w.manifest['tuck_hinge_v'], 0.]).copy()
    base_fr = d.w.poses()['front_tuck'][:3, :3].copy()
    lid = d.w.poses()['lid'].copy()
    hinge_local = lid[:3, :3].T@(hinge-lid[:3, 3])
    frame_local = lid[:3, :3].T@base_fr
    grip_local = base_fr.T@(p-hinge)
    wrist_local = base_fr.T@r

    def target(angle):
        # Both hands brace the hinge while the tuck bends.
        current_lid = d.w.poses()['lid'] if partial else lid
        parent = current_lid[:3, :3]
        fr = parent@frame_local@rx(angle-start)
        pivot = current_lid[:3, 3]+parent@hinge_local
        return pivot+fr@grip_local, fr@wrist_local, fr

    label = 'partially fold front tuck' if partial else 'finish front-tuck fold'
    fold_to(d, side, 'front_tuck', start, np.deg2rad(peak), target, label)
    p, r = d.pose(side)
    _, _, fr = target(d.w.angles()['front_tuck'])
    d.move(side, p, r, .030, 1., 'release front-tuck grip')
    clear = p-.060*r[:, 2]+[0., 0., .020] if partial else p+.040*fr[:, 1]
    d.move(side, clear, r, .030, 2., 'withdraw from front-tuck free edge')
    if partial:
        home_with_gap(d, side, .030)
        check_released_tuck(d, partial=True)
    else:
        save_checkpoint(d, 'tuck_clear', 'right')
        finish_tuck_withdrawal(d)


def finish_tuck_withdrawal(d):
    withdraw_open_hand(d, 'left', ('front_tuck',), .030)
    check_released_tuck(d, partial=False)
    if d.panel_force('right', 'lid_right_dust') < .5:
        raise RuntimeError('Right hand lost the lid support during tuck withdrawal')


def check_released_tuck(d, partial):
    d.hold(1., 'check released front-tuck fold')
    value = np.rad2deg(d.w.angles()['front_tuck'])
    print('TUCK_RELEASED', value, 'partial', partial, flush=True)
    lower, upper = (37., 53.) if partial else (95.5, 120.)
    if not (lower < value < upper or (partial and tuck_ready(d))):
        raise RuntimeError('Front tuck did not retain the required intermediate fold')
    d.w.snapshot('front-tuck-partial-released' if partial else 'front-tuck-final-released')


def fold_to(d, side, name, start, goal, target, label, grip_panel=None, tracking_lead_deg=0., measure=None):
    if measure is None:
        measure = lambda: d.w.angles()[name]
    direction = 1. if goal >= start else -1.
    for step in range(300):
        measured = measure()
        if direction*(goal-measured) <= np.deg2rad(.5):
            return
        fraction = d.ik.smoothstep(min(1., (step+1)/200))
        nominal_progress = (abs(goal-start)+np.deg2rad(tracking_lead_deg))*fraction
        measured_progress = direction*(measured-start)
        progress = min(nominal_progress, max(0., measured_progress+np.deg2rad(4.)))
        point, rotation = target(start+direction*progress)[:2]
        correction = .6*(point-d.pose(side)[0])
        correction *= min(1., .005/max(np.linalg.norm(correction), 1e-9))
        d.command_pose(side, point+correction, rotation, holding_gap(d, side, grip_panel or name), label)
        force_guard(d, side, label)
        for holder in d.support:
            if holder != side:
                force_guard(d, holder, label)
    measured = np.rad2deg(measure())
    raise RuntimeError(f'{name} fold reached {measured:.1f} deg, required {np.rad2deg(goal):.1f} deg')


def settle_measured_grip(d, side, name):
    """Remove arm tracking preload accumulated while the fingers settled."""
    offset = 0 if side == 'left' else 7
    initial = d.action[offset:offset+6].copy()
    measured = d.w.joints()[offset:offset+6].copy()
    for step in range(8):
        blend = d.ik.smoothstep((step+1)/8)
        d.action[offset:offset+6] = initial+(measured-initial)*blend
        d.action[offset+6] = holding_gap(d, side, name)/.0939
        d.step('settle measured grip '+name, side)
        force_guard(d, side, 'settle measured grip '+name)
    if d.panel_force(side, name) < .1:
        raise RuntimeError('Measured grip lost contact with '+name)


def fold_flap(d, kind, side, keep_grip=False):
    sign = 1 if side == 'left' else -1
    ear = kind == 'ear'
    name = 'tuck_'+side+'_ear' if ear else 'lid_'+side+'_dust'
    parent_name = 'front_tuck' if ear else 'lid'
    hu, v, lever, delta = (.1018, d.w.manifest.get('tuck_hinge_v', .288)+.0184, .022, -.8) if ear else (.1034, .220, .031, 0.)  # by grok: ear grip follows the 158 mm lid; r12: wrist delta -.4 -> -.8 (offline: 96-102 deg fold end unreachable with -.4 in the -90 deg yawed box)
    parent_pose = d.w.poses()[parent_name].copy()
    parent = parent_pose[:3, :3]
    child_frame = d.w.poses()[name][:3, :3].copy()
    hinge = d.w.point(name, [-sign*hu, v, 0.]).copy()
    start = d.w.angles()[name]
    local = frame(np.array([0., 0., sign]), np.array([sign*np.cos(delta), np.sin(delta), 0.]))
    hinge_local = parent.T@(hinge-parent_pose[:3, 3])
    child_local = parent.T@child_frame
    def target(angle):
        current = d.w.poses()[parent_name]
        axes = current[:3, :3]
        rr = axes@child_local@ry(angle-start)
        pivot = current[:3, 3]+axes@hinge_local
        return pivot+rr@grip_local, rr@wrist_local
    grip_local = np.array([-sign*lever, 0., 0.])
    wrist_local = local
    p, r = target(start)
    if not ear:
        if side == 'left':
            options = ((.031, 0., 0., .15), (.028, 0., 0., .15), (.027, .015, 0., .15),
                       (.031, 0., 0., .28), (.031, 0., -15., 0.))
        elif absolute_lid(d) > np.deg2rad(105.):
            options = ((.031, 0., -20., -.28), (.028, 0., -20., -.28), (.031, 0., -20., -.46))
        else:
            options = ((.031, 0., 0., -.28), (.031, 0., 0., -.46))
    elif abs(d.w.angles()['front_tuck']) < np.deg2rad(30.):
        options = ((.022, .008, 0., 0.), (.022, .013, 0., 0.), (.022, 0., 0., 0.)) if side == 'left' else (
            (.022, .008, -30., -.46), (.022, .008, 0., -.46), (.022, .013, 0., -.46))
        if absolute_lid(d) > np.deg2rad(100.):
            options = ((.022, .008, -30., .25), (.022, .013, -30., .25), (.022, .013, -60., .25)) if side == 'left' else (
                (.022, 0., -30., 0.), (.024, .008, -30., -.25), (.022, .013, -45., 0.), (.024, .008, -45., 0.))
    elif tuck_ready(d):
        options = ((.022, .008, -90., 0.), (.018, .013, -90., 0.), (.024, .008, -90., 0.)) if side == 'left' else (
            (.022, .008, -90., .28), (.022, .013, -90., 0.), (.022, .013, -90., -.28))
    else:
        options = ((.022, .013, -30., -.28), (.018, .013, -45., -.28), (.024, .008, -30., -.28)) if side == 'left' else (
            (.022, .008, -30., -.28), (.022, .008, -30., -.46), (.022, .008, -45., 0.))
    candidates = []
    for reach, offset, angle, tilt in options:
        angle = np.deg2rad(angle)
        normal = child_frame[:, 2]*sign
        along = child_frame@np.array([sign*np.cos(angle), np.sin(angle), 0.])
        x = normal*np.cos(tilt)+along*np.sin(tilt)
        z = along*np.cos(tilt)-normal*np.sin(tilt)
        contact = hinge+child_frame@np.array([-sign*reach, offset, 0.])
        candidates.append((contact-z*.003, frame(x, z)))
    candidates.append((p, r))
    gaps = (.030, .045, .060, .080) if ear else (.080, .060, .045)
    p, r, gap = approach_pinch(d, side, name, candidates, gaps, .001)
    pinch(d, side, p, r, gap, name, floor=.001)
    settle_measured_grip(d, side, name)
    p, r = d.pose(side)
    start = d.w.angles()[name]
    child_frame = d.w.poses()[name][:3, :3].copy()
    hinge = d.w.point(name, [-sign*hu, v, 0.]).copy()
    parent_pose = d.w.poses()[parent_name]
    axes = parent_pose[:3, :3]
    hinge_local = axes.T@(hinge-parent_pose[:3, 3])
    child_local = axes.T@child_frame
    grip_local = child_frame.T@(p-hinge)
    wrist_local = child_frame.T@r
    fold_to(d, side, name, start, sign*np.deg2rad(102), target, 'fold '+name)
    if keep_grip:
        hold_folded_dust(d, side)
        return
    p, r = d.pose(side)
    d.move(side, p, r, .030, 1., 'release '+name)
    if ear:
        _, clear_r = target(sign*np.deg2rad(70))
        clear = p-.045*r[:, 2]
        d.move(side, clear, r, .030, 1.5, 'withdraw away from parent of '+name)
        d.move(side, clear+[0., 0., .050], clear_r, .030, 1.5, 'lift cleared '+name+' gripper')
        home_with_gap(d, side, .030)
    else:
        from pack_checkpoint import save_checkpoint
        holder = 'right' if side == 'left' else 'left'
        save_checkpoint(d, f'{side}_dust_release', holder)
        finish_dust_release(d, side)
        return
    d.hold(1., 'check released '+name)
    angle = sign*np.rad2deg(d.w.angles()[name])
    print('LID_FLAP_RELEASED', name, angle, flush=True)
    if not 78. < angle < 108.:
        raise RuntimeError('Lid flap did not retain fold: '+name)
    d.w.snapshot(name+'-released')


def finish_dust_release(d, side):
    name = f'lid_{side}_dust'
    withdraw_open_hand(d, side, (name,), .030)
    d.hold(1., 'check released '+name)
    angle = np.rad2deg(d.w.angles()[name])*(1. if side == 'left' else -1.)
    print('LID_FLAP_RELEASED', name, angle, flush=True)
    if not 78. < angle < 108.:
        raise RuntimeError('Lid flap did not retain fold: '+name)
    d.w.snapshot(name+'-released')


def hold_folded_dust(d, side):
    name = f'lid_{side}_dust'
    sign = 1. if side == 'left' else -1.
    angle = sign*np.rad2deg(d.w.angles()[name])
    if not 97. < angle < 110. or d.panel_force(side, name) < .8:
        raise RuntimeError(f'Cannot keep physical hold of {name} at {angle:.1f} deg')
    gap = d.action[6 if side == 'left' else 13]*.0939
    if not hasattr(d, 'fold_gaps'):
        d.fold_gaps = {}
        d.fold_gap_max = {}
    d.fold_gaps[side, name] = gap
    d.fold_gap_max[side, name] = max(.004, gap+.002)
    remember_lid_clamp(d, side, name)
    print('DUST_HELD', side, angle, d.panel_force(side, name), flush=True)


def relax_held_dust(d, side, angle_deg=90.3):
    name = f'lid_{side}_dust'
    sign = 1. if side == 'left' else -1.
    poses = d.w.poses()
    lid = poses['lid']
    flap = poses[name][:3, :3]
    pivot = d.w.point(name, [-sign*.1034, .220, 0.])
    p, r = d.pose(side)
    local_hinge = lid[:3, :3].T@(pivot-lid[:3, 3])
    local_frame = lid[:3, :3].T@flap
    local_grip = flap.T@(p-pivot)
    local_wrist = flap.T@r
    start = d.w.angles()[name]

    def target(angle):
        parent = d.w.poses()['lid']
        frame = parent[:3, :3]@local_frame@ry(angle-start)
        hinge = parent[:3, 3]+parent[:3, :3]@local_hinge
        return hinge+frame@local_grip, frame@local_wrist

    fold_to(d, side, name, start, sign*np.deg2rad(angle_deg), target,
            'settle folded flap in the grip', tracking_lead_deg=3.5)
    print('DUST_RELAXED', side, float(sign*np.rad2deg(d.w.angles()[name])), flush=True)
    remember_lid_clamp(d, side, name)


def transfer_support(d, incoming):
    outgoing = 'left' if incoming == 'right' else 'right'
    if getattr(d, 'lid_clamps', None):
        clamp_lid(d, incoming)
        d.hold(.25, 'hold both lid clamps before transfer')
        release_panel_support(d, outgoing)
        d.w.snapshot('lid-support-transferred-to-'+incoming)
        return
    ref = reference(d)
    p, r, normal = push_pose(d, ref, absolute_lid(d), incoming)
    prepare = p+.030*normal if incoming == 'right' else p+.020*normal+[0., .050, -.020]
    establish_support(d, incoming, p, r, normal, prepare)
    d.hold(.5, 'hold both lid supports before transfer')
    d.support.pop(outgoing)
    p, r = d.pose(outgoing)
    if outgoing == 'left':
        d.move(outgoing, p+.025*normal+[0., .050, -.020], r, 0., 2., 'clear outgoing left lid support')
        d.return_home(outgoing)
    else:
        d.move(outgoing, p+.030*normal, r, 0., 2., 'clear outgoing right lid support')
        home_with_gap(d, outgoing, 0.)
    d.w.snapshot('lid-support-transferred-to-'+incoming)


def close_prepared_lid(d):
    d.support.pop('left')
    ref = reference(d)
    start = absolute_lid(d)
    for k in range(360):
        f = d.ik.smoothstep((k+1)/360)
        angle = start*(1-f)+np.pi*f
        shift = np.clip((np.rad2deg(angle)-88)/22, 0., 1.)
        p, r, _ = push_pose(d, ref, angle, 'left', u=-.070+.030*shift, v=.240-.020*shift)
        d.feedback_step('left', p, r, 0., 'close main lid and feed real return ears into slots')
        force_guard(d, 'left', 'lid insertion')
    d.hold(.5, 'settle physical lid closure')
    d.w.snapshot('lid-closed-loaded')
    p, r = d.pose('left')
    d.move('left', p+[0., .025, .085], r, 0., 2., 'release closed-lid pusher')
    d.return_home('left')
    d.hold(5., 'verify complete box after both hands release')
    d.w.snapshot('lid-complete-released')
    (d.w.output/'lid-result.json').write_text(json.dumps(d.w.last, indent=2))
    if not d.w.last['success']:
        raise RuntimeError('Full physical closure and released dwell not achieved')


def verify_tray(d):
    d.hold(3., 'verify tray bottom-tab retention before lid work')
    if not all(v['inserted'] for v in d.w.last['locks'].values()):
        raise RuntimeError('Tray bottom tabs did not retain before lid work')


def run(d):
    verify_tray(d)
    lift_flat_lid(d)
    fold_tuck(d, partial=True)
    fold_flap(d, 'ear', 'right')
    transfer_support(d, 'right')
    fold_flap(d, 'ear', 'left')
    fold_tuck(d, partial=False)
    fold_flap(d, 'dust', 'left')
    transfer_support(d, 'left')
    fold_flap(d, 'dust', 'right')
    close_prepared_lid(d)
