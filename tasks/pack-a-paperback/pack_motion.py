"""pack_grok002 (by grok): scripted bimanual packing controller (robot contact only).

Privileged object poses are read for targeting (scripted feasibility controller, like revision035);
nothing is written to the simulator except arm joint/gripper targets.
"""
import json
import numpy as np
import pack_params as pp
from robot_fold import Driver

GAP_SCALE = .0939


def front_pinch(yaw=-.4, pitch=.4):
    z = np.array([np.cos(yaw)*np.cos(pitch), np.sin(yaw)*np.cos(pitch), -np.sin(pitch)])
    x = np.cross(z, np.cross(np.array([0., 0., 1.]), z)); x /= np.linalg.norm(x)
    return np.column_stack([x, np.cross(z, x), z])


def end_pinch(side):
    z = np.array([0., -1., 0.]) if side == 'left' else np.array([0., 1., 0.])
    x = np.array([0., 0., 1.]); return np.column_stack([x, np.cross(z, x), z])


def base_z(d):
    return d.w.support_top+.00065


def box_point(d, xyz):
    base = d.w.poses()['base']
    return base[:3, 3]+base[:3, :3]@np.asarray(xyz)


def force_guard(d, side, label):
    import lid_sequence as ls
    ls.force_guard(d, side, label)


def content_force(d, side, name):
    ids = [0, 1] if side == 'left' else [2, 3]; k = d.w.filter_names.index(name)
    return sum(float(d.w.hand_panel_forces(i)[k]) for i in ids)


def pad_force(d, side):
    return content_force(d, side, 'pad_top')+content_force(d, side, 'pad_base')


def move2(d, targets, seconds, label):
    """Both arms together: targets = {side: (point, rotation, gap or None)}."""
    print('ROBOT_PHASE', label, flush=True)
    start = {}
    for side, (p, r, g) in targets.items():
        o = 0 if side == 'left' else 7
        p0, r0 = d.ik.tcp_pose(d.action[o:o+6], d.w.station.BASES[side]); start[side] = (p0, r0, d.action[o+6]*GAP_SCALE)
    n = max(1, round(seconds*20))
    for i in range(n):
        s = d.ik.smoothstep((i+1)/n)
        for side, (p, r, g) in targets.items():
            p0, r0, g0 = start[side]; g1 = g0 if g is None else g
            d.solve_command(side, p0*(1-s)+np.asarray(p)*s, d.ik.slerp_matrix(r0, r, s), g0*(1-s)+g1*s, label)
        d.step(label)
    d.w.snapshot(f'{d.w.steps:05d}-'+label.replace(' ', '-'))


def record(d, name, extra=None):
    m = d.w.last; row = {'stage': name, 'steps': d.w.steps, 'pack_flags': m['pack_flags'], 'score': m['score'],
                         'contents': m['pack_contents'], 'closure': m['pack_closure'], **(extra or {})}
    (d.w.output/f'stage-{name}.json').write_text(json.dumps(row, indent=2, default=lambda o: o.item() if hasattr(o, 'item') else str(o)))
    print('STAGE_RESULT', name, json.dumps({'flags': m['pack_flags'], 'score': m['score'], **(extra or {})}, default=lambda o: o.item() if hasattr(o, 'item') else str(o)), flush=True)
    return row


# ---------------------------------------------------------------- P1: pad (left hand)
PAD_RELEASE_BOTTOM = .062   # pad bottom above base mid-plane at release (lower finger clears the 50 mm front wall)


def place_pad(d):
    side = 'left'; P = pp.PAD['size']; R = front_pinch(); a = R[:, 2]
    pb, _, _ = d.w.content_poses()
    bottom = pb[:3, 3].copy()                      # pad bottom centre, world
    front_x = bottom[0]-P[1]/2
    grasp = np.array([front_x+.010, bottom[1], bottom[2]+P[2]/2])
    pre = grasp-.060*a+[0., 0., .015]
    d.move_joints_safe(side, pre, R, 3., 'prepare pad front-edge pinch')
    d.move(side, pre, R, .046, .7, 'open jaw before pad front edge')
    d.move(side, grasp, R, .046, 1.5, 'straddle pad front edge')
    d.move(side, grasp, R, 0., 1., 'pinch pad front edge')
    d.hold(.3, 'settle pad pinch')
    f = pad_force(d, side); print('PAD_GRIP', f, flush=True)
    if f < .1: raise RuntimeError('No measured physical grip on the foam pad')
    rel = grasp-bottom
    # by grok q2: one-edge pinch lets the pad sag ~23 deg; its far edge hung below the 50 mm wall top and
    # dragged the box off the riser in q2b. Carry high (pad bottom ~150 mm above the base) and descend vertically.
    PAD_CARRY = .150
    d.move(side, grasp+[0., 0., PAD_CARRY], R, 0., 2., 'lift pad off pedestal')
    target_bottom = np.array([.35+pp.PAD_TARGET_V, 0., base_z(d)+PAD_RELEASE_BOTTOM])
    release = target_bottom+rel
    high = release.copy(); high[2] = grasp[2]+PAD_CARRY
    d.move(side, high, R, 0., 3., 'carry pad over open box')
    d.move(side, release, R, 0., 2.5, 'lower pad to release height')
    d.hold(.3, 'steady pad before release')
    pb, _, _ = d.w.content_poses(); print('PAD_AT_RELEASE', pb[:3, 3].tolist(), flush=True)
    d.move(side, release, R, .034, .4, 'ease pad pinch')
    d.move(side, release-.050*a, R, .034, 1., 'withdraw lower finger from under pad')
    d.move(side, release-.050*a+[0., 0., .060], R, .034, 1., 'lift hand clear of box after pad')
    d.hold(1.5, 'let dropped pad settle')
    k = d.w.last['pack_contents']; front_gap = k['pad_center_box_m'][1]-pp.PAD['size'][1]/2+pp.INTERIOR_V
    print('PAD_DROPPED', json.dumps({'pad_in_box': d.w.last['pack_flags']['pad_in_box'], 'center': k['pad_center_box_m'], 'front_gap_m': front_gap, 'tilt': k['pad_tilt_cos']}), flush=True)
    if front_gap < .012 and d.w.last['pack_flags']['pad_in_box']:
        nudge_pad(d, front_gap)
    d.return_home(side)
    d.hold(1., 'pad placed, hands home')
    d.w.snapshot('stage-place_pad-result')
    return record(d, 'place_pad', {'pad_front_gap_m': float(d.w.last['pack_contents']['pad_center_box_m'][1]-pp.PAD['size'][1]/2+pp.INTERIOR_V)})


def nudge_pad(d, front_gap):
    """Fingertips (closed jaw, wide face along y) push the pad's front face back toward the back wall."""
    side = 'left'; k = d.w.last['pack_contents']
    front_world_x = .35+k['pad_center_box_m'][1]-pp.PAD['size'][1]/2
    contact = np.array([front_world_x-.005, 0.03, base_z(d)+.016])
    R = press_rot(d, side, contact); tip = d.face_tool_pose(contact, np.array([0., 0., 1.]), R, -.004, True)
    d.move(side, tip+[0., 0., .060], R, 0., 2., 'above pad front gap for nudge')
    d.move(side, tip, R, 0., 1.5, 'lower fingertips into pad front gap')
    push = .012-front_gap+.004
    d.move(side, tip+[push, 0., 0.], R, 0., 1.5, 'nudge pad back toward back wall')
    d.move(side, tip+[push-.004, 0., .065], R, 0., 1.5, 'lift fingertips after nudge')
    d.hold(.5, 'settle nudged pad')


# ---------------------------------------------------------------- P2: book (both hands, one per end)
BOOK_RELEASE_BOTTOM = .080   # book bottom above base mid-plane: thicker finger roots must clear the side walls
END_INSET = .012
BOOK_AIM_V = .012   # r5: book drifts ~+3.5 mm toward the back wall on the drop and squaring pivots it back further; was .018
BOOK_AIM_V_OLD = .018   # by grok q3: book centre v target (back gap 7.4 mm, front gap 43 mm for the tuck)


def rz(a):
    c, s_ = np.cos(a), np.sin(a); return np.array([[c, -s_, 0.], [s_, c, 0.], [0., 0., 1.]])


def book_axis_yaw(bk):
    """World yaw error of the book long axis from the world y axis (box u), wrapped to [-90, 90] deg."""
    e = bk[:3, 0]; a = np.arctan2(e[1], e[0])-np.pi/2
    return (a+np.pi/2) % np.pi-np.pi/2


def square_on_stand(d, tries=3):
    """by grok (pack_grok002, r2 lesson): the arms cannot reach a pinch whose hand yaw follows a 10 deg book
    (both hands need ~8 deg of misalignment, and the open lower finger then strikes the end face). So the robot
    first squares the book ON ITS STAND with fingertip pushes (a couple: left fingertips push the near face of the
    end that sits closer to the robot toward +x, right fingertips push the far face of the other end toward -x),
    closed-loop on the measured yaw. Falls back to a single-end push if the couple is unreachable. No pose writes."""
    B = pp.BOOK['size']; up = np.array([0., 0., 1.])
    for it in range(tries):
        d.hold(.4, 'measure book yaw on stand')
        _, _, bk = d.w.content_poses(); yaw = book_axis_yaw(bk); c = bk[:3, 3].copy()
        print('STAND_SQUARE', it, float(np.rad2deg(yaw)), flush=True)
        if abs(np.rad2deg(yaw)) < 1.5: return float(np.rad2deg(yaw))
        ey = rz(yaw)@np.array([0., 1., 0.]); ex = rz(yaw)@np.array([1., 0., 0.]); r = B[0]/2-.018
        amt = r*abs(np.sin(yaw))+.006
        # end with smaller x (closer to the robot) gets pushed +x from its near face; the other end -x from its far face
        ends = {'left': c+ey*r, 'right': c-ey*r}
        near_side = 'left' if ends['left'][0] < ends['right'][0] else 'right'; far_side = 'right' if near_side == 'left' else 'left'
        plan = {}
        for side, sgn in [(near_side, 1.), (far_side, -1.)]:
            face = ends[side]-sgn*ex*(B[1]/2+.011)
            contact = np.array([face[0], face[1], c[2]])
            try:
                Rp = press_rot(d, side, contact); tip = d.face_tool_pose(contact, up, Rp, -.004, True)
                plan[side] = (tip, Rp, sgn)
            except RuntimeError as e:
                print('STAND_SQUARE_SIDE_UNREACHABLE', side, e, flush=True)
        if not plan: raise RuntimeError('No reachable fingertip push to square the book on its stand')
        if len(plan) == 1: amt = 2*amt
        for side, (tip, Rp, sgn) in plan.items():
            d.move_joints_safe(side, tip+[0., 0., .050], Rp, 2., 'above book face for stand squaring '+side)
        move2(d, {s_: (tip, Rp, 0.) for s_, (tip, Rp, sgn) in plan.items()}, 2.5, 'lower fingertips beside book faces')
        move2(d, {s_: (tip+[sgn*amt, 0., 0.], Rp, 0.) for s_, (tip, Rp, sgn) in plan.items()}, 3.8, 'push book ends to square on stand')
        move2(d, {s_: (tip+[sgn*amt-sgn*.006, 0., .060], Rp, 0.) for s_, (tip, Rp, sgn) in plan.items()}, 1., 'lift fingertips after stand squaring')
        for side in plan: d.return_home(side)
    d.hold(.4, 'measure book yaw on stand')
    _, _, bk = d.w.content_poses(); y = float(np.rad2deg(book_axis_yaw(bk))); print('STAND_SQUARE_FINAL', y, flush=True)
    return y


def place_book(d):
    """pack_grok002: the book is staged 10 deg skewed. Both hands pinch its ends along the ACTUAL book axis
    (hand yaw follows the book), then rotate together back to square during the lift (in-hand squaring by
    the robot's own grip). No pose writes."""
    B = pp.BOOK['size']
    _, _, bk = d.w.content_poses(); print('BOOK_STAGED_YAW_DEG', float(np.rad2deg(book_axis_yaw(bk))), flush=True)
    square_on_stand(d)
    _, _, bk = d.w.content_poses(); c = bk[:3, 3].copy(); yaw = book_axis_yaw(bk); print('BOOK_YAW_AT_PICK_DEG', float(np.rad2deg(yaw)), flush=True)
    ey = rz(yaw)@np.array([0., 1., 0.])                         # book long axis, pointing to +y (left arm side)
    grasp = {'left': c+ey*(B[0]/2-END_INSET), 'right': c-ey*(B[0]/2-END_INSET)}
    sq0 = {s: c+rz(-yaw)@(grasp[s]-c) for s in grasp}
    # hand yaw at the grasp = yaw + d0; after in-hand squaring = d0. Pick the first d0 with all poses reachable.
    def ok(side, p, Rm):
        q, ep, er = d.ik.solve(p, d.ik.matrix_to_quat_xyzw(Rm), d.w.constants.READY, d.w.station.BASES[side], iters=600, damping=.018)
        return ep < .001 and er < .01
    d0 = None
    # r1 lesson: with d0 = -yaw/2 the palm hit the book end during the straddle and spun it 10 -> 28 deg.
    # Use the SMALLEST reachable misalignment (1 deg steps).
    for cand in [-np.sign(yaw)*np.deg2rad(k) for k in range(0, int(abs(np.rad2deg(yaw)))+9)]:
        Rg = {s: rz(yaw+cand)@end_pinch(s) for s in grasp}; Rf = {s: rz(cand)@end_pinch(s) for s in grasp}
        if all(ok(s, grasp[s], Rg[s]) and ok(s, grasp[s]-.035*Rg[s][:, 2]+[0., 0., .025], Rg[s]) and ok(s, grasp[s]-.032*Rg[s][:, 2], Rg[s]) and ok(s, sq0[s]+[0., 0., .095], Rf[s]) for s in grasp):
            d0 = cand; break
    if d0 is None: raise RuntimeError('No reachable hand yaw for the skewed book pick')
    print('BOOK_HAND_YAW_OFFSET_DEG', float(np.rad2deg(d0)), flush=True)
    R = {s: rz(yaw+d0)@end_pinch(s) for s in grasp}
    for side in ['left', 'right']:
        pre = grasp[side]-.035*R[side][:, 2]+[0., 0., .025]
        d.move_joints_safe(side, pre, R[side], 3., 'prepare book end pinch '+side)
        d.move(side, pre, R[side], .050, .6, 'open jaw before book end '+side)
    # r1 lesson 2: the diagonal pre->grasp path dropped the open lower finger through the book end face
    # (pre is 25 mm up, jaw half-opening ~25 mm) and spun the book. Descend vertically OUTSIDE the end first, then slide in.
    move2(d, {s: (grasp[s]-.032*R[s][:, 2], R[s], .050) for s in grasp}, 1.2, 'descend beside both book ends')
    _, _, bk = d.w.content_poses(); print('BOOK_YAW_BEFORE_STRADDLE_DEG', float(np.rad2deg(book_axis_yaw(bk))), flush=True)
    move2(d, {s: (grasp[s], R[s], .050) for s in grasp}, 1.6, 'straddle both book ends')
    _, _, bk = d.w.content_poses(); print('BOOK_YAW_AFTER_STRADDLE_DEG', float(np.rad2deg(book_axis_yaw(bk))), flush=True)
    move2(d, {s: (grasp[s], R[s], 0.) for s in grasp}, 1., 'pinch both book ends')
    d.hold(.3, 'settle book pinch')
    f = {s: content_force(d, s, 'book') for s in grasp}; print('BOOK_GRIP', f, flush=True)
    if min(f.values()) < .1: raise RuntimeError('Missing physical grip on a book end: '+str(f))
    # lift and square in hand: rotate the grasp offsets by -yaw about the book centre, hands back to plain end pinches
    sq = sq0
    R0 = {s: rz(d0)@end_pinch(s) for s in grasp}
    move2(d, {s: (grasp[s]+[0., 0., .050], R[s], 0.) for s in grasp}, 1.2, 'lift book with both hands')
    move2(d, {s: (sq[s]+[0., 0., .095], R0[s], 0.) for s in grasp}, 1.5, 'square book in hand while lifting')
    _, _, bk = d.w.content_poses(); print('BOOK_YAW_AFTER_INHAND_SQUARE_DEG', float(np.rad2deg(book_axis_yaw(bk))), flush=True)
    # closed-loop residual correction: rotate the pinch points about the MEASURED book centre (hand yaw kept)
    cur = {s: sq[s]+[0., 0., .095] for s in grasp}
    for it in range(2):
        _, _, bk = d.w.content_poses(); res = book_axis_yaw(bk)
        if abs(np.rad2deg(res)) < 1.5: break
        cc = bk[:3, 3]; new = {s: cc+rz(-res)@(cur[s]-cc) for s in grasp}
        try:
            move2(d, {s: (new[s], R0[s], 0.) for s in grasp}, 1., 'correct book yaw in hand')
            cur = new
        except RuntimeError as e:
            print('INHAND_CORRECTION_SKIPPED', e, flush=True); break
        _, _, bk = d.w.content_poses(); print('BOOK_YAW_AFTER_CORRECTION_DEG', it, float(np.rad2deg(book_axis_yaw(bk))), flush=True)
    # re-express the hold so the later release offsets use the actual pinch points vs the actual book centre
    _, _, bk = d.w.content_poses(); c = bk[:3, 3].copy(); sq = {s: cur[s] for s in grasp}
    grasp, R = sq, R0
    tgt = np.array([.35+BOOK_AIM_V, 0., base_z(d)+BOOK_RELEASE_BOTTOM+B[2]/2])
    release = {s: tgt+(grasp[s]-c) for s in grasp}
    move2(d, {s: (release[s]+[0., 0., .030], R[s], 0.) for s in grasp}, 3.5, 'carry book over box')
    for it in range(2):
        d.hold(.3, 'steady book above box')
        _, _, bk = d.w.content_poses(); err = tgt[:2]-bk[:3, 3][:2]
        print('BOOK_ALIGN', it, err.tolist(), flush=True)
        if np.linalg.norm(err) < .0015: break
        for s in release: release[s] = release[s]+[err[0], err[1], 0.]
        move2(d, {s: (release[s]+[0., 0., .030], R[s], 0.) for s in grasp}, 1., 'correct book over box')
    move2(d, {s: (release[s], R[s], 0.) for s in grasp}, 1.5, 'lower book to release height')
    d.hold(.3, 'steady book before release')
    move2(d, {s: (release[s], R[s], .033) for s in grasp}, .4, 'ease both book pinches')
    # r3 lesson: a slow (0.8 s) slide-out let the last supporting finger drag the falling book 14.6 mm sideways
    # into the side panel (book_on_pad False via inside()). Withdraw both lower fingers quickly and together.
    move2(d, {s: (release[s]-.035*R[s][:, 2], R[s], .033) for s in grasp}, .15, 'slide both lower fingers out from under book')
    move2(d, {s: (release[s]-.035*R[s][:, 2]+[0., 0., .050], R[s], .033) for s in grasp}, 1., 'lift both hands clear after book')
    d.hold(1.5, 'let book settle on pad')
    print('BOOK_LANDED', json.dumps({k: d.w.last['pack_contents'][k] for k in ('book_center_box_m', 'book_bottom_z_m', 'pad_top_z_m', 'book_tilt_cos', 'book_yaw_err_deg')}, default=float), flush=True)
    if not d.w.last['pack_flags']['book_on_pad']:
        try:
            seat_book(d)
        except RuntimeError as e:
            print('SEAT_BOOK_SKIPPED', e, flush=True)
    for side in ['left', 'right']:
        d.return_home(side)
    d.hold(1., 'book placed, hands home')
    d.w.snapshot('stage-place_book-result')
    return record(d, 'place_book')


def seat_book(d, tries=2):
    """r4 lesson: after the drop the book can rest propped on one long edge (tilt cos .987). One fingertip press on
    the HIGH side of the book top, pushing it down until that point is 2 mm into the pad, then lift. Right hand
    first, left as fallback (reachability). Force-guarded. No pose writes."""
    B = pp.BOOK['size']; up = np.array([0., 0., 1.])
    for it in range(tries):
        d.hold(.3, 'measure book seating')
        k = d.w.last['pack_contents']; print('SEAT_CHECK', it, k['book_tilt_cos'], k['book_bottom_z_m']-k['pad_top_z_m'], flush=True)
        if d.w.last['pack_flags']['book_on_pad']: return True
        _, _, bk = d.w.content_poses(); n = bk[:3, 2]*np.sign(bk[2, 2]); c = bk[:3, 3]
        h = -n[:2]/max(np.linalg.norm(n[:2]), 1e-9)                  # horizontal direction of the high side
        local = bk[:3, :3].T@np.array([h[0], h[1], 0.]); local[2] = 0.; local /= max(np.linalg.norm(local), 1e-9)
        local = local*np.array([.040, .030, 0.])+np.array([0., 0., np.sign(bk[2, 2])*B[2]/2])
        pt = c+bk[:3, :3]@local
        target_z = base_z(d)+pp.BASE_TOP_Z+pp.PAD['size'][2]+B[2]-.002
        done = False
        for side in ['right', 'left']:
            try: R = press_rot(d, side, pt)
            except RuntimeError as e: print('SEAT_SIDE_UNREACHABLE', side, e, flush=True); continue
            ok_ = False
            for hgt in (.030, .020, .012):                       # r4 IK scan: +30 mm often out of reach, lower works
                try:
                    d.move_joints_safe(side, d.face_tool_pose(pt, up, R, -hgt, True), R, 2., 'above high side of book for seat press '+side); ok_ = True; break
                except RuntimeError as e: print('SEAT_APPROACH_UNREACHABLE', side, hgt, e, flush=True)
            if not ok_: continue
            d.move(side, d.face_tool_pose(pt, up, R, -.002, True), R, 0., 1., 'touch high side of book '+side)
            depth = pt[2]-target_z
            for j in range(40):
                f = d.ik.smoothstep((j+1)/40)
                d.feedback_step(side, d.face_tool_pose(pt, up, R, depth*f, True), R, 0., 'press high side of book down '+side); force_guard(d, side, 'seat press')
            d.hold(.3, 'hold seat press')
            p, r = d.pose(side); d.move(side, p+[0., 0., .05], r, 0., .8, 'lift after seat press '+side)
            d.return_home(side); done = True; break
        if not done: raise RuntimeError('No reachable seat press')
    d.hold(.5, 'settle seated book')
    return bool(d.w.last['pack_flags']['book_on_pad'])


# ---------------------------------------------------------------- L1/L2: lid lift, dust flaps, tuck pre-fold
def verify_box(d):
    d.hold(2., 'verify formed box before lid work')
    b = d.w.last['box']
    # by grok q4: side tab locks are never seated in the inherited start state (diagnostic only); require walls + corner ears
    if not all(b['corner_ears_captured'].values()) or not all(b['walls'].values()):
        raise RuntimeError('Formed box lost a wall or bottom tab before lid work')


def lift_lid(d):
    import lid_sequence as ls
    verify_box(d); ls.lift_flat_lid(d, clamp_support=True)
    d.w.snapshot('stage-lift_lid-result')
    return record(d, 'lift_lid', {'lid_absolute_deg': float(np.rad2deg(ls.absolute_lid(d)))})


def flaps(d):
    import lid_sequence as ls
    ls.move_lid_to_work_angle(d, 'left', 108., 'raise lid for deep tuck grip')
    ls.transfer_support(d, 'right')
    ls.brace_tuck(d, 'left')
    ls.release_panel_support(d, 'right')
    ls.fold_flap(d, 'ear', 'right')
    ls.brace_tuck(d, 'right')
    ls.release_panel_support(d, 'left')
    ls.fold_flap(d, 'ear', 'left')
    ls.clamp_lid(d, 'left')
    ls.release_panel_support(d, 'right')
    ls.move_lid_to_work_angle(d, 'left', 89., 'bring lid upright to clear the carton walls')
    ls.transfer_support(d, 'right')
    return finish_flaps(d, 'left_dust')


TAIL_HOLDERS = {'left_dust': 'right', 'left_dust_hold': 'right', 'right_dust': 'left',
                'right_dust_hold': 'left', 'right_lid_raise': 'right', 'front_tuck': 'right', 'tuck_grip': 'right', 'tuck_clear': 'right', 'tuck_folded': 'right', 'close_tuck': 'right', 'lid_release': 'right'}
TAIL_PANELS = {'right_dust': 'lid_left_dust', 'right_dust_hold': 'lid_left_dust', 'right_lid_raise': 'lid_right_dust',
               'front_tuck': 'lid_right_dust', 'tuck_grip': 'lid_right_dust', 'tuck_clear': 'lid_right_dust',
               'tuck_folded': 'lid_right_dust', 'close_tuck': 'lid_right_dust',
               'lid_release': 'lid_right_dust'}


def finish_flaps(d, phase):
    import lid_sequence as ls
    from pack_checkpoint import save_checkpoint
    if phase == 'left_dust':
        save_checkpoint(d, 'left_dust', 'right')
        if abs(np.rad2deg(ls.absolute_lid(d))-89.) > 3.:
            ls.move_lid_to_work_angle(d, 'right', 89., 'bring right-held lid upright for the left flap')
        ls.fold_flap(d, 'dust', 'left', keep_grip=True)
    elif phase == 'left_dust_hold':
        ls.hold_folded_dust(d, 'left')
    if phase in ('left_dust', 'left_dust_hold'):
        save_checkpoint(d, 'left_dust_hold', 'right')
        ls.release_panel_support(d, 'right')
    if phase in ('left_dust', 'left_dust_hold', 'right_dust'):
        save_checkpoint(d, 'right_dust', 'left')
        ls.fold_flap(d, 'dust', 'right', keep_grip=True)
    elif phase == 'right_dust_hold':
        ls.hold_folded_dust(d, 'right')
    if phase not in ('right_lid_raise', 'front_tuck', 'tuck_grip', 'tuck_clear', 'tuck_folded'):
        save_checkpoint(d, 'right_dust_hold', 'left')
        ls.relax_held_dust(d, 'right')
        ls.release_panel_support(d, 'left')
        d.hold(.5, 'settle left flap after support transfer')
        left_angle = np.rad2deg(d.w.angles()['lid_left_dust'])
        if not 85. < left_angle < 100.:
            raise RuntimeError(f'Left flap changed to {left_angle:.1f} deg during support transfer')
    if phase not in ('front_tuck', 'tuck_grip', 'tuck_clear', 'tuck_folded'):
        save_checkpoint(d, 'right_lid_raise', 'right')
        ls.move_lid_to_work_angle(d, 'right', 100., 'raise lid for the final tuck fold')
        if abs(np.rad2deg(d.w.angles()['lid_right_dust'])) > 91.:
            ls.relax_held_dust(d, 'right')
    if phase != 'tuck_folded':
        if phase == 'tuck_clear':
            ls.finish_tuck_withdrawal(d)
        else:
            if phase != 'tuck_grip':
                save_checkpoint(d, 'front_tuck', 'right')
            ls.fold_tuck(d, partial=False, already_gripped=phase == 'tuck_grip')
        save_checkpoint(d, 'tuck_folded', 'right')
    a = d.w.angles()
    d.w.snapshot('stage-flaps-result')
    return record(d, 'flaps', {'angles_deg': {n: float(np.rad2deg(a[n])) for n in ['lid', 'front_tuck', 'lid_left_dust', 'lid_right_dust', 'tuck_left_ear', 'tuck_right_ear']}})


def resume_tail(d, phase):
    if phase == 'lid_release':
        return [close_from_right_support(d, already_pressed=True)]
    stages = [] if phase == 'close_tuck' else [finish_flaps(d, phase)]
    stages.append(close_tuck(d))
    return stages


# ---------------------------------------------------------------- L3: press and tuck (core)
def press_and_tuck(d):
    import lid_sequence as ls
    ref = ls.reference(d); start = ls.absolute_lid(d)
    d.support.pop('left', None)
    # 1) left hand rotates the supported lid down to ~12 deg open (tuck tip swings in behind the front wall)
    near = np.pi-np.deg2rad(12.)
    for k in range(320):
        f = d.ik.smoothstep((k+1)/320); angle = start*(1-f)+near*f
        shift = np.clip((np.rad2deg(angle)-88)/22, 0., 1.)
        p, r, _ = ls.push_pose(d, ref, angle, 'left', u=-.070+.030*shift, v=.240-.020*shift)
        d.feedback_step('left', p, r, 0., 'rotate lid down toward box'); ls.force_guard(d, 'left', 'lid lowering')
    d.w.snapshot('lid-near-closed')
    # 2) right fingertips on the lid front edge (just behind the tuck crease) push down; tuck slides in
    hv = d.w.manifest['tuck_hinge_v']; up = np.array([0., 0., 1.])
    lid_front = d.w.point('lid', [.040, hv-.012, 0.]).copy()          # current (12 deg open) lid point
    flush_front = box_point(d, [.040, -.066, .0506])
    Rr = press_rot(d, 'right', flush_front)
    tf = lambda q, pre=.0003: d.face_tool_pose(q, up, Rr, pre, tips_only=True)
    d.move_joints_safe('right', tf(lid_front)+[0., 0., .050], Rr, 3., 'prepare right fingertips over lid front edge')
    d.move('right', tf(lid_front, -.002), Rr, 0., 1.5, 'touch lid front edge')
    start_p = tf(lid_front, -.002)
    for k in range(220):
        f = d.ik.smoothstep((k+1)/220)
        d.feedback_step('right', start_p*(1-f)+tf(flush_front, .0015)*f, Rr, 0., 'push lid front edge down: tuck slides inside front wall')
        ls.force_guard(d, 'right', 'tuck insertion')
    d.hold(.5, 'hold front edge flush')
    d.w.snapshot('tuck-pushed-in')
    # 3) left presses the lid centre over the book (compress the foam to flush)
    centre = box_point(d, [-.040, .020, .0506])
    Rl = press_rot(d, 'left', centre)
    p0, r0 = d.pose('left')
    d.move('left', p0+[0., 0., .03], r0, 0., 1., 'lift left off lid back')
    d.move('left', d.face_tool_pose(centre, up, Rl, -.04, True), Rl, 0., 2., 'left over lid centre')
    for k in range(60):
        f = d.ik.smoothstep((k+1)/60)
        d.feedback_step('left', d.face_tool_pose(centre, up, Rl, -.04*(1-f)+.003*f, True), Rl, 0., 'left presses lid centre flush')
    d.hold(1., 'both hands hold lid flush')
    d.w.snapshot('lid-pressed-flush-both-hands')
    held = dict(d.w.last['pack_flags'])
    # 4) release: right first, then left slowly
    p, _ = d.pose('right'); d.move('right', p+[0., 0., .07], Rr, 0., 1.5, 'release right hand from lid front')
    d.hold(.5, 'tuck holding with left only')
    p, _ = d.pose('left')
    d.move('left', p+[0., 0., .008], Rl, 0., 2., 'slowly ease left press')
    d.move('left', p+[0., 0., .08], Rl, 0., 1.5, 'lift left hand clear')
    for side in ['left', 'right']:
        d.return_home(side)
    d.hold(5., 'verify packed closed box with both hands released')
    d.w.snapshot('stage-press_and_tuck-result')
    return record(d, 'press_and_tuck', {'flags_while_held': held, 'success': d.w.last['success'], 'hold_s': d.w.last['hold_s']})



# ---------------------------------------------------------------- S1: square the book (pack_grok002)
def square_book(d):
    """Verify |yaw| <= 3 deg; if not, fingertips push the FRONT face of the forward end back toward the back wall
    (front gap ~40 mm is the only reachable side; the back gap is ~5 mm), closed loop, max 3 pushes. Re-seat after."""
    B = pp.BOOK['size']; up = np.array([0., 0., 1.])
    for it in range(4):
        d.hold(.5, 'measure book squareness')
        k = d.w.last['pack_contents']; err = k['book_yaw_err_deg']; print('BOOK_SQUARE', it, err, flush=True)
        if err <= pp.BOOK_SQUARE_TOL_DEG-1. or it == 3: break
        _, _, bk = d.w.content_poses(); yaw = book_axis_yaw(bk); c = bk[:3, 3]
        ey = rz(yaw)@np.array([0., 1., 0.]); ex = rz(yaw)@np.array([1., 0., 0.]); r = B[0]/2-.030
        ends = {'left': c+ey*r, 'right': c-ey*r}
        side = 'left' if ends['left'][0] < ends['right'][0] else 'right'
        face = ends[side]-ex*(B[1]/2+.011); contact = np.array([face[0], face[1], c[2]])
        push = 2*r*abs(np.sin(yaw))+.006
        Rp = press_rot(d, side, contact); tip = d.face_tool_pose(contact, up, Rp, -.004, True)
        d.move_joints_safe(side, tip+[0., 0., .050], Rp, 2., 'above book front face for squaring push '+side)
        d.move(side, tip, Rp, 0., 2.5, 'lower fingertips in front of book '+side)
        for j in range(64):
            f = d.ik.smoothstep((j+1)/64)
            d.feedback_step(side, tip+[push*f, 0., 0.], Rp, 0., 'push book front face back to square '+side); force_guard(d, side, 'squaring push')
            if d.w.last['pack_contents']['book_yaw_err_deg'] < 1.0: print('SQUARE_PUSH_STOPPED_AT', j, flush=True); break
        # r5: lifting 60 mm while still 'pushed' was out of reach; back off 6 mm, then lift 30 mm, then home
        p, r = d.pose(side)
        d.move(side, p+[-.006, 0., 0.], r, 0., .5, 'back off from book front face '+side)
        try: d.move(side, p+[-.006, 0., .030], r, 0., .8, 'lift after squaring push '+side)
        except RuntimeError as e: print('SQUARE_LIFT_PARTIAL', e, flush=True)
        d.return_home(side)
    if not d.w.last['pack_flags']['book_on_pad']:
        try: seat_book(d)
        except RuntimeError as e: print('SEAT_BOOK_SKIPPED', e, flush=True)
    d.w.snapshot('stage-square_book-result')
    return record(d, 'square_book', {'book_yaw_err_deg': d.w.last['pack_contents']['book_yaw_err_deg']})


# ---------------------------------------------------------------- C1: press contents (pack_grok002)
def press_contents(d, release=True, side=None):
    """Compress the pad with one hand while the other supports the lid."""
    B = pp.BOOK['size']; up = np.array([0., 0., 1.])
    import lid_sequence as ls
    if side is None:
        side = 'left' if 'right' in d.support else 'right'
    _, _, bk = d.w.content_poses(); c = bk[:3, 3].copy()
    top = c+[0., 0., B[2]/2]
    # by grok r13: offline IK - 40 mm above the book centre is out of reach; try nearer press points/lower approaches
    done = False
    contacts = (((-.025, .035), .020), ((-.025, .035), .010)) if side == 'left' else (
        ((-.020, -.025), .020), ((-.010, -.025), .012), ((-.030, -.020), .020),
        ((-.020, -.025), .012), ((-.010, -.025), .040))
    for off, hgt in contacts:
        pt = top+[off[0], off[1], 0.]
        try:
            if side == 'left':
                z = np.array([np.sin(.5), 0., -np.cos(.5)])
                Rr = ls.frame(np.array([0., -1., 0.]), z)
            else:
                Rr = press_rot(d, side, pt)
            d.move_joints_safe(side, d.face_tool_pose(pt, up, Rr, -hgt, True), Rr, 2.5, 'prepare book compression'); done = True
            print('PRESS_APPROACH', off, hgt, flush=True); break
        except RuntimeError as e:
            print('PRESS_APPROACH_UNREACHABLE', off, hgt, e, flush=True)
    if not done: raise RuntimeError('No reachable approach above the book for the content press')
    d.move(side, d.face_tool_pose(pt, up, Rr, -.002, True), Rr, 0., 1., 'touch book top')
    initial_top = d.w.last['pack_contents']['book_top_z_m']
    contact = d.face_tool_pose(pt, up, Rr, .0005, True)
    preload = 0.
    t_reach = None
    for k in range(175):
        top_now = d.w.last['pack_contents']['book_top_z_m']
        contact_force = content_force(d, side, 'book')
        desired_force = 6.*d.ik.smoothstep(min(1., (k+1)/24))
        adjustment = np.clip(.0003*(desired_force-contact_force), -.001, .001)
        preload = np.clip(preload+adjustment, 0., .028)
        d.command_pose(side, contact+up*(top_now-initial_top-preload), Rr, 0., 'press book down with measured force')
        force_guard(d, side, 'content press')
        ex = d.w.last['pack_contents']['book_top_z_m']-pp.LID_UNDERSIDE_Z
        if t_reach is None and ex <= 0.:
            t_reach = k*.05
        if ex <= -.0024:
            break
    k = d.w.last['pack_contents']; res = {'book_top_excess_mm': (k['book_top_z_m']-pp.LID_UNDERSIDE_Z)*1e3, 'pad_joint_mm': d.w.last['pad_compression_joint_m']*1e3,
                                          'time_to_flush_s': t_reach, 'compressed_flag': d.w.last['pack_flags']['compressed_to_height'], 'press_force_n': d.hand_force(side)}
    print('CONTENTS_PRESSED', json.dumps(res, default=float), flush=True)
    d.w.snapshot('contents-pressed')
    if not d.w.last['pack_flags']['compressed_to_height']:
        raise RuntimeError('Contents not compressed to lid height: '+json.dumps(res, default=float))
    if release:
        p, r = d.pose(side); d.move(side, p+[0., 0., .05], r, 0., .6, 'lift off pressed book')
        t0 = d.w.steps
        for _ in range(int(6/.05)):
            d.step('watch rebound after release (diagnostic of stage)')
            if d.w.last['pack_contents']['book_top_z_m'] > pp.LID_UNDERSIDE_Z: break
        res['robot_release_hold_window_s'] = round((d.w.steps-t0)*.05+.6, 2)
        d.return_home(side)
    return record(d, 'press_contents', res)


# ---------------------------------------------------------------- C2: close + tuck inside the hold window (pack_grok002 core)
LID_PRE_OPEN_DEG = 65.    # lid pre-lowered to this opening while the right hand still holds the book down
LID_NEAR_OPEN_DEG = 12.


def push_lid_down(d, goal=162.):
    import lid_sequence as ls
    from gripper_geometry import hand_collision_bounds

    base = d.w.station.BASES['left']
    start = d.action[:6].copy()
    clear = ls.panel_clearance(d, ('lid',))
    bounds = hand_collision_bounds(0.)
    angle = ls.absolute_lid(d)
    for yaw in (1.4, 1.2):
        z = np.array([0., -np.sin(yaw), -np.cos(yaw)])
        x = np.cross(z, np.cross([0., -1., 0.], z))
        rot = ls.frame(x/np.linalg.norm(x), z)

        def contact(theta):
            hinge, axes = ls.lid_frame(ls.reference(d), theta)
            point = hinge+axes@np.array([0., .220-.130, 0.])
            normal = -axes[:, 2]
            return d.face_tool_pose(point, normal, rot, .0003), normal

        point, normal = contact(angle)
        prepare = point+normal*.025+[0., 0., .020]
        q, ep, er = d.ik.solve(prepare, d.ik.matrix_to_quat_xyzw(rot), start, base,
                              iters=240, damping=.018)
        if ep > .001 or er > .01:
            continue
        if not clear((d.ik.tcp_pose(j, base) for j in np.linspace(start, q, 40)), *bounds):
            continue
        if clear(((p, rot) for p in np.linspace(prepare, point, 15)), *bounds, allow_target=True):
            break
    else:
        raise RuntimeError('No clear approach for active lid closing')

    for k in range(26):
        d.action[:6] = start+(q-start)*d.ik.smoothstep((k+1)/26)
        d.action[6] = 0.
        d.step('approach above the lid with closed fingertips', 'left')
        force_guard(d, 'left', 'approach lid pusher')
    point, _ = contact(ls.absolute_lid(d))
    d.move('left', point, rot, 0., .6, 'touch outer lid surface')
    for _ in range(110):
        angle = ls.absolute_lid(d)
        if angle >= np.deg2rad(goal):
            break
        point, _ = contact(min(np.deg2rad(goal+1.), angle+np.deg2rad(1.5)))
        actual, _ = d.pose('left')
        correction = np.clip(.4*(point-actual), -.003, .003)
        d.command_pose('left', point+correction, rot, 0., 'push lid toward the front slot')
        force_guard(d, 'left', 'active lid closing')
    p, rot = d.pose('left')
    d.move('left', p+[0., 0., .040], rot, 0., .45, 'lift from the lid before seating the tuck')


def nudge_tuck_inward(d):
    import lid_sequence as ls

    hv = d.w.manifest['tuck_hinge_v']
    base = d.w.poses()['base']
    inward, up = base[:3, 1], base[:3, 2]
    normal = d.w.poses()['front_tuck'][:3, 2]
    if normal@inward > 0.:
        normal = -normal
    point = d.w.point('front_tuck', [.035, hv+.024, 0.])
    z = np.array([np.cos(.15)*np.cos(.12), -np.sin(.15)*np.cos(.12), -np.sin(.12)])
    x = np.cross(z, np.cross(up, z))
    rot = ls.frame(x/np.linalg.norm(x), z)
    touch = d.face_tool_pose(point, normal, rot, -.002, tips_only=True)
    d.move('right', touch-.040*z+[0., 0., .012], rot, 0., 1.1, 'approach the front of the tuck')
    d.move('right', touch, rot, 0., .55, 'position fingertips outside the tuck')
    for k in range(24):
        location = d.ik.smoothstep((k+1)/24)
        d.command_pose('right', touch+location*(.010*inward+.004*up), rot, 0., 'nudge tuck over the rim')
        force_guard(d, 'right', 'align tuck over the rim')
        c = d.w.last['pack_closure']
        if c['tuck_v_range_m'][0] >= -pp.INTERIOR_V-.0003:
            break
    measured = d.w.last['pack_closure']['tuck_v_range_m'][0]
    print('TUCK_RIM_ALIGNMENT_MM', float(1e3*measured), flush=True)
    p, rot = d.pose('right')
    d.move('right', p-.045*inward+.020*up, rot, 0., .5, 'clear fingers after aligning tuck')


def seat_closed_lid(d, u=-.040):
    import lid_sequence as ls
    from gripper_geometry import hand_collision_bounds

    up = np.array([0., 0., 1.])
    hv = d.w.manifest['tuck_hinge_v']
    x = np.array([0., -1., 0.])
    start = d.action[:6].copy()
    base = d.w.station.BASES['left']
    clearness = ls.panel_clearance(d, ('lid',))
    bounds = hand_collision_bounds(0.)
    front = d.w.point('lid', [u, hv-.018, 0.])
    flush = box_point(d, [u, -.060, .0506])
    for tilt in (.35, .20, .50, .70, 0.):
        z = np.array([np.sin(tilt), 0., -np.cos(tilt)])
        rot = ls.frame(x, z)
        touch = d.face_tool_pose(front, up, rot, -.003, tips_only=True)
        end = d.face_tool_pose(flush, up, rot, .004, tips_only=True)
        prepare = touch+[0., .012, .055]
        quat = d.ik.matrix_to_quat_xyzw(rot)
        q, ep, er = d.ik.solve(prepare, quat, start, base, iters=250, damping=.018)
        if ep > .001 or er > .01:
            continue
        if not clearness((d.ik.tcp_pose(k, base) for k in np.linspace(start, q, 40)), *bounds):
            continue
        if not clearness(((point, rot) for point in np.linspace(prepare, touch, 16)), *bounds, allow_target=True):
            continue
        for point in np.linspace(prepare, end, 12)[1:]:
            out, ep, er = d.ik.solve(point, quat, q, base, iters=180, damping=.018)
            if ep > .001 or er > .01:
                break
            q = out
        else:
            entry, _, _ = d.ik.solve(prepare, quat, start, base, iters=250, damping=.018)
            break
    else:
        raise RuntimeError('No clear downward press over the prepared paperboard lid')
    for k in range(25):
        d.action[:6] = start+(entry-start)*d.ik.smoothstep((k+1)/25)
        d.action[6] = 0.
        d.step('prepare fingertips above the front of the box', 'left')
        force_guard(d, 'left', 'prepare front lid press')
    d.move('left', touch, rot, 0., .65, 'touch top of lid near front edge')
    preload = 0.
    for k in range(90):
        front = d.w.point('lid', [u, hv-.018, 0.])
        desired_force = 5.*d.ik.smoothstep(min(1., (k+1)/20))
        preload = np.clip(preload+np.clip(.00025*(desired_force-d.panel_force('left', 'lid')), -.001, .001), 0., .019)
        target = d.face_tool_pose(front, up, rot, preload, tips_only=True)
        d.command_pose('left', target, rot, 0., 'press lid flush over the seated tuck')
        force_guard(d, 'left', 'seat front tuck')
        c = d.w.last['pack_closure']
        if c['lid_flush'] and c['tuck_inside_front_wall'] and d.w.last['pack_flags']['folded_flaps_seated']:
            return k+1
    return None


def close_from_right_support(d, already_pressed=False):
    import lid_sequence as ls
    from pack_checkpoint import save_checkpoint
    if not already_pressed:
        press_contents(d, release=False, side='left')
    up = np.array([0., 0., 1.])
    u = -.040
    flush = box_point(d, [u, -.066, .0506])
    r = press_rot(d, 'left', flush)
    def tool(point, depth):
        return d.face_tool_pose(point, up, r, depth, tips_only=True)

    release_step = d.w.steps
    insert_s = None
    if not already_pressed:
        p, rotation = d.pose('left')
        d.move('left', p+[0., 0., .025], rotation, 0., .35, 'lift fingertips clear of the book')
        d.move('left', tool(flush+[-.050, 0., .050], .0003), r, 0., .7, 'get ahead of the lid sweep')
        save_checkpoint(d, 'lid_release', 'right')
    hinge, axes = ls.lid_frame(ls.reference(d), np.deg2rad(114.))
    prop_point = hinge+axes@np.array([-.020, .235-.130, 0.])
    prop_z = np.array([np.sin(.5), -np.cos(.5)*np.sin(1.4), -np.cos(.5)*np.cos(1.4)])
    prop_x = np.cross(prop_z, np.cross([0., -1., 0.], prop_z))
    prop_x /= np.linalg.norm(prop_x)
    prop_rot = ls.frame(prop_x, prop_z)
    normal = axes[:, 2]
    prop = d.face_tool_pose(prop_point, normal, prop_rot, .0002)
    prepare = prop+normal*.045+[0., .020, .045]
    d.move('left', prepare, prop_rot, 0., 1.2, 'clear approach to guide the lid')
    d.move('left', prop, prop_rot, 0., .45, 'position left fingertips to catch the lid')
    if content_force(d, 'left', 'book') > .15:
        raise RuntimeError('Lid guide touched the book during its approach')
    d.support['left'] = (prop, prop_rot, 0.)
    d.support.pop('right')
    d.lid_clamps.pop('right', None)
    d.lid_hold_panels.pop('right', None)
    p, rotation = d.pose('right')
    d.move('right', p, rotation, .040, .2, 'open fingers before withdrawing')
    clear = p-.080*rotation[:, 2]+[-.015, -.015, .060]
    d.move('right', clear, rotation, .040, .55, 'retract along the right fingertip axis')
    force_guard(d, 'right', 'release prepared lid')
    for _ in range(5):
        d.step('verify right-hand clearance while holding guide')
        if d.panel_force('right', 'lid_right_dust') <= .15:
            break
    force_guard(d, 'left', 'hold lid for right-hand clearance')
    if d.panel_force('right', 'lid_right_dust') > .15:
        raise RuntimeError('Right fingertips still touch the flap after release')
    a = d.w.angles()
    if np.rad2deg(a['front_tuck']) < 95. or any(abs(abs(np.rad2deg(a[name]))-90.) > 4.
            for name in ('lid_left_dust', 'lid_right_dust')):
        raise RuntimeError('A folded panel moved out of position during the lid handoff')
    d.support.pop('left')
    p, rotation = d.pose('left')
    # Clear the swept path before the falling lid reaches the folded tab.
    retreat = np.array([-.165, .080, -.034])
    for k in range(5):
        d.command_pose('left', p+retreat*((k+1)/5), rotation, 0., 'retract ahead of the lid sweep')
        force_guard(d, 'left', 'clear lid guide')
    for _ in range(24):
        if np.rad2deg(ls.absolute_lid(d)) >= 165.:
            break
        d.step('let the prepared lid close')
    if np.rad2deg(ls.absolute_lid(d)) < 155.:
        ls.home_with_gap(d, 'left', 0.)
        p, rotation = d.pose('left')
        d.move('left', p, rotation, 0., .3, 'prepare closed fingertips for the lid')
        push_lid_down(d)
    angle = np.rad2deg(ls.absolute_lid(d))
    print('RELEASED_LID_DEG', angle, flush=True)
    if angle < 150.:
        raise RuntimeError(f'Prepared lid stalled at {angle:.1f} degrees after release')

    if d.w.last['pack_closure']['tuck_v_range_m'][0] < -pp.INTERIOR_V-.0003:
        nudge_tuck_inward(d)
    seated = seat_closed_lid(d, u)
    if seated is not None:
        insert_s = round((d.w.steps-release_step)*.05, 2)
    d.hold(.4, 'verify the tabs seated under the lid')
    held = dict(d.w.last['pack_flags'])
    p, rotation = d.pose('left')
    d.move('left', p+[0., 0., .050], rotation, 0., .7, 'clear the closed lid')
    d.return_home('left')
    p, rotation = d.pose('right')
    if p[2] < d.w.support_top+.135:
        d.move('right', p+[0., 0., .045], rotation, 0., .8, 'raise right hand before returning home')
    d.return_home('right')
    d.hold(5., 'verify packed closed box with both hands released')
    d.w.snapshot('stage-close_tuck-result')
    return record(d, 'close_tuck', {'flags_while_held': held, 'released_lid_deg': angle,
                                  'closing_start_to_insert_s': insert_s,
                                  'success': d.w.last['success'], 'hold_s': d.w.last['hold_s']})


def close_tuck(d):
    """pack_grok002 (timing fix): while the right fingertips still hold the contents pressed, the left hand pre-lowers
    the lid to ~65 deg open (clear of the right wrist, which sits in front). The hold window starts when the right
    hand lets go. Then, in one coordinated loop, the left lowers the lid 65 -> 12 deg (1.5 s) while the right goes to
    its waiting pose in front of the front wall; the right touches the lid front edge and pushes it down (1.5 s) so
    the tuck slides inside the front wall; the left presses the lid centre flush; both release; 5 s verify."""
    import lid_sequence as ls
    from pack_checkpoint import save_checkpoint
    if d.lid_hold_panels.get('right') == 'lid_right_dust':
        save_checkpoint(d, 'close_tuck', 'right')
        return close_from_right_support(d)
    save_checkpoint(d, 'close_tuck', 'left')
    press_contents(d, release=False)
    hv = d.w.manifest['tuck_hinge_v']; up = np.array([0., 0., 1.])
    flush_front = box_point(d, [.040, -.066, .0506])
    Rr = press_rot(d, 'right', flush_front)
    tf = lambda q, pre=.0003: d.face_tool_pose(q, up, Rr, pre, tips_only=True)
    wait = tf(flush_front+[-.050, 0., .050])   # r13: offline IK, [-.03,0,.06] out of reach; still outside lid+tuck sweep (0.20 m from hinge)
    start = ls.absolute_lid(d); d.support.pop('left', None)
    clamped = 'left' in getattr(d, 'lid_clamps', {})
    near_open = LID_NEAR_OPEN_DEG
    if clamped:
        near_open = 18. if d.lid_clamps['left'][0][1] > .08 else 32.
    pre = np.pi-np.deg2rad(LID_PRE_OPEN_DEG); near = np.pi-np.deg2rad(near_open)
    print('LID_START_ABS_DEG', float(np.rad2deg(start)), flush=True)
    def lid_target(angle):
        if clamped:
            return ls.lid_clamp_target(d, 'left', angle)
        shift = np.clip((np.rad2deg(angle)-88)/22, 0., 1.)
        return ls.push_pose(d, ls.reference(d), angle, 'left', u=-.070+.030*shift, v=.240-.020*shift)[:2]
    if start < pre:
        for k in range(60):
            f = d.ik.smoothstep((k+1)/60); p, r = lid_target(start*(1-f)+pre*f)
            gap = ls.holding_gap(d, 'left', 'lid') if clamped else 0.
            d.feedback_step('left', p, r, gap, 'pre-lower lid while contents held'); ls.force_guard(d, 'left', 'lid pre-lowering'); ls.force_guard(d, 'right', 'content hold')
        start = pre
    d.w.snapshot('lid-prelowered-contents-held')
    excess = lambda: d.w.last['pack_contents']['book_top_z_m']-pp.LID_UNDERSIDE_Z
    print('PRE_RELEASE_EXCESS_MM', excess()*1e3, flush=True)
    # ---- hold window starts: right lets go
    t_rel = d.w.steps; trace = []
    p_r0, r_r0 = d.pose('right'); lift = p_r0+[0., 0., .030]
    n = 30
    for k in range(n):
        f = d.ik.smoothstep((k+1)/n); p, r = lid_target(start*(1-f)+near*f)
        gap = ls.holding_gap(d, 'left', 'lid') if clamped else 0.
        d.feedback_command('left', p, r, gap, 'lower lid fast toward box')
        if k < 6: q = p_r0*(1-(k+1)/6)+lift*((k+1)/6); rr = r_r0
        else:
            g = d.ik.smoothstep(min(1., (k-5)/14)); q = lift*(1-g)+wait*g; rr = d.ik.slerp_matrix(r_r0, Rr, g)
        d.solve_command('right', q, rr, 0., 'right to waiting pose in front of front wall')
        d.step('lower lid fast + right to waiting pose'); ls.force_guard(d, 'left', 'lid lowering')
        trace.append(((d.w.steps-t_rel)*.05, excess()))
    lid_front = d.w.point('lid', [.040, hv-.012, 0.]).copy()
    d.move('right', tf(lid_front, -.002), Rr, 0., .4, 'touch lid front edge')
    if clamped:
        p, r = d.pose('left')
        d.move('left', p, r, .045, .25, 'release lid to the right-hand press')
        d.lid_clamps.pop('left')
    s0 = tf(lid_front, -.002); t_tuck = None
    for k in range(30):
        f = d.ik.smoothstep((k+1)/30)
        d.feedback_step('right', s0*(1-f)+tf(flush_front, .0015)*f, Rr, 0., 'push lid front edge down: tuck inside front wall'); ls.force_guard(d, 'right', 'tuck insertion')
        trace.append(((d.w.steps-t_rel)*.05, excess()))
        if t_tuck is None and d.w.last['pack_closure']['tuck_inside_front_wall']: t_tuck = (d.w.steps-t_rel)*.05
    t_push_end = (d.w.steps-t_rel)*.05
    d.hold(.3, 'hold front edge flush')
    if t_tuck is None and d.w.last['pack_closure']['tuck_inside_front_wall']: t_tuck = (d.w.steps-t_rel)*.05
    print('TUCK_TIMING', json.dumps({'t_push_end_s': t_push_end, 't_tuck_inside_s': t_tuck, 'hold_window_ref_s': 4.80,
                                     'excess_mm_at_push_end': trace[-1][1]*1e3}, default=float), flush=True)
    d.w.snapshot('tuck-pushed-in')
    centre = box_point(d, [-.040, .020, .0506]); Rl = press_rot(d, 'left', centre)
    p0, r0 = d.pose('left')
    d.move('left', p0+[0., 0., .03], r0, .045 if clamped else 0., .5, 'lift left off lid')
    d.move('left', d.face_tool_pose(centre, up, Rl, -.02, True), Rl, 0., 1., 'left over lid centre')
    for k in range(30):
        f = d.ik.smoothstep((k+1)/30)
        d.feedback_step('left', d.face_tool_pose(centre, up, Rl, -.02*(1-f)+.003*f, True), Rl, 0., 'left presses lid centre flush')
    d.hold(1., 'both hands hold lid flush')
    d.w.snapshot('lid-pressed-flush-both-hands'); held = dict(d.w.last['pack_flags'])
    p, _ = d.pose('right'); d.move('right', p+[0., 0., .07], Rr, 0., 1., 'release right hand from lid front')
    d.hold(.5, 'tuck holding with left only')
    p, _ = d.pose('left'); d.move('left', p+[0., 0., .008], Rl, 0., 1.5, 'slowly ease left press'); d.move('left', p+[0., 0., .08], Rl, 0., 1., 'lift left hand clear')
    for side in ['left', 'right']: d.return_home(side)
    d.hold(5., 'verify packed closed box with both hands released')
    d.w.snapshot('stage-close_tuck-result')
    return record(d, 'close_tuck', {'flags_while_held': held, 'tuck_in_s_after_release': t_tuck, 'push_end_s_after_release': t_push_end,
                                    'within_hold_window': bool(t_tuck is not None and t_tuck <= 4.80),
                                    'excess_trace_mm': [(round(t, 2), round(e*1e3, 2)) for t, e in trace[::3]],
                                    'success': d.w.last['success'], 'hold_s': d.w.last['hold_s']})


def press_rot(d, side, point):
    """Fingertips-down press orientation that the arm can reach at this point (tilt searched)."""
    s = 1 if side == 'left' else -1
    for t in (.35, .2, .5, 0., .7):
        for flip in (1., -1.):
            zz = np.array([np.sin(t), 0., -np.cos(t)]); x = np.array([0., -s*flip, 0.]); R = np.column_stack([x, np.cross(zz, x), zz])
            q, ep, er = d.ik.solve(point+[0., 0., .02], d.ik.matrix_to_quat_xyzw(R), d.w.constants.READY, d.w.station.BASES[side], iters=600, damping=.018)
            if ep < .001 and er < .01:
                return R
    raise RuntimeError('No reachable press orientation for '+side)


ORDER = ['place_pad', 'place_book', 'square_book', 'lift_lid', 'flaps', 'press_contents', 'close_tuck']   # pack_grok002
FUNCS = {'place_pad': place_pad, 'place_book': place_book, 'square_book': square_book, 'lift_lid': lift_lid, 'flaps': flaps,
         'press_contents': press_contents, 'close_tuck': close_tuck}


def run(w, last):
    d = Driver(w); d.hold(1., 'settle start scene'); w.snapshot('pack-start')
    results = []
    names = ORDER[:ORDER.index(last)+1]
    if 'close_tuck' in names: names.remove('press_contents')   # close_tuck re-presses itself (timing-coupled)
    for name in names:
        results.append(FUNCS[name](d))
    return results
