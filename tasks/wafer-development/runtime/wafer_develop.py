"""Timed photoresist-development task; the process response is a declared surrogate."""
import math
import numpy as np

SPEC = dict(version='wafer_develop_v3_no_gun', timer_mode='countdown', diameter_m=.1, thickness_m=.001, flat_m=.0325,
    target_dwell_s=30., good_dwell_s=[28., 32.], maximum_duration_s=120.,
    beaker_inner_radius_m=.085, beaker_outer_radius_m=.09, beaker_height_m=.16,
    beaker_floor_m=.006, liquid_height_m=.13, wall_clearance_m=.002,
    vertical_tolerance_deg=5., dry_hand_clearance_m=.01, withdrawal_clearance_m=.002,
    withdrawal_hold_s=1., return_hold_s=1., timer_start_sync_s=1., timer_press_m=.001,
    timer_press_force_N=.2, timer_debounce_s=.05, timer_release_m=.0003,
    timer_screen_tilt_deg=60., timer_yaw_deg=-115., beaker_origin_world_m=[.24,-.12,.7475],
    timer_origin_world_m=[.24,-.4,.7475],
    timer_button_local_m=[.032,-.020,.034],
    initial_right_arm_q_rad=[-.5063752205871798,1.4440168668703728,1.2735138901132785,-1.4002964340317847,-.0000012075757193059913,-.07003292889226712],
    return_position_tolerance_m=.001, return_orientation_tolerance_deg=2.,
    placement_support_window_s=.05, placement_min_mean_support_N=.05,
    placement_ready_gap_m=.00005, placement_release_gap_m=.0005,
    placement_seating_step_m=.00005, placement_seating_max_commands=10,
    release_feedback_gain_per_command=1.,
    minimum_final_pillar_contact_N=.005,
    maximum_wafer_slip_m=.001, maximum_wafer_rotation_deg=2.,
    maximum_unexpected_contact_N=.01, maximum_grip_contact_gap_s=.02,
    process_scope='Pattern quality uses continuous full-immersion time only. Partial wetting during insertion/withdrawal, chemistry, diffusion, temperature and real resist kinetics are not calibrated. Jaw tips may enter liquid; handles and robot hands must stay dry.')


def pose_matrix(pose):
    p = np.asarray(pose, dtype=float)
    if p.shape != (7,) or not np.isfinite(p).all():
        raise ValueError('Invalid measured pose.')
    x, y, z, w = p[3:]
    if abs(np.linalg.norm(p[3:]) - 1.) > 1e-4:
        raise ValueError('Quaternion is not normalized.')
    r = np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
    return p[:3], r


def make_develop_return_feedback(rows, previous, tcp_pose, solve_ik, lower, upper):
    """Bound additional correction about the existing handover command, not the
    unloaded target. Measured placement limits and the shared step guards remain.
    """
    from wafer_grasp_protocol import _pose, _angle
    from wafer_grasp_feedback import BoundedTCPFeedback
    baseline = _pose(next(r for r in reversed(rows) if r['phase'] == 'settle')['wafer_pose_world'], 'baseline')
    current = rows[-1]
    wafer = _pose(current['wafer_pose_world'], 'wafer')
    position_error = float(current['wafer_return_position_error_m'])
    angle_error = float(current['wafer_return_rotation_deg'])
    pose_position_error = float(np.linalg.norm(wafer[:3, 3]-baseline[:3, 3]))
    pose_angle_error = math.degrees(_angle(wafer[:3, :3], baseline[:3, :3]))
    if (not np.isfinite([position_error, angle_error]).all() or position_error < 0. or angle_error < 0.
        or max(position_error, pose_position_error) > SPEC['return_position_tolerance_m']
        or max(angle_error, pose_angle_error) > SPEC['return_orientation_tolerance_deg']):
        raise RuntimeError('Wafer is outside the existing placement limits at return handover.')
    entry = dict(source='Recorded native wafer pose against last settle pose; reported errors also checked.',
        position_error_m=pose_position_error, rotation_error_deg=pose_angle_error,
        maximum_position_error_m=SPEC['return_position_tolerance_m'],
        maximum_rotation_error_deg=SPEC['return_orientation_tolerance_deg'])
    relation = np.linalg.inv(wafer) @ _pose(current['measured_tcp_pose_world'], 'tcp')
    desired = baseline @ relation
    nominal = np.asarray(previous, dtype=float).copy()
    if nominal.shape != (6,) or not np.isfinite(nominal).all():
        raise ValueError('Invalid return handover command.')
    servo = BoundedTCPFeedback(tcp_pose, solve_ik, lower, upper)
    if np.any(nominal < servo.lower) or np.any(nominal > servo.upper):
        raise ValueError('Return handover command violates a joint limit.')
    nominal_p, nominal_r = (np.array(x, copy=True) for x in tcp_pose(nominal))

    class DevelopReturnFeedback:
        def step(self, nominal_q, wafer_p, wafer_r, previous_q, *, integrate=True):
            if not np.array_equal(nominal_q, nominal):
                raise ValueError('Return handover nominal must remain fixed.')
            measured_p = wafer_p + wafer_r @ relation[:3, 3]
            measured_r = wafer_r @ relation[:3, :3]
            virtual_p = nominal_p + measured_p - desired[:3, 3]
            virtual_r = measured_r @ desired[:3, :3].T @ nominal_r
            q, record = servo.step(nominal, virtual_p, virtual_r, previous_q, integrate=integrate)
            record['measurement'] = 'Virtual TCP error from measured wafer pose; not a measured TCP pose.'
            record['correction_reference'] = 'Additional correction about the fixed handover FK command.'
            record['entry_check'] = entry.copy()
            record['actual_wafer_position_world_m'] = np.asarray(wafer_p).tolist()
            record['actual_wafer_rotation_world'] = np.asarray(wafer_r).tolist()
            record['measured_equivalent_tcp_position_world_m'] = measured_p.tolist()
            record['desired_equivalent_tcp_position_world_m'] = desired[:3, 3].tolist()
            record['desired_equivalent_tcp_rotation_world'] = desired[:3, :3].tolist()
            record['virtual_tcp_position_world_m'] = record.pop('measured_tcp_position_world_m')
            return q, record

    return DevelopReturnFeedback(), nominal.copy()


def make_develop_release_feedback(current, previous, tcp_pose, solve_ik, lower, upper):
    """Counter measured wafer motion as passive jaws unload; freeze once released."""
    from wafer_grasp_protocol import _pose
    from wafer_grasp_feedback import BoundedTCPFeedback, PROFILE, rotation_vector
    if not current['wafer_on_pillars'] or not placement_geometry_ok(current,SPEC['placement_ready_gap_m']):
        raise RuntimeError('Release feedback requires a seated wafer.')
    initial_wafer=_pose(current['wafer_pose_world'],'seated wafer')
    desired=_pose(current['measured_tcp_pose_world'],'release TCP')
    relation=np.linalg.inv(initial_wafer)@desired
    nominal=np.asarray(previous,dtype=float).copy();frozen_q=None;count=0
    if nominal.shape!=(6,)or not np.isfinite(nominal).all():raise ValueError('Invalid release handover command.')
    def solve_request(p,r,q):
        if frozen_q is None:return solve_ik(p,r,q)
        actual_p,actual_r=tcp_pose(frozen_q)
        return frozen_q.copy(),float(np.linalg.norm(actual_p-p)),float(np.linalg.norm(rotation_vector(r@actual_r.T)))
    servo=BoundedTCPFeedback(tcp_pose,solve_request,lower,upper)
    if np.any(nominal<servo.lower)or np.any(nominal>servo.upper):raise ValueError('Release handover violates a joint limit.')
    nominal_p,nominal_r=(np.array(x,copy=True)for x in tcp_pose(nominal))
    def bounded(value,limit):
        length=np.linalg.norm(value)
        return value if length<=limit else value*(limit/length)

    class ReleaseFeedback:
        def step(self,row,previous_q):
            nonlocal frozen_q,count
            wafer=_pose(row['wafer_pose_world'],'release wafer');measured=wafer@relation
            ep=desired[:3,3]-measured[:3,3];er=rotation_vector(desired[:3,:3]@measured[:3,:3].T)
            released=bool(row['wafer_released']and row['wafer_all_pillars_loaded'])
            if released and frozen_q is None:frozen_q=np.asarray(previous_q,dtype=float).copy()
            if frozen_q is None:
                for error,kind in ((ep,'translation'),(er,'rotation')):
                    unit='m'if kind=='translation'else'rad';correction=getattr(servo,kind).copy()
                    if np.linalg.norm(error)>PROFILE[f'{kind}_deadband_{unit}']:
                        correction+=bounded(SPEC['release_feedback_gain_per_command']*error,PROFILE[f'maximum_{kind}_increment_{unit}'])
                    setattr(servo,kind,bounded(correction,PROFILE[f'maximum_{kind}_correction_{unit}']))
            virtual_p=nominal_p-ep;virtual_r=measured[:3,:3]@desired[:3,:3].T@nominal_r
            q,record=servo.step(nominal,virtual_p,virtual_r,previous_q,integrate=False);count+=1
            record.update(command_index=count,controller='measured_wafer_release_v1',
                correction_reference='Additional correction about the fixed pre-opening arm command.',
                measurement='Virtual TCP error from measured wafer pose; not a measured TCP pose.',
                gain_per_command=SPEC['release_feedback_gain_per_command'],frozen_after_release=frozen_q is not None,
                released_and_all_pillars_loaded=released,
                actual_wafer_position_world_m=wafer[:3,3].tolist(),actual_wafer_rotation_world=wafer[:3,:3].tolist(),
                desired_wafer_pose_world=list(current['wafer_pose_world']),
                measured_equivalent_tcp_position_world_m=measured[:3,3].tolist(),
                desired_equivalent_tcp_position_world_m=desired[:3,3].tolist(),desired_equivalent_tcp_rotation_world=desired[:3,:3].tolist())
            record['virtual_tcp_position_world_m']=record.pop('measured_tcp_position_world_m')
            return q,record
    return ReleaseFeedback()


def placement_geometry_ok(row, maximum_gap_m, spec=SPEC):
    gaps = np.asarray(row['wafer_pillar_plane_gaps_m'], dtype=float)
    errors = np.asarray([row['wafer_return_position_error_m'], row['wafer_return_rotation_deg']], dtype=float)
    return bool(gaps.shape == (3,) and np.isfinite(gaps).all() and np.isfinite(errors).all()
        and row['wafer_pillar_footprints_covered'] and np.max(np.abs(gaps)) <= maximum_gap_m
        and 0. <= errors[0] <= spec['return_position_tolerance_m']
        and 0. <= errors[1] <= spec['return_orientation_tolerance_deg'])


def placement_preopen_gate(rows, physics_dt_s, spec=SPEC):
    """Require sustained first contact, not full load transfer while jaws are closed."""
    dt = float(physics_dt_s)
    if not math.isfinite(dt) or dt <= 0.:
        raise ValueError('Invalid placement physics step.')
    count = round(spec['placement_support_window_s']/dt)
    if count < 1 or abs(count*dt-spec['placement_support_window_s']) > 1e-9:
        raise ValueError('Placement window must contain whole physics steps.')
    if len(rows) < count:
        return dict(passed=False, samples=len(rows), reason='Insufficient placement contact history.')
    window = list(rows[-count:])
    times = np.asarray([r['time_s'] for r in window], dtype=float)
    forces = np.asarray([r['pillar_contact_normal_force_N'] for r in window], dtype=float)
    gaps = np.asarray([r['wafer_pillar_plane_gaps_m'] for r in window], dtype=float)
    if (forces.shape != (count, 3) or gaps.shape != (count, 3)
        or not all(np.isfinite(x).all() for x in (times, forces, gaps)) or np.any(forces < 0.)):
        raise ValueError('Invalid measured placement contact history.')
    totals = forces.sum(axis=1)
    record = dict(window_s=count*dt, samples=count, start_time_s=float(times[0]), end_time_s=float(times[-1]),
        mean_support_force_N=float(totals.mean()), minimum_support_force_N=float(totals.min()),
        maximum_absolute_gap_m=float(np.max(np.abs(gaps))),
        all_footprints_covered=all(r['wafer_pillar_footprints_covered'] for r in window),
        current_on_pillars=bool(window[-1]['wafer_on_pillars']),
        continuous=bool(np.all(np.abs(np.diff(times)-dt) <= 1e-8)))
    record['passed'] = bool(record['continuous'] and record['current_on_pillars']
        and record['mean_support_force_N'] > spec['placement_min_mean_support_N']
        and record['minimum_support_force_N'] > 0.
        and all(placement_geometry_ok(r, spec['placement_ready_gap_m'], spec) for r in window))
    return record


def wafer_vertices(spec=SPEC):
    radius = spec['diameter_m']/2
    flat = -math.sqrt(radius**2 - (spec['flat_m']/2)**2)
    angle = math.acos(flat/radius)
    theta = np.linspace(-angle, angle, 129)
    ring = np.column_stack([radius*np.cos(theta), radius*np.sin(theta)])
    return np.concatenate([np.column_stack([ring, np.full(len(ring), z)])
        for z in (-spec['thickness_m']/2, spec['thickness_m']/2)])


def immersion_geometry(wafer_pose, beaker_origin, spec=SPEC):
    p, r = pose_matrix(wafer_pose)
    vertices = wafer_vertices(spec) @ r.T + p - np.asarray(beaker_origin)
    low, high = float(vertices[:, 2].min()), float(vertices[:, 2].max())
    radial = float(np.linalg.norm(vertices[:, :2], axis=1).max())
    liquid = spec['liquid_height_m']
    return dict(fully_immersed=high <= liquid and low >= spec['beaker_floor_m'] + spec['wall_clearance_m']
        and radial <= spec['beaker_inner_radius_m'] - spec['wall_clearance_m'],
        fully_withdrawn=low >= liquid + spec['withdrawal_clearance_m'],
        bottom_m=low, top_m=high, radial_extent_m=radial,
        deviation_from_vertical_deg=math.degrees(math.asin(min(1., abs(float(r[2, 2]))))))


def pattern_quality(dwell_s, spec=SPEC):
    if not math.isfinite(dwell_s) or dwell_s < 0:
        raise ValueError('Invalid development time.')
    low, high = spec['good_dwell_s']
    return 'UNDERDEVELOPED' if dwell_s < low else 'OVERDEVELOPED' if dwell_s > high else 'GOOD'


class ButtonTimer:
    def __init__(self, spec=SPEC):
        self.spec = spec
        self.started_at = None
        self.stopped_at = None
        self.press_since = None
        self.latched = False
        self.start_count = 0

    def update(self, time_s, displacement_m, force_N):
        if not all(math.isfinite(x) for x in (time_s, displacement_m, force_N)):
            raise ValueError('Non-finite timer evidence.')
        pressed = displacement_m >= self.spec['timer_press_m'] and force_N >= self.spec['timer_press_force_N']
        if self.latched:
            if displacement_m <= self.spec['timer_release_m']:
                self.press_since=None;self.latched=False
            return self.elapsed(time_s)
        if not pressed:
            self.press_since = None
            self.latched = False
        elif self.press_since is None:
            self.press_since = time_s
        elif not self.latched and time_s-self.press_since >= self.spec['timer_debounce_s']-1e-9:
            self.latched = True
            self.start_count += 1
            if self.started_at is None:
                self.started_at = time_s
            elif self.stopped_at is None:
                self.stopped_at = time_s
        return self.elapsed(time_s)

    def elapsed(self,time_s):
        return 0. if self.started_at is None else (time_s if self.stopped_at is None else self.stopped_at)-self.started_at

    def remaining(self,time_s):
        return max(0.,self.spec['target_dwell_s']-self.elapsed(time_s))


class DevelopmentScore:
    def __init__(self, beaker_origin, physics_dt_s, spec=SPEC):
        self.spec = dict(spec)
        self.origin = np.asarray(beaker_origin, dtype=float)
        self.dt = float(physics_dt_s)
        self.timer = ButtonTimer(spec)
        self.timer_started_fully_immersed = None
        self.last = None
        self.entered_at = None
        self.left_at = None
        self.withdrawn_s = 0.
        self.withdrawal_verified = False
        self.returned_s = 0.
        self.released = False
        self.failures = set()
        self.readings = []
        self.pickup = False
        self.count = 0

    def update(self, row):
        t = float(row['time_s'])
        numeric = ('time_s', 'timer_button_displacement_m', 'timer_button_force_N', 'timer_elapsed_s', 'timer_remaining_s',
            'unexpected_contact_N', 'hand_handle_liquid_clearance_m', 'wafer_slip_m', 'wafer_rotation_error_deg',
            'wafer_return_position_error_m', 'wafer_return_rotation_deg')
        if not all(math.isfinite(float(row[key])) for key in numeric):
            raise ValueError('Non-finite task evidence.')
        pillar_forces = np.asarray(row['pillar_contact_normal_force_N'], dtype=float)
        if pillar_forces.shape != (3,) or not np.isfinite(pillar_forces).all() or np.any(pillar_forces < 0.):
            raise ValueError('Invalid measured pillar forces.')
        all_pillars_loaded = bool(np.min(pillar_forces) > self.spec['minimum_final_pillar_contact_N'])
        if bool(row['wafer_all_pillars_loaded']) != all_pillars_loaded:
            self.failures.add('pillar_support_disagrees_with_contact_history')
        if self.last is not None and abs(t-self.last-self.dt) > 1e-8:
            raise ValueError('Discontinuous physics evidence.')
        if self.last is None and abs(t) > 1e-8:
            raise ValueError('Task must be recorded from its initial state.')
        self.last = t
        self.count += 1
        g = immersion_geometry(row['wafer_pose_world'], self.origin, self.spec)
        previous_start = self.timer.started_at
        elapsed = self.timer.update(t, row['timer_button_displacement_m'], row['timer_button_force_N'])
        if previous_start is None and self.timer.started_at is not None:
            self.timer_started_fully_immersed = bool(g['fully_immersed'])
            if not self.timer_started_fully_immersed:
                self.failures.add('timer_started_before_full_immersion')
        displayed = self.timer.remaining(t)
        if abs(elapsed-row['timer_elapsed_s']) > 1e-6 or abs(displayed-row['timer_remaining_s']) > 1e-6:
            self.failures.add('timer_display_disagrees_with_contact_history')
        if row.get('timer_reading_s') is not None:
            reading = float(row['timer_reading_s'])
            if not math.isfinite(reading):raise ValueError('Non-finite timer reading.')
            if abs(reading-displayed) > 1e-6:
                self.failures.add('controller_reading_disagrees_with_timer')
            self.readings.append((t, reading))
        if row.get('pickup_verified'):
            self.pickup = True
        if not row['objects_intact'] or not row['tool_hand_retained']:
            self.failures.add('integrity_or_tool_retention')
        if self.pickup and row['unexpected_contact_N'] > self.spec['maximum_unexpected_contact_N']:
            self.failures.add('unexpected_contact')
        if row['hand_handle_liquid_clearance_m'] < self.spec['dry_hand_clearance_m']:
            self.failures.add('hand_or_handle_entered_liquid')
        if self.pickup and not self.released:
            supported_release = (self.withdrawal_verified and row['wafer_on_pillars']
                and row['wafer_return_position_error_m'] <= self.spec['return_position_tolerance_m']
                and row['wafer_return_rotation_deg'] <= self.spec['return_orientation_tolerance_deg'])
            if not supported_release and (row['wafer_slip_m'] > self.spec['maximum_wafer_slip_m'] or row['wafer_rotation_error_deg'] > self.spec['maximum_wafer_rotation_deg']):
                self.failures.add('wafer_slip')
            if not row['wafer_bilaterally_held'] and not supported_release:
                self.failures.add('lost_wafer_grip')
            if row['wafer_released'] and supported_release:
                self.released = True
        if g['fully_immersed']:
            if self.entered_at is None:
                self.entered_at = t
            if self.left_at is not None:
                self.failures.add('repeated_or_interrupted_dip')
            if g['deviation_from_vertical_deg'] > self.spec['vertical_tolerance_deg']:
                self.failures.add('wafer_not_vertical')
        elif self.entered_at is not None and self.left_at is None:
            self.left_at = t
        if self.left_at is not None and g['fully_withdrawn']:
            self.withdrawn_s += self.dt
        else:
            self.withdrawn_s = 0.
        self.withdrawal_verified |= self.withdrawn_s+1e-8 >= self.spec['withdrawal_hold_s']
        good_return = (self.released and row['wafer_released'] and row['wafer_on_pillars']
            and all_pillars_loaded and row['wafer_return_stable']
            and row['wafer_return_position_error_m'] <= self.spec['return_position_tolerance_m']
            and row['wafer_return_rotation_deg'] <= self.spec['return_orientation_tolerance_deg'])
        self.returned_s = self.returned_s+self.dt if good_return else 0.
        return g

    def result(self):
        dwell = None if self.entered_at is None else round((self.last if self.left_at is None else self.left_at)-self.entered_at, 9)
        quality = 'NOT_DEVELOPED' if dwell is None else pattern_quality(dwell, self.spec)
        timer_ok = (self.timer.start_count == 2 and self.entered_at is not None
            and self.timer.started_at is not None and self.timer.stopped_at is not None
            and self.left_at is not None
            and self.timer_started_fully_immersed is True
            and 0. <= self.timer.started_at-self.entered_at <= self.spec['timer_start_sync_s']
            and abs(self.timer.stopped_at-self.left_at) <= self.spec['timer_start_sync_s'])
        read_before_lift = self.left_at is not None and any(self.left_at-1. <= t <= self.left_at for t, _ in self.readings)
        checks = dict(pickup=self.pickup, full_immersion=self.entered_at is not None,
            timer_operated_and_read=timer_ok and read_before_lift,
            dwell_in_28_to_32_seconds=quality == 'GOOD',
            withdrawn_and_held=self.withdrawal_verified,
            returned_and_released=self.returned_s+1e-8 >= self.spec['return_hold_s'],
            duration=self.last is not None and self.last <= self.spec['maximum_duration_s'],
            integrity_and_handling=not self.failures)
        weights = dict(pickup=.15, full_immersion=.15, timer_operated_and_read=.15,
            dwell_in_28_to_32_seconds=.25, withdrawn_and_held=.15, returned_and_released=.15)
        passed = all(checks.values())
        score = sum(weight for key, weight in weights.items() if checks[key]) if checks['integrity_and_handling'] and checks['duration'] else 0.
        return dict(success=passed, score=round(score, 6), checks=checks,
            failures=sorted(self.failures), immersion_start_s=self.entered_at,
            immersion_end_s=self.left_at, development_seconds=dwell, pattern_quality=quality,
            pattern_visible=self.left_at is not None and quality == 'GOOD' and checks['withdrawn_and_held'],
            timer_started_s=self.timer.started_at, timer_stopped_s=self.timer.stopped_at,
            recorded_samples=self.count, scope=self.spec['process_scope'])
