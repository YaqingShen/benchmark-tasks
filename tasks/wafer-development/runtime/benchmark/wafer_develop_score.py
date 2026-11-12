"""Submission rubric v1: visible handling, mechanical timer and immersion time."""
from collections import deque
import math
import numpy as np
from wafer_develop import SPEC, immersion_geometry
from wafer_grasp_protocol import _pose, _angle, pad_contact_geometry, pillar_plane_clearances

RUBRIC = dict(version='wafer_development_practical_v1', sample_hz=20,
    pliers_lift_m=.03, wafer_lift_m=.01, pickup_hold_s=.2,
    tool_shift_m=.002, tool_angle_deg=3., dry_clearance_m=.01,
    immersion_angle_deg=5., start_delay_s=1., exit_after_countdown_s=2.,
    stop_delay_s=1., hold_s=1., return_position_m=.001, return_angle_deg=2.,
    pillar_gap_m=.0005, release_pad_clearance_m=.001,
    fixture_shift_m=.001, fixture_angle_deg=2., maximum_s=120.)
WEIGHTS = dict(pickup=.15, full_immersion=.15, timer_operation=.15,
    immersion_time=.25, withdrawal=.15, unassisted_return=.15)


class DevelopmentTaskScorer:
    def __init__(self, metadata, sample_dt_s=.05):
        if abs(float(sample_dt_s)-.05)>1e-10:
            raise ValueError('The practical rubric requires complete 20 Hz observations.')
        self.metadata=metadata;self.dt=float(sample_dt_s);self.last=None;self.count=0
        self.initial=None;self.initial_tool=None;self.fixtures=None;self.beaker=None;self.timer_base=None
        self.tool_hand=None;self.tool_reference=None;self.pickup_since=None
        self.wafer_return_completed=False;self.tool_stand_since=None;self.tool_retained=True
        self.checks=dict.fromkeys(WEIGHTS,False);self.invalid=set();self.complete=True
        self.entered=None;self.exited=None;self.withdraw_since=None;self.return_window=deque()
        self.press_since=None;self.button_latched=False;self.button_events=[]
        self.clock_started=None;self.clock_stopped=None;self.last_measurements={}
        self.robot_solids=None;self.button_solid=None

    def reset(self,row):
        self.__init__(self.metadata,self.dt);self.update(row)

    def _fail(self,reason):
        self.invalid.add(reason)

    @staticmethod
    def _all(value):
        return bool(value) and (all(value.values()) if isinstance(value,dict) else value is True)

    def _support(self,row,wafer):
        if wafer[2,2]<=.5:return False
        geometry=pillar_plane_clearances(row['wafer_pose_world'],self.metadata)
        return (geometry['all_pillars_under_wafer']
            and max(abs(v)for v in geometry['clearances_m'])<=RUBRIC['pillar_gap_m'])

    def _tool_relation(self,row,hand):
        pads=[_pose(p,'handle paddle')for p in row['handle_paddle_poses_world']]
        if len(pads)!=2:raise ValueError('Two handle paddles are required.')
        frame=np.eye(4);frame[:3,3]=(pads[0][:3,3]+pads[1][:3,3])/2
        u,_,v=np.linalg.svd(pads[0][:3,:3]+pads[1][:3,:3]);frame[:3,:3]=u@np.diag([1.,1.,np.linalg.det(u@v)])@v
        return np.linalg.inv(_pose(row[hand],hand))@frame

    def _right_finger_at_button(self,row,button):
        from wafer_robot_geometry import RobotGeometryGuard
        from collision_geometry import prepare_mesh,transform_mesh,convex_screen
        if self.robot_solids is None:
            self.robot_solids=RobotGeometryGuard.from_evidence(self.metadata['robot_geometry'])
            n=16;angles=np.arange(n)*2*math.pi/n
            vertices=[[.01*math.cos(a),.01*math.sin(a),z]for z in(-.003,.003)for a in angles]
            faces=[(0,i+1,i)for i in range(1,n-1)]+[(n,n+i,n+i+1)for i in range(1,n-1)]
            for i in range(n):
                j=(i+1)%n;faces.extend([(i,j,n+j),(i,n+j,n+i)])
            self.button_solid=prepare_mesh(vertices,faces,name='timer_button')
        cap=transform_mesh(self.button_solid,button)
        poses=row['robot_link_poses_world']
        if len(poses)!=len(self.robot_solids.keys):raise ValueError('Incomplete robot link poses.')
        for key,pieces,pose in zip(self.robot_solids.keys,self.robot_solids.meshes,poses):
            if key[0]!='right'or key[1]not in('tip_left','tip_right'):continue
            transform=_pose(pose,'right fingertip')
            for piece in pieces:
                finger=transform_mesh(piece,transform)
                if convex_screen(finger,cap,margin=.0005).status in('overlap','touching','near'):return True
        return False

    def update(self,row):
        try:self._update(row)
        except (KeyError,TypeError,ValueError,IndexError)as exc:
            self.complete=False;self._fail('incomplete_evidence: '+str(exc))

    def _update(self,row):
        t=float(row['time_s'])
        if not math.isfinite(t)or (self.last is None and abs(t)>1e-8)or(self.last is not None and abs(t-self.last-self.dt)>1e-8):
            raise ValueError('Missing or reordered 20 Hz observations.')
        self.last=t;self.count+=1
        wafer=_pose(row['wafer_pose_world'],'wafer');tool=_pose(row['clamp_pose_world'],'pliers')
        beaker=_pose(row['beaker_pose_world'],'beaker');timer=_pose(row['timer_base_pose_world'],'timer')
        fixtures={k:_pose(v,k)for k,v in row['fixture_poses_world'].items()}
        if self.initial is None:
            self.initial=wafer.copy();self.initial_tool=tool.copy();self.fixtures=fixtures
            self.beaker=beaker.copy();self.timer_base=timer.copy()
        if t>RUBRIC['maximum_s']+1e-8:self._fail('time_limit')
        for field in ('objects_present','objects_intact','develop_objects_intact','robot_geometry_intact'):
            if not self._all(row[field]):self._fail('object_or_robot_modified: '+field)
        screen=row['robot_geometry_check']
        if screen.get('status')!='CLEAR' or screen.get('complete')is not True:self._fail('robot_collision_or_unresolved_geometry')
        pairs=[(beaker,self.beaker),(timer,self.timer_base)]
        if set(fixtures)!=set(self.fixtures):raise ValueError('Fixture inventory changed.')
        pairs.extend((fixtures[k],v)for k,v in self.fixtures.items())
        for actual,reference in pairs:
            if np.linalg.norm(actual[:3,3]-reference[:3,3])>RUBRIC['fixture_shift_m']or math.degrees(_angle(actual[:3,:3],reference[:3,:3]))>RUBRIC['fixture_angle_deg']:
                self._fail('fixture_moved')
        dry=float(row['hand_handle_liquid_clearance_m'])
        if not math.isfinite(dry):raise ValueError('Invalid dry-part clearance.')
        if dry<RUBRIC['dry_clearance_m']:self._fail('hand_or_handle_too_close_to_liquid')
        pad=pad_contact_geometry(row,self.metadata);held=pad['both_engaged'];support=self._support(row,wafer)
        tool_lift=tool[2,3]-self.initial_tool[2,3]
        if self.tool_reference is None and tool_lift>=RUBRIC['pliers_lift_m']:
            hands=('measured_hand_pose_world','right_hand_pose_world')
            relations={name:self._tool_relation(row,name)for name in hands}
            tcp_offset=np.array([0.,0.,.134])
            self.tool_hand=min(hands,key=lambda name:np.linalg.norm(relations[name][:3,3]-tcp_offset))
            self.tool_reference=relations[self.tool_hand]
            if np.linalg.norm(self.tool_reference[:3,3]-tcp_offset)>.02:self._fail('pliers_lifted_without_hand')
        if self.tool_reference is not None:
            relation=self._tool_relation(row,self.tool_hand)
            self.tool_retained=(np.linalg.norm(relation[:3,3]-self.tool_reference[:3,3])<=RUBRIC['tool_shift_m']
                and math.degrees(_angle(relation[:3,:3],self.tool_reference[:3,:3]))<=RUBRIC['tool_angle_deg'])
            on_stand=(self.wafer_return_completed and np.linalg.norm(tool[:3,3]-self.initial_tool[:3,3])<=.002
                and math.degrees(_angle(tool[:3,:3],self.initial_tool[:3,:3]))<=3.)
            self.tool_stand_since=(t if self.tool_stand_since is None else self.tool_stand_since)if on_stand else None
            if not self.tool_retained and not on_stand:
                self._fail('pliers_lost_in_hand')
        pickup=(self.tool_reference is not None and held and wafer[2,3]-self.initial[2,3]>=RUBRIC['wafer_lift_m'])
        self.pickup_since=(t if self.pickup_since is None else self.pickup_since)if pickup else None
        if self.pickup_since is not None and t-self.pickup_since+1e-8>=RUBRIC['pickup_hold_s']:self.checks['pickup']=True
        displaced=np.linalg.norm(wafer[:3,3]-self.initial[:3,3])>.002
        if displaced and not held and not support:self._fail('wafer_dropped_or_not_held_by_pliers')
        geometry=immersion_geometry(row['wafer_pose_world'],beaker[:3,3])
        full=geometry['fully_immersed']
        if full:
            if not held or not self.checks['pickup']:self._fail('immersion_without_valid_pickup')
            if geometry['deviation_from_vertical_deg']>RUBRIC['immersion_angle_deg']:self._fail('nonvertical_immersion')
            if self.exited is not None:self._fail('repeated_immersion')
            if self.entered is None:self.entered=t
            self.checks['full_immersion']=self.checks['pickup']and held
        elif self.entered is not None and self.exited is None:self.exited=t
        button=_pose(row['timer_button_pose_world'],'timer button')
        local=timer[:3,:3].T@(button[:3,3]-timer[:3,3])
        displacement=SPEC['timer_button_local_m'][2]-local[2]
        if not -.0005<=displacement<=.0035:self._fail('timer_button_outside_mechanism')
        if self.button_latched:
            if displacement<=SPEC['timer_release_m']:self.button_latched=False;self.press_since=None
        elif displacement>=SPEC['timer_press_m']:
            if self.press_since is None:self.press_since=t
            if t-self.press_since+1e-8>=SPEC['timer_debounce_s']:
                self.button_latched=True;self.button_events.append(t)
                if not self._right_finger_at_button(row,button):self._fail('timer_not_pressed_by_right_finger')
                if len(self.button_events)==1:
                    self.clock_started=t
                    if not full:self._fail('timer_started_before_full_immersion')
                elif len(self.button_events)==2:self.clock_stopped=t
                else:self._fail('timer_restarted')
        else:self.press_since=None
        observed_elapsed=float(row['timer_elapsed_s']);observed_remaining=float(row['timer_remaining_s'])
        end=t if self.clock_stopped is None else self.clock_stopped
        expected_elapsed=0. if self.clock_started is None else end-self.clock_started
        if not all(math.isfinite(v)for v in(observed_elapsed,observed_remaining)):
            raise ValueError('Nonfinite timer display.')
        if abs(observed_elapsed-expected_elapsed)>self.dt+1e-6 or abs(observed_remaining-max(0.,30.-observed_elapsed))>1e-6:
            self._fail('timer_display_does_not_follow_mechanical_presses')
        if self.exited is not None:
            dwell=self.exited-self.entered
            self.checks['immersion_time']=self.checks['full_immersion']and 28.-1e-8<=dwell<=32.+1e-8
            if self.clock_started is not None and self.clock_stopped is not None:
                self.checks['timer_operation']=(0.<=self.clock_started-self.entered<=RUBRIC['start_delay_s']+1e-8
                    and -1e-8<=self.exited-self.clock_started-30.<=RUBRIC['exit_after_countdown_s']+1e-8
                    and 0.<=self.clock_stopped-self.exited<=RUBRIC['stop_delay_s']+1e-8 and len(self.button_events)==2)
            clear=geometry['fully_withdrawn']and held
            self.withdraw_since=(t if self.withdraw_since is None else self.withdraw_since)if clear else None
            if self.withdraw_since is not None and t-self.withdraw_since+1e-8>=RUBRIC['hold_s']:self.checks['withdrawal']=True
        pos_error=float(np.linalg.norm(wafer[:3,3]-self.initial[:3,3]));angle_error=math.degrees(_angle(wafer[:3,:3],self.initial[:3,:3]))
        pads_clear=all(not p['overlaps_strip']or p['signed_gap_m']>=RUBRIC['release_pad_clearance_m']for p in pad['pads'])
        returned=(self.checks['withdrawal']and support and pads_clear
            and float(row['measured_jaw_gap_m'])>.002 and pos_error<=RUBRIC['return_position_m']and angle_error<=RUBRIC['return_angle_deg'])
        if returned:self.return_window.append((t,wafer.copy()))
        else:self.return_window.clear()
        while len(self.return_window)>1 and self.return_window[1][0]<=t-RUBRIC['hold_s']+1e-8:self.return_window.popleft()
        stable=bool(self.return_window)and t-self.return_window[0][0]+1e-8>=RUBRIC['hold_s']
        if stable:
            reference=self.return_window[0][1]
            stable=all(np.linalg.norm(p[:3,3]-reference[:3,3])<=.001 and math.degrees(_angle(p[:3,:3],reference[:3,:3]))<=2. for _,p in self.return_window)
        self.checks['unassisted_return']=stable
        self.wafer_return_completed|=stable
        self.last_measurements=dict(return_position_error_m=pos_error,return_orientation_error_deg=angle_error,
            fully_immersed=full,wafer_held_by_pad_geometry=held,pillar_support_geometry=support,pads_clear=pads_clear)

    def result(self):
        invalid=set(self.invalid)
        pending=not self.tool_retained and self.tool_stand_since is not None and self.last-self.tool_stand_since+1e-8<1.
        checks=dict(self.checks)
        if pending:checks['unassisted_return']=False
        valid=self.complete and not invalid and self.count>0
        score=sum(WEIGHTS[k]for k,v in checks.items()if v)if valid else 0.
        return dict(rubric=RUBRIC['version'],score=round(score,6),success=valid and all(checks.values()),
            checks=checks,tool_return_settling=pending,invalid_reasons=sorted(invalid),evidence_complete=self.complete and self.count>0,
            observed_samples=self.count,observation_dt_s=self.dt,elapsed_s=self.last,
            immersion_start_s=self.entered,immersion_end_s=self.exited,
            immersion_duration_s=None if self.exited is None else round(self.exited-self.entered,9),
            timer_button_events_s=list(self.button_events),measurements=self.last_measurements,
            scope='New submission rubric, sampled geometry and mechanical-button motion at 20 Hz. No force magnitudes, controller phase, OCR claim or chemical-quality measurement. Strict engineering audits remain separate.')
