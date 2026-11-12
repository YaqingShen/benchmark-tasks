"""Measured task progress; no scripted phase labels or drying-physics claim."""
from collections import deque
import copy
import math
import numpy as np
from wafer_grasp_protocol import _pose, _angle, pillar_plane_clearances, pad_contact_geometry
from wafer_cycle_protocol import coverage_points, SCAN


PROFILE=dict(version='measured_task_progress_v7',stage_weight=.2,hold_s=1.,
    milestone_evaluation_hz=20.,maximum_milestone_latency_s=.05,
    support_dropout_tolerance_s=.1,support_force_window_s=1.,intermediate_hold_requires_world_static_pose=False,
    release_requires_prior_measured_support=True,mixed_support_window_s=.05,
    mixed_support_basis='Signed normal force on the wafer; prior fixture contact, continuous support geometry, positive mean pad load and50ms mean signed load carrying at least80percent of weight. No completion credit.',
    tool_lift_m=.030,wafer_lift_m=.010,pillar_clearance_m=.005,gun_lift_m=.020,
    return_position_m=.001,gun_return_position_m=.002,return_angle_deg=2.,pose_envelope_m=.0005,pose_envelope_deg=.5,
    fixture_position_m=.0001,fixture_angle_deg=.05,maximum_unwanted_force_N=.01,
    minimum_grip_force_N=.005,minimum_support_fraction=.8,tool_retreat_m=.020,
    hand_retreat_m=.040,trigger_on_m=.0022,trigger_off_m=.001,
    scope='Geometry, contact support and observed air coverage; no evaporation, contamination or material-damage model.')
STAGES=('tool_pickup','wafer_lift','air_coverage','wafer_return','gun_return')
TASK_QUALIFICATION=dict(version='measured_full_task_v1',task_score_required=1.,
    legacy_paddle_torque_warning_Nm=.025,legacy_warning_effect='Diagnostic only; actual joint, speed and drive-cap bounds remain mandatory.')


class WaferTaskScorer:
    def __init__(self,metadata,sample_dt_s=None):
        self.metadata=copy.deepcopy(metadata)
        from wafer_finger_preload_v12 import workflow_timing
        timing=workflow_timing(metadata)
        self.maximum_seconds=float(timing['maximum_simulation_seconds'])if timing else None
        self.dt=float(sample_dt_s if sample_dt_s is not None else metadata.get('task_score_sample_dt_s',.05))
        if not math.isfinite(self.dt)or self.dt<=0:raise ValueError('Invalid task scoring cadence.')
        self.points=coverage_points();edges=np.linspace(self.points[:,1].min()-1e-9,self.points[:,1].max()+1e-9,19)
        self.bands=np.clip(np.searchsorted(edges,self.points[:,1],side='right')-1,0,17)
        self.reset()

    def reset(self,initial_row=None,*,report=True):
        self.invalid=[];self.stage=0;self.last_time=None;self.initialized=False
        self.mixed_support_window=deque();self.mixed_support_samples=0
        self.exposure=np.zeros(len(self.points));self.windows={k:deque()for k in STAGES}
        self.completed=[];self.gun_picked=False;self.gun_pick_window=deque()
        self.wafer_relation=None;self.gun_hand_relation=None;self.initial=None
        self.initial_supported={'clamp':False,'gun':False};self.mechanism_diagnostics=[]
        self.unsupported_since={name:None for name in ('clamp','wafer','gun')}
        self.lift_observed={name:False for name in ('clamp','wafer','gun')}
        self.release_support_seen={name:False for name in ('wafer','gun')}
        self.release_pending={name:None for name in ('wafer','gun')}
        self.support_samples={name:deque()for name in ('wafer','gun')}
        self.support_sums={'wafer':np.zeros(3),'gun':np.zeros(1)};self.support_duration={'wafer':0.,'gun':0.}
        if initial_row is not None:
            try:self._initialize(initial_row)
            except (KeyError,ValueError,TypeError,IndexError,RuntimeError)as exc:self._invalidate(str(exc))
        return self.result() if report else None

    def _invalidate(self,reason):
        if reason not in self.invalid:self.invalid.append(reason)

    @staticmethod
    def _number(row,key):
        value=row[key]
        if isinstance(value,(bool,np.bool_))or not isinstance(value,(int,float,np.number))or not math.isfinite(float(value)):
            raise ValueError('Invalid numeric evidence: '+key)
        return float(value)

    @staticmethod
    def _vector(row,key,length,nonnegative=False):
        value=np.asarray(row[key])
        if value.shape!=(length,)or value.dtype.kind not in 'fiu'or not np.isfinite(value).all():raise ValueError('Invalid vector evidence: '+key)
        if nonnegative and np.any(value<0):raise ValueError('Negative contact evidence: '+key)
        return value.astype(float)

    def _initialize(self,row):
        force_scope=self.metadata['wafer_support_force_measurement']
        if any(force_scope.get(k)!=v for k,v in dict(version='signed_wafer_normal_support_v1',frame='world',force_on='wafer').items()):raise ValueError('Missing signed wafer support-force provenance.')
        ref=self.metadata['task_score_reference'];names=ref['object_names']
        if len(names)!=len(set(names))or not {'wafer','clamp','gun'}.issubset(names):raise ValueError('Invalid required object inventory.')
        fixtures=ref['fixture_poses_world']
        if not fixtures or not set(fixtures).issubset(names):raise ValueError('Missing approved fixture inventory.')
        stands=self.metadata['tool_pickup']['stand_geoms']
        required={'heater','pillar_0','pillar_1','pillar_2','gun_stand',*stands}
        if len(stands)!=2 or len(set(stands))!=2 or not required.issubset(fixtures):raise ValueError('Missing task support fixtures.')
        self.fixture_reference={k:_pose(v,k)for k,v in fixtures.items()}
        support=next(s for s in self.metadata['tool_pickup']['stand_shapes']if s['name']=='tool_stand_hinge')
        if support['type']!='cylinder'or not 0<float(support['size_m'][0])<.1:raise ValueError('Invalid initial pliers support footprint.')
        self.tool_support_radius_m=float(support['size_m'][0]);self.tool_support_center_xy_m=self.fixture_reference[support['name']][:2,3].copy()
        self.object_reference={k:_pose(ref['initial_object_poses_world'][k],k)for k in ('wafer','clamp','gun')}
        self.object_names=set(names);self._validate(row)
        for name,key in [('wafer','wafer_pose_world'),('clamp','clamp_pose_world'),('gun','gun_pose_world')]:
            pose=_pose(row[key],name);expected=self.object_reference[name]
            if np.linalg.norm(pose[:3,3]-expected[:3,3])>.001 or math.degrees(_angle(pose[:3,:3],expected[:3,:3]))>2:
                raise ValueError('Initial object pose differs from the approved scene: '+name)
        if max(self._vector(row,'finger_handle_contact_normal_force_N',2,True))>.005 or max(self._vector(row,'gun_grip_normal_force_N',2,True))>.005:
            raise ValueError('The task must start with both hands empty.')
        self.initial=copy.deepcopy(row);self.last_time=self._number(row,'time_s');self.initialized=True

    def _validate(self,row):
        self._number(row,'time_s')
        for key in ('wafer_pose_world','clamp_pose_world','gun_pose_world','measured_hand_pose_world','right_hand_pose_world'):_pose(row[key],key)
        for key in ('objects_present','objects_intact'):
            values=row[key]
            if not isinstance(values,dict)or any(values.get(name)is not True for name in self.object_names):
                raise ValueError('Missing or changed required object: '+key)
        poses=row['fixture_poses_world']
        for name,reference in self.fixture_reference.items():
            actual=_pose(poses[name],name)
            if np.linalg.norm(actual[:3,3]-reference[:3,3])>PROFILE['fixture_position_m']or math.degrees(_angle(actual[:3,:3],reference[:3,:3]))>PROFILE['fixture_angle_deg']:
                raise ValueError('A fixed fixture was moved: '+name)
        for key,length in [('finger_handle_contact_normal_force_N',2),('jaw_contact_normal_force_N',2),('pillar_contact_normal_force_N',3),('gun_grip_normal_force_N',2)]:
            values=self._vector(row,key,length,True)
            if key=='jaw_contact_normal_force_N'and values.max()>15:raise ValueError('Wafer contact load exceeds the declared bound.')
            if key=='finger_handle_contact_normal_force_N'and values.max()>20:raise ValueError('Finger/tool jam load exceeds the declared bound.')
        for key,scalar,count in [('wafer_jaw_normal_force_vectors_world_N','jaw_contact_normal_force_N',2),('wafer_pillar_normal_force_vectors_world_N','pillar_contact_normal_force_N',3)]:
            vectors=np.asarray(row[key],dtype=float)
            if vectors.shape!=(count,3)or not np.isfinite(vectors).all():raise ValueError('Invalid signed support-force vectors: '+key)
            if np.any(np.linalg.norm(vectors,axis=1)>np.asarray(row[scalar])+1e-6):raise ValueError('Signed force exceeds measured partner normal magnitude: '+key)
        for key in ('forbidden_wafer_contact_force_N','unexpected_robot_tool_contact_normal_force_N','unexpected_robot_environment_contact_normal_force_N','tool_unexpected_environment_contact_normal_force_N'):
            force=self._number(row,key)
            if force<0 or force>PROFILE['maximum_unwanted_force_N']:raise ValueError('Unexpected contact: '+key)
        contacts=row['unexpected_gun_right_contacts']
        if not isinstance(contacts,list)or any(len(c)!=3 or not math.isfinite(c[2])or c[2]<0 or c[2]>.01 for c in contacts):
            raise ValueError('Unexpected right-hand contact.')
        for key in ('tool_stand_normal_force_N','gun_stand_vertical_force_N','gun_foot_gap_m','gun_support_margin_m','trigger_compression_m'):
            self._number(row,key)
        if self._number(row,'gun_stand_normal_force_N')<0:raise ValueError('Invalid gun stand normal force.')
        if self.metadata.get('handle_paddles'):
            from wafer_tool_pickup import guard_handle_paddles,guard_handle_paddle_capacity
            q=self._vector(row,'handle_paddle_q_rad',2);v=self._vector(row,'handle_paddle_qvel_rad_s',2)
            torque=self._vector(row,'handle_paddle_passive_torque_Nm',2)
            expected=[-s['stiffness_Nm_rad']*(p-s['rest_rad'])-s['damping_Nms_rad']*w for s,p,w in zip(self.metadata['handle_paddles'],q,v)]
            if not np.allclose(torque,expected,atol=1e-7,rtol=0):raise ValueError('Inconsistent passive paddle evidence.')
            guard_handle_paddle_capacity(row,self.metadata)
            try:guard_handle_paddles(row,self.metadata)
            except RuntimeError as exc:
                if not self.mechanism_diagnostics:self.mechanism_diagnostics.append(str(exc))
        wafer=_pose(row['wafer_pose_world'],'wafer')
        if wafer[2,3]<self.object_reference['wafer'][2,3]-.005:raise ValueError('The wafer fell below its support level.')
        _pose(row['jaw_link_poses_world'][0],'lower jaw');_pose(row['jaw_link_poses_world'][1],'upper jaw')
        axis=self._vector(row,'nozzle_axis_world',3);self._vector(row,'nozzle_tip_world_m',3)
        if abs(np.linalg.norm(axis)-1)>1e-5:raise ValueError('Invalid measured nozzle axis.')

    def _held(self,row):
        wafer=_pose(row['wafer_pose_world'],'wafer');tool=_pose(row['clamp_pose_world'],'clamp')
        clearance=pillar_plane_clearances(row['wafer_pose_world'],self.metadata)
        alignment=math.degrees(math.acos(float(np.clip(abs(wafer[:3,1]@tool[:3,1]),-1,1))))
        return (wafer[2,3]-self.object_reference['wafer'][2,3]>=PROFILE['wafer_lift_m']
            and min(clearance['clearances_m'])>=PROFILE['pillar_clearance_m']
            and min(row['jaw_contact_normal_force_N'])>=.005
            and min(row['finger_handle_contact_normal_force_N'])>=.005
            and max(row['pillar_contact_normal_force_N'])<=.005
            and alignment<=2 and pad_contact_geometry(row,self.metadata)['both_engaged'])

    def _gun_held(self,row):
        gun=_pose(row['gun_pose_world'],'gun')
        return gun[2,3]-self.object_reference['gun'][2,3]>=PROFILE['gun_lift_m']and min(row['gun_grip_normal_force_N'])>=.005 and row['gun_stand_normal_force_N']<=.005

    def _pliers_held(self,row):
        return min(row['finger_handle_contact_normal_force_N'])>=.005 and row['tool_stand_normal_force_N']<=.005

    def _window(self,window,row,condition,pose_keys):
        t=float(row['time_s'])
        if not condition:window.clear();return False
        item={'time_s':t,'poses':np.asarray([row[key]for key in pose_keys],dtype=float).copy()}
        window.append(item)
        while len(window)>1 and window[1]['time_s']<=t-PROFILE['hold_s']+1e-8:window.popleft()
        if t-window[0]['time_s']<PROFILE['hold_s']-1e-8:return False
        period=1/PROFILE['milestone_evaluation_hz']
        previous=getattr(self,'_previous_update_time',t-self.dt)
        if math.floor((t+1e-8)/period)==math.floor((previous+1e-8)/period):return False
        if not pose_keys:return True
        poses=np.stack([x['poses']for x in window]);positions=poses[:,:,:3];quats=poses[:,:,3:]
        if np.max(np.linalg.norm(positions-positions[0],axis=2))>PROFILE['pose_envelope_m']:return False
        quats=quats/np.linalg.norm(quats,axis=2,keepdims=True)
        angles=2*np.arccos(np.clip(abs(np.einsum('nbq,bq->nb',quats,quats[0])),-1.,1.))
        if float(np.max(angles))*180/math.pi>PROFILE['pose_envelope_deg']:return False
        return True

    def _wafer_support_geometry(self,row):
        wafer=_pose(row['wafer_pose_world'],'wafer');base=self.object_reference['wafer'];gaps=pillar_plane_clearances(row['wafer_pose_world'],self.metadata)
        return (math.degrees(_angle(wafer[:3,:3],base[:3,:3]))<=2
            and gaps['all_pillars_under_wafer']and max(abs(x)for x in gaps['clearances_m'])<=.0005)

    def _wafer_geometry(self,row):
        return(self._wafer_support_geometry(row)and max(row['jaw_contact_normal_force_N'])<=.005
            and np.linalg.norm(_pose(row['wafer_pose_world'],'wafer')[:3,3]-self.object_reference['wafer'][:3,3])<=.001)

    def _wafer_rest(self,row):
        weight=float(self.metadata['wafer_mass_kg'])*abs(float(self.metadata['gravity_world_m_s2'][2]))
        return self._wafer_geometry(row)and min(row['pillar_contact_normal_force_N'])>=.005 and sum(row['pillar_contact_normal_force_N'])>=.8*weight

    def _gun_support_geometry(self,row):
        gun=_pose(row['gun_pose_world'],'gun');base=self.object_reference['gun']
        return (math.degrees(_angle(gun[:3,:3],base[:3,:3]))<=2
            and abs(row['gun_foot_gap_m'])<=.0005 and row['gun_support_margin_m']>=.002
            and row['gun_stand_normal_force_N']<=3.)

    def _gun_geometry(self,row):
        return(self._gun_support_geometry(row)and max(row['gun_grip_normal_force_N'])<=.005
            and row['trigger_compression_m']<PROFILE['trigger_off_m']
            and np.linalg.norm(_pose(row['gun_pose_world'],'gun')[:3,3]-self.object_reference['gun'][:3,3])<=(PROFILE['gun_return_position_m']if self.gun_picked else PROFILE['return_position_m']))

    def _gun_supported(self,row):
        weight=(float(self.metadata['gun']['mass_kg'])+float(self.metadata['gun']['trigger_mass_kg']))*abs(float(self.metadata['gravity_world_m_s2'][2]))
        position=np.linalg.norm(_pose(row['gun_pose_world'],'gun')[:3,3]-self.object_reference['gun'][:3,3])
        return (self._gun_support_geometry(row)and position<=PROFILE['gun_return_position_m']
            and .8*weight<=row['gun_stand_vertical_force_N']<=3.)

    def _gun_rest(self,row):
        weight=(float(self.metadata['gun']['mass_kg'])+float(self.metadata['gun']['trigger_mass_kg']))*abs(float(self.metadata['gravity_world_m_s2'][2]))
        return self._gun_geometry(row)and .8*weight<=row['gun_stand_vertical_force_N']<=3.

    def _support_update(self,row,dt):
        for name,force in [('wafer',np.asarray(row['pillar_contact_normal_force_N'])),('gun',np.array([row['gun_stand_vertical_force_N']]))]:
            weighted=force*dt;window=self.support_samples[name];window.append((row['time_s'],dt,weighted))
            self.support_sums[name]+=weighted;self.support_duration[name]+=dt
            while window and window[0][0]<=row['time_s']-PROFILE['hold_s']+1e-8:
                _,span,value=window.popleft();self.support_sums[name]-=value;self.support_duration[name]-=span

    def _support_ready(self,name):
        duration=self.support_duration[name]
        if duration<PROFILE['hold_s']-1e-8:return False
        force=self.support_sums[name]/duration;gravity=abs(float(self.metadata['gravity_world_m_s2'][2]))
        if name=='wafer':return bool(force.min()>=.005 and force.sum()>=.8*float(self.metadata['wafer_mass_kg'])*gravity)
        weight=(float(self.metadata['gun']['mass_kg'])+float(self.metadata['gun']['trigger_mass_kg']))*gravity
        return bool(.8*weight<=force[0]<=3.)

    def _coverage(self,row,dt):
        if not(self.gun_picked and self._held(row)and self._gun_held(row))or row['trigger_compression_m']<PROFILE['trigger_on_m']:return
        wafer=_pose(row['wafer_pose_world'],'wafer');normal=wafer[:3,2]
        axis=np.asarray(row['nozzle_axis_world']);tip=np.asarray(row['nozzle_tip_world_m']);incidence=-float(axis@normal)
        if not math.cos(math.radians(60))<=incidence<=math.cos(math.radians(30)):return
        distance=float((tip-wafer[:3,3]-normal*float(self.metadata['wafer_thickness_m'])/2)@normal/incidence)
        if not .050<=distance<=.110:return
        hit=(tip+axis*distance-wafer[:3,3])@wafer[:3,:3]
        self.exposure[np.sum((self.points-hit[:2])**2,axis=1)<=SCAN['footprint_radius_m']**2]+=dt

    def _check_release_order(self,name,row,held):
        geometry=self._wafer_support_geometry(row)if name=='wafer'else self._gun_support_geometry(row)
        bearing=(sum(row['pillar_contact_normal_force_N'])if name=='wafer'else row['gun_stand_vertical_force_N'])>=.005
        if held:
            self.release_pending[name]=None
            if not geometry:self.release_support_seen[name]=False
            elif bearing:self.release_support_seen[name]=True
        else:
            if self.release_pending[name]is None:
                self.release_pending[name]=bool(self.release_support_seen[name]and geometry)
            elif not geometry:self.release_pending[name]=False
            if geometry and bearing and not self.release_pending[name]:
                raise ValueError('A lifted object was released before measured support was established: '+name)

    def _wafer_mixed_support(self,row):
        t=float(row['time_s']);g=np.asarray(self.metadata['gravity_world_m_s2'],dtype=float)
        up=-g/np.linalg.norm(g);weight=float(self.metadata['wafer_mass_kg'])*np.linalg.norm(g)
        jaws=np.asarray(row['wafer_jaw_normal_force_vectors_world_N']);pillars=np.asarray(row['wafer_pillar_normal_force_vectors_world_N'])
        wafer=_pose(row['wafer_pose_world'],'wafer')
        condition=(self.release_support_seen['wafer']and not self._wafer_rest(row)and self._wafer_support_geometry(row)
            and np.linalg.norm(wafer[:3,3]-self.object_reference['wafer'][:3,3])<=PROFILE['return_position_m']
            and float(pillars.sum(axis=0)@up)>=.005)
        if condition:
            geometry=pad_contact_geometry(row,self.metadata)
            condition=any(face['within_tolerance']for face in geometry['pads'])
        if not condition:self.mixed_support_window.clear();return False
        window=self.mixed_support_window;window.append((t,wafer,float((jaws.sum(axis=0)+pillars.sum(axis=0))@up),float(jaws.sum(axis=0)@up)))
        span=PROFILE['mixed_support_window_s']
        while len(window)>1 and window[1][0]<=t-span+1e-8:window.popleft()
        if t-window[0][0]<span-1e-8:return False
        first=window[0][1]
        durations=np.diff(np.maximum([v[0]for v in window],t-span))
        mean_total=float(np.dot(durations,[v[2]for v in list(window)[1:]])/span)
        mean_pad=float(np.dot(durations,[v[3]for v in list(window)[1:]])/span)
        supported=(mean_total>=PROFILE['minimum_support_fraction']*weight and mean_pad>=.005
            and max(np.linalg.norm(v[1][:3,3]-first[:3,3])for v in window)<=PROFILE['pose_envelope_m']
            and max(math.degrees(_angle(v[1][:3,:3],first[:3,:3]))for v in window)<=PROFILE['pose_envelope_deg'])
        if supported:self.mixed_support_samples+=1
        return supported

    def _check_drops(self,row):
        contact_keys={'clamp':'finger_handle_contact_normal_force_N','wafer':'jaw_contact_normal_force_N','gun':'gun_grip_normal_force_N'}
        pose_keys={'clamp':'clamp_pose_world','wafer':'wafer_pose_world','gun':'gun_pose_world'}
        mixed_wafer_support=self._wafer_mixed_support(row)
        supported={'clamp':row['tool_stand_normal_force_N']>.01 and np.linalg.norm(_pose(row['clamp_pose_world'],'clamp')[:3,3]-self.object_reference['clamp'][:3,3])<=.001,
            'wafer':self._wafer_rest(row)or mixed_wafer_support,'gun':self._gun_supported(row)}
        for name in pose_keys:
            rise=_pose(row[pose_keys[name]],name)[2,3]-self.object_reference[name][2,3]
            held=min(row[contact_keys[name]])>=.005
            if rise>.005 and held:self.lift_observed[name]=True
            if name in self.release_pending and self.lift_observed[name]:
                self._check_release_order(name,row,held)
            if not self.lift_observed[name]or held or supported[name]:self.unsupported_since[name]=None
            elif self.unsupported_since[name]is None:self.unsupported_since[name]=row['time_s']
            elif row['time_s']-self.unsupported_since[name]>PROFILE['support_dropout_tolerance_s']+1e-9:raise ValueError('A lifted object lost both its grip and its intended support: '+name)

    def update(self,row,*,report=True):
        if self.invalid:return self.result() if report else None
        if not self.initialized:return self.reset(row,report=report)
        try:
            self._validate(row);t=self._number(row,'time_s');dt=t-self.last_time
            if dt< -1e-9 or dt>self.dt+max(1e-8,self.dt*1e-5):raise ValueError('Task observations are missing or out of order.')
            if dt<=1e-9:return self.result() if report else None
            self._previous_update_time=self.last_time;self.last_time=t
            if self.maximum_seconds is not None and t>self.maximum_seconds+1e-8 and self.stage<5:
                raise ValueError('Task deadline exceeded: full workflow was not completed within 120 simulation seconds.')
            self._support_update(row,dt)
            self._check_drops(row)
            if self.stage<2 and not self._gun_geometry(row):raise ValueError('The gun was disturbed before the wafer lift was established.')
            if self.stage==0 and not self._wafer_geometry(row):raise ValueError('The wafer was disturbed before the pliers were picked up.')
            if self.gun_picked and self.stage<4 and row['gun_stand_normal_force_N']>.01:raise ValueError('The gun was returned before the wafer was returned.')
            if row['tool_stand_normal_force_N']>.01:
                tool=_pose(row['clamp_pose_world'],'clamp')
                if self.stage>=1 or np.linalg.norm(tool[:2,3]-self.tool_support_center_xy_m)>self.tool_support_radius_m or max(row['jaw_contact_normal_force_N'])>.005:
                    raise ValueError('The pliers contacted their stand outside the initial pickup.')
            for name,key,force in [('clamp','clamp_pose_world','tool_stand_normal_force_N'),('gun','gun_pose_world','gun_stand_vertical_force_N')]:
                if np.linalg.norm(_pose(row[key],name)[:3,3]-self.object_reference[name][:3,3])<=.001 and row[force]>.01:
                    self.initial_supported[name]=True
            if self.stage>=4 and not self._wafer_geometry(row):raise ValueError('The returned wafer moved off its supports or was regrasped.')
            if self.stage==5:
                if not self._gun_geometry(row):raise ValueError('The returned gun moved off its support or air was re-enabled.')
                if not self._pliers_held(row):raise ValueError('Final pliers retention was lost or the tool contacted its stand.')
                return self.result() if report else None
            stage=STAGES[self.stage];complete=False
            if self.stage==0:
                tool=_pose(row['clamp_pose_world'],'clamp')
                condition=all(self.initial_supported.values())and tool[2,3]-self.object_reference['clamp'][2,3]>=PROFILE['tool_lift_m']and row['tool_stand_normal_force_N']<=.005 and min(row['finger_handle_contact_normal_force_N'])>=.005
                complete=self._window(self.windows[stage],row,condition,[])
            elif self.stage==1:
                complete=self._window(self.windows[stage],row,self._held(row),[])
                if complete:self.wafer_relation=np.linalg.inv(_pose(row['wafer_pose_world'],'wafer'))@_pose(row['clamp_pose_world'],'clamp')
            elif self.stage==2:
                if not self.gun_picked and self._window(self.gun_pick_window,row,self._gun_held(row),[]):
                    self.gun_picked=True;self.gun_hand_relation=np.linalg.inv(_pose(row['gun_pose_world'],'gun'))@_pose(row['right_hand_pose_world'],'right hand')
                self._coverage(row,dt);complete=bool(np.all(self.exposure>=SCAN['minimum_exposure_s']))
            elif self.stage==3:
                relative=np.linalg.inv(_pose(row['wafer_pose_world'],'wafer'))@_pose(row['clamp_pose_world'],'clamp')
                retreat=float(np.linalg.norm(relative[:3,3]-self.wafer_relation[:3,3]))
                complete=self._window(self.windows[stage],row,self._wafer_geometry(row)and retreat>=PROFILE['tool_retreat_m'],['wafer_pose_world'])and self._support_ready('wafer')
            elif self.stage==4:
                relative=np.linalg.inv(_pose(row['gun_pose_world'],'gun'))@_pose(row['right_hand_pose_world'],'right hand')
                retreat=float(np.linalg.norm(relative[:3,3]-self.gun_hand_relation[:3,3]))
                complete=self._window(self.windows[stage],row,self._gun_geometry(row)and self._pliers_held(row)and retreat>=PROFILE['hand_retreat_m'],['wafer_pose_world','gun_pose_world'])and self._support_ready('gun')and self._support_ready('wafer')
            if complete:self.completed.append(dict(stage=stage,time_s=t));self.stage+=1
        except (KeyError,ValueError,TypeError,IndexError,RuntimeError)as exc:self._invalidate(str(exc))
        return self.result() if report else None

    def result(self):
        covered=self.exposure>=SCAN['minimum_exposure_s'];fraction=float(covered.mean())
        bands=[i for i in range(18)if np.any(self.bands==i)and bool(np.all(covered[self.bands==i]))]
        partial=fraction if self.stage==2 and self.gun_picked else 0.
        progress=round(min(1.,.2*(self.stage+partial)),12)
        return dict(score=0. if self.invalid else progress,success=not self.invalid and self.stage==5,
            stage='complete'if self.stage==5 else STAGES[self.stage],stage_progress=partial,
            maximum_simulation_seconds=self.maximum_seconds,
            completion_time_s=self.completed[-1]['time_s']if self.stage==5 else None,
            invalid_reasons=list(self.invalid),diagnostics=dict(completed=copy.deepcopy(self.completed),
                diagnostic_progress=progress,mixed_support_samples=self.mixed_support_samples,coverage_fraction=fraction,covered_row_bands=bands,
                eligible_points=len(self.points),minimum_exposure_s=float(self.exposure.min()),
                initial_support_observed=dict(self.initial_supported),mechanism_diagnostics=list(self.mechanism_diagnostics),
                geometry_integrity_scope='Authored geometry and topology only; no material fracture model.',
                excluded_surface='1 mm outer rim and 15 mm clamp-shadow strip.',profile=copy.deepcopy(PROFILE)))
