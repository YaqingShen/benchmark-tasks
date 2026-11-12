"""Visible-marker handling rubric; 20 Hz geometry, not a cleanliness model."""
from collections import deque
import copy
import math
import numpy as np
from wafer_grasp_protocol import _pose, _angle, pad_contact_geometry, pillar_plane_clearances

MIXED_VERSION='visible_debris_practical_v6'
RUBRIC = dict(version='visible_debris_practical_v5', sample_hz=20., maximum_s=120.,
    tool_lift_m=.030, wafer_lift_m=.010, pin_clearance_m=.005, gun_lift_m=.020,
    pickup_hold_s=.2, hold_s=1., tool_shift_m=.002, tool_angle_deg=3.,
    gun_shift_m=.0015, gun_angle_deg=2., hand_grip_radius_m=.020,
    support_dropout_s=.1, return_position_m=.001, gun_return_position_m=.002,
    contact_band_tolerance_m=.0001,
    return_angle_deg=2., support_gap_m=.0005, support_margin_m=.002,
    pose_envelope_m=.0005, pose_envelope_deg=.5, release_pad_clearance_m=.001,
    tool_retreat_m=.020, hand_retreat_m=.040, trigger_on_m=.0022,
    trigger_off_m=.001, fixture_shift_m=.0001, fixture_angle_deg=.05,
    nozzle_min_m=.050, nozzle_max_m=.110, incidence_min_deg=30., incidence_max_deg=60.,
    horizontal_angle_deg=10., cleaning_speed_m_s=.020, cleaning_angle_speed_deg_s=5.,
    marker_departure_m=.001, marker_surface_tolerance_m=.002,
    marker_clear_margin_m=.001, marker_count=12)
WEIGHTS = dict(wafer_lift=.20, gun_operation=.10, marker_removal=.40,
    wafer_return=.15, gun_return=.15)
HANDS = ('measured_hand_pose_world', 'right_hand_pose_world')
TCP_OFFSET = np.array([0., 0., .134])


def _finite(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)) or not math.isfinite(value):
        raise ValueError('Invalid numeric evidence: '+name)
    return float(value)


def _array(value, shape, name):
    a=np.asarray(value)
    if a.shape!=shape or a.dtype.kind not in 'fiu' or not np.isfinite(a).all():
        raise ValueError('Invalid geometry: '+name)
    return a.astype(float)


def _plain(value):
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {k:_plain(v)for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [_plain(v)for v in value]
    return value


def _convex_xy(vertices):
    points=sorted(set(map(tuple,np.asarray(vertices,dtype=float)[:,:2])))
    def cross(a,b,c):return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
    if len(points)<=2:return np.asarray(points)
    lower=[];upper=[]
    for chain,sequence in ((lower,points),(upper,reversed(points))):
        for point in sequence:
            while len(chain)>1 and cross(chain[-2],chain[-1],point)<=0.:chain.pop()
            chain.append(point)
    return np.asarray(lower[:-1]+upper[:-1])


def _particle_footprints_overlap(first,second):
    a,b=_convex_xy(first),_convex_xy(second)
    for polygon in (a,b):
        for start,end in zip(polygon,np.roll(polygon,-1,axis=0)):
            edge=end-start;normal=np.array([-edge[1],edge[0]])
            length=np.linalg.norm(normal)
            if length<1e-15:continue
            normal/=length;pa=a@normal;pb=b@normal
            if pa.max()<=pb.min()+1e-9 or pb.max()<=pa.min()+1e-9:return False
    return True


def _footprint_overlap(vertices,radius,flat,margin=0.):
    """Conservative whole-hull overlap with the disk/flat footprint and margin."""
    hull=_convex_xy(vertices);poly=[]
    for a,b in zip(hull,np.roll(hull,-1,axis=0)):
        da=a[0]-flat+margin;db=b[0]-flat+margin
        if da>=0.:poly.append(a)
        if (da>=0.)!=(db>=0.):poly.append(a+(b-a)*da/(da-db))
    if not poly:return False
    xy=np.asarray(poly);boundary=radius+margin
    if np.any(np.linalg.norm(xy,axis=1)<=boundary):return True
    cross=[]
    for a,b in zip(xy,np.roll(xy,-1,axis=0)):
        edge=b-a;length=float(edge@edge)
        nearest=a if length<1e-24 else a+np.clip(-float(a@edge)/length,0.,1.)*edge
        if np.linalg.norm(nearest)<=boundary:return True
        cross.append(float(edge[0]*(-a[1])-edge[1]*(-a[0])))
    area=abs(float(np.sum(xy[:,0]*np.roll(xy[:,1],-1)-xy[:,1]*np.roll(xy[:,0],-1))))/2
    return bool(area>1e-16 and(min(cross)>=0. or max(cross)<=0.))


def _mixed_shapes(dust):
    if dust.get('shape_schema')!='mixed_convex_debris_v1':raise ValueError('Unknown debris shape schema.')
    parts=dust['particles'];ids=dust['ids'];binding=dust['geometry_evidence']
    if len(parts)!=12 or [p['id']for p in parts]!=list(ids):raise ValueError('Mixed particle inventory differs.')
    if binding.get('status')!='COMPOSED_COLLIDERS_VERIFIED' or set(binding['particles'])!=set(ids):
        raise ValueError('Missing actual mixed-particle collider evidence.')
    vertices=[];initial=[]
    for item in parts:
        points=np.asarray(item['vertices_local_m'])
        if points.ndim!=2 or points.shape[1]!=3 or not 4<=len(points)<=64:
            raise ValueError('Invalid mixed-particle vertices.')
        points=_array(points,points.shape,'particle local vertices')
        if np.linalg.matrix_rank(points-points.mean(axis=0),tol=1e-12)!=3:
            raise ValueError('A particle needs a three-dimensional convex hull.')
        measured=_array(binding['particles'][item['id']]['vertices_local_m'],points.shape,'composed particle vertices')
        if not np.allclose(points,measured,rtol=0.,atol=1e-8):raise ValueError('Declared particle collider changed.')
        q=_array(item['initial_rotation_xyzw'],(4,),'particle initial rotation')
        initial.append(_pose(np.r_[np.zeros(3),q],'particle initial rotation')[:3,:3])
        vertices.append(points)
    return vertices,initial


def _particle_geometry(vertices,pose,wafer,radius,flat,top):
    local=(vertices@pose[:3,:3].T+pose[:3,3]-wafer[:3,3])@wafer[:3,:3]
    near=local[:,2].min()<=top+RUBRIC['marker_surface_tolerance_m'] and local[:,2].max()>=top-RUBRIC['marker_surface_tolerance_m']
    on=bool(near and _footprint_overlap(local,radius,flat))
    clear=not _footprint_overlap(local,radius,flat,RUBRIC['marker_clear_margin_m'])
    return local,on,clear


def _outside_band_contact(vertices,x_min,surface_z,radius,tolerance_m=.00025):
    """Intersect the forbidden face area with the wafer's contact-plane tolerance."""
    poly=list(vertices)
    for axis,bound,sign in ((0,x_min,1),(2,surface_z-tolerance_m,1),(2,surface_z+tolerance_m,-1)):
        clipped=[]
        for a,b in zip(poly,poly[1:]+poly[:1]):
            da=(a[axis]-bound)*sign;db=(b[axis]-bound)*sign
            if da>=0.:clipped.append(a)
            if (da>=0.)!=(db>=0.):clipped.append(a+(b-a)*da/(da-db))
        poly=clipped
        if len(poly)<3:return False
    xy=np.asarray(poly)[:,:2]
    area=abs(float(np.sum(xy[:,0]*np.roll(xy[:,1],-1)-xy[:,1]*np.roll(xy[:,0],-1))))/2
    if area<=1e-16:return False
    cross=[]
    for a,b in zip(xy,np.roll(xy,-1,axis=0)):
        edge=b-a;length=float(edge@edge)
        nearest=a if length<1e-24 else a+np.clip(-float(a@edge)/length,0.,1.)*edge
        if np.linalg.norm(nearest)<radius:return True
        cross.append(float(edge[0]*(-a[1])-edge[1]*(-a[0])))
    return bool(min(cross)>=0. or max(cross)<=0.)


def _grip_geometry(row, metadata):
    depths=metadata.get('pad_contact_band_depths_m')
    if depths is None:return pad_contact_geometry(row,metadata)
    depths=_array(depths,(2,),'lower/upper contact band depths')
    if not np.allclose(depths,[.005,.003],rtol=0.,atol=1e-12):
        raise ValueError('The edge-pick draft requires lower 5 mm and upper 3 mm bands.')
    pads=[pad_contact_geometry(row,{**metadata,'pad_intrusion_m':float(depth)})['pads'][i]
          for i,depth in enumerate(depths)]
    wafer=_pose(row['wafer_pose_world'],'wafer pose')
    radius=float(metadata['wafer_radius_m'])
    flat=-math.sqrt(radius**2-(float(metadata['wafer_flat_length_m'])/2)**2)
    bottom=float(metadata['wafer_local_bottom_z_m']);top=bottom+float(metadata['wafer_thickness_m'])
    faces=_array(metadata['contact_faces_local_m'],(2,4,3),'pad contact faces')
    links=_array(row['jaw_link_poses_world'],(2,7),'jaw link poses')
    for i,pad in enumerate(pads):
        link=_pose(links[i],'jaw link pose')
        vertices=(faces[i]@link[:3,:3].T+link[:3,3]-wafer[:3,3])@wafer[:3,:3]
        pad['maximum_face_inset_m']=float(vertices[:,0].max()-flat)
        limit=depths[i]+RUBRIC['contact_band_tolerance_m']
        pad['within_band']=pad['maximum_face_inset_m']<=limit+1e-9
        pad['outside_band_contact']=_outside_band_contact(vertices,flat+limit+1e-9,
            bottom if i==0 else top,radius)
        pad['within_tolerance']=bool(pad['within_tolerance']and pad['within_band'])
    return dict(pads=pads,both_engaged=all(p['within_tolerance']for p in pads))


def _wafer_release_geometry(row,metadata):
    """Check actual pad faces against the whole wafer, including its interior."""
    wafer=_pose(row['wafer_pose_world'],'wafer pose')
    faces=_array(metadata['contact_faces_local_m'],(2,4,3),'pad contact faces')
    links=_array(row['jaw_link_poses_world'],(2,7),'jaw link poses')
    radius=_finite(metadata['wafer_radius_m'],'wafer radius')
    flat=-math.sqrt(radius**2-(float(metadata['wafer_flat_length_m'])/2)**2)
    bottom=float(metadata['wafer_local_bottom_z_m']);top=bottom+float(metadata['wafer_thickness_m'])
    records=[]
    for i in range(2):
        link=_pose(links[i],'jaw link pose')
        vertices=(faces[i]@link[:3,:3].T+link[:3,3]-wafer[:3,3])@wafer[:3,:3]
        poly=[]
        for a,b in zip(vertices,np.roll(vertices,-1,axis=0)):
            da=a[0]-flat;db=b[0]-flat
            if da>=0.:poly.append(a)
            if (da>=0.)!=(db>=0.):poly.append(a+(b-a)*da/(da-db))
        overlaps=False
        if poly:
            points=np.asarray(poly);xy=points[:,:2]
            overlaps=bool(np.any(np.linalg.norm(xy,axis=1)<=radius))
            cross=[]
            for a,b in zip(xy,np.roll(xy,-1,axis=0)):
                edge=b-a;length=float(edge@edge)
                nearest=a if length<1e-24 else a+np.clip(-float(a@edge)/length,0.,1.)*edge
                overlaps|=bool(np.linalg.norm(nearest)<=radius)
                cross.append(float(edge[0]*(-a[1])-edge[1]*(-a[0])))
            # A face can surround the disk without any vertex or edge entering it.
            area=abs(float(np.sum(xy[:,0]*np.roll(xy[:,1],-1)-xy[:,1]*np.roll(xy[:,0],-1))))/2
            if area>1e-16 and (min(cross)>=-1e-12 or max(cross)<=1e-12):overlaps=True
            gap=float(bottom-points[:,2].max())if i==0 else float(points[:,2].min()-top)
        else:gap=None
        records.append(dict(overlaps_wafer=overlaps,signed_gap_m=gap))
    clear=all(not p['overlaps_wafer']or p['signed_gap_m']>=RUBRIC['release_pad_clearance_m']for p in records)
    return dict(pads=records,clear=clear,withdrawn=all(not p['overlaps_wafer']for p in records))


def _shared_support(row,metadata,pads,gaps):
    """Pillars and a held lower pad may jointly support the wafer during handoff."""
    if gaps is None or not gaps['all_pillars_under_wafer'] or not pads['pads'][0]['within_tolerance']:
        return False
    centers=_array(metadata['pillar_top_centers_world_m'],(3,3),'pillar tops')
    active=np.abs(gaps['clearances_m'])<=RUBRIC['support_gap_m']
    if not np.any(active):return False
    wafer=_pose(row['wafer_pose_world'],'wafer pose');link=_pose(row['jaw_link_poses_world'][0],'lower jaw')
    face=_array(metadata['contact_faces_local_m'],(2,4,3),'pad faces')[0]
    world=face@link[:3,:3].T+link[:3,3];local=(world-wafer[:3,3])@wafer[:3,:3]
    radius=float(metadata['wafer_radius_m']);half=float(metadata['wafer_flat_length_m'])/2
    flat=-math.sqrt(radius**2-half**2)
    valid=(np.abs(local[:,2]-float(metadata['wafer_local_bottom_z_m']))<=RUBRIC['support_gap_m'])&(local[:,0]>=flat)&(np.linalg.norm(local[:,:2],axis=1)<=radius)
    if not np.any(valid):return False
    points=sorted(set(map(tuple,np.vstack([centers[active,:2],world[valid,:2]]))))
    def cross(a,b,c):return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
    lower=[];upper=[]
    for collection,sequence in ((lower,points),(upper,list(reversed(points)))):
        for point in sequence:
            while len(collection)>=2 and cross(collection[-2],collection[-1],point)<=0:collection.pop()
            collection.append(point)
    hull=np.asarray(lower[:-1]+upper[:-1])
    if len(hull)<3:return False
    a=-flat;segment=radius**2*math.acos(a/radius)-a*half
    centroid_x=(2/3)*half**3/(math.pi*radius**2-segment)
    center=(wafer[:3,3]+wafer[:3,:3]@np.array([centroid_x,0.,0.]))[:2]
    margins=[cross(p,q,center)/np.linalg.norm(q-p)for p,q in zip(hull,np.roll(hull,-1,axis=0))]
    return bool(min(margins)>=RUBRIC['support_margin_m'])


class VisibleDebrisScorer:
    def __init__(self, metadata, sample_dt_s=.05):
        if abs(float(sample_dt_s)-.05)>1e-10:
            raise ValueError('The practical rubric requires complete 20 Hz observations.')
        self.metadata=copy.deepcopy(metadata);self.dt=float(sample_dt_s)
        self.mixed='shape_schema'in self.metadata.get('dust',{})
        self.version=MIXED_VERSION if self.mixed else RUBRIC['version']
        self.reset(report=False)

    def reset(self, initial_row=None, report=True):
        self.invalid=[];self.complete=True;self.count=0;self.last=None;self.previous=None
        self.initial=None;self.reference=None;self.fixtures=None;self.names=None
        self.checks=dict.fromkeys(WEIGHTS, False);self.checks['marker_removal']=0.
        self.tool_hand={};self.hand_reference={};self.retained={'clamp':False,'gun':False}
        self.picked={'clamp':False,'gun':False};self.pick_since={'clamp':None,'gun':None,'wafer':None}
        self.unsupported={name:None for name in ('clamp','gun','wafer')}
        self.windows={name:deque()for name in ('wafer','gun','clamp')}
        self.wafer_grip_relation=None;self.wafer_returned=False;self.wafer_return_started=False;self.tool_return_pending=False
        self.first_air=None;self.previous_air_valid=False
        self.launched=np.zeros(12,dtype=bool);self.moved=np.zeros(12,dtype=bool);self.clear_since=[None]*12
        self.cleared=np.zeros(12,dtype=bool);self.last_measurements={}
        self.marker_vertices_reference=None
        if initial_row is not None:self.update(initial_row,report=False)
        return self.result() if report else None

    def _invalidate(self, reason):
        if reason not in self.invalid:self.invalid.append(reason)

    def _pose(self, row, name):
        return _pose(row[name], name)

    def _initialize(self,row):
        ref=self.metadata['task_score_reference'];names=ref['object_names']
        ids=self.metadata['dust']['ids'];profile=self.metadata['dust']['profile']
        if len(ids)!=12 or len(set(ids))!=12 or int(profile['count'])!=12:
            raise ValueError('Exactly twelve named test markers are required.')
        if self.mixed:self.marker_vertices,self.marker_initial_rotations=_mixed_shapes(self.metadata['dust'])
        elif abs(float(profile['side_m'])-.001)>1e-10:
            raise ValueError('Legacy evidence requires 1 mm cube markers.')
        if len(names)!=len(set(names))or not {'wafer','clamp','gun'}.issubset(names):
            raise ValueError('Incomplete required object inventory.')
        if set(names)&set(ids) not in (set(),set(ids)):
            raise ValueError('Partial marker inventory in the task-body map.')
        self.names=set(names);self.ids=tuple(ids)
        self.anchors=_array(self.metadata['dust']['initial_centers_wafer_m'],(12,3),'initial marker centers')
        self.marker_reference=self.anchors.copy()
        calibration=self.metadata.get('hand_grasp_points_local_m',{hand:TCP_OFFSET for hand in HANDS})
        if set(calibration)!=set(HANDS):raise ValueError('Both measured hand grasp points are required.')
        self.hand_grasp_points={hand:_array(calibration[hand],(3,),'hand grasp point')for hand in HANDS}
        self.reference={name:_pose(ref['initial_object_poses_world'][name],name)for name in ('wafer','clamp','gun')}
        self.fixtures={k:_pose(v,k)for k,v in ref['fixture_poses_world'].items()if k not in self.ids}
        required={'heater','pillar_0','pillar_1','pillar_2','gun_stand',*self.metadata['tool_pickup']['stand_geoms']}
        if not required.issubset(self.fixtures):raise ValueError('Missing required support fixtures.')
        self.initial={k:self._pose(row,k+'_pose_world')for k in ('wafer','clamp','gun')}
        for name,actual in self.initial.items():
            expected=self.reference[name]
            if np.linalg.norm(actual[:3,3]-expected[:3,3])>.001 or math.degrees(_angle(actual[:3,:3],expected[:3,:3]))>2.:
                raise ValueError('Initial pose differs from the declared scene: '+name)
        self.radius=float(self.metadata['wafer_radius_m']);self.flat=-math.sqrt(self.radius**2-(float(self.metadata['wafer_flat_length_m'])/2)**2)
        self.top=float(self.metadata['wafer_local_bottom_z_m'])+float(self.metadata['wafer_thickness_m'])
        self.marker_radius=None if self.mixed else math.sqrt(3)*float(profile['side_m'])/2
        self.cone=float(profile['cone_half_angle_deg']);self.nozzle_radius=float(profile['nozzle_radius_m'])
        if not 0.<self.cone<45. or not 0.<self.nozzle_radius<.01:raise ValueError('Invalid declared air cone geometry.')
        compression=_finite(row['trigger_compression_m'],'trigger')
        if compression>=RUBRIC['trigger_off_m']:raise ValueError('The gun must start with its trigger released.')

    def _validate(self,row):
        for key in ('objects_present','objects_intact'):
            values=row[key]
            if not isinstance(values,dict)or any(values.get(name)is not True for name in self.names):
                self._invalidate('missing_or_modified_object: '+key)
        intact=row['robot_geometry_intact']
        if not (intact is True or isinstance(intact,dict)and bool(intact)and all(v is True for v in intact.values())):
            self._invalidate('robot_geometry_modified')
        screen=row['robot_geometry_check']
        if screen.get('status')=='COLLISION':self._invalidate('robot_collision')
        elif screen.get('status')!='CLEAR' or screen.get('complete')is not True:
            raise ValueError('Unresolved or missing robot geometry evidence.')
        if row['dust_intact']is not True:self._invalidate('missing_or_modified_marker')
        actual={k:_pose(v,k)for k,v in row['fixture_poses_world'].items()if k not in self.ids}
        if set(actual)!=set(self.fixtures):raise ValueError('Fixture inventory changed.')
        for name,reference in self.fixtures.items():
            if np.linalg.norm(actual[name][:3,3]-reference[:3,3])>RUBRIC['fixture_shift_m'] or math.degrees(_angle(actual[name][:3,:3],reference[:3,:3]))>RUBRIC['fixture_angle_deg']:
                self._invalidate('fixture_moved: '+name)

    def _tool_relation(self,row,name,hand):
        if name=='clamp':
            pads=[_pose(p,'handle paddle')for p in row['handle_paddle_poses_world']]
            if len(pads)!=2:raise ValueError('Two handle paddles are required.')
            frame=np.eye(4);frame[:3,3]=(pads[0][:3,3]+pads[1][:3,3])/2
            u,_,v=np.linalg.svd(pads[0][:3,:3]+pads[1][:3,:3]);frame[:3,:3]=u@np.diag([1.,1.,np.linalg.det(u@v)])@v
        else:
            frame=self._pose(row,'gun_pose_world').copy()
            point=_array(self.metadata['gun']['grip_local_m'],(3,),'gun grip center').copy()
            point[0]-=_finite(row['trigger_compression_m'],'trigger')/2
            frame[:3,3]+=frame[:3,:3]@point
        return np.linalg.inv(self._pose(row,hand))@frame

    def _on_tool_stand(self,row,name,pose):
        reference=self.initial[name]
        close=(np.linalg.norm(pose[:3,3]-reference[:3,3])<=.002
            and math.degrees(_angle(pose[:3,:3],reference[:3,:3]))<=2.)
        if name=='gun':
            return (close and abs(_finite(row['gun_foot_gap_m'],'gun foot gap'))<=RUBRIC['support_gap_m']
                and _finite(row['gun_support_margin_m'],'gun support margin')>=RUBRIC['support_margin_m'])
        return close

    def _track_tool(self,row,name,pose,t):
        relations={hand:self._tool_relation(row,name,hand)for hand in HANDS}
        distances={hand:float(np.linalg.norm(rel[:3,3]-self.hand_grasp_points[hand]))for hand,rel in relations.items()}
        rise=pose[2,3]-self.initial[name][2,3]
        on_stand=self._on_tool_stand(row,name,pose)
        if name not in self.hand_reference and not on_stand:
            hand=min(HANDS,key=distances.get)
            if distances[hand]<=RUBRIC['hand_grip_radius_m']:
                self.tool_hand[name]=hand;self.hand_reference[name]=relations[hand].copy()
        retained=False
        if name in self.hand_reference:
            hand=self.tool_hand[name];relation=relations[hand];reference=self.hand_reference[name]
            shift=RUBRIC['tool_shift_m']if name=='clamp'else RUBRIC['gun_shift_m']
            angle=RUBRIC['tool_angle_deg']if name=='clamp'else RUBRIC['gun_angle_deg']
            retained=(distances[hand]<=RUBRIC['hand_grip_radius_m']
                and np.linalg.norm(relation[:3,3]-reference[:3,3])<=shift
                and math.degrees(_angle(relation[:3,:3],reference[:3,:3]))<=angle)
        self.retained[name]=bool(retained)
        threshold=RUBRIC['tool_lift_m']if name=='clamp'else RUBRIC['gun_lift_m']
        condition=retained and rise>=threshold
        self.pick_since[name]=(t if self.pick_since[name]is None else self.pick_since[name])if condition else None
        if self.pick_since[name]is not None and t-self.pick_since[name]+1e-8>=RUBRIC['pickup_hold_s']:self.picked[name]=True
        if name=='clamp'and self.picked[name]and on_stand and not retained and not self.wafer_returned:
            self._invalidate('pliers_released_before_wafer_return')
        self._drop(name,t,retained or on_stand)
        return on_stand,distances

    def _drop(self,name,t,supported):
        if supported:self.unsupported[name]=None
        elif self.unsupported[name]is None:self.unsupported[name]=t
        elif t-self.unsupported[name]+1e-8>=RUBRIC['support_dropout_s']:
            self._invalidate('dropped_or_unretained: '+name)

    def _stable(self,name,t,condition,pose):
        window=self.windows[name]
        if not condition:window.clear();return False
        window.append((t,pose.copy()))
        while len(window)>1 and window[1][0]<=t-RUBRIC['hold_s']+1e-8:window.popleft()
        if t-window[0][0]+1e-8<RUBRIC['hold_s']:return False
        reference=window[0][1]
        return all(np.linalg.norm(p[:3,3]-reference[:3,3])<=RUBRIC['pose_envelope_m']
            and math.degrees(_angle(p[:3,:3],reference[:3,:3]))<=RUBRIC['pose_envelope_deg']for _,p in window)

    def _air_geometry(self,row,wafer,gun):
        tip=_array(row['nozzle_tip_world_m'],(3,),'nozzle tip');axis=_array(row['nozzle_axis_world'],(3,),'nozzle axis')
        expected=gun[:3,3]+gun[:3,:3]@_array(self.metadata['gun']['nozzle_local_m'],(3,),'nozzle local')
        if abs(np.linalg.norm(axis)-1)>1e-5 or np.linalg.norm(tip-expected)>1e-5 or np.linalg.norm(axis-gun[:3,0])>1e-5:
            raise ValueError('Nozzle geometry differs from measured gun pose.')
        incidence=-float(axis@wafer[:3,2])
        good=math.cos(math.radians(RUBRIC['incidence_max_deg']))<=incidence<=math.cos(math.radians(RUBRIC['incidence_min_deg']))
        distance=float((tip-wafer[:3,3]-wafer[:3,2]*self.top)@wafer[:3,2]/incidence)if incidence>1e-8 else -1.
        hit=tip+axis*distance;local=wafer[:3,:3].T@(hit-wafer[:3,3])
        good=good and RUBRIC['nozzle_min_m']<=distance<=RUBRIC['nozzle_max_m'] and np.linalg.norm(local[:2])<=self.radius and local[0]>=self.flat
        return bool(good),tip,axis,distance

    def _markers(self,row,t,wafer,air_valid,removable):
        positions=_array(row['dust_position_world_m'],(12,3),'named marker positions')
        poses=None
        if 'dust_pose_world'in row:
            poses=_array(row['dust_pose_world'],(12,7),'marker poses')
            if not np.allclose(poses[:,:3],positions,atol=1e-6,rtol=0):raise ValueError('Marker pose/position disagreement.')
        local=(positions-wafer[:3,3])@wafer[:3,:3]
        local_vertices=None
        if self.mixed:
            if poses is None:raise ValueError('Mixed particles require measured orientation.')
            geometry=[_particle_geometry(vertices,_pose(pose,'particle pose'),wafer,self.radius,self.flat,self.top)
                for vertices,pose in zip(self.marker_vertices,poses)]
            local_vertices=[item[0]for item in geometry]
            on=np.array([item[1]for item in geometry]);clear=np.array([item[2]for item in geometry])
        else:
            on=(np.linalg.norm(local[:,:2],axis=1)<=self.radius+self.marker_radius)&(local[:,0]>=self.flat-self.marker_radius)&(np.abs(local[:,2]-(self.top+.0005))<=RUBRIC['marker_surface_tolerance_m'])
            clear=(np.linalg.norm(local[:,:2],axis=1)>self.radius+self.marker_radius+RUBRIC['marker_clear_margin_m'])|(local[:,0]<self.flat-self.marker_radius-RUBRIC['marker_clear_margin_m'])
        if self.count==1:
            if not on.all() or np.max(np.linalg.norm(local-self.anchors,axis=1))>.001:
                raise ValueError('Markers do not match the named initial layout.')
            if self.mixed:
                for vertices,rotation,anchor,actual in zip(self.marker_vertices,self.marker_initial_rotations,self.anchors,local_vertices):
                    expected=vertices@rotation.T+anchor
                    if (np.any(np.linalg.norm(expected[:,:2],axis=1)>self.radius+1e-8)
                            or expected[:,0].min()<self.flat-1e-8
                            or abs(expected[:,2].min()-self.top)>.00001):
                        raise ValueError('Declared particle footprint is not fully on the wafer.')
                    if (np.any(np.linalg.norm(actual[:,:2],axis=1)>self.radius+1e-8)
                            or actual[:,0].min()<self.flat-1e-8
                            or np.max(np.linalg.norm(actual-expected,axis=1))>.001):
                        raise ValueError('Actual particle footprint differs from its declared layout.')
                for i,points in enumerate(local_vertices):
                    if any(_particle_footprints_overlap(points,other)for other in local_vertices[:i]):
                        raise ValueError('Initial particle footprints overlap.')
                self.marker_vertices_reference=[v.copy()for v in local_vertices]
        for i in range(12):
            if on[i]and self.launched[i]:
                self.launched[i]=False;self.moved[i]=False;self.marker_reference[i]=local[i]
                if self.mixed:self.marker_vertices_reference[i]=local_vertices[i].copy()
            displacement=(np.max(np.linalg.norm(local_vertices[i]-self.marker_vertices_reference[i],axis=1))
                if self.mixed else np.linalg.norm(local[i]-self.marker_reference[i]))
            departed=not on[i] or displacement>RUBRIC['marker_departure_m']
            # Valid blowing can also dislodge markers indirectly through other debris.
            authorized=bool(removable and not self.wafer_return_started
                and (air_valid or self.previous_air_valid))
            if departed and not self.moved[i]:
                if authorized:self.moved[i]=True
                else:self._invalidate('marker_departed_without_valid_blowing: '+self.ids[i])
            if not on[i]and not self.launched[i]:
                exit_authorized=bool(self.moved[i]and removable and not self.wafer_return_started
                    and (air_valid or self.previous_air_valid))
                if exit_authorized:self.launched[i]=True
                else:self._invalidate('marker_exited_surface_without_valid_blowing: '+self.ids[i])
            if clear[i]and self.launched[i]:
                if self.clear_since[i]is None:self.clear_since[i]=t
            else:self.clear_since[i]=None
            self.cleared[i]=self.clear_since[i]is not None and t-self.clear_since[i]+1e-8>=RUBRIC['hold_s']
        self.previous_air_valid=bool(air_valid)
        self.checks['marker_removal']=float(np.mean(self.cleared))

    def update(self,row,report=True):
        if self.invalid:return self.result()if report else None
        try:self._update(row)
        except (KeyError,ValueError,TypeError,IndexError,RuntimeError,np.linalg.LinAlgError)as exc:
            self.complete=False;self._invalidate('incomplete_evidence: '+str(exc))
        return self.result()if report else None

    def _update(self,row):
        t=_finite(row['time_s'],'time')
        if self.last is None:
            if abs(t)>1e-8:raise ValueError('The initial observation must be at time zero.')
            self._initialize(row)
        elif abs(t-self.last-self.dt)>1e-8:raise ValueError('Missing or reordered 20 Hz observations.')
        self.count+=1;self.last=t
        if t>RUBRIC['maximum_s']+1e-8:raise ValueError('Observations exceed the 120 second budget.')
        self._validate(row)
        wafer=self._pose(row,'wafer_pose_world');tool=self._pose(row,'clamp_pose_world');gun=self._pose(row,'gun_pose_world')
        tool_stand,_=self._track_tool(row,'clamp',tool,t);gun_stand,gun_distances=self._track_tool(row,'gun',gun,t)
        if all(self.retained.values())and self.tool_hand['clamp']==self.tool_hand['gun']:
            self._invalidate('one_hand_cannot_retain_both_tools')
        pads=_grip_geometry(row,self.metadata);held=bool(pads['both_engaged']and self.retained['clamp'])
        for i,pad in enumerate(pads['pads']):
            if pad.get('outside_band_contact',False):self._invalidate('pad_contact_outside_band: '+('lower','upper')[i])
        tilt=math.degrees(math.acos(float(np.clip(wafer[2,2],-1.,1.))))
        support=False;gaps=None
        if wafer[2,2]>.5:
            gaps=pillar_plane_clearances(row['wafer_pose_world'],self.metadata)
            support=gaps['all_pillars_under_wafer']and max(abs(x)for x in gaps['clearances_m'])<=RUBRIC['support_gap_m']
        if self.checks['wafer_lift']and support:self.wafer_return_started=True
        shared_support=bool(self.retained['clamp']and _shared_support(row,self.metadata,pads,gaps))
        self._drop('wafer',t,held or support or shared_support)
        if wafer[2,3]<self.initial['wafer'][2,3]-.005:self._invalidate('wafer_below_support_level')
        pickup=(held and self.picked['clamp']and wafer[2,3]-self.initial['wafer'][2,3]>=RUBRIC['wafer_lift_m']
            and gaps is not None and min(gaps['clearances_m'])>=RUBRIC['pin_clearance_m'])
        self.pick_since['wafer']=(t if self.pick_since['wafer']is None else self.pick_since['wafer'])if pickup else None
        if self.pick_since['wafer']is not None and t-self.pick_since['wafer']+1e-8>=RUBRIC['pickup_hold_s']:
            self.checks['wafer_lift']=True
            if self.wafer_grip_relation is None:self.wafer_grip_relation=np.linalg.inv(wafer)@tool
        speed=0. if self.previous is None else float(np.linalg.norm(wafer[:3,3]-self.previous[:3,3])/self.dt)
        angular=0. if self.previous is None else math.degrees(_angle(wafer[:3,:3],self.previous[:3,:3]))/self.dt
        compression=_finite(row['trigger_compression_m'],'trigger')
        if not -.0005<=compression<=float(self.metadata['gun']['trigger_travel_m'])+.0005:self._invalidate('trigger_outside_mechanism')
        active=compression>=RUBRIC['trigger_on_m']
        if active and self.first_air is None:self.first_air=t
        aim,tip,axis,distance=self._air_geometry(row,wafer,gun)
        stable_for_air=tilt<=RUBRIC['horizontal_angle_deg']and speed<=RUBRIC['cleaning_speed_m_s']and angular<=RUBRIC['cleaning_angle_speed_deg_s']
        air_valid=bool(active and aim and held and self.checks['wafer_lift']and self.picked['gun']and self.retained['gun']and stable_for_air)
        if air_valid:self.checks['gun_operation']=True
        self._markers(row,t,wafer,air_valid,held and not support and stable_for_air)
        position=float(np.linalg.norm(wafer[:3,3]-self.initial['wafer'][:3,3]));angle=math.degrees(_angle(wafer[:3,:3],self.initial['wafer'][:3,:3]))
        release=_wafer_release_geometry(row,self.metadata);pads_clear=release['clear']
        jaw_gap=_finite(row['measured_jaw_gap_m'],'jaw gap')
        release_ready=pads_clear and (release['withdrawn']or jaw_gap>.002)
        retreat=0. if self.wafer_grip_relation is None else float(np.linalg.norm((np.linalg.inv(wafer)@tool)[:3,3]-self.wafer_grip_relation[:3,3]))
        returned=(self.checks['wafer_lift']and support and release_ready
            and position<=RUBRIC['return_position_m']and angle<=RUBRIC['return_angle_deg']and retreat>=RUBRIC['tool_retreat_m'])
        self.checks['wafer_return']=self._stable('wafer',t,returned,wafer)
        self.wafer_returned|=self.checks['wafer_return']
        gun_retreat=0.
        if 'gun'in self.hand_reference:
            gun_retreat=float(np.linalg.norm(self._tool_relation(row,'gun',self.tool_hand['gun'])[:3,3]-self.hand_reference['gun'][:3,3]))
        gun_released=(self.picked['gun']and gun_stand and compression<RUBRIC['trigger_off_m']and gun_retreat>=RUBRIC['hand_retreat_m']
            and min(gun_distances.values())>RUBRIC['hand_grip_radius_m'])
        self.checks['gun_return']=self._stable('gun',t,gun_released,gun)
        pliers_home=self.wafer_returned and tool_stand and not self.retained['clamp']
        tool_stable=self._stable('clamp',t,pliers_home,tool)
        self.tool_return_pending=bool(pliers_home and not tool_stable)
        self.previous=wafer.copy()
        self.last_measurements=dict(wafer_held=held,wafer_tilt_deg=tilt,wafer_speed_m_s=speed,
            wafer_angular_speed_deg_s=angular,aimed_trigger_operation=air_valid,nozzle_surface_distance_m=distance,
            pliers_retained=self.retained['clamp'],gun_retained=self.retained['gun'],
            wafer_return_position_error_m=position,wafer_return_angle_deg=angle,
            pads_clear=pads_clear,pads_withdrawn=release['withdrawn'],pillar_support_geometry=support,
            shared_pad_pillar_support=shared_support,gun_stand_geometry=gun_stand)

    def result(self):
        valid=not self.invalid and self.complete and self.count>0
        checks=dict(self.checks)
        if self.tool_return_pending:checks['wafer_return']=False
        score=sum(WEIGHTS[k]*float(v)for k,v in checks.items())if valid else 0.
        success=bool(valid and all(checks.values())and checks['marker_removal']==1.)
        return _plain(dict(rubric=self.version,score=round(score,8),success=success,
            checks=checks,removed_markers=int(self.cleared.sum()),total_markers=12,
            cleared_marker_ids=[name for name,clear in zip(getattr(self,'ids',()),self.cleared)if clear],
            invalid_reasons=list(self.invalid),evidence_complete=self.complete and self.count>0,
            observed_samples=self.count,observation_dt_s=self.dt,elapsed_s=self.last,
            completion_time_s=self.last if success else None,tool_return_settling=self.tool_return_pending,
            measurements=dict(self.last_measurements),
            scope='Visible specified-marker removal and handling at 20 Hz. Geometry does not prove contact force or sub-frame causality. No nanoscale cleanliness, drying, force, jet-impulse or coverage-grid score.'))
