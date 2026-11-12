"""Independent evaluation of recorded contact-based wafer-grasp simulations.

This module reads metadata and full physics-step measurements only. It never
steps an engine, changes a state, or attaches an object. A fixture result does
not validate the YAM arm transport, brittle-wafer stresses, or a drying process.
All poses are world XYZ in meters followed by quaternion XYZW.
"""
import copy
import math
import numpy as np


DEFAULT_LIMITS = dict(
    settle_window_s=.5, contact_window_s=.25, lift_window_s=1., return_window_s=1.,
    settle_linear_speed_m_s=.001, settle_angular_speed_rad_s=.02,
    lift_linear_speed_m_s=.005, lift_angular_speed_rad_s=.05,
    minimum_lift_m=.010, minimum_pillar_clearance_m=.005,
    maximum_xy_drift_m=.005, maximum_wafer_tilt_deg=5.,
    maximum_relative_translation_m=.001, maximum_relative_rotation_deg=2.,
    maximum_flat_alignment_deg=2., minimum_jaw_contact_N=.005,
    maximum_jaw_contact_N=15., minimum_pillar_contact_N=.005,
    maximum_unwanted_contact_N=.01, support_gap_tolerance_m=.0005,
    minimum_measured_gap_reduction_m=.001, minimum_closed_gap_m=.00075,
    maximum_closed_gap_m=.0015, return_position_m=.001, return_rotation_deg=2.,
    minimum_retracted_clamp_rise_m=.020,pad_geometry_gap_tolerance_m=.00025)


def _array(value,shape,name):
    a=np.asarray(value)
    if a.shape!=shape or a.dtype.kind not in 'fiu' or not np.isfinite(a).all():
        raise ValueError('Invalid '+name)
    return a.astype(float,copy=True)


def _number(value,name):
    if isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,float,np.number)) or not math.isfinite(float(value)):
        raise ValueError('Invalid '+name)
    return float(value)


def _pose(value,name):
    p=_array(value,(7,),name);n=np.linalg.norm(p[3:])
    if abs(n-1)>.001:raise ValueError('Non-unit quaternion: '+name)
    x,y,z,w=p[3:]/n;t=np.eye(4);t[:3,3]=p[:3]
    t[:3,:3]=[[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
              [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
              [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]]
    return t


def _angle(a,b):
    r=a.T@b
    # atan2 remains useful near zero, unlike trace-only arccos.
    sine=np.linalg.norm([r[2,1]-r[1,2],r[0,2]-r[2,0],r[1,0]-r[0,1]])/2
    return float(math.atan2(sine,float(np.clip((np.trace(r)-1)/2,-1,1))))


def _relative(reference,target):
    t=np.eye(4);t[:3,:3]=reference[:3,:3].T@target[:3,:3]
    t[:3,3]=reference[:3,:3].T@(target[:3,3]-reference[:3,3]);return t


def _stats(values):
    if not values:return dict(count=0,minimum=None,maximum=None)
    return dict(count=len(values),minimum=float(min(values)),maximum=float(max(values)))


def _window(rows,phase,seconds,dt):
    n=round(seconds/dt)
    if abs(n*dt-seconds)>1e-8:raise ValueError('Window is not an integer number of physics intervals.')
    ids=[i for i,row in enumerate(rows) if row['phase']==phase]
    if not ids:return None
    end=ids[-1];start=end
    while start>0 and rows[start-1]['phase']==phase:start-=1
    boundary=end-n
    if boundary<0 or end-start+1<n:return None
    selected=rows[boundary:end+1]
    if any(row['phase']!=phase for row in selected[1:]):return None
    return selected


def _rates(rows,dt):
    poses=[row['_wafer'] for row in rows]
    linear=[float(np.linalg.norm(b[:3,3]-a[:3,3])/dt) for a,b in zip(poses,poses[1:])]
    angular=[_angle(a[:3,:3],b[:3,:3])/dt for a,b in zip(poses,poses[1:])]
    if any(not math.isfinite(x) for x in linear+angular):raise ValueError('Nonfinite pose rate.')
    return dict(linear_speed_m_s=_stats(linear),angular_speed_rad_s=_stats(angular))


def pillar_plane_clearances(wafer_pose,metadata):
    """Conservative lower-face clearance above all finite pillar-top disks.

    The full pillar disk must project under the wafer. The plane minimum over
    each disk is exact; the wafer-footprint disk-inclusion bound is conservative
    for tilt. This is geometric clearance, not contact or load-transfer proof.
    """
    t=_pose(wafer_pose,'wafer pose');normal=t[:3,2]
    if normal[2]<=.5:raise ValueError('Wafer is too tilted for the pillar-plane protocol.')
    centers=_array(metadata['pillar_top_centers_world_m'],(3,3),'pillar tops')
    radius=float(metadata['pillar_radius_m']);wafer_radius=float(metadata['wafer_radius_m'])
    flat=-math.sqrt(wafer_radius**2-(float(metadata['wafer_flat_length_m'])/2)**2)
    bottom=t[:3,3]+normal*float(metadata['wafer_local_bottom_z_m'])
    gaps=[];covered=[]
    for center in centers:
        center_gap=float(normal@(bottom-center)/normal[2])
        gap=center_gap-radius*float(np.linalg.norm(normal[:2]))/float(normal[2])
        projection=center+np.array([0.,0.,center_gap])
        local=t[:3,:3].T@(projection-t[:3,3]);bound=radius/float(normal[2])
        inside=local[0]>=flat+bound and np.linalg.norm(local[:2])<=wafer_radius-bound
        gaps.append(gap);covered.append(bool(inside))
    return dict(clearances_m=gaps,all_pillars_under_wafer=all(covered),pillar_footprint_covered=covered)


def pad_contact_geometry(sample,metadata,tolerance_m=.00025):
    """Corroborate intended pad faces against a finite wafer-edge strip.

    Link-filtered force cannot identify the force-bearing collider. This check
    confirms only that both authored pad faces geometrically engage the wafer;
    it does not attribute every link force to a pad or prove full-area contact.
    """
    wafer=_pose(sample['wafer_pose_world'],'wafer pose')
    faces=_array(metadata['contact_faces_local_m'],(2,4,3),'pad contact-face corners')
    links=_array(sample['jaw_link_poses_world'],(2,7),'jaw link poses')
    radius=float(metadata['wafer_radius_m']);flat=-math.sqrt(radius**2-(float(metadata['wafer_flat_length_m'])/2)**2)
    intrusion=_number(metadata['pad_intrusion_m'],'pad intrusion')
    if not 0<intrusion<=.010+1e-12:raise ValueError('Unsupported pad engagement strip; maximum authored depth is 10 mm.')
    half=float(metadata['wafer_flat_length_m'])/2
    bottom=float(metadata['wafer_local_bottom_z_m']);top=bottom+float(metadata['wafer_thickness_m'])
    def clip(poly,axis,bound,greater):
        out=[]
        for a,b in zip(poly,poly[1:]+poly[:1]):
            da=(a[axis]-bound)*(1 if greater else -1);db=(b[axis]-bound)*(1 if greater else -1)
            if da>=0:out.append(a)
            if (da>=0)!=(db>=0):out.append(a+(b-a)*(da/(da-db)))
        return out
    records=[]
    for index in range(2):
        link=_pose(links[index],'jaw link pose')
        vertices=(faces[index]@link[:3,:3].T+link[:3,3]-wafer[:3,3])@wafer[:3,:3]
        poly=list(vertices)
        for axis,bound,greater in ((0,flat,True),(0,flat+intrusion,False),(1,-half,True),(1,half,False)):
            poly=clip(poly,axis,bound,greater)
        if len(poly)<3:
            records.append(dict(overlaps_strip=False,signed_gap_m=None,within_tolerance=False));continue
        points=np.asarray(poly);area=abs(np.sum(points[:,0]*np.roll(points[:,1],-1)-points[:,1]*np.roll(points[:,0],-1)))/2
        gap=float(bottom-points[:,2].max()) if index==0 else float(points[:,2].min()-top)
        overlaps=bool(area>1e-10)
        records.append(dict(overlaps_strip=overlaps,signed_gap_m=gap,projected_overlap_area_m2=float(area),
                            within_tolerance=overlaps and abs(gap)<=tolerance_m))
    return dict(pads=records,both_engaged=all(r['within_tolerance'] for r in records),
        scope='Measured link poses plus authored finite pad-face geometry. This does not isolate pad force from whole-link force.')


def evaluate_wafer_grasp_trial(metadata,samples,limits=None,require_return=False):
    """Evaluate a complete recorded fixture trial; missing data never passes.

    Mandatory rows: time_s, phase, wafer_pose_world, clamp_pose_world,
    jaw_contact_normal_force_N[lower,upper], pillar_contact_normal_force_N[3],
    forbidden_wafer_contact_force_N, measured_jaw_gap_m. Contact forces must be
    filtered by the wafer and the named partner; robot-finger totals do not count.
    """
    report=dict(result='INCOMPLETE',lift_result='NOT_EVALUATED',return_result='NOT_EVALUATED',
        protocol='contact_wafer_fixture_v1',checks={},errors=[],
        scope='Recorded dynamic-wafer/hinged-clamp fixture only. No arm transport, real-wafer strength, full pad-area contact, cleanliness, or drying result.',
        measurement_scope='Pose-speed limits apply to all recorded physics intervals. Native engine velocities may differ and are not silently substituted. Metadata attachment/write declarations also require source audit.')
    try:
        cfg=dict(DEFAULT_LIMITS)
        if limits is not None:
            if not isinstance(limits,dict) or set(limits)-set(cfg):raise ValueError('Unknown protocol limit.')
            cfg.update(limits)
        for key,value in cfg.items():
            if _number(value,key)<=0:raise ValueError('Limits must be positive.')
        report['limits']=copy.deepcopy(cfg)
        if type(require_return) is not bool:raise ValueError('require_return must be boolean.')
        dt=_number(metadata['physics_dt_s'],'physics_dt_s')
        if not 0<dt<=1/240+1e-12:raise ValueError('Full physics sampling must be at least 240 Hz.')
        for key,expected in (('wafer_radius_m',.05),('wafer_thickness_m',.001),('wafer_flat_length_m',.0325),('pillar_radius_m',.0005)):
            if abs(_number(metadata[key],key)-expected)>1e-10:raise ValueError('Geometry differs from the declared wafer protocol: '+key)
        bottom=_number(metadata['wafer_local_bottom_z_m'],'wafer_local_bottom_z_m')
        if abs(bottom)>.001:raise ValueError('Unexpected wafer body origin.')
        mass=_number(metadata['wafer_mass_kg'],'wafer_mass_kg')
        if mass<=0:raise ValueError('Wafer mass must be positive.')
        gravity=_array(metadata['gravity_world_m_s2'],(3,),'gravity')
        if np.linalg.norm(gravity[:2])>1e-9 or gravity[2]>=0:raise ValueError('Expected downward world-Z gravity.')
        for key,expected in (('wafer_dynamic',True),('wafer_has_free_joint',True),('wafer_is_mocap',False)):
            if metadata.get(key) is not expected:raise ValueError('Unverified dynamic-wafer declaration: '+key)
        for key in ('wafer_attachment_count','post_start_wafer_pose_write_count'):
            if type(metadata.get(key)) is not int or metadata[key]!=0:raise ValueError('Wafer attachment or state-write evidence is not zero: '+key)
        force_scope=metadata.get('contact_force_scope')
        if force_scope not in ('wafer_partner_filtered','wafer_link_filtered'):raise ValueError('Partner-filtered wafer contact forces are required.')
        require_pad_geometry=force_scope=='wafer_link_filtered' or 'contact_faces_local_m' in metadata
        report['contact_evidence_scope']=('Wafer-to-lever force totals plus separately checked pad-face engagement; not isolated pad forces.'
            if force_scope=='wafer_link_filtered' else 'Wafer-to-named-pad partner-filtered forces.')
        centers=_array(metadata['pillar_top_centers_world_m'],(3,3),'pillar tops')
        if np.ptp(centers[:,2])>1e-8:raise ValueError('Pillar top heights differ.')
        sides=[np.linalg.norm(centers[i,:2]-centers[(i+1)%3,:2]) for i in range(3)]
        if max(abs(s-.05543) for s in sides)>1e-5:raise ValueError('Pillars do not form the specified equilateral triangle.')
        if not isinstance(samples,(list,tuple)) or len(samples)<2:raise ValueError('Missing full-timestep trace.')
        phase_order={'initialization':0,'settle':1,'approach':2,'open_hold':3,'close':4,'grip_hold':5,
                     'lift':6,'lift_hold':7,'lower':8,'return_hold':9,'release':10,'retreat':11,'unassisted_hold':12}
        rows=[]
        for source in samples:
            row=dict(source)
            row['time_s']=_number(row['time_s'],'sample time')
            if row.get('phase') not in phase_order:raise ValueError('Missing or unknown sample phase.')
            row['_wafer']=_pose(row['wafer_pose_world'],'wafer pose')
            row['_clamp']=_pose(row['clamp_pose_world'],'clamp pose')
            row['_jaw_forces']=_array(row['jaw_contact_normal_force_N'],(2,),'jaw forces')
            row['_pillar_forces']=_array(row['pillar_contact_normal_force_N'],(3,),'pillar forces')
            row['_forbidden_force']=_number(row['forbidden_wafer_contact_force_N'],'forbidden contact force')
            row['_gap']=_number(row['measured_jaw_gap_m'],'measured jaw gap')
            if np.any(row['_jaw_forces']<0) or np.any(row['_pillar_forces']<0) or row['_forbidden_force']<0 or row['_gap']<0:
                raise ValueError('Forces and jaw gap must be nonnegative.')
            rows.append(row)
        times=np.array([r['time_s'] for r in rows])
        if not np.allclose(np.diff(times),dt,atol=1e-9,rtol=0):raise ValueError('Trace has missing, duplicate, or irregular physics timestamps.')
        if any(phase_order[b['phase']]<phase_order[a['phase']] for a,b in zip(rows,rows[1:])):
            raise ValueError('Trial phases are out of order.')
        report['trace']=dict(sample_count=len(rows),physics_dt_s=dt,start_time_s=float(times[0]),end_time_s=float(times[-1]))
        report['assumptions']={k:copy.deepcopy(v) for k,v in metadata.items() if k not in ('named_indices',)}
        def check(name,passed,**details):
            report['checks'][name]=dict(passed=bool(passed),**details);return bool(passed)
        def stationary(window,linear,angular):
            stats=_rates(window,dt)
            return (stats['linear_speed_m_s']['maximum']<linear and stats['angular_speed_rad_s']['maximum']<angular),stats
        maximum_jaw=max(float(np.max(r['_jaw_forces'])) for r in rows)
        maximum_forbidden=max(r['_forbidden_force'] for r in rows)
        check('bounded_contact_forces',maximum_jaw<=cfg['maximum_jaw_contact_N'] and maximum_forbidden<=cfg['maximum_unwanted_contact_N'],
            maximum_jaw_force_N=maximum_jaw,maximum_forbidden_force_N=maximum_forbidden,
            scope='Simulation force bounds only, not brittle-wafer stress limits.')
        settled=_window(rows,'settle',cfg['settle_window_s'],dt)
        if settled is None:raise ValueError('Missing complete settling window.')
        baseline=settled[-1]['_wafer'];base_pose=settled[-1]['wafer_pose_world']
        settle_geometry=[pillar_plane_clearances(r['wafer_pose_world'],metadata) for r in settled]
        steady,settle_rates=stationary(settled,cfg['settle_linear_speed_m_s'],cfg['settle_angular_speed_rad_s'])
        pillar_forces=np.array([r['_pillar_forces'] for r in settled[1:]])
        check('initial_support',steady and all(g['all_pillars_under_wafer'] and max(abs(v) for v in g['clearances_m'])<=cfg['support_gap_tolerance_m'] for g in settle_geometry)
            and np.min(pillar_forces)>=cfg['minimum_pillar_contact_N']
            and min(pillar_forces.sum(1))>=.8*mass*abs(gravity[2])
            and max(float(np.max(r['_jaw_forces'])) for r in settled[1:])<=cfg['maximum_unwanted_contact_N'],
            pose_rates=settle_rates,pillar_force_N=_stats(pillar_forces.flatten().tolist()),
            settled_wafer_pose_world=list(base_pose))
        close_phase='grip_hold' if any(r['phase']=='grip_hold' for r in rows) else 'close'
        grasp=_window(rows,close_phase,cfg['contact_window_s'],dt)
        if grasp is None:raise ValueError('Missing complete bilateral-grasp window.')
        reference=_relative(grasp[-1]['_clamp'],grasp[-1]['_wafer'])
        open_gap=float(np.median([r['_gap'] for r in settled]));closed=[r['_gap'] for r in grasp[1:]]
        jaw_forces=np.array([r['_jaw_forces'] for r in grasp[1:]])
        check('moving_jaw_bilateral_grasp',open_gap-max(closed)>=cfg['minimum_measured_gap_reduction_m']
            and min(closed)>=cfg['minimum_closed_gap_m'] and max(closed)<=cfg['maximum_closed_gap_m']
            and np.min(jaw_forces)>=cfg['minimum_jaw_contact_N'],open_gap_m=open_gap,closed_gap_m=_stats(closed),
            jaw_contact_force_N=_stats(jaw_forces.flatten().tolist()))
        held=_window(rows,'lift_hold',cfg['lift_window_s'],dt)
        if held is None:raise ValueError('Missing complete lift-hold window.')
        if require_pad_geometry:
            engaged=[pad_contact_geometry(r,metadata,cfg['pad_geometry_gap_tolerance_m']) for r in grasp[1:]+held[1:]]
            gaps=[p['signed_gap_m'] for g in engaged for p in g['pads'] if p['signed_gap_m'] is not None]
            check('measured_pad_face_engagement',all(g['both_engaged'] for g in engaged),evaluated_sample_count=len(engaged),
                signed_gap_m=_stats(gaps),scope='Finite pad/wafer strip geometry only; link forces are not reclassified as pad-only forces.')
        metrics={name:[] for name in ('rise_m','pillar_clearance_m','xy_drift_m','tilt_deg','relative_translation_m','relative_rotation_deg','flat_alignment_deg')}
        covered=True
        for row in held:
            wafer,clamp=row['_wafer'],row['_clamp'];relation=_relative(clamp,wafer)
            gaps=pillar_plane_clearances(row['wafer_pose_world'],metadata);covered=covered and gaps['all_pillars_under_wafer']
            metrics['rise_m'].append(float(wafer[2,3]-baseline[2,3]))
            metrics['pillar_clearance_m'].append(min(gaps['clearances_m']))
            metrics['xy_drift_m'].append(float(np.linalg.norm(wafer[:2,3]-baseline[:2,3])))
            metrics['tilt_deg'].append(math.degrees(math.acos(float(np.clip(wafer[:3,2]@baseline[:3,2],-1,1)))))
            metrics['relative_translation_m'].append(float(np.linalg.norm(relation[:3,3]-reference[:3,3])))
            metrics['relative_rotation_deg'].append(math.degrees(_angle(reference[:3,:3],relation[:3,:3])))
            metrics['flat_alignment_deg'].append(math.degrees(math.acos(float(np.clip(abs(wafer[:3,1]@clamp[:3,1]),-1,1)))))
        steady,rates=stationary(held,cfg['lift_linear_speed_m_s'],cfg['lift_angular_speed_rad_s'])
        check('measured_lift_and_clearance',covered and min(metrics['rise_m'])>=cfg['minimum_lift_m'] and min(metrics['pillar_clearance_m'])>=cfg['minimum_pillar_clearance_m']
            and max(metrics['xy_drift_m'])<=cfg['maximum_xy_drift_m'] and max(metrics['tilt_deg'])<=cfg['maximum_wafer_tilt_deg'],
            metrics={k:_stats(v) for k,v in metrics.items()})
        check('retention_and_bilateral_contact',max(metrics['relative_translation_m'])<=cfg['maximum_relative_translation_m']
            and max(metrics['relative_rotation_deg'])<=cfg['maximum_relative_rotation_deg']
            and max(metrics['flat_alignment_deg'])<=cfg['maximum_flat_alignment_deg']
            and min(float(np.min(r['_jaw_forces'])) for r in held[1:])>=cfg['minimum_jaw_contact_N']
            and max(float(np.max(r['_pillar_forces'])) for r in held[1:])<=cfg['maximum_unwanted_contact_N'])
        report['hold_motion_diagnostic']=dict(acceptance_criterion=False,within_previous_stationary_limits=steady,pose_rates=rates,scope='Secure retained lifting is required; exact whole-wafer stationarity is not a task requirement.')
        lift_passed=all(v['passed'] for v in report['checks'].values())
        report['lift_result']='WAFER_LIFT_RETAINED_IN_THIS_SIMULATION' if lift_passed else 'FAILED'
        report['result']=report['lift_result']
        unassisted=_window(rows,'unassisted_hold',cfg['return_window_s'],dt)
        if unassisted is not None:
            returned=_window(rows,'return_hold',cfg['return_window_s'],dt)
            observed_release=any(r['phase']=='release' for r in rows) and any(r['phase']=='retreat' for r in rows)
            stable,return_rates=stationary(unassisted,cfg['settle_linear_speed_m_s'],cfg['settle_angular_speed_rad_s'])
            placement=[];clearance=[]
            for row in unassisted:
                t=row['_wafer'];g=pillar_plane_clearances(row['wafer_pose_world'],metadata)
                placement.append(np.linalg.norm(t[:3,3]-baseline[:3,3])<=cfg['return_position_m']
                    and math.degrees(_angle(baseline[:3,:3],t[:3,:3]))<=cfg['return_rotation_deg']
                    and g['all_pillars_under_wafer'] and max(abs(v) for v in g['clearances_m'])<=cfg['support_gap_tolerance_m'])
                clearance.append(float(row['_clamp'][2,3]-grasp[-1]['_clamp'][2,3]))
            contacts=all(np.min(r['_pillar_forces'])>=cfg['minimum_pillar_contact_N']
                and np.sum(r['_pillar_forces'])>=.8*mass*abs(gravity[2])
                and np.max(r['_jaw_forces'])<=cfg['maximum_unwanted_contact_N']
                and r['_gap']>=open_gap-.0002 for r in unassisted[1:])
            returned_ok=check('returned_released_unassisted',returned is not None and observed_release and stable and all(placement)
                and contacts and min(clearance)>=cfg['minimum_retracted_clamp_rise_m'],pose_rates=return_rates,
                minimum_clamp_retraction_m=min(clearance),unassisted_interval_count=len(unassisted)-1)
            report['return_result']='WAFER_RETURNED_AND_RELEASED_IN_THIS_SIMULATION' if returned_ok else 'FAILED'
        elif require_return:
            report['return_result']='INCOMPLETE';report['errors'].append('Return/release evidence was requested but is incomplete.')
        if require_return:
            report['result']=('WAFER_PICKUP_RETURN_SUCCEEDED_IN_THIS_SIMULATION' if lift_passed and report['return_result']=='WAFER_RETURNED_AND_RELEASED_IN_THIS_SIMULATION'
                              else 'INCOMPLETE' if report['return_result']=='INCOMPLETE' else 'FAILED')
        report['failed_checks']=[key for key,value in report['checks'].items() if not value['passed']]
    except (KeyError,TypeError,ValueError,OverflowError) as exc:
        report['errors'].append(type(exc).__name__+': '+str(exc))
        report['result']='INCOMPLETE'
    return report
