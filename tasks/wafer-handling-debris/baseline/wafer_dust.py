"""Visible surrogate debris with contact and a declared, uncalibrated air jet."""
import copy
import math
import xml.etree.ElementTree as ET
import numpy as np

PROFILE = dict(version='surrogate_debris_v1', count=12, side_m=.001, mass_kg=1e-6,
    friction=.4, air_density_kg_m3=1.2, drag_coefficient=1., reference_speed_m_s=20.,
    reference_distance_m=.08, nozzle_radius_m=.002, cone_half_angle_deg=2.15,
    maximum_distance_m=.18, clear_hold_s=1., minimum_jet_impulse_N_s=1e-8,
    scope='Millimetre surrogate debris; no adhesion, calibrated microdust, CFD, evaporation or thermal model. Jet forces act only on debris.')

ADHESIVE_PROFILE = dict(PROFILE, version='cohesive_surrogate_debris_v2',
    adhesion_stiffness_N_m=.08, adhesion_damping_N_s_m=.0002,
    adhesion_break_force_N=45e-6, cone_half_angle_deg=7.5,
    scope='Millimetre surrogate debris with a breakable cohesive surface bond and quadratic jet drag. Parameters are task assumptions, not calibrated dust or CFD. Adhesion has an equal and opposite wafer wrench; no pose locks or particle deletion.')

MIXED_ASSUMPTIONS = dict(version='mixed_visible_debris_v1', physics_dt_s=.0005,
    bond_frequency_hz=80., bond_damping_ratio=.7, bond_break_gravity_multiple=6.,
    gravity_m_s2=9.81, maximum_relative_speed_for_stability_m_s=40.,
    edge_margin_m=.0015, grip_margin_m=.00075, separation_m=.002,
    initial_surface_gap_m=.000001,
    scope='Sparse coarse visible cleanroom-contaminant surrogates, not calibrated composition or nanofabrication cleanliness. '
          'Nominal densities and 6-weight breakable point bonds are engineering assumptions. '
          'Quadratic drag uses measured-orientation projected mesh area; no chemistry, airflow torque, moisture or film model.')


def _mesh_properties(vertices, faces):
    vertices=np.asarray(vertices,dtype=float);tri=vertices[np.asarray(faces,dtype=int)]
    signed=np.einsum('ij,ij->i',tri[:,0],np.cross(tri[:,1],tri[:,2]))/6.
    volume=float(signed.sum())
    if not volume>0:raise ValueError('Convex mesh must have positive outward-wound volume.')
    centroid=(signed[:,None]*tri.sum(axis=1)/4.).sum(axis=0)/volume
    areas=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])/2.
    return volume,centroid,areas


def _mixed_templates():
    """Fixed outward convex meshes keep generated assets identical across platforms."""
    templates={
        'dust_grain': ([
            [0.0, -0.309016994375, -0.5],
            [0.0, -0.309016994375, 0.5],
            [0.0, 0.309016994375, -0.5],
            [0.0, 0.309016994375, 0.5],
            [-0.309016994375, -0.5, 0.0],
            [-0.309016994375, 0.5, 0.0],
            [0.309016994375, -0.5, 0.0],
            [0.309016994375, 0.5, 0.0],
            [-0.5, 0.0, -0.309016994375],
            [-0.5, 0.0, 0.309016994375],
            [0.5, 0.0, -0.309016994375],
            [0.5, 0.0, 0.309016994375],
        ], [[0, 2, 10], [0, 4, 8], [0, 6, 4], [0, 8, 2], [0, 10, 6], [1, 3, 9], [1, 4, 6], [1, 6, 11], [1, 9, 4], [1, 11, 3], [2, 5, 7], [2, 7, 10], [2, 8, 5], [3, 5, 9], [3, 7, 5], [3, 11, 7], [4, 9, 8], [5, 8, 9], [6, 10, 11], [7, 11, 10]]),
        'fiber': ([
            [-0.5, 0.5, 0.0],
            [-0.5, 0.25, 0.5],
            [-0.5, -0.25, 0.5],
            [-0.5, -0.5, 0.0],
            [-0.5, -0.25, -0.5],
            [-0.5, 0.25, -0.5],
            [0.5, 0.5, 0.0],
            [0.5, 0.25, 0.5],
            [0.5, -0.25, 0.5],
            [0.5, -0.5, 0.0],
            [0.5, -0.25, -0.5],
            [0.5, 0.25, -0.5],
        ], [[0, 1, 7], [0, 5, 1], [0, 6, 11], [0, 7, 6], [0, 11, 5], [1, 2, 7], [1, 3, 2], [1, 4, 3], [1, 5, 4], [2, 3, 9], [2, 8, 7], [2, 9, 8], [3, 4, 9], [4, 5, 11], [4, 10, 9], [4, 11, 10], [6, 7, 9], [6, 9, 11], [7, 8, 9], [9, 10, 11]]),
        'flake': ([
            [0.532680221963, -0.002601570777, -0.5],
            [0.250502004141, 0.420821852646, -0.5],
            [-0.254448490909, 0.492893924718, -0.5],
            [-0.467319778037, -0.002601570777, -0.5],
            [-0.259398985958, -0.507106075282, -0.5],
            [0.240601014042, -0.408006976183, -0.5],
            [0.532680221963, -0.002601570777, 0.5],
            [0.250502004141, 0.420821852646, 0.5],
            [-0.254448490909, 0.492893924718, 0.5],
            [-0.467319778037, -0.002601570777, 0.5],
            [-0.259398985958, -0.507106075282, 0.5],
            [0.240601014042, -0.408006976183, 0.5],
        ], [[0, 1, 7], [0, 5, 1], [0, 6, 11], [0, 7, 6], [0, 11, 5], [1, 2, 7], [1, 3, 2], [1, 4, 3], [1, 5, 4], [2, 3, 8], [2, 8, 7], [3, 4, 10], [3, 9, 8], [3, 10, 9], [4, 5, 11], [4, 11, 10], [6, 7, 11], [7, 8, 9], [7, 9, 10], [7, 10, 11]]),
        'wear_fragment': ([
            [-0.510082150859, -0.47879407256, -0.448842419716],
            [0.489917849141, -0.320899335718, -0.448842419716],
            [0.339917849141, 0.52120592744, -0.448842419716],
            [-0.410082150859, 0.415942769545, -0.448842419716],
            [-0.310082150859, -0.373530914665, 0.551157580284],
            [0.439917849141, -0.268267756771, 0.551157580284],
            [0.189917849141, 0.363311190598, 0.551157580284],
            [-0.260082150859, 0.31067961165, 0.551157580284],
        ], [[0, 1, 5], [0, 3, 1], [0, 4, 7], [0, 5, 4], [0, 7, 3], [1, 2, 5], [1, 3, 2], [2, 3, 7], [2, 6, 5], [2, 7, 6], [4, 5, 6], [4, 6, 7]]),
    }
    return {kind:(np.asarray(v,dtype=float),np.asarray(f,dtype=int)) for kind,(v,f) in templates.items()}

def rotations_from_xyzw(quaternions):
    q=np.asarray(quaternions,dtype=float)
    if q.ndim!=2 or q.shape[1]!=4 or not np.isfinite(q).all():raise ValueError('Expected finite Nx4 XYZW quaternions.')
    norm=np.linalg.norm(q,axis=1)
    if np.any(abs(norm-1)>1e-4):raise ValueError('Particle orientation is not a unit quaternion.')
    x,y,z,w=(q/norm[:,None]).T
    return np.stack((1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),
                     2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),
                     2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)),axis=1).reshape(-1,3,3)


def mixed_layout(seed=0,*,wafer_radius_m=.05,wafer_thickness_m=.001,wafer_flat_m=.0325,
                 top_z_m=.0005,protected_band_depth_m=.005,protected_band_half_width_m=.01625,
                 physics_dt_s=.0005):
    """Twelve reproducible convex shapes; geometry, masses and force parameters share one specification."""
    if isinstance(seed,bool)or not isinstance(seed,(int,np.integer))or seed<0:raise ValueError('Seed must be a nonnegative integer.')
    if physics_dt_s!=MIXED_ASSUMPTIONS['physics_dt_s']:raise ValueError('Mixed debris is qualified only at 2 kHz.')
    if not(wafer_radius_m>0 and 0<wafer_flat_m<2*wafer_radius_m and wafer_thickness_m>0
            and 0<protected_band_depth_m<wafer_radius_m and protected_band_half_width_m>0):
        raise ValueError('Invalid wafer or protected-band dimensions.')
    rng=np.random.default_rng(seed);templates=_mixed_templates();particles=[];centers=[]
    flat=-math.sqrt(wafer_radius_m**2-(wafer_flat_m/2)**2)
    kinds=np.asarray(['dust_grain','fiber','flake','wear_fragment']*3);rng.shuffle(kinds)
    labels=dict(dust_grain='silicon/quartz-like grain',fiber='garment/lint or hair-like fiber',
                flake='skin-like or cleanroom-polymer flake',wear_fragment='equipment-wear fragment')
    densities=dict(dust_grain=2400.,fiber=1200.,flake=1050.,wear_fragment=2700.)
    colors=dict(dust_grain=[.20,.22,.25],fiber=[.78,.74,.65],flake=[.72,.66,.56],wear_fragment=[.30,.33,.37])
    omega=2*math.pi*MIXED_ASSUMPTIONS['bond_frequency_hz']
    for index,kind_value in enumerate(kinds):
        kind=str(kind_value)
        if kind=='dust_grain':
            length=rng.uniform(.0004,.001);scale=np.array([length,length*rng.uniform(.65,.95),length*rng.uniform(.55,.85)])
        elif kind=='fiber':
            length=rng.uniform(.001,.002);diameter=rng.uniform(.00008,.00016);scale=np.array([length,diameter,diameter])
        elif kind=='flake':
            length=rng.uniform(.0005,.0012);scale=np.array([length,length*rng.uniform(.55,.9),rng.uniform(.00003,.00007)])
        else:
            length=rng.uniform(.00045,.0009);scale=np.array([length,length*rng.uniform(.55,.85),length*rng.uniform(.35,.65)])
        canonical,faces=templates[kind];vertices=canonical*scale
        volume,center,areas=_mesh_properties(vertices,faces)
        if np.linalg.norm(center)>1e-12:raise ValueError('Particle mesh is not centered at its volume centroid.')
        mass=volume*densities[kind];yaw=float(rng.uniform(-math.pi,math.pi))
        quat=[0.,0.,math.sin(yaw/2),math.cos(yaw/2)]
        radius=float(np.linalg.norm(vertices[:,:2],axis=1).max())
        for attempt in range(10000):
            angle=rng.uniform(-math.pi,math.pi);r=wafer_radius_m*math.sqrt(rng.random())
            xy=np.array([r*math.cos(angle),r*math.sin(angle)])
            if np.linalg.norm(xy)+radius>wafer_radius_m-MIXED_ASSUMPTIONS['edge_margin_m']:continue
            if xy[0]-radius<flat+MIXED_ASSUMPTIONS['edge_margin_m']:continue
            q=np.maximum(np.abs(xy-[flat+protected_band_depth_m/2,0.])-[protected_band_depth_m/2,protected_band_half_width_m],0.)
            if np.linalg.norm(q)<radius+MIXED_ASSUMPTIONS['grip_margin_m']:continue
            if any(np.linalg.norm(xy-np.asarray(p)[:2])<radius+old['bounding_radius_xy_m']+MIXED_ASSUMPTIONS['separation_m']
                   for p,old in zip(centers,particles)):continue
            break
        else:raise ValueError('Could not place nonoverlapping debris on the exposed wafer.')
        centers.append([*xy,float(top_z_m-vertices[:,2].min()+MIXED_ASSUMPTIONS['initial_surface_gap_m'])])
        particles.append(dict(id=f'dust_{index:02d}',kind=kind,description=labels[kind],
            canonical_vertices=canonical.tolist(),scale_m=scale.tolist(),vertices_local_m=vertices.tolist(),faces=faces.tolist(),
            initial_rotation_xyzw=quat,initial_yaw_rad=yaw,color_rgb=np.clip(np.asarray(colors[kind])+rng.uniform(-.04,.04,3),.05,.9).tolist(),
            mass_kg=mass,density_kg_m3=densities[kind],volume_m3=volume,bounding_radius_xy_m=radius,
            adhesion_stiffness_N_m=mass*omega**2,adhesion_damping_N_s_m=2*MIXED_ASSUMPTIONS['bond_damping_ratio']*mass*omega,
            adhesion_break_force_N=MIXED_ASSUMPTIONS['bond_break_gravity_multiple']*mass*MIXED_ASSUMPTIONS['gravity_m_s2'],
            drag_face_area_vectors_local_m2=areas.tolist()))
    max_faces=max(len(p['faces'])for p in particles);area_vectors=np.zeros((12,max_faces,3))
    for index,p in enumerate(particles):area_vectors[index,:len(p['faces'])]=p['drag_face_area_vectors_local_m2']
    profile=dict(copy.deepcopy(ADHESIVE_PROFILE),version='mixed_cohesive_debris_v1',
        mass_kg=[p['mass_kg']for p in particles],side_m=max(max(p['scale_m'])for p in particles),
        drag_face_area_vectors_local_m2=area_vectors.tolist(),scope=MIXED_ASSUMPTIONS['scope'])
    for key in ('adhesion_stiffness_N_m','adhesion_damping_N_s_m','adhesion_break_force_N'):
        profile[key]=[p[key]for p in particles]
    spec=dict(shape_schema='mixed_convex_debris_v1',seed=int(seed),ids=[p['id']for p in particles],particles=particles,
        initial_centers_wafer_m=centers,profile=profile,post_start_particle_pose_writes=0,
        assumptions=copy.deepcopy(MIXED_ASSUMPTIONS),
        placement=dict(wafer_radius_m=wafer_radius_m,wafer_thickness_m=wafer_thickness_m,wafer_flat_m=wafer_flat_m,
                       wafer_top_z_m=top_z_m,protected_band_depth_m=protected_band_depth_m,
                       protected_band_half_width_m=protected_band_half_width_m),
        category_sources=['https://sma.nasa.gov/sma-disciplines/planetary-protection/explore/explore-item/cleanroom-gowning-or-how-to-dress-in-the-cleanroom',
                          'https://research.yale.edu/documents/cleanroom-handbook-40'],scope=MIXED_ASSUMPTIONS['scope'])
    spec['cpu_stability']=mixed_stability(spec)
    return spec


def mixed_stability(spec):
    p=spec['profile'];dt=spec['assumptions']['physics_dt_s'];mass=np.asarray(p['mass_kg'])
    k=np.asarray(p['adhesion_stiffness_N_m']);c=np.asarray(p['adhesion_damping_N_s_m'])
    area_bound=np.linalg.norm(p['drag_face_area_vectors_local_m2'],axis=2).sum(axis=1)/2
    drag_damping=p['air_density_kg_m3']*p['drag_coefficient']*area_bound*spec['assumptions']['maximum_relative_speed_for_stability_m_s']
    q2=k/mass*dt**2;d=(c+drag_damping)/mass*dt
    # Sufficient semi-implicit Euler stability bound for each damped translation axis.
    bound=q2+2*d
    if np.any(~np.isfinite(bound))or np.any(bound>=4):raise ValueError('Mixed-debris force timestep fails conservative stability bound.')
    return dict(physics_dt_s=dt,maximum_sufficient_stability_expression=float(bound.max()),required_less_than=4.,
                gravity_equilibrium_displacement_m=float((mass*9.81/k).max()),
                break_to_weight_ratio=float((np.asarray(p['adhesion_break_force_N'])/(mass*9.81)).min()),
                scope='Translation force-equation bound up to 40 m/s relative air speed; not contact-solver or native-run validation.')


def mixed_cpu_checks(count=20):
    """Geometry and force-equation checks, with no native simulator or file writes."""
    from scipy.spatial import ConvexHull
    if not isinstance(count,int)or count<20:raise ValueError('Use at least20 randomized layouts.')
    maximum_stability=maximum_equilibrium_error=maximum_tilt_error=maximum_break_time=0.
    kinds_seen=set();previous=None;dt=MIXED_ASSUMPTIONS['physics_dt_s']
    for seed in range(count):
        spec=mixed_layout(seed)
        if spec!=mixed_layout(seed):raise AssertionError('Seed is not deterministic.')
        centers=np.asarray(spec['initial_centers_wafer_m']);parts=spec['particles'];profile=spec['profile']
        if previous is not None and np.array_equal(centers,previous):raise AssertionError('Seeds do not vary the layout.')
        previous=centers.copy();rotations=rotations_from_xyzw([p['initial_rotation_xyzw']for p in parts])
        flat=-math.sqrt(.05**2-(.0325/2)**2)
        for index,part in enumerate(parts):
            vertices=np.asarray(part['vertices_local_m']);world=vertices@rotations[index].T+centers[index]
            volume,com,areas=_mesh_properties(vertices,part['faces']);kinds_seen.add(part['kind'])
            if not np.allclose(com,0.,atol=1e-13,rtol=0.)or not math.isclose(part['mass_kg'],volume*part['density_kg_m3'],rel_tol=1e-12):
                raise AssertionError('Mesh center or density-derived mass is inconsistent.')
            if not(np.linalg.norm(world[:,:2],axis=1).max()<.05 and world[:,0].min()>flat
                   and abs(world[:,2].min()-(.0005+MIXED_ASSUMPTIONS['initial_surface_gap_m']))<1e-12):
                raise AssertionError('The actual footprint is not supported by the wafer.')
            q=np.maximum(np.abs(centers[index,:2]-[flat+.0025,0.])-[.0025,.01625],0.)
            if np.linalg.norm(q)<part['bounding_radius_xy_m']+MIXED_ASSUMPTIONS['grip_margin_m']-1e-12:
                raise AssertionError('A particle overlaps the protected grip band.')
            for other in range(index):
                if np.linalg.norm(centers[index,:2]-centers[other,:2])<part['bounding_radius_xy_m']+parts[other]['bounding_radius_xy_m']+MIXED_ASSUMPTIONS['separation_m']-1e-12:
                    raise AssertionError('Particle footprints overlap.')
            if not np.allclose(areas,part['drag_face_area_vectors_local_m2'],atol=1e-18,rtol=1e-12):
                raise AssertionError('Drag geometry differs from the collision mesh.')
        mass=np.asarray(profile['mass_kg'])[:,None];gravity=np.array([0.,0.,-9.81])
        direction=np.array([.6,0.,.8]);basis=np.array([[0.,1.,0.],[-.8,0.,.6]]).T
        measured=jet_force(np.zeros((12,3)),np.zeros((12,3)),-.08*direction,direction,True,profile,rotations_world=rotations)
        for index,part in enumerate(parts):
            projected=np.asarray(part['vertices_local_m'])@rotations[index].T@basis
            independent_area=ConvexHull(projected).volume
            expected=.5*profile['air_density_kg_m3']*profile['drag_coefficient']*independent_area*20.**2*direction
            if not np.allclose(measured[index],expected,atol=1e-18,rtol=1e-11):
                raise AssertionError('Projected-mesh drag does not match the independent2D hull area.')
        maximum_stability=max(maximum_stability,spec['cpu_stability']['maximum_sufficient_stability_expression'])
        for angle in (0.,math.pi/3,math.pi/2,math.pi):
            rotation=np.array([[math.cos(angle),0.,math.sin(angle)],[0.,1.,0.],[-math.sin(angle),0.,math.cos(angle)]])
            targets=centers@rotation.T;position=targets.copy();velocity=np.zeros_like(position)
            bonds=SurfaceAdhesion(centers,profile)
            for _ in range(1000):
                force,reaction,torque=bonds.forces(position,velocity,np.zeros(3),rotation,np.zeros(3),np.zeros(3))
                velocity+=(force/mass+gravity)*dt;position+=velocity*dt
            if not bonds.bound.all():raise AssertionError('Gravity alone broke a cohesive bond.')
            equilibrium=mass*gravity/np.asarray(profile['adhesion_stiffness_N_m'])[:,None]
            maximum_equilibrium_error=max(maximum_equilibrium_error,float(abs(position-targets-equilibrium).max()))
            if not np.allclose(force.sum(axis=0)+reaction,0.,atol=1e-18)or not np.allclose(np.cross(position-velocity*dt,force).sum(axis=0)+torque,0.,atol=1e-18):
                raise AssertionError('Adhesion does not conserve the applied force/torque pair.')
        bonds=SurfaceAdhesion(centers,profile);position=centers.copy();velocity=np.zeros_like(position)
        for tick in range(3000):
            t=min(tick*dt,1.);angle=(math.pi/2)*(3*t*t-2*t*t*t)
            omega=(math.pi/2)*(6*t-6*t*t)if t<1 else 0.
            rotation=np.array([[math.cos(angle),0.,math.sin(angle)],[0.,1.,0.],[-math.sin(angle),0.,math.cos(angle)]])
            force,_,_=bonds.forces(position,velocity,np.zeros(3),rotation,np.zeros(3),[0.,omega,0.])
            velocity+=(force/mass+gravity)*dt;position+=velocity*dt
            maximum_tilt_error=max(maximum_tilt_error,float(np.linalg.norm(position-centers@rotation.T,axis=1).max()))
        if not bonds.bound.all():raise AssertionError('A slow90-degree tilt broke an unblown bond.')
        zeros=np.zeros((12,3));axis=np.array([math.cos(math.pi/9),0.,-math.sin(math.pi/9)])
        if np.any(jet_force(centers,zeros,[.3,0.,-.08],[0.,0.,1.],True,profile,rotations_world=rotations)):
            raise AssertionError('An off-axis jet applied a removal force.')
        if np.any(jet_force(centers,zeros,[0.,0.,.08],axis,False,profile,rotations_world=rotations)):
            raise AssertionError('A switched-off gun applied a removal force.')
        # Independent point-mass shear test: an ideal flat support supplies normal reaction and Coulomb friction.
        position=zeros.copy();velocity=zeros.copy();bonds=SurfaceAdhesion(zeros,profile)
        for tick in range(1000):
            cohesive,_,_=bonds.forces(position,velocity,np.zeros(3),np.eye(3),np.zeros(3),np.zeros(3))
            drag=jet_force(position,velocity,-.08*axis,axis,True,profile,rotations_world=rotations)
            total=cohesive+drag+mass*gravity;normal=np.maximum(-total[:,2],0.)
            tangential=total[:,:2];norm=np.linalg.norm(tangential,axis=1)
            friction=profile['friction']*normal
            tangential*=np.maximum(0.,1-friction/np.maximum(norm,1e-30))[:,None]
            velocity[:,:2]+=tangential/mass*dt;position[:,:2]+=velocity[:,:2]*dt
            if not bonds.bound.any():
                maximum_break_time=max(maximum_break_time,(tick+1)*dt)
                break
        else:raise AssertionError('A directed reference jet did not detach every surrogate in the force test.')
    if kinds_seen!={'dust_grain','fiber','flake','wear_fragment'}or maximum_equilibrium_error>1e-9:
        raise AssertionError('Mixed category or force-equilibrium check failed.')
    return dict(passed=True,layouts=count,particles_per_layout=12,
        checks=['deterministic_seed','non_grid_layout_variation','centroid_and_mass','on_wafer_actual_footprint',
                'protected_grip_band','nonoverlap','mesh_area_drag','gravity_and_inverted_retention',
                'slow_tilt_retention','action_reaction_balance','gun_off_no_jet','off_axis_no_jet',
                'directed_reference_jet_detachment'],
        maximum_stability_expression=maximum_stability,maximum_gravity_equilibrium_error_m=maximum_equilibrium_error,
        maximum_slow_tilt_tracking_error_m=maximum_tilt_error,maximum_directed_jet_detachment_s=maximum_break_time,
        scope='CPU geometry and explicit force equations, including an ideal-plane shear check. No Isaac/contact-solver, actual material, cleanroom cleanliness or full-task success claim.')


class SurfaceAdhesion:
    """Breakable spring bonds, independent of the gun switch."""
    def __init__(self, anchors_wafer_m, profile=ADHESIVE_PROFILE):
        self.anchors=np.asarray(anchors_wafer_m,dtype=float).copy()
        self.profile=profile;self.bound=np.ones(len(self.anchors),dtype=bool)
        self.stiffness=np.broadcast_to(profile['adhesion_stiffness_N_m'],(len(self.anchors),))[:,None]
        self.damping=np.broadcast_to(profile['adhesion_damping_N_s_m'],(len(self.anchors),))[:,None]
        self.break_force=np.broadcast_to(profile['adhesion_break_force_N'],(len(self.anchors),))

    def forces(self, positions, velocities, wafer_position, wafer_rotation,
               wafer_velocity, wafer_angular_velocity):
        p=np.asarray(positions);v=np.asarray(velocities);origin=np.asarray(wafer_position)
        offsets=self.anchors@np.asarray(wafer_rotation).T
        targets=origin+offsets
        target_velocity=np.asarray(wafer_velocity)+np.cross(wafer_angular_velocity,offsets)
        elastic=-self.stiffness*(p-targets)
        trial=elastic-self.damping*(v-target_velocity)
        self.bound &= np.linalg.norm(trial,axis=1)<self.break_force
        force=np.where(self.bound[:,None],trial,0.)
        reaction=-force.sum(axis=0)
        torque=-np.cross(p-origin,force).sum(axis=0)
        return force,reaction,torque


def cohesive_forces(bonds,row,profile):
    from wafer_grasp_protocol import _pose
    pose=_pose(row['wafer_pose_world'],'wafer')
    adhesion,reaction,torque=bonds.forces(row['dust_position_world_m'],row['dust_velocity_world_m_s'],
        pose[:3,3],pose[:3,:3],row['dust_wafer_link_velocity_world_m_s'],row['dust_wafer_angular_velocity_world_rad_s'])
    rotations=None
    if 'drag_face_area_vectors_local_m2'in profile:
        rotations=rotations_from_xyzw(np.asarray(row['dust_pose_world'])[:,3:7])
    jet=jet_force(row['dust_position_world_m'],row['dust_velocity_world_m_s'],
        row['nozzle_tip_world_m'],row['nozzle_axis_world'],row['air_on'],profile,rotations_world=rotations)
    torque-=np.cross(np.asarray(row['dust_wafer_com_world_m'])-pose[:3,3],reaction)
    return adhesion+jet,jet,reaction,torque


class CohesiveDustScore:
    def __init__(self,metadata):
        self.spec=metadata['dust'];self.profile=self.spec['profile'];self.dt=float(metadata['physics_dt_s'])
        self.bonds=SurfaceAdhesion(self.spec['initial_centers_wafer_m'],self.profile)
        self.last=None;self.first_air=None;self.clear_s=0.;self.early_loss=False;self.initial=False;self.pre_return_clear_s=None
        self.error=0.;self.wrench_error=0.;self.other_force=0.;self.intact=True;self.states_match=True
        self.impulse=np.zeros(self.profile['count']);self.count=0;self.remaining=self.profile['count']

    def update(self,row):
        from wafer_grasp_protocol import _pose
        p=np.asarray(row['dust_position_world_m']);pose=_pose(row['wafer_pose_world'],'wafer')
        local=(p-pose[:3,3])@pose[:3,:3];radius=math.sqrt(3)*self.profile['side_m']/2
        flat=-math.sqrt(.05**2-(.0325/2)**2)
        on=(np.linalg.norm(local[:,:2],axis=1)<.05-radius)&(local[:,0]>flat+radius)&(np.abs(local[:,2]-.001)<.002)
        clear=((np.linalg.norm(local[:,:2],axis=1)>.05+radius)|(local[:,0]<flat-radius))&(np.asarray(row['dust_wafer_contact_N'])<1e-8)
        if row['phase']=='lower' and self.pre_return_clear_s is None:self.pre_return_clear_s=self.clear_s
        if self.last is None:self.initial=bool(on.all())
        else:
            if abs(row['time_s']-self.last['time_s']-self.dt)>1e-8:raise ValueError('Particle trace is discontinuous.')
            force,jet,reaction,torque=cohesive_forces(self.bonds,self.last,self.profile)
            self.error=max(self.error,float(np.max(abs(force-np.asarray(row['dust_applied_force_world_N'])))))
            self.wrench_error=max(self.wrench_error,float(np.max(abs(np.r_[reaction,torque]-row['dust_applied_wafer_wrench_world_N_Nm']))))
            self.impulse+=np.linalg.norm(jet,axis=1)*self.dt
            self.states_match &= np.array_equal(self.bonds.bound,row['dust_bonds_bound'])
        if self.first_air is None:
            if row['air_on']:self.first_air=row['time_s']
            elif not on.all() or not all(row['dust_bonds_bound']):self.early_loss=True
        self.intact &= bool(row['dust_intact']) and p.shape==(self.profile['count'],3) and np.isfinite(p).all()
        self.other_force=max(self.other_force,float(row['dust_other_external_wrench_max_abs']))
        self.clear_s=self.clear_s+self.dt if clear.all() and self.first_air is not None else 0.
        self.remaining=int((~clear).sum());self.count+=1;self.last=row

    def result(self):
        checks=dict(initially_on_wafer=self.initial,no_loss_before_air=not self.early_loss,
            all_particles_present_and_unchanged=bool(self.intact),all_particles_received_jet=bool(min(self.impulse)>=self.profile['minimum_jet_impulse_N_s']),
            all_particles_clear=self.clear_s+1e-8>=self.profile['clear_hold_s'],
            all_clear_before_wafer_return=self.pre_return_clear_s is not None and self.pre_return_clear_s+1e-8>=self.profile['clear_hold_s'],
            force_equations_replayed=self.error<=1e-10,wafer_reaction_replayed=self.wrench_error<=1e-10,
            bond_states_replayed=bool(self.states_match),no_other_external_wrench=self.other_force==0.,
            no_particle_pose_writes=self.spec['post_start_particle_pose_writes']==0)
        return dict(passed=all(checks.values()),checks=checks,remaining=self.remaining,
            clear_hold_s=self.clear_s,pre_return_clear_hold_s=self.pre_return_clear_s,first_air_s=self.first_air,samples=self.count,
            maximum_force_replay_error_N=self.error,maximum_wafer_wrench_error=self.wrench_error,
            jet_impulse_N_s=self.impulse.tolist(),scope=self.profile['scope'])


def layout(seed=0):
    p=np.array([[x,y,.001001] for y in (-.03,-.01,.01,.03) for x in (-.025,0.,.025)])
    if seed:p[:,:2]+=np.random.default_rng(seed).uniform(-.002,.002,(12,2))
    return p


def jet_force(position,velocity,tip,axis,air_on,profile=PROFILE,*,rotations_world=None):
    position=np.asarray(position);velocity=np.asarray(velocity)
    if not air_on:return np.zeros_like(position)
    axis=np.asarray(axis);delta=position-np.asarray(tip);along=delta@axis
    radial=np.linalg.norm(delta-along[:,None]*axis,axis=1)
    width=profile['nozzle_radius_m']+np.maximum(along,0)*math.tan(math.radians(profile['cone_half_angle_deg']))
    speed=profile['reference_speed_m_s']*np.minimum(1.,profile['reference_distance_m']/np.maximum(along,1e-6))*np.exp(-2*(radial/width)**2)
    active=(along>0)&(along<profile['maximum_distance_m'])&(radial<3*width)
    relative=speed[:,None]*axis-velocity
    magnitude=np.linalg.norm(relative,axis=1)
    if 'drag_face_area_vectors_local_m2'in profile:
        rotations=np.asarray(rotations_world,dtype=float)
        if rotations.shape!=(len(position),3,3):raise ValueError('Mixed drag requires measured particle orientations.')
        local_direction=np.einsum('ni,nij->nj',relative/np.maximum(magnitude[:,None],1e-30),rotations)
        areas=np.asarray(profile['drag_face_area_vectors_local_m2'])
        if areas.shape[0]!=len(position):raise ValueError('Mixed drag particle count differs.')
        area=.5*np.abs(np.einsum('nfi,ni->nf',areas,local_direction)).sum(axis=1)
    else:area=np.full(len(position),profile['side_m']**2)
    force=.5*profile['air_density_kg_m3']*profile['drag_coefficient']*area[:,None]*magnitude[:,None]*relative
    return np.where(active[:,None],force,0.)


def add_particles(old_model,old_data,metadata,xml,seed=0):
    import mujoco
    if old_data.time!=0:raise ValueError('Particles must exist before the trial starts.')
    tree=ET.fromstring(xml);world=tree.find('worldbody');wafer=old_data.body('wafer')
    positions=layout(seed);rotation=wafer.xmat.reshape(3,3)
    for i,p in enumerate(positions):
        body=ET.SubElement(world,'body',name=f'dust_{i:02d}',pos=' '.join(map(str,wafer.xpos+rotation@p)),quat=' '.join(map(str,wafer.xquat)))
        ET.SubElement(body,'freejoint',name=f'dust_joint_{i:02d}')
        ET.SubElement(body,'geom',name=f'dust_geom_{i:02d}',type='box',size='.0005 .0005 .0005',mass=str(PROFILE['mass_kg']),
            friction='.4 .00001 .000001',condim='3',rgba='1 .22 .02 1',solref='.002 1',solimp='.95 .99 .0001',margin='0',gap='0')
    xml=ET.tostring(tree,encoding='unicode');model=mujoco.MjModel.from_xml_string(xml);data=mujoco.MjData(model)
    for i in range(old_model.njnt):
        if model.jnt_qposadr[i]!=old_model.jnt_qposadr[i]:raise ValueError('Existing joint ordering changed.')
    data.qpos[:old_model.nq]=old_data.qpos;data.qvel[:old_model.nv]=old_data.qvel;data.ctrl[:]=old_data.ctrl
    mujoco.mj_forward(model,data)
    metadata['dust']=dict(profile=copy.deepcopy(PROFILE),seed=seed,initial_centers_wafer_m=positions.tolist(),
        ids=[f'dust_{i:02d}'for i in range(12)],post_start_particle_pose_writes=0,
        force_timing='Measured at interval start, held for one physics step; no forces on robot, tools or wafer.')
    return model,data,metadata,xml


class DustPhysics:
    def __init__(self,model,data,metadata):
        self.m,self.d=model,data;self.spec=metadata['dust'];self.profile=self.spec['profile']
        self.ids=[model.body(n).id for n in self.spec['ids']]
        self.geoms={model.geom(f'dust_geom_{i:02d}').id:i for i in range(len(self.ids))}
        self.wafer={model.geom(n).id for n in metadata.get('wafer_geom_names',['wafer_geom'])}
        self.impulse=np.zeros(len(self.ids));self.force=np.zeros((len(self.ids),3))
        self.applied_air=False;self.applied_tip=np.zeros(3);self.applied_axis=np.zeros(3)

    def measure(self):
        import mujoco
        p=self.d.xpos[self.ids].copy();v=np.zeros_like(p)
        for i,b in enumerate(self.ids):
            velocity=np.zeros(6);mujoco.mj_objectVelocity(self.m,self.d,mujoco.mjtObj.mjOBJ_BODY,b,velocity,0);v[i]=velocity[3:]
        contact=np.zeros(len(self.ids))
        for j,c in enumerate(self.d.contact):
            if c.geom1 in self.geoms and c.geom2 in self.wafer:i=self.geoms[c.geom1]
            elif c.geom2 in self.geoms and c.geom1 in self.wafer:i=self.geoms[c.geom2]
            else:continue
            force=np.zeros(6);mujoco.mj_contactForce(self.m,self.d,j,force);contact[i]+=max(0.,force[0])
        return dict(dust_position_world_m=p.tolist(),dust_velocity_world_m_s=v.tolist(),dust_wafer_contact_N=contact.tolist(),
            dust_applied_force_world_N=self.d.xfrc_applied[self.ids,:3].copy().tolist(),dust_applied_air_on=self.applied_air,
            dust_applied_nozzle_tip_world_m=self.applied_tip.tolist(),dust_applied_nozzle_axis_world=self.applied_axis.tolist(),
            dust_jet_impulse_N_s=self.impulse.tolist(),dust_present=all(self.m.body(n).id==b for n,b in zip(self.spec['ids'],self.ids)))

    def next_step(self,row,enabled=True):
        self.applied_air=bool(row['air_on']and enabled);self.applied_tip=np.asarray(row['nozzle_tip_world_m']);self.applied_axis=np.asarray(row['nozzle_axis_world'])
        self.force=jet_force(row['dust_position_world_m'],row['dust_velocity_world_m_s'],self.applied_tip,self.applied_axis,self.applied_air,self.profile)
        self.d.xfrc_applied[self.ids,:3]=self.force;self.d.xfrc_applied[self.ids,3:]=0


class DustScore:
    def __init__(self,metadata):
        self.spec=metadata['dust'];self.dt=float(metadata['physics_dt_s']);self.profile=self.spec['profile']
        self.first_air=None;self.last=None;self.clear_s=0.;self.early_loss=False;self.initial=None;self.force_error=0.;self.other_force=0.;self.present=True;self.count=0
        self.impulse=np.zeros(self.profile['count']);self.input_consistent=True

    def update(self,row):
        from wafer_grasp_protocol import _pose
        p=np.asarray(row['dust_position_world_m']);v=np.asarray(row['dust_velocity_world_m_s']);pose=_pose(row['wafer_pose_world'],'wafer')
        local=(p-pose[:3,3])@pose[:3,:3];radius=math.sqrt(3)*self.profile['side_m']/2;flat=-math.sqrt(.05**2-(.0325/2)**2)
        on=(np.linalg.norm(local[:,:2],axis=1)<.05-radius)&(local[:,0]>flat+radius)&(np.abs(local[:,2]-.001)<.002)
        clear=((np.linalg.norm(local[:,:2],axis=1)>.05+radius)|(local[:,0]<flat-radius))&(np.asarray(row['dust_wafer_contact_N'])<1e-8)
        if self.initial is None:self.initial=bool(on.all())
        if self.first_air is None:
            if row['air_on']:self.first_air=row['time_s']
            elif not on.all():self.early_loss=True
        if self.last is not None:
            if abs(row['time_s']-self.last['time_s']-self.dt)>1e-8:raise ValueError('Dust trace is discontinuous.')
            self.input_consistent=self.input_consistent and row['dust_applied_air_on']==self.last['air_on']and np.allclose(row['dust_applied_nozzle_tip_world_m'],self.last['nozzle_tip_world_m'],atol=1e-12,rtol=0)and np.allclose(row['dust_applied_nozzle_axis_world'],self.last['nozzle_axis_world'],atol=1e-12,rtol=0)
            expected=jet_force(self.last['dust_position_world_m'],self.last['dust_velocity_world_m_s'],row['dust_applied_nozzle_tip_world_m'],row['dust_applied_nozzle_axis_world'],row['dust_applied_air_on'],self.profile)
            self.force_error=max(self.force_error,float(np.max(np.abs(expected-np.asarray(row['dust_applied_force_world_N'])))))
            self.impulse+=np.linalg.norm(np.asarray(row['dust_applied_force_world_N']),axis=1)*self.dt
            self.input_consistent=self.input_consistent and np.allclose(self.impulse,row['dust_jet_impulse_N_s'],atol=1e-12,rtol=0)
        self.present=self.present and row['dust_present']and p.shape==(self.profile['count'],3)and np.isfinite(p).all()and np.isfinite(v).all()
        self.other_force=max(self.other_force,row.get('external_wrench_other_body_max_abs_N_Nm',0.))
        self.clear_s=self.clear_s+self.dt if clear.all()and self.first_air is not None else 0.
        self.last=row;self.count+=1;self.remaining=int((~clear).sum())

    def result(self):
        exposed=bool(self.last is not None and min(self.last['dust_jet_impulse_N_s'])>=self.profile['minimum_jet_impulse_N_s'])
        checks=dict(initially_on_wafer=self.initial is True,no_loss_before_air=not self.early_loss,all_particles_present=bool(self.present),
            all_particles_received_jet=exposed,all_clear_for_one_second=self.clear_s+1e-8>=self.profile['clear_hold_s'],
            force_equation_replayed=self.force_error<1e-12,no_other_external_force=self.other_force==0.,
            jet_follows_measured_gun=bool(self.input_consistent),
            no_particle_pose_writes=self.spec['post_start_particle_pose_writes']==0)
        return dict(passed=all(checks.values()),checks=checks,remaining=self.remaining if self.last else self.profile['count'],
            clear_hold_s=self.clear_s,first_air_s=self.first_air,samples=self.count,maximum_force_replay_error_N=self.force_error,
            scope=self.profile['scope'])
