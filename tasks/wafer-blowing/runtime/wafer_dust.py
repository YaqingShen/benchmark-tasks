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


class SurfaceAdhesion:
    """Breakable spring bonds, independent of the gun switch."""
    def __init__(self, anchors_wafer_m, profile=ADHESIVE_PROFILE):
        self.anchors=np.asarray(anchors_wafer_m,dtype=float).copy()
        self.profile=profile;self.bound=np.ones(len(self.anchors),dtype=bool)

    def forces(self, positions, velocities, wafer_position, wafer_rotation,
               wafer_velocity, wafer_angular_velocity):
        p=np.asarray(positions);v=np.asarray(velocities);origin=np.asarray(wafer_position)
        offsets=self.anchors@np.asarray(wafer_rotation).T
        targets=origin+offsets
        target_velocity=np.asarray(wafer_velocity)+np.cross(wafer_angular_velocity,offsets)
        elastic=-self.profile['adhesion_stiffness_N_m']*(p-targets)
        trial=elastic-self.profile['adhesion_damping_N_s_m']*(v-target_velocity)
        self.bound &= np.linalg.norm(trial,axis=1)<self.profile['adhesion_break_force_N']
        force=np.where(self.bound[:,None],trial,0.)
        reaction=-force.sum(axis=0)
        torque=-np.cross(p-origin,force).sum(axis=0)
        return force,reaction,torque


def cohesive_forces(bonds,row,profile):
    from wafer_grasp_protocol import _pose
    pose=_pose(row['wafer_pose_world'],'wafer')
    adhesion,reaction,torque=bonds.forces(row['dust_position_world_m'],row['dust_velocity_world_m_s'],
        pose[:3,3],pose[:3,:3],row['dust_wafer_link_velocity_world_m_s'],row['dust_wafer_angular_velocity_world_rad_s'])
    jet=jet_force(row['dust_position_world_m'],row['dust_velocity_world_m_s'],
        row['nozzle_tip_world_m'],row['nozzle_axis_world'],row['air_on'],profile)
    torque-=np.cross(np.asarray(row['dust_wafer_com_world_m'])-pose[:3,3],reaction)
    return adhesion+jet,jet,reaction,torque


class CohesiveDustScore:
    def __init__(self,metadata):
        self.spec=metadata['dust'];self.profile=self.spec['profile'];self.dt=float(metadata['physics_dt_s'])
        self.bonds=SurfaceAdhesion(self.spec['initial_centers_wafer_m'],self.profile)
        self.last=None;self.first_air=None;self.clear_s=0.;self.early_loss=False;self.initial=False;self.pre_return_clear_s=None
        self.error=0.;self.wrench_error=0.;self.other_force=0.;self.intact=True;self.states_match=True
        self.impulse=np.zeros(self.profile['count']);self.count=0;self.remaining=self.profile['count']
        self.return_tracker=MeasuredReturnContact() if self.spec.get('return_detection')=='measured_pin_recontact_v1' else None

    def update(self,row):
        from wafer_grasp_protocol import _pose
        p=np.asarray(row['dust_position_world_m']);pose=_pose(row['wafer_pose_world'],'wafer')
        local=(p-pose[:3,3])@pose[:3,:3];radius=math.sqrt(3)*self.profile['side_m']/2
        flat=-math.sqrt(.05**2-(.0325/2)**2)
        on=(np.linalg.norm(local[:,:2],axis=1)<.05-radius)&(local[:,0]>flat+radius)&(np.abs(local[:,2]-.001)<.002)
        clear=((np.linalg.norm(local[:,:2],axis=1)>.05+radius)|(local[:,0]<flat-radius))&(np.asarray(row['dust_wafer_contact_N'])<1e-8)
        returning=self.return_tracker.update(row) if self.return_tracker is not None else row['phase']=='lower'
        if returning and self.pre_return_clear_s is None:self.pre_return_clear_s=self.clear_s
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


LAYOUT = dict(version='seeded_disk_v2', seed=0, radius_m=.04,
    minimum_spacing_m=.008, minimum_x_m=-.034)


def layout(seed=0):
    rng = np.random.default_rng(seed)
    points = []
    for _ in range(10000):
        xy = rng.uniform(-LAYOUT['radius_m'], LAYOUT['radius_m'], 2)
        if np.linalg.norm(xy) > LAYOUT['radius_m'] or xy[0] < LAYOUT['minimum_x_m']:
            continue
        if any(np.linalg.norm(xy - old[:2]) < LAYOUT['minimum_spacing_m'] for old in points):
            continue
        points.append(np.r_[xy, .001001])
        if len(points) == PROFILE['count']:
            return np.asarray(points)
    raise ValueError('Could not place separated particles on the wafer.')


def jet_force(position,velocity,tip,axis,air_on,profile=PROFILE):
    position=np.asarray(position);velocity=np.asarray(velocity)
    if not air_on:return np.zeros_like(position)
    axis=np.asarray(axis);delta=position-np.asarray(tip);along=delta@axis
    radial=np.linalg.norm(delta-along[:,None]*axis,axis=1)
    width=profile['nozzle_radius_m']+np.maximum(along,0)*math.tan(math.radians(profile['cone_half_angle_deg']))
    speed=profile['reference_speed_m_s']*np.minimum(1.,profile['reference_distance_m']/np.maximum(along,1e-6))*np.exp(-2*(radial/width)**2)
    active=(along>0)&(along<profile['maximum_distance_m'])&(radial<3*width)
    relative=speed[:,None]*axis-velocity
    force=.5*profile['air_density_kg_m3']*profile['drag_coefficient']*profile['side_m']**2*np.linalg.norm(relative,axis=1)[:,None]*relative
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
            friction='.4 .00001 .000001',condim='3',rgba='.95 .015 .025 1',solref='.002 1',solimp='.95 .99 .0001',margin='0',gap='0')
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


class MeasuredReturnContact:
    """First pin contact after a measured 10 mm lift, independent of action labels."""
    def __init__(self):
        self.initial_z=None
        self.lifted=False
        self.returned=False

    def update(self,row):
        z=float(row['wafer_pose_world'][2])
        forces=np.asarray(row['wafer_pillar_normal_force_vectors_world_N'],dtype=float)
        if not math.isfinite(z) or forces.shape!=(3,3) or not np.isfinite(forces).all():
            raise ValueError('Invalid measured wafer return evidence.')
        if self.initial_z is None:self.initial_z=z
        contact=float(np.linalg.norm(forces,axis=1).sum())
        if z-self.initial_z>=.010 and contact<=.005:self.lifted=True
        if self.lifted and contact>.005:self.returned=True
        return self.returned
