"""Passive pliers pickup and transport from a separate stand."""
from pathlib import Path
import argparse, copy, gzip, importlib, json, math, sys
import xml.etree.ElementTree as ET
import numpy as np
from wafer_grasp_protocol import _pose, _angle
from wafer_grasp_feedback import BoundedTCPFeedback,rotation_vector,rotation_matrix,_bounded,PROFILE as FEEDBACK_PROFILE

PROFILE=dict(revision='opposed_handle_seats_v28',stand_offset_world_m=[.11,.08,0.],
    paddle_base_size_m=[.034,.030,.002],
    axial_shoulder_centers_xy_m=[[.004,.004],[.016,.008]],axial_shoulder_size_m=[.002,.003,.005],
    opposing_shoulder_centers_xy_m=[None,[-.0055,.012]],opposing_shoulder_size_m=[.002,.003,.004],
    maximum_translation_correction_m=.010,alignment_integral_gain=.12,alignment_proportional_gain=.1,alignment_orientation_settle_s=.75,docking_acceptance='measured_open_jaw_clearance_v1',
    handle_paddle_stiffness_Nm_rad=.8,handle_paddle_rest_rad=.030,handle_paddle_pivot_shift_m=[.010,-.005],lower_backing_extension_m=.008,
    upper_paddle_joint_anchor_local_m=[0.,0.,0.],
    handle_contact_land_size_m=[.016,.006,.001],handle_contact_land_x_m=[-.005,.005],handle_contact_land_y_m=[0.,.008],
    upper_fore_aft_shoulder_center_body_m=[.011,.004,-.0009],upper_fore_aft_shoulder_size_m=[.002,.003,.0008],
    finger_seat_endstop_y_m=.0145,finger_seat_endstop_size_m=[.012,.001,.002],
    handle_center_shift_tool_m=[.0144,-.006,0.],approach_distance_m=.050,lift_m=.040,behind_flat_m=.025,
    open_finger_m=-.035,transport_finger_m=-.0245,handoff_finger_m=-.0245,
    minimum_transport_gap_m=.002,maximum_relative_slip_m=.002,
    maximum_relative_rotation_deg=3.,minimum_finger_contact_N=.05,
    maximum_unexpected_contact_N=1.,maximum_docking_tangential_error_m=.001,
    maximum_docking_angle_deg=2.,minimum_50ms_mean_finger_force_N=.75,maximum_contact_gap_s=.001,
    maximum_feedback_rotation_deg=4.,alignment_hold_s=6.,close_duration_s=8.,minimum_insertion_clearance_m=.0001,
    translation_only_phases=[])

PADDLE_CAPACITY_PROFILE=dict(revision='passive_drive_operating_envelope_v2',joint_limit_allowance_rad=.001,
    torque_law_tolerance_Nm=1e-7,acceptance_fields=['limits_rad','velocity_guard_rad_s','operating_torque_guard_Nm'],
    legacy_warning_field='torque_guard_Nm',
    numerical_drive_cap='100 mNm prevents clipping the full-range linear spring; accepted raw torque remains at most 60 mNm. Historical specifications without an operating guard use their authored force cap.',
    rationale='A 20 g wafer about 150 mm ahead of the grip contributes about 29.5 mNm gravity moment. The 25 mNm diagnostic is below this ordinary load; the accepted operating torque limit is 60 mNm. Contact, retention and task checks remain independent.')

LOCAL_CONTACT_SOLVER_PROFILE=dict(revision='full_workflow_verified_baseline_v2',impratio=10.,noslip_iterations=0,
    scope='Restores the numerical settings of the completed v12 pickup and 150-second loaded hold. Geometry, friction coefficients, springs, actuator limits and timestep are unchanged; PhysX retains its own solver.')

TOOL_TIMING_PROFILE=dict(revision='full_workflow_120s_tool_v1',
    phases_s=dict(tool_settle=1.5,tool_approach=1.5,tool_close=4.5,tool_grip_hold=1.,
        tool_lift=1.5,tool_lift_hold=1.,tool_transport=2.,tool_transport_hold=1.,
        tool_lower_behind=1.5,tool_realign=0.,tool_align_hold=2.,tool_insert=2.,tool_dock=2.),
    closure=dict(free_travel_s=1.5,baseline_duration_s=8.,baseline_tail_start_s=5.),
    acceptance_hold_s=1.,scope='Faster clear-space motion; the original final three seconds of finger closure are unchanged.')


def tool_timing(metadata):
    profile=metadata['tool_pickup'].get('timing_profile')
    if profile is None:
        return dict(tool_settle=2.,tool_approach=4.,tool_close=PROFILE['close_duration_s'],tool_grip_hold=2.,
            tool_lift=4.,tool_lift_hold=2.,tool_transport=6.,tool_transport_hold=2.,tool_lower_behind=4.,
            tool_realign=2.,tool_align_hold=PROFILE['alignment_hold_s'],tool_insert=6.,tool_dock=3.)
    if profile!=TOOL_TIMING_PROFILE:raise ValueError('Unknown tool timing profile.')
    return dict(profile['phases_s'])


def tool_pickup_duration(metadata,*,stop_after_lift=False,stop_after_seating=False):
    total=0.;last='tool_grip_hold'if stop_after_seating else'tool_lift_hold'if stop_after_lift else'tool_dock'
    for phase,duration in tool_timing(metadata).items():
        total+=duration
        if phase==last:return total
    raise ValueError('Missing tool timing endpoint.')


def tool_finger_target(phase,elapsed_s,duration_s,initial_fingers,target,metadata):
    s=float(np.clip(elapsed_s/duration_s,0.,1.));u=s*s*(3.-2.*s)
    profile=metadata['tool_pickup'].get('timing_profile')
    if phase=='tool_close'and profile is not None:
        tool_timing(metadata);close=profile['closure'];free=close['free_travel_s'];base=close['baseline_duration_s'];tail=close['baseline_tail_start_s']
        if abs(duration_s-(free+base-tail))>1e-12:raise ValueError('Tool closure duration differs.')
        if elapsed_s<=free:
            a=tail/base;value=a*a*(3.-2.*a);slope=6.*a*(1.-a)/base;t=elapsed_s/free
            u=(3.*t*t-2.*t*t*t)*value+(t*t*t-t*t)*free*slope
        else:
            t=min(1.,(tail+elapsed_s-free)/base);u=t*t*(3.-2.*t)
    return np.asarray(initial_fingers)+(target-np.asarray(initial_fingers))*u


def jaw_normal_center_error(metadata,row):
    wafer=_pose(row['wafer_pose_world'],'wafer');normal=wafer[:3,2];levels=[]
    for pose,face in zip(row['jaw_link_poses_world'],metadata['contact_faces_local_m']):
        pad=_pose(pose,'pad');points=np.asarray(face)@pad[:3,:3].T+pad[:3,3]
        levels.append((points-wafer[:3,3])@normal)
    return -float((levels[0].max()+levels[1].min())/2)


class ToolPickupFeedback(BoundedTCPFeedback):
    def __init__(self,fk,solve,lower,upper,*,tool_to_tcp):
        super().__init__(fk,solve,lower,upper);self.tool_to_tcp=np.asarray(tool_to_tcp).copy();self.previous_correction_error=None;self.dock_hold_command=None

    def step(self,nominal_q,measured_p,measured_R,previous_command_q,*,integrate=True,integrate_rotation=True,normal_axis=None,normal_error_m=None,normal_only=True,damped_translation=False,rotation_pivot_world=None,settle_hold=False):
        nominal=np.asarray(nominal_q);previous=np.asarray(previous_command_q)
        p,r=self.fk(nominal);relation=self.tool_to_tcp
        tool_r=r@relation[:3,:3].T;tool_p=p-tool_r@relation[:3,3]
        error=tool_p-np.asarray(measured_p);angular=rotation_vector(tool_r@np.asarray(measured_R).T)
        correction_error=error.copy()
        if normal_axis is not None:
            normal_axis=np.asarray(normal_axis,dtype=float)
            if normal_axis.shape!=(3,)or not np.isfinite(normal_axis).all()or abs(np.linalg.norm(normal_axis)-1.)>1e-5:
                raise ValueError('Insertion normal must be a measured unit vector.')
            if normal_error_m is not None and not math.isfinite(float(normal_error_m)):raise ValueError('Nonfinite measured jaw center error.')
            correction_error=normal_axis*float(error@normal_axis if normal_error_m is None else normal_error_m)
            if normal_error_m is not None and abs(normal_error_m)<=.00005:correction_error[:]=0.
            if not normal_only:correction_error+=error-normal_axis*float(error@normal_axis)
        if not settle_hold:self.dock_hold_command=None
        elif normal_error_m is None:raise ValueError('Dock hold requires measured jaw centering.')
        elif np.linalg.norm(error)>.0009 or np.linalg.norm(angular)>math.radians(.8) or abs(normal_error_m)>.00014:
            self.dock_hold_command=None
        elif self.dock_hold_command is None and np.linalg.norm(error)<=.0008 and np.linalg.norm(angular)<=math.radians(.5) and abs(normal_error_m)<=.00012:
            self.dock_hold_command=previous.copy()
        if self.dock_hold_command is not None:integrate=False;integrate_rotation=False
        old_rotation=self.rotation.copy();old_translation=self.translation.copy()
        if integrate:
            increment=.2*correction_error
            if damped_translation:
                increment=PROFILE['alignment_integral_gain']*correction_error
                if self.previous_correction_error is not None:
                    increment+=PROFILE['alignment_proportional_gain']*(correction_error-self.previous_correction_error)
            if np.linalg.norm(correction_error)>FEEDBACK_PROFILE['translation_deadband_m'] or damped_translation:
                self.translation+=_bounded(increment,FEEDBACK_PROFILE['maximum_translation_increment_m'])
            if integrate_rotation and np.linalg.norm(angular)>FEEDBACK_PROFILE['rotation_deadband_rad']:
                self.rotation+=_bounded(.2*angular,FEEDBACK_PROFILE['maximum_rotation_increment_rad'])
        self.rotation=_bounded(self.rotation,math.radians(PROFILE['maximum_feedback_rotation_deg']))
        if rotation_pivot_world is not None and integrate_rotation and normal_axis is not None:
            lever=np.asarray(rotation_pivot_world)-np.asarray(measured_p)
            delta=rotation_matrix(self.rotation)@rotation_matrix(old_rotation).T
            self.translation-=normal_axis*float(normal_axis@(delta@lever-lever))
        self.translation=old_translation+_bounded(self.translation-old_translation,FEEDBACK_PROFILE['maximum_translation_increment_m'])
        self.previous_correction_error=correction_error.copy()
        self.translation=_bounded(self.translation,PROFILE['maximum_translation_correction_m']);self.rotation=_bounded(self.rotation,math.radians(PROFILE['maximum_feedback_rotation_deg']))
        corrected_r=rotation_matrix(self.rotation)@tool_r
        target_p=tool_p+self.translation+corrected_r@relation[:3,3];target_r=corrected_r@relation[:3,:3]
        if self.dock_hold_command is None:q,_,_=self.solve(target_p,target_r,previous)
        else:
            q=self.dock_hold_command.copy();target_p,target_r=self.fk(q)
        q=np.asarray(q);actual_p,actual_r=self.fk(q)
        pe=float(np.linalg.norm(actual_p-target_p));ae=float(np.linalg.norm(rotation_vector(target_r@actual_r.T)))
        increment=float(np.max(np.abs(q-previous)))
        if not np.isfinite(q).all()or np.any(q<self.lower)or np.any(q>self.upper)or pe>.00002 or ae>math.radians(.01):
            raise RuntimeError('Tool feedback IK residual or joint limit failed.')
        if increment>math.radians(1.):raise RuntimeError('Tool feedback exceeds one degree per command.')
        self.command_count+=1
        return q,dict(command_index=self.command_count,stationary_command_hold=self.dock_hold_command is not None,integrated=bool(integrate),rotation_integrated=bool(integrate and integrate_rotation),nominal_q_rad=nominal.tolist(),
            measured_position_error_m=error.tolist(),measured_orientation_error_rad=angular.tolist(),
            normal_axis_world=None if normal_axis is None else normal_axis.tolist(),
            measured_jaw_center_error_m=normal_error_m,normal_only=normal_only,damped_translation=bool(damped_translation),
            translation_correction_world_m=self.translation.tolist(),rotation_correction_world_rad=self.rotation.tolist(),
            ik_position_residual_m=pe,ik_orientation_residual_rad=ae,maximum_joint_increment_rad=increment)


def simplify_handle_geometry(root,metadata):
    removed=[];mass_changes=[]
    for half in ('lower','upper'):
        body=root.find('.//body[@name="'+half+'_half"]');handle=body.find('geom[@name="'+half+'_robot_handle"]')
        transferred=0.
        for index in (0,1):
            name=half+'_handle_side_rail_'+str(index);geom=body.find('geom[@name="'+name+'"]')
            if geom is None:raise ValueError('Missing lateral guide: '+name)
            transferred+=float(geom.get('mass'));body.remove(geom);removed.append(name)
        handle.set('mass',format(float(handle.get('mass'))+transferred,'.17g'))
        mass_changes.append(dict(body=half+'_half',mass_to_broad_handle_kg=transferred))
    metadata['geometry']['primitive_parts_original_tool_frame']=[p for p in metadata['geometry']['primitive_parts_original_tool_frame']if p['name']not in removed]
    metadata['full_cycle_handle_geometry']=dict(revision='broad_handles_no_side_rails_v1',removed_geoms=removed,
        mass_redistribution=mass_changes,total_mass_kg=metadata['tool_mass_kg'],
        mass_rule='Lateral-guide mass redistributed into each physical broad handle; COM and inertia recomputed.')
    return removed


def configure_handle_paddles(root,metadata):
    pivot=np.asarray(metadata['geometry']['tool_original_pivot_m']);parts=metadata['geometry']['primitive_parts_original_tool_frame']
    assets=root.find('asset');specs=[]
    for half,sign in (('lower',1.),('upper',-1.)):
        parent=root.find('.//body[@name="'+half+'_half"]');name=half+'_robot_handle';geom=parent.find('geom[@name="'+name+'"]')
        part=next(p for p in parts if p['name']==name);part['name']=half+'_handle_backing';part['depth']=.002
        if half=='lower':part['outline']=[[x+(PROFILE['lower_backing_extension_m']if x>0 else 0.),y]for x,y in part['outline']]
        geom.set('name',part['name']);geom.set('mass',format(float(geom.get('mass'))-.010,'.17g'))
        vertices=np.array([(x,y,z)for z in(-.001,.001)for x,y in part['outline']])
        assets.find('mesh[@name="'+geom.get('mesh')+'"]').set('vertex',' '.join(format(v,'.12g')for v in vertices.ravel()))
        index=0 if half=='lower'else 1
        position=[-.07245727892431036+PROFILE['handle_center_shift_tool_m'][0]+PROFILE['handle_paddle_pivot_shift_m'][index],PROFILE['handle_center_shift_tool_m'][1],sign*.0237];body_name=half+'_grip_paddle';joint_name=body_name+'_pitch'
        anchor=np.asarray(PROFILE['upper_paddle_joint_anchor_local_m']if half=='upper'else[0.,0.,0.])
        position=(np.asarray(position)+(rotation_matrix(np.array([0.,PROFILE['handle_paddle_rest_rad'],0.]))-np.eye(3))@anchor).tolist()
        body=ET.SubElement(parent,'body',name=body_name,pos=' '.join(map(str,position)))
        ET.SubElement(body,'joint',name=joint_name,type='hinge',pos=' '.join(map(str,anchor)),axis='0 1 0',range='-.08 .08',stiffness=str(PROFILE['handle_paddle_stiffness_Nm_rad']),springref=str(PROFILE['handle_paddle_rest_rad']),damping='.001')
        base_size=np.asarray(PROFILE['paddle_base_size_m']);base_center=np.array([0.,0.,-sign*.0005])
        land_size=np.asarray(PROFILE['handle_contact_land_size_m']);land_centers=[np.array([PROFILE['handle_contact_land_x_m'][index],y,sign*.001])for y in PROFILE['handle_contact_land_y_m']]
        centers=[base_center,*land_centers];sizes=[base_size,*[land_size for _ in land_centers]];shoulders=[]
        collider_names=[name,*[half+'_handle_side_rail_'+str(i)for i in range(len(land_centers))]]
        if half=='upper':
            centers.append(np.asarray(PROFILE['upper_fore_aft_shoulder_center_body_m']));sizes.append(np.asarray(PROFILE['upper_fore_aft_shoulder_size_m']));collider_names.append('upper_handle_rib_0')
            shoulders.append(dict(name=collider_names[-1],center_body_m=centers[-1].tolist(),size_m=sizes[-1].tolist()))
        centers.append(np.array([PROFILE['handle_contact_land_x_m'][index],PROFILE['finger_seat_endstop_y_m'],sign*.0015]))
        stop_size=np.asarray(PROFILE['finger_seat_endstop_size_m']).copy();stop_size[2]=.004
        sizes.append(stop_size);collider_names.append(half+'_handle_endstop')
        shoulders.append(dict(name=collider_names[-1],center_body_m=centers[-1].tolist(),size_m=sizes[-1].tolist()))
        centers.append(np.array([*PROFILE['axial_shoulder_centers_xy_m'][index],sign*.002]))
        sizes.append(np.asarray(PROFILE['axial_shoulder_size_m']));collider_names.append(half+'_handle_axial_stop')
        shoulders.append(dict(name=collider_names[-1],center_body_m=centers[-1].tolist(),size_m=sizes[-1].tolist()))
        counterstop=PROFILE['opposing_shoulder_centers_xy_m'][index]
        if counterstop is not None:
            size=np.asarray(PROFILE['opposing_shoulder_size_m'])
            centers.append(np.array([*counterstop,sign*(size[2]/2-.0005)]))
            sizes.append(size);collider_names.append(half+'_handle_opposing_stop')
            shoulders.append(dict(name=collider_names[-1],center_body_m=centers[-1].tolist(),size_m=size.tolist()))
        volumes=[float(np.prod(size))for size in sizes];total_volume=sum(volumes)
        for collider,center,size,volume in zip(collider_names,centers,sizes,volumes):
            ET.SubElement(body,'geom',name=collider,type='box',pos=' '.join(map(str,center)),size=' '.join(map(str,size/2)),mass=format(.010*volume/total_volume,'.17g'),
                rgba='.30 .32 .34 1',friction=geom.get('friction','.6 .0001 .0001'),contype=geom.get('contype','1'),conaffinity=geom.get('conaffinity','1'))
            parts.append(dict(name=collider,kind='box',center=(pivot+position+center).tolist(),size=size.tolist(),color=[.30,.32,.34],group=body_name,rotate_y=0.))
        spec=dict(body_name=body_name,joint_name=joint_name,parent_body=half+'_half',collider_name=name,collider_names=collider_names,
            base_size_m=base_size.tolist(),base_center_body_m=base_center.tolist(),contact_land_size_m=land_size.tolist(),contact_land_centers_body_m=[p.tolist()for p in land_centers],registration_shoulders=shoulders,
            position_parent_m=position,joint_anchor_local_m=anchor.tolist(),
            quaternion_parent_xyzw=[0.,0.,0.,1.],axis_local=[0.,1.,0.],limits_rad=[-.08,.08],stiffness_Nm_rad=PROFILE['handle_paddle_stiffness_Nm_rad'],
            damping_Nms_rad=.001,rest_rad=PROFILE['handle_paddle_rest_rad'],mass_kg=.010,size_m=[*base_size[:2],.003],force_cap_Nm=.100,operating_torque_guard_Nm=.060,
            torque_guard_Nm=.025,velocity_guard_rad_s=10.,velocity_limit_rad_s=20.)
        specs.append(spec)
    metadata['handle_paddles']=specs;metadata['geometry']['body_names']+=list(s['body_name']for s in specs)
    metadata['full_workflow_no_wafer_side_cradle']=True
    metadata['handle_paddle_capacity_profile']=dict(PADDLE_CAPACITY_PROFILE)
    metadata['full_cycle_handle_geometry'].update(revision=PROFILE['revision'],
        backing_depth_m=.002,paddle_mass_from_each_backing_kg=.010,
        lower_backing_extension_m=PROFILE['lower_backing_extension_m'],paddle_pivot_shift_m=list(PROFILE['handle_paddle_pivot_shift_m']),
        sizing='Connected edge and axial shoulders locate the fingers. Each paddle retains 10 g distributed by physical volume.',
        topology='Two unpowered pitch hinges on physical handle paddles; robot fingers provide all squeezing force.')
    return specs


def guard_handle_paddle_capacity(row,metadata):
    """Physical operating envelope; the legacy torque warning is evaluated separately."""
    try:values=np.asarray([row[k]for k in ('handle_paddle_q_rad','handle_paddle_qvel_rad_s','handle_paddle_passive_torque_Nm')],dtype=float)
    except (KeyError,TypeError,ValueError)as exc:raise RuntimeError('Incomplete passive handle-paddle measurements.')from exc
    if values.shape!=(3,2)or not np.isfinite(values).all():raise RuntimeError('Incomplete passive handle-paddle measurements.')
    if len(metadata['handle_paddles'])!=2:raise RuntimeError('Expected two passive handle paddles.')
    for spec,q,v,torque in zip(metadata['handle_paddles'],row['handle_paddle_q_rad'],row['handle_paddle_qvel_rad_s'],row['handle_paddle_passive_torque_Nm']):
        expected=-spec['stiffness_Nm_rad']*(q-spec['rest_rad'])-spec['damping_Nms_rad']*v
        limit=min(spec['force_cap_Nm'],spec.get('operating_torque_guard_Nm',spec['force_cap_Nm']))
        if not np.isfinite(limit)or limit<=0:raise RuntimeError('Invalid passive handle-paddle operating torque limit.')
        if abs(torque-expected)>PADDLE_CAPACITY_PROFILE['torque_law_tolerance_Nm']or not spec['limits_rad'][0]-.001<=q<=spec['limits_rad'][1]+.001 or abs(v)>spec['velocity_guard_rad_s']or abs(torque)>limit:
            raise RuntimeError(f"{spec['joint_name']}: pitch {q:.6f} rad, speed {v:.6f} rad/s, torque {torque:.8f} Nm exceeds the authored operating envelope.")


def guard_handle_paddles(row,metadata):
    """Original stricter diagnostic; historical acceptance remains unchanged."""
    guard_handle_paddle_capacity(row,metadata)
    for spec,q,v,torque in zip(metadata['handle_paddles'],row['handle_paddle_q_rad'],row['handle_paddle_qvel_rad_s'],row['handle_paddle_passive_torque_Nm']):
        if abs(torque)>spec['torque_guard_Nm']:
            raise RuntimeError(f"{spec['joint_name']}: pitch {q:.6f} rad, speed {v:.6f} rad/s, torque {torque:.8f} Nm exceeds a passive-paddle guard.")


def guard_handle_paddles_for_qualification(row,metadata):
    if metadata.get('task_qualification',{}).get('version')=='measured_full_task_v1':
        guard_handle_paddle_capacity(row,metadata)
    else:guard_handle_paddles(row,metadata)


def _pose_body(data,name):
    body=data.body(name);q=body.xquat
    return [*body.xpos.tolist(),*q[1:].tolist(),float(q[0])]


def _json_default(value):
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,np.ndarray):return value.tolist()
    raise TypeError('Unsupported report value: '+type(value).__name__)


def _bottom_at(model,data,name,xy):
    geom=model.geom(name);mesh=int(model.geom_dataid[geom.id])
    start=int(model.mesh_vertadr[mesh]);count=int(model.mesh_vertnum[mesh])
    vertices=model.mesh_vert[start:start+count]@data.geom_xmat[geom.id].reshape(3,3).T+data.geom_xpos[geom.id]
    first=int(model.mesh_faceadr[mesh]);faces=model.mesh_face[first:first+model.mesh_facenum[mesh]]
    hits=[]
    for ids in faces:
        triangle=vertices[ids];basis=np.c_[triangle[1,:2]-triangle[0,:2],triangle[2,:2]-triangle[0,:2]]
        if abs(np.linalg.det(basis))<1e-12:continue
        u,v=np.linalg.solve(basis,np.asarray(xy)-triangle[0,:2])
        if u>=-1e-8 and v>=-1e-8 and u+v<=1+1e-8:
            hits.append(float(triangle[0,2]+u*(triangle[1,2]-triangle[0,2])+v*(triangle[2,2]-triangle[0,2])))
    if not hits:raise ValueError('Tool stand has no handle surface above it.')
    return min(hits)


def prepare_scene(model,data,metadata,xml):
    import mujoco
    import wafer_cycle as cycle
    if abs(data.time)>1e-12:raise ValueError('Tool layout can only be initialized before physics starts.')
    metadata=copy.deepcopy(metadata);root=ET.fromstring(xml);world=root.find('worldbody')
    if float(model.opt.impratio)!=LOCAL_CONTACT_SOLVER_PROFILE['impratio']:raise ValueError('Unexpected parent friction impedance ratio.')
    option=root.find('option')
    if option is None:raise ValueError('Missing explicit parent simulation options.')
    option.set('noslip_iterations',str(LOCAL_CONTACT_SOLVER_PROFILE['noslip_iterations']))
    simplify_handle_geometry(root,metadata)
    configure_handle_paddles(root,metadata)
    target=_pose(_pose_body(data,'lower_half'),'tool');delta=np.asarray(PROFILE['stand_offset_world_m'])
    body=world.find("body[@name='lower_half']");body.set('pos',cycle.pickup.vector(target[:3,3]+delta))
    original_q=np.asarray(metadata['initial_arm_q_rad']);target_p,target_r=cycle.ik.tcp_pose(original_q,cycle.station.BASES['left'])
    target_p=target_p+target[:3,:3]@np.asarray(PROFILE['handle_center_shift_tool_m'])
    grip_p=target_p+delta;grip_q=cycle.solve_pose(grip_p,target_r,original_q)
    start_q=cycle.solve_pose(grip_p-target_r[:,2]*PROFILE['approach_distance_m'],target_r,grip_q)
    temp_model=mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'));temp_data=mujoco.MjData(temp_model)
    for joint in range(model.njnt):
        source=model.joint(joint);dest=temp_model.joint(source.name);free=int(model.jnt_type[joint])==0
        nq,nv=(7,6)if free else(1,1)
        temp_data.qpos[dest.qposadr[0]:dest.qposadr[0]+nq]=data.qpos[source.qposadr[0]:source.qposadr[0]+nq]
        temp_data.qvel[dest.dofadr[0]:dest.dofadr[0]+nv]=data.qvel[source.dofadr[0]:source.dofadr[0]+nv]
    temp_data.ctrl[:]=data.ctrl
    address=temp_model.jnt_qposadr[temp_model.body('lower_half').jntadr[0]]
    temp_data.qpos[address:address+3]=target[:3,3]+delta
    temp_data.qpos[temp_model.joint('pliers_hinge').qposadr[0]]=-.04
    for spec in metadata['handle_paddles']:
        temp_data.qpos[temp_model.joint(spec['joint_name']).qposadr[0]]=spec['rest_rad']
    mujoco.mj_forward(temp_model,temp_data)
    hinge_support=copy.deepcopy(world.find("geom[@name='clamp_cradle']"))
    hinge_support.set('name','tool_stand_hinge')
    hinge_support.set('pos',cycle.pickup.vector(np.fromstring(hinge_support.get('pos'),sep=' ')+delta))
    world.append(hinge_support)
    world.remove(world.find("geom[@name='clamp_cradle']"))
    handle_center=target[:3,3]+delta+target[:3,:3]@np.array([-.125,-.008,0.])
    top=_bottom_at(temp_model,temp_data,'upper_handle_backing',handle_center[:2])
    height=top-cycle.station.TABLE_TOP
    if height<=.005:raise ValueError('Tool stand cannot fit above the table.')
    ET.SubElement(world,'geom',name='tool_stand_handle',type='box',
        pos=cycle.pickup.vector([*handle_center[:2],cycle.station.TABLE_TOP+height/2]),
        size=cycle.pickup.vector([.008,.002,height/2]),rgba='.30 .32 .34 1',
        contype='2',conaffinity='7',friction='.3 .0001 .0001')
    text=ET.tostring(root,encoding='unicode');result=mujoco.MjModel.from_xml_string(text);state=mujoco.MjData(result)
    if result.opt.noslip_iterations!=LOCAL_CONTACT_SOLVER_PROFILE['noslip_iterations'] or result.opt.impratio!=model.opt.impratio or result.opt.noslip_tolerance!=model.opt.noslip_tolerance or result.opt.timestep!=model.opt.timestep:
        raise ValueError('Full-workflow numerical solver profile differs.')
    metadata['local_contact_solver_profile']=dict(LOCAL_CONTACT_SOLVER_PROFILE,engine='MuJoCo',engine_version=mujoco.__version__,
        impratio=float(result.opt.impratio),noslip_iterations=int(result.opt.noslip_iterations),noslip_tolerance=float(result.opt.noslip_tolerance),
        timestep_s=float(result.opt.timestep),parent_noslip_iterations=int(model.opt.noslip_iterations))
    state.qpos[:]=temp_data.qpos;state.qvel[:]=temp_data.qvel;state.ctrl[:]=temp_data.ctrl
    for i,v in enumerate([*start_q,PROFILE['open_finger_m'],PROFILE['open_finger_m']],1):
        state.qpos[result.joint('joint'+str(i)).qposadr[0]]=v
        state.ctrl[result.actuator('drive_joint'+str(i)).id]=v
    mujoco.mj_forward(result,state)
    for name in ('lower_half','upper_half','lower_grip_paddle','upper_grip_paddle'):
        b=result.body(name);q=b.iquat
        metadata['body_inertials'][name]=dict(mass_kg=float(b.mass[0]),center_of_mass_local_m=b.ipos.tolist(),
            diagonal_inertia_kg_m2=b.inertia.tolist(),principal_axes_xyzw=[*q[1:].tolist(),float(q[0])])
    if result.neq or np.any(state.xfrc_applied):raise ValueError('No attachment or applied wrench is permitted.')
    contact=cycle.pickup.read_robot_contact_diagnostics(result,state)
    if any(c['signed_distance_m']<-.00001 for c in contact['contact_details']):
        raise ValueError('Empty-hand starting pose intersects an object.')
    metadata['tool_pickup']=dict(profile=dict(PROFILE),timing_profile=copy.deepcopy(TOOL_TIMING_PROFILE),target_tool_pose_world=[*target[:3,3],*cycle.ik.matrix_to_quat_xyzw(target[:3,:3])],
        target_tcp_position_world_m=target_p.tolist(),target_tcp_rotation=target_r.tolist(),
        stand_grasp_arm_q_rad=grip_q.tolist(),start_arm_q_rad=start_q.tolist(),wafer_arm_q_rad=original_q.tolist(),
        initial_tool_pose_world=_pose_body(state,'lower_half'),stand_geoms=['tool_stand_hinge','tool_stand_handle'],
        initial_empty_hand_contact_force_N=contact['finger_handle_contact_normal_force_N'],
        initial_passive_hinge_rad=-.04,stand_shapes=[dict(name=g.get('name'),type=g.get('type'),position_world_m=np.fromstring(g.get('pos'),sep=' ').tolist(),size_m=np.fromstring(g.get('size'),sep=' ').tolist())for g in world if g.get('name')in ('tool_stand_hinge','tool_stand_handle')],
        post_start_pose_write_count=0,stage_clock='Global physics time is continuous; handoff uses a declared stage-local clock.')
    metadata['tool_pickup']['initial_joint_positions']={name:float(state.qpos[result.joint(name).qposadr[0]])for name in
        ['pliers_hinge','lower_pad_compression','upper_pad_compression',*[s['joint_name']for s in metadata['handle_paddles']]]}
    metadata['initial_condition']='Both hands empty; passive pliers rest on a separate stand, wafer rests on three pillars.'
    return result,state,metadata,text


def tool_stand_contact_allowed(phase):
    return phase in ('tool_initialization','tool_settle','tool_approach','tool_close','tool_grip_hold','tool_lift')


def read_tool_environment_contacts(model,data,metadata,phase):
    import mujoco
    stand=0.;external={}
    tool_names={p['name']for p in metadata['geometry']['primitive_parts_original_tool_frame']}
    for i,c in enumerate(data.contact):
        names=[model.geom(c.geom1).name,model.geom(c.geom2).name]
        if sum(n in tool_names for n in names)!=1:continue
        part=next(n for n in names if n in tool_names);other=next(n for n in names if n not in tool_names)
        if other.startswith(('robot_','right_robot_','wafer_'))or other=='wafer':continue
        force=np.zeros(6);mujoco.mj_contactForce(model,data,i,force);normal=max(0.,float(force[0]))
        if other in metadata['tool_pickup']['stand_geoms']:
            stand+=normal
            if tool_stand_contact_allowed(phase):continue
        if normal>0.:external[(part,other)]=external.get((part,other),0.)+normal
    return dict(tool_stand_normal_force_N=stand,tool_unexpected_environment_contact_normal_force_N=sum(external.values()),
        tool_unexpected_environment_contacts_N=[[a,b,f]for(a,b),f in sorted(external.items())])


def measurement(model,data,metadata,phase):
    import wafer_cycle as cycle
    mechanism=importlib.import_module(metadata['mechanism_module'])
    row=mechanism.read_fixture_state(model,data,metadata,phase)
    row.update(cycle.pickup.read_robot_contact_diagnostics(model,data));row.pop('contact_details',None)
    row['measured_hand_pose_world']=_pose_body(data,'gripper')
    row['measured_finger_q_m']=[float(data.qpos[model.joint('joint'+str(i)).qposadr[0]])for i in (7,8)]
    row['commanded_arm_q_rad']=[float(data.ctrl[model.actuator('drive_joint'+str(i)).id])for i in range(1,7)]
    row['commanded_finger_q_m']=[float(data.ctrl[model.actuator('drive_joint'+str(i)).id])for i in (7,8)]
    row.update(read_tool_environment_contacts(model,data,metadata,phase))
    row['handle_paddle_poses_world']=[_pose_body(data,s['body_name'])for s in metadata['handle_paddles']]
    row['handle_paddle_q_rad']=[float(data.qpos[model.joint(s['joint_name']).qposadr[0]])for s in metadata['handle_paddles']]
    row['handle_paddle_qvel_rad_s']=[float(data.qvel[model.joint(s['joint_name']).dofadr[0]])for s in metadata['handle_paddles']]
    row['handle_paddle_passive_torque_Nm']=[-s['stiffness_Nm_rad']*(q-s['rest_rad'])-s['damping_Nms_rad']*v for s,q,v in zip(metadata['handle_paddles'],row['handle_paddle_q_rad'],row['handle_paddle_qvel_rad_s'])]
    return row



def _transform(value,name):
    a=np.asarray(value,dtype=float)
    return a.copy()if a.shape==(4,4)else _pose(value,name)


def insertion_clearance(metadata,row):
    wafer=_pose(row['wafer_pose_world'],'wafer');tool=_pose(row['clamp_pose_world'],'tool')
    target=np.asarray(metadata['tool_pickup']['target_tool_pose_world'][:3]);shift=target-tool[:3,3]
    shift-=wafer[:3,2]*float(shift@wafer[:3,2])
    local_shift=wafer[:3,:3].T@shift;half=metadata['wafer_thickness_m']/2
    faces=[];bounds=[];pivot=np.asarray(metadata['geometry']['tool_original_pivot_m'])
    corners=np.array([[x,y,z]for x in(-1,1)for y in(-1,1)for z in(-1,1)])
    polygon=np.asarray(metadata['wafer_collision_partition']['original_outline_m'])
    wafer_low=np.r_[polygon.min(0),-half];wafer_high=np.r_[polygon.max(0),half]
    for pose,face in zip(row['jaw_link_poses_world'],metadata['contact_faces_local_m']):
        link=_pose(pose,'pad');v=(np.asarray(face)@link[:3,:3].T+link[:3,3]-wafer[:3,3])@wafer[:3,:3]
        faces.append(np.vstack([v,v+local_shift]))
    for part in metadata['geometry']['primitive_parts_original_tool_frame']:
        name=part['name'];kind=part['kind']
        if kind=='box':v=corners*np.asarray(part['size'])/2
        elif kind=='cylinder':
            size=np.repeat(part['radius'],3);size['XYZ'.index(part.get('axis','Z'))]=part['height']/2;v=corners*size
        elif kind=='prism':
            v=np.array([(x,y,z)for z in(-part['depth']/2,part['depth']/2)for x,y in part['outline']])
            if part.get('plane','XY')=='XZ':v=v[:,[0,2,1]]*[1,-1,1]
        else:raise ValueError('Unknown tool primitive: '+kind)
        index=1 if name.startswith('upper_')else 0
        if name.endswith('_contact_pad'):link=_pose(row['jaw_link_poses_world'][index],'pad')
        elif part.get('group','').endswith('_grip_paddle'):
            spec=metadata['handle_paddles'][index]
            local=_pose([*spec['position_parent_m'],*spec['quaternion_parent_xyzw']],'paddle origin')
            v=(v@rotation_matrix(np.array([0.,math.radians(part.get('rotate_y',0.)),0.])).T+np.asarray(part['center'])-pivot-local[:3,3])@local[:3,:3]
            link=_pose(row['handle_paddle_poses_world'][index],'grip paddle')
        else:
            v=v@rotation_matrix(np.array([0.,math.radians(part.get('rotate_y',0.)),0.])).T+np.asarray(part['center'])-pivot
            link=_pose(row['lever_link_poses_world'][index],'lever')
        v=(v@link[:3,:3].T+link[:3,3]-wafer[:3,3])@wafer[:3,:3]
        swept=np.vstack([v,v+local_shift]);low=swept.min(0);high=swept.max(0)
        separation=float(np.max(np.maximum(wafer_low-high,low-wafer_high)))
        bounds.append(dict(name=name,separating_axis_clearance_m=separation))
    lower=-half-float(faces[0][:,2].max());upper=float(faces[1][:,2].min())-half
    margin=PROFILE['minimum_insertion_clearance_m']
    return dict(passed=bool(min(lower,upper)>=margin and min(b['separating_axis_clearance_m']for b in bounds)>=margin),
        lower_face_z_range_wafer_m=[float(faces[0][:,2].min()),float(faces[0][:,2].max())],
        upper_face_z_range_wafer_m=[float(faces[1][:,2].min()),float(faces[1][:,2].max())],
        lower_clearance_m=lower,upper_clearance_m=upper,required_clearance_m=margin,swept_part_bounds=bounds,
        scope='Conservative tangential insertion sweep at the measured jaw-centered height; actual motion remains contact-guarded.')


def plan_tool_pickup(metadata,read_tool_pose,read_tcp_pose,read_command_q,solve_pose,slerp,controller_state,*,stop_after_lift=False,stop_after_seating=False,read_state=None):
    """20 Hz Cartesian waypoints shared by both physics backends."""
    spec=metadata['tool_pickup'];target=_pose(spec['target_tool_pose_world'],'target')
    grip=np.asarray(spec['stand_grasp_arm_q_rad']);start=np.asarray(spec['start_arm_q_rad'])
    carry=PROFILE['transport_finger_m'];opened=PROFILE['open_finger_m'];timing=tool_timing(metadata)
    for phase,q,finger in [('tool_settle',start,opened),('tool_approach',grip,opened),
                 ('tool_close',grip,carry),('tool_grip_hold',grip,carry)]:yield phase,timing[phase],q,finger
    if stop_after_seating:return
    desired=_transform(read_tool_pose(),'tool');relation=np.linalg.inv(desired)@_transform(read_tcp_pose(),'tcp')
    controller_state.update(tool_to_tcp=relation.copy(),relation_revision=1)
    q=np.asarray(read_command_q());last_nominal=q.copy()
    def segment(phase,position,rotation,duration,finger):
        nonlocal q,desired,last_nominal
        p0=desired[:3,3].copy();r0=desired[:3,:3].copy();commands=[]
        for i in range(round(duration*20)):
            s=(i+1)/(duration*20);u=s*s*(3-2*s);p=p0+(position-p0)*u;r=slerp(r0,rotation,u)
            q=solve_pose(p+r@relation[:3,3],r@relation[:3,:3],q);commands.append(np.asarray(q).copy())
        desired[:3,3]=position;desired[:3,:3]=rotation;last_nominal=q.copy()
        controller_state['desired_tool_pose']=desired.copy()
        return phase,duration,np.array(commands),finger
    yield segment('tool_lift',desired[:3,3]+[0,0,PROFILE['lift_m']],desired[:3,:3].copy(),timing['tool_lift'],carry)
    yield 'tool_lift_hold',timing['tool_lift_hold'],last_nominal.copy(),carry
    if stop_after_lift:return
    behind=target[:3,3]+[0,PROFILE['behind_flat_m'],0.]
    yield segment('tool_transport',behind+[0,0,PROFILE['lift_m']],target[:3,:3],timing['tool_transport'],carry)
    yield 'tool_transport_hold',timing['tool_transport_hold'],last_nominal.copy(),carry
    yield segment('tool_lower_behind',behind,target[:3,:3],timing['tool_lower_behind'],carry)
    if timing['tool_realign']>0:yield segment('tool_realign',behind,target[:3,:3],timing['tool_realign'],carry)
    yield 'tool_align_hold',timing['tool_align_hold'],last_nominal.copy(),carry
    actual=_transform(read_tool_pose(),'tool')
    if read_state is None:raise RuntimeError('Measured pad and lever states are required before insertion.')
    clearance=insertion_clearance(metadata,read_state());controller_state['insertion_clearance']=clearance
    normal=_pose(read_state()['wafer_pose_world'],'wafer')[:3,2];delta=actual[:3,3]-behind
    if not clearance['passed']or np.linalg.norm(delta-normal*float(delta@normal))>.0005 or math.degrees(_angle(actual[:3,:3],target[:3,:3]))>PROFILE['maximum_docking_angle_deg']:
        raise RuntimeError('Open jaw alignment is not ready for wafer insertion.')
    yield segment('tool_insert',target[:3,3],target[:3,:3],timing['tool_insert'],carry)
    yield 'tool_dock',timing['tool_dock'],last_nominal.copy(),PROFILE['handoff_finger_m']

def handle_retention_error(row,reference):
    """Measure finger-contact surfaces; passive lever rotation is not grip slip."""
    def poses(value):
        inverse=np.linalg.inv(_pose(value['measured_hand_pose_world'],'hand'))
        parts=value['handle_paddle_poses_world']
        if len(parts)!=2:raise ValueError('Two measured contact paddles are required.')
        return [inverse@_pose(p,'contact paddle')for p in parts]
    current,baseline=poses(row),poses(reference)
    shift=np.linalg.norm(np.mean([p[:3,3]for p in current],axis=0)-np.mean([p[:3,3]for p in baseline],axis=0))
    rotation=max(math.degrees(_angle(a[:3,:3],b[:3,:3]))for a,b in zip(current,baseline))
    return float(shift),float(rotation)


def run_tool_pickup(model,data,metadata,record=None,command_record=None,*,stop_after_lift=False,stop_after_seating=False):
    import mujoco
    import wafer_cycle as cycle
    spec=metadata['tool_pickup'];dt=float(model.opt.timestep);steps=round(.05/dt)
    if abs(steps*dt-.05)>1e-12:raise ValueError('Expected the shared 20 Hz control cadence.')
    arm=[model.actuator('drive_joint'+str(i)).id for i in range(1,7)]
    fingers=[model.actuator('drive_joint'+str(i)).id for i in (7,8)]
    target=_pose(spec['target_tool_pose_world'],'target');target_p=np.array(spec['target_tcp_position_world_m']);target_r=np.array(spec['target_tcp_rotation'])
    grip=np.asarray(spec['stand_grasp_arm_q_rad']);start=data.ctrl[arm].copy();open_q=PROFILE['open_finger_m'];carry=PROFILE['transport_finger_m']
    initial=measurement(model,data,metadata,'tool_initialization');rows=[initial]
    if record is not None:record('tool_initialization',initial)
    servo=BoundedTCPFeedback(lambda q:cycle.ik.tcp_pose(q,cycle.station.BASES['left']),
        lambda p,r,q:cycle.ik.solve(p,cycle.ik.matrix_to_quat_xyzw(r),q,base_pos=cycle.station.BASES['left'],iters=100,damping=.015,tol=1e-7),
        cycle.ik.LOWER,cycle.ik.UPPER)
    reference=None;handle_reference=None;tool_to_tcp=None;tool_servo=None;desired_tool=None;last_tool_nominal=None;maximum_slip=0.;maximum_root_shift=0.;maximum_angle=0.;minimum_gap=math.inf;minimum_force=math.inf
    grasp_point=target[:3,:3].T@(target_p-target[:3,3]);contact_gap=np.zeros(2);maximum_contact_gap=0.;held_forces=[]
    error=None
    controller_state={};relation_revision=0
    def tcp_pose():
        hand=_pose(_pose_body(data,'gripper'),'hand');hand[:3,3]+=hand[:3,:3]@np.asarray(cycle.constants.TCP_OFFSET)
        return hand
    try:
        for phase,duration,qfinal,finger in plan_tool_pickup(metadata,lambda:_pose_body(data,'lower_half'),tcp_pose,
                lambda:data.ctrl[arm].copy(),cycle.solve_pose,cycle.ik.slerp_matrix,controller_state,stop_after_lift=stop_after_lift,
                stop_after_seating=stop_after_seating,read_state=lambda:measurement(model,data,metadata,'tool_align_hold')):
            if controller_state.get('relation_revision',0)!=relation_revision:
                relation_revision=controller_state['relation_revision'];tool_to_tcp=np.asarray(controller_state['tool_to_tcp'])
                tool_servo=ToolPickupFeedback(servo.fk,servo.solve,cycle.ik.LOWER,cycle.ik.UPPER,tool_to_tcp=tool_to_tcp)
            print('Tool phase:',phase,flush=True);q0=data.ctrl[arm].copy();f0=data.ctrl[fingers].copy()
            for i in range(round(duration/dt)):
                if i%steps==0:
                    s=min(1.,(i//steps+1)*.05/duration);u=s*s*(3-2*s)
                    nominal=qfinal[i//steps]if qfinal.ndim==2 else qfinal if tool_to_tcp is not None else q0+(qfinal-q0)*u
                    hand=data.body('gripper');r=hand.xmat.reshape(3,3);p=hand.xpos+r@np.asarray(cycle.constants.TCP_OFFSET)
                    if tool_to_tcp is not None:
                        actual=_pose(_pose_body(data,'lower_half'),'tool')
                        q,feedback=tool_servo.step(nominal,actual[:3,3],actual[:3,:3],data.ctrl[arm],
                            integrate=phase in ('tool_lift_hold','tool_transport_hold','tool_align_hold','tool_insert','tool_dock'),
                            integrate_rotation=phase not in PROFILE['translation_only_phases']and phase!='tool_insert'and(phase not in ('tool_align_hold','tool_dock')or i*dt<duration-PROFILE['alignment_orientation_settle_s']),
                            normal_axis=_pose(rows[-1]['wafer_pose_world'],'wafer')[:3,2]if phase in ('tool_align_hold','tool_insert','tool_dock')else None,
                            normal_error_m=jaw_normal_center_error(metadata,rows[-1])if phase in ('tool_align_hold','tool_insert','tool_dock')else None,normal_only=phase=='tool_insert',damped_translation=phase in ('tool_align_hold','tool_insert','tool_dock'),rotation_pivot_world=np.mean([p[:3]for p in rows[-1]['jaw_link_poses_world']],axis=0)if phase=='tool_dock'else None,settle_hold=phase=='tool_dock')
                        feedback['measurement']='Measured tool-root pose; corrected root pose transformed into TCP commands.'
                    else:q,feedback=servo.step(nominal,p,r,data.ctrl[arm],integrate=phase=='tool_grip_hold')
                    data.ctrl[arm]=q;data.ctrl[fingers]=tool_finger_target(phase,(i//steps+1)*.05,duration,f0,finger,metadata)
                    if command_record is not None:command_record(dict(time_s=float(data.time),phase=phase,arm_q_rad=q.tolist(),finger_q_m=data.ctrl[fingers].tolist(),tcp_feedback=feedback))
                mujoco.mj_step(model,data);mujoco.mj_forward(model,data)
                row=measurement(model,data,metadata,phase);rows.append(row)
                if record is not None:record(phase,row)
                cycle.guard_finger_handle_load(row,engine='local');cycle.guard_passive_pad_state(row,metadata['pad_compliance']);cycle.guard_passive_hinge_state(row,metadata['passive_hinge'])
                guard_handle_paddles_for_qualification(row,metadata)
                if row['unexpected_robot_environment_contact_normal_force_N']>1 or row['unexpected_robot_tool_contact_normal_force_N']>1:
                    raise RuntimeError('Unexpected tool-pickup robot contact.')
                if row['tool_unexpected_environment_contact_normal_force_N']>.01:
                    raise RuntimeError('Unexpected carried-tool environment contact: '+str(row['tool_unexpected_environment_contacts_N']))
                if max(row['jaw_contact_normal_force_N'])>.05 or row['forbidden_wafer_contact_force_N']>.01:
                    raise RuntimeError('Tool contacted wafer before the wafer grasp stage.')
                if reference is not None:
                    rel=np.linalg.inv(_pose(row['measured_hand_pose_world'],'hand'))@_pose(row['clamp_pose_world'],'tool')
                    point=rel[:3,3]+rel[:3,:3]@grasp_point;original=reference[:3,3]+reference[:3,:3]@grasp_point
                    slip=float(np.linalg.norm(point-original));angle=math.degrees(_angle(rel[:3,:3],reference[:3,:3]))
                    maximum_root_shift=max(maximum_root_shift,float(np.linalg.norm(rel[:3,3]-reference[:3,3])))
                    slip,angle=handle_retention_error(row,handle_reference)
                    force=np.array(row['finger_handle_contact_normal_force_N']);held_forces.append(force)
                    contact_gap=np.where(force<=.005,contact_gap+dt,0.);maximum_contact_gap=max(maximum_contact_gap,float(contact_gap.max()))
                    if maximum_contact_gap>PROFILE['maximum_contact_gap_s']+1e-10:raise RuntimeError('Finger contact was lost during tool transport.')
                    maximum_slip=max(maximum_slip,slip);maximum_angle=max(maximum_angle,angle)
                    minimum_gap=min(minimum_gap,row['measured_jaw_gap_m']);minimum_force=min(minimum_force,min(row['finger_handle_contact_normal_force_N']))
                    if slip>PROFILE['maximum_relative_slip_m'] or angle>PROFILE['maximum_relative_rotation_deg']:
                        raise RuntimeError('Empty pliers slipped relative to the hand.')
                    if row['measured_jaw_gap_m']<PROFILE['minimum_transport_gap_m']:
                        raise RuntimeError('Transport closed the jaw opening below the insertion clearance.')
                if not np.isfinite(data.qpos).all():raise RuntimeError('Nonfinite tool-pickup state.')
            if phase=='tool_grip_hold':
                reference=np.linalg.inv(_pose(row['measured_hand_pose_world'],'hand'))@_pose(row['clamp_pose_world'],'tool');handle_reference=row
                onset=np.array([r['finger_handle_contact_normal_force_N']for r in rows[-steps:]])
                gap=np.zeros(2);maximum=0.
                for force in onset:
                    gap=np.where(force<=.005,gap+dt,0.);maximum=max(maximum,float(gap.max()))
                if min(onset.mean(0))<PROFILE['minimum_50ms_mean_finger_force_N']or maximum>PROFILE['maximum_contact_gap_s']+1e-10:
                    raise RuntimeError('No bilateral finger contact before tool lift.')
            print(f"t={data.time:.2f}; jaw={row['measured_jaw_gap_m']*1000:.3f}mm; fingers={row['finger_handle_contact_normal_force_N']}; stand={row['tool_stand_normal_force_N']:.3f}N",flush=True)
    except Exception as exc:error=type(exc).__name__+': '+str(exc)
    report=evaluate_tool_pickup(metadata,rows,stop_after_lift=stop_after_lift,stop_after_seating=stop_after_seating)
    report['insertion_clearance']=controller_state.get('insertion_clearance')
    report['error']=error
    if error is not None:
        report['passed']=False;report['result']='FAILED';report['errors'].append(error)
    if report.get('docked')and error is None:
        metadata['initial_arm_q_rad']=data.ctrl[arm].copy().tolist()
        if not np.allclose(data.ctrl[fingers],[-.025,-.025],atol=1e-12,rtol=0):raise ValueError('Wafer-stage handoff finger target differs.')
    metadata['tool_pickup']['result']=report
    return report


def evaluate_tool_pickup(metadata,rows,*,stop_after_lift=False,stop_after_seating=False):
    """Recheck recorded contacts and poses without a physics engine."""
    spec=metadata['tool_pickup'];dt=float(metadata['physics_dt_s'])
    fields='time_s phase handle_paddle_q_rad handle_paddle_qvel_rad_s handle_paddle_passive_torque_Nm clamp_pose_world measured_hand_pose_world measured_jaw_gap_m finger_handle_contact_normal_force_N tool_stand_normal_force_N unexpected_robot_tool_contact_normal_force_N unexpected_robot_environment_contact_normal_force_N tool_unexpected_environment_contact_normal_force_N tool_unexpected_environment_contacts_N jaw_contact_normal_force_N forbidden_wafer_contact_force_N commanded_finger_q_m wafer_pose_world jaw_link_poses_world handle_paddle_poses_world lever_link_poses_world'.split()
    rows=[{k:r[k]for k in fields if k in r}for r in rows if r['phase'].startswith('tool_')]
    target=_pose(spec['target_tool_pose_world'],'target');point=target[:3,:3].T@(np.asarray(spec['target_tcp_position_world_m'])-target[:3,3])
    errors=[];checks={};result=dict(result='INCOMPLETE',passed=False,errors=errors,checks=checks)
    try:
        if spec['profile']!=PROFILE:raise ValueError('Tool-pickup profile differs.')
        times=np.array([r['time_s']for r in rows]);expected_end=tool_pickup_duration(metadata,stop_after_lift=stop_after_lift,stop_after_seating=stop_after_seating)
        if not len(rows)>1 or not np.allclose(np.diff(times),dt,atol=1e-8,rtol=0):raise ValueError('Tool trace cadence is incomplete.')
        initial=rows[0];final=rows[-1]
        paddle_ok=True
        for row in rows:
            try:guard_handle_paddles_for_qualification(row,metadata)
            except (KeyError,TypeError,ValueError,RuntimeError):paddle_ok=False;break
        checks['passive_handle_paddles']=dict(passed=paddle_ok,
            maximum_absolute_pitch_rad=np.max(np.abs([r['handle_paddle_q_rad']for r in rows]),axis=0).tolist(),
            maximum_absolute_speed_rad_s=np.max(np.abs([r['handle_paddle_qvel_rad_s']for r in rows]),axis=0).tolist(),
            maximum_absolute_passive_torque_Nm=np.max(np.abs([r['handle_paddle_passive_torque_Nm']for r in rows]),axis=0).tolist())
        if metadata.get('task_qualification',{}).get('version')=='measured_full_task_v1':
            checks['passive_handle_paddles']['acceptance_profile']=metadata.get('handle_paddle_capacity_profile',PADDLE_CAPACITY_PROFILE)['revision']
            legacy_limits=np.array([s['torque_guard_Nm']for s in metadata['handle_paddles']])
            raw=np.abs([r['handle_paddle_passive_torque_Nm']for r in rows])
            result['legacy_handle_paddle_torque_diagnostic']=dict(acceptance_criterion=False,
                limits_Nm=legacy_limits.tolist(),maximum_absolute_torque_Nm=raw.max(0).tolist(),
                exceeded_sample_count=(raw>legacy_limits).sum(0).tolist())
        checks['fresh_empty_hand_start']=dict(passed=initial['phase']=='tool_initialization'and
            max(initial['finger_handle_contact_normal_force_N'])<=.005,
            initial_handle_force_N=initial['finger_handle_contact_normal_force_N'])
        settled=[r for r in rows if r['phase']=='tool_settle'][-round(1/dt):]
        checks['tool_stand_settled']=dict(passed=len(settled)==round(1/dt)and min(r['tool_stand_normal_force_N']for r in settled)>.1,
            minimum_settled_support_force_N=min((r['tool_stand_normal_force_N']for r in settled),default=None))
        checks['complete_tool_stage']=dict(passed=abs(times[-1]-times[0]-expected_end)<1e-7 and
            final['phase']==('tool_grip_hold'if stop_after_seating else'tool_lift_hold'if stop_after_lift else'tool_dock'),duration_s=float(times[-1]-times[0]))
        seating=[r for r in rows if r['phase']=='tool_grip_hold']
        if not seating:
            checks['seating_completed']=dict(passed=False,reason='Stopped before the seated grip hold.')
            for check in checks.values():check['passed']=bool(check['passed'])
            result.update(result='FAILED',scope='Incomplete physical tool seating.');return result
        reference=seating[-1]
        if stop_after_seating:
            window=seating[-round(.05/dt):];force=np.array([r['finger_handle_contact_normal_force_N']for r in window]);gap=np.zeros(2);maximum=0.
            for f in force:gap=np.where(f<=.005,gap+dt,0.);maximum=max(maximum,float(gap.max()))
            checks['seated_bilateral_contact']=dict(passed=bool(len(window)==round(.05/dt)and min(force.mean(0))>=PROFILE['minimum_50ms_mean_finger_force_N']and maximum<=PROFILE['maximum_contact_gap_s']+1e-10),mean_force_N=force.mean(0).tolist(),maximum_contact_gap_s=maximum)
            checks['open_jaws']=dict(passed=bool(min(r['measured_jaw_gap_m']for r in window)>=PROFILE['minimum_transport_gap_m']))
            for check in checks.values():check['passed']=bool(check['passed'])
            passed=all(c['passed']for c in checks.values());result.update(passed=passed,result='TOOL_SEATING_PASSED'if passed else'FAILED',scope='Physical seating only; no free lift or transport evaluated.');return result
        def relative(row):return np.linalg.inv(_pose(row['measured_hand_pose_world'],'hand'))@_pose(row['clamp_pose_world'],'tool')
        ref=relative(reference);ref_point=ref[:3,3]+ref[:3,:3]@point
        held=[r for r in rows if r['phase']not in ('tool_initialization','tool_settle','tool_approach','tool_close','tool_grip_hold')]
        translation=[];rotation=[];root=[];gaps=[];force=[];contact_gap=np.zeros(2);maximum_gap=0.
        for row in held:
            rel=relative(row);shift,turn=handle_retention_error(row,reference)
            translation.append(shift);rotation.append(turn);root.append(float(np.linalg.norm(rel[:3,3]-ref[:3,3])))
            gaps.append(float(row['measured_jaw_gap_m']));f=np.asarray(row['finger_handle_contact_normal_force_N']);force.append(f)
            contact_gap=np.where(f<=.005,contact_gap+dt,0.);maximum_gap=max(maximum_gap,float(contact_gap.max()))
        force=np.asarray(force);width=round(.05/dt)
        if len(force)<width:raise ValueError('No complete held-tool contact window.')
        sums=np.vstack([np.zeros((1,2)),np.cumsum(force,axis=0)]);minimum_mean=float(np.min((sums[width:]-sums[:-width])/width))
        checks['tool_retention']=dict(passed=max(translation)<=PROFILE['maximum_relative_slip_m']and max(rotation)<=PROFILE['maximum_relative_rotation_deg'],
            maximum_grasp_point_translation_m=max(translation),maximum_relative_rotation_deg=max(rotation),maximum_root_displacement_m=max(root),measurement='Measured contact-paddle midpoint and each paddle orientation relative to the hand; fixed end-of-grip reference.')
        checks['bilateral_handle_contact']=dict(passed=minimum_mean>=PROFILE['minimum_50ms_mean_finger_force_N']and maximum_gap<=PROFILE['maximum_contact_gap_s']+1e-10,
            minimum_50ms_mean_finger_force_N=minimum_mean,maximum_contact_gap_s=maximum_gap,minimum_instantaneous_force_N=float(force.min()))
        checks['open_jaw_transport']=dict(passed=min(gaps)>=PROFILE['minimum_transport_gap_m'],minimum_gap_m=min(gaps))
        lift=[r for r in rows if r['phase']=='tool_lift_hold'][-round(1/dt):]
        checks['free_tool_lift']=dict(passed=len(lift)==round(1/dt)and min(r['clamp_pose_world'][2]-initial['clamp_pose_world'][2]for r in lift)>=.030
            and max(r['tool_stand_normal_force_N']for r in lift)<.01,
            minimum_root_rise_m=min((r['clamp_pose_world'][2]-initial['clamp_pose_world'][2]for r in lift),default=None))
        maximum_unexpected=max(r['unexpected_robot_tool_contact_normal_force_N']+r['unexpected_robot_environment_contact_normal_force_N']for r in rows)
        maximum_tool_environment=max(r['tool_unexpected_environment_contact_normal_force_N']for r in rows)
        for row in rows:
            forces=np.asarray([p[2]for p in row['tool_unexpected_environment_contacts_N']],dtype=float)
            if not np.isfinite(forces).all()or np.any(forces<0)or abs(float(forces.sum())-row['tool_unexpected_environment_contact_normal_force_N'])>1e-6:
                raise ValueError('Tool/environment contact details and total disagree.')
        maximum_wafer=max(max(r['jaw_contact_normal_force_N'])for r in rows)
        checks['collision_and_wafer_clearance']=dict(passed=maximum_unexpected<=.01 and maximum_tool_environment<=.01 and maximum_wafer<=.05 and max(r['forbidden_wafer_contact_force_N']for r in rows)<=.01,
            maximum_unexpected_robot_contact_N=maximum_unexpected,maximum_unexpected_tool_environment_contact_N=maximum_tool_environment,maximum_early_pad_wafer_force_N=maximum_wafer)
        pose=_pose(final['clamp_pose_world'],'tool');position=float(np.linalg.norm(pose[:3,3]-target[:3,3]));angle=math.degrees(_angle(pose[:3,:3],target[:3,:3]))
        if not stop_after_lift:
            alignment=[r for r in rows if r['phase']=='tool_align_hold']
            checks['insertion_sweep']=insertion_clearance(metadata,alignment[-1])if alignment else dict(passed=False,error='No measured insertion state.')
            target_tolerance=1e-7 if metadata.get('engine')=='Isaac Lab / PhysX'else 1e-12
            center_error=abs(jaw_normal_center_error(metadata,final))
            checks['docked_jaw_clearance']=insertion_clearance(metadata,final)
            checks['docked_jaw_clearance']['center_error_m_diagnostic']=center_error
            normal=_pose(final['wafer_pose_world'],'wafer')[:3,2];delta=pose[:3,3]-target[:3,3]
            tangential=float(np.linalg.norm(delta-normal*float(delta@normal)))
            checks['wafer_stage_docking']=dict(passed=tangential<=PROFILE['maximum_docking_tangential_error_m']and angle<=PROFILE['maximum_docking_angle_deg']
                and np.allclose(final['commanded_finger_q_m'],[PROFILE['handoff_finger_m']]*2,atol=target_tolerance,rtol=0),position_error_m_diagnostic=position,tangential_position_error_m=tangential,maximum_tangential_error_m=PROFILE["maximum_docking_tangential_error_m"],orientation_error_deg=angle,
                commanded_target_readback_tolerance_m=target_tolerance,target_tolerance_scope='Native float32 target representation only; no motion/contact tolerance changes.')
        passed=all(c['passed']for c in checks.values())
        result.update(result='TOOL_FREE_LIFT_PASSED'if passed and stop_after_lift else'TOOL_PICKUP_AND_DOCK_PASSED'if passed else'FAILED',passed=passed,
            failed_checks=[k for k,v in checks.items()if not v['passed']],free_lift=checks['free_tool_lift']['passed'],docked=checks.get('wafer_stage_docking',{}).get('passed',False),
            maximum_relative_slip_m=max(translation),maximum_relative_rotation_deg=max(rotation),maximum_root_displacement_m=max(root),
            checked_grasp_point_tool_m=point.tolist(),minimum_50ms_mean_finger_force_N=minimum_mean,maximum_contact_gap_s=maximum_gap,
            bilateral_contact_passed=checks['bilateral_handle_contact']['passed'],maximum_unexpected_robot_contact_N=maximum_unexpected,
            minimum_transport_gap_m=min(gaps),minimum_finger_contact_N=float(force.min()),docking_position_error_m=position,docking_angle_error_deg=angle,
            physics_sample_count=len(rows),final_time_s=float(times[-1]),post_start_pose_write_count=spec['post_start_pose_write_count'])
    except (KeyError,TypeError,ValueError,StopIteration)as exc:errors.append(type(exc).__name__+': '+str(exc))
    for check in checks.values():check['passed']=bool(check['passed'])
    result['scope']='Recorded contact-held passive pliers; no motorized jaw or attachment.'
    return result



if __name__=='__main__':
    import mujoco
    import wafer_cycle as cycle
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--lift-only',action='store_true');parser.add_argument('--seat-only',action='store_true');args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    model,data,metadata,xml=cycle.pickup.build_yam_model()
    model,data,metadata,xml=cycle.add_gun_and_right_arm(model,data,metadata,xml)
    model,data,metadata,xml=prepare_scene(model,data,metadata,xml)
    (args.output/'scene.xml').write_text(xml);replay=[];last_frame=-1
    with gzip.open(args.output/'samples.jsonl.gz','wt',compresslevel=1)as trace,(args.output/'commands.jsonl').open('w')as commands:
        def sample(phase,row):
            global last_frame
            trace.write(json.dumps(row,allow_nan=False)+'\n')
            frame=round(data.time*20)
            if abs(data.time*20-frame)<1e-7 and frame!=last_frame:
                replay.append((float(data.time),data.qpos.copy(),data.qvel.copy(),data.ctrl.copy(),phase));last_frame=frame
        def command(row):commands.write(json.dumps(row,allow_nan=False)+'\n')
        report=run_tool_pickup(model,data,metadata,sample,command,stop_after_lift=args.lift_only,stop_after_seating=args.seat_only)
    (args.output/'report.json').write_text(json.dumps(report,indent=2,default=_json_default)+'\n');(args.output/'metadata.json').write_text(json.dumps(metadata,indent=2,default=_json_default)+'\n')
    np.savez_compressed(args.output/'replay_states.npz',time_s=np.array([r[0]for r in replay]),qpos=np.stack([r[1]for r in replay]),qvel=np.stack([r[2]for r in replay]),ctrl=np.stack([r[3]for r in replay]),phase=np.array([r[4]for r in replay]))
    print(json.dumps(report,indent=2,default=_json_default),flush=True)
    sys.exit(0 if report['passed']else 2)
