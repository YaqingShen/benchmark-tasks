"""Bounded measured-TCP compensation shared by local and cloud simulations.

This controller changes position targets only. It has no physics-engine access,
does not attach objects, and does not change actuator effort or velocity limits.
All rotations and correction vectors use world coordinates. Call once per 50 ms.
Collision screening of the resulting recorded motion remains a separate check.
"""
import math
import numpy as np


PROFILE = dict(command_hz=20, integral_gain_per_command=.20,
    maximum_translation_correction_m=.006,
    maximum_rotation_correction_rad=math.radians(2.),
    maximum_translation_increment_m=.00015,
    maximum_rotation_increment_rad=math.radians(.04),
    maximum_joint_increment_rad=math.radians(1.),
    maximum_ik_position_residual_m=.00002,
    maximum_ik_rotation_residual_rad=math.radians(.01),
    translation_deadband_m=.00001,
    rotation_deadband_rad=math.radians(.005))


def _array(value, shape, name):
    result = np.asarray(value, dtype=float)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError('Invalid '+name)
    return result.copy()


def _rotation(value):
    result = _array(value, (3, 3), 'rotation')
    if np.linalg.norm(result.T@result-np.eye(3)) > 1e-5 or abs(np.linalg.det(result)-1) > 1e-5:
        raise ValueError('Rotation is not a proper orthonormal matrix.')
    return result


def _bounded(vector, maximum):
    norm = np.linalg.norm(vector)
    return vector if norm <= maximum else vector*(maximum/norm)


def rotation_vector(rotation):
    """Principal rotation vector, including a stable near-pi branch."""
    rotation = _rotation(rotation)
    cosine = float(np.clip((np.trace(rotation)-1)/2, -1., 1.))
    skew = np.array([rotation[2,1]-rotation[1,2], rotation[0,2]-rotation[2,0],
                     rotation[1,0]-rotation[0,1]])/2
    sine = float(np.linalg.norm(skew)); angle = math.atan2(sine, cosine)
    if sine > 1e-8:
        return skew*(angle/sine)
    if cosine > 0:
        return skew
    values, vectors = np.linalg.eigh((rotation+np.eye(3))/2)
    axis = vectors[:, np.argmax(values)]
    if axis@skew < 0: axis = -axis
    return axis*angle


def rotation_matrix(vector):
    vector = _array(vector, (3,), 'rotation vector')
    angle = float(np.linalg.norm(vector))
    if angle < 1e-14: return np.eye(3)
    x,y,z = vector/angle
    skew = np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])
    return np.eye(3)+math.sin(angle)*skew+(1-math.cos(angle))*(skew@skew)


class BoundedTCPFeedback:
    """Small integral compensation for constant loaded-arm position error.

    fk(q) returns world position and rotation; solve(p,R,q_seed) returns joint
    target, position residual and angular residual. Measured TCP poses must come
    from the live engine, not from desired or previously commanded joint angles.
    Set integrate=False during a moving segment to avoid integrating motion lag.
    """
    def __init__(self, fk, solve, lower, upper):
        self.fk, self.solve = fk, solve
        self.lower = _array(lower, (6,), 'joint lower limits')
        self.upper = _array(upper, (6,), 'joint upper limits')
        if np.any(self.lower >= self.upper): raise ValueError('Invalid joint bounds.')
        self.translation = np.zeros(3); self.rotation = np.zeros(3)
        self.command_count = 0

    def step(self, nominal_q, measured_p, measured_R, previous_command_q, *, integrate=True):
        nominal_q = _array(nominal_q, (6,), 'nominal joints')
        previous = _array(previous_command_q, (6,), 'previous command')
        measured_p = _array(measured_p, (3,), 'measured TCP position')
        measured_R = _rotation(measured_R)
        if np.any(nominal_q < self.lower) or np.any(nominal_q > self.upper):
            raise ValueError('Nominal arm target violates a joint limit.')
        p, R = self.fk(nominal_q); p = _array(p, (3,), 'nominal TCP'); R = _rotation(R)
        position_error = p-measured_p
        orientation_error = rotation_vector(R@measured_R.T)
        translation, rotation = self.translation.copy(), self.rotation.copy()
        if integrate:
            if np.linalg.norm(position_error) > PROFILE['translation_deadband_m']:
                translation += _bounded(PROFILE['integral_gain_per_command']*position_error,
                                        PROFILE['maximum_translation_increment_m'])
            if np.linalg.norm(orientation_error) > PROFILE['rotation_deadband_rad']:
                rotation += _bounded(PROFILE['integral_gain_per_command']*orientation_error,
                                     PROFILE['maximum_rotation_increment_rad'])
        translation = _bounded(translation, PROFILE['maximum_translation_correction_m'])
        rotation = _bounded(rotation, PROFILE['maximum_rotation_correction_rad'])
        requested_p = p+translation; requested_R = rotation_matrix(rotation)@R
        q, pe, ae = self.solve(requested_p, requested_R, previous)
        q = _array(q, (6,), 'IK command')
        # Recompute residuals; callback-reported success is not accepted blindly.
        actual_p, actual_R = self.fk(q)
        pe = float(np.linalg.norm(np.asarray(actual_p)-requested_p))
        ae = float(np.linalg.norm(rotation_vector(requested_R@np.asarray(actual_R).T)))
        if pe > PROFILE['maximum_ik_position_residual_m'] or ae > PROFILE['maximum_ik_rotation_residual_rad']:
            raise RuntimeError('Bounded feedback IK residual exceeds its fixed limit.')
        if np.any(q < self.lower) or np.any(q > self.upper):
            raise RuntimeError('Bounded feedback command violates a joint limit.')
        if np.max(np.abs(q-previous)) > PROFILE['maximum_joint_increment_rad']:
            raise RuntimeError('Bounded feedback joint increment exceeds one degree per command.')
        self.translation, self.rotation = translation, rotation
        self.command_count += 1
        return q, dict(command_index=self.command_count, integrated=bool(integrate),
            nominal_q_rad=nominal_q.tolist(), measured_tcp_position_world_m=measured_p.tolist(),
            nominal_tcp_position_world_m=p.tolist(), measured_position_error_m=position_error.tolist(),
            measured_orientation_error_rad=orientation_error.tolist(),
            translation_correction_world_m=translation.tolist(), rotation_correction_world_rad=rotation.tolist(),
            requested_tcp_position_world_m=requested_p.tolist(), ik_position_residual_m=pe,
            ik_orientation_residual_rad=ae, maximum_joint_increment_rad=float(np.max(np.abs(q-previous))),
            saturated_translation=bool(np.linalg.norm(translation)>=PROFILE['maximum_translation_correction_m']-1e-12),
            saturated_rotation=bool(np.linalg.norm(rotation)>=PROFILE['maximum_rotation_correction_rad']-1e-12))


JAW_PROFILE = dict(revision='measured_jaw_plane_settled_hold_v28', hold_acquire_after_s=1., hold_position_m=.001, hold_angle_rad=math.radians(.5), resume_position_m=.0015, resume_angle_rad=math.radians(.75), integral_gain=.04,
    proportional_gain=.20, maximum_rotation_correction_rad=math.radians(4.))


def _pose7(value):
    value=_array(value,(7,),'measured pose');x,y,z,w=value[3:]/np.linalg.norm(value[3:])
    r=np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
        [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
        [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
    return value[:3],r


def measured_jaw_frame(row,faces):
    poses=[_pose7(p)for p in row['jaw_link_poses_world']]
    centers=[p+r@np.mean(face,axis=0)for (p,r),face in zip(poses,faces,strict=True)]
    r0,r1=[r for p,r in poses]
    return np.mean(centers,axis=0),r0@rotation_matrix(.5*rotation_vector(r0.T@r1))


class JawPlaneFeedback(BoundedTCPFeedback):
    """Track the measured jaw midplane as the passive pliers close."""
    def __init__(self,base,metadata,initial,nominal_q,read_row):
        super().__init__(base.fk,base.solve,base.lower,base.upper)
        self.translation=base.translation.copy();self.rotation=base.rotation.copy()
        self.read_row=read_row;self.faces=np.asarray(metadata['contact_faces_local_m'])
        self.p0,self.r0=self.fk(nominal_q)
        center,r=measured_jaw_frame(initial,self.faces);wafer,wr=_pose7(initial['wafer_pose_world'])
        normal=wr[:,2];x=r[:,0]-normal*float(r[:,0]@normal);x/=np.linalg.norm(x)
        self.center0=center-normal*float((center-wafer)@normal)
        self.frame0=np.column_stack((x,np.cross(normal,x),normal))
        self.previous_position=None;self.previous_angle=None;self.hold_started=None;self.hold_command=None
        metadata['jaw_plane_feedback']=dict(profile=dict(JAW_PROFILE),
            initial_center_world_m=self.center0.tolist(),initial_frame_world=self.frame0.tolist(),
            scope='Measured jaw pose controls arm targets; fixed initial wafer plane and continuous nominal lift. No tool attachment or jaw actuation.')

    def step(self,nominal_q,measured_p,measured_R,previous_command_q,*,integrate=True):
        nominal=_array(nominal_q,(6,),'nominal joints');previous=_array(previous_command_q,(6,),'previous command')
        p,r=self.fk(nominal);delta=r@self.r0.T
        desired_center=np.asarray(p)+delta@(self.center0-self.p0);desired_frame=delta@self.frame0
        row=self.read_row();center,frame=measured_jaw_frame(row,self.faces)
        error=desired_center-center;angle=rotation_vector(desired_frame@frame.T)
        if row.get('phase')=='lift_hold':
            if self.hold_started is None:self.hold_started=float(row['time_s'])
            if np.linalg.norm(error)>JAW_PROFILE['resume_position_m'] or np.linalg.norm(angle)>JAW_PROFILE['resume_angle_rad']:
                self.hold_command=None
            elif self.hold_command is None and float(row['time_s'])-self.hold_started>=JAW_PROFILE['hold_acquire_after_s'] and np.linalg.norm(error)<=JAW_PROFILE['hold_position_m'] and np.linalg.norm(angle)<=JAW_PROFILE['hold_angle_rad']:
                self.hold_command=previous.copy()
        else:self.hold_started=None;self.hold_command=None
        for value,old,attr,cap in [(error,self.previous_position,'translation',PROFILE['maximum_translation_increment_m']),
                                  (angle,self.previous_angle,'rotation',PROFILE['maximum_rotation_increment_rad'])]:
            if self.hold_command is not None:continue
            inc=JAW_PROFILE['integral_gain']*value
            if old is not None:inc+=JAW_PROFILE['proportional_gain']*(value-old)
            setattr(self,attr,getattr(self,attr)+_bounded(inc,cap))
        self.previous_position=error.copy();self.previous_angle=angle.copy()
        self.translation=_bounded(self.translation,PROFILE['maximum_translation_correction_m'])
        self.rotation=_bounded(self.rotation,JAW_PROFILE['maximum_rotation_correction_rad'])
        requested_p=p+self.translation;requested_r=rotation_matrix(self.rotation)@r
        if self.hold_command is None:q,_,_=self.solve(requested_p,requested_r,previous)
        else:
            q=self.hold_command.copy();requested_p,requested_r=self.fk(q)
        q=_array(q,(6,),'jaw feedback command')
        actual_p,actual_r=self.fk(q)
        pe=float(np.linalg.norm(actual_p-requested_p));ae=float(np.linalg.norm(rotation_vector(requested_r@actual_r.T)))
        inc=float(np.max(np.abs(q-previous)))
        if np.any(q<self.lower)or np.any(q>self.upper)or pe>PROFILE['maximum_ik_position_residual_m']or ae>PROFILE['maximum_ik_rotation_residual_rad']or inc>PROFILE['maximum_joint_increment_rad']:
            raise RuntimeError('Jaw-plane feedback violates the existing IK, joint or command-increment bound.')
        self.command_count+=1
        return q,dict(command_index=self.command_count,integrated=self.hold_command is None,stationary_command_hold=self.hold_command is not None,controller=JAW_PROFILE['revision'],
            nominal_q_rad=nominal.tolist(),measured_tcp_position_world_m=np.asarray(measured_p).tolist(),
            nominal_tcp_position_world_m=np.asarray(p).tolist(),measured_position_error_m=error.tolist(),
            measured_orientation_error_rad=angle.tolist(),measured_jaw_center_world_m=center.tolist(),
            measured_jaw_frame_world=frame.tolist(),desired_jaw_center_world_m=desired_center.tolist(),
            desired_jaw_frame_world=desired_frame.tolist(),translation_correction_world_m=self.translation.tolist(),
            rotation_correction_world_rad=self.rotation.tolist(),requested_tcp_position_world_m=requested_p.tolist(),
            ik_position_residual_m=pe,ik_orientation_residual_rad=ae,maximum_joint_increment_rad=inc)
