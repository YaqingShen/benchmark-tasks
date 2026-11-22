"""Receive the molded ring with a stock fingertip just inside its distal arc."""
import numpy as np
from scipy.spatial.transform import Rotation

def receive_frame(hinge,ring_rotation,slide,alternate=False):
    radial,lateral,normal=ring_rotation.T
    if alternate:
        lateral_slope,roll,radial_center,side,advance=.25,35.,.010,-.0015,.004
    else:
        lateral_slope,roll,radial_center,side,advance=.55,45.,.013,0.,.006
    approach=-normal+lateral_slope*lateral-.55*radial
    approach/=np.linalg.norm(approach)
    jaw=-radial+(radial@approach)*approach
    jaw/=np.linalg.norm(jaw)
    roll=np.radians(roll)
    jaw=np.cos(roll)*jaw+np.sin(roll)*np.cross(approach,jaw)
    rotation=np.column_stack((jaw,np.cross(approach,jaw),approach))
    center=hinge+radial_center*radial+side*lateral
    entry=center-rotation@(slide*.012+np.array([.002,0.,.014]))
    return rotation,entry,entry+advance*jaw

def receive(plan,t,label,ns,generation,status,old):
    from yam_arena import ik
    from yam_arena.constants import FINGER_OPEN
    from yam_arena.urdf import quat_xyzw_to_matrix
    from grasp_geometry import ReceivingGeometry
    moves={
        'align right fingers with ring':(1.2,-.012,.012,False),
        'straddle near ring wire':(2.,0.,.012,False),
        'pinch ring directly':(1.4,0.,.002,True),
        'confirm direct finger capture':(1.6,0.,.002,True),
        'retry right grip from clear':(.7,-.012,.012,False),
        'reseat right fingers':(1.2,0.,.012,False),
        'close right fingers again':(1.2,0.,.002,True),
        'confirm second grip':(.7,0.,.002,True),
        'keep direct grip':(1.,0.,.002,True)}
    if label not in moves or ns['labkit'].env() is None:return old(plan,t)
    env=ns['labkit'].env()
    arm=env.unwrapped.scene['right_robot']
    pose=ns['labkit']._np(arm.data.body_link_pose_w)[0,list(arm.body_names).index('gripper')]
    origin=ns['labkit']._np(env.unwrapped.scene.env_origins[0])
    rotation=quat_xyzw_to_matrix(pose[3:]);tcp=pose[:3]-origin+.134*rotation[:,2]
    measured=ns['labkit']._np(arm.data.joint_pos)[0]
    gaps=-measured[6:8]
    if getattr(plan,'_inside_generation',None)!=generation:
        plan._inside_generation=generation;plan._inside_label=None;plan._inside_hold=False
        plan._inside_geometry=ReceivingGeometry(ns)
        plan._inside_bias=np.zeros(3);plan._inside_misses=0
        fk,_=ik.tcp_pose(measured[:6],plan.base_pos)
        plan._inside_base=plan.base_pos+tcp-fk
    if plan._inside_hold:
        plan._inside_misses=0 if status.get('independent_grip') else plan._inside_misses+1
        if plan._inside_misses<6 or status.get('withdrawn'):
            return np.r_[plan.q,-plan._holding_opening/FINGER_OPEN],label
        plan._inside_hold=False;ns['TRACK'].pop('secure_grip_generation',None)
    duration,retreat,gap,closing=moves[label]
    if label!=plan._inside_label:
        plan._inside_label=label;plan._inside_time=t
        plan._inside_start=tcp.copy();plan._inside_start_r=rotation.copy();plan._inside_initial_gap=float(np.mean(gaps))
        if label in ('align right fingers with ring','reseat right fingers'):
            seal=ns['labkit']._part_state(env,ns['SEAL'])
            ring=seal.rotation()@Rotation.from_rotvec([0,seal.joints['pullring'],0]).as_matrix()@ns['S']
            plan._inside_r,plan._entry,plan._closed=receive_frame(seal.to_world(ns['anchor']),ring,plan._inside_geometry.slides[1],alternate=label=='reseat right fingers')
    if not hasattr(plan,'_entry'):return np.r_[plan.q,-.012/FINGER_OPEN],label
    frac=ik.smoothstep(np.clip((t-plan._inside_time+.05)/duration,0.,1.))
    r=ik.slerp_matrix(plan._inside_start_r,plan._inside_r,frac)
    goal=(plan._closed if closing else plan._entry)+retreat*plan._inside_r[:,2]
    raw_target=(1-frac)*plan._inside_start+frac*goal
    gap_target=(1-frac)*plan._inside_initial_gap+frac*gap
    if closing and status.get('independent_grip') and status.get('pinch_frames',0)>=3:
        plan._holding_opening=max(0.,min(gaps)-.002)
        plan._inside_hold=True
        ns['TRACK']['secure_grip_generation']=generation
        return np.r_[plan.q,-plan._holding_opening/FINGER_OPEN],label
    error=raw_target-tcp
    if np.linalg.norm(error)<.011 and not status.get('independent_grip'):
        plan._inside_bias=np.clip(plan._inside_bias+np.clip(.065*error,-.00020,.00020),-.0035,.0035)
    target=raw_target+plan._inside_bias
    carton=ns['labkit']._part_state(env,ns['CARTON']);hook=ns['labkit']._part_state(env,ns['HOOK'])
    geometry=plan._inside_geometry
    commanded=np.full(2,gap_target)
    # If advancing the open jaw would hit the hook, first close in place.
    # Check the real jaw width and intermediate fingertip positions at the
    # stationary wrist before issuing that ordinary gripper command.
    unsafe=min(geometry.clearance(carton,hook,r,target,g)
               for g in (gaps,(gaps+commanded)/2,commanded))<.0006
    if unsafe:
        held_gap=float(np.mean(gaps))
        if closing and gap_target<held_gap:
            for candidate in (gap_target,(gap_target+held_gap)/2):
                sweep=[gaps+a*(candidate-gaps) for a in (.25,.5,.75,1.)]
                if min(geometry.clearance(carton,hook,rotation,tcp,g) for g in sweep)>.00055:
                    held_gap=candidate
                    break
        return np.r_[plan.q,float(np.clip(-held_gap/FINGER_OPEN,0.,1.))],label
    plan.q,_,_=ik.solve(target,ik.matrix_to_quat_xyzw(r),plan.q,base_pos=plan._inside_base,iters=70)
    return np.r_[plan.q,float(np.clip(-gap_target/FINGER_OPEN,0.,1.))],label
