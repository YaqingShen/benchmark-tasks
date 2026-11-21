"""Scripted robot targets calibrated to the observed handle pose.

Only robot joint commands are changed. Appliance states, scoring and joint
constraints are untouched. Dream-1 does not call this scripted ArmPlan hook.
"""
import math
import numpy as np


def configure(ns):
    from yam_arena import demo as dm, ik
    from yam_arena.constants import LEFT_BASE_POS, RIGHT_BASE_POS, FINGER_OPEN
    from yam_arena.urdf import quat_xyzw_to_matrix
    original = dm.ArmPlan.action7
    generation = [0]
    ns['on_reset'](lambda env: generation.__setitem__(0, generation[0]+1))

    def action7(self, t):
        if getattr(self, '_handle_generation', None) != generation[0]:
            self._handle_frame = None
            self._handle_generation = generation[0]
        pos, rot, finger, label = self.target(t)
        env = ns['labkit']._ENV.get('env')
        if env is None or not label.startswith('open:'):
            return original(self, t)
        left = np.linalg.norm(self.base_pos-LEFT_BASE_POS) < np.linalg.norm(self.base_pos-RIGHT_BASE_POS)
        if not left and label == 'open: withdraw from lid path':
            return original(self,t)
        st = ns['labkit']._part_state(env, next(o for o in ns['OBJECTS'] if o.name=='waffle'))
        H = np.asarray(ns['HOME'])+np.asarray(ns['HINGE'])
        A = np.array([0.,0.,ns['ZA']])
        if self._handle_frame is None:
            base_R = st.rotation()
            spin_R = ns['_Rx'](st.joints['spin'])
            hinge = st.pos+base_R@(A+spin_R@(np.asarray(ns['HINGE'])-A))
            axis_R = base_R@spin_R
            if left:
                # The proximal handoff plan starts at HANDOFF_DEG; align it to
                # the actual lid, including small stand movement and grip slip.
                correction = axis_R@ns['_Ry'](st.joints['lid']-math.radians(ns['HANDOFF_DEG']))
                self._handle_frame = dict(start=float(t), hinge=hinge, correction=correction)
            else:
                robot=env.unwrapped.scene['right_robot']
                pose=ns['labkit']._np(robot.data.body_link_pose_w)[0,list(robot.body_names).index('gripper')]
                R=quat_xyzw_to_matrix(pose[3:])
                origin=ns['labkit']._np(env.unwrapped.scene.env_origins[0])
                tcp=pose[:3]-origin+.134*R[:,2]
                lid_R=axis_R@ns['_Ry'](st.joints['lid'])
                self._handle_frame=dict(start=float(t),hinge=hinge,axis_R=axis_R,
                    grip=lid_R.T@(tcp-hinge),grip_R=lid_R.T@R)
        c=self._handle_frame
        if left:
            blend=ik.smoothstep(min(1.,max(0.,float(t)-c['start'])/1.))
            target=np.asarray(pos)+blend*(c['hinge']+c['correction']@(np.asarray(pos)-H)-pos)
            target_R=ik.slerp_matrix(rot,c['correction']@rot,blend)
        else:
            # Infer the planned hinge angle from the exterior-handle arc.
            from yam_arena.demo import DOWN_GRIP
            DOWN=quat_xyzw_to_matrix(np.asarray(DOWN_GRIP))
            FWD=ns['_Ry'](-math.pi/2)@DOWN@ns['_Rz'](-math.pi/2)
            rel=rot@FWD.T
            th=math.atan2(rel[0,2],rel[0,0])
            local_R=ns['_Ry'](th)
            nominal=H+local_R@(np.asarray(ns['HOME'])+np.array([ns['GRIP_X'],0.,(ns['HANDLE_Z0']+ns['HANDLE_Z1'])/2])-H)
            lid_R=c['axis_R']@local_R
            target=c['hinge']+lid_R@c['grip']+np.asarray(pos)-nominal
            target_R=lid_R@c['grip_R']
        q,pe,ae=ik.solve(target,ik.matrix_to_quat_xyzw(target_R),self.q,base_pos=self.base_pos,iters=60)
        self.q=q
        self.worst_ik_mm=max(self.worst_ik_mm,1000*pe)
        self.worst_ik_deg=max(self.worst_ik_deg,float(np.degrees(ae)))
        ns['trace_evidence'].feedback=dict(kind='scripted handle pose calibration',phase=label,
            arm='left' if left else 'right',target=target.tolist(),ik_error_mm=float(pe*1000))
        return np.r_[q,float(np.clip(finger/FINGER_OPEN,0.,1.))],label
    dm.ArmPlan.action7=action7
