"""Robot-only tool pose control from measured tool-in-hand transforms.

Calibrate the held transform in free space, above the food and above the cup.
Only robot joint targets are changed; no object states or scoring data are written.
"""
import math
import numpy as np


def configure(ns):
    from yam_arena import demo as dm,ik
    from yam_arena.constants import LEFT_BASE_POS,RIGHT_BASE_POS,FINGER_OPEN
    from yam_arena.urdf import quat_xyzw_to_matrix
    original=dm.ArmPlan.action7
    if getattr(original,'_tong_alignment',False):return
    generation=[0]
    def reset_controller(env):generation[0]+=1
    ns['on_reset'](reset_controller)
    selected={'tongs: clear side rim','tongs: exit above side rim','tongs: above front lugs',
              'tongs: align above lugs','tongs: approach lugs','tongs: tips around lugs',
              'tongs: align tips','tongs: withdraw from food','tongs: return above cup'}
    cup_stages={'tongs: align over cup','tongs: lower into cup','tongs: release into cup','tongs: clear cup'}
    def held_transform(env,tool):
        robot=env.unwrapped.scene['left_robot']
        poses=ns['labkit']._np(robot.data.body_link_pose_w)[0]
        g=poses[list(robot.body_names).index('gripper')]
        rg=quat_xyzw_to_matrix(g[3:]);rt=tool.rotation()
        origin=ns['labkit']._np(env.unwrapped.scene.env_origins[0])
        tcp=g[:3]-origin+.134*rg[:,2]
        center=tool.pos+.5*tool.joints['jaw']*rt[:,0]
        return rt.T@(tcp-center),rt.T@rg
    def action7(self,t):
        if getattr(self,'_tool_generation',None)!=generation[0]:
            for key in ('_tong_food','_tool_relative','_tool_calibration_time','_cup_relative','_cup_jaw'):
                if hasattr(self,key):delattr(self,key)
            self._tool_generation=generation[0]
        pos,rot,finger,label=self.target(t)
        left=np.linalg.norm(self.base_pos-LEFT_BASE_POS)<np.linalg.norm(self.base_pos-RIGHT_BASE_POS)
        env=ns['labkit']._ENV.get('env')
        if not left or env is None or not (label in selected or label in cup_stages or label.startswith(('loosen:','transfer:'))):
            return original(self,t)
        elapsed=float(t);duration=1.
        for w in self.waypoints:
            if elapsed<w.duration:duration=w.duration;break
            elapsed-=w.duration
        tool=ns['labkit']._part_state(env,next(s for s in ns['OBJECTS'] if s.name=='kitchen_tongs'))
        actual_rt=tool.rotation();actual_center=tool.pos+.5*tool.joints['jaw']*actual_rt[:,0]
        rt0=ns['_Rz'](ns['TONG_YAW'])
        cup_rotation=rt0@dm.grip_rotation(math.pi/2)@ns['_Rz'](math.pi)
        pos=np.asarray(pos)
        if label in cup_stages:
            desired_rt=rt0
            root_goal=pos+.007*rot[:,2]-desired_rt@ns['TONG_GRIP']
            if not hasattr(self,'_cup_jaw'):self._cup_jaw=float(tool.joints['jaw'])
            jaw_blend=ik.smoothstep(min(1.,elapsed/.5)) if label=='tongs: align over cup' else 1.
            desired_center=root_goal+.5*self._cup_jaw*desired_rt[:,0]*jaw_blend
            if label=='tongs: align over cup' and elapsed>.2 and np.linalg.norm(tool.lin_vel)<.03:
                rp,rr=held_transform(env,tool)
                if not hasattr(self,'_cup_relative'):
                    self._cup_relative=(rp,rr)
                else:
                    oldp,oldr=self._cup_relative
                    self._cup_relative=(.8*oldp+.2*rp,ik.slerp_matrix(oldr,rr,.2))
            relative=getattr(self,'_cup_relative',getattr(self,'_tool_relative',None))
            mix=1.-ik.smoothstep(elapsed/duration) if label=='tongs: clear cup' else 1.
        else:
            if not hasattr(self,'_tong_food'):
                f=ns['labkit']._part_state(env,next(s for s in ns['OBJECTS'] if s.name=='belgian_waffle'))
                self._tong_food=(f.pos.copy(),f.rotation().copy())
            fp,fr=self._tong_food
            if label=='transfer: clear grid':
                anchor_blend=1.-ik.smoothstep(elapsed/duration)
            elif label.startswith('transfer:') or label=='tongs: return above cup':
                anchor_blend=0.
            else:
                anchor_blend=ik.smoothstep(elapsed/duration) if label=='tongs: clear side rim' else 1.
            pos=pos+anchor_blend*(fp+fr@(pos-np.asarray(ns['HOME']))-pos)
            rot=ik.slerp_matrix(rot,fr@rot,anchor_blend)
            desired_rt=rot@cup_rotation.T@rt0
            desired_center=pos+.007*rot[:,2]-desired_rt@ns['TONG_GRIP']
            if label=='tongs: align above lugs' and elapsed>.2 and np.linalg.norm(tool.lin_vel)<.03:
                rp,rr=held_transform(env,tool)
                if not hasattr(self,'_tool_relative'):
                    self._tool_relative=(rp,rr);self._tool_calibration_time=elapsed
                else:
                    oldp,oldr=self._tool_relative
                    self._tool_relative=(.8*oldp+.2*rp,ik.slerp_matrix(oldr,rr,.2))
            relative=getattr(self,'_tool_relative',None)
            mix=ik.smoothstep(min(1.,max(0.,elapsed-getattr(self,'_tool_calibration_time',elapsed))/.8)) if label=='tongs: align above lugs' else 1.
        target=pos.copy();target_rot=rot;applied=False
        if relative is not None:
            rp,rr=relative
            calibrated_pos=desired_center+desired_rt@rp
            calibrated_rot=desired_rt@rr
            angle=math.acos(np.clip((np.trace(rot.T@calibrated_rot)-1.)/2.,-1.,1.))
            if np.linalg.norm(calibrated_pos-pos)<.060 and angle<math.radians(35.):
                applied=True
                target=pos+mix*(calibrated_pos-pos)
                target_rot=ik.slerp_matrix(rot,calibrated_rot,mix)
        q,pe,ae=ik.solve(target,ik.matrix_to_quat_xyzw(target_rot),self.q,base_pos=self.base_pos,iters=60)
        self.q=q;self.worst_ik_mm=max(self.worst_ik_mm,pe*1000);self.worst_ik_deg=max(self.worst_ik_deg,float(np.degrees(ae)))
        ns['trace_evidence'].feedback={'kind':'scripted measured tool-in-hand pose control','phase':label,
            'tool_center_error_mm':((desired_center-actual_center)*1000).tolist(),
            'robot_position_correction_mm':((target-pos)*1000).tolist(),
            'calibrated':relative is not None,'calibration_applied':applied,'ik_error_mm':float(pe*1000)}
        return np.r_[q,float(np.clip(finger/FINGER_OPEN,0.,1.))],label
    action7._tong_alignment=True
    dm.ArmPlan.action7=action7
