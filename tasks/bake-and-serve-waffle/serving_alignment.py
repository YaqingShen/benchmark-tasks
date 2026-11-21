"""Robot-only serving targets relative to the measured appliance stand.

The initial cooking/opening sequence can move the loose stand a few millimetres.
The bracing hand applies a bounded horizontal correction toward the stand marks.
The turning hand follows measured base translation and stops wrist overtravel once
the actual unit reaches home. No object state, force or scoring data is written.
"""
import numpy as np


def configure(ns):
    from yam_arena import demo as dm,ik
    from yam_arena.constants import LEFT_BASE_POS,RIGHT_BASE_POS,FINGER_OPEN
    original=dm.ArmPlan.action7
    if getattr(original,'_serving_alignment',False):return
    shared={};generation=[0]
    def reset_controller(env):
        shared.clear();generation[0]+=1
    ns['on_reset'](reset_controller)
    def action7(self,t):
        if getattr(self,'_serving_generation',None)!=generation[0]:
            self._serving_started=None;self._serving_generation=generation[0]
        pos,rot,finger,label=self.target(t)
        env=ns['labkit']._ENV.get('env')
        left=np.linalg.norm(self.base_pos-LEFT_BASE_POS)<np.linalg.norm(self.base_pos-RIGHT_BASE_POS)
        eligible=label.startswith('finish:') or (not left and label in {'hold: let go','hold: slide off the handle','hold: up'})
        finishing=label=='park' and getattr(self,'_serving_started',None) is not None
        if env is None or not (eligible or finishing):return original(self,t)
        st=ns['labkit']._part_state(env,next(s for s in ns['OBJECTS'] if s.name=='waffle'))
        if shared.get('env') is not env:
            shared.update(env=env,position=st.pos.copy(),live_position=st.pos.copy(),rotation=st.rotation().copy())
        if not left:shared['live_position']=.85*shared['live_position']+.15*st.pos
        if getattr(self,'_serving_started',None) is None:self._serving_started=float(t)
        blend=ik.smoothstep(min(1.,max(0.,float(t)-self._serving_started)))
        if finishing:
            elapsed=float(t)
            for w in self.waypoints:
                if elapsed<w.duration:
                    blend=1.-ik.smoothstep(elapsed/w.duration);break
                elapsed-=w.duration
        pos=np.asarray(pos).copy()
        if not left and label=='finish: return unit upright' and st.joints['spin']<np.radians(1.) and 'home_pos' not in shared:
            shared.update(home_pos=pos.copy(),home_rot=rot.copy())
        if not left and 'home_pos' in shared:
            if label in {'finish: return unit upright','finish: settle upright','hold: let go'}:
                pos=shared['home_pos'];rot=shared['home_rot']
            elif label=='hold: slide off the handle':
                release=np.asarray(ns['HOME'])+[ns['GRIP_X'],0.,ns['ZA']]+ns['_Rx'](np.radians(-12.))@np.array([0.,0.,(ns['HANDLE_Z0']+ns['HANDLE_Z1'])/2-ns['ZA']])
                pos=shared['home_pos']+(pos-release);rot=shared['home_rot']
        frame=shared['position'] if left else shared['live_position']
        actual=frame+shared['rotation']@(pos-np.asarray(ns['HOME']))
        correction=np.zeros(3)
        if left and label in {'finish: brace the stand','finish: hold stand'}:
            correction[:2]=np.clip(.8*(np.asarray(ns['HOME'])-st.pos)[:2],-.006,.006)
            actual+=correction

        target=np.asarray(pos)+blend*(actual-pos)
        rotation=ik.slerp_matrix(rot,shared['rotation']@rot,blend)
        q,pe,ae=ik.solve(target,ik.matrix_to_quat_xyzw(rotation),self.q,base_pos=self.base_pos,iters=60)
        self.q=q;self.worst_ik_mm=max(self.worst_ik_mm,1000*pe);self.worst_ik_deg=max(self.worst_ik_deg,float(np.degrees(ae)))
        ns['trace_evidence'].feedback={'kind':'scripted measured-stand serving alignment','phase':label,
            'stand_frame_translation_mm':((shared['position']-np.asarray(ns['HOME']))*1000).tolist(),
            'command_target':target.tolist(),'brace_correction_mm':(correction*1000).tolist(),'home_wrist_stopped':'home_pos' in shared,'ik_error_mm':float(pe*1000)}
        return np.r_[q,float(np.clip(finger/FINGER_OPEN,0.,1.))],label
    action7._serving_alignment=True
    dm.ArmPlan.action7=action7
