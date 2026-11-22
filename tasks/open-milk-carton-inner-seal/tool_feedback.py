"""Script-only tool-frame calibration, changing robot targets exclusively."""
import numpy as np

def configure(ns):
 from yam_arena import demo as dm,ik
 from yam_arena.constants import LEFT_BASE_POS,RIGHT_BASE_POS,FINGER_OPEN
 from yam_arena.urdf import quat_xyzw_to_matrix
 old=dm.ArmPlan.action7;generation=[0]
 ns['on_reset'](lambda e:generation.__setitem__(0,generation[0]+1))
 active={'hold raised ring','unload open hook','back hook out of ring','carry clear of held ring','orient above closed ring','approach ring opening','insert hook through ring','engage underside of ring','raise pull ring','pull seal through physical resistance','clear milk mouth','carry to discard tray','level hook over tray','lower seal into tray','release handle','withdraw from tray'}
 align={'hold raised ring','unload open hook','back hook out of ring','orient above closed ring','approach ring opening','insert hook through ring','engage underside of ring','raise pull ring','pull seal through physical resistance','clear milk mouth'}
 def action7(self,t):
  pos,R,f,label=self.target(t)
  is_left=np.linalg.norm(self.base_pos-LEFT_BASE_POS)<np.linalg.norm(self.base_pos-RIGHT_BASE_POS)
  status=ns['TRACK'].get('transfer_status',{})
  live=ns['labkit'].env() is not None
  retries={'retry right grip from clear','reseat right fingers','close right fingers again','confirm second grip'}
  if live:
   # Stop commanding the remaining waypoints after a real inter-arm collision.
   # Hold the measured joint position so a failed handoff cannot push the carton over.
   if getattr(self,'_stop_generation',None)!=generation[0]:
    self._stop_generation=generation[0];self._stop_action=None
   evidence=ns['TRACK'].get('outcome_evidence',{})
   if self._stop_action is None and 'unsafe_robot_contact' in evidence.get('persistent_violations',()):
    arm=ns['labkit'].env().unwrapped.scene['left_robot' if is_left else 'right_robot']
    measured=ns['labkit']._np(arm.data.joint_pos)[0]
    self._stop_action=np.r_[measured[:6],np.clip(measured[6:8].mean()/FINGER_OPEN,0.,1.)]
   if self._stop_action is not None:return self._stop_action.copy(),label
   if is_left and label in {'relieve carton grasp','withdraw above seam','clear carton top','hands clear'} and not status.get('parked'):
    return np.r_[self.q,0.],label
   if not is_left and label in {'align right fingers with ring','straddle near ring wire','pinch ring directly','confirm direct finger capture'}|retries and not status.get('raised'):
    return np.r_[self.q,.44],label
   if is_left and label=='hold raised ring' and status.get('raised'):
    # Maintain the established world-frame arm target. Following a freely
    # tilting carton during the wait can apply further lateral load to its top.
    return np.r_[self.q,0.],label
   if is_left and label in {'unload open hook','back hook out of ring','carry clear of held ring'} and not status.get('withdrawn'):
    if not (ns['TRACK'].get('secure_grip_generation')==generation[0] and status.get('right_two_fingers')):
     return np.r_[self.q,0.],label
   if is_left and label in {'unload open hook','back hook out of ring','carry clear of held ring'}:
    if not status.get('captured'):return np.r_[self.q,0.],label
    if label=='carry clear of held ring' and not status.get('withdrawn'):
     return np.r_[self.q,0.],label
    env=ns['labkit'].env()
    carton=ns['labkit']._part_state(env,ns['CARTON'])
    hook=ns['labkit']._part_state(env,ns['HOOK'])
    toe=hook.to_world(ns['HOOK_TOE'])
    if getattr(self,'_unload_generation',None)!=generation[0]:
     self._unload_generation=generation[0];self._unload_label=None
     self._unload_origin,self._unload_rotation=ik.tcp_pose(self.q,self.base_pos)
     self._unload_axes=carton.rotation()@ns['S']
     self._unload_toe_local=carton.to_local(toe)
     self._unload_local=np.zeros(3);self._unload_bias=np.zeros(3)
    if self._unload_label!=label:
     self._unload_label=label;self._unload_elapsed=0.;self._unload_last=t-.05
     self._unload_start=self._unload_local.copy()
    # Start from the wrist command that is actually supporting the ring.
    # A raised ring can settle below its nominal waypoint under contact load.
    duration,local={
     'unload open hook':(1.25,[-.006,-.004,-.004]),
     'back hook out of ring':(1.5,[-.012,0.,0.]),
     'carry clear of held ring':(1.4,[-.012,.065,.045]),
    }[label]
    end=np.array(local)
    self._unload_elapsed+=float(np.clip(t-self._unload_last,0.,.05))
    self._unload_last=t
    progress=ik.smoothstep(float(np.clip(self._unload_elapsed/duration,0.,1.)))
    self._unload_local=self._unload_start+progress*(end-self._unload_start)
    expected=carton.to_world(self._unload_toe_local+ns['S']@self._unload_local)
    error=expected-toe
    if label=='carry clear of held ring':
     self._unload_bias*=.96
    else:
     self._unload_bias=np.clip(self._unload_bias+np.clip(.10*error,-.00020,.00020),-.005,.005)
    correction=np.clip(.50*error+self._unload_bias,-.006,.006)
    target=self._unload_origin+self._unload_axes@self._unload_local+correction
    self.q,_,_=ik.solve(target,ik.matrix_to_quat_xyzw(self._unload_rotation),self.q,base_pos=self.base_pos,iters=60)
    return np.r_[self.q,0.],label
   if is_left and label in {'return hook above rest','set hook on clean posts','release parked hook','clear hook rest'} and not status.get('withdrawn'):
    return np.r_[self.q,0.],label
   if is_left and label=='hold' and abs(pos[2]-(ns['TOP']+ns['PEAK']-.010))<=.012 and not status.get('parked'):
    return np.r_[self.q,0.],label
   if is_left and label in {'pre-shape left hand above carton center','straddle top seam at midpoint','pinch center of sealed fin','seat on both carton folds','steady carton before pull'} and not status.get('parked'):
    return np.r_[self.q,0.],label
   if label in {'unload open hook','back hook out of ring','carry clear of held ring','return hook above rest','set hook on clean posts','release parked hook','clear hook rest'} and not status.get('captured'):
    return np.r_[self.q,0.],label
   if not is_left and label in {'keep direct grip','draw seal with direct grip'}:
    if getattr(self,'_pull_generation',None)!=generation[0]:
     self._pull_generation=generation[0];self._pull_origin=None;self._pull_reference=None
     self._milk_pull_authorized=False
    if label=='keep direct grip' and status.get('right_two_fingers'):
     self._pull_reference=np.asarray(pos).copy()
   if not is_left and label=='seat inside finger':
    grip=-getattr(self,'_holding_opening',0.)/FINGER_OPEN
    if not all(status.get(k) for k in ('parked','centered_left','right_two_fingers')):
     return np.r_[self.q,grip],label
    env=ns['labkit'].env();arm=env.unwrapped.scene['right_robot']
    actual=ns['labkit']._np(arm.data.body_link_pose_w)[0,list(arm.body_names).index('gripper')]
    wrist=quat_xyzw_to_matrix(actual[3:])
    tcp=actual[:3]-ns['labkit']._np(env.unwrapped.scene.env_origins[0])+.134*wrist[:,2]
    if getattr(self,'_bearing_generation',None)!=generation[0]:
     self._bearing_generation=generation[0];self._bearing_time=t
     self._bearing_cmd,self._bearing_r=ik.tcp_pose(self.q,self.base_pos)
     self._bearing_tcp=tcp.copy()
    step=.0045*ik.smoothstep((t-self._bearing_time+.05)/2.)
    offset=step*self._bearing_r[:,2]
    carton=ns['labkit']._part_state(env,ns['CARTON']);hook=ns['labkit']._part_state(env,ns['HOOK'])
    gaps=-ns['labkit']._np(arm.data.joint_pos)[0,6:8]
    if self._inside_geometry.clearance(carton,hook,self._bearing_r,self._bearing_tcp+offset,gaps)>.0014:
     self.q,_,_=ik.solve(self._bearing_cmd+offset,ik.matrix_to_quat_xyzw(self._bearing_r),
                          self.q,base_pos=self.base_pos,iters=60)
    return np.r_[self.q,grip],label
   if not is_left and label=='draw seal with direct grip':
    if not getattr(self,'_milk_pull_authorized',False):
     self._milk_pull_authorized=all(status.get(k) for k in ('parked','right_two_fingers','centered_left'))
    if not self._milk_pull_authorized:return np.r_[self.q,0.],label
    self._last_draw_nominal=(np.asarray(pos).copy(),R.copy())
    from supported_pull import compliant_pull
    return compliant_pull(self,ns,False,label,status,generation[0])
   if not is_left and label in {'lift seal clear','carry seal to tray'} and ns['TRACK'].get('seal_material_open'):
    if getattr(self,'_carry_generation',None)!=generation[0]:
     self._carry_generation=generation[0];self._carry_start=t
     loaded_tcp,loaded_rotation=ik.tcp_pose(self.q,self.base_pos)
     nominal_tcp,nominal_rotation=self._last_draw_nominal
     self._carry_offset=loaded_tcp-nominal_tcp
     self._carry_rot_offset=loaded_rotation@nominal_rotation.T
    weight=1.-ik.smoothstep((t-self._carry_start)/1.6)
    target=np.asarray(pos)+weight*self._carry_offset
    orientation=ik.slerp_matrix(np.eye(3),self._carry_rot_offset,weight)@R
    self.q,_,_=ik.solve(target,ik.matrix_to_quat_xyzw(orientation),self.q,base_pos=self.base_pos,iters=60)
    return np.r_[self.q,0.],label
   if not is_left and label=='hands clear' and not getattr(self,'_milk_pull_authorized',False):
    return np.r_[self.q,0.],label
   if not is_left and label in {'lift seal clear','carry seal to tray','turn wrist above tray','lower held seal','release seal in tray','withdraw right hand'} and not getattr(self,'_milk_pull_authorized',False):
    return np.r_[self.q,0.],label
  if not is_left:
   from inner_receiver import receive
   return receive(self,t,label,ns,generation[0],status,old)
  if label=='hold' and abs(pos[2]-(ns['TOP']+ns['PEAK']-.010))<=.012:
   from supported_pull import compliant_pull
   return compliant_pull(self,ns,True,label,status,generation[0])
  if label not in active:return old(self,t)
  env=ns['labkit'].env()
  if env is None:return old(self,t)
  if getattr(self,'_milk_generation',None)!=generation[0]:
   self._milk_grip=None;self._milk_generation=generation[0]
   self._entry_trim=np.zeros(3)
   self._seal_lifted=False
   self._capture_trim=np.zeros(2)
   self._capture_last=np.zeros(2)
  tool=ns['labkit']._part_state(env,ns['HOOK'])
  if label not in {'release handle','withdraw from tray'}:
   robot=env.unwrapped.scene['left_robot'];pose=ns['labkit']._np(robot.data.body_link_pose_w)[0,list(robot.body_names).index('gripper')];rg=quat_xyzw_to_matrix(pose[3:]);origin=ns['labkit']._np(env.unwrapped.scene.env_origins[0]);tcp=pose[:3]-origin+.134*rg[:,2]
   self._milk_grip=(tool.rotation().T@(tcp-tool.pos),tool.rotation().T@rg)
   if not hasattr(self,'_milk_reported'):
    print('MILK_GRIP_CALIBRATION',self._milk_grip[0].tolist(),flush=True);self._milk_reported=True
  down=quat_xyzw_to_matrix(np.asarray(dm.DOWN_GRIP));rt=R@down.T;g=ns['HOOK_GRIP'];pt=np.asarray(pos)+.007*R[:,2]-rt@g
  if label in align:
   st=ns['labkit']._part_state(env,ns['CARTON']);pref=ns['P'].copy();pref[2]=ns['TOP'];pt=st.pos+st.rotation()@(pt-pref);rt=st.rotation()@rt
  if label in {'insert hook through ring','engage underside of ring','raise pull ring','pull seal through physical resistance'}:
   # Correct slow tool drift using its observed pose relative to the carton.
   # The ring is 3 mm thick and its centre starts 4 mm above the disk.
   # With the 0.5 mm disk top, the midplane of the free gap is at 1.5 mm.
   axes=st.rotation()@ns['S']
   toe=ns['HOOK_TOE']
   desired=pt+rt@toe
   actual=tool.pos+tool.rotation()@toe
   mouth=st.to_world(ns['C']+ns['S']@np.array([0.,0.,ns['SEAL_Z']]))
   height=(desired-mouth)@axes[:,2]
   if label in {'insert hook through ring','engage underside of ring'} and height<.004:
    error=(desired-actual)@axes
    observed_height=(actual-mouth)@axes[:,2]
    # Downward motion lags its command. Integrating negative height error on
    # descent put the toe into the membrane even as the command was settling.
    error[2]=max(0.,.00165-observed_height)
    self._entry_trim=np.clip(self._entry_trim+np.clip(.10*error,-.000075,.000075),-.0025,.0025)
    if observed_height<.0025:
     pt+=axes[:,2]*np.clip(1.2*(.00165-observed_height),0.,.0015)
   elif label in {'raise pull ring','pull seal through physical resistance'}:
    self._entry_trim*=.97
   pt=pt+axes@self._entry_trim
  if label in {'hold raised ring','raise pull ring','pull seal through physical resistance'} and not self._seal_lifted:
   # Keep the far arc over the metal toe as the ring rotates. With a simple
   # vertical pull the wire can run beyond the 5 mm free end before it tears.
   disk=ns['labkit']._part_state(env,ns['SEAL'])
   angle=disk.joints['pullring']
   if angle<-.45:
    contact_y=ns['HOOK_RING_Y']
    contact_u=.0115+np.sqrt(.0115**2-contact_y**2)
    radial=np.array([contact_u*np.cos(angle),contact_y,-contact_u*np.sin(angle)])
    wire=disk.to_world(ns['anchor']+ns['S']@radial)
    toe=tool.to_world(ns['HOOK_TOE'])
    along=(wire-toe)@tool.rotation()
    error=np.array([along[0]+.001,along[1]])
    self._capture_trim=np.clip(self._capture_trim+np.clip(.12*error,-.00025,.00025),-.012,.012)
    correction=np.clip(self._capture_trim+.65*error,-.014,.014)
    self._capture_last=correction
    pt+=rt[:,:2]@correction
  elif self._seal_lifted and label in {'pull seal through physical resistance','clear milk mouth'}:
   self._capture_last*=.985
   pt+=rt[:,:2]@self._capture_last
  if label in {'pull seal through physical resistance','clear milk mouth'}:
   material=ns['C']+ns['S']@np.array([0.,0.,ns['SEAL_Z']])
   disk=ns['labkit']._part_state(env,ns['SEAL'])
   outward=st.rotation()@ns['N']
   self._seal_lifted |= (disk.to_world(material)-st.to_world(material))@outward>.003
   if self._seal_lifted:
    # Once the disk moves outward, shed the stored robot tracking error.
    # The tool and seal continue to move solely under physical contact.
    toe=ns['HOOK_TOE']
    actual=tool.pos+tool.rotation()@toe
    delta=pt+rt@toe-actual
    length=np.linalg.norm(delta)
    if length>.003:pt-=delta*(1.-.003/length)
  target=pt+rt@self._milk_grip[0];targetR=rt@self._milk_grip[1]
  q,pe,ae=ik.solve(target,ik.matrix_to_quat_xyzw(targetR),self.q,base_pos=self.base_pos,iters=60);self.q=q
  self.worst_ik_mm=max(self.worst_ik_mm,1000*pe);self.worst_ik_deg=max(self.worst_ik_deg,float(np.degrees(ae)))
  ns['TRACK']['feedback']={'phase':label,'ik_error_mm':float(pe*1000),'target':target.tolist()}
  return np.r_[q,float(np.clip(f/FINGER_OPEN,0.,1.))],label
 dm.ArmPlan.action7=action7
 # The native scripted policy can end a failed episode after the final retry.
 # This prevents a later return-to-ready movement across an uncompleted grasp.
 from yam_arena.scripted import YamScriptedPolicy
 previous_act=YamScriptedPolicy.act
 def act(policy,observation):
  chunk=previous_act(policy,observation)
  if not getattr(policy,'_plans',None) or not ns['labkit'].env():return chunk
  now=float(observation.state_time)+policy._dt
  status=ns['TRACK'].get('transfer_status',{})
  left=policy._plans['left'].target(now)[3]
  right=policy._plans['right'].target(now)[3]
  fail=((left=='return hook above rest' and not status.get('withdrawn'))
        or (left=='pre-shape left hand above carton center' and not status.get('parked'))
        or (right=='lift seal clear' and not ns['TRACK'].get('seal_material_open')))
  if fail:
   chunk.actions[-1].meta['request_stop']=True
   print('MILK_SCRIPTED_STOP',left,right,'missing completed transfer or material release',flush=True)
  return chunk
 YamScriptedPolicy.act=act
