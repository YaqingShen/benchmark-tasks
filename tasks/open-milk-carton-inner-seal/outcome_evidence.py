"""Read-only live outcome evidence. Missing guards prohibit a validated score."""
import json,os,pathlib,math
import numpy as np
from geometry_observer import GeometryObserver,ring_inside_tray,penetrates_carton_interior
from contact_rules import classify,centered_top_grip
from rules import Observation,Tracker
from handoff import Transfer
from rest_support import HandleSupport

def configure(ns):
 hinge=np.asarray(ns['ringjoint'].anchor)
 ring_offsets=np.concatenate([mesh.points for mesh in ns['ringjoint'].meshes])-hinge
 observer=GeometryObserver(ns);tracker=Tracker();history={'seal_cleared':False, 'missing':set()}
 @ns['on_reset']
 def reset(env):
  history['handle_support']=HandleSupport();history['transfer']=Transfer();history['was_open']=False;observer.previous=None;history['seal_cleared']=False;history['missing'].clear();ns['TRACK'].pop('outcome_evidence',None);tracker.violations.clear();tracker.final_hold_s=0.;tracker.max_spill_ml=0.;tracker.supported_pull_seen=False
 @ns['on_step']
 def record(env):
  states={p.name:ns['labkit']._part_state(env,p) for p in ns['OBJECTS']}
  g=observer.read(states,dt=.05);c=ns['TRACK'].get('contact_window',{})
  contact=classify(c.get('pairs',[]),ns.get('FOOD_CONTACT_COLLIDERS',()))
  centered, fingers=centered_top_grip(c.get('pairs',[]),states['carton'].to_local,ns['PEAK'])
  transfer=history['transfer']
  tool=states['hook'];mat=states['tool_mat']
  # Contact reports can stop after the hook settles. Retain the most recent
  # native two-post support only while the bottom corners stay at that pose.
  half=.5*np.asarray(ns['HOOK'].boxes[0][1])
  bottom=[np.array([x,y,-1.])*half+ns['HOOK_GRIP'] for x in (-1.,1.) for y in (-1.,1.)]
  corners=[mat.to_local(tool.to_world(point)) for point in bottom]
  on_rest=history['handle_support'].observe(c.get('pairs',[]),corners)
  material=bool(ns['TRACK']['seal_material_open'])
  direct=transfer.observe(c.get('pairs',[]),states['seal'].joints['pullring'],material,on_rest)
  contact['bimanual_support_active']=centered and direct and transfer.parked and on_rest
  at_tear=material and not history['was_open']
  history['was_open']=material
  ns['TRACK']['transfer_status']={**vars(transfer),'right_two_fingers':direct,'centered_left':centered,'tool_on_rest':on_rest}
  carton_rotation=states['carton'].rotation()
  yaw_deg=float(np.degrees(np.arctan2(carton_rotation[1,0],carton_rotation[0,0])))
  s=states['seal'];tray=states['discard'];cap=states['cap'];rest=states['cap_rest']
  seal_center=np.asarray(g['seal_center']);theta=np.linspace(0,2*math.pi,96,endpoint=False)
  edge=seal_center+np.stack([ns['DISK_R']*np.cos(theta),ns['DISK_R']*np.sin(theta),np.zeros(96)],axis=1)@(s.rotation()@ns['S']).T
  local=(edge-tray.pos)@tray.rotation()
  in_tray=bool((np.abs(local[:,0])<=.098).all() and (np.abs(local[:,1])<=.058).all() and (local[:,2]>=.003).all() and (local[:,2]<=.032).all())
  in_tray=in_tray and ring_inside_tray(ring_offsets,hinge,s.joints['pullring'],s,tray)
  cap_center=np.asarray(g['cap_center']);cl=rest.to_local(cap_center)
  # Sleeping rigid bodies stop reporting impulses. Measure the actual base-to-pad
  # gap and containment in the pad frame, including all edges of the cap base.
  cap_base=cap.pos+np.stack([.020*np.cos(theta),.020*np.sin(theta),np.zeros(96)],axis=1)@cap.rotation().T
  base_local=(cap_base-rest.pos)@rest.rotation()
  base_seated=bool((np.abs(base_local[:,:2])<=.040).all() and np.max(np.abs(base_local[:,2]-.004))<=.00025)
  cap_rest=bool(abs(cl[0])<=.005 and abs(cl[1])<=.005 and 0<cl[2]<.04 and cap.tilt_deg()<=5 and (contact['cap_supported_by_rest'] or base_seated))
  relative=np.asarray(g['seal_center_relative_mm'])/1000
  mouth_axis=states['carton'].rotation()@ns['S']
  edge_in_mouth=(edge-np.asarray(g['mouth_center']))@mouth_axis
  # A flat seal starts recessed in a short neck. Distinguish movement within
  # that neck from crossing the actual interior-side opening into the carton.
  inner_plane=ns['NECK_MID_Z']-ns['NECK_HEIGHT']/2-ns['SEAL_Z']
  half_thickness=(s.rotation()@ns['S'])[:,2]@mouth_axis*.0005
  skin=np.concatenate([edge_in_mouth-half_thickness,edge_in_mouth+half_thickness])
  pushed_in=penetrates_carton_interior(relative,skin,ns['RIM'],inner_plane)
  liquid=ns['TRACK'].get('liquid');unavailable=[]
  if not liquid:unavailable.append('native_liquid_evidence_missing')
  if not contact['interior_guard_available']:unavailable.append('reserved_clean_cap_guard_unvalidated')
  if c.get('callbacks',0)!=12:unavailable.append('incomplete_240Hz_contact_coverage')
  violations=contact['violations']
  if history['seal_cleared']:
   for pair in c.get('pairs',[]):
    if pair['max_force_n']<=1e-5:continue
    if any('/seal/' in pair[k] for k in ['actor0','actor1']):
     for key in ['collider0','collider1']:
      path=pair[key]
      if '/carton/body/mesh_' in path and 1<=int(path.rsplit('mesh_',1)[1])<=32:violations.append('discarded_seal_mouth_contact')
  history['seal_cleared']|=g['seal_clear_of_mouth']
  obs=Observation(cap_off=g['cap_clear_of_mouth'],cap_clean_supported=cap_rest,
   seal_detached=g['seal_clear_of_mouth'],seal_material_open=ns['TRACK']['seal_material_open'],open_fraction=g['sampled_open_fraction'],seal_in_tray=in_tray,
   seal_supported=contact['seal_supported_by_tray'] and transfer.parked and on_rest,seal_speed=g['seal_center_speed_mps'] if g['seal_center_speed_mps'] is not None else 1.,
   carton_supported=contact['carton_supported_by_table'],carton_tilt_deg=g['carton_tilt_deg'],hands_released=contact['hands_released'],
   contamination=any(x in violations for x in ['worktop_mouth_contact','gripper_mouth_contact','tool_tip_worktop_contact','protected_interior_contact','discarded_seal_mouth_contact','reserved_cap_contact']),
   spill_ml=liquid['escaped_ml'] if liquid else 0.,fragment_in_carton=pushed_in,
   unsafe_robot_contact='arm_arm_collision' in violations,
   supported_ring_load=contact['opening_under_load'] and contact['bimanual_support_active'],
   unsupported_opening=at_tear and not contact['bimanual_support_active'])
  provisional=tracker.update(obs)
  history['missing'].update(unavailable)
  measured_score=provisional if not history['missing'] else 0.
  result={'step':ns['TRACK']['step'],'observation':vars(obs),'persistent_violations':sorted(tracker.violations),'validation_missing':sorted(history['missing']),
   'provisional_score_not_official':provisional,'official_score':measured_score,'transfer':dict(ns['TRACK']['transfer_status']),'center_grip_fingers':fingers,'carton_yaw_deg':yaw_deg,'contact_callbacks':c.get('callbacks',0),'opening_under_load':contact['opening_under_load'],'bimanual_support_active':contact['bimanual_support_active']}
  ns['TRACK']['outcome_evidence']=result
  filename=os.environ.get('TASK_TRACE_PATH')
  if filename:
   with pathlib.Path(filename).with_name('outcomes.jsonl').open('a') as f:f.write(json.dumps(result)+'\n')
