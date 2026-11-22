"""Native per-physics-step contact evidence; diagnostic only, never alters bodies.
Reference: NVIDIA Omni Physics Contact Reports API. Impulse/dt is a simulated
contact force, not a physical measurement. Substep coverage is counted explicitly.
"""
import json,os,pathlib
import numpy as np
STATE={'sub':None,'pairs':{},'callbacks':0,'samples':0,'window':{},'window_callbacks':0}

def configure(ns):
 labkit=ns['labkit'];original=labkit._author
 def author(prim_path,cfg,**kwargs):
  out=original(prim_path,cfg,**kwargs)
  if cfg.part_name not in {'carton','cap','seal','hook'}:return out
  from isaaclab.app.settings_manager import get_settings_manager
  get_settings_manager().set_bool('/physics/disableContactProcessing',False)
  from isaaclab.sim.utils.stage import get_current_stage
  from pxr import Usd,UsdPhysics,PhysxSchema
  stage=get_current_stage()
  for p in Usd.PrimRange(stage.GetPrimAtPath(prim_path)):
   if p.HasAPI(UsdPhysics.RigidBodyAPI):PhysxSchema.PhysxContactReportAPI.Apply(p).CreateThresholdAttr().Set(0.)
  return out
 labkit._author=author
 @ns['on_reset']
 def reset(env):
  from omni.physx import get_physx_simulation_interface
  from isaaclab.app.settings_manager import get_settings_manager
  get_settings_manager().set_bool('/physics/disableContactProcessing',False)
  from pxr import PhysicsSchemaTools
  STATE.update(pairs={},callbacks=0,samples=0,window={},window_callbacks=0,table_vertical_impulse=0.,external_seal_forces=[],seal_normal=[],seal_tangent=[],seal_anchors=[],tool_left_forces=[],tool_left_normal=[],tool_left_tangent=[],tool_left_anchors=[],left_support_impulse=np.zeros(3))
  dt=float(env.unwrapped.sim.get_physics_dt())
  def callback(headers,data,anchors):
   STATE['callbacks']+=1;STATE['window_callbacks']+=1
   seal_normal=np.zeros(3);seal_tangent=np.zeros(3);seal_count=0
   tool_normal=np.zeros(3);tool_tangent=np.zeros(3);tool_count=0
   for h in headers:
    paths=[str(PhysicsSchemaTools.intToSdfPath(p)) for p in [h.actor0,h.actor1,h.collider0,h.collider1]]
    count=int(h.num_contact_data)
    anchor_count=int(h.num_friction_anchors_data)
    if count<=0 and not anchor_count:continue
    key=' | '.join(paths);record=STATE['pairs'].setdefault(key,{'actor0':paths[0],'actor1':paths[1],'collider0':paths[2],'collider1':paths[3],'substeps':0,'max_force_n':0.,'last_points':[]})
    contacts=[data[i] for i in range(h.contact_data_offset,h.contact_data_offset+count)]
    impulse=sum((np.asarray(c.impulse) for c in contacts),np.zeros(3))
    tangent=sum((np.asarray(anchors[i].impulse) for i in range(h.friction_anchors_offset,h.friction_anchors_offset+anchor_count)),np.zeros(3))
    owners={path.split('/env_0/')[-1].split('/')[0] for path in paths[:2]}
    if 'seal' in owners and 'carton' not in owners and len(owners)==2:
     # The contact normal points from the second shape toward the first.
     sign=1 if paths[0].split('/env_0/')[-1].startswith('seal/') else -1
     seal_normal+=sign*impulse;seal_tangent+=sign*tangent;seal_count+=anchor_count
    if owners=={'hook','LeftArm'}:
     sign=1 if paths[0].split('/env_0/')[-1].startswith('hook/') else -1
     tool_normal+=sign*impulse;tool_tangent+=sign*tangent;tool_count+=anchor_count
    if owners=={'carton','LeftArm'}:
     sign=1 if paths[0].split('/env_0/')[-1].startswith('carton/') else -1
     STATE['left_support_impulse']+=sign*impulse
    if owners=={'carton','table'}:
     STATE['table_vertical_impulse']+=abs(float(impulse[2]))
    force=float(np.linalg.norm(impulse)/dt)
    record['substeps']+=1;record['max_force_n']=max(record['max_force_n'],force);record['last_points']=[list(c.position) for c in contacts[:8]];STATE['samples']+=count
    win=STATE['window'].setdefault(key,{'actor0':paths[0],'actor1':paths[1],'collider0':paths[2],'collider1':paths[3],'substeps':0,'max_force_n':0.,'last_points':[]})
    win['substeps']+=1;win['max_force_n']=max(win['max_force_n'],force);win['last_points']=record['last_points']
   # PhysX point impulses report the normal component. Separate friction
   # anchors supply the tangential component of the same pair and substep.
   STATE['seal_normal'].append((seal_normal/dt).tolist())
   STATE['seal_tangent'].append((seal_tangent/dt).tolist())
   STATE['external_seal_forces'].append(((seal_normal+seal_tangent)/dt).tolist())
   STATE['seal_anchors'].append(seal_count)
   STATE['tool_left_normal'].append((tool_normal/dt).tolist())
   STATE['tool_left_tangent'].append((tool_tangent/dt).tolist())
   STATE['tool_left_forces'].append(((tool_normal+tool_tangent)/dt).tolist())
   STATE['tool_left_anchors'].append(tool_count)
  STATE['sub']=get_physx_simulation_interface().subscribe_full_contact_report_events(callback)
  print('MILK_CONTACT_REPORTS '+json.dumps({'physics_dt':dt,'threshold':0,'score_guard_validated':False}),flush=True)
 @ns['on_step']
 def record(env):
  ns['TRACK']['contact_diagnostics']={'callbacks':STATE['callbacks'],'point_samples':STATE['samples'],'pair_count':len(STATE['pairs']),'validated_for_scoring':False}
  window={'step':ns['TRACK']['step'],'callbacks':STATE['window_callbacks'],'pairs':list(STATE['window'].values())}
  window['table_normal_n']=STATE['table_vertical_impulse']/(max(1,STATE['window_callbacks'])*float(env.unwrapped.sim.get_physics_dt()))
  window['left_on_carton_force_n']=(STATE['left_support_impulse']/(max(1,STATE['window_callbacks'])*float(env.unwrapped.sim.get_physics_dt()))).tolist()
  STATE['left_support_impulse']=np.zeros(3)
  window['external_seal_forces_n']=STATE['external_seal_forces']
  # Independent read-only articulation measurement. Store its native frame
  # as well as the link orientation so the sign can be checked in replay.
  seal=env.unwrapped.scene['seal']
  idx=list(seal.body_names).index('pullring')
  if (not STATE.get('wrench_unavailable',False)):
   try:
    raw=seal.root_physx_view.get_link_incoming_joint_force()
    wrench=(labkit._np(raw) if hasattr(raw,'detach') or hasattr(raw,'torch') else np.asarray(raw.numpy()))[0,idx]
    rotation=labkit._np(seal.data.body_link_pose_w)[0,idx,3:7]
    window['ring_incoming_wrench_child']=wrench.tolist()
    window['ring_child_quat_xyzw']=rotation.tolist()
   except (AttributeError,RuntimeError) as error:
    STATE['wrench_unavailable']=True
    print('RING_WRENCH_UNAVAILABLE',repr(error),flush=True)
  for key in ('seal_normal','seal_tangent','seal_anchors','tool_left_forces','tool_left_normal','tool_left_tangent','tool_left_anchors'):
   window[key]=STATE[key];STATE[key]=[]
  STATE['external_seal_forces']=[]
  STATE['table_vertical_impulse']=0.
  ns['TRACK']['contact_window']=window
  STATE['window']={};STATE['window_callbacks']=0
  filename=os.environ.get('TASK_TRACE_PATH')
  if filename:
   with pathlib.Path(filename).with_name('contact_windows.jsonl').open('a') as f:f.write(json.dumps(window)+'\n')
  if ns['TRACK']['step']%20:return
  if filename:
   out=pathlib.Path(filename).with_name('contacts.jsonl')
   with out.open('a') as f:f.write(json.dumps({'step':ns['TRACK']['step'],'callbacks':STATE['callbacks'],'pairs':list(STATE['pairs'].values())})+'\n')
