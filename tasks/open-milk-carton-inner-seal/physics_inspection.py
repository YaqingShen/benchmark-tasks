"""Read-only native collision cooking and transform consistency diagnostics."""
import json,os,pathlib
import numpy as np

def configure(ns):
 # Subscribe while the stage is authored so a failure during initial settling
 # is distinguishable from one caused by the robot after reset.
 original=ns['labkit']._author
 def author(prim_path,cfg,**kwargs):
  result=original(prim_path,cfg,**kwargs)
  if str(prim_path).endswith('/seal'):
   from omni.physx import get_physx_interface
   from omni.physx.bindings._physx import SimulationEvent
   from pxr import PhysicsSchemaTools
   def on_event(event):
    if event.type!=int(SimulationEvent.JOINT_BREAK):return
    joint=str(PhysicsSchemaTools.decodeSdfPath(*event.payload['jointPath']))
    if '/seal/perimeter_' not in joint:return
    data={'joint':joint,'last_completed_control_step':ns['TRACK']['step']}
    print('MILK_PERIMETER_BREAK '+json.dumps(data),flush=True)
    filename=os.environ.get('TASK_TRACE_PATH')
    if filename:
     with pathlib.Path(filename).with_name('joint_breaks.jsonl').open('a') as f:f.write(json.dumps(data)+'\n')
   ns['PERIMETER_SUBSCRIPTION']=get_physx_interface().get_simulation_event_stream_v2().create_subscription_to_pop(on_event)
  return result
 ns['labkit']._author=author
 @ns['on_reset']
 def inspect(env):
  import omni.physx,omni.usd
  from pxr import PhysicsSchemaTools,Sdf
  from isaaclab.sim.utils.stage import get_current_stage
  stage=get_current_stage();records=[]
  paths=['carton/body/mesh_10','carton/body/mesh_1','cap/body/mesh_46','cap/body/mesh_81'] if ns['FULL_CAP'] else []
  for tail in paths:
   path='/World/envs/env_0/'+tail
   def callback(result,convexes,path=path):
    records.append({'path':path,'result':str(result),'convexes':[{'vertices':[list(x) for x in c.vertices],'indices':list(c.indices),'polygons':[{'index_base':int(p.index_base),'num_vertices':int(p.num_vertices)} for p in c.polygons]} for c in convexes]})
   try:
    omni.physx.get_physx_cooking_interface().request_convex_collision_representation(omni.usd.get_context().get_stage_id(),PhysicsSchemaTools.sdfPathToInt(Sdf.Path(path)),False,callback)
   except Exception as e:records.append({'path':path,'error':str(e)})
  path=os.environ.get('TASK_TRACE_PATH')
  if path:pathlib.Path(path).with_name('cooked_hulls.json').write_text(json.dumps(records))
 @ns['on_step']
 def poses(env):
  import omni.physx
  item=omni.physx.get_physx_interface().get_rigidbody_transformation('/World/envs/env_0/cap/body')
  cap=ns['labkit']._part_state(env,ns['CAP'])
  if item.get('ret_val'):
   pos=np.asarray(item['position']);q=np.asarray(item['rotation']);error=float(np.linalg.norm(pos-cap.pos));angle=2*np.arccos(np.clip(abs(np.dot(q,cap.quat_xyzw)),0,1))
   result={'step':ns['TRACK']['step'],'position_error_m':error,'angle_error_rad':float(angle),'native_pos':pos.tolist(),'native_quat':q.tolist()}
  else:result={'step':ns['TRACK']['step'],'native_transform_available':False}
  path=os.environ.get('TASK_TRACE_PATH')
  if path:
   with pathlib.Path(path).with_name('transform_consistency.jsonl').open('a') as f:f.write(json.dumps(result)+'\n')
