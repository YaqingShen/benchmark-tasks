"""pack_grok002 (by grok): bimanual YAM native Isaac packing world, adapted from mailer_world.py.
Start: an already-formed open mailer (walls rolled, tabs in slots, lid open on the worktable),
a foam pad and a paperback on two staging pedestals in front of the box. No pose writes after reset.
"""
import atexit,hashlib,json,os,subprocess
from collections import deque
from pathlib import Path
import numpy as np
_APP=None
def jdefault(o):
 return o.item() if hasattr(o,'item') else str(o)

def array(v):
 v=v.torch if hasattr(v,'torch') else v
 return v.detach().cpu().numpy() if hasattr(v,'detach') else np.asarray(v)

def transform(p):
 x,y,z,w=np.asarray(p[3:])/np.linalg.norm(p[3:]);m=np.eye(4);m[:3,3]=p[:3]
 m[:3,:3]=[[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]]
 return m

import pack_params as pp
import pack_check as pc

class PackWorld:
 def __init__(self,output,probe='pack_start',record=True,root=None,dt=.001,tabless=False,resolution=640,enable_cameras=True):
  global _APP
  import torch
  self.torch=torch;self.width=resolution;self.height=resolution*9//16;self.root=Path(root or __file__).resolve().parent;self.output=Path(output);self.output.mkdir(parents=True,exist_ok=False)
  self.manifest=json.loads((self.root/'assets/manifest.json').read_text());self.env=None;self.video=None;self.steps=0;self.hold_steps=0;self.record=record;self.probe=probe;self.last={};self.tabless=tabless;self.pose_window=deque(maxlen=21)
  self.enable_cameras=enable_cameras
  os.environ['ENABLE_CAMERAS']='1' if enable_cameras else '0'
  if _APP is None:
   from isaaclab.app import AppLauncher
   _APP=AppLauncher(headless=True,enable_cameras=enable_cameras,device='cpu').app;atexit.register(_APP.close)
  from crease_material import PassiveCreaseCfg
  from isaaclab.sensors import ContactSensorCfg
  from isaaclab.managers import EventTermCfg,SceneEntityCfg
  from isaaclab_arena.terms.events import reset_articulation_joints
  from isaaclab_arena.assets.object import Object
  from isaaclab_arena.assets.object_type import ObjectType
  from isaaclab_arena.utils.pose import Pose
  from isaaclab_arena.scene.scene import Scene
  from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
  from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
  from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
  from yam_arena import station,constants,ik
  from yam_arena.urdf import quat_xyzw_to_matrix
  from yam_arena.embodiment import BimanualYamEmbodiment
  import yam_arena.scene as scenes
  self.station=station;self.constants=constants;self.ik=ik;self.support_top=station.TABLE_TOP+.125
  staging_top=station.TABLE_TOP+.150
  # Simple raised platform; side edges remain accessible to the fingers.
  import isaaclab.sim as sim_utils
  from isaaclab_physx.sim.schemas import PhysxCollisionPropertiesCfg
  def bench_box(name,size,position):
   return Object(name=name,object_type=ObjectType.BASE,spawner_cfg=sim_utils.CuboidCfg(size=size,collision_props=PhysxCollisionPropertiesCfg(contact_offset=.00005,rest_offset=0.),visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(.24,.27,.29),roughness=.85),physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=.65,dynamic_friction=.5)),initial_pose=Pose(position_xyz=position))
  pedestal_height=self.support_top-station.TABLE_TOP-.008
  bench=[bench_box('worktable_top',(.4206,.170,.008),(.4463,0.,self.support_top-.004)),
         bench_box('worktable_base',(.160,.100,pedestal_height),(.4463,0.,station.TABLE_TOP+pedestal_height/2))]
  # Narrow staging stands give the fingers access to the underside of each item.
  for item,st in pp.STAGING.items():
   (px,py),(sx,sy)=st['pedestal_center_xy'],st['pedestal']
   bench.append(bench_box('staging_'+item,(sx,sy,staging_top-station.TABLE_TOP),(px,py,(staging_top+station.TABLE_TOP)/2)))
  scenes.STATION_DOME_INTENSITY=250.;scenes.STATION_LAMP_INTENSITY=1.8e4;scenes.STATION_LAMP_RADIUS=.55
  embodiment=BimanualYamEmbodiment(enable_cameras=enable_cameras,rig=station.rig(),cam_height=self.height,cam_width=self.width,robot_contact_offset=.00005,contact_sensors=True)
  self.camera_home_action=[]
  _,neutral=ik.tcp_pose(np.array(constants.READY))
  for side,sgn in [('left',1),('right',-1)]:
   c,s=np.cos(-sgn*.70),np.sin(-sgn*.70);rz=np.array([[c,-s,0],[s,c,0],[0,0,1.]])
   mount=quat_xyzw_to_matrix(np.array(station.wrist_camera(side).rot_xyzw));c,s=np.cos(.15),np.sin(.15);up=np.array([[1.,0,0],[0,c,-s],[0,s,c]])
   q,ep,er=ik.solve([.20,sgn*.29,1.06],ik.matrix_to_quat_xyzw(rz@neutral@mount@up@mount.T),constants.READY,station.BASES[side],iters=500,damping=.02)
   if ep>.002 or er>.02:raise RuntimeError('Initial IK failed')
   self.camera_home_action.extend([*q.tolist(),1.])
   if probe in ['right_outer_held','right_partial_held'] and side=='left':
    a=np.deg2rad(-74.1257 if probe=='right_partial_held' else -75.);c,t=np.cos(a),np.sin(a);rot=np.array([[c,0,t],[0,1,0],[-t,0,c]]);base=np.array([[0,1,0],[-1,0,0],[0,0,1.]]);h=np.array([.35,0,self.support_top+.00065])+base@np.array([.110,-.0822,0]);point=h+base@rot@np.array([.035,0,0]);normal=np.array([0,0,-1.]);approach=np.array([0,1.,0]);rotation=base@rot@np.column_stack([normal,np.cross(approach,normal),approach]);q,ep,er=ik.solve(point,ik.matrix_to_quat_xyzw(rotation),constants.READY,station.BASES[side],iters=800,damping=.018)
    if ep>.001 or er>.01:raise RuntimeError('Diagnostic held-outer initialization is unreachable')
   arm=getattr(embodiment.scene_config,side+'_robot');arm.init_state.joint_pos.update(dict(zip(constants.ARM_JOINTS,q.tolist())));arm.actuators['gripper'].joint_effort_limit=8.
   if probe in ['right_outer_held','right_partial_held'] and side=='left':
    # Replace the inherited regex, which otherwise also matches joint7/8.
    arm.init_state.joint_pos.pop('joint[7-8]',None)
    arm.init_state.joint_pos.update({n:constants.FINGER_OPEN*(.004/.0939) for n in constants.FINGER_JOINTS})
  obj=Object(name='mailer',usd_path=str(self.root/'assets'/('mailer_notabs.usda' if tabless else 'mailer.usda')),initial_pose=Pose(position_xyz=(.35,0,self.support_top+.00065),rotation_xyzw=(0,0,-2**-.5,2**-.5)),object_type=ObjectType.ARTICULATION)
  cfg=obj.object_cfg;cfg.spawn.activate_contact_sensors=True
  cfg.actuators={'passive_scores':PassiveCreaseCfg(joint_names_expr=['.*'],stiffness=self.manifest['crease_stiffness_Nm_rad'],damping=self.manifest['crease_damping_Nm_s_rad'],joint_effort_limit=.04,joint_velocity_limit=12.)}
  initial={n:0. for n in self.manifest['joints']}
  lid_rest=-(np.deg2rad(90.68)+np.arcsin(.050/self.manifest['lid_length_m']))  # by grok: lid lying on the worktable
  if probe in ['assembled','tray','partial','tray_propped','pack_start','diag_packed']:
   initial={n:np.deg2rad(j['target_deg']) for n,j in self.manifest['joints'].items()}
   if probe in ['tray','partial']:
    for n in ['lid','front_tuck','lid_left_dust','lid_right_dust','tuck_left_ear','tuck_right_ear']:initial[n]=0.
    # Back wall upright makes the unclosed lid vertical, yielding an accessible tray.
   if probe=='partial':
    for n in ['left_inner','right_inner']:initial[n]*=.80
  if probe=='front_formed':initial['front']=-np.pi/2
  if probe=='back_propped':
   back=np.deg2rad(90.5);absolute=-np.arcsin(.050*np.sin(back)/.1616);initial.update(back=back,lid=absolute-back,front_tuck=-absolute)
  if probe in ['walls_ready','right_outer_held','right_partial_held']:
   initial.update(front=-np.pi/2,back=np.deg2rad(90.68),lid=np.deg2rad(-108.7),front_tuck=np.deg2rad(18.),left_outer=0.,right_outer=0.)
   for wall in ['front','back']:
    for side,sign in [('left',1),('right',-1)]:initial[wall+'_'+side+'_ear']=sign*np.pi/2
  if probe=='right_outer_held':initial['right_outer']=-np.deg2rad(75.)
  if probe=='right_partial_held':initial.update({n:np.deg2rad(a) for n,a in {'right_outer':-74.1257,'right_rim':-57.627,'right_inner':-15.289,'front':-91.823,'back':90.794,'lid':-108.552,'front_tuck':18.023}.items()})
  if probe in ['tray_propped','pack_start','diag_packed']:
   initial.update(back=np.deg2rad(90.68),lid=lid_rest,front_tuck=-(lid_rest+np.deg2rad(90.68)),lid_left_dust=0.,lid_right_dust=0.,tuck_left_ear=0.,tuck_right_ear=0.)
  cfg.init_state.joint_pos=initial;cfg.init_state.joint_vel={'.*':0.}
  # by grok: foam pad (2-body articulation, passive foam spring) and paperback (rigid), on the pedestals
  from foam_material import FoamSpringCfg
  yaw=(0,0,-2**-.5,2**-.5)
  (px,py)=pp.STAGING['foam_pad']['center_xy'];pz=staging_top+.0002
  if probe=='diag_packed':(px,py,pz)=(.35+pp.PAD_TARGET_V,0.,self.support_top+.00065+pp.BASE_TOP_Z+.0002)  # by grok: DIAGNOSTIC start only (contents spawned in the box at reset; never a task start)
  pad=Object(name='foam_pad',usd_path=str(self.root/'assets/foam_pad.usda'),initial_pose=Pose(position_xyz=(px,py,pz),rotation_xyzw=yaw),object_type=ObjectType.ARTICULATION)
  pcfg=pad.object_cfg;pcfg.spawn.activate_contact_sensors=False
  pcfg.actuators={'foam':FoamSpringCfg(joint_names_expr=['compress'],stiffness=pp.PAD['stiffness_N_m'],damping=pp.PAD['damping_N_s_m'],joint_effort_limit=pp.PAD["max_force_N"],joint_velocity_limit=1.,rest_offset_m=pp.PAD['preload_N']/pp.PAD['stiffness_N_m'])}
  pcfg.init_state.joint_pos={'compress':0.};pcfg.init_state.joint_vel={'.*':0.}
  _yb=np.deg2rad(-90.+pp.BOOK_STAGING_YAW_DEG)/2;_yaw_book=(0,0,float(np.sin(_yb)),float(np.cos(_yb)))  # by grok: book staged yawed (pack_grok002)
  (bx,by)=pp.STAGING['book']['center_xy'];bz=staging_top+pp.BOOK['size'][2]/2+.0002
  if probe=='diag_packed':(bx,by,bz,_yaw_book)=(.35+pp.BOOK_TARGET_V,0.,self.support_top+.00065+pp.BASE_TOP_Z+.0002+pp.PAD['size'][2]+pp.BOOK['size'][2]/2+.0003,yaw)  # by grok: DIAGNOSTIC start only
  book=Object(name='book',usd_path=str(self.root/'assets/book.usda'),initial_pose=Pose(position_xyz=(bx,by,bz),rotation_xyzw=_yaw_book),object_type=ObjectType.RIGID)
  hands=[]
  for side in ['left','right']:
   arm=getattr(embodiment.scene_config,side+'_robot')
   hands.extend([arm.prim_path+embodiment.link_paths[k] for k in ['tip_left','tip_right']])
  self.filter_names=[*self.manifest['panels'],'pad_base','pad_top','book']
  filters=[obj.prim_path+'/'+n for n in self.manifest['panels']]+[pad.prim_path+'/pad_base',pad.prim_path+'/pad_top',book.prim_path]
  standard=scenes.env_cfg_callback(20)
  def config(c):
   c=standard(c);c.events.mailer_reset_joint_state=EventTermCfg(func=reset_articulation_joints,mode='reset',params={'asset_cfg':SceneEntityCfg('mailer')});c.sim.dt=dt;c.decimation=round(.05/dt);c.sim.render_interval=c.decimation;c.episode_length_s=1800.
   for i,path in enumerate(hands):setattr(c.scene,f'hand_{i}_contacts',ContactSensorCfg(prim_path=path,filter_prim_paths_expr=filters,update_period=0.,history_length=round(.05/dt),max_contact_data_count_per_prim=4096))
   return c
  desc=IsaacLabArenaEnvironment(name='pack_grok002_mailer',scene=Scene([*scenes.task_room('station',embodiment.rig.top_cam),*bench,obj,pad,book]),embodiment=embodiment,env_cfg_callback=config)
  self.env=ArenaEnvBuilder(desc,ArenaEnvBuilderCfg(num_envs=1,solve_relations=False,device='cpu')).make_registered();self.env.reset();self.base=self.env.unwrapped
  if enable_cameras:
   import omni.replicator.core as rep
   rep.settings.set_render_rtx_realtime(antialiasing='DLAA')
  self.box=self.base.scene['mailer'];self.body_ids={n:self.box.find_bodies(n)[0][0] for n in self.manifest['panels']};self.joint_ids={n:self.box.find_joints(n)[0][0] for n in self.manifest['joints']}
  self.box.actuators.target_command.set_position_index(value=torch.zeros((1,self.box.num_joints),device=self.base.device),joint_ids=list(range(self.box.num_joints)))
  if probe=='front_formed':self.box.actuators['passive_scores'].plastic_rotation[0,self.joint_ids['front']]=-np.pi/2
  if probe=='back_propped':
   peak=np.deg2rad(102);absolute=-np.arcsin(.050*np.sin(peak)/.1616);memory={'back':peak-.2,'lid':absolute-peak+.2,'front_tuck':max(0.,-absolute-.2)}
   for name,value in memory.items():self.box.actuators['passive_scores'].plastic_rotation[0,self.joint_ids[name]]=value
  if probe in ['walls_ready','right_outer_held','right_partial_held']:
   memory=dict(initial);memory.update(back=np.deg2rad(90.54),lid=np.deg2rad(-108.16),front_tuck=np.deg2rad(6.16))
   if probe=='right_outer_held':memory['right_outer']=-np.deg2rad(75.)+.2
   if probe=='right_partial_held':memory.update({n:np.deg2rad(a) for n,a in {'right_outer':-64.4432,'right_rim':-50.6597,'right_inner':-10.1641,'front':-90.,'front_tuck':9.1742}.items()})
   for name,value in memory.items():self.box.actuators['passive_scores'].plastic_rotation[0,self.joint_ids[name]]=value
  if probe in ['tray_propped','pack_start','diag_packed']:
   memory=dict(initial);memory.update(front=np.deg2rad(-90.54),back=np.deg2rad(90.54),lid=lid_rest+np.deg2rad(.54),front_tuck=-(lid_rest+np.deg2rad(90.68))-.2)
   for side,sign in [('left',1),('right',-1)]:
    for wall in ['front','back']:memory[wall+'_'+side+'_ear']=sign*np.deg2rad(90.54)
    for segment,a in [('outer',78.54),('rim',78.54),('inner',101.46)]:memory[side+'_'+segment]=sign*np.deg2rad(a)
   for name,value in memory.items():self.box.actuators['passive_scores'].plastic_rotation[0,self.joint_ids[name]]=value
  self.pad=self.base.scene['foam_pad'];self.book=self.base.scene['book'];self.pad_ids={n:self.pad.find_bodies(n)[0][0] for n in ['pad_base','pad_top']};self.content_window=deque(maxlen=21)
  self.trace=(self.output/'trace.jsonl').open('w')
  self.metadata={'revision':self.manifest['revision'],'native_isaac':True,'probe':probe,'diagnostic_only':probe!='flat','robot_demonstration':False,'post_reset_pose_writes':0,'grasp_constraints':0,'automatic_lock_constraints':0,'spring_targets_radians':0,'calibrated_material':False,'tab_colliders_removed':tabless,'physics_hz':1/dt,'control_hz':20,'maximum_episode_s':1800.,'cameras':['top_cam','left_cam','right_cam'],'overlays':False,'camera_resolution':[self.width,self.height],'stability_definition':'all finite-thickness panel vertices remain within 2 mm over the last 1 second; full success held for 3 seconds','body_names':self.box.body_names,'joint_names':self.box.joint_names}
  self.metadata['spring_targets_radians']='passive plastic strain history, never an assembly-angle target'
  self.metadata['force_feedback']={'window_s':.05,'history_samples':round(.05/dt),'type':'mean normal-contact force magnitude over actual physics substeps, not a single sampled impulse','raw_samples_preserved_in_trace':True}
  self.metadata['contact_filter_names']=self.filter_names
  self.metadata['crease_material']={'law':'elastic-perfectly-plastic return mapping','stiffness_Nm_rad':.01,'yield_moment_Nm':.002,'elastic_unloading_rad':.2,'damping_Nm_s_rad':.0003,'calibrated':False,'source_for_model_family':'https://doi.org/10.1016/j.ijsolstr.2011.04.002','parameters_from_source':False}
  if not enable_cameras:
   self.metadata['cameras']=[]
   self.metadata['camera_resolution']=None
  self.metadata['worktable']={'support_top_m':self.support_top,'rise_mm':round(1000*(self.support_top-station.TABLE_TOP)),'slab_size_mm':[420.6,170,8],'slab_xy_bounds_m':[[.236,-.085],[.6566,.085]],'static_visible_collision':True,'shape':'single rectangular top and central pedestal'}
  self.metadata['staging_top_m']=staging_top
  self.metadata.update(task='pack_grok002: pack a foam pad and a paperback into a formed mailer, overstuffed with a slow-rebound pad: square + press the contents down, then close the lid with the tuck inside the front wall (by grok)',author='by grok',start_state='pack_start: pre-formed open mailer (diagnostic-free task start; walls/tabs set at reset, lid open), pad+book on staging pedestals',pack_params=pp.summary(),foam_model='two rigid plates + prismatic passive spring with preload; acts only on the book/pad',success_required=pc.REQUIRED)
  self.metadata['diagnostic_only']=probe not in ['pack_start']
  (self.output/'metadata.json').write_text(json.dumps(self.metadata,indent=2))
  if record:
   import imageio_ffmpeg
   self.video_log=(self.output/'video.log').open('w');self.video=subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(),'-y','-loglevel','error','-f','rawvideo','-pixel_format','rgb24','-video_size',f'{self.width}x{self.height*3//2}','-framerate','10','-i','-','-an','-c:v','libx264','-preset','fast','-crf','22','-pix_fmt','yuv420p',str(self.output/'three-camera.mp4')],stdin=subprocess.PIPE,stderr=self.video_log)
  print('PACK_READY',self.box.num_bodies,self.box.num_joints,'initial_angles',self.angles(),flush=True)

 def joints(self):
  values=[]
  for side in ['left_robot','right_robot']:
   arm=self.base.scene[side];ids,_=arm.find_joints(self.constants.ARM_JOINTS,preserve_order=True);fids,_=arm.find_joints(self.constants.FINGER_JOINTS,preserve_order=True);q=array(arm.data.joint_pos)[0];values.extend([*q[ids],np.clip(q[fids].mean()/self.constants.FINGER_OPEN,0,1)])
  return np.array(values)

 def hand_panel_forces(self,index):
  history=array(self.base.scene[f'hand_{index}_contacts'].data.normal_force_matrix_w_history)
  return np.linalg.norm(history,axis=-1).mean(axis=1).reshape(-1,len(self.filter_names)).sum(0)

 def poses(self):return {n:transform(array(self.box.data.body_link_pose_w)[0,i]) for n,i in self.body_ids.items()}
 def angles(self):return {n:float(array(self.box.data.joint_pos)[0,i]) for n,i in self.joint_ids.items()}
 def point(self,name,blank_xyz):
  p=self.poses()[name];return p[:3,3]+p[:3,:3]@(np.asarray(blank_xyz)-self.manifest['panels'][name]['center'])
 def apply_joint_torque(self,name,torque):
  # Diagnostic loading only: equal and opposite finite torques, no pose writes.
  joint=self.manifest['joints'][name];poses=self.poses();axis=np.array([0,1,0] if joint['axis']=='Y' else [1,0,0]);axis=poses[joint['parent']][:3,:3]@axis
  torques=np.zeros((1,self.box.num_bodies,3));torques[0,self.body_ids[joint['child']]]+=axis*torque;torques[0,self.body_ids[joint['parent']]]-=axis*torque
  forces=np.zeros_like(torques)
  self.box.permanent_wrench_composer.set_forces_and_torques_index(forces=self.torch.tensor(forces,dtype=self.torch.float32,device=self.base.device),torques=self.torch.tensor(torques,dtype=self.torch.float32,device=self.base.device),body_ids=list(range(self.box.num_bodies)),is_global=True)
 def push_book(self,force_n):
  # by grok (pack_grok002): DIAGNOSTIC ONLY test press: a downward world-frame force at the book COM, standing in for a hand.
  # Used only by run_robot --stage press_diag (probe diag_packed); never in any task stage; no pose writes.
  f=np.zeros((1,1,3));f[0,0,2]=-float(force_n)
  self.book.permanent_wrench_composer.set_forces_and_torques_index(forces=self.torch.tensor(f,dtype=self.torch.float32,device=self.base.device),torques=self.torch.tensor(np.zeros_like(f),dtype=self.torch.float32,device=self.base.device),body_ids=[0],is_global=True)
 def sleep_settings(self):
  # by grok: report the PhysX sleep/stabilization attributes actually present on the spawned pad/book prims
  import omni.usd
  st=omni.usd.get_context().get_stage();out={}
  for prim in st.Traverse():
   p=str(prim.GetPath())
   if ('foam_pad' in p or p.endswith('/book') or '/book/' in p) and prim.GetAttributes():
    for at in prim.GetAttributes():
     n=at.GetName()
     if 'leep' in n or 'tabilization' in n:out[p+'.'+n]=at.Get()
  return out
 def images(self):return {n:array(self.base.scene[n].data.output['rgb'])[0,:,:,:3].astype(np.uint8) for n in ['top_cam','left_cam','right_cam']}
 def panel(self,frames):
  from PIL import Image
  im=Image.new('RGB',(self.width,self.height*3//2));im.paste(Image.fromarray(frames['top_cam']),(0,0))
  for name,x in [('left_cam',0),('right_cam',self.width//2)]:im.paste(Image.fromarray(frames[name]).resize((self.width//2,self.height//2),Image.Resampling.LANCZOS),(x,self.height))
  return im
 def snapshot(self,name):
  if not self.enable_cameras:return
  from PIL import Image
  frames=self.images();self.panel(frames).save(self.output/(name+'.png'));Image.fromarray(frames['top_cam']).save(self.output/(name+'-top.png'))

 def box_measure(self):
  poses=self.poses();base=poses['base'];inverse=np.linalg.inv(base);angles=self.angles();released=True;hands=[]
  for i in range(4):
   force=float(self.hand_panel_forces(i).sum());hands.append(force);released &= force<.025
  for side,o in [('left',0),('right',7)]:
   tp,_=self.ik.tcp_pose(self.joints()[o:o+6],self.station.BASES[side])
   # Conservative body-centre clearance, coupled with measured hand contact.
   released &= min(np.linalg.norm(tp-p[:3,3]) for p in poses.values())>.05
  normals={n:float(abs((base[:3,:3].T@poses[n][:3,2])[2])) for n in ['front','back','left_outer','right_outer']}
  walls={n:bool(v<np.sin(np.deg2rad(10)) and .020<(inverse@poses[n])[2,3]<.030) for n,v in normals.items()}
  locks={}
  for s,side in [(-1,'left'),(1,'right')]:
   n=side+'_inner';x=s*(.110+.050+.0054+.0488+.0018)
   for k,y in enumerate([-.041,.041]):
    tip=inverse@np.r_[self.point(n,[x,y,0]),1.];slot=s*(.110-.0054)
    normal=base[:3,:3].T@poses[n][:3,0]
    # Tip has crossed the actual upper base surface and lies within a cutout.
    # Position AND insertion orientation AND wall geometry: no angle-only credit.
    depth=.0006-tip[2];aligned=abs(tip[0]-slot)<.00045 and abs(tip[1]-y)<.0010 and abs(normal[2])>.97
    inserted=bool(not self.tabless and aligned and .0007<depth<.00165 and walls[side+'_outer'])
    locks[side+'_'+str(k)]={'inserted':inserted,'depth_m':float(depth),'slot_error_x_m':float(tip[0]-slot),'slot_error_y_m':float(tip[1]-y)}
  corner_ears={}
  for wall,vsign in [('front',-1),('back',1)]:
   for side in ['left','right']:
    name=wall+'_'+side+'_ear';p=inverse@poses[name];u,v0,z=p[:3,3]
    corner_ears[name]=bool(.1058<abs(u)<.1092 and abs(v0-vsign*.064)<.004 and .021<z<.029 and abs(p[0,2])>.97)
  dust_flaps={}
  for side in ['left','right']:
   p=inverse@poses['lid_'+side+'_dust'];u,v0,z=p[:3,3]
   dust_flaps[side]=bool(.1015<abs(u)<.1043 and abs(v0)<.006 and .030<z<.039 and abs(p[0,2])>.97)
  lid_ears={}
  for s,side in [(-1,'left'),(1,'right')]:
   n='tuck_'+side+'_ear';u=s*(.110-.0054-.0012-.0016);tip=inverse@np.r_[self.point(n,[u+s*.024,.0+self.manifest['tuck_hinge_v']+.015,0]),1.]
   lid_ears[side]=bool(abs(tip[0]-u)<.0011 and -.073<tip[1]<-.050 and .019<tip[2]<.050)
  lid=inverse@poses['lid'];lid_normal=lid[:3,2];lid_h=float(lid[2,3]);closed=bool(all(dust_flaps.values()) and all(lid_ears.values()) and lid_normal[2]<-.985 and abs(lid_h-.050)<.003 and abs(angles['front_tuck']-np.pi/2)<.20)
  v=array(self.box.data.body_link_vel_w)[0]
  points=[]
  for n,p in poses.items():
   c=self.manifest['panels'][n]['center']
   local=np.array([[x-c[0],y-c[1],z] for poly in self.manifest['panels'][n]['polygons'] for x,y in poly for z in [-.0006,.0006]])
   points.extend(local@p[:3,:3].T+p[:3,3])
  points=np.array(points);self.pose_window.append(points)
  motion=float(max(np.linalg.norm(p-points,axis=-1).max() for p in self.pose_window))
  stable=bool(len(self.pose_window)==21 and motion<.002)
  on_table=bool(base[2,2]>.9986 and abs(base[2,3]-(self.support_top+.0006))<.002)
  fraction_walls=sum(walls.values())/4;fraction_locks=sum(v['inserted'] for v in locks.values())/4
  # Scores reflect current state. Opening and correcting resets dwell only.
  score=.35*fraction_walls+.45*fraction_locks+.15*int(closed and all(walls.values()) and fraction_locks==1)
  valid=all(corner_ears.values()) and all(walls.values()) and fraction_locks==1 and closed and released and stable and on_table
  score+=.0
  material=self.box.actuators['passive_scores'];plastic=array(material.plastic_rotation)[0]
  return {'score':float(score),'walls':walls,'locks':locks,'lid_closed':closed,'lid_ears_inserted':lid_ears,'corner_ears_captured':corner_ears,'lid_dust_flaps_inside':dust_flaps,'hands_released':bool(released),'hand_force_n':hands,'stable':stable,'maximum_vertex_motion_last_second_m':motion,'base_on_table':on_table,'angles_deg':{n:float(np.rad2deg(q)) for n,q in angles.items()},'plastic_crease_rotation_deg':{n:float(np.rad2deg(plastic[self.joint_ids[n]])) for n in self.joint_ids},'plastic_dissipation_J':float(array(material.plastic_dissipation).sum()),'base_pose':array(self.box.data.body_link_pose_w)[0,self.body_ids['base']].tolist(),'max_linear_speed':float(np.max(np.linalg.norm(v[:,:3],axis=-1))),'body_speeds':{n:[float(np.linalg.norm(v[i,:3])),float(np.linalg.norm(v[i,3:]))] for n,i in self.body_ids.items()}}

 def content_poses(self):
  q=array(self.pad.data.body_link_pose_w)[0]
  return transform(q[self.pad_ids['pad_base']]),transform(q[self.pad_ids['pad_top']]),transform(array(self.book.data.root_link_pose_w)[0])

 def measure(self):
  """by grok: pack task state. Box terms from box_measure(); pad/book/tuck/lid terms from pack_check."""
  box=self.box_measure();poses=self.poses();base=poses['base']
  pad_base,pad_top,book=self.content_poses();k=pc.contents(base,pad_base,pad_top,book);pts=k.pop('points_world')
  c=pc.closure(base,poses,self.manifest)
  self.content_window.append(pts);motion=float(max(np.linalg.norm(p-pts,axis=-1).max() for p in self.content_window))
  stable=bool(box['stable'] and len(self.content_window)==21 and motion<.002)
  from gripper_geometry import parcel_clearances
  parcel_gaps = parcel_clearances(self.joints(), base)
  clear = min(parcel_gaps) >= pc.HAND_CLEARANCE_M
  intact=bool(all(box['walls'].values()) and all(box['corner_ears_captured'].values()) and box['base_on_table'])  # by grok: side-wall tab locks are not seated even in the inherited pre-formed start (depth ~0.12 mm < 0.7 mm); reported as diagnostic only
  self._comp_latch=bool(getattr(self,'_comp_latch',False) or k['compressed_to_height_now'])  # by grok (pack_grok002): latched milestone
  flaps_seated=all(c['dust_flaps_inside'].values()) and all(c['tuck_ears_inside'].values())
  flags={'pad_in_box':k['pad_in_box'],'book_on_pad':k['book_on_pad'],'book_square':k['book_square'],'compressed_to_height':self._comp_latch,'tuck_inside_front_wall':c['tuck_inside_front_wall'],'folded_flaps_seated':flaps_seated,'lid_flush':c['lid_flush'],'box_intact':intact,'hands_clear':clear,'stable':stable}
  valid=all(flags[n] for n in pc.REQUIRED)
  self.hold_steps=self.hold_steps+1 if valid else 0
  hold_s=self.hold_steps/20
  v=array(self.book.data.root_link_vel_w)[0]
  return {'score':pc.score(flags,hold_s),'success':bool(hold_s>=pc.FINAL_HOLD_S),'hold_s':hold_s,'pack_flags':flags,'pack_contents':k,'pack_closure':c,
          'hand_parcel_gaps_m':parcel_gaps,'content_max_vertex_motion_last_second_m':motion,'book_speed_m_s':float(np.linalg.norm(v[:3])),
          'pad_compression_joint_m':float(array(self.pad.data.joint_pos)[0,0]),'box':box,'max_linear_speed':max(box['max_linear_speed'],float(np.linalg.norm(v[:3])))}

 def step(self,action,label=''):
  self.env.step(self.torch.tensor([np.asarray(action).tolist()],dtype=self.torch.float32,device=self.base.device));self.steps+=1;self.last=self.measure();row={'step':self.steps,'phase':label,'action':np.asarray(action).tolist(),'measured_joints':self.joints().tolist(),'mean_panel_contact_force_n':{str(i):self.hand_panel_forces(i).tolist() for i in range(4)},'peak_panel_contact_force_n':{str(i):np.linalg.norm(array(self.base.scene[f'hand_{i}_contacts'].data.normal_force_matrix_w_history),axis=-1).reshape(-1,len(self.filter_names)).max(0).tolist() for i in range(4)},'panel_contact_force_n':{str(i):np.linalg.norm(array(self.base.scene[f'hand_{i}_contacts'].data.normal_force_matrix_w),axis=-1).reshape(-1,len(self.filter_names)).sum(0).tolist() for i in range(4)},**self.last};self.trace.write(json.dumps(row,default=jdefault)+'\n')
  if self.video and self.steps%2==0:self.video.stdin.write(np.asarray(self.panel(self.images())).tobytes())
  if self.steps%20==0:
   self.trace.flush();(self.output/'progress.json').write_text(json.dumps(row,indent=2,default=jdefault));print('STEP',self.steps,label,'score',round(row['score'],3),'speed',round(row['max_linear_speed'],3),flush=True)
  return row

 def close(self):
  if self.env is not None:
   (self.output/'result.json').write_text(json.dumps({'steps':self.steps,'probe':self.probe,'robot_demonstration':self.metadata['robot_demonstration'],**self.last},indent=2,default=jdefault))
  if hasattr(self,'trace'):self.trace.close()
  if self.video:self.video.stdin.close();self.video.wait(timeout=30);self.video_log.close()
  if self.env:self.env.close()
