"""Native PhysX PBD liquid diagnostic; never prescribes particle trajectories.
Particle mass represents 1.89 L at nominal 1030 kg/m3. Rheology and dimensions
are uncalibrated. Counts cannot resolve films or drops smaller than one particle.
"""
import os,json
import numpy as np
VOLUME_ML=1890.;DENSITY=1030.;MASS=VOLUME_ML*1e-6*DENSITY
# Native particles carry a small share of the actual milk inertia. The rest is
# assigned once to the dynamic carton. All 1.89 L remain represented spatially.
PARTICLE_INERTIA_FRACTION=.02
PARTICLE_MASS=MASS*PARTICLE_INERTIA_FRACTION
ENABLED=os.environ.get('MILK_FLUID','1')=='1'
STATE={}

def configure(ns):
 if not ENABLED:return
 labkit=ns['labkit'];original=labkit._author
 def author(prim_path,cfg,**kwargs):
  out=original(prim_path,cfg,**kwargs)
  if cfg.part_name!='cap_rest':return out
  from isaaclab.sim.utils.stage import get_current_stage
  from pxr import UsdShade,PhysxSchema,Sdf,Gf
  from omni.physx.scripts import particleUtils,physicsUtils
  import carb,particle_backend_compat
  particle_backend_compat.install();stage=get_current_stage()
  physics=PhysxSchema.PhysxSceneAPI(stage.GetPrimAtPath('/physicsScene'))
  physics.CreateEnableGPUDynamicsAttr(True);physics.CreateBroadphaseTypeAttr('GPU')
  carb.settings.get_settings().set_int('/physics/cudaDevice',0)
  carb.settings.get_settings().set_bool('/physics/suppressReadback',False)
  parent=prim_path.rsplit('/',1)[0];system=parent+'/milk_system';particles=parent+'/milk_particles';material=parent+'/milk_material'
  particleUtils.add_physx_particle_system(stage,system,simulation_owner=Sdf.Path('/physicsScene'),
   contact_offset=.0026,rest_offset=.002,particle_contact_offset=.0038,
   solid_rest_offset=.00228,fluid_rest_offset=.00228,enable_ccd=True,
   solver_position_iterations=8,max_velocity=3.,max_neighborhood=96)
  particleUtils.add_pbd_particle_material(stage,material,viscosity=.02,surface_tension=.03,cohesion=.01,friction=.1,damping=0.,density=DENSITY*PARTICLE_INERTIA_FRACTION)
  physicsUtils.add_physics_material_to_prim(stage,stage.GetPrimAtPath(system),material)
  mat=UsdShade.Material(stage.GetPrimAtPath(material));shader=UsdShade.Shader.Define(stage,material+'/surface');shader.CreateIdAttr('UsdPreviewSurface')
  shader.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(.94,.93,.89));shader.CreateInput('roughness',Sdf.ValueTypeNames.Float).Set(.2)
  mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(),'surface');UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath(system)).Bind(mat)
  # A 4 mm lattice fills from the bottom, with a partial top layer; no overlap
  # with the carton shell, gable, spout, cap, or seal is imposed at reset.
  count=round(VOLUME_ML*1e-6/.004**3)
  local=np.array([(x,y,ns['BASE_FOLD']+.0025+z*.004) for z in range(48) for x in np.linspace(-.0475,.0475,25) for y in np.linspace(-.0475,.0475,25)])[:count]
  points=local+ns['P'];obj=particleUtils.add_physx_particleset_points(stage,particles,points.tolist(),np.zeros_like(points).tolist(),[.004]*len(points),system,True,True,0,PARTICLE_MASS/len(points),0.)
  obj.CreateDisplayColorAttr([Gf.Vec3f(.94,.93,.89)])
  particleUtils.add_physx_particle_smoothing(stage,system,enabled=True,strength=.5)
  particleUtils.add_physx_particle_isosurface(stage,system,enabled=True,grid_spacing=.0025,surface_distance=.0025,num_mesh_smoothing_passes=2,max_vertices=250000,max_triangles=500000,max_subgrids=8192)
  carb.settings.get_settings().set_bool('/physics/updateParticlesToUsd',True)
  STATE.update(system=system,particles=particles,initial=points.copy(),escaped=set(),step=0)
  print('MILK_FLUID_AUTHORED '+json.dumps({'particle_count':len(points),'volume_ml':VOLUME_ML,'represented_milk_mass_kg':MASS,'dynamic_particle_mass_kg':PARTICLE_MASS,'ml_per_particle':VOLUME_ML/len(points),'physical_calibration':False}),flush=True)
  return out
 labkit._author=author
 @ns['on_reset']
 def reset(env):
  from isaaclab.sim.utils.stage import get_current_stage
  from pxr import UsdGeom,Vt
  stage=get_current_stage();particles=UsdGeom.Points(stage.GetPrimAtPath(STATE['particles']))
  before=np.asarray(particles.GetPointsAttr().Get(),dtype=float)
  initial=STATE['initial'].astype(np.float32)
  # Reset is the only post-authoring particle write. No writes occur in record().
  particles.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(initial))
  particles.GetVelocitiesAttr().Set(Vt.Vec3fArray.FromNumpy(np.zeros_like(initial)))
  print('MILK_PARTICLE_RESET '+json.dumps({'particle_count':len(initial),'pre_reset_max_motion_mm':float(np.linalg.norm(before-initial,axis=1).max()*1000)}),flush=True)
  STATE['escaped']=set();STATE['step']=0;STATE['previous_local']=None
 @ns['on_step']
 def record(env):
  from isaaclab.sim.utils.stage import get_current_stage
  from pxr import UsdGeom,PhysxSchema
  import omni.physx,carb,omni.usd,usdrt
  carb.settings.get_settings().set_bool('/physics/updateParticlesToUsd',True)
  omni.physx.get_physx_interface().update_transformations(False,True,True,False)
  env.unwrapped.sim.physics_manager.forward();stage=get_current_stage();p=stage.GetPrimAtPath(STATE['particles'])
  points=UsdGeom.Points(p).GetPointsAttr().Get()
  rt=usdrt.Usd.Stage.Attach(omni.usd.get_context().get_stage_id());rp=rt.GetPrimAtPath(STATE['particles']);raw=rp.GetAttribute('points').Get();a=np.asarray(raw,dtype=float)
  if a.shape!=STATE['initial'].shape:a=np.asarray(points,dtype=float)
  if a.shape!=STATE['initial'].shape or not np.isfinite(a).all():raise RuntimeError('Invalid native milk particle readback')
  c=labkit._part_state(env,ns['CARTON']);local=(a-c.pos)@c.rotation()
  roof=ns['H']+(ns['PEAK']-ns['H'])*(1-2*np.abs(local[:,0])/ns['W'])
  # Count each escaped particle once, including particles retained in the tray.
  escaped=(np.abs(local[:,0])>ns['W']/2+.003)|(np.abs(local[:,1])>ns['DEP']/2+.003)|(local[:,2]<-.003)|(local[:,2]>roof+.008)
  new=sorted(set(np.flatnonzero(escaped).tolist())-STATE['escaped'])
  if new:
   previous=STATE.get('previous_local')
   event={'step':STATE['step']+1,'new_escaped':len(new),'particles':[{'id':i,'initial_local':(STATE['initial'][i]-ns['P']).tolist(),'previous_local':previous[i].tolist() if previous is not None else None,'local':local[i].tolist()} for i in new]}
   path=os.environ.get('TASK_TRACE_PATH')
   if path:
    with open(os.path.join(os.path.dirname(path),'liquid_escapes.jsonl'),'a') as f:f.write(json.dumps(event)+'\n')
  STATE['previous_local']=local.copy()
  STATE['escaped'].update(np.flatnonzero(escaped).tolist());STATE['step']+=1
  evidence={'native_pbd':True,'count':len(a),'escaped_particles':len(STATE['escaped']),'escaped_ml':len(STATE['escaped'])*VOLUME_ML/len(a),'resolution_ml':VOLUME_ML/len(a),'max_motion_mm':float(np.linalg.norm(a-STATE['initial'],axis=1).max()*1000),'z_range_local_m':[float(local[:,2].min()),float(local[:,2].max())],'physical_calibration':False}
  ns['TRACK']['liquid']=evidence
  if STATE['step']%20==0:print('MILK_LIQUID '+json.dumps(evidence),flush=True)
