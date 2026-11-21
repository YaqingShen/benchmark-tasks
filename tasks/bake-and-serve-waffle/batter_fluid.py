"""Native PBD batter with user-authorized abstract baking state swap.

Viscosity/cohesion are uncalibrated numerical parameters. This probe establishes
fluid motion and container contacts; the subsequent timer-based solid swap is an explicit debug abstraction.
"""
import numpy as np
_PATHS={}


def configure(ns):
    from cooking_state import State,BATTER_MASS_KG,TIME_SCALE
    import phase_conversion
    thermal=State()
    # Resume only after a recorded native bake-state swap. This is explicitly
    # a solid-only suffix diagnostic; final demonstrations always reset normally.
    import json
    from pathlib import Path
    cp_path=Path(ns['__file__']).with_name('checkpoint.json')
    cp=json.loads(cp_path.read_text()) if cp_path.exists() else None
    if cp is not None:
        event=cp.get('bake_state')
        if not event or event['robot_time_s']>=cp['sim_time']:
            raise ValueError('Cooking suffix requires a recorded completed bake-state swap')
        @ns['on_reset']
        def restore_baked_state(env):
            phase_conversion.STATE.update(solid=True,event=event,block_reason=None)
            print('BAKED_SUFFIX_ONLY',{'source':cp['source_run'],'step':cp['step']},flush=True)
        return
    labkit=ns['labkit'];original=labkit._author
    def author(prim_path,cfg,translation=None,orientation=None,**kwargs):
        out=original(prim_path,cfg,translation=translation,orientation=orientation,**kwargs)
        if cfg.part_name!='belgian_waffle':return out
        from isaaclab.sim.utils.stage import get_current_stage
        from pxr import Usd,UsdGeom,UsdPhysics,UsdShade,PhysxSchema,Sdf,Gf
        from omni.physx.scripts import particleUtils,physicsUtils
        import carb
        import particle_backend_compat
        particle_backend_compat.install()
        stage=get_current_stage();root=stage.GetPrimAtPath(prim_path)
        physics=PhysxSchema.PhysxSceneAPI(stage.GetPrimAtPath('/physicsScene'))
        physics.CreateEnableGPUDynamicsAttr(True)
        physics.CreateBroadphaseTypeAttr('GPU')
        carb.settings.get_settings().set_int('/physics/cudaDevice',0)
        carb.settings.get_settings().set_bool('/physics/suppressReadback',False)
        # Disable both the collision shapes and dynamics of the dormant cooked
        # mesh. A concealed solid must not supply the fluid's contact behavior.
        active_colliders=[]
        for p in Usd.PrimRange(root):
            if p.HasAPI(UsdPhysics.CollisionAPI):
                if UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get():active_colliders.append(str(p.GetPath()))
                UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Set(False)
            if p.IsA(UsdGeom.Gprim):UsdGeom.Imageable(p).MakeInvisible()
            if p.HasAPI(UsdPhysics.RigidBodyAPI):PhysxSchema.PhysxRigidBodyAPI.Apply(p).CreateDisableGravityAttr(True)
        UsdGeom.Imageable(root).MakeInvisible()
        parent=prim_path.rsplit('/',1)[0]
        system=parent+'/batter_system';particles=parent+'/batter_particles';material=parent+'/batter_material'
        particleUtils.add_physx_particle_system(stage,system,simulation_owner=Sdf.Path('/physicsScene'),
            contact_offset=.0016,rest_offset=.0012,particle_contact_offset=.0024,
            solid_rest_offset=.0014,fluid_rest_offset=.0014,enable_ccd=True,
            solver_position_iterations=8,max_velocity=2.,max_neighborhood=96)
        particleUtils.add_pbd_particle_material(stage,material,viscosity=60.,surface_tension=.03,
            cohesion=.02,friction=.2,damping=.0,density=1050.)
        physicsUtils.add_physics_material_to_prim(stage,stage.GetPrimAtPath(system),material)
        mat=UsdShade.Material(stage.GetPrimAtPath(material))
        shader=UsdShade.Shader.Define(stage,material+'/surface');shader.CreateIdAttr('UsdPreviewSurface')
        shader.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(.80,.64,.37))
        shader.CreateInput('roughness',Sdf.ValueTypeNames.Float).Set(.28)
        mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(),'surface')
        UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath(system)).Bind(mat)
        # Nominal 250 mL fill at assumed 1050 kg/m3, preheated-grid debug reset.
        origin=np.asarray(ns['HOME'],float)
        spacing=.0028
        local=np.array([(x,y,ns['ZA']+.002+z) for x in np.arange(-.077,.078,spacing)
                        for y in np.arange(-.077,.078,spacing) if x*x+y*y<.077**2
                        for z in np.arange(5)*spacing])
        points=local+origin
        obj=particleUtils.add_physx_particleset_points(stage,particles,points.tolist(),
            np.zeros_like(points).tolist(),[.0028]*len(points),system,True,True,0,BATTER_MASS_KG/len(points),0.)
        obj.CreateDisplayColorAttr([Gf.Vec3f(.80,.64,.37)])
        particleUtils.add_physx_particle_smoothing(stage,system,enabled=True,strength=.5)
        particleUtils.add_physx_particle_isosurface(stage,system,enabled=True,
            grid_spacing=.0015,surface_distance=.0017,num_mesh_smoothing_passes=2,
            max_vertices=200000,max_triangles=400000,max_subgrids=4096)
        carb.settings.get_settings().set_bool('/physics/updateParticlesToUsd',True)
        _PATHS.update(system=system,particles=particles,cooked_path=prim_path,initial=points.copy(),count=len(points),step=0,active_collider_paths=active_colliders)
        print('BATTER_AUTHORED',{'count':len(points),'mass_kg':BATTER_MASS_KG,'fluid':True,'cooked_colliders_enabled':False},flush=True)
        return out
    labkit._author=author
    @ns['on_reset']
    def reset_particle_output(env):
        phase_conversion.reset()
        thermal.__dict__.update(State().__dict__)
        import carb
        carb.settings.get_settings().set_bool('/physics/updateParticlesToUsd',True)
        from isaaclab.sim.utils.stage import get_current_stage
        from pxr import Usd,UsdGeom,UsdPhysics
        stage=get_current_stage()
        colliders=[]
        for p in Usd.PrimRange(stage.GetPrimAtPath(_PATHS['cooked_path'])):
            if p.IsA(UsdGeom.Imageable):UsdGeom.Imageable(p).MakeInvisible()
            if p.HasAPI(UsdPhysics.CollisionAPI):
                enabled=UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get()
                colliders.append(bool(enabled))
        if any(colliders):raise RuntimeError('Dormant cooked body has active collision shapes')
        print('BATTER_COOKED_BODY_CHECK',{'active_colliders':sum(colliders),'hidden':True},flush=True)
        from pxr import PhysxSchema
        sc=PhysxSchema.PhysxSceneAPI(stage.GetPrimAtPath('/physicsScene'))
        sy=PhysxSchema.PhysxParticleSystem(stage.GetPrimAtPath(_PATHS['system']))
        import omni.physx
        print('BATTER_SCENE_CHECK',{'readback_suppressed':omni.physx.get_physx_interface().is_readback_suppressed(),'gpu':sc.GetEnableGPUDynamicsAttr().Get(),'system_enabled':sy.GetParticleSystemEnabledAttr().Get(),'owners':[str(x) for x in sy.GetSimulationOwnerRel().GetTargets()]},flush=True)
    @ns['on_step']
    def record(env):
        if not _PATHS:return
        if phase_conversion.STATE['solid']:
            ns['trace_evidence'].feedback={'kind':'abstract baking state','solid':True,'thermal':thermal.report(),'material_transition':phase_conversion.STATE['event']}
            return
        from isaaclab.sim.utils.stage import get_current_stage
        from pxr import UsdGeom,PhysxSchema
        import omni.physx,carb
        carb.settings.get_settings().set_bool('/physics/updateParticlesToUsd',True)
        omni.physx.get_physx_interface().update_transformations(False,True,True,False)
        env.unwrapped.sim.physics_manager.forward()
        stage=get_current_stage();ps=stage.GetPrimAtPath(_PATHS['particles'])
        api=PhysxSchema.PhysxParticleSetAPI(ps)
        raw=api.GetSimulationPointsAttr().Get()
        pts=UsdGeom.Points(ps).GetPointsAttr().Get()
        if pts is None or not len(pts):pts=raw
        fabric_motion=None;particle_mem=None
        try:
            import usdrt,omni.usd
            rt=usdrt.Usd.Stage.Attach(omni.usd.get_context().get_stage_id())
            rp=rt.GetPrimAtPath(_PATHS['particles'])
            rpts=rp.GetAttribute('points').Get()
            ra=np.asarray(rpts,dtype=float)
            if ra.shape==_PATHS['initial'].shape:
                fabric_motion=float(np.linalg.norm(ra-_PATHS['initial'],axis=1).max()*1000)
                if fabric_motion>.01:pts=rpts
            if _PATHS['step']==0:
                from pxr import PhysicsSchemaTools,Sdf
                from omni.physx.bindings import _physx
                stats=_physx.PhysicsSceneStats()
                ok=_physx.acquire_physx_statistics_interface().get_physx_scene_statistics(
                    omni.usd.get_context().get_stage_id(),PhysicsSchemaTools.sdfPathToInt(Sdf.Path('/physicsScene')),stats)
                print('BATTER_NATIVE_STATS',{'ok':ok,'particle_bytes':stats.gpu_mem_particles,'particle_contacts':stats.gpu_mem_particle_contacts,'fabric_attrs':[str(x.GetName()) for x in rp.GetAttributes()]},flush=True)
        except Exception as error:
            if _PATHS['step']==0:print('BATTER_NATIVE_STATS_ERROR',repr(error),flush=True)
        a=np.asarray(pts,dtype=float);_PATHS['step']+=1
        finite=bool(a.shape==_PATHS['initial'].shape and np.isfinite(a).all())
        if not finite:raise RuntimeError('Invalid native batter particle state')
        w=labkit._part_state(env,next(o for o in ns['OBJECTS'] if o.name=='waffle'))
        closed=abs(w.joints['lid'])<np.radians(5.) and w.joints['latch']>-.003
        thermal.advance(float(env.unwrapped.step_dt),closed=closed)
        shifted=np.linalg.norm(a-_PATHS['initial'],axis=1)
        mold_local,center,mold_rotation=phase_conversion.mold_coordinates(ns,w,a)
        r=np.linalg.norm(mold_local[:,:2],axis=1)
        velocities=UsdGeom.Points(ps).GetVelocitiesAttr().Get()
        phase_conversion.attempt(ns,env,_PATHS,a,velocities,thermal,w)
        evidence={'kind':'native PBD batter and heating diagnostic','thermal':thermal.report(),'particle_count':len(a),'mass_kg':BATTER_MASS_KG,
                  'max_motion_mm':float(shifted.max()*1000),'mean_motion_mm':float(shifted.mean()*1000),
                  'radial_95_mm':float(np.quantile(r,.95)*1000),
                  'height_range_mm':(mold_local[:,2].min()*1000,mold_local[:,2].max()*1000),
                  'outside_mold_fraction':float(np.mean((r>.109)|(np.abs(mold_local[:,2])>.043))),
                  'cooked':phase_conversion.STATE['solid'],'phase_block_reason':phase_conversion.STATE['block_reason'],'recipe_calibrated':False,'debug_time_compressed':TIME_SCALE>1,
                  'fabric_motion_mm':fabric_motion,'simulation_points_motion_mm':None if raw is None or len(raw)!=len(a) else float(np.linalg.norm(np.asarray(raw)-_PATHS['initial'],axis=1).max()*1000)}
        ns['trace_evidence'].feedback=evidence
        if _PATHS['step']%20==0:
            import json
            print('BATTER '+json.dumps(evidence),flush=True)
