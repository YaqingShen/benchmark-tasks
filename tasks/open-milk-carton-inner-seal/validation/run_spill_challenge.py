"""Independent unscored drop test of the pan geometry shared with the task."""
import argparse
import json
from pathlib import Path
import sys
parser=argparse.ArgumentParser()
parser.add_argument('source',type=Path)
parser.add_argument('output',type=Path)
parser.add_argument('--hz',type=int,choices=[240,480,960],default=240)
parser.add_argument('--open-control',action='store_true',help='Remove the enclosure and secondary pan; retain the same numerical support under the primary floor.')
args=parser.parse_args()
sys.path.insert(0,str(args.source.resolve()))
from spill_station import build_pan, usable_litres, X_MIN, X_MAX, Y_MIN, Y_MAX, THICK, HOOD, APRON, OUTER_RIM
from isaacsim import SimulationApp
app=SimulationApp({'headless':True,'width':64,'height':64})
import carb
import numpy as np
import omni.physx
import omni.usd
from pxr import Gf, UsdGeom, UsdPhysics, PhysxSchema, UsdUtils, Sdf
from omni.physx.scripts import particleUtils,physicsUtils

def run():
    stage=omni.usd.get_context().get_stage()
    UsdGeom.SetStageMetersPerUnit(stage,1.)
    UsdGeom.SetStageUpAxis(stage,UsdGeom.Tokens.z)
    physics=UsdPhysics.Scene.Define(stage,'/physicsScene')
    physics.CreateGravityDirectionAttr(Gf.Vec3f(0,0,-1))
    physics.CreateGravityMagnitudeAttr(9.81)
    api=PhysxSchema.PhysxSceneAPI.Apply(physics.GetPrim())
    api.CreateEnableGPUDynamicsAttr(True)
    api.CreateBroadphaseTypeAttr('GPU')
    api.CreateTimeStepsPerSecondAttr(args.hz)
    settings=carb.settings.get_settings()
    settings.set_bool('/physics/suppressReadback',False)
    settings.set_bool('/physics/updateParticlesToUsd',True)
    scenarios=[('arm_side',-1,0),('far_side',1,0),('left_end',0,1),('right_end',0,-1)]
    tests=[]
    # Four physically separated copies in one physics scene, without time-varying forcing.
    n=round(.00189/.004**3)
    xyz=np.array([(x,y,z) for z in range(48) for x in range(25) for y in range(25)],dtype=float)[:n]*.004
    for i,(name,dx,dy) in enumerate(scenarios):
        origin='/World/'+name;floor=i*4.
        build_pan(stage,origin,floor,separate_floor=True)
        if args.open_control:
            pan=stage.GetPrimAtPath(origin+'/spill_pan')
            keep={'arm_side','far_side','left_end','right_end','floor','bottom_backing'}
            for child in list(pan.GetChildren()):
                if child.GetName() not in keep:stage.RemovePrim(child.GetPath())
        # Occupied wet zone: closed conservative bounding boxes for the fallen
        # carton, discard tray, tool-holder and cap rest displace liquid.
        for label,xy,size in [('fallen_carton',(.52,.035),(.245,.102,.102)),('discard',(.455,-.18),(.20,.12,.022)),('tool_rest',(.280,.150),(.140,.110,.036)),('cap_rest',(.24,-.30),(.08,.08,.025))]:
            box=UsdGeom.Cube.Define(stage,origin+'/'+label)
            box.CreateSizeAttr(1);box.AddTranslateOp().Set(Gf.Vec3d(*xy,floor+size[2]/2))
            box.AddScaleOp().Set(Gf.Vec3f(*size));UsdPhysics.CollisionAPI.Apply(box.GetPrim())
        system=origin+'/liquid'
        particleUtils.add_physx_particle_system(stage,system,simulation_owner=Sdf.Path('/physicsScene'),
            contact_offset=.0026,rest_offset=.002,particle_contact_offset=.0038,
            solid_rest_offset=.00228,fluid_rest_offset=.00228,enable_ccd=True,solver_position_iterations=8,max_velocity=4.,max_neighborhood=96)
        material=origin+'/water'
        particleUtils.add_pbd_particle_material(stage,material,viscosity=.02,surface_tension=.03,cohesion=.01,friction=.1,damping=0.,density=1000.)
        physicsUtils.add_physics_material_to_prim(stage,stage.GetPrimAtPath(system),material)
        points=xyz+np.array([.367-.048,.035-.048,floor+.226])
        vel=np.tile([.6*dx,.6*dy,0.],(n,1))
        obj=particleUtils.add_physx_particleset_points(stage,origin+'/points',points.tolist(),vel.tolist(),[.004]*n,system,True,True,0,1.89/n,0.)
        tests.append(dict(name=name,floor=floor,attr=obj.GetPointsAttr(),outside=np.zeros(n,bool),left_inner=np.zeros(n,bool),points=points,first=None,max_overrun_mm=0.))
    sim=omni.physx.get_physx_simulation_interface()
    iface=omni.physx.get_physx_interface()
    sim.attach_stage(UsdUtils.StageCache.Get().Insert(stage).ToLongInt())
    dt=1/args.hz
    for step in range(1,4*args.hz+1):
        sim.simulate(dt,(step-1)*dt);sim.fetch_results()
        iface.update_transformations(False,True,True,False)
        for t in tests:
            a=np.asarray(t['attr'].Get())
            if a.shape!=(n,3) or not np.isfinite(a).all():raise RuntimeError('unavailable liquid positions')
            z=a[:,2]-t['floor']
            clearance=np.maximum.reduce([X_MIN-THICK-a[:,0],a[:,0]-X_MAX-THICK,Y_MIN-THICK-a[:,1],a[:,1]-Y_MAX-THICK,-.003-z,z-HOOD-.003])
            peak=float(clearance.max()*1000)
            t['max_overrun_mm']=max(t['max_overrun_mm'],peak)
            if peak>0 and t['first'] is None:
                at=int(clearance.argmax())
                t['first']=dict(step=step,id=at,x=float(a[at,0]),y=float(a[at,1]),z=float(z[at]),overrun_mm=peak)
            crossed=clearance>0
            t['left_inner']|=crossed
            # Count a primary excursion separately. Any such point outside
            # the ACTUAL secondary pan's inner faces, or above its lip, fails.
            caught=(a[:,0]>=X_MIN-APRON)&(a[:,0]<=X_MAX+APRON)&(a[:,1]>=Y_MIN-APRON)&(a[:,1]<=Y_MAX+APRON)&(z>=-.003)&(z<=OUTER_RIM-.003)
            t['outside']|=crossed if args.open_control else (crossed&~caught)
            t['z']=z
    rows=[]
    for t in tests:
        moved=float(np.median(t['points'][:,2]-t['floor']-t['z']))
        outside=int(t['outside'].sum())
        rows.append(dict(direction=t['name'],first_primary_overrun=t['first'],max_primary_overrun_mm=t['max_overrun_mm'],left_primary_ever=int(t['left_inner'].sum()),input_ml=1890,particles=n,outside_ever=outside,
            outside_equivalent_ml=1890*outside/n,median_vertical_drop_m=moved,
            final_max_z_m=float(t['z'].max()),passed=bool(moved>.17 and outside==0)))
    result=dict(model='Isaac PhysX PBD full-charge instantaneous directional release; no physical apparatus',
        fixture='open_control' if args.open_control else 'sealed_double_catch',capacity_lower_bound_l=usable_litres(),initial_bottom_m=.226,initial_horizontal_mps=.6,physics_steps=4*args.hz,
        physics_hz=args.hz,nominal_ml_per_particle=1890/n,scenarios=rows,passed=all(r['passed'] for r in rows))
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
    sim.detach_stage()
    return result['passed']
try: ok=run()
except BaseException:
    import traceback
    traceback.print_exc();ok=False
sys.stdout.flush()
app.close(exit_code=0 if ok else 1)
sys.exit(0 if ok else 1)
