"""Wet-zone boundaries in metres, relative to the station table surface."""
import json
import os
from pathlib import Path

X_MIN, X_MAX = .095, .695
Y_MIN, Y_MAX = -.400, .400
RIM = .040
THICK = .003
# Side and far splash screens extend from the base to 160 mm.
SCREENS = .160
HOOD = .550
APRON = .040
OUTER_RIM = .030
WORKING_DEPTH = .020
BLOCKED_AREA = .120

def usable_litres():
    return ((X_MAX-X_MIN)*(Y_MAX-Y_MIN)-BLOCKED_AREA)*WORKING_DEPTH*1000

def build_pan(stage, parent, floor_z, separate_floor=False):
    from pxr import Gf, UsdGeom, UsdPhysics
    root = parent + '/spill_pan'
    x, y = (X_MIN+X_MAX)/2, (Y_MIN+Y_MAX)/2
    width, length = X_MAX-X_MIN, Y_MAX-Y_MIN
    def box(name, pos, size, collide, color):
        cube=UsdGeom.Cube.Define(stage, root+'/'+name)
        cube.CreateSizeAttr(1.)
        cube.AddTranslateOp().Set(Gf.Vec3d(*pos))
        cube.AddScaleOp().Set(Gf.Vec3f(*size))
        cube.CreateDisplayColorAttr([Gf.Vec3f(*color)])
        if collide:
            UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        return str(cube.GetPath())
    # The coloured face and existing table collider share the pan-floor datum.
    # The standalone spill fixture has its own colliding bottom at that same datum.
    h=.003
    box('floor',(x,y,floor_z-h/2+(0 if separate_floor else .00002)),
        (width+2*THICK,length+2*THICK,h),separate_floor,(.42,.49,.50))
    walls=[]
    for name,edge,height in [('arm_side',X_MIN-THICK/2,RIM),('far_side',X_MAX+THICK/2,SCREENS)]:
        walls.append(box(name,(edge,y,floor_z+height/2),(THICK,length+2*THICK,height),True,(.29,.44,.46)))
    for name,edge in [('left_end',Y_MAX+THICK/2),('right_end',Y_MIN-THICK/2)]:
        walls.append(box(name,(x,edge,floor_z+SCREENS/2),(width,THICK,SCREENS),True,(.29,.44,.46)))
    # A gasketed clear hood completes the wet boundary above the rim.
    # Outward numerical backing prevents a 240 Hz particle step from crossing
    # a thin sheet between collision samples. Interior faces stay at their
    # physical coordinates. The real panels are 5 mm clear polycarbonate.
    backing=.030
    hood=[]
    for name,px,py,sx,sy in [
        ('far_clear',X_MAX+backing/2,y,backing,length+2*backing),
        ('left_clear',x,Y_MAX+backing/2,width,backing),
        ('right_clear',x,Y_MIN-backing/2,width,backing),
        ('left_sleeve',X_MIN-backing/2,y+length/4,backing,length/2),
        ('right_sleeve',X_MIN-backing/2,y-length/4,backing,length/2)]:
        hood.append(box(name,(px,py,floor_z+HOOD/2),(sx,sy,HOOD),True,(.65,.82,.85)))
    hood.append(box('sealed_lid',(x,y,floor_z+HOOD+backing/2),
                    (width+2*backing,length+2*backing,backing),True,(.65,.82,.85)))
    for path in hood:
        UsdGeom.Imageable(stage.GetPrimAtPath(path)).MakeInvisible()
    if separate_floor:
        # Same supporting surface as the 60 mm stock station worktop.
        box('bottom_backing',(x,y,floor_z-.030),(width+2*backing,length+2*backing,.060),True,(.42,.49,.50))
    for side in (-1,1):
        box('clear_frame_'+('left' if side>0 else 'right'),
            (X_MAX+THICK/2,y+side*(length/2+THICK/2),floor_z+HOOD/2),
            (.004,.004,HOOD),False,(.40,.56,.56))
    # A second seamless pan extends past all four primary seams. Its
    # stainless bottom is 3 mm below the inner working datum in the fixture.
    # The calibrated station table supplies the common supporting collider.
    outer=[]
    outer_z=floor_z-.003
    ow,ol=width+2*APRON,length+2*APRON
    # Render the 40 mm annular ledge just above the supporting table.
    for label,xx,yy,dx,dy in [('near',X_MIN-APRON/2,y,APRON,ol),
                              ('far',X_MAX+APRON/2,y,APRON,ol),
                              ('left',x,Y_MAX+APRON/2,width,APRON),
                              ('right',x,Y_MIN-APRON/2,width,APRON)]:
        box('apron_'+label,(xx,yy,floor_z-.001+.00004),(dx,dy,.002),False,(.31,.41,.43))
    if separate_floor:
        box('secondary_bottom',(x,y,outer_z-.030),(ow+2*THICK,ol+2*THICK,.060),True,(.31,.41,.43))
    for name,xx,yy,dx,dy in [('near',X_MIN-APRON-THICK/2,y,THICK,ol+2*THICK),
                              ('far',X_MAX+APRON+THICK/2,y,THICK,ol+2*THICK),
                              ('left',x,Y_MAX+APRON+THICK/2,ow,THICK),
                              ('right',x,Y_MIN-APRON-THICK/2,ow,THICK)]:
        outer.append(box('secondary_'+name,(xx,yy,outer_z+OUTER_RIM/2),
                         (dx,dy,OUTER_RIM),True,(.26,.39,.42)))
    return walls+hood+outer

def configure(ns):
    labkit=ns['labkit']
    original=labkit._author
    def author(prim_path,cfg,**kwargs):
        result=original(prim_path,cfg,**kwargs)
        if cfg.part_name=='cap_rest':
            from isaaclab.sim.utils.stage import get_current_stage
            stage=get_current_stage()
            parent=str(prim_path).rsplit('/',1)[0]
            ns['PAN_WALLS']=build_pan(stage,parent,ns['TOP'])
            # Each liquid-tight sleeve deforms around its own arm. Only its
            # own arm is excluded from that sleeve proxy's solid contacts.
            from pxr import UsdPhysics, Sdf
            from yam_arena.robot import asset_link_paths
            for side,arm in [('left','LeftArm'),('right','RightArm')]:
                prim=stage.GetPrimAtPath(parent+'/spill_pan/'+side+'_sleeve')
                rel=UsdPhysics.FilteredPairsAPI.Apply(prim).CreateFilteredPairsRel()
                for relative in asset_link_paths().values():
                    rel.AddTarget(Sdf.Path(parent+'/'+arm+'/'+str(relative).strip('/')))
        return result
    labkit._author=author

    @ns['on_reset']
    def inspect(env):
        from isaaclab.sim.utils.stage import get_current_stage
        from pxr import Gf, Usd, UsdShade, UsdPhysics, Sdf
        stage=get_current_stage()
        material=UsdShade.Material.Define(stage,'/World/WashableBootFinish')
        shader=UsdShade.Shader.Define(stage,'/World/WashableBootFinish/Surface')
        shader.CreateIdAttr('UsdPreviewSurface')
        shader.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(.08,.39,.43))
        shader.CreateInput('roughness',Sdf.ValueTypeNames.Float).Set(.82)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(),'surface')
        fitted=[]
        for arm in ('LeftArm','RightArm'):
            root=stage.GetPrimAtPath('/World/envs/env_0/'+arm)
            for prim in Usd.PrimRange(root):
                if prim.GetName() not in ('gripper','tip_left','tip_right') or not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    continue
                # The unsuffixed child instances the visual mesh. The numbered
                # siblings carry the original collision shapes and materials.
                visual=prim.GetChild(prim.GetName())
                if not visual.IsValid():
                    raise RuntimeError('Missing protected visual mesh: '+str(prim.GetPath()))
                binding=UsdShade.MaterialBindingAPI.Apply(visual)
                binding.Bind(material,UsdShade.Tokens.strongerThanDescendants)
                mesh=visual.GetChild(prim.GetName())
                resolved=UsdShade.MaterialBindingAPI(mesh).ComputeBoundMaterial()[0]
                if not resolved or resolved.GetPath()!=material.GetPath():
                    raise RuntimeError('Protective exterior material did not resolve on '+str(mesh.GetPath()))
                fitted.append(str(mesh.GetPath()))
        if len(fitted)!=6 or not all(stage.GetPrimAtPath(p).HasAPI(UsdPhysics.CollisionAPI) for p in ns['PAN_WALLS']):
            raise RuntimeError('Incomplete wet-zone protection geometry: '+repr(fitted))
        if os.environ.get('TASK_TRACE_PATH'):
            out=Path(os.environ['TASK_TRACE_PATH']).with_name('wet_station.json')
            out.write_text(json.dumps(dict(walls=ns['PAN_WALLS'],covered_links=fitted,
                capacity_lower_bound_l=usable_litres(),product_temperature_measured=False,
                fitted_exterior_contact_geometry=True,boot_seal_tested_physically=False),indent=2))
