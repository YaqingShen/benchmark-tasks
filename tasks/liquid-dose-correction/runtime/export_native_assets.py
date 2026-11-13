"""Export simplified visuals plus compound collision geometry to OpenUSD.

Run with Blender, after build_scene.py. Exported rigid assets are preparation
for Isaac validation, not a completed task: threaded cap constraints and the
pipette's compliant mechanism still need a native backend implementation.
"""
import bpy
import hashlib
import json
import math
from pathlib import Path
from mathutils import Matrix
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'assets'/'native'
OUT.mkdir(parents=True,exist_ok=True)
SCENE=ROOT.parent.parent/'outputs'/'formulation-prototype'/'formulation-scene.blend'
bpy.ops.wm.open_mainfile(filepath=str(SCENE))
CONFIG=json.loads((ROOT/'config.json').read_text())

# Convert visible lettering/linework into ordinary mesh geometry for portable
# renderers. The saved authoring scene is not overwritten.
bpy.ops.object.select_all(action='DESELECT')
for obj in list(bpy.context.scene.objects):
    if obj.type in {'FONT','CURVE'}:
        obj.select_set(True);bpy.context.view_layer.objects.active=obj
        bpy.ops.object.convert(target='MESH')
        bpy.context.object.select_set(False)

SPECS={
 'bottle_A':{'mass':.12+.2*CONFIG['density_g_ml']['A'],'dynamic':True},
 'bottle_B':{'mass':.12+.2*CONFIG['density_g_ml']['B'],'dynamic':True},
 'bottle_A_cap':{'mass':.005,'dynamic':True},
 'bottle_B_cap':{'mass':.005,'dynamic':True},
 'cylinder_A':{'mass':.045,'dynamic':True},
 'cylinder_B':{'mass':.045,'dynamic':True},
 'pipette_A':{'mass':.003,'dynamic':True},
 'pipette_B':{'mass':.003,'dynamic':True},
 'beaker':{'mass':.12,'dynamic':True},
 'digital_balance':{'mass':1.,'dynamic':False},
 'waste':{'mass':.06,'dynamic':False},
 'trash_bin':{'mass':.1,'dynamic':False},
}

SPECS={n:s for n,s in SPECS.items() if n in bpy.data.objects}

def physical_mesh(stage,path,points,faces,material):
    mesh=UsdGeom.Mesh.Define(stage,path)
    mesh.CreatePointsAttr([Gf.Vec3f(*p) for p in points])
    mesh.CreateFaceVertexCountsAttr([len(f) for f in faces])
    mesh.CreateFaceVertexIndicesAttr([i for f in faces for i in f])
    mesh.CreateSubdivisionSchemeAttr('none')
    mesh.CreatePurposeAttr('guide')
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr('convexHull')
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material,materialPurpose='physics')
    return mesh


def box(stage,path,center,size,material):
    cube=UsdGeom.Cube.Define(stage,path);cube.CreateSizeAttr(1.)
    cube.AddTranslateOp().Set(Gf.Vec3d(*center));cube.AddScaleOp().Set(Gf.Vec3f(*size))
    cube.CreatePurposeAttr('guide');UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(material,materialPurpose='physics')


def ring(stage,base,z0,z1,ri0,ro0,ri1,ro1,material,n=24):
    faces=[(0,3,2,1),(4,5,6,7),(0,1,5,4),(1,2,6,5),(2,3,7,6),(3,0,4,7)]
    for i in range(n):
        a0=2*math.pi*i/n;a1=2*math.pi*(i+1)/n
        points=[]
        for z,ri,ro in [(z0,ri0,ro0),(z1,ri1,ro1)]:
            for r,a in [(ri,a0),(ro,a0),(ro,a1),(ri,a1)]:
                points.append((r*math.cos(a),r*math.sin(a),z))
        physical_mesh(stage,base+f'_{i}',points,faces,material)


def solid_cylinder(stage,path,r,h,z,material):
    obj=UsdGeom.Cylinder.Define(stage,path)
    obj.CreateRadiusAttr(r);obj.CreateHeightAttr(h);obj.CreateAxisAttr('Z')
    obj.AddTranslateOp().Set(Gf.Vec3d(0,0,z));obj.CreatePurposeAttr('guide')
    UsdPhysics.CollisionAPI.Apply(obj.GetPrim())
    UsdShade.MaterialBindingAPI.Apply(obj.GetPrim()).Bind(material,materialPurpose='physics')


def add_colliders(stage,root,name,material):
    base=str(root.GetPath())+'/collisions'
    UsdGeom.Xform.Define(stage,base)
    if name.startswith('bottle_') and not name.endswith('_cap'):
        solid_cylinder(stage,base+'/bottom',.043,.004,.002,material)
        ring(stage,base+'/body',.004,.12,.0405,.043,.0405,.043,material)
        ring(stage,base+'/shoulder',.12,.174,.0405,.043,.0185,.021,material)
        ring(stage,base+'/neck',.174,.194,.0185,.021,.0185,.021,material)
    elif name.endswith('_cap'):
        ring(stage,base+'/wall',0,.024,.0217,.024,.0217,.024,material)
        solid_cylinder(stage,base+'/top',.024,.003,.0255,material)
    elif name.startswith('cylinder_'):
        solid_cylinder(stage,base+'/foot',.026,.008,.004,material)
        solid_cylinder(stage,base+'/bottom',.0125,.005,.0105,material)
        ring(stage,base+'/wall',.013,.194,.011,.0125,.011,.0125,material)
    elif name=='beaker':
        solid_cylinder(stage,base+'/bottom',.044,.004,.002,material)
        ring(stage,base+'/wall',.004,.122,.0415,.044,.0415,.044,material)
    elif name=='waste':
        solid_cylinder(stage,base+'/bottom',.025,.004,.002,material)
        ring(stage,base+'/wall',.004,.07,.023,.025,.028,.030,material)
    elif name.startswith('pipette_'):
        # The first export is a rigid proxy for reach/collision checks only.
        # It must not be passed off as a physical squeezing mechanism.
        solid_cylinder(stage,base+'/stem',.003,.104,.052,material)
        solid_cylinder(stage,base+'/bulb',.012,.047,.1325,material)
    elif name=='digital_balance':
        box(stage,base+'/body',(0,0,.022),(.166,.159,.04),material)
        # The pan is offset in Y in the visual model.
        pan=UsdGeom.Cylinder.Define(stage,base+'/pan')
        pan.CreateRadiusAttr(.065);pan.CreateHeightAttr(.009);pan.CreateAxisAttr('Z')
        pan.AddTranslateOp().Set(Gf.Vec3d(0,.013,.047));pan.CreatePurposeAttr('guide')
        UsdPhysics.CollisionAPI.Apply(pan.GetPrim())
        UsdShade.MaterialBindingAPI.Apply(pan.GetPrim()).Bind(material,materialPurpose='physics')
    elif name=='trash_bin':
        box(stage,base+'/bottom',(0,0,.004),(.10,.09,.008),material)
        for i,x in enumerate((-.048,.048)):box(stage,base+f'/side_{i}',(x,0,.065),(.004,.09,.13),material)
        for i,y in enumerate((-.043,.043)):box(stage,base+f'/front_back_{i}',(0,y,.065),(.10,.004,.13),material)
    else:raise ValueError(name)


report={'status':'EXPORTED_RIGID_ASSETS_ONLY','mechanisms_ready':False,
 'limitations':['Cap threading/attachment not implemented in these assets.','Pipettes are rigid proxies; squeeze physics pending.','Liquid mass is initial only; runtime update pending.','Collider proxies are unvalidated in PhysX.'],
 'table_top_m':.7475,'assets':{}}
rotation=Matrix.Rotation(-math.pi/2,4,'Z')
for name,spec in SPECS.items():
    obj=bpy.data.objects[name]
    world=obj.matrix_world.copy();parent=obj.parent;basis=obj.matrix_basis.copy()
    native=rotation@world
    native.translation+=__import__('mathutils').Vector((.42,0,.7475))
    q=native.to_quaternion()
    obj.parent=None;obj.matrix_world=Matrix.Identity(4)
    bpy.ops.object.select_all(action='DESELECT')
    selected=[]
    def collect(o):
        if name.startswith('bottle_') and not name.endswith('_cap') and (o.name.endswith('_cap') or '_liquid' in o.name):
            return
        o.select_set(True);selected.append(o)
        for child in o.children:collect(child)
    collect(obj)
    path=OUT/(name+'.usda')
    bpy.ops.wm.usd_export(filepath=str(path),selected_objects_only=True,visible_objects_only=False,
       export_animation=False,export_materials=True,export_textures=False,relative_paths=True,
       root_prim_path='')
    obj.parent=parent;obj.matrix_basis=basis
    stage=Usd.Stage.Open(str(path))
    UsdGeom.SetStageMetersPerUnit(stage,1.)
    UsdGeom.SetStageUpAxis(stage,UsdGeom.Tokens.z)
    prim=stage.GetPrimAtPath('/'+name)
    if not prim:raise RuntimeError('Missing exported root: '+name)
    stage.SetDefaultPrim(prim)
    # Blender emits materials beside the default prim. Move them inside it,
    # otherwise a reference to the default prim loses all visual bindings.
    old=Sdf.Path('/_materials');new=Sdf.Path('/'+name+'/Looks')
    if stage.GetPrimAtPath(old):
        Sdf.CopySpec(stage.GetRootLayer(),old,stage.GetRootLayer(),new)
        for child in Usd.PrimRange(prim):
            for rel in child.GetRelationships():
                targets=rel.GetTargets()
                if any(t.HasPrefix(old) for t in targets):
                    rel.SetTargets([t.ReplacePrefix(old,new) if t.HasPrefix(old) else t for t in targets])
            for attr in child.GetAttributes():
                targets=attr.GetConnections()
                if any(t.HasPrefix(old) for t in targets):
                    attr.SetConnections([t.ReplacePrefix(old,new) if t.HasPrefix(old) else t for t in targets])
        stage.RemovePrim(old)
    rigid=UsdPhysics.RigidBodyAPI.Apply(prim)
    rigid.CreateRigidBodyEnabledAttr(True);rigid.CreateKinematicEnabledAttr(not spec['dynamic'])
    UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(spec['mass'])
    mat=UsdShade.Material.Define(stage,'/'+name+'/physics_material')
    pm=UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    pm.CreateStaticFrictionAttr(.65);pm.CreateDynamicFrictionAttr(.55);pm.CreateRestitutionAttr(0.)
    add_colliders(stage,prim,name,mat)
    stage.GetRootLayer().Save()
    reference=Usd.Stage.CreateInMemory()
    reference.DefinePrim('/instance').GetReferences().AddReference(str(path))
    for child in reference.Traverse():
        for rel in child.GetRelationships():
            if rel.GetName().startswith('material:binding'):
                if not all(reference.GetPrimAtPath(p).IsValid() for p in rel.GetTargets()):
                    raise RuntimeError('Unresolved referenced material: '+str(child.GetPath()))
    count=sum(p.HasAPI(UsdPhysics.CollisionAPI) for p in Usd.PrimRange(prim))
    assert count>0 and prim.HasAPI(UsdPhysics.RigidBodyAPI)
    report['assets'][name]={'path':name+'.usda','sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'mass_kg_initial':spec['mass'],'dynamic':spec['dynamic'],'colliders':count,'referenced_materials_verified':True,
        'position_xyz':list(native.translation),'rotation_xyzw':[q.x,q.y,q.z,q.w]}
(OUT/'native-assets.json').write_text(json.dumps(report,indent=2))
print('NATIVE_ASSETS_EXPORTED',len(report['assets']))
