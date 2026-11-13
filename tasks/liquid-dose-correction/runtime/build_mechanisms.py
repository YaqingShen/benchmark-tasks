"""Build candidate physical cap threads and passive pipette articulations.

Run with Blender's bundled Python (pxr), after export_native_assets.py. The
rigid asset export is preserved. Outputs go to assets/mechanisms. These
mechanisms are engineering approximations pending native contact validation.
No task actions, grasp attachments, or object pose updates are authored.
"""
import hashlib
import json
import math
from pathlib import Path
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade
from pipette_geometry import STEM_LENGTH_M,BULB_CENTER_M,STEM_DRY_MASS_KG,BULB_SPRING_N_M

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT/'assets/native'
OUT=ROOT/'assets/mechanisms'
OUT.mkdir(parents=True,exist_ok=True)
PITCH=.003
HALF_WIDTH=.00045


def physx(prim,schema,attrs):
    prim.AddAppliedSchema(schema)
    for key,kind,value in attrs:
        prim.CreateAttribute(key,kind,custom=False).Set(value)


def collision(prim,material,thread=False):
    UsdPhysics.CollisionAPI.Apply(prim)
    if prim.IsA(UsdGeom.Mesh):
        # PhysX's default 1 mm minimum hull thickness is larger than these
        # thread wedges. Preserve their authored clearance during cooking.
        physx(prim,'PhysxConvexHullCollisionAPI',[
            ('physxConvexHullCollision:minThickness',Sdf.ValueTypeNames.Float,.00005),
        ])
    physx(prim,'PhysxCollisionAPI',[
        ('physxCollision:contactOffset',Sdf.ValueTypeNames.Float,.00008 if thread else .0002),
        ('physxCollision:restOffset',Sdf.ValueTypeNames.Float,0.),
    ])
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material,materialPurpose='physics')


def rigid(prim,mass):
    api=UsdPhysics.RigidBodyAPI.Apply(prim)
    api.CreateRigidBodyEnabledAttr(True);api.CreateKinematicEnabledAttr(False)
    UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(mass)
    physx(prim,'PhysxRigidBodyAPI',[
        ('physxRigidBody:solverPositionIterationCount',Sdf.ValueTypeNames.Int,32),
        ('physxRigidBody:solverVelocityIterationCount',Sdf.ValueTypeNames.Int,8),
        ('physxRigidBody:maxDepenetrationVelocity',Sdf.ValueTypeNames.Float,.5),
        ('physxRigidBody:enableCCD',Sdf.ValueTypeNames.Bool,True),
    ])


def thread(stage,path,z_start,turns,base_radius,crest_radius,material):
    """Triangular helical ridge, split into convex wedges with real end faces."""
    count=round(turns*64)
    faces=[(0,2,1),(3,4,5),(0,1,4,3),(1,2,5,4),(2,0,3,5)]
    for i in range(count):
        pts=[]
        for theta in (2*math.pi*turns*i/count,2*math.pi*turns*(i+1)/count):
            z=z_start+PITCH*theta/(2*math.pi)
            for r,dz in ((base_radius,-HALF_WIDTH),(crest_radius,0),(base_radius,HALF_WIDTH)):
                pts.append(Gf.Vec3f(r*math.cos(theta),r*math.sin(theta),z+dz))
        mesh=UsdGeom.Mesh.Define(stage,path+f'/segment_{i:03d}')
        mesh.CreatePointsAttr(pts)
        mesh.CreateFaceVertexCountsAttr([len(f) for f in faces])
        mesh.CreateFaceVertexIndicesAttr([v for f in faces for v in f])
        mesh.CreateSubdivisionSchemeAttr('none')
        # Visible ridge matches the collider; no invisible attachment constraint.
        UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr('convexHull')
        collision(mesh.GetPrim(),material,thread=True)
        mesh.CreateDisplayColorAttr([Gf.Vec3f(.16,.25,.35)])
    return count


def cylinder(stage,path,r,h,z,material):
    obj=UsdGeom.Cylinder.Define(stage,path)
    obj.CreateRadiusAttr(r);obj.CreateHeightAttr(h);obj.CreateAxisAttr('Z')
    obj.AddTranslateOp().Set(Gf.Vec3d(0,0,z))
    collision(obj.GetPrim(),material)
    return obj


def thread_followers(stage,path,material):
    """Three rounded physical lugs following the male helical ridge.

    This is an interrupted-thread approximation. All constraints come from
    contact; there is no joint, attachment, or prescribed cap motion.
    """
    for i in range(3):
        theta=2*math.pi*i/3
        sphere=UsdGeom.Sphere.Define(stage,path+f'/lug_{i:02d}')
        sphere.CreateRadiusAttr(.0008)
        sphere.AddTranslateOp().Set(Gf.Vec3d(.0226*math.cos(theta),.0226*math.sin(theta),
                                            .0015+PITCH*theta/(2*math.pi)))
        collision(sphere.GetPrim(),material,thread=True)
        sphere.CreateDisplayColorAttr([Gf.Vec3f(.16,.25,.35)])
    return 3


def collar(stage,path,material):
    faces=[(0,3,2,1),(4,5,6,7),(0,1,5,4),(1,2,6,5),(2,3,7,6),(3,0,4,7)]
    for i in range(48):
        a0=2*math.pi*i/48;a1=2*math.pi*(i+1)/48
        pts=[Gf.Vec3f(r*math.cos(a),r*math.sin(a),z)
             for z in (.183,.185) for r,a in ((.0185,a0),(.025,a0),(.025,a1),(.0185,a1))]
        m=UsdGeom.Mesh.Define(stage,path+f'/segment_{i:02d}')
        m.CreatePointsAttr(pts);m.CreateFaceVertexCountsAttr([len(f) for f in faces])
        m.CreateFaceVertexIndicesAttr([v for f in faces for v in f])
        m.CreateSubdivisionSchemeAttr('none')
        UsdPhysics.MeshCollisionAPI.Apply(m.GetPrim()).CreateApproximationAttr('convexHull')
        collision(m.GetPrim(),material)
        m.CreateDisplayColorAttr([Gf.Vec3f(.27,.12,.025)])


def make_pipette(stage,name,material):
    root=stage.GetPrimAtPath('/'+name)
    root.RemoveAPI(UsdPhysics.RigidBodyAPI);root.RemoveAPI(UsdPhysics.MassAPI)
    for a in ('physics:mass','physics:kinematicEnabled','physics:rigidBodyEnabled'):
        root.RemoveProperty(a)
    stage.RemovePrim('/'+name+'/collisions')
    for part in Usd.PrimRange(root):
        if not part.IsA(UsdGeom.Mesh):continue
        mesh=UsdGeom.Mesh(part)
        matrix=UsdGeom.Xformable(part).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        inverse=matrix.GetInverse();points=[]
        for p in mesh.GetPointsAttr().Get():
            v=matrix.Transform(Gf.Vec3d(p))
            v[2]+=(STEM_LENGTH_M-.104)*min(max(v[2]/.104,0.),1.)
            points.append(Gf.Vec3f(inverse.Transform(v)))
        mesh.GetPointsAttr().Set(points)
    stem=UsdGeom.Xform.Define(stage,'/'+name+'/stem')
    rigid(stem.GetPrim(),STEM_DRY_MASS_KG)
    UsdPhysics.ArticulationRootAPI.Apply(stem.GetPrim())
    physx(stem.GetPrim(),'PhysxArticulationAPI',[
        ('physxArticulation:enabledSelfCollisions',Sdf.ValueTypeNames.Bool,True),
        ('physxArticulation:solverPositionIterationCount',Sdf.ValueTypeNames.Int,32),
        ('physxArticulation:solverVelocityIterationCount',Sdf.ValueTypeNames.Int,8),
    ])
    # Preserve the tapered stem, markings and ID. Replace the rigid bulb with
    # two compliant contact surfaces. Their displacement is the sensor input.
    for child in list(root.GetChildren()):
        old=child.GetPath()
        if child.GetName() in ('stem','Looks','physics_material'):continue
        if child.GetName().endswith('_bulb'):
            stage.RemovePrim(old);continue
        new=stem.GetPath().AppendChild(child.GetName())
        Sdf.CopySpec(stage.GetRootLayer(),old,stage.GetRootLayer(),new)
        stage.RemovePrim(old)
    c=cylinder(stage,str(stem.GetPath())+'/shaft_collider',.0025,STEM_LENGTH_M,STEM_LENGTH_M/2,material)
    c.CreatePurposeAttr('guide')
    # Small lower connection stays below the two moving surfaces.
    c=cylinder(stage,str(stem.GetPath())+'/connection',.004,.007,STEM_LENGTH_M+.0035,material)
    c.CreateDisplayColorAttr([Gf.Vec3f(.5,.66,.74)])
    springs=[]
    for side,sign in (('left',-1),('right',1)):
        x=sign*.009
        body=UsdGeom.Xform.Define(stage,'/'+name+'/bulb_'+side)
        body.AddTranslateOp().Set(Gf.Vec3d(x,0,BULB_CENTER_M))
        rigid(body.GetPrim(),.0006)
        shape=UsdGeom.Cube.Define(stage,str(body.GetPath())+'/surface')
        shape.CreateSizeAttr(1.)
        shape.AddScaleOp().Set(Gf.Vec3f(.006,.021,.048))
        shape.CreateDisplayColorAttr([Gf.Vec3f(.5,.66,.74)])
        collision(shape.GetPrim(),material)
        joint=UsdPhysics.PrismaticJoint.Define(stage,'/'+name+'/squeeze_'+side)
        joint.CreateBody0Rel().SetTargets([stem.GetPath()])
        joint.CreateBody1Rel().SetTargets([body.GetPath()])
        joint.CreateAxisAttr('X')
        joint.CreateLocalPos0Attr(Gf.Vec3f(x,0,BULB_CENTER_M))
        joint.CreateLocalPos1Attr(Gf.Vec3f(0,0,0))
        joint.CreateLowerLimitAttr(0. if sign<0 else -.0058)
        joint.CreateUpperLimitAttr(.0058 if sign<0 else 0.)
        joint.CreateCollisionEnabledAttr(False)
        drive=UsdPhysics.DriveAPI.Apply(joint.GetPrim(),'linear')
        drive.CreateTypeAttr('force');drive.CreateTargetPositionAttr(0.)
        drive.CreateTargetVelocityAttr(0.);drive.CreateStiffnessAttr(BULB_SPRING_N_M)
        drive.CreateDampingAttr(.25);drive.CreateMaxForceAttr(3.)
        springs.append({'joint_name':joint.GetPrim().GetName(),'stiffness':BULB_SPRING_N_M,
                        'damping':.25,'effort_limit':3.,'velocity_limit':.1,'target':0.})
    return springs


def make_fixtures(report):
    path=OUT/'station_fixtures.usda'
    stage=Usd.Stage.CreateNew(str(path)) if not path.exists() else Usd.Stage.Open(str(path))
    stage.GetRootLayer().Clear()
    root=UsdGeom.Xform.Define(stage,'/station_fixtures');stage.SetDefaultPrim(root.GetPrim())
    UsdGeom.SetStageMetersPerUnit(stage,1.);UsdGeom.SetStageUpAxis(stage,'Z')
    rigid(root.GetPrim(),.1)
    UsdPhysics.RigidBodyAPI(root.GetPrim()).CreateKinematicEnabledAttr(True)
    mat=UsdShade.Material.Define(stage,'/station_fixtures/material')
    material=UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    material.CreateStaticFrictionAttr(.6);material.CreateDynamicFrictionAttr(.5)
    top=report['table_top_m']
    for ingredient,y in (('A',.16),('B',-.16)):
        if 'bottle_'+ingredient not in report['assets']:continue
        # Support the narrow stem, keeping both sides of the bulb clear for
        # the gripper. Supports under the bulb would block its closing pads.
        for i,x in enumerate((.055,.125)):
            rest=UsdGeom.Cube.Define(stage,f'/station_fixtures/rest_{ingredient}_{i}')
            rest.CreateSizeAttr(1.);rest.AddTranslateOp().Set(Gf.Vec3d(x,y,top+.007))
            rest.AddScaleOp().Set(Gf.Vec3f(.012,.010,.014))
            rest.CreateDisplayColorAttr([Gf.Vec3f(.88,.88,.83)])
            collision(rest.GetPrim(),mat)
        center=report['assets']['bottle_'+ingredient]['position_xyz']
        points=[];indices=[]
        for i in range(96):
            a=2*math.pi*i/96
            for r in (.048,.049):points.append(Gf.Vec3f(center[0]+r*math.cos(a),center[1]+r*math.sin(a),top+.00015))
            indices.extend([2*i,2*i+1,(2*((i+1)%96)+1),2*((i+1)%96)])
        ring=UsdGeom.Mesh.Define(stage,'/station_fixtures/home_'+ingredient)
        ring.CreatePointsAttr(points);ring.CreateFaceVertexCountsAttr([4]*96);ring.CreateFaceVertexIndicesAttr(indices)
        ring.CreateSubdivisionSchemeAttr('none');ring.CreateDisplayColorAttr([Gf.Vec3f(.04,.07,.10)])
    for ingredient,y in (('A',-.04),('B',.04)):
        if 'bottle_'+ingredient not in report['assets']:continue
        cap_x=.20
        for suffix,radius,height,z in [('base',.026,.005,top+.0025),('rest',.018,.055,top+.0325)]:
            support=UsdGeom.Cylinder.Define(stage,f'/station_fixtures/cap_{ingredient}_{suffix}')
            support.CreateRadiusAttr(radius);support.CreateHeightAttr(height);support.CreateAxisAttr('Z')
            support.AddTranslateOp().Set(Gf.Vec3d(cap_x,y,z))
            support.CreateDisplayColorAttr([Gf.Vec3f(.78,.86,.80)])
            collision(support.GetPrim(),mat)
    stage.GetRootLayer().Save()
    return {'path':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'mass_kg_initial':.1,'dynamic':False,'object_type':'RIGID','position_xyz':[0.,0.,0.],
            'rotation_xyzw':[0.,0.,0.,1.],'colliders':4*sum('bottle_'+k in report['assets']for k in ('A','B')),'rigid_bodies':['/station_fixtures'],'joints':[],
            'cap_rest_centers':{k:[.20,y,top+.06]for k,y in [('A',-.04),('B',.04)]if 'bottle_'+k in report['assets']}}


def main():
    report=json.loads((SOURCE/'native-assets.json').read_text())
    config=json.loads((ROOT/'config.json').read_text())
    for ingredient in ('A','B'):
        if ingredient not in config['targets_g']:
            for name in ('bottle_'+ingredient,'bottle_'+ingredient+'_cap','pipette_'+ingredient):report['assets'].pop(name,None)
    if not config.get('trash_bin_in_task',True):report['assets'].pop('trash_bin',None)
    if not config.get('scale_in_task',True):report['assets'].pop('digital_balance',None)
    if not config.get('cylinders_in_task',True):
        for name in ('cylinder_A','cylinder_B'):report['assets'].pop(name,None)
    layout=json.loads((ROOT/'native-layout.json').read_text())
    report.update(status='UNVALIDATED_CONTACT_MECHANISM_CANDIDATE',mechanisms_ready=False,
        limitations=['Physical thread contacts and passive bulb springs need native validation.',
                     'Pipette bulb is a two-slider approximation, not a calibrated deformable polymer.',
                     'Reduced liquid coupling and force-based balance are implemented; full robot validation is pending.'],
        thread_geometry={'pitch_m':PITCH,'male_start_z':.182,'male_turns':3.,
                         'female_start_z':.0015,'female_lug_spacing_turns':1/3,'initial_cap_z':.185,
                         'radial_overlap_m':.0007,'half_thread_width_m':HALF_WIDTH,
                         'male_crest_radius_m':.0225,'female_crest_radius_m':.0218,
                         'cap_bore_radius_m':.0232,
                         'female_model':'three rounded contact lugs',
                         'female_lug_radius_m':.0008,'female_lug_center_radius_m':.0226})
    for name,spec in report['assets'].items():
        spec.update(layout['poses'][name])
        dst=OUT/spec['path'];dst.write_bytes((SOURCE/spec['path']).read_bytes())
        stage=Usd.Stage.Open(str(dst));prim=stage.GetDefaultPrim()
        mat=UsdShade.Material(stage.GetPrimAtPath('/'+name+'/physics_material'))
        for child in Usd.PrimRange(prim):
            if child.HasAPI(UsdPhysics.CollisionAPI):collision(child,mat)
        if name=='digital_balance':
            from balance_display import build_screen
            build_screen(stage,'/'+name,mat)
        if name=='trash_bin':
            # A 219 mm pipette must fit completely below the rim. Preserve
            # the footprint/floor and extend the visible and collision walls.
            height=.24
            for child in prim.GetChildren():
                if not child.GetName().startswith(('trash_side','trash_front_back')):continue
                for part in Usd.PrimRange(child):
                    if not part.IsA(UsdGeom.Mesh):continue
                    mesh=UsdGeom.Mesh(part)
                    matrix=UsdGeom.Xformable(part).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                    inverse=matrix.GetInverse();points=[]
                    for point in mesh.GetPointsAttr().Get():
                        world=matrix.Transform(Gf.Vec3d(point));world[2]*=height/.13
                        points.append(Gf.Vec3f(inverse.Transform(world)))
                    mesh.GetPointsAttr().Set(points)
                    mesh.GetExtentAttr().Set(UsdGeom.Mesh.ComputeExtent(points))
                    part.RemoveProperty('normals')
            for part in ('side_0','side_1','front_back_0','front_back_1'):
                wall=stage.GetPrimAtPath('/'+name+'/collisions/'+part)
                position=wall.GetAttribute('xformOp:translate').Get();position[2]=height/2
                size=wall.GetAttribute('xformOp:scale').Get();size[2]=height
                wall.GetAttribute('xformOp:translate').Set(position)
                wall.GetAttribute('xformOp:scale').Set(size)
            # Publish the actual inner collision faces, not an arbitrary
            # shrunken box that rejects a released tool resting on a wall.
            def box(part):
                path='/'+name+'/collisions/'+part
                p=stage.GetPrimAtPath(path)
                return p.GetAttribute('xformOp:translate').Get(),p.GetAttribute('xformOp:scale').Get()
            xp,xs=box('side_1');yp,ys=box('front_back_1');bp,bs=box('bottom')
            spec['interior_bounds_m']={'half_x':float(xp[0]-xs[0]/2),
                'half_y':float(yp[1]-ys[1]/2),'floor_z':float(bp[2]+bs[2]/2),
                'rim_z':float(xp[2]+xs[2]/2)}
        if name.startswith('bottle_'):
            rigid(prim,spec['mass_kg_initial'])
            tm=UsdShade.Material.Define(stage,'/'+name+'/thread_material')
            pm=UsdPhysics.MaterialAPI.Apply(tm.GetPrim())
            pm.CreateStaticFrictionAttr(.12);pm.CreateDynamicFrictionAttr(.10);pm.CreateRestitutionAttr(0.)
            if name.endswith('_cap'):
                # A polygonal bore's apothem is smaller than its vertex radius.
                # Widen it enough that the thread crests clear the 24-gon wall.
                for meshprim in Usd.PrimRange(prim):
                    if meshprim.IsA(UsdGeom.Mesh):
                        mesh=UsdGeom.Mesh(meshprim);points=mesh.GetPointsAttr().Get()
                        adjusted=[]
                        for p in points:
                            radius=math.hypot(p[0],p[1])
                            scale=.0232/radius if abs(radius-.0217)<1e-6 else (.026/radius if abs(radius-.024)<1e-6 else 1.)
                            adjusted.append(Gf.Vec3f(p[0]*scale,p[1]*scale,p[2]))
                        mesh.GetPointsAttr().Set(adjusted)
                UsdGeom.Cylinder(stage.GetPrimAtPath('/'+name+'/collisions/top')).CreateRadiusAttr(.026)
                count=thread_followers(stage,'/'+name+'/threads',tm)
                # A recessed inner sealing disk meets the 194 mm neck rim
                # when the cap root rests at 185 mm. The taller outside shell
                # supplies a grasp surface without an invisible mouth seal.
                seal=cylinder(stage,'/'+name+'/inner_seal',.0229,.0016,.0098,mat)
                seal.CreateDisplayColorAttr([Gf.Vec3f(.025,.12,.65)])
            else:
                count=thread(stage,'/'+name+'/threads',.182,3.,.021,.0225,tm)
                # Annular closed axial stop; preserve the open bore so the
                # pipette can physically enter the bottle after uncapping.
                collar(stage,'/'+name+'/collar',mat)
            spec['thread_contact_segments']=count
        if name.startswith('pipette_'):
            spec['passive_springs']=make_pipette(stage,name,mat)
            spec['object_type']='ARTICULATION'
            spec['root_body']='stem'
            spec['stroke_per_side_m']=.0058
            spec['mass_kg_initial']=STEM_DRY_MASS_KG+.0012
            spec['stem_length_m']=STEM_LENGTH_M
            spec['bulb_center_m']=BULB_CENTER_M
        else:spec['object_type']='RIGID'
        stage.GetRootLayer().Save()
        spec['sha256']=hashlib.sha256(dst.read_bytes()).hexdigest()
        spec['colliders']=sum(p.HasAPI(UsdPhysics.CollisionAPI) for p in stage.Traverse())
        spec['rigid_bodies']=[str(p.GetPath()) for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
        spec['joints']=[str(p.GetPath()) for p in stage.Traverse() if p.IsA(UsdPhysics.Joint)]
    report['assets']['station_fixtures']=make_fixtures(report)
    report['layout']=layout['status']
    (OUT/'native-assets.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'assets':len(report['assets']),
                      'pipette_joints':report['assets']['pipette_A']['joints']}))


if __name__=='__main__':main()
