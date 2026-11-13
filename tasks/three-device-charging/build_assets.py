"""Isaac/USD charging-scene assets. Dimensions sourced; mechanical parameters provisional.

Run in the 5090 Isaac venv as a separate process before starting SimulationApp.
The generated assets contain no grasp attachments or charging-success triggers.
"""
from pathlib import Path
import json, math
import numpy as np
from PIL import Image
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

HERE=Path(__file__).resolve().parent
OUT=HERE/'assets'; OUT.mkdir(exist_ok=True)
TABLE=.7475
# Device envelopes are from Apple. Charger internals are a Belkin-inspired
# approximation, widened at the user's request; not a measured WIZ023 CAD model.
PHONE_PUCK=(.023,.050,.096)
WATCH_PUCK=(.028,-.050,.092)
PUCK_NORMAL=(-.8660254038,0,.5)
RECESS_CENTER=(-.012,0)
RECESS_RADIUS=.039
RECESS_FLOOR=.010
BASE_TOP=.012

def physx(prim,schema,items):
    prim.AddAppliedSchema(schema)
    for n,t,v in items:prim.CreateAttribute(n,t,custom=False).Set(v)

def body(stage,path,mass,pos=(0,0,0)):
    x=UsdGeom.Xform.Define(stage,path);x.AddTranslateOp().Set(Gf.Vec3d(*pos))
    UsdPhysics.RigidBodyAPI.Apply(x.GetPrim()).CreateRigidBodyEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(x.GetPrim()).CreateMassAttr(mass)
    physx(x.GetPrim(),'PhysxRigidBodyAPI',[
        ('physxRigidBody:solverPositionIterationCount',Sdf.ValueTypeNames.Int,32),
        ('physxRigidBody:solverVelocityIterationCount',Sdf.ValueTypeNames.Int,8),
        ('physxRigidBody:maxDepenetrationVelocity',Sdf.ValueTypeNames.Float,.5),
        ('physxRigidBody:enableCCD',Sdf.ValueTypeNames.Bool,True)])
    return x

def material(stage,path,rgb,metallic=0,rough=.5,emission=0):
    m=UsdShade.Material.Define(stage,path);s=UsdShade.Shader.Define(stage,path+'/surface')
    s.CreateIdAttr('UsdPreviewSurface')
    s.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
    s.CreateInput('roughness',Sdf.ValueTypeNames.Float).Set(rough)
    s.CreateInput('metallic',Sdf.ValueTypeNames.Float).Set(metallic)
    if emission:s.CreateInput('emissiveColor',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*(np.array(rgb)*emission)))
    m.CreateSurfaceOutput().ConnectToSource(s.ConnectableAPI(),'surface')
    return m

def setup(name):
    path=OUT/(name+'.usda'); stage=Usd.Stage.CreateNew(str(path)) if not path.exists() else Usd.Stage.Open(str(path))
    stage.GetRootLayer().Clear();root=UsdGeom.Xform.Define(stage,'/'+name)
    stage.SetDefaultPrim(root.GetPrim());UsdGeom.SetStageMetersPerUnit(stage,1.);UsdGeom.SetStageUpAxis(stage,'Z')
    phys=UsdShade.Material.Define(stage,'/'+name+'/Looks/physics')
    api=UsdPhysics.MaterialAPI.Apply(phys.GetPrim());api.CreateStaticFrictionAttr(.65);api.CreateDynamicFrictionAttr(.5);api.CreateRestitutionAttr(0.)
    colors={'white':(.88,.89,.90),'black':(.009,.012,.016),'silver':(.55,.58,.60),'blue':(.40,.51,.63),
            'band':(.018,.035,.05),'sensor':(.025,.03,.025),'seam':(.35,.37,.39)}
    mats={n:material(stage,'/'+name+'/Looks/'+n,c,metallic=.7 if n=='silver' else 0,rough=.28 if n=='silver' else .55) for n,c in colors.items()}
    return stage,phys,mats

def finish_shape(shape,mat,phys=None):
    UsdShade.MaterialBindingAPI.Apply(shape.GetPrim()).Bind(mat)
    if phys:
        UsdPhysics.CollisionAPI.Apply(shape.GetPrim())
        if shape.GetPrim().IsA(UsdGeom.Mesh):UsdPhysics.MeshCollisionAPI.Apply(shape.GetPrim()).CreateApproximationAttr('convexHull')
        physx(shape.GetPrim(),'PhysxCollisionAPI',[
            ('physxCollision:contactOffset',Sdf.ValueTypeNames.Float,.0001),
            ('physxCollision:restOffset',Sdf.ValueTypeNames.Float,0.)])
        UsdShade.MaterialBindingAPI.Apply(shape.GetPrim()).Bind(phys,materialPurpose='physics')
    return shape

def rounded(stage,path,size,radius,mat,phys=None,pos=(0,0,0),bevel=None):
    # Convex, rounded rectangle with a small bevel at front and back.
    w,h,t=size;pts=[];loops=40
    bevel=min(.001,t*.2) if bevel is None else min(bevel,t*.45,radius*.8)
    for z,inset in [(-t/2,bevel),(-t/2+bevel,0),(t/2-bevel,0),(t/2,bevel)]:
        for cx,cy,start in [(1,1,0),(-1,1,90),(-1,-1,180),(1,-1,270)]:
            for a in np.linspace(start,start+90,10,endpoint=False):
                a=math.radians(a);rr=max(.0001,radius-inset)
                pts.append(Gf.Vec3f(cx*(w/2-radius)+rr*math.cos(a),cy*(h/2-radius)+rr*math.sin(a),z))
    faces=[list(range(loops-1,-1,-1)),list(range(3*loops,4*loops))]
    for layer in range(3):
        for j in range(loops):faces.append([layer*loops+j,layer*loops+(j+1)%loops,(layer+1)*loops+(j+1)%loops,(layer+1)*loops+j])
    m=UsdGeom.Mesh.Define(stage,path);m.CreatePointsAttr(pts)
    m.CreateFaceVertexCountsAttr([len(f) for f in faces]);m.CreateFaceVertexIndicesAttr([v for f in faces for v in f])
    m.CreateSubdivisionSchemeAttr('none');m.AddTranslateOp().Set(Gf.Vec3d(*pos))
    return finish_shape(m,mat,phys)

def cylinder(stage,path,r,h,mat,phys=None,pos=(0,0,0),normal=(0,0,1)):
    count=64
    pts=[Gf.Vec3f(r*math.cos(a),r*math.sin(a),z) for z in [-h/2,h/2] for a in np.linspace(0,2*math.pi,count,endpoint=False)]
    faces=[list(range(count-1,-1,-1)),list(range(count,2*count))]
    faces += [[j,(j+1)%count,(j+1)%count+count,j+count] for j in range(count)]
    c=UsdGeom.Mesh.Define(stage,path);c.CreatePointsAttr(pts)
    c.CreateFaceVertexCountsAttr([len(f) for f in faces]);c.CreateFaceVertexIndicesAttr([v for f in faces for v in f]);c.CreateSubdivisionSchemeAttr('none')
    c.AddTranslateOp().Set(Gf.Vec3d(*pos))
    c.AddOrientOp().Set(Gf.Quatf(Gf.Rotation(Gf.Vec3d(0,0,1),Gf.Vec3d(*normal)).GetQuat()))
    return finish_shape(c,mat,phys)

def rod(stage,path,a,b,r,mat,phys=None):
    a,b=np.array(a),np.array(b);return cylinder(stage,path,r,float(np.linalg.norm(b-a)),mat,phys,tuple((a+b)/2),tuple(b-a))

def screen(stage,parent,w,h,z,name):
    # Standby black screen. The 30% state is metadata, never a starting overlay.
    size=(512,1024) if name=='iphone17' else (512,600)
    im=Image.new('RGB',size,(0,0,0))
    im.save(OUT/(name+'_screen.png'))
    path='/'+name+'/Looks/display';mat=UsdShade.Material.Define(stage,path)
    sh=UsdShade.Shader.Define(stage,path+'/surface');sh.CreateIdAttr('UsdPreviewSurface')
    uv=UsdShade.Shader.Define(stage,path+'/uv');uv.CreateIdAttr('UsdPrimvarReader_float2');uv.CreateInput('varname',Sdf.ValueTypeNames.Token).Set('st')
    tex=UsdShade.Shader.Define(stage,path+'/texture');tex.CreateIdAttr('UsdUVTexture')
    tex.CreateInput('file',Sdf.ValueTypeNames.Asset).Set('./'+name+'_screen.png')
    tex.CreateInput('sourceColorSpace',Sdf.ValueTypeNames.Token).Set('sRGB')
    tex.CreateInput('st',Sdf.ValueTypeNames.Float2).ConnectToSource(uv.ConnectableAPI(),'result')
    sh.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(.015))
    sh.CreateInput('emissiveColor',Sdf.ValueTypeNames.Color3f).ConnectToSource(tex.ConnectableAPI(),'rgb')
    sh.CreateInput('roughness',Sdf.ValueTypeNames.Float).Set(.45)
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(),'surface')
    mesh=UsdGeom.Mesh.Define(stage,parent+'/display')
    mesh.CreatePointsAttr([(-w/2,-h/2,z),(w/2,-h/2,z),(w/2,h/2,z),(-w/2,h/2,z)])
    mesh.CreateFaceVertexCountsAttr([4]);mesh.CreateFaceVertexIndicesAttr([0,1,2,3]);mesh.CreateSubdivisionSchemeAttr('none')
    UsdGeom.PrimvarsAPI(mesh).CreatePrimvar('st',Sdf.ValueTypeNames.TexCoord2fArray,UsdGeom.Tokens.vertex).Set([(0,0),(1,0),(1,1),(0,1)])
    finish_shape(mesh,mat)

def indicator(stage,parent,z,radius,name):
    root=parent+'/charging_indicator';x=UsdGeom.Xform.Define(stage,root)
    x.AddScaleOp().Set(Gf.Vec3f(1.))
    UsdGeom.Imageable(x).CreateVisibilityAttr('invisible')
    color=(1.,.25,.005) if name=='airpods_pro' else (.015,.85,.10)
    mat=material(stage,'/'+name+'/Looks/charge_light',color,rough=.45,emission=1.5)
    if name=='airpods_pro':
        cylinder(stage,root+'/led',radius,.0001,mat,None,(0,-.003,z));return
    pts=[];indices=[]
    for i in range(64):
        a=2*np.pi*i/64
        pts += [(radius*np.cos(a),radius*np.sin(a),z),(.84*radius*np.cos(a),.84*radius*np.sin(a),z)]
        j=2*i;k=2*((i+1)%64);indices.extend([j,k,k+1,j+1])
    mesh=UsdGeom.Mesh.Define(stage,root+'/ring');mesh.CreatePointsAttr(pts)
    mesh.CreateFaceVertexCountsAttr([4]*64);mesh.CreateFaceVertexIndicesAttr(indices);mesh.CreateSubdivisionSchemeAttr('none');finish_shape(mesh,mat)
    bolt=[(-.12,.60),(-.48,-.08),(-.07,-.08),(-.25,-.58),(.48,.12),(.05,.12)]
    mesh=UsdGeom.Mesh.Define(stage,root+'/bolt');mesh.CreatePointsAttr([(x*radius,y*radius,z+.00002) for x,y in bolt])
    mesh.CreateFaceVertexCountsAttr([6]);mesh.CreateFaceVertexIndicesAttr(list(range(6)));mesh.CreateDoubleSidedAttr(True);mesh.CreateSubdivisionSchemeAttr('none');finish_shape(mesh,mat)

def make_phone():
    name='iphone17';st,ph,m=setup(name);body(st,'/'+name,.177)
    rounded(st,'/'+name+'/frame',(.0715,.1496,.00795),.010,m['silver'],ph)
    rounded(st,'/'+name+'/back',(.0697,.1478,.0006),.009,m['blue'],None,(0,0,-.00405))
    rounded(st,'/'+name+'/glass',(.0697,.1478,.0005),.009,m['black'],None,(0,0,.00402))
    for y in [.043,.063]:
        cylinder(st,'/'+name+'/lens_'+str(round(y*1000)),.007,.003,m['black'],ph,(-.021,y,-.0050))
    screen(st,'/'+name,.062,.132,.00432,name)
    indicator(st,'/'+name,.0046,.024,name)
    rounded(st,'/'+name+'/island',(.020,.005,.0002),.002,m['black'],None,(0,.061,.00445))
    st.GetRootLayer().Save()

def make_case():
    name='airpods_pro';st,ph,m=setup(name);body(st,'/'+name,.0614) # case plus two earbuds
    rounded(st,'/'+name+'/shell',(.0606,.0452,.0217),.010,m['white'],ph,bevel=.004)
    rounded(st,'/'+name+'/lid_seam',(.057,.00045,.00015),.0001,m['seam'],None,(0,.010,.01091))
    cylinder(st,'/'+name+'/status_light',.00085,.00015,m['seam'],None,(0,-.003,.01095))
    indicator(st,'/'+name,.01115,.0012,name)
    rounded(st,'/'+name+'/port',(.008,.002,.0002),.0009,m['black'],None,(0,-.0227,0)).AddRotateXOp().Set(90)
    st.GetRootLayer().Save()

def recessed_deck(stage,parent,mat,phys):
    # A ring must not be one convex collider: that would invisibly fill its hole.
    # Each sector is a separate convex prism, above a solid lower base/floor.
    count=96;inner=np.array(RECESS_CENTER)
    for i in range(count):
        angles=np.array([i,i+1])*2*np.pi/count
        rays=np.stack([np.cos(angles),np.sin(angles)],axis=1)
        pts=[]
        for z,r in [(RECESS_FLOOR,RECESS_RADIUS),(BASE_TOP,RECESS_RADIUS+.001)]:
            outline=[inner+r*rays[0],.059*rays[0],.059*rays[1],inner+r*rays[1]]
            pts.extend([Gf.Vec3f(float(p[0]),float(p[1]),z) for p in outline])
        faces=[[3,2,1,0],[4,5,6,7],[0,1,5,4],[1,2,6,5],[2,3,7,6],[3,0,4,7]]
        mesh=UsdGeom.Mesh.Define(stage,parent+f'/deck_sector_{i:03d}')
        mesh.CreatePointsAttr(pts);mesh.CreateFaceVertexCountsAttr([4]*6)
        mesh.CreateFaceVertexIndicesAttr([v for f in faces for v in f]);mesh.CreateSubdivisionSchemeAttr('none')
        finish_shape(mesh,mat,phys)

def make_charger():
    name='belkin';st,ph,m=setup(name);body(st,'/'+name,.55) # mass is provisional, pending weighing
    cylinder(st,'/'+name+'/base',.059,RECESS_FLOOR,m['white'],ph,(0,0,RECESS_FLOOR/2))
    recessed_deck(st,'/'+name,m['white'],ph)
    inset=material(st,'/'+name+'/Looks/recess',(.78,.79,.80),rough=.8)
    cylinder(st,'/'+name+'/recess_surface',RECESS_RADIUS,.00002,inset,None,(*RECESS_CENTER,RECESS_FLOOR+.00001))
    cylinder(st,'/'+name+'/foot',.055,.001,m['black'],ph,(0,0,.0005))
    rod(st,'/'+name+'/stem',(.034,0,BASE_TOP),(.037,0,.090),.005,m['silver'],ph)
    rod(st,'/'+name+'/crossbar',(.037,-.050,.090),(.037,.050,.090),.004,m['silver'],ph)
    # Shape and contact-surface locations approximate WIZ023 product photographs.
    normal=np.array(PUCK_NORMAL)
    for label,center,r,h in [('phone',PHONE_PUCK,.028,.007),('watch',WATCH_PUCK,.014,.006)]:
        center=np.array(center)
        rod(st,'/'+name+'/'+label+'_mount',(.037,center[1],.090),center-normal*h/2,.0035,m['silver'],ph)
        cylinder(st,'/'+name+'/'+label+'_puck',r,h,m['white'],ph,tuple(center),tuple(normal))
        cylinder(st,'/'+name+'/'+label+'_face',r-.0015,.0002,m['white'],None,tuple(center+normal*(h/2+.0001)),tuple(normal))
    # Short fixed visual cable stub only; cable mechanics are outside this preview.
    rod(st,'/'+name+'/cable_stub',(.054,0,.006),(.105,0,.006),.0017,m['white'])
    st.GetRootLayer().Save()

def joint(st,name,b0,b1,p0,p1,axis):
    j=UsdPhysics.RevoluteJoint.Define(st,'/apple_watch/'+name)
    j.CreateBody0Rel().SetTargets([b0]);j.CreateBody1Rel().SetTargets([b1]);j.CreateAxisAttr(axis)
    j.CreateLocalPos0Attr(Gf.Vec3f(*p0));j.CreateLocalPos1Attr(Gf.Vec3f(*p1))
    j.CreateLowerLimitAttr(-65. if axis=='X' else -25.);j.CreateUpperLimitAttr(65. if axis=='X' else 25.)
    j.CreateCollisionEnabledAttr(False)
    drive=UsdPhysics.DriveAPI.Apply(j.GetPrim(),'angular')
    drive.CreateTypeAttr('force');drive.CreateStiffnessAttr(.0025);drive.CreateDampingAttr(.00015)
    drive.CreateMaxForceAttr(.025);drive.CreateTargetPositionAttr(0.)

def make_watch():
    name='apple_watch';st,ph,m=setup(name);root='/'+name;case=body(st,root+'/case',.0303)
    UsdPhysics.ArticulationRootAPI.Apply(case.GetPrim())
    physx(case.GetPrim(),'PhysxArticulationAPI',[
        ('physxArticulation:enabledSelfCollisions',Sdf.ValueTypeNames.Bool,True),
        ('physxArticulation:solverPositionIterationCount',Sdf.ValueTypeNames.Int,32),
        ('physxArticulation:solverVelocityIterationCount',Sdf.ValueTypeNames.Int,8)])
    rounded(st,root+'/case/frame',(.036,.042,.0097),.008,m['silver'],ph)
    rounded(st,root+'/case/glass',(.034,.040,.0006),.008,m['black'],None,(0,0,.00475))
    cylinder(st,root+'/case/sensor',.012,.0014,m['sensor'],ph,(0,0,-.0050))
    cylinder(st,root+'/case/crown',.003,.003,m['silver'],ph,(.019, .009,0),(1,0,0))
    screen(st,root+'/case',.028,.033,.00511,name)
    indicator(st,root+'/case',.0053,.010,name)
    # Contact-bearing elastic articulated bands. Two bend/twist DOFs per segment.
    # This is a reduced flexible-band model, not a calibrated continuum material.
    for side,sgn in [('long',1),('short',-1)]:
        prev=root+'/case'
        for i in range(8):
            edge=sgn*(.021+i*.009);cy=edge+sgn*.0045
            pivot=root+f'/{side}_pivot_{i}';link=root+f'/{side}_band_{i}'
            body(st,pivot,.00002,(0,edge,0));body(st,link,.00073,(0,cy,0))
            UsdPhysics.MassAPI(st.GetPrimAtPath(pivot)).CreateDiagonalInertiaAttr(Gf.Vec3f(1e-9))
            UsdPhysics.MassAPI(st.GetPrimAtPath(pivot)).CreateCenterOfMassAttr(Gf.Vec3f(0))
            rounded(st,link+'/rubber',(.020,.0088,.0022),.001,m['band'],ph)
            joint(st,f'{side}_bend_{i}',prev,pivot,(0,sgn*(.021 if i==0 else .0045),0),(0,0,0),'X')
            joint(st,f'{side}_twist_{i}',pivot,link,(0,0,0),(0,-sgn*.0045,0),'Y')
            if side=='long' and i>1:
                cylinder(st,link+'/hole_mark',.001,.0001,m['black'],None,(0,0,.00115))
            prev=link
    st.GetRootLayer().Save()

def main():
    make_phone();make_case();make_charger();make_watch()
    rng=np.random.default_rng(17)
    specs={
      'belkin':{'position_xyz':[.51,0,TABLE+.001],'rotation_xyzw':[0,0,0,1],'object_type':'RIGID'},
      'iphone17':{'position_xyz':[.32,.16,TABLE+.010],'object_type':'RIGID'},
      'airpods_pro':{'position_xyz':[.29,-.025,TABLE+.014],'object_type':'RIGID'},
      'apple_watch':{'position_xyz':[.37,-.18,TABLE+.009],'object_type':'ARTICULATION'},
    }
    for n,s in specs.items():
        s['path']=n+'.usda'
        if n!='belkin':
            s['position_xyz'][:2]=(np.array(s['position_xyz'][:2])+rng.uniform(-.018,.018,2)).tolist()
            yaw=float(rng.uniform(-.65,.65));s['rotation_xyzw']=[0,0,math.sin(yaw/2),math.cos(yaw/2)]
            s['initial_battery_percent']=30
    (OUT/'manifest.json').write_text(json.dumps({'seed':17,'table_top_m':TABLE,'assets':specs,
       'status':'scene-preview-only','sources':{
       'iphone':'https://www.apple.com/iphone-17/specs/',
       'airpods':'https://support.apple.com/en-us/111834',
       'watch':'https://support.apple.com/en-ie/125093',
       'charger':'https://www.belkin.com/support-article/?articleNum=336338'},
       'geometry':{
       'device_envelopes_mm':{'iphone17':[71.5,149.6,7.95],'airpods_pro':[60.6,45.2,21.7],'apple_watch_case':[36,42,9.7]},
       'charger_base_diameter_mm':118,
       'phone_watch_lateral_center_spacing_mm':100,
       'phone_puck_center_m':PHONE_PUCK,'watch_puck_center_m':WATCH_PUCK,
       'puck_normal':PUCK_NORMAL,
       'airpods_recess':{'center_xy_m':RECESS_CENTER,'floor_z_m':RECESS_FLOOR,'floor_diameter_mm':78,'depth_mm':2,'collision_sectors':96},
       'charger_internal_dimensions_status':'user-adjusted approximate geometry, not manufacturer measurements'},
       'approximations':['Belkin-inspired widened support: internal dimensions, mass and friction need measurement',
       '32-DOF articulated silicone band; stiffness is provisional',
       'Black standby screens; 30 percent battery is internal state only',
       'No charging or magnetic interaction implemented in this layout preview']},indent=2))
    print('Generated native USD assets and seed-17 layout.')

if __name__=='__main__':main()
