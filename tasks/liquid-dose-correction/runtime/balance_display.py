"""Readable physical seven-segment balance screen, attached to the balance."""
import math

SEGMENTS={'0':'abcdef','1':'bc','2':'abdeg','3':'abcdg','4':'bcfg',
          '5':'acdfg','6':'acdefg','7':'abc','8':'abcdefg','9':'abcdfg',
          '-':'g',' ':''}


def build_screen(stage,root_path,physical_material):
    from pxr import Gf,Sdf,UsdGeom,UsdPhysics,UsdShade
    root=stage.GetPrimAtPath(root_path)
    for child in list(root.GetChildren()):
        if child.GetName() in ('balance_screen','balance_reading') or child.GetName().startswith(('balance_button_','button_text_')):
            stage.RemovePrim(child.GetPath())
    panel=UsdGeom.Xform.Define(stage,root_path+'/readout')
    panel.AddTranslateOp().Set(Gf.Vec3d(0,-.105,.030))
    panel.AddRotateXOp().Set(45.)
    housing=UsdGeom.Cube.Define(stage,str(panel.GetPath())+'/housing')
    housing.CreateSizeAttr(1.);housing.AddScaleOp().Set(Gf.Vec3f(.158,.050,.006))
    housing.CreateDisplayColorAttr([Gf.Vec3f(.012,.025,.03)])
    UsdPhysics.CollisionAPI.Apply(housing.GetPrim())
    UsdShade.MaterialBindingAPI.Apply(housing.GetPrim()).Bind(physical_material,materialPurpose='physics')
    lit=UsdShade.Material.Define(stage,root_path+'/Looks/Screen_light')
    shader=UsdShade.Shader.Define(stage,str(lit.GetPath())+'/Surface')
    shader.CreateIdAttr('UsdPreviewSurface')
    shader.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(.35,1.,.78))
    shader.CreateInput('emissiveColor',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(.25,.8,.6))
    shader.CreateInput('roughness',Sdf.ValueTypeNames.Float).Set(.9)
    lit.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(),'surface')
    def rectangle(path,x,y,w,h):
        mesh=UsdGeom.Mesh.Define(stage,path)
        mesh.CreatePointsAttr([Gf.Vec3f(x-w/2,y-h/2,.0032),Gf.Vec3f(x+w/2,y-h/2,.0032),
                               Gf.Vec3f(x+w/2,y+h/2,.0032),Gf.Vec3f(x-w/2,y+h/2,.0032)])
        mesh.CreateFaceVertexCountsAttr([4]);mesh.CreateFaceVertexIndicesAttr([0,1,2,3])
        mesh.CreateSubdivisionSchemeAttr('none');mesh.CreateDoubleSidedAttr(True)
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(lit)
    # Six digit slots plus a fixed decimal point. Leading slots remain blank.
    layout={'a':(0,.014,.012,.0025),'g':(0,0,.012,.0025),'d':(0,-.014,.012,.0025),
            'f':(-.007,.007,.0025,.010),'b':(.007,.007,.0025,.010),
            'e':(-.007,-.007,.0025,.010),'c':(.007,-.007,.0025,.010)}
    digits=UsdGeom.Xform.Define(stage,str(panel.GetPath())+'/digits')
    # Keep native mesh topology fixed. Replacing one variable-length digit
    # mesh left stale lit strokes in RTX captures when the reading changed.
    # Move unlit strokes behind the opaque housing. Visibility-only animation
    # left dim RTX ghosts of inactive segments in long dynamic recordings.
    for i in range(6):
        center=-.061+i*.020+(.004 if i>=3 else 0)
        for segment,(x,y,w,h)in layout.items():
            rectangle(str(digits.GetPath())+f'/slot_{i}_{segment}',center+x,y,w,h)
    rectangle(str(panel.GetPath())+'/decimal',-.009,-.014,.0028,.0028)
    # A small physical 'g' unit, built from the same strokes.
    for segment in 'abcdfg':
        dx,dy,w,h=layout[segment]
        rectangle(str(panel.GetPath())+'/unit_'+segment,.065+dx*.55,-.007+dy*.55,w*.55,h*.55)
    Screen(stage,root_path).set_value(0.)


class Screen:
    def __init__(self,stage,root_path):
        from pxr import UsdGeom
        self.segments={};self.offsets={}
        for i in range(6):
            for segment in 'abcdefg':
                prim=stage.GetPrimAtPath(root_path+f'/readout/digits/slot_{i}_{segment}')
                if not prim:raise ValueError('Balance segment mesh missing; rebuild the native assets')
                self.segments[i,segment]=UsdGeom.Imageable(prim)
                attr=prim.GetAttribute('xformOp:translate:segment')
                self.offsets[i,segment]=(UsdGeom.XformOp(attr) if attr else
                    UsdGeom.Xformable(prim).AddTranslateOp(opSuffix='segment'))
                self.segments[i,segment].MakeVisible()
        self.previous=None

    def set_value(self,value):
        from pxr import Gf
        if value is not None and math.isfinite(value) and round(value,3)==0:value=0.
        text='------' if value is None or not math.isfinite(value) or not -99.999<=value<=999.999 else f'{value:7.3f}'.replace('.','')
        if text==self.previous:return
        for i,char in enumerate(text):
            for segment in 'abcdefg':
                self.offsets[i,segment].Set(Gf.Vec3d(0.,0.,0. if segment in SEGMENTS[char] else -.012))
        self.previous=text
