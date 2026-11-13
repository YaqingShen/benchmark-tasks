"""Procedural reference-based visual assets. Run with Blender 4.1+.

These are visual meshes and separate movable parts, not validated collision
meshes. Dimensions other than stated capacities are design estimates.
"""
import bpy
import math
import json
import sys
from pathlib import Path
from mathutils import Vector

ROOT=Path(__file__).resolve().parent
OUT=ROOT.parent.parent/"outputs"/"formulation-prototype"
OUT.mkdir(parents=True,exist_ok=True)
CONFIG=json.loads((ROOT/"config.json").read_text())
bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False)


def material(name,color,metal=0.,rough=.3,trans=0.):
    m=bpy.data.materials.new(name)
    m.diffuse_color=(*color,1)
    m.use_nodes=True
    p=m.node_tree.nodes.get("Principled BSDF")
    p.inputs["Base Color"].default_value=(*color,1)
    p.inputs["Metallic"].default_value=0.
    p.inputs["Roughness"].default_value=.85
    p.inputs["Specular IOR Level"].default_value=0.
    p.inputs["Transmission Weight"].default_value=0.
    if trans and name != "Amber glass":
        alpha=.22 if name=="Borosilicate glass" else .68
        m.diffuse_color=(*color,alpha)
        p.inputs["Alpha"].default_value=alpha
        m.blend_method="BLEND"
        m.use_screen_refraction=False
        m.use_nodes=True
    p.inputs["IOR"].default_value=1.46 if trans else 1.5
    return m


glass=material("Borosilicate glass",(.40,.69,.78),rough=.07,trans=1.)
edge=material("Glass outline",(.20,.39,.46),rough=.85)
amber=material("Amber glass",(.42,.135,.018),rough=.13,trans=.83)
blue=material("Blue polypropylene",(.018,.12,.65),rough=.24)
white=material("White label",(.95,.95,.91),rough=.65)
ink=material("Blue graduation ink",(.025,.13,.26),rough=.65)
grey=material("Balance body",(.75,.78,.79),metal=.1,rough=.3)
dark=material("Dark feet and display",(.015,.032,.041),rough=.3)
steel=material("Stainless pan",(.52,.58,.63),metal=.8,rough=.24)
pipmat=material("Translucent LDPE",(.50,.66,.74),rough=.3,trans=.48)
water=material("Colorless nonreactive substitute",(.8,.95,1.),rough=.08,trans=.8)
mint=material("Balance digits",(.2,.85,.57),rough=.5)
base=material("Work surface",(.72,.79,.80),rough=.68)


def assign(obj,mat,parent=None):
    if mat: obj.data.materials.append(mat)
    if parent: obj.parent=parent
    if obj.type=="MESH":
        for p in obj.data.polygons: p.use_smooth=False
    return obj


def empty(name,loc=(0,0,0)):
    o=bpy.data.objects.new(name,None)
    bpy.context.collection.objects.link(o)
    o.location=loc
    return o


def lathe(name,profile,mat,parent=None,n=96,spout=False):
    n=min(n,32)
    vertices=[]
    for r,z in profile:
        for j in range(n):
            a=2*math.pi*j/n
            threshold=.191 if name.startswith("cylinder_") else .112
            amplitude=.004 if name.startswith("cylinder_") else .009
            extra=amplitude*max(0.,math.cos(a))**24 if spout and z>threshold else 0.
            vertices.append(((r+extra)*math.cos(a),(r+extra)*math.sin(a),z))
    faces=[]
    for i in range(len(profile)-1):
        for j in range(n):
            a=i*n+j;b=i*n+(j+1)%n
            faces.append((a,b,b+n,a+n))
    mesh=bpy.data.meshes.new(name)
    mesh.from_pydata(vertices,[],faces)
    mesh.update()
    obj=bpy.data.objects.new(name,mesh)
    bpy.context.collection.objects.link(obj)
    return assign(obj,mat,parent)


def cube(name,loc,size,mat,parent=None,bevel=.003):
    bpy.ops.mesh.primitive_cube_add(size=1,location=loc)
    obj=bpy.context.object;obj.name=name;obj.dimensions=size
    bpy.ops.object.transform_apply(location=False,rotation=False,scale=True)
    if bevel:
        m=obj.modifiers.new("Rounded edges","BEVEL");m.width=bevel;m.segments=3
        obj.modifiers.new("Weighted normals","WEIGHTED_NORMAL")
    return assign(obj,mat,parent)


def cyl(name,loc,r,depth,mat,parent=None,vertices=64):
    vertices=min(vertices,32)
    bpy.ops.mesh.primitive_cylinder_add(vertices=vertices,radius=r,depth=depth,location=loc)
    obj=bpy.context.object;obj.name=name
    return assign(obj,mat,parent)


def line(name,points,r,mat,parent=None):
    curve=bpy.data.curves.new(name,"CURVE");curve.dimensions="3D"
    curve.resolution_u=1;curve.bevel_depth=r;curve.bevel_resolution=2
    spline=curve.splines.new("POLY");spline.points.add(len(points)-1)
    for p,co in zip(spline.points,points):p.co=(*co,1)
    obj=bpy.data.objects.new(name,curve);bpy.context.collection.objects.link(obj)
    return assign(obj,mat,parent)


def arc(name,r,z,start,end,mat,parent,thickness=.00022):
    return line(name,[(r*math.cos(start+(end-start)*i/40),r*math.sin(start+(end-start)*i/40),z) for i in range(41)],thickness,mat,parent)


def text(name,body,loc,size,mat,parent=None,rotation=(math.pi/2,0,0)):
    c=bpy.data.curves.new(name,"FONT");c.body=body;c.size=size
    c.align_x="CENTER";c.align_y="CENTER";c.extrude=.000015
    o=bpy.data.objects.new(name,c);bpy.context.collection.objects.link(o)
    o.location=loc;o.rotation_euler=rotation
    return assign(o,mat,parent)


def bottle(name,loc):
    root=empty(name,loc)
    profile=[(0,0),(.043,0),(.043,.12),(.021,.174),(.021,.194),(.0185,.194),(.0185,.174),(.0405,.119),(.0405,.004),(0,.004)]
    lathe(name+"_glass",profile,amber,root)
    # White band and large A/B identifier, with no brand or chemical claim.
    lathe(name+"_label",[(.0433,.045),(.0433,.101)],white,root)
    text(name+"_ID",name[-1],(0,-.0437,.081),.026,ink,root)
    text(name+"_type",CONFIG["ingredient_labels"][name[-1]],(0,-.044,.061),.006,ink,root)
    text(name+"_capacity","500 mL",(0,-.044,.0515),.005,ink,root)
    # A separate plain cap retains the moving-part distinction.
    cap=empty(name+"_cap",(0,0,.185));cap.parent=root
    lathe(name+"_cap_shell",[(.0217,0),(.024,0),(.024,.027),(0,.027),(0,.024),(.021,.024),(.0217,0)],blue,cap,24)
    # Starting stock: 200 mL in a cylindrical lower body, visual approximation.
    liquid=cyl(name+"_liquid",(0,0,.004+(.2e-3/(math.pi*.0404**2))/2),.0403,.2e-3/(math.pi*.0404**2),water,root)
    return root,cap,liquid


bottles={}
bottles["bottle_A"]=bottle("bottle_A",(-.255,.04,0))
if "B" in CONFIG["targets_g"]:bottles["bottle_B"]=bottle("bottle_B",(-.145,.13,0))

cylinders={}
for ingredient,loc in ([("A",(-.035,-.005,0)),("B",(.045,.13,0))] if CONFIG.get("cylinders") else []):
    name="cylinder_"+ingredient
    cylinder=empty(name,loc);cylinders[ingredient]=cylinder
    cyl(name+"_hex_foot",(0,0,.004),.030,.008,glass,cylinder,6)
    lathe(name+"_glass",[(0,.008),(.0115,.008),(.0125,.014),(.0125,.194),(.012,.196),(.011,.194),(.011,.013),(0,.013)],glass,cylinder,spout=True)
    arc(name+"_rim_outline",.0125,.195,0,2*math.pi,edge,cylinder,.00028)
    for a in (-2.7,.44):
        line(name+"_edge",[(.0125*math.cos(a),.0125*math.sin(a),z) for z in (.014,.194)],.00022,edge,cylinder)
    for ml in range(1,51):
        z=.013+ml*1e-6/(math.pi*.011**2)
        start,end=(-2.7,-.4) if ml%10==0 else ((-2.5,-1.1) if ml%5==0 else (-2.4,-1.7))
        arc(name+f"_tick_{ml}",.0127,z,start,end,ink,cylinder,.00016 if ml%10 else .00025)
        if ml%10==0:
            text(name+f"_number_{ml}",str(ml),(.004,-.0121,z+.003),.0045,ink,cylinder)
    text(name+"_capacity","50 mL",(0,-.0129,.162),.0045,ink,cylinder)
    text(name+"_ingredient",ingredient,(0,-.0129,.179),.012,ink,cylinder)


balance=empty("digital_balance",(.18,.005,0))
cube("balance_body",(0,0,.022),(.166,.159,.04),grey,balance,.009)
for x in (-.06,.06):
    for y in (-.05,.05):cyl("balance_foot",(x,y,.003),.01,.006,dark,balance)
cyl("balance_pan",(0,.013,.047),.065,.009,steel,balance)
cube("balance_screen",(-.012,-.081,.024),(.078,.003,.025),dark,balance,.001)
display=text("balance_reading","0.000 g",(-.012,-.083,.026),.012,mint,balance)
for x,label in ((.043,"T"),(.065,"O")):
    cube("balance_button_"+label,(x,-.081,.025),(.015,.003,.012),blue,balance,.002)
    text("button_text_"+label,label,(x,-.083,.026),.007,white,balance)

beaker=empty("beaker",(.18,.018,.052))
lathe("beaker_glass",[(0,0),(.037,0),(.044,.004),(.044,.116),(.045,.12),(.044,.122),(.042,.121),(.0415,.117),(.0415,.006),(.037,.004),(0,.004)],glass,beaker,spout=True)
line("beaker_rim_outline",[((.044+.009*max(0.,math.cos(a))**24)*math.cos(a),(.044+.009*max(0.,math.cos(a))**24)*math.sin(a),.122) for a in [i*2*math.pi/96 for i in range(97)]],.00035,edge,beaker)
arc("beaker_base_outline",.044,.004,0,2*math.pi,edge,beaker,.0003)
for ml in (100,200,300,400):
    z=.004+ml*1e-6/(math.pi*.0415**2)
    arc("beaker_tick_"+str(ml),.0444,z,-2.5,-.5,white,beaker,.0004)
    text("beaker_mark_"+str(ml),str(ml),(0,-.0446,z+.005),.006,white,beaker)
text("beaker_capacity","500 mL",(0,-.0446,.104),.008,white,beaker)

waste=empty("waste",(.33,-.07,0))
lathe("waste_cup",[(0,0),(.025,0),(.030,.07),(.028,.07),(.023,.004),(0,.004)],glass,waste)
arc("waste_rim_outline",.030,.07,0,2*math.pi,edge,waste,.0003)
text("waste_label","WASTE",(0,-.029,.043),.007,ink,waste)
trash=empty("trash_bin",(.32,.17,0))
cube("trash_bin_floor",(0,0,.004),(.10,.09,.008),grey,trash,.002)
for x in (-.048,.048):cube("trash_side",(x,0,.065),(.004,.09,.13),grey,trash,.002)
for y in (-.043,.043):cube("trash_front_back",(0,y,.065),(.10,.004,.13),grey,trash,.002)
text("trash_bin_label","USED",(0,-.046,.078),.012,ink,trash)
text("trash_bin_label_2","PIPETTES",(0,-.046,.059),.008,ink,trash)

pipettes={}
for ingredient,y in [(k,y) for k,y in [("A",-.12),("B",-.16)] if k in CONFIG["targets_g"]]:
    root=empty("pipette_"+ingredient,(-.23,y,.016))
    root.rotation_euler=(0,math.pi/2,0)
    # Tip at local z=0; 5 mL LDPE bulb at the other end.
    lathe("pipette_"+ingredient+"_shaft",[(.0012,0),(.003,.02),(.003,.104),(.002,.104),(.002,.02),(.0008,0),(.0012,0)],pipmat,root,16)
    bulb=lathe("pipette_"+ingredient+"_bulb",[(.003,.104),(.012,.115),(.012,.15),(.006,.161),(0,.161)],pipmat,root,16)
    pipettes[ingredient]=(root,bulb)
    cube("pipette_"+ingredient+"_rest",(-.112,y,.006),(.022,.025,.012),white,bevel=.003)
    text("pipette_"+ingredient+"_rest_label",ingredient,(-.111,y-.013,.006),.006,ink)

# Physical station markers, not overlays.
for name,(root,_,_) in bottles.items():
    x,y,_=root.location
    arc(name+"_home",.049,.0007,0,2*math.pi,ink,empty(name+"_home_marker",(x,y,0)),.00045)
cube("table",(0,.03,-.023),(.90,.66,.045),base,bevel=.018)
floor=material("Backdrop",(.87,.9,.91),rough=.8)
cube("floor",(0,0,-.07),(200,200,.02),floor,bevel=0)


def camera(name,loc,target,scale):
    data=bpy.data.cameras.new(name);o=bpy.data.objects.new(name,data)
    bpy.context.collection.objects.link(o);o.location=loc
    o.rotation_euler=(Vector(target)-o.location).to_track_quat("-Z","Y").to_euler()
    data.type="ORTHO";data.ortho_scale=scale;data.lens=55
    return o


scene=bpy.context.scene
scene.camera=camera("review_camera",(.49,-.95,.64),(.01,.035,.091),.90)
for name,loc,power,size in [("Key",(-.5,-.5,1.2),14.,.75),("Rim",(.3,.4,.9),19.,.5),("Fill",(.7,-.1,.6),7.,.5)]:
    data=bpy.data.lights.new(name,"AREA");data.energy=power;data.shape="DISK";data.size=size
    o=bpy.data.objects.new(name,data);bpy.context.collection.objects.link(o);o.location=loc
    o.rotation_euler=(Vector((0,0,.08))-o.location).to_track_quat("-Z","Y").to_euler()
scene.world.color=(.7,.7,.7)
scene.render.engine="CYCLES"
scene.cycles.device="CPU"
scene.cycles.samples=24
scene.cycles.use_denoising=True
scene.cycles.max_bounces=10
scene.cycles.transmission_bounces=8
scene.render.resolution_x=1500;scene.render.resolution_y=1000;scene.render.resolution_percentage=100
scene.render.image_settings.file_format="PNG"
scene.view_settings.view_transform="AgX"
scene.unit_settings.system="METRIC"
scene.render.film_transparent=False
scene.render.filepath=str(OUT/"design-reference.png")
scene["prototype_status"]="Visual/component prototype only; no robot dynamics, fluid validation or Dream-1 run."
scene["reference_dimensions"]="Capacities from supplied images. Outer dimensions estimated; verify before physical replication."
bpy.ops.wm.save_as_mainfile(filepath=str(OUT/"formulation-scene.blend"))
bpy.ops.render.render(write_still=True)

# A second frame makes the required small final volume visible in the large beaker.
final_liquid=cyl("beaker_liquid_final",(0,0,.004+(sum(CONFIG["targets_g"][k]/CONFIG["density_g_ml"][k] for k in CONFIG["targets_g"])*1e-6/(math.pi*.0414**2))/2),.0413,sum(CONFIG["targets_g"][k]/CONFIG["density_g_ml"][k] for k in CONFIG["targets_g"])*1e-6/(math.pi*.0414**2),water,beaker)
display.data.body=f'{sum(CONFIG["targets_g"].values()):.3f} g'
for i,(root,bulb) in enumerate(pipettes.values()):
    root.location=(.30+i*.035,.17,.008)
    root.rotation_euler=(0,(1 if i else -1)*.12,0)
scene.render.filepath=str(OUT/"final-state.png")
bpy.ops.render.render(write_still=True)
bpy.data.objects.remove(final_liquid,do_unlink=True)

manifest={"units":"m", "status":"visual assets; collision and grasp validation pending", "objects":[
    {"id":"/".join(bottles),"material":"amber glass, blue polypropylene screw cap","nominal_ml":500,"height":.212,"body_diameter":.086,"neck_inner_diameter":.037,"cap_diameter":.048,"initial_ml":200},
    {"id":"cylinder_A/cylinder_B","material":"borosilicate glass; no plastic rings","nominal_ml":50,"height":.196,"inner_diameter":.022,"outer_diameter":.025,"division_ml":1,"calibration":"TC style; cylinder reading and transfer loss require validation"},
    {"id":"beaker","material":"borosilicate glass","nominal_ml":500,"height":.122,"inner_diameter":.083},
    {"id":"/".join("pipette_"+k for k in pipettes),"material":"LDPE","nominal_ml":5,"length":.161,"bulb_diameter":.024,"tip_inner_diameter":.0016},
    {"id":"waste","material":"glass","nominal_ml":100,"height":.07,"top_diameter":.06},
    {"id":"trash_bin","material":"plastic","width":.10,"depth":.09,"height":.13}
]}
manifest["objects"]=[o for o in manifest["objects"] if not o["id"].startswith("cylinder_")]
manifest["objects"].append({"id":"balance","material":"plastic/metal body and metal pan","resolution_g":.001,"width":.166,"depth":.159})
(OUT/"asset-manifest.json").write_text(json.dumps(manifest,indent=2))
print("SCENE_READY",str(OUT))
