"""Development-only USD export for the passive compliant-pad v12 candidate.

No pxr/Isaac/MuJoCo import occurs while exporting. Optional schema audit imports
pxr only when explicitly called in a separate local CPU process. The passive
hinge has an explicit force-type angular drive representing its fixed mechanical
spring. USD angular drive quantities use degrees; runtime tensor APIs use radians.
"""
from pathlib import Path
import copy
import hashlib
import json
import math
import re
import xml.etree.ElementTree as ET
import numpy as np
from wafer_grasp_settings import (PHYSICS_DT_S, PAD_STIFFNESS_N_M,
    PAD_DAMPING_N_S_M, PAD_MASS_KG, PAD_TRAVEL_M, HINGE_CLOSED_LIMIT_RAD)


def _v(values):return '('+', '.join(f'{float(v):.12g}' for v in values)+')'
def _vec(element,key,size,default=None):
    value=element.get(key,default)
    if value is None:raise ValueError('Missing model attribute '+key)
    a=np.fromstring(value,sep=' ')
    if a.shape!=(size,) or not np.isfinite(a).all():raise ValueError('Invalid model attribute '+key)
    return a


def _tool_collision_profile(tree,bodies):
    """Match the six moving weld groups; reject unsupported filtering changes."""
    flags=tree.find('./option/flag')
    if flags is not None and any(flags.get(k)=='disable'for k in ('contact','constraint','filterparent')):
        raise ValueError('Full tool collision export requires active parent/contact filtering.')
    if tree.findall('./contact/exclude'):raise ValueError('Explicit body exclusions need a reviewed collision export.')
    parent={child.get('name'):body.get('name')for body in bodies.values()for child in body.findall('body')}
    inventory={name:sorted(g.get('name')for g in body.findall('geom'))for name,body in bodies.items()}
    masks={g.get('name'):[int(g.get('contype','1')),int(g.get('conaffinity','1'))]
           for body in bodies.values()for g in body.findall('geom')}
    if any(not name for names in inventory.values()for name in names):raise ValueError('Named tool colliders are required.')
    if len(masks)!=sum(map(len,inventory.values())):raise ValueError('Tool collider names must be unique.')
    for name,body in bodies.items():
        joints=body.findall('joint')+body.findall('freejoint')
        if len(joints)!=1:raise ValueError('Each full tool body must be a separate moving weld group.')
    explicit=sorted(sorted((p.get('geom1'),p.get('geom2')))for p in tree.findall('./contact/pair')
                    if p.get('geom1')in masks and p.get('geom2')in masks)
    filtered=[];allowed=[];names=sorted(bodies)
    for i,a in enumerate(names):
        for b in names[i+1:]:
            connected=parent.get(a)==b or parent.get(b)==a;decisions=set()
            for ga in inventory[a]:
                for gb in inventory[b]:
                    ca,aa=masks[ga];cb,ab=masks[gb]
                    decisions.add(sorted((ga,gb))in explicit or (not connected and bool((ca&ab)or(cb&aa))))
            if len(decisions)!=1:raise ValueError('Mixed collider filtering within a body pair needs explicit export support.')
            (allowed if decisions.pop()else filtered).append([a,b])
    if ['lower_contact_pad','upper_contact_pad']not in explicit or ['lower_pad','upper_pad']not in allowed:
        raise ValueError('The explicit opposing-pad collision must remain enabled.')
    return dict(version='six_body_mujoco_pairs_v1',enabled_self_collisions=True,
        body_colliders=inventory,body_parents=parent,collider_masks=masks,
        filtered_body_pairs=filtered,allowed_body_pairs=allowed,explicit_geom_pairs=explicit,
        scope='Internal collision eligibility only; contact responses differ between physics engines.')


def verify_compiled_tool_collisions(scene_path,profile):
    """Check every cross-body tool collider pair in the qualified local scene."""
    import mujoco
    model=mujoco.MjModel.from_xml_path(str(scene_path))
    disabled=int(model.opt.disableflags)
    for flag in ('mjDSBL_CONSTRAINT','mjDSBL_CONTACT','mjDSBL_FILTERPARENT'):
        if disabled&int(getattr(mujoco.mjtDisableBit,flag)):raise ValueError('Compiled tool contact filtering is disabled.')
    if model.nexclude or mujoco.get_mjcb_contactfilter()is not None:
        raise ValueError('Compiled collision exclusions/callback differ from the exported profile.')
    ids={name:model.body(name).id for name in profile['body_colliders']};geom_ids={}
    for name,bid in ids.items():
        if int(model.body_weldid[bid])!=bid:raise ValueError('A full tool body was welded to another body.')
        parent=profile['body_parents'].get(name)
        if parent is not None and int(model.body_parentid[bid])!=ids[parent]:raise ValueError('Compiled tool parent differs.')
        geoms=list(range(int(model.body_geomadr[bid]),int(model.body_geomadr[bid]+model.body_geomnum[bid])))
        if sorted(model.geom(g).name for g in geoms)!=profile['body_colliders'][name]:raise ValueError('Compiled tool colliders differ.')
        geom_ids[name]=geoms
    tool_ids={g for geoms in geom_ids.values()for g in geoms}
    explicit={tuple(sorted((int(a),int(b))))for a,b in zip(model.pair_geom1,model.pair_geom2)if a in tool_ids and b in tool_ids}
    if sorted(sorted((model.geom(a).name,model.geom(b).name))for a,b in explicit)!=profile['explicit_geom_pairs']:
        raise ValueError('Compiled explicit tool contact pairs differ.')
    count=enabled=0;names=sorted(ids)
    for i,a in enumerate(names):
        for b in names[i+1:]:
            wa=int(model.body_weldid[ids[a]]);wb=int(model.body_weldid[ids[b]])
            pa=int(model.body_weldid[model.body_parentid[wa]]);pb=int(model.body_weldid[model.body_parentid[wb]])
            body_filtered=wa==wb or (wa!=0 and wb!=0 and(wa==pb or wb==pa))
            expected=[a,b]in profile['allowed_body_pairs']
            for ga in geom_ids[a]:
                for gb in geom_ids[b]:
                    actual=(tuple(sorted((ga,gb)))in explicit or(not body_filtered and bool(
                        (model.geom_contype[ga]&model.geom_conaffinity[gb])or(model.geom_contype[gb]&model.geom_conaffinity[ga]))))
                    if actual!=expected:raise ValueError('Compiled/exported tool collision pair differs: '+model.geom(ga).name+'/'+model.geom(gb).name)
                    count+=1;enabled+=int(actual)
    return dict(status='COMPILED_TOOL_COLLISION_PAIRS_MATCH',collider_pair_count=count,
        enabled_collider_pairs=enabled,filtered_collider_pairs=count-enabled,
        compiled_collider_masks={model.geom(g).name:[int(model.geom_contype[g]),int(model.geom_conaffinity[g])]for g in sorted(tool_ids)},
        scene_sha256=hashlib.sha256(Path(scene_path).read_bytes()).hexdigest(),mujoco_version=mujoco.__version__)


def verify_tool_collision_prim(root,profile):
    """Read composed filtering; this does not certify a physical contact response."""
    from pxr import Usd,UsdPhysics
    if profile.get('version')!='six_body_mujoco_pairs_v1':raise ValueError('Unknown tool collision profile.')
    if root.GetAttribute('physxArticulation:enabledSelfCollisions').Get()is not True:
        raise ValueError('Full tool articulation self-collision is disabled.')
    prims=list(Usd.PrimRange(root,Usd.TraverseInstanceProxies()))
    bodies={p.GetName():p for p in prims if p.HasAPI(UsdPhysics.RigidBodyAPI)}
    if set(bodies)!=set(profile['body_colliders']):raise ValueError('Composed collision body inventory differs.')
    names=sorted(bodies);pairs={(a,b)for i,a in enumerate(names)for b in names[i+1:]}
    filtered={tuple(p)for p in profile['filtered_body_pairs']};allowed={tuple(p)for p in profile['allowed_body_pairs']}
    if filtered&allowed or filtered|allowed!=pairs:raise ValueError('Incomplete or contradictory tool pair map.')
    actual=set()
    for name,body in bodies.items():
        colliders=[p.GetName()for p in Usd.PrimRange(body,Usd.TraverseInstanceProxies())if p.HasAPI(UsdPhysics.CollisionAPI)]
        if sorted(colliders)!=profile['body_colliders'][name]:raise ValueError('Composed tool collider inventory differs.')
    paths={str(p.GetPath()):name for name,p in bodies.items()}
    for prim in prims:
        if prim.IsA(UsdPhysics.Joint)and UsdPhysics.Joint(prim).GetCollisionEnabledAttr().Get()is not True:
            raise ValueError('Implicit joint collision filtering overrides the explicit tool pair map.')
        if prim.HasAPI(UsdPhysics.FilteredPairsAPI):
            name=paths.get(str(prim.GetPath()))
            if name is None:raise ValueError('Unexpected root/collider-level tool collision filter.')
            for target in UsdPhysics.FilteredPairsAPI(prim).GetFilteredPairsRel().GetTargets():
                other=paths.get(str(target))
                if other is None or other==name:raise ValueError('Unexpected external/self tool filter target.')
                actual.add(tuple(sorted((name,other))))
    expected={tuple(p)for p in profile['filtered_body_pairs']}
    if actual!=expected:raise ValueError('Composed tool filtered pairs differ from the local model.')
    return dict(status='COMPOSED_TOOL_COLLISION_PAIRS_MATCH',enabled_self_collisions=True,
        filtered_body_pairs=sorted(map(list,actual)),enabled_body_pairs=profile['allowed_body_pairs'],
        scope='Composed USD eligibility readback; physical collision response remains run-specific.')


def _rotation(element):
    if element.get('quat') is not None:
        w,x,y,z=_vec(element,'quat',4)
        if abs(w*w+x*x+y*y+z*z-1)>1e-8:raise ValueError('Non-unit authored quaternion.')
        return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                         [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                         [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
    angles=_vec(element,'euler',3,'0 0 0')
    if np.count_nonzero(np.abs(angles)>1e-14)>1:raise ValueError('Multi-axis MJCF Euler export needs explicit convention review.')
    r=np.eye(3)
    for axis,angle in enumerate(angles):
        c,s=math.cos(angle),math.sin(angle)
        a=np.eye(3);i,j=(axis+1)%3,(axis+2)%3;a[i,i]=a[j,j]=c;a[i,j]=-s;a[j,i]=s;r=r@a
    return r


def _quat(r):
    tr=float(np.trace(r))
    if tr>0:
        s=2*math.sqrt(1+tr);q=np.array([(r[2,1]-r[1,2])/s,(r[0,2]-r[2,0])/s,(r[1,0]-r[0,1])/s,s/4])
    else:
        i=int(np.argmax(np.diag(r)));j=(i+1)%3;k=(i+2)%3;s=2*math.sqrt(1+r[i,i]-r[j,j]-r[k,k])
        q=np.empty(4);q[i]=s/4;q[j]=(r[j,i]+r[i,j])/s;q[k]=(r[k,i]+r[i,k])/s;q[3]=(r[k,j]-r[j,k])/s
    q/=np.linalg.norm(q)
    if q[3]<0:q=-q
    return q


def _prism_mass(vertices,mass):
    if len(vertices)%2 or len(vertices)<6:raise ValueError('Expected a two-face convex prism mesh.')
    n=len(vertices)//2;a,b=vertices[:n],vertices[n:];delta=b-a
    if not np.allclose(delta,delta[0],atol=1e-10,rtol=0):raise ValueError('Mesh is not a parallel prism.')
    depth=np.linalg.norm(delta[0]);w=delta[0]/depth;u=a[1]-a[0];u/=np.linalg.norm(u);v=np.cross(w,u)
    if depth<=0 or abs(u@w)>1e-8 or np.max(np.abs((a-a[0])@w))>1e-9:raise ValueError('Invalid prism planes.')
    frame=np.column_stack([u,v,w]);xy=(a-a[0])@frame[:,:2];x,y=xy.T;xx,yy=np.roll(x,-1),np.roll(y,-1)
    cross=x*yy-xx*y;area=cross.sum()/2
    if abs(area)<1e-12:raise ValueError('Degenerate prism area.')
    center=np.array([np.sum((x+xx)*cross)/(6*area),np.sum((y+yy)*cross)/(6*area)])
    moments=np.array([[np.sum((x*x+x*xx+xx*xx)*cross)/(12*area),np.sum((2*x*y+x*yy+xx*y+2*xx*yy)*cross)/(24*area)],
                      [0,np.sum((y*y+y*yy+yy*yy)*cross)/(12*area)]])
    moments[1,0]=moments[0,1];cov=np.zeros((3,3));cov[:2,:2]=moments-np.outer(center,center);cov[2,2]=depth**2/12
    cov=frame@cov@frame.T;com=a[0]+frame@np.r_[center,depth/2]
    return com,mass*(np.trace(cov)*np.eye(3)-cov)


def _inertial(geoms,meshes):
    pieces=[]
    for geom in geoms:
        mass=float(geom.get('mass','0'))
        if not math.isfinite(mass) or mass<=0:raise ValueError('Dynamic collider lacks positive authored mass.')
        kind=geom.get('type');center=np.zeros(3)
        if kind=='box':
            size=2*_vec(geom,'size',3);inertia=np.diag(mass*(np.sum(size**2)-size**2)/12)
        elif kind=='cylinder':
            radius,half=_vec(geom,'size',2);inertia=np.diag([mass*(3*radius**2+4*half**2)/12]*2+[mass*radius**2/2])
        elif kind=='mesh':center,inertia=_prism_mass(meshes[geom.get('mesh')],mass)
        else:raise ValueError('Unsupported dynamic primitive '+str(kind))
        rotation=_rotation(geom);position=_vec(geom,'pos',3,'0 0 0')
        pieces.append((mass,position+rotation@center,rotation@inertia@rotation.T))
    total=sum(p[0] for p in pieces);com=sum(m*p for m,p,i in pieces)/total
    inertia=sum(i+m*(np.dot(p-com,p-com)*np.eye(3)-np.outer(p-com,p-com)) for m,p,i in pieces)
    eig,axes=np.linalg.eigh(inertia)
    if np.min(eig)<=0:raise ValueError('Nonpositive compound inertia.')
    if np.linalg.det(axes)<0:axes[:,0]*=-1
    return dict(mass_kg=float(total),center_of_mass_local_m=com.tolist(),diagonal_inertia_kg_m2=eig.tolist(),
                principal_axes_xyzw=_quat(axes).tolist(),inertia_tensor_local_kg_m2=inertia.tolist())


def _body_inertial(body,meshes,compiler):
    """Respect explicit body inertials, including massless overlap colliders."""
    element=body.find('inertial')
    if element is None:return _inertial(body.findall('geom'),meshes)
    if compiler.get('inertiafromgeom','auto') not in ('auto','false'):
        raise ValueError('Compiler overrides explicit body inertia; export needs source review.')
    mass=float(element.get('mass','nan'));center=_vec(element,'pos',3,'0 0 0')
    if not math.isfinite(mass) or mass<=0:raise ValueError('Invalid explicit body mass.')
    if element.get('fullinertia') is not None:
        if element.get('quat') or element.get('euler'):raise ValueError('Ambiguous fullinertia orientation.')
        xx,yy,zz,xy,xz,yz=_vec(element,'fullinertia',6)
        inertia=np.array([[xx,xy,xz],[xy,yy,yz],[xz,yz,zz]])
    else:
        rotation=_rotation(element);inertia=rotation@np.diag(_vec(element,'diaginertia',3))@rotation.T
    eig,axes=np.linalg.eigh(inertia)
    if eig.min()<=0 or 2*eig.max()>eig.sum()+1e-14:raise ValueError('Invalid explicit physical inertia.')
    if np.linalg.det(axes)<0:axes[:,0]*=-1
    return dict(mass_kg=mass,center_of_mass_local_m=center.tolist(),diagonal_inertia_kg_m2=eig.tolist(),
        principal_axes_xyzw=_quat(axes).tolist(),inertia_tensor_local_kg_m2=inertia.tolist(),
        source='Explicit MJCF body inertial; zero-mass overlapping collision pieces do not contribute mass.')


def _header(body,root_apis=''):
    schema='' if not root_apis else ' (prepend apiSchemas = ['+root_apis+'])'
    return '#usda 1.0\n(\n defaultPrim = "Asset"\n metersPerUnit = 1\n upAxis = "Z"\n)\ndef Xform "Asset"'+schema+' {\n'+body+'\n}\n'


def _rigid_properties(inertial,kinematic=False):
    lines=['bool physics:rigidBodyEnabled = true','bool physics:kinematicEnabled = '+str(kinematic).lower(),
           'bool physxRigidBody:disableGravity = '+str(kinematic).lower(),
           'bool physxRigidBody:enableCCD = '+str(not kinematic).lower(),'int physxRigidBody:solverPositionIterationCount = 64',
           'int physxRigidBody:solverVelocityIterationCount = 4']
    if inertial is not None:
        q=inertial['principal_axes_xyzw']
        lines += ['float physics:mass = '+repr(inertial['mass_kg']),
                  'point3f physics:centerOfMass = '+_v(inertial['center_of_mass_local_m']),
                  'float3 physics:diagonalInertia = '+_v(inertial['diagonal_inertia_kg_m2']),
                  'quatf physics:principalAxes = '+_v([q[3],*q[:3]])]
    return '\n'.join(lines)


def _material(name,friction):
    return ('def Material "'+name+'" (prepend apiSchemas = ["PhysicsMaterialAPI", "PhysxMaterialAPI"]) {\n'
            f'float physics:staticFriction = {friction:.12g}\nfloat physics:dynamicFriction = {friction:.12g}\n'
            'float physics:restitution = 0\ntoken physxMaterial:frictionCombineMode = "max"\n}\n')


def _effective_pad_friction(tree):
    """Map explicit wafer-pad pair sliding friction to PhysX pad materials.

    PhysX material max-combination cannot independently override arbitrary
    pairs. This mapping is exact for the intended wafer/pad pair only when the
    wafer's own coefficient does not exceed the authored pair coefficient.
    Other pad contacts can differ and are explicitly outside that equivalence.
    """
    wafer=tree.find(".//body[@name='wafer']")
    if wafer is None or not wafer.findall('geom'):raise ValueError('Missing wafer colliders for friction mapping.')
    wafer_mu={g.get('name'):float(_vec(g,'friction',3,'.3 .0001 .0001')[0]) for g in wafer.findall('geom')}
    mapping={};covered=set()
    for pair in tree.findall('./contact/pair'):
        names=[pair.get('geom1'),pair.get('geom2')]
        present=[name for name in names if name in wafer_mu]
        if not present:continue
        if len(present)!=1:raise ValueError('Unexpected wafer-to-wafer explicit pair.')
        wafer_name=present[0];other=names[1-names.index(wafer_name)]
        if not other or '_contact_pad' not in other:raise ValueError('Unexpected explicit wafer contact pair.')
        mu=_vec(pair,'friction',5)
        if not np.allclose(mu[:2],mu[0],atol=1e-12,rtol=0) or mu[0]<wafer_mu[wafer_name]-1e-12:
            raise ValueError('Explicit anisotropic/lower-friction pair is not representable by this material mapping.')
        if other in mapping and abs(mapping[other]-float(mu[0]))>1e-12:raise ValueError('Conflicting pad pair friction.')
        mapping[other]=float(mu[0])
        covered.add((wafer_name,other))
    pads={g.get('name') for g in tree.findall('.//geom') if '_contact_pad' in g.get('name','')}
    if set(mapping)!=pads or covered!={(w,p) for w in wafer_mu for p in pads}:
        raise ValueError('Each wafer collider/contact-pad pair requires an explicit friction pair.')
    return mapping


def _shape(geom,meshes,material_path,zero_position=False):
    name=geom.get('name')
    if not name or not re.fullmatch('[A-Za-z_][A-Za-z0-9_]*',name):raise ValueError('Invalid collider name.')
    kind=geom.get('type');usd_kind={'box':'Cube','cylinder':'Cylinder','mesh':'Mesh'}.get(kind)
    if usd_kind is None:raise ValueError('Unsupported shared primitive '+str(kind))
    schemas='"PhysicsCollisionAPI", "PhysxCollisionAPI", "MaterialBindingAPI"'
    if kind=='mesh':schemas+=', "PhysicsMeshCollisionAPI"'
    position=np.zeros(3) if zero_position else _vec(geom,'pos',3,'0 0 0');q=_quat(_rotation(geom))
    lines=[f'def {usd_kind} "{name}" (prepend apiSchemas = [{schemas}]) {{',
           'double3 xformOp:translate = '+_v(position),'quatf xformOp:orient = '+_v([q[3],*q[:3]]),
           'color3f[] primvars:displayColor = ['+_v(_vec(geom,'rgba',4,'.7 .7 .7 1')[:3])+']',
           'float physxCollision:contactOffset = 0.0001','float physxCollision:restOffset = 0',
           f'rel material:binding:physics = <{material_path}>']
    order=['xformOp:translate','xformOp:orient']
    if kind=='box':
        lines+=['double size = 1','double3 xformOp:scale = '+_v(2*_vec(geom,'size',3))];order+=['xformOp:scale']
    elif kind=='cylinder':
        radius,half=_vec(geom,'size',2);lines += [f'double radius = {radius:.12g}',f'double height = {2*half:.12g}','uniform token axis = "Z"']
    else:
        vertices=meshes[geom.get('mesh')];_prism_mass(vertices,1.);n=len(vertices)//2
        faces=[list(reversed(range(n))),list(range(n,2*n))]+[[i,(i+1)%n,(i+1)%n+n,i+n] for i in range(n)]
        lines += ['uniform token subdivisionScheme = "none"','uniform token physics:approximation = "convexHull"',
                  'point3f[] points = ['+', '.join(_v(v) for v in vertices)+']',
                  'int[] faceVertexCounts = ['+', '.join(str(len(f)) for f in faces)+']',
                  'int[] faceVertexIndices = ['+', '.join(str(i) for f in faces for i in f)+']']
    lines += ['uniform token[] xformOpOrder = ['+', '.join('"'+v+'"' for v in order)+']','}']
    return '\n'.join(lines)


def usd_drive_specification(actuation):
    """Map SI spring constants to OpenUSD force-drive quantities, not acceleration.

    Angular USD stiffness/damping are torque per degree and per degree/second:
    https://openusd.org/release/api/class_usd_physics_drive_a_p_i.html
    Tensor gain setters alone do not create a drive and can otherwise be inert:
    https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/extensions/runtime/source/omni.physics.tensors/docs/api/python.html
    """
    angular=actuation['joint_type']=='hinge'
    if actuation['joint_type'] not in ('hinge','slide'):raise ValueError('Unsupported drive joint type.')
    mode=actuation['mode']
    if mode in ('constant_opening_spring','constant_passive_spring'):target=float(actuation['constant_target'])
    elif mode=='powered_position':target=float(actuation['target_limits'][0])
    else:raise ValueError('Unsupported mechanism drive mode.')
    values=[float(actuation[k]) for k in ('stiffness','damping','effort_limit')]
    if not np.isfinite([*values,target]).all() or min(values)<0 or values[0]<=0 or values[2]<=0:
        raise ValueError('Invalid mechanism drive coefficients.')
    scale=math.pi/180 if angular else 1.
    return dict(instance='angular' if angular else 'linear',type='force',
        stiffness=values[0]*scale,damping=values[1]*scale,maxForce=values[2],
        targetPosition=math.degrees(target) if angular else target,targetVelocity=0.,
        position_unit='deg' if angular else 'm',
        stiffness_unit='Nm/deg' if angular else 'N/m',
        damping_unit='Nm*s/deg' if angular else 'N*s/m',effort_unit='Nm' if angular else 'N')


def verify_drive_prim(prim,actuation):
    """Require the explicit composed drive; gain readbacks cannot prove presence."""
    from pxr import UsdPhysics
    expected=usd_drive_specification(actuation);instance=expected['instance']
    if not actuation.get('authored_usd_drive') or not prim.HasAPI(UsdPhysics.DriveAPI,instance):
        raise ValueError('Missing authored mechanism DriveAPI: '+instance)
    drives=[str(x) for x in prim.GetAppliedSchemas() if str(x).startswith('PhysicsDriveAPI:')]
    if drives!=['PhysicsDriveAPI:'+instance]:raise ValueError('Unexpected additional mechanism drive instance.')
    drive=UsdPhysics.DriveAPI(prim,instance)
    if drive.GetTypeAttr().Get()!='force':raise ValueError('Mechanism drive must use force, not acceleration.')
    measured={}
    for key,method in [('stiffness','GetStiffnessAttr'),('damping','GetDampingAttr'),
                       ('maxForce','GetMaxForceAttr'),('targetPosition','GetTargetPositionAttr'),
                       ('targetVelocity','GetTargetVelocityAttr')]:
        attr=getattr(drive,method)()
        if not attr.HasAuthoredValueOpinion():raise ValueError('Missing explicit drive '+key)
        value=float(attr.Get())
        if not math.isfinite(value) or not math.isclose(value,expected[key],rel_tol=1e-6,abs_tol=1e-9):
            raise ValueError('Composed mechanism drive '+key+' differs from the fixed source profile.')
        measured[key]=value
    return dict(status='COMPOSED_FORCE_DRIVE_VERIFIED',joint_path=str(prim.GetPath()),
        instance=instance,type='force',values=measured,units={k:v for k,v in expected.items() if k.endswith('_unit')},
        scope='Drive presence/type and authored units; native parameters and targets require separate backend readback.')


def verify_joint_frame_prim(prim,specification):
    from pxr import UsdPhysics
    joint=UsdPhysics.Joint(prim)
    typed=(UsdPhysics.RevoluteJoint if specification['type']=='hinge' else UsdPhysics.PrismaticJoint)(prim)
    expected_limits=np.asarray(specification['limits'])*(180/math.pi if specification['type']=='hinge' else 1.)
    tolerance=1e-5 if specification['type']=='hinge' else 1e-9
    if typed.GetAxisAttr().Get()!='Z' or not np.allclose([typed.GetLowerLimitAttr().Get(),typed.GetUpperLimitAttr().Get()],expected_limits,atol=tolerance,rtol=0):
        raise ValueError('Composed passive joint axis or travel differs: '+specification['name'])
    for key,getter in (('local_pos0_m','GetLocalPos0Attr'),('local_pos1_m','GetLocalPos1Attr')):
        actual=np.asarray(getattr(joint,getter)().Get(),dtype=float)
        if not np.allclose(actual,specification[key],atol=1e-8,rtol=0):
            raise ValueError('Composed passive joint anchor differs: '+specification['name'])
    for key,getter in (('local_rot0_xyzw','GetLocalRot0Attr'),('local_rot1_xyzw','GetLocalRot1Attr')):
        value=getattr(joint,getter)().Get();actual=np.r_[value.GetImaginary(),value.GetReal()]
        expected=np.asarray(specification[key])
        if min(np.linalg.norm(actual-expected),np.linalg.norm(actual+expected))>1e-7:
            raise ValueError('Composed passive joint axis frame differs: '+specification['name'])
    return dict(status='COMPOSED_JOINT_FRAMES_VERIFIED',joint=specification['name'])


def _joint_export(tree,parent,child,passive_pad_mass=None,handle_profile=None):
    joints=child.findall('joint')
    if len(joints)!=1:raise ValueError('Each exported moving child requires one explicit joint.')
    joint=joints[0];kind=joint.get('type','hinge');name=joint.get('name')
    if kind not in ('hinge','slide'):raise ValueError('Unsupported mechanism joint type: '+kind)
    axis=_vec(joint,'axis',3);axis/=np.linalg.norm(axis)
    if not np.isfinite(axis).all():raise ValueError('Invalid joint axis.')
    helper=np.array([0.,1.,0.]) if abs(axis[1])<.9 else np.array([1.,0.,0.])
    x=np.cross(helper,axis);x/=np.linalg.norm(x);basis=np.column_stack([x,np.cross(axis,x),axis])
    child_r=_rotation(child);child_p=_vec(child,'pos',3,'0 0 0');anchor=_vec(joint,'pos',3,'0 0 0')
    q0=_quat(child_r@basis);q1=_quat(basis);limits=_vec(joint,'range',2)
    authored_limits=np.degrees(limits) if kind=='hinge' else limits
    schema='PhysicsRevoluteJoint' if kind=='hinge' else 'PhysicsPrismaticJoint'
    source=(f'def {schema} "{name}" {{\n'
        f'rel physics:body0 = </Asset/{parent.get("name")}>\nrel physics:body1 = </Asset/{child.get("name")}>\n'
        'point3f physics:localPos0 = '+_v(child_p+child_r@anchor)+'\npoint3f physics:localPos1 = '+_v(anchor)+'\n'
        'quatf physics:localRot0 = '+_v([q0[3],*q0[:3]])+'\nquatf physics:localRot1 = '+_v([q1[3],*q1[:3]])+'\n'
        f'uniform token physics:axis = "Z"\nfloat physics:lowerLimit = {authored_limits[0]:.12g}\nfloat physics:upperLimit = {authored_limits[1]:.12g}\n'
        'bool physics:collisionEnabled = false\n}\n')
    drives=[x for x in tree.findall('./actuator/position') if x.get('joint')==name]
    if len(drives)>1:raise ValueError('Multiple drive definitions for one mechanism joint.')
    if drives:
        drive=drives[0];force=_vec(drive,'forcerange',2);control=_vec(drive,'ctrlrange',2)
        if abs(force[0]+force[1])>1e-12:raise ValueError('Asymmetric drive limit requires separate implementation.')
        drive_damping=float(drive.get('kv'));passive_damping=float(joint.get('damping','0'))
        actuation=dict(mode='powered_position',joint_name=name,joint_type=kind,
            stiffness=float(drive.get('kp')),damping=drive_damping+passive_damping,
            drive_damping=drive_damping,joint_viscous_damping=passive_damping,effort_limit=float(force[1]),
            target_limits=control.tolist(),position_unit='rad' if kind=='hinge' else 'm',
            effort_unit='Nm' if kind=='hinge' else 'N',authored_usd_drive=False)
    else:
        stiffness=float(joint.get('stiffness'));damping=float(joint.get('damping'));rest=float(joint.get('springref'))
        if handle_profile is not None:
            h=handle_profile
            if kind!='hinge' or name!=h['joint_name'] or child.get('name')!=h['body_name'] or parent.get('name')!=h['parent_body']:
                raise ValueError('Passive handle-paddle topology differs.')
            if (not np.array_equal(axis,h['axis_local']) or not np.allclose(limits,h['limits_rad'],atol=1e-12,rtol=0)
                    or stiffness!=h['stiffness_Nm_rad'] or damping!=h['damping_Nms_rad'] or rest!=h['rest_rad']):
                raise ValueError('Passive handle-paddle SI profile differs.')
            if not np.allclose(anchor,h.get('joint_anchor_local_m',[0.,0.,0.]),atol=1e-12,rtol=0)or not np.allclose(child_p,h['position_parent_m'],atol=1e-12,rtol=0):
                raise ValueError('Passive handle-paddle joint anchor or body origin differs.')
            bound=stiffness*float(np.max(np.abs(limits-rest)))+damping*h['velocity_guard_rad_s']
            if not bound<h['force_cap_Nm'] or not 0<h['torque_guard_Nm']<h['force_cap_Nm']:
                raise ValueError('Handle-paddle torque guard or native cap can conceal clipping.')
            actuation=dict(mode='constant_passive_spring',joint_name=name,joint_type=kind,
                stiffness=stiffness,damping=damping,constant_target=rest,effort_limit=h['force_cap_Nm'],
                velocity_limit=h['velocity_limit_rad_s'],diagnostic_velocity_limit=h['velocity_guard_rad_s'],
                diagnostic_force_limit=h['torque_guard_Nm'],mass_kg=h['mass_kg'],position_unit='rad',effort_unit='Nm',authored_usd_drive=False)
            if 'operating_torque_guard_Nm' in h:
                if h['operating_torque_guard_Nm']!=.060 or not h['operating_torque_guard_Nm']<h['force_cap_Nm']:
                    raise ValueError('The accepted torque limit must be 60 mNm and lower than the numerical drive cap.')
                actuation['operating_force_limit']=h['operating_torque_guard_Nm']
        elif kind=='hinge':
            speed_limit=10.;static_bound=stiffness*float(np.max(np.abs(limits-rest)))
            if name!='pliers_hinge' or stiffness!=.45 or damping!=.2 or rest!=-.2:
                raise ValueError('Expected the reviewed V11 passive opening hinge SI profile.')
            effort_limit=.3
            actuation=dict(mode='constant_opening_spring',joint_name=name,joint_type=kind,
                stiffness=stiffness,damping=damping,constant_target=rest,effort_limit=effort_limit,
                velocity_limit=speed_limit,maximum_static_spring_torque_Nm=static_bound,
                effort_limit_basis='Fixed0.3Nm native cap. Every recorded physics step must also pass the unchanged0.3Nm raw spring/damper torque guard; no automatic cap inflation.',
                position_unit='rad',effort_unit='Nm',authored_usd_drive=False)
        else:
            expected={'lower_pad_compression':('lower_pad',[0.,0.,-1.]),
                      'upper_pad_compression':('upper_pad',[0.,0.,1.])}
            if name not in expected or child.get('name')!=expected[name][0]:raise ValueError('Unknown passive pad slide.')
            if not np.array_equal(axis,np.asarray(expected[name][1])):raise ValueError('Pad compression axis changed.')
            if passive_pad_mass is None or not math.isfinite(passive_pad_mass) or passive_pad_mass<=0:raise ValueError('Pad mass is required.')
            if not np.allclose(limits,[0.,PAD_TRAVEL_M],atol=1e-12,rtol=0) or rest!=0. or stiffness!=PAD_STIFFNESS_N_M:
                raise ValueError('Pad spring travel/rest/stiffness differs from reviewed SI profile.')
            if not math.isclose(passive_pad_mass,PAD_MASS_KG,abs_tol=1e-12,rel_tol=0) or damping!=PAD_DAMPING_N_S_M:
                raise ValueError('Pad mass or damping differs from shared settings.')
            bound=stiffness*PAD_TRAVEL_M+damping*.05
            if bound>=8.:raise ValueError('Native8N cap can bind inside the declared position/speed rectangle.')
            actuation=dict(mode='constant_passive_spring',joint_name=name,joint_type=kind,
                stiffness=stiffness,damping=damping,constant_target=rest,effort_limit=8.,
                velocity_limit=1.,diagnostic_velocity_limit=.05,diagnostic_force_limit=4.,mass_kg=passive_pad_mass,
                maximum_spring_damper_force_in_position_speed_rectangle_N=bound,
                effort_limit_basis='8N exceeds K*travel+D*0.05m/s; stop on saved speed>0.05m/s or uncapped force magnitude>4N. Native1m/s ceiling does not conceal this diagnostic.',
                position_unit='m',effort_unit='N',authored_usd_drive=False)
    # Presence must be authored before PhysX creates the articulation. Setting
    # runtime stiffness/damping/force limits does not enable a missing drive.
    drive=usd_drive_specification(actuation);instance=drive['instance']
    source=source.replace(f'def {schema} "{name}" {{',
        f'def {schema} "{name}" (prepend apiSchemas = ["PhysicsDriveAPI:{instance}"]) {{')
    attributes=f'uniform token drive:{instance}:physics:type = "force"\n'
    attributes+=''.join(f'float drive:{instance}:physics:{key} = {drive[key]:.12g}\n'
        for key in ('stiffness','damping','maxForce','targetPosition','targetVelocity'))
    source=source[:-2]+attributes+'}\n'
    actuation.update(authored_usd_drive=True,usd_drive=drive)
    return source,dict(name=name,type=kind,parent=parent.get('name'),child=child.get('name'),
        axis_local=axis.tolist(),limits=limits.tolist(),position_unit=actuation['position_unit'],
        local_pos0_m=(child_p+child_r@anchor).tolist(),local_pos1_m=anchor.tolist(),
        local_rot0_xyzw=q0.tolist(),local_rot1_xyzw=q1.tolist()),actuation


def export_assets(destination,scenario=None,model_module='wafer_retention_candidate_v12',geometry_transform=None):
    """Export the current shared fixture geometry; no physics engine is run."""
    import importlib
    if model_module != 'wafer_retention_candidate_v12':
        raise ValueError('Select an explicitly reviewed mechanism source module.')
    model=importlib.import_module(model_module);build_model_xml=model.build_model_xml
    passive=importlib.import_module('wafer_clamp_model')
    parent_model=importlib.import_module('wafer_retention_candidate_v3')
    dependency_paths={'mechanism':Path(model.__file__),'retention_parent':Path(parent_model.__file__),'passive_base':Path(passive.__file__),
                      'design':Path(passive._design().__file__),'settings':Path(__file__).with_name('wafer_grasp_settings.py')}
    dependencies={name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in dependency_paths.items()}
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    xml,metadata=build_model_xml(scenario);tree=ET.fromstring(xml)
    if geometry_transform is not None:
        geometry_transform(tree,metadata)
        xml=ET.tostring(tree,encoding='unicode')
    if metadata.get('physics_dt_s')!=PHYSICS_DT_S or float(tree.find('option').get('timestep'))!=PHYSICS_DT_S:
        raise ValueError('Export cadence differs from shared settings.')
    meshes={m.get('name'):np.fromstring(m.get('vertex',''),sep=' ').reshape(-1,3) for m in tree.findall('./asset/mesh')}
    if any(not np.isfinite(v).all() for v in meshes.values()):raise ValueError('Invalid shared mesh.')
    world=tree.find('worldbody')
    if world is None:raise ValueError('Missing shared worldbody.')
    lower=world.find("body[@name='lower_half']");wafer=world.find("body[@name='wafer']")
    if lower is None or wafer is None:raise ValueError('Missing free tool/wafer roots.')
    bodies={body.get('name'):body for body in lower.iter('body')}
    handles=metadata.get('handle_paddles',[])
    if handles and ([h['body_name']for h in handles]!=['lower_grip_paddle','upper_grip_paddle'] or
                    [h['joint_name']for h in handles]!=['lower_grip_paddle_pitch','upper_grip_paddle_pitch']):
        raise ValueError('Unexpected passive handle-paddle identities.')
    if set(bodies)!={'lower_half','upper_half','lower_pad','upper_pad',*[h['body_name']for h in handles]}:
        raise ValueError('Unexpected free-tool rigid bodies.')
    upper=bodies['upper_half'];lower_pad=bodies['lower_pad'];upper_pad=bodies['upper_pad']
    expected_lower={'upper_half','lower_pad'}|({'lower_grip_paddle'}if handles else set())
    expected_upper={'upper_pad'}|({'upper_grip_paddle'}if handles else set())
    if set(b.get('name') for b in lower.findall('body'))!=expected_lower or set(b.get('name')for b in upper.findall('body'))!=expected_upper:
        raise ValueError('Passive pad parent topology differs.')
    if lower_pad.findall('body') or upper_pad.findall('body'):raise ValueError('Unexpected descendants under pads.')
    body_pairs=[(name,bodies[name]) for name in ['lower_half','upper_half','lower_pad','upper_pad',*[h['body_name']for h in handles]]]
    if len(tree.findall('.//equality/*')) or wafer.find('freejoint') is None or lower.find('freejoint') is None:
        raise ValueError('Free-wafer/free-clamp topology is required.')
    compiler=tree.find('compiler')
    if compiler is None:raise ValueError('Missing shared compiler configuration.')
    inertials={name:_body_inertial(body,meshes,compiler) for name,body in [*body_pairs,('wafer',wafer)]}
    joints=[];joint_descriptions=[];spring_profiles=[]
    for parent,child in ((lower,upper),(lower,lower_pad),(upper,upper_pad)):
        text,description,profile=_joint_export(tree,parent,child,
            passive_pad_mass=inertials[child.get('name')]['mass_kg'] if child in (lower_pad,upper_pad) else None)
        if profile['mode'] not in ('constant_opening_spring','constant_passive_spring'):raise ValueError('Adaptive powered drive is forbidden.')
        joints.append(text);joint_descriptions.append(description);spring_profiles.append(profile)
    for h in handles:
        if abs(inertials[h['body_name']]['mass_kg']-h['mass_kg'])>1e-12:
            raise ValueError('Handle-paddle mass differs from declared profile.')
        text,description,profile=_joint_export(tree,bodies[h['parent_body']],bodies[h['body_name']],handle_profile=h)
        joints.append(text);joint_descriptions.append(description);spring_profiles.append(profile)
    collision_profile=(_tool_collision_profile(tree,bodies)
        if handles and metadata.get('full_workflow_no_wafer_side_cradle')is True else None)
    joint=''.join(joints);joint_description=joint_descriptions[0];actuation=spring_profiles[0]
    if collision_profile is not None:
        joint=joint.replace('bool physics:collisionEnabled = false','bool physics:collisionEnabled = true')
    if actuation['mode']!='constant_opening_spring' or any(x['mode']!='constant_passive_spring' for x in spring_profiles[1:]):
        raise ValueError('Expected one opening hinge spring and two constant pad springs.')
    root_transforms={'lower_half':np.eye(4)}
    for name,body in body_pairs[1:]:
        parent=next((h['parent_body']for h in handles if h['body_name']==name),'upper_half' if name=='upper_pad' else 'lower_half')
        local=np.eye(4);local[:3,:3]=_rotation(body);local[:3,3]=_vec(body,'pos',3,'0 0 0')
        root_transforms[name]=root_transforms[parent]@local
    rigid_schemas='"PhysicsRigidBodyAPI", "PhysicsMassAPI", "PhysxRigidBodyAPI"'
    models={};poses={};friction_default=.3;pad_friction=_effective_pad_friction(tree)
    def body_shapes(body):
        material='';shapes=''
        for geom in body.findall('geom'):
            name=geom.get('name');mu=pad_friction.get(name,float(_vec(geom,'friction',3,f'{friction_default} .0001 .0001')[0]))
            material+=_material(name+'_material',mu)
            shapes+=_shape(geom,meshes,'/Asset/'+name+'_material')+'\n'
        return material,shapes
    materials='';bodies=''
    for name,body in body_pairs:
        mats,shapes=body_shapes(body);materials+=mats
        q=_quat(root_transforms[name][:3,:3]);position=root_transforms[name][:3,3]
        xform='double3 xformOp:translate = '+_v(position)+'\nquatd xformOp:orient = '+_v([q[3],*q[:3]])+'\nuniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient"]\n'
        schemas=rigid_schemas;filters=''
        if collision_profile is not None:
            targets=[b for a,b in collision_profile['filtered_body_pairs']if a==name]
            if targets:
                schemas+=', "PhysicsFilteredPairsAPI"'
                filters='rel physics:filteredPairs = ['+', '.join('</Asset/'+b+'>'for b in targets)+']\n'
        bodies+=f'def Xform "{name}" (prepend apiSchemas = [{schemas}]) {{\n'+xform+_rigid_properties(inertials[name])+'\n'+filters+shapes+'}\n'
    clamp=_header(materials+bodies+joint,'"PhysicsArticulationRootAPI", "PhysxArticulationAPI"')
    clamp=clamp.replace('def Xform "Asset" (prepend apiSchemas = ["PhysicsArticulationRootAPI", "PhysxArticulationAPI"]) {',
        'def Xform "Asset" (prepend apiSchemas = ["PhysicsArticulationRootAPI", "PhysxArticulationAPI"]) {\nbool physxArticulation:enabledSelfCollisions = '
        +('true'if collision_profile is not None else'false')+'\nint physxArticulation:solverPositionIterationCount = 64\nint physxArticulation:solverVelocityIterationCount = 4')
    sources={'clamp':clamp}
    for name,body in [('clamp',lower),('wafer',wafer)]:
        q=_quat(_rotation(body));poses[name]=[*_vec(body,'pos',3,'0 0 0'),*q]
    mats,shapes=body_shapes(wafer)
    sources['wafer']=_header(_rigid_properties(inertials['wafer'])+'\n'+mats+shapes,rigid_schemas)
    for name in ('heater','pillar_0','pillar_1','pillar_2'):
        geom=world.find(f"geom[@name='{name}']")
        if geom is None:raise ValueError('Missing shared support '+name)
        q=_quat(_rotation(geom));poses[name]=[*_vec(geom,'pos',3,'0 0 0'),*q]
        # Body placement carries source rotation; its child geometry is identity.
        local=copy.deepcopy(geom);local.attrib.pop('euler',None);local.attrib.pop('quat',None)
        mu=_vec(geom,'friction',3,'.3 .0001 .0001')[0]
        sources[name]=_header(_rigid_properties(None,True)+'\n'+_material(name+'_material',mu)
            +_shape(local,meshes,'/Asset/'+name+'_material',zero_position=True),rigid_schemas)
    # The actual-YAM adapter adds this separate 25 mm pedestal below the local
    # fixture's z=0 plane. It is not added under the CPU carriage-only fixture.
    pedestal=ET.Element('geom',name='pedestal',type='cylinder',pos='0 0 0',size='.075 .0125',rgba='.4 .4 .4 1')
    poses['pedestal']=[0.,0.,-.0125,0.,0.,0.,1.]
    sources['pedestal']=_header(_rigid_properties(None,True)+'\n'+_material('pedestal_material',.3)
        +_shape(pedestal,meshes,'/Asset/pedestal_material'),rigid_schemas)
    if metadata.get('cycle_layout') and not metadata.get('full_workflow_no_wafer_side_cradle'):
        pivot=np.array(metadata['geometry']['fixture_pivot_world_m']);top=pivot[2]-.005
        cradle=ET.Element('geom',name='clamp_cradle',type='cylinder',
            pos=' '.join(str(v)for v in [*pivot[:2],(top-.025)/2+.0125]),
            size=f'.007 {(top+.025)/2:.12g}',rgba='.35 .38 .40 1')
        sources['pedestal']=_header(_rigid_properties(None,True)+'\n'+_material('pedestal_material',.3)
            +_shape(pedestal,meshes,'/Asset/pedestal_material')+'\n'+_shape(cradle,meshes,'/Asset/pedestal_material'),rigid_schemas)
        metadata['clamp_cradle_local']=dict(radius_m=.007,top_z_m=float(top),center_xy_m=pivot[:2].tolist())
    for name,source in sources.items():
        path=destination/(name+'.usda');path.write_text(source,encoding='utf-8');models[name]=str(path.resolve())
    metadata=copy.deepcopy(metadata)
    metadata['model_source_sha256']=hashlib.sha256(Path(model.__file__).read_bytes()).hexdigest()
    if dependencies!={name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in dependency_paths.items()}:
        raise RuntimeError('Mechanism dependencies changed during export; discard these provisional assets.')
    metadata['model_dependency_sha256']=dependencies
    metadata['mechanism_source_module']=model_module
    metadata['usd_export']=dict(source_mjcf_sha256=hashlib.sha256(xml.encode()).hexdigest(),
        body_inertials=inertials,shared_geometry='Exact authored MJCF primitive parameters and prism vertices; analytic compound mass properties.',
        joint=joint_description,actuation=actuation,joints=joint_descriptions,passive_springs=spring_profiles,
        contact_link_names=['lower_pad','upper_pad'],lever_link_names=['lower_half','upper_half'],
        physx_contact_offset_m=.0001,physx_rest_offset_m=0.,physx_friction_combine='max',
        effective_wafer_pad_sliding_friction=pad_friction,
        added_yam_pedestal=dict(radius_m=.075,height_m=.025,center_local_m=[0.,0.,-.0125]),
        physx_position_iterations=64,physx_velocity_iterations=4,
        limitations=['Schema/geometry export is not a PhysX physical grasp result.',
            'MuJoCo and PhysX solvers/contact compliance differ despite shared geometry and inertia.',
            'Explicit MuJoCo wafer-pad sliding friction is mapped to pad materials; other pad contacts and rolling/torsional friction are not equivalent.',
            'Two explicit SI pad springs use constant linear force drives. Native8N caps exceed the7N position/speed rectangle bound; the separate prospective force guard remains4N. No accepted state reaches native clipping; this is not engine equivalence.',
            ('Full-tool internal collision eligibility matches the local pair map; connected bodies remain filtered. No wafer attachment or world joint is authored.'
             if collision_profile is not None else'Legacy articulation-wide self collision is disabled; no wafer attachment or world joint is authored.')])
    if collision_profile is not None:metadata['usd_export']['tool_self_collision']=collision_profile
    result=dict(models=models,metadata=metadata,initial_local_poses={k:[float(v) for v in p] for k,p in poses.items()},
                tool_link_names=[name for name,body in body_pairs],lever_link_names=['lower_half','upper_half'],
                contact_link_names=['lower_pad','upper_pad'],mechanism_joint_name=joint_description['name'],
                passive_joint_names=[x['joint_name'] for x in spring_profiles])
    (destination/'export_manifest.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result


def verify_passive_spring_specifications(exported):
    """Validate exact SI spring topology without importing a physics engine."""
    usd=exported['metadata']['usd_export'];springs=usd['passive_springs'];joints=usd['joints']
    if exported['metadata'].get('physics_dt_s')!=PHYSICS_DT_S:
        raise ValueError('Export metadata cadence differs from shared settings.')
    names=['pliers_hinge','lower_pad_compression','upper_pad_compression']
    handles=exported['metadata'].get('handle_paddles',[])
    names += [h['joint_name']for h in handles]
    if [x['joint_name'] for x in springs]!=names or [x['name'] for x in joints]!=names:
        raise ValueError('Expected the opening hinge and two explicitly named passive pad springs.')
    if exported['tool_link_names']!=['lower_half','upper_half','lower_pad','upper_pad',*[h['body_name']for h in handles]]:
        raise ValueError('Unexpected free-tool body order.')
    if exported['lever_link_names']!=['lower_half','upper_half'] or exported['contact_link_names']!=['lower_pad','upper_pad']:
        raise ValueError('Lever and contact-pad identities are ambiguous.')
    hinge=springs[0]
    expected_hinge=dict(joint_name='pliers_hinge',stiffness_Nm_rad=.45,damping_Nms_rad=.2,
        rest_rad=-.2,limits_rad=[-.04,HINGE_CLOSED_LIMIT_RAD],force_cap_Nm=.3,torque_guard_Nm=.3,velocity_limit_rad_s=10.)
    if exported['metadata'].get('passive_hinge')!=expected_hinge:
        raise ValueError('Passive hinge diagnostic metadata differs from the reviewed SI profile.')
    if joints[0]['limits']!=[-.04,HINGE_CLOSED_LIMIT_RAD] or joints[0]['parent']!='lower_half' or joints[0]['child']!='upper_half':
        raise ValueError('Opening hinge topology/travel differs from the reviewed model.')
    for key,value in dict(mode='constant_opening_spring',joint_type='hinge',stiffness=.45,damping=.2,constant_target=-.2,effort_limit=.3,velocity_limit=10.).items():
        matches=math.isclose(hinge[key],value,rel_tol=0,abs_tol=1e-12) if isinstance(value,float) else hinge[key]==value
        if not matches:
            raise ValueError('Opening hinge source profile differs: '+key)
    for index,side in enumerate(('lower','upper'),1):
        profile=springs[index];joint=joints[index];mass=usd['body_inertials'][side+'_pad']['mass_kg']
        expected=dict(mode='constant_passive_spring',joint_type='slide',stiffness=PAD_STIFFNESS_N_M,constant_target=0.,effort_limit=8.,velocity_limit=1.,diagnostic_velocity_limit=.05,diagnostic_force_limit=4.,position_unit='m',effort_unit='N')
        for key,value in expected.items():
            if profile[key]!=value:raise ValueError('Passive pad SI source profile differs: '+key)
        if not math.isclose(profile['mass_kg'],mass,rel_tol=1e-12,abs_tol=1e-15) or not math.isclose(mass,PAD_MASS_KG,rel_tol=0,abs_tol=1e-12) or profile['damping']!=PAD_DAMPING_N_S_M:
            raise ValueError('Passive pad mass or damping differs from shared settings.')
        if joint['parent']!=side+'_half' or joint['child']!=side+'_pad' or joint['limits']!=[0.,PAD_TRAVEL_M] or joint['axis_local']!=[0.,0.,-1. if side=='lower' else 1.]:
            raise ValueError('Passive pad compression frame or travel differs.')
        if PAD_STIFFNESS_N_M*PAD_TRAVEL_M+profile['damping']*.05>=8.:raise ValueError('Passive pad drive can saturate inside the position/speed rectangle.')
        if exported['metadata']['pad_compliance'][index-1].get('force_guard_N')!=4.:
            raise ValueError('Prospective4N pad-force guard differs from the native8N cap.')
    if handles and ([h['body_name']for h in handles]!=['lower_grip_paddle','upper_grip_paddle']or
                    [h['joint_name']for h in handles]!=['lower_grip_paddle_pitch','upper_grip_paddle_pitch']):
        raise ValueError('Unexpected passive handle-paddle profiles.')
    for index,h in enumerate(handles,3):
        profile=springs[index];joint=joints[index]
        expected=dict(mode='constant_passive_spring',joint_type='hinge',stiffness=h['stiffness_Nm_rad'],damping=h['damping_Nms_rad'],
            constant_target=h['rest_rad'],effort_limit=h['force_cap_Nm'],velocity_limit=h['velocity_limit_rad_s'],
            diagnostic_velocity_limit=h['velocity_guard_rad_s'],diagnostic_force_limit=h['torque_guard_Nm'])
        if 'operating_torque_guard_Nm' in h:
            if h['operating_torque_guard_Nm']!=.060 or not h['operating_torque_guard_Nm']<h['force_cap_Nm']:
                raise ValueError('Accepted paddle torque differs from 60 mNm or reaches the numerical drive cap.')
            expected['operating_force_limit']=h['operating_torque_guard_Nm']
        if any(profile.get(k)!=v for k,v in expected.items()):raise ValueError('Handle-paddle spring metadata differs.')
        if joint['parent']!=h['parent_body']or joint['child']!=h['body_name']or joint['limits']!=h['limits_rad']or joint['axis_local']!=h['axis_local']:
            raise ValueError('Handle-paddle joint frame or travel differs.')
        if 'joint_anchor_local_m' in h and not np.allclose(joint['local_pos1_m'],h['joint_anchor_local_m'],atol=1e-12,rtol=0):
            raise ValueError('Handle-paddle local anchor differs from the mechanical profile.')
        if abs(usd['body_inertials'][h['body_name']]['mass_kg']-h['mass_kg'])>1e-12:raise ValueError('Handle-paddle mass differs.')
    for profile in springs:
        if profile.get('authored_usd_drive') is not True or profile.get('usd_drive')!=usd_drive_specification(profile):
            raise ValueError('Missing or inconsistent explicit passive force DriveAPI.')
    return dict(status='FIVE_CONSTANT_PASSIVE_SI_SPRINGS_VERIFIED'if handles else'THREE_CONSTANT_PASSIVE_SI_SPRINGS_VERIFIED',joint_names=names,
        target_positions=[x['constant_target'] for x in springs],
        scope='Authored SI profile only; native readback and physical behavior require their own checks.')


def audit_assets(exported):
    """Optional CPU-only schema audit. Run before/after Kit in another process."""
    import sys
    for parent in list(Path(__file__).resolve().parents)[:3]:
        vendor=parent/'wafer-design'/'deps'
        if vendor.is_dir():
            if str(vendor) not in sys.path:sys.path.insert(0,str(vendor))
            break
    from pxr import Usd,UsdPhysics
    verify_passive_spring_specifications(exported)
    summaries={}
    for name,path in exported['models'].items():
        stage=Usd.Stage.Open(path)
        if stage is None:raise ValueError('Cannot open exported '+name)
        bodies=[p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
        joints=[p for p in stage.Traverse() if p.IsA(UsdPhysics.Joint)]
        colliders=[p for p in stage.Traverse() if p.HasAPI(UsdPhysics.CollisionAPI)]
        if name=='clamp':
            specifications=exported['metadata']['usd_export']['joints']
            springs=exported['metadata']['usd_export']['passive_springs']
            if len(bodies)!=len(exported['tool_link_names']) or len(joints)!=len(springs):raise ValueError('Unexpected tool body or passive joint count.')
            if sorted(p.GetName() for p in bodies)!=sorted(exported['tool_link_names']):raise ValueError('Unexpected tool body names.')
            if not stage.GetPrimAtPath('/Asset').HasAPI(UsdPhysics.ArticulationRootAPI):raise ValueError('Missing floating articulation root.')
            if exported['metadata']['usd_export'].get('tool_self_collision')is not None:
                verify_tool_collision_prim(stage.GetPrimAtPath('/Asset'),exported['metadata']['usd_export']['tool_self_collision'])
            by_name={p.GetName():p for p in joints}
            if set(by_name)!={x['name'] for x in specifications}:raise ValueError('Unexpected tool joint names.')
            for specification,spring in zip(specifications,springs,strict=True):
                prim=by_name[specification['name']]
                joint_class=UsdPhysics.RevoluteJoint if specification['type']=='hinge' else UsdPhysics.PrismaticJoint
                if not prim.IsA(joint_class):raise ValueError('Wrong joint type.')
                joint=joint_class(prim)
                if [str(t) for t in joint.GetBody0Rel().GetTargets()]!=['/Asset/'+specification['parent']] or [str(t) for t in joint.GetBody1Rel().GetTargets()]!=['/Asset/'+specification['child']]:
                    raise ValueError('Wrong passive joint body references.')
                verify_drive_prim(prim,spring)
                if 'local_pos0_m' in specification:verify_joint_frame_prim(prim,specification)
        elif len(bodies)!=1 or joints:raise ValueError('Unexpected support/wafer joint topology.')
        expected_kinematic=name not in ('clamp','wafer')
        for body in bodies:
            api=UsdPhysics.RigidBodyAPI(body)
            if api.GetKinematicEnabledAttr().Get()!=expected_kinematic:raise ValueError('Incorrect dynamic/kinematic body flag.')
            if not expected_kinematic and (UsdPhysics.MassAPI(body).GetMassAttr().Get() or 0)<=0:raise ValueError('Missing dynamic mass.')
        summaries[name]=dict(rigid_body_count=len(bodies),joint_count=len(joints),collider_count=len(colliders),
                             kinematic=expected_kinematic,sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest())
    return dict(result='USD_SCHEMA_AND_TOPOLOGY_CHECKED',assets=summaries,scope='No physics engine was run.')
