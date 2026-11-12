"""Export a free passive tool from its compiled MJCF, without running physics."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import math
import re
import sys

import mujoco
import numpy as np

try:
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'wafer-design/deps'))
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def quaternion(rotation):
    q = np.empty(4)
    mujoco.mju_mat2Quat(q, np.asarray(rotation).ravel())
    return q


def rotation(q):
    result = np.empty(9)
    mujoco.mju_quat2Mat(result, np.asarray(q))
    return result.reshape(3, 3)


def quat(q, double=False):
    q = np.asarray(q)
    return (Gf.Quatd if double else Gf.Quatf)(float(q[0]), *(float(x) for x in q[1:]))


def vector(x):
    return Gf.Vec3f(*(float(v) for v in x))


def name(model, kind, index):
    value = mujoco.mj_id2name(model, kind, index)
    if not value or not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', value):
        raise ValueError('All exported bodies, joints and colliders need valid unique names.')
    return value


def transform(prim, position, q):
    xform = UsdGeom.Xformable(prim)
    xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*map(float, position)))
    xform.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(quat(q, True))


def close(actual, expected, label, atol=2e-7, rtol=2e-6):
    if not np.allclose(np.asarray(actual), np.asarray(expected), atol=atol, rtol=rtol):
        raise ValueError('USD/MJCF mismatch: ' + label)


def joint_basis(axis):
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    helper = np.array([1., 0., 0.]) if abs(axis[1]) > .9 else np.array([0., 1., 0.])
    x = np.cross(helper, axis)
    x /= np.linalg.norm(x)
    return np.column_stack((x, np.cross(axis, x), axis))


def export_geometry(stage, model, g, parent_path, zero_position=False):
    gn = name(model, mujoco.mjtObj.mjOBJ_GEOM, g)
    gp = parent_path + '/' + gn
    kind, size = int(model.geom_type[g]), model.geom_size[g]
    mesh_data = None
    if kind == mujoco.mjtGeom.mjGEOM_BOX:
        shape = UsdGeom.Cube.Define(stage, gp); shape.CreateSizeAttr(1.)
    elif kind in (mujoco.mjtGeom.mjGEOM_CYLINDER, mujoco.mjtGeom.mjGEOM_CAPSULE):
        shape = (UsdGeom.Cylinder if kind == mujoco.mjtGeom.mjGEOM_CYLINDER else UsdGeom.Capsule).Define(stage, gp)
        shape.CreateRadiusAttr(float(size[0])); shape.CreateHeightAttr(float(2 * size[1])); shape.CreateAxisAttr('Z')
    elif kind == mujoco.mjtGeom.mjGEOM_SPHERE:
        shape = UsdGeom.Sphere.Define(stage, gp); shape.CreateRadiusAttr(float(size[0]))
    elif kind == mujoco.mjtGeom.mjGEOM_MESH:
        mid = model.geom_dataid[g]; start, count = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
        vertices = model.mesh_vert[start:start + count].copy()
        start, count = model.mesh_faceadr[mid], model.mesh_facenum[mid]
        faces = model.mesh_face[start:start + count].copy()
        shape = UsdGeom.Mesh.Define(stage, gp)
        shape.CreatePointsAttr([vector(v) for v in vertices])
        shape.CreateFaceVertexCountsAttr([3] * len(faces)); shape.CreateFaceVertexIndicesAttr(faces.ravel().tolist())
        shape.CreateSubdivisionSchemeAttr('none')
        UsdPhysics.MeshCollisionAPI.Apply(shape.GetPrim()).CreateApproximationAttr('convexHull')
        mesh_data = dict(vertices_m=vertices.tolist(), faces=faces.tolist())
    else:
        raise ValueError('Unsupported collider: ' + gn)
    prim = shape.GetPrim()
    position = np.zeros(3) if zero_position else model.geom_pos[g]
    orientation = np.array([1., 0., 0., 0.]) if zero_position else model.geom_quat[g]
    transform(prim, position, orientation)
    if kind == mujoco.mjtGeom.mjGEOM_BOX:
        UsdGeom.Xformable(prim).AddScaleOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*map(float, 2 * size)))
    shape.CreateDisplayColorAttr([vector(model.geom_rgba[g, :3])])
    shape.CreateDisplayOpacityAttr([float(model.geom_rgba[g, 3])])
    UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(bool(model.geom_contype[g] or model.geom_conaffinity[g]))
    prim.AddAppliedSchema('PhysxCollisionAPI')
    prim.CreateAttribute('physxCollision:contactOffset', Sdf.ValueTypeNames.Float).Set(.0001)
    prim.CreateAttribute('physxCollision:restOffset', Sdf.ValueTypeNames.Float).Set(0.)
    material_path = '/Asset/material_' + gn
    material_prim = stage.DefinePrim(material_path, 'Material')
    material = UsdPhysics.MaterialAPI.Apply(material_prim)
    material.CreateStaticFrictionAttr(float(model.geom_friction[g, 0]))
    material.CreateDynamicFrictionAttr(float(model.geom_friction[g, 0]))
    material_prim.AddAppliedSchema('PhysxMaterialAPI')
    material_prim.CreateAttribute('physxMaterial:frictionCombineMode', Sdf.ValueTypeNames.Token).Set('max')
    prim.CreateRelationship('material:binding:physics').SetTargets([material_path]); prim.AddAppliedSchema('MaterialBindingAPI')
    return dict(name=gn, path=gp, kind=kind, half_sizes_m=size.tolist(), position_m=position.tolist(),
        quaternion_wxyz=orientation.tolist(), friction=model.geom_friction[g].tolist(), material_path=material_path,
        collision_enabled=bool(model.geom_contype[g] or model.geom_conaffinity[g]), mesh=mesh_data,
        source_contact_margin_m=float(model.geom_margin[g]), native_contact_offset_m=.0001, native_rest_offset_m=0.)


def collision_pairs(model, bodies):
    explicit = {tuple(sorted((int(a), int(b)))) for a, b in zip(model.pair_geom1, model.pair_geom2)}
    excluded = {int(x) for x in model.exclude_signature}
    filtering = not (int(model.opt.disableflags) & int(mujoco.mjtDisableBit.mjDSBL_FILTERPARENT))
    pairs = []
    for index, a in enumerate(bodies):
        for b in bodies[index + 1:]:
            connected = model.body_parentid[a] == b or model.body_parentid[b] == a
            eligibility = set()
            ga = range(model.body_geomadr[a], model.body_geomadr[a] + model.body_geomnum[a])
            gb = range(model.body_geomadr[b], model.body_geomadr[b] + model.body_geomnum[b])
            for x in ga:
                for y in gb:
                    eligible = tuple(sorted((x, y))) in explicit or (
                        ((min(a, b) << 16) + max(a, b)) not in excluded
                        and not (filtering and connected)
                        and bool((model.geom_contype[x] & model.geom_conaffinity[y])
                                 or (model.geom_contype[y] & model.geom_conaffinity[x])))
                    eligibility.add(bool(eligible))
            if len(eligibility) != 1:
                raise ValueError('Mixed or empty collider-pair filtering requires an explicit exporter extension.')
            pairs.append((a, b, eligibility.pop()))
    return pairs


def export(scene, output, root_name, *, asset_name='edge_pick', existing=False, write_report=True):
    scene, output = Path(scene).resolve(), Path(output).resolve()
    if output.exists() and not existing:
        raise FileExistsError('Preserve existing exports: ' + str(output))
    asset = output / (asset_name + '.usda')
    if asset.exists():raise FileExistsError(asset)
    before = sha(scene)
    model = mujoco.MjModel.from_xml_path(str(scene))
    if model.neq or mujoco.get_mjcb_contactfilter() is not None:
        raise ValueError('Equality constraints or contact callbacks require explicit transfer support.')
    if int(model.opt.disableflags) & (int(mujoco.mjtDisableBit.mjDSBL_CONTACT) | int(mujoco.mjtDisableBit.mjDSBL_CONSTRAINT)):
        raise ValueError('Contacts and constraints must be enabled.')
    root = int(model.body(root_name).id)
    bodies = [root]
    for b in range(root + 1, model.nbody):
        if int(model.body_parentid[b]) in bodies:
            bodies.append(b)
    if int(model.body_jntnum[root]) != 1 or model.jnt_type[model.body_jntadr[root]] != mujoco.mjtJoint.mjJNT_FREE:
        raise ValueError('The tool root must be a free dynamic body.')
    tool_joints = {j for b in bodies for j in range(model.body_jntadr[b], model.body_jntadr[b] + model.body_jntnum[b])}
    for kind, target in zip(model.actuator_trntype, model.actuator_trnid[:, 0]):
        if int(kind) != int(mujoco.mjtTrn.mjTRN_JOINT):
            raise ValueError('Only ordinary fixture joint actuators are supported.')
        if int(target) in tool_joints:
            raise ValueError('The tool root, jaw and trigger must not have commanded actuators.')
    if any(model.body_gravcomp[b] != 0 for b in bodies):
        raise ValueError('Tool gravity compensation is not transferable as a passive asset.')
    body_names = {b: name(model, mujoco.mjtObj.mjOBJ_BODY, b) for b in bodies}
    joints = []
    for b in bodies[1:]:
        if int(model.body_jntnum[b]) != 1 or model.body_mocapid[b] != -1:
            raise ValueError('Each moving child requires one ordinary passive joint.')
        j = int(model.body_jntadr[b])
        if model.jnt_type[j] not in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
            raise ValueError('Only revolute or prismatic passive joints are supported.')
        if model.dof_armature[model.jnt_dofadr[j]] != 0 or model.dof_frictionloss[model.jnt_dofadr[j]] != 0:
            raise ValueError('Nonzero joint armature/friction loss needs a separate native model.')
        if model.jnt_stiffness[j] <= 0 or not model.jnt_limited[j]:
            raise ValueError('An explicitly limited positive passive spring is required.')
        joints.append(j)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    origin = data.xpos[root].copy()
    root_rotation = data.xmat[root].reshape(3, 3).copy()
    output.mkdir(parents=True, exist_ok=existing)
    stage = Usd.Stage.CreateNew(str(asset))
    top = UsdGeom.Xform.Define(stage, '/Asset').GetPrim()
    stage.SetDefaultPrim(top)
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    if joints:
        UsdPhysics.ArticulationRootAPI.Apply(top)
        top.AddAppliedSchema('PhysxArticulationAPI')
        top.CreateAttribute('physxArticulation:enabledSelfCollisions', Sdf.ValueTypeNames.Bool).Set(True)
    body_records, geom_records, joint_records = [], [], []
    paths = {b: '/Asset/' + body_names[b] for b in bodies}
    for b in bodies:
        prim = UsdGeom.Xform.Define(stage, paths[b]).GetPrim()
        p = root_rotation.T @ (data.xpos[b] - origin)
        r = root_rotation.T @ data.xmat[b].reshape(3, 3)
        transform(prim, p, quaternion(r))
        body = UsdPhysics.RigidBodyAPI.Apply(prim)
        body.CreateRigidBodyEnabledAttr(True)
        body.CreateKinematicEnabledAttr(False)
        mass = UsdPhysics.MassAPI.Apply(prim)
        mass.CreateMassAttr(float(model.body_mass[b]))
        mass.CreateCenterOfMassAttr(vector(model.body_ipos[b]))
        mass.CreateDiagonalInertiaAttr(vector(model.body_inertia[b]))
        mass.CreatePrincipalAxesAttr(quat(model.body_iquat[b]))
        body_records.append(dict(name=body_names[b], path=paths[b], position_m=p.tolist(),
            quaternion_wxyz=quaternion(r).tolist(), mass_kg=float(model.body_mass[b]),
            com_m=model.body_ipos[b].tolist(), inertia_kg_m2=model.body_inertia[b].tolist(),
            inertia_quaternion_wxyz=model.body_iquat[b].tolist()))
        for g in range(model.body_geomadr[b], model.body_geomadr[b] + model.body_geomnum[b]):
            geom_records.append(export_geometry(stage, model, g, paths[b]))
    pairs = collision_pairs(model, bodies)
    for a, b, enabled in pairs:
        if not enabled:
            UsdPhysics.FilteredPairsAPI.Apply(stage.GetPrimAtPath(paths[a])).CreateFilteredPairsRel().AddTarget(paths[b])
    for j in joints:
        child = int(model.jnt_bodyid[j]); parent = int(model.body_parentid[child])
        jn = name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        angular = model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE
        joint = (UsdPhysics.RevoluteJoint if angular else UsdPhysics.PrismaticJoint).Define(stage, '/Asset/' + jn)
        joint.CreateBody0Rel().SetTargets([paths[parent]]); joint.CreateBody1Rel().SetTargets([paths[child]])
        rp = data.xmat[parent].reshape(3, 3); rc = data.xmat[child].reshape(3, 3)
        basis = joint_basis(model.jnt_axis[j])
        anchor = data.xpos[child] + rc @ model.jnt_pos[j]
        p0 = rp.T @ (anchor - data.xpos[parent]); p1 = model.jnt_pos[j]
        q0, q1 = quaternion(rp.T @ rc @ basis), quaternion(basis)
        joint.CreateLocalPos0Attr(vector(p0)); joint.CreateLocalPos1Attr(vector(p1))
        joint.CreateLocalRot0Attr(quat(q0)); joint.CreateLocalRot1Attr(quat(q1))
        joint.CreateAxisAttr('Z'); joint.CreateCollisionEnabledAttr(True)
        offset = float(model.qpos0[model.jnt_qposadr[j]])
        limits = model.jnt_range[j] - offset
        scale = math.pi / 180 if angular else 1.
        joint.CreateLowerLimitAttr(float(limits[0] / scale)); joint.CreateUpperLimitAttr(float(limits[1] / scale))
        k, c = float(model.jnt_stiffness[j]), float(model.dof_damping[model.jnt_dofadr[j]])
        rest = float(model.qpos_spring[model.jnt_qposadr[j]] - offset)
        drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), 'angular' if angular else 'linear')
        drive.CreateTypeAttr('force'); drive.CreateStiffnessAttr(k * scale); drive.CreateDampingAttr(c * scale)
        drive.CreateTargetPositionAttr(rest / scale); drive.CreateTargetVelocityAttr(0.)
        drive.CreateMaxForceAttr(float('inf'))
        joint_records.append(dict(name=jn, path=str(joint.GetPath()), angular=angular,
            parent=paths[parent], child=paths[child], local_pos0_m=p0.tolist(), local_pos1_m=p1.tolist(),
            local_rot0_wxyz=q0.tolist(), local_rot1_wxyz=q1.tolist(),
            mjcf_reference_offset=offset, limits=limits.tolist(), stiffness_si=k, damping_si=c,
            rest_si=rest, native_gain_scale=scale, force_cap='unlimited, matching the passive MJCF spring'))
    stage.GetRootLayer().Save()
    stage = None
    verify(asset, body_records, geom_records, joint_records, pairs, paths)
    if sha(scene) != before:
        raise ValueError('Source MJCF changed during export.')
    report = dict(result='GEOMETRY_MASS_AND_PASSIVE_SPRING_PARITY_VERIFIED', source_scene_sha256=before,
        exporter_sha256=sha(__file__), asset_file=asset.name, asset_sha256=sha(asset), mujoco_version=mujoco.__version__,
        usd_version=list(Usd.GetVersion()), body_count=len(bodies), collider_count=len(geom_records),
        joint_count=len(joints), initial_root_pose_world_wxyz=[*origin.tolist(), *quaternion(root_rotation).tolist()],
        bodies=body_records, colliders=geom_records, joints=joint_records,
        collision_pairs=[dict(a=body_names[a], b=body_names[b], enabled=enabled) for a, b, enabled in pairs],
        scope='Compiled MJCF geometry, inertias and passive spring parameters verified from reopened USD. No Isaac physics run or contact-response equivalence claim.',
        transfer_limits=['MuJoCo contact solref/solimp and PhysX contact solvers are not equivalent.',
            'Native contact/rest offsets are declared 0.1 mm/0; they are not an equivalence to the source MJCF contact margin.',
            'Sliding friction coefficients are copied; torsional/rolling friction and inter-asset contact pairs require native integration review.',
            'Robot, stand, wafer and initial root placement are not included in this isolated tool asset.'])
    if write_report:(output / 'parity.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    return report


def verify(asset, bodies, geoms, joints, pairs, paths):
    stage = Usd.Stage.Open(str(asset))
    if UsdGeom.GetStageMetersPerUnit(stage) != 1. or UsdGeom.GetStageUpAxis(stage) != 'Z':
        raise ValueError('Invalid USD coordinate units.')
    for b in bodies:
        prim = stage.GetPrimAtPath(b['path']); api = UsdPhysics.MassAPI(prim)
        if UsdPhysics.RigidBodyAPI(prim).GetKinematicEnabledAttr().Get():raise ValueError('Tool body became kinematic.')
        close(prim.GetAttribute('xformOp:translate').Get(), b['position_m'], b['name'] + ' pose')
        q = prim.GetAttribute('xformOp:orient').Get()
        close(rotation([q.GetReal(), *q.GetImaginary()]), rotation(b['quaternion_wxyz']), b['name'] + ' orientation')
        close(api.GetMassAttr().Get(), b['mass_kg'], b['name'] + ' mass', atol=1e-10)
        close(api.GetCenterOfMassAttr().Get(), b['com_m'], b['name'] + ' COM')
        close(api.GetDiagonalInertiaAttr().Get(), b['inertia_kg_m2'], b['name'] + ' inertia', atol=1e-13)
        q = api.GetPrincipalAxesAttr().Get()
        close(rotation([q.GetReal(), *q.GetImaginary()]), rotation(b['inertia_quaternion_wxyz']), b['name'] + ' inertia axes')
    for g in geoms:
        prim = stage.GetPrimAtPath(g['path']); size = np.asarray(g['half_sizes_m'])
        close(prim.GetAttribute('xformOp:translate').Get(), g['position_m'], g['name'] + ' geometry position')
        q = prim.GetAttribute('xformOp:orient').Get()
        close(rotation([q.GetReal(), *q.GetImaginary()]), rotation(g['quaternion_wxyz']), g['name'] + ' geometry orientation')
        if g['kind'] == mujoco.mjtGeom.mjGEOM_BOX:
            close(prim.GetAttribute('xformOp:scale').Get(), 2 * size, g['name'] + ' size')
        elif g['kind'] == mujoco.mjtGeom.mjGEOM_MESH:
            shape = UsdGeom.Mesh(prim)
            close(shape.GetPointsAttr().Get(), g['mesh']['vertices_m'], g['name'] + ' mesh vertices')
            close(np.asarray(shape.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3), g['mesh']['faces'], g['name'] + ' mesh faces', atol=0, rtol=0)
            if UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get() != 'convexHull':raise ValueError('Mesh collision approximation differs.')
        else:
            close(prim.GetAttribute('radius').Get(), size[0], g['name'] + ' radius')
            if g['kind'] != mujoco.mjtGeom.mjGEOM_SPHERE:close(prim.GetAttribute('height').Get(), 2 * size[1], g['name'] + ' height')
        if UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() != g['collision_enabled']:raise ValueError('Collider state differs.')
        material = UsdPhysics.MaterialAPI(stage.GetPrimAtPath(g['material_path']))
        close(material.GetStaticFrictionAttr().Get(), g['friction'][0], 'friction')
        close(material.GetDynamicFrictionAttr().Get(), g['friction'][0], 'friction')
        close(prim.GetAttribute('physxCollision:contactOffset').Get(), g['native_contact_offset_m'], 'native contact offset')
        close(prim.GetAttribute('physxCollision:restOffset').Get(), g['native_rest_offset_m'], 'native rest offset')
    for j in joints:
        api = (UsdPhysics.RevoluteJoint if j['angular'] else UsdPhysics.PrismaticJoint)(stage.GetPrimAtPath(j['path']))
        if list(map(str, api.GetBody0Rel().GetTargets())) != [j['parent']] or list(map(str, api.GetBody1Rel().GetTargets())) != [j['child']]:raise ValueError('Joint body binding differs.')
        close(api.GetLocalPos0Attr().Get(), j['local_pos0_m'], 'joint parent anchor')
        close(api.GetLocalPos1Attr().Get(), j['local_pos1_m'], 'joint child anchor')
        for attr, key in ((api.GetLocalRot0Attr(), 'local_rot0_wxyz'), (api.GetLocalRot1Attr(), 'local_rot1_wxyz')):
            q = attr.Get(); close(rotation([q.GetReal(), *q.GetImaginary()]), rotation(j[key]), 'joint axes')
        scale = j['native_gain_scale']
        close([api.GetLowerLimitAttr().Get() * scale, api.GetUpperLimitAttr().Get() * scale], j['limits'], 'joint limits')
        drive = UsdPhysics.DriveAPI(api.GetPrim(), 'angular' if j['angular'] else 'linear')
        close(drive.GetStiffnessAttr().Get() / scale, j['stiffness_si'], 'spring stiffness')
        close(drive.GetDampingAttr().Get() / scale, j['damping_si'], 'spring damping')
        close(drive.GetTargetPositionAttr().Get() * scale, j['rest_si'], 'spring rest')
        if drive.GetTypeAttr().Get() != 'force' or not math.isinf(drive.GetMaxForceAttr().Get()):raise ValueError('Passive spring force semantics differ.')
    for a, b, enabled in pairs:
        prim = stage.GetPrimAtPath(paths[a])
        targets = UsdPhysics.FilteredPairsAPI(prim).GetFilteredPairsRel().GetTargets()
        if (Sdf.Path(paths[b]) in targets) == enabled:raise ValueError('Internal contact filtering differs.')


def export_scene(scene, output, root_name, with_supports=False):
    report = export(scene, output, root_name, write_report=False)
    output = Path(output).resolve()
    if with_supports:
        wafer = export(scene, output, 'wafer', asset_name='wafer', existing=True, write_report=False)
        model = mujoco.MjModel.from_xml_path(str(Path(scene).resolve()))
        asset = output / 'supports.usda'
        if asset.exists():raise FileExistsError(asset)
        stage = Usd.Stage.CreateNew(str(asset))
        top = UsdGeom.Xform.Define(stage, '/Asset').GetPrim()
        stage.SetDefaultPrim(top); UsdGeom.SetStageMetersPerUnit(stage, 1.); UsdGeom.SetStageUpAxis(stage, 'Z')
        supports = []
        for g in range(model.ngeom):
            if model.geom_bodyid[g] != 0 or model.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE:
                continue
            gn = name(model, mujoco.mjtObj.mjOBJ_GEOM, g)
            path = '/Asset/' + gn
            prim = UsdGeom.Xform.Define(stage, path).GetPrim()
            transform(prim, model.geom_pos[g], model.geom_quat[g])
            body = UsdPhysics.RigidBodyAPI.Apply(prim)
            body.CreateRigidBodyEnabledAttr(True); body.CreateKinematicEnabledAttr(True)
            record = export_geometry(stage, model, g, path, zero_position=True)
            supports.append(dict(name=gn, path=path, collider=record,
                pose_world_wxyz=[*model.geom_pos[g].tolist(), *model.geom_quat[g].tolist()]))
        if not {'pillar_0', 'pillar_1', 'pillar_2'}.issubset(s['name'] for s in supports):
            raise ValueError('The original three support pillars are required.')
        stage.GetRootLayer().Save(); stage = None
        verify(asset, [], [s['collider'] for s in supports], [], [], {})
        stage = Usd.Stage.Open(str(asset))
        for support in supports:
            prim = stage.GetPrimAtPath(support['path'])
            if UsdPhysics.RigidBodyAPI(prim).GetKinematicEnabledAttr().Get() is not True:
                raise ValueError('A fixed support became dynamic.')
            expected = support['pose_world_wxyz']
            close(prim.GetAttribute('xformOp:translate').Get(), expected[:3], support['name'] + ' support position')
            q = prim.GetAttribute('xformOp:orient').Get()
            close(rotation([q.GetReal(), *q.GetImaginary()]), rotation(expected[3:]), support['name'] + ' support orientation')
        if sha(scene) != report['source_scene_sha256'] or wafer['source_scene_sha256'] != report['source_scene_sha256']:
            raise ValueError('Fixture source changed during the asset export.')
        report.update(wafer_asset=wafer, support_asset=dict(asset_file=asset.name, asset_sha256=sha(asset),
            supports=supports, initial_root_pose_world_wxyz=[0., 0., 0., 1., 0., 0., 0.]),
            excluded_fixture='The plane floor and actuated carriage/fingers are deliberately not exported.')
    generator = Path(scene).resolve().parent / 'edge_pick.py'
    if generator.exists():
        module_spec = importlib.util.spec_from_file_location('edge_pick_export_source', generator)
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        generated, _ = module.build_model_xml()
        generated_sha = hashlib.sha256(generated.encode()).hexdigest()
        batch_path = generator.parent / 'batch_report.json'
        batch = json.loads(batch_path.read_text())
        successes = sum(bool(t['success']) for t in batch['trials'])
        if (generated_sha != sha(scene) or batch['nominal_scene_sha256'] != sha(scene)
                or batch['source_sha256'] != sha(generator)
                or not batch['source_unchanged_during_batch']
                or batch['count'] != len(batch['trials']) or batch['count'] != 20
                or batch['successes'] != successes or successes < 16):
            raise ValueError('Local mechanism qualification does not bind this export.')
        report['source_generator_verification'] = dict(file='../edge_pick.py', sha256=sha(generator),
            generated_nominal_scene_sha256=generated_sha, matches_saved_scene=True)
        report['local_fixture_qualification'] = dict(file='../batch_report.json', sha256=sha(batch_path),
            count=batch['count'], successes=successes, required_successes=16, passed=True,
            scope='Recorded MuJoCo fixture batch only; no Isaac or full YAM validation.')
    (output / 'parity.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('scene', type=Path); parser.add_argument('output', type=Path)
    parser.add_argument('--body', required=True, help='Free tool root body in the MJCF.')
    parser.add_argument('--with-supports', action='store_true', help='Also export the wafer and one combined fixed-support asset.')
    args = parser.parse_args()
    report = export_scene(args.scene, args.output, args.body, args.with_supports)
    print(json.dumps({k: report[k] for k in ('result', 'body_count', 'collider_count', 'joint_count', 'asset_sha256')}))
