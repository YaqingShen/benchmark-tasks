"""Four reusable convex meshes, shared by native physics and measured-pose replay."""
from collections import Counter
import hashlib
from pathlib import Path
import numpy as np


SHAPE_SCHEMA = 'mixed_convex_debris_v1'
COLORS = {'dust_grain': (.30, .33, .37), 'fiber': (.78, .76, .67),
          'flake': (.70, .62, .50), 'wear_fragment': (.23, .27, .31)}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _geometry(particle):
    kind = particle['kind']
    vertices = np.asarray(particle['canonical_vertices'], dtype=float)
    faces = np.asarray(particle['faces'])
    scale = np.asarray(particle['scale_m'], dtype=float)
    actual = np.asarray(particle['vertices_local_m'], dtype=float)
    color = np.asarray(particle['color_rgb'], dtype=float)
    if (kind not in COLORS or vertices.ndim != 2 or vertices.shape[1] != 3
            or not 4 <= len(vertices) <= 24 or not np.isfinite(vertices).all()
            or faces.ndim != 2 or faces.shape[1] != 3
            or not np.issubdtype(faces.dtype, np.integer)
            or faces.min() < 0 or faces.max() >= len(vertices)
            or scale.shape != (3,) or not np.isfinite(scale).all() or np.any(scale <= 0)
            or actual.shape != vertices.shape
            or not np.allclose(actual, vertices * scale, rtol=0, atol=1e-12)
            or color.shape != (3,) or not np.isfinite(color).all()
            or np.any(color < 0) or np.any(color > 1)
            or not np.isfinite(float(particle['mass_kg'])) or float(particle['mass_kg']) <= 0):
        raise ValueError('Invalid mixed-particle geometry: ' + str(particle.get('id')))
    edges = Counter(tuple(sorted((int(f[i]), int(f[(i + 1) % 3]))))
                    for f in faces for i in range(3))
    if any(count != 2 for count in edges.values()):
        raise ValueError('A particle mesh must be closed.')
    a, b, c = (vertices[faces[:, i]] for i in range(3))
    normals = np.cross(b - a, c - a)
    lengths = np.linalg.norm(normals, axis=1)
    if np.any(lengths <= 1e-12):
        raise ValueError('Degenerate particle mesh triangle.')
    normals /= lengths[:, None]
    if np.any(np.einsum('fvi,fi->fv', vertices[None] - a[:, None], normals) > 1e-9):
        raise ValueError('Particle faces must describe an outward-wound convex mesh.')
    volume = np.einsum('ij,ij->i', a, np.cross(b, c)).sum() / 6
    if volume <= 0:
        raise ValueError('A particle mesh must have positive volume.')
    return kind, vertices, faces.astype(int), scale, color


def _asset_text(kind, vertices, faces):
    def vec(value):
        return '(' + ', '.join(format(float(v), '.17g') for v in value) + ')'
    points = ', '.join(vec(v) for v in vertices)
    counts = ', '.join('3' for _ in faces)
    indices = ', '.join(str(int(v)) for v in faces.ravel())
    return f'''#usda 1.0
(defaultPrim = "Asset"
 metersPerUnit = 1
 upAxis = "Z")
def Xform "Asset" (prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI", "PhysxRigidBodyAPI"]) {{
 bool physics:rigidBodyEnabled = true
 bool physics:kinematicEnabled = false
 bool physxRigidBody:disableGravity = false
 bool physxRigidBody:enableCCD = true
 int physxRigidBody:solverPositionIterationCount = 64
 int physxRigidBody:solverVelocityIterationCount = 4
 float physics:mass = 0.000001
 point3f physics:centerOfMass = (0, 0, 0)
 def Material "Material" (prepend apiSchemas = ["PhysicsMaterialAPI", "PhysxMaterialAPI"]) {{
  float physics:staticFriction = 0.4
  float physics:dynamicFriction = 0.4
  float physics:restitution = 0
  token physxMaterial:frictionCombineMode = "max"
 }}
 def Mesh "Particle" (prepend apiSchemas = ["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI", "PhysxCollisionAPI", "MaterialBindingAPI"]) {{
  point3f[] points = [{points}]
  int[] faceVertexCounts = [{counts}]
  int[] faceVertexIndices = [{indices}]
  token subdivisionScheme = "none"
  token orientation = "rightHanded"
  token physics:approximation = "convexHull"
  bool physics:collisionEnabled = true
  color3f[] primvars:displayColor = [{vec(COLORS[kind])}]
  float physxCollision:contactOffset = 0.00001
  float physxCollision:restOffset = 0
  rel material:binding:physics = </Asset/Material>
 }}
}}
'''


def _templates(dust):
    if dust.get('shape_schema') != SHAPE_SCHEMA:
        raise ValueError('Unsupported mixed-particle shape schema.')
    particles, ids = dust['particles'], dust['ids']
    if len(particles) != 12 or len(set(ids)) != 12 or [p['id'] for p in particles] != ids:
        raise ValueError('Mixed-particle order differs from measured body order.')
    templates = {}
    for particle in particles:
        kind, vertices, faces, _, _ = _geometry(particle)
        text = _asset_text(kind, vertices, faces)
        if kind in templates and templates[kind] != text:
            raise ValueError('Particles of one kind must share a canonical mesh.')
        templates[kind] = text
    if set(templates) != set(COLORS):
        raise ValueError('All four mixed-particle shapes are required.')
    return templates


def export_mixed_assets(root, dust):
    """Write four deterministic assets before simulation; bind them into dust metadata."""
    root = Path(root).resolve()
    templates = _templates(dust)
    assets = {}
    for kind, text in sorted(templates.items()):
        name = 'native/debris_' + kind + '.usda'
        path = root / name
        data = text.encode()
        if path.exists() and path.read_bytes() != data:
            raise ValueError('Refusing to replace a different debris asset: ' + name)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_bytes(data)
        assets[kind] = dict(asset_file=name, sha256=hashlib.sha256(data).hexdigest())
    dust['assets'] = assets
    for particle in dust['particles']:
        asset = assets[particle['kind']]
        particle['asset_file'] = asset['asset_file']
        particle['asset_sha256'] = asset['sha256']
    return assets


def particle_asset_bindings(root, dust):
    """Read-only validation of the exact source meshes and per-particle geometry."""
    root = Path(root).resolve()
    templates = _templates(dust)
    assets = dust['assets']
    if set(assets) != set(templates):
        raise ValueError('Incomplete recorded mixed-particle assets.')
    bindings = {}
    for kind, text in templates.items():
        expected_name = 'native/debris_' + kind + '.usda'
        expected_hash = hashlib.sha256(text.encode()).hexdigest()
        if assets[kind] != dict(asset_file=expected_name, sha256=expected_hash):
            raise ValueError('Recorded debris mesh differs from its geometry: ' + kind)
        path = (root / expected_name).resolve()
        if not path.is_relative_to(root) or not path.is_file() or sha(path) != expected_hash:
            raise ValueError('Recorded debris asset is missing or changed: ' + kind)
        bindings[expected_name] = (path, expected_hash)
    for particle in dust['particles']:
        asset = assets[particle['kind']]
        if (particle['asset_file'] != asset['asset_file']
                or particle['asset_sha256'] != asset['sha256']):
            raise ValueError('Particle-to-asset binding changed: ' + particle['id'])
    return bindings
