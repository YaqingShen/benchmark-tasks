"""labkit: multi-shape parts with sliding and hinged joints for yam_arena task files.

yam_arena task files only take single primitives (Box, Cylinder, Sphere, Asset). labkit adds:

- ``Part``: ONE rigid body built from many boxes and cylinders (a bottle with a neck and cap, a tray with walls, a
  wafer cassette with slots), loose (give it a mass) or fixed to the table (``fixed=True``).
- ``Joint``: a moving piece on a part, either a ``"slide"`` (prismatic) or a ``"hinge"`` (revolute) along/about one
  of the part's axes, with limits, an optional spring back to a target, damping and joint friction. A spring clip, a
  focus knob, a microscope stage axis, a lid. ``parent`` lets a joint ride on another (a stage's x carriage on its y
  carriage on its focus slide).
- ``PartState``: what ``success()`` / ``score()`` receive for a part: the body origin's pose, velocities, bounds, and
  every joint's position (m for slides, rad for hinges).
- ``on_step(fn)`` / ``on_reset(fn)``: callbacks with the Isaac Lab environment after every control step / reset, for
  scoring that watches the whole trial.
- ``Model``: a real USD model (for example from NVIDIA SimReady) with its own bodies and joints, fixed to the table
  or loose, with per-joint drive settings and starting positions.
- Arena library assets are loaded from NVIDIA's production bucket (see ``arena_assets_fix``).
- A box may carry a fourth item, its yaw (rad) about the part's vertical, or a (yaw, pitch) pair (pitch about y), for walls that are not axis-aligned (a
  polygonal vial). Colours added to ``GLASS_COLORS`` are drawn with Omniverse's glass material (see-through), or,
  should that material fail to load, as a translucent surface.
- ``meshes``: triangle meshes (from STL files of real parts, see ``stl_mesh``) on a part or joint, drawn with flat
  shading. Each collides as its convex hull (``"hull"``), as a convex decomposition (``"decomp"``), or not at all
  (``"none"``: drawn only, with boxes and cylinders doing the colliding).

The technique (hooking ``yam_arena.scene`` and the embodiment from a task file's import, without modifying
yam_arena) follows robocurve/benchmark-tasks PR #40 (tasks/bearing-press/partkit.py). This is an independent
implementation that adds hinges, fixed parts and joint friction.

    import pathlib, sys
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    from labkit import Joint, part, on_step

    CLIP = Joint("clip", kind="hinge", axis="Y", anchor=(-0.03, 0, 0.012), lower=0.0, upper=0.6,
                 stiffness=0.4, target=0.0, boxes=[((0.0, 0, 0.012), (0.06, 0.012, 0.003), (0.75, 0.75, 0.78))])
    HOLDER = part("holder", pos=(0.30, 0.0, TOP + 0.005), fixed=True,
                  boxes=[((0, 0, 0), (0.08, 0.06, 0.01), (0.2, 0.2, 0.2))], joints=[CLIP])

Coordinates: shape centres and joint anchors are in the part frame (metres), at rest. ``pos`` is the part's body
origin in the world frame; ``yaw`` turns the part about the vertical. Everything that needs Isaac Sim is imported
lazily, so importing this module needs only numpy and the pure-Python parts of yam_arena.
"""

from __future__ import annotations

import importlib.abc
import importlib.util
import math
import pathlib
import sys
from dataclasses import dataclass, field

import numpy as np
from yam_arena.task_file import Box, ObjectState

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import arena_assets_fix  # noqa: E402,F401

__all__ = ["Joint", "Part", "PartState", "part", "Model", "JointSet", "model", "on_step", "on_reset", "env", "install",
           "GLASS_COLORS", "stl_mesh", "Mesh"]

CONTACT_REPORT: set[str] = set()
MASS_PROPERTIES: dict = {}
JOINT_FRAME_ROT: dict = {}  # identical parent/child frames preserve the rest pose
SURFACE_FRICTION: dict = {}
MATTE_COLORS: dict = {}  # tray-sort: colour -> roughness (matte foam, paper)
TEXTURES: dict = {}  # tray-sort: part name -> [(centre xyz in the part frame, width, height, facing yaw, png path)]
TABLE_CUTS: list = []  # tray-sort: holes through the station table [(centre x, centre y, size x, size y)]
TABLE_TRIM = dict(width=0.012, height=0.003, color=(0.16, 0.16, 0.17))  # tray-sort: grommet trim round each hole
GLASS_COLORS: set = set()  # colours drawn as glass

Color = tuple[float, float, float]
_AXES = ("X", "Y", "Z")

# Physics defaults for every part body and collider.
CONTACT_OFFSET = 0.0005  # m: below 1 mm fits, so tight parts do not touch on every face at once
SOLVER_POS_ITERS = 32
SOLVER_VEL_ITERS = 4
ANGULAR_DAMPING = 1.0
MAX_DEPEN_VEL = 1.0
MAX_LIN_VEL = None  # tray-sort: m/s cap on every part body (None: PhysX default)
MAX_ANG_VEL = None  # tray-sort: deg/s cap
LINEAR_DAMPING = None
ARTICULATION_SLEEP = None  # set to 0.0 to keep articulated parts awake


# -- specs -----------------------------------------------------------------------------------------------------------


def _vec3(v, what) -> tuple[float, float, float]:
    out = tuple(float(x) for x in v)
    if len(out) != 3 or not all(math.isfinite(x) for x in out):
        raise ValueError(f"{what} must be 3 finite numbers, got {v!r}")
    return out


def _boxes(specs, what):
    """Each box: (centre, size, colour[, yaw about the vertical, rad])."""
    out = []
    for spec in specs:
        c, s, k = spec[:3]
        turn = spec[3] if len(spec) > 3 else 0.0
        yaw, pitch = (float(turn[0]), float(turn[1])) if isinstance(turn, (tuple, list)) else (float(turn), 0.0)
        if not (math.isfinite(yaw) and math.isfinite(pitch)):
            raise ValueError(f"{what}: box turn must be finite")
        out.append((_vec3(c, what), _vec3(s, what), _vec3(k, what), (yaw, pitch)))
    return tuple(out)


@dataclass(frozen=True, eq=False)
class Mesh:
    """A triangle mesh in the part frame (metres): ``points`` (N, 3) and ``tris`` (M, 3) vertex indices, a colour, and
    how it collides: "hull", "decomp" or "none"."""

    points: np.ndarray
    tris: np.ndarray
    color: tuple
    collide: str = "none"

    def bounds(self):
        return self.points.min(axis=0), self.points.max(axis=0)


def _read_stl(path) -> np.ndarray:
    """Triangles (M, 3, 3) from a binary or ASCII STL file, in the file's units."""
    import struct

    data = pathlib.Path(path).read_bytes()
    if data[:5] == b"solid" and b"facet" in data[:400]:
        v = [[float(x) for x in ln.split()[1:4]] for ln in data.decode(errors="ignore").splitlines()
             if ln.strip().startswith("vertex")]
        return np.asarray(v, float).reshape(-1, 3, 3)
    n = struct.unpack("<I", data[80:84])[0]
    rec = np.frombuffer(data[84:84 + 50 * n], dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]))
    return rec["v"].astype(float)


def _rot(rpy) -> np.ndarray:
    """Rotation matrix from (roll, pitch, yaw) in rad, applied about x, then y, then z (fixed axes)."""
    r, p, y = (float(a) for a in rpy)
    cx, sx, cy, sy, cz, sz = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return rz @ ry @ rx


def stl_mesh(path, color, pos=(0.0, 0.0, 0.0), rpy=(0.0, 0.0, 0.0), scale=0.001, origin=None, collide="none",
             mirror_y=False) -> Mesh:
    """Load an STL (millimetres by default: ``scale`` 0.001) into the part frame: the file's point ``origin`` (file
    units; default its own origin) is scaled, rotated by ``rpy`` (or a 3 x 3 rotation matrix) and placed at ``pos``. ``mirror_y`` mirrors the part
    (a left-hand copy of a right-hand part)."""
    tri = _read_stl(path).reshape(-1, 3)
    if origin is not None:
        tri = tri - np.asarray(origin, float)
    tri = tri * float(scale)
    if mirror_y:
        tri[:, 1] *= -1.0
    rot = np.asarray(rpy, float)
    rot = rot if rot.shape == (3, 3) else _rot(rpy)
    pts = tri @ rot.T + np.asarray(pos, float)
    tris = np.arange(len(pts)).reshape(-1, 3)
    if mirror_y:
        tris = tris[:, ::-1]  # keep the faces pointing outwards
    if collide not in ("hull", "decomp", "none"):
        raise ValueError("collide must be 'hull', 'decomp' or 'none'")
    return Mesh(points=pts, tris=tris, color=_vec3(color, str(path)), collide=collide)


def _meshes(specs, what):
    out = tuple(specs)
    for m in out:
        if not isinstance(m, Mesh):
            raise ValueError(f"{what}: meshes must be made with stl_mesh()")
    return out


def _cyls(specs, what):
    """Each cylinder: (centre, radius, height, colour[, axis])."""
    out = []
    for spec in specs:
        c, r, h, k = spec[:4]
        axis = str(spec[4]).upper() if len(spec) > 4 else "Z"
        if axis not in _AXES:
            raise ValueError(f"{what}: cylinder axis must be X, Y or Z")
        out.append((_vec3(c, what), float(r), float(h), _vec3(k, what), axis))
    return tuple(out)


@dataclass(frozen=True)
class Joint:
    """Shapes on a joint. ``anchor`` is the joint's position in the part frame; the shapes are given in the part
    frame too, at the joint's zero position. ``kind`` is ``"slide"`` (metres along ``axis``) or ``"hinge"`` (radians
    about ``axis``, right-handed). ``lower``/``upper`` limit the joint; ``stiffness`` (N/m or N鐠虹棳/rad) pulls it
    towards ``target``; ``damping`` resists speed; ``friction`` (N or N鐠虹棳) holds it where it is left (a stiff knob).
    ``initial`` is the joint position at every reset."""

    name: str
    kind: str = "slide"
    axis: str = "X"
    anchor: tuple[float, float, float] = (0.0, 0.0, 0.0)
    boxes: tuple = ()
    cylinders: tuple = ()
    meshes: tuple = field(default=(), compare=False, hash=False)
    lower: float = -0.01
    upper: float = 0.0
    stiffness: float = 0.0
    damping: float = 0.0
    target: float = 0.0
    friction: float = 0.0
    initial: float = 0.0
    mass: float = 0.01
    parent: str = ""  # another joint of the same part this one rides on ("" = the part's main body)

    def checked(self) -> "Joint":
        if self.kind not in ("slide", "hinge"):
            raise ValueError(f"{self.name}: kind must be 'slide' or 'hinge'")
        if self.axis.upper() not in _AXES:
            raise ValueError(f"{self.name}: axis must be X, Y or Z")
        if not self.boxes and not self.cylinders and not self.meshes:
            raise ValueError(f"{self.name}: a joint needs at least one shape")
        if not self.lower <= self.initial <= self.upper:
            raise ValueError(f"{self.name}: initial must lie within [lower, upper]")
        if not self.lower <= self.target <= self.upper:  # Isaac Lab rejects a default joint position past a limit
            raise ValueError(f"{self.name}: target must lie within [lower, upper]; raise stiffness for more force")
        return Joint(
            name=self.name, kind=self.kind, axis=self.axis.upper(), anchor=_vec3(self.anchor, self.name),
            boxes=_boxes(self.boxes, self.name), cylinders=_cyls(self.cylinders, self.name),
            meshes=_meshes(self.meshes, self.name), lower=float(self.lower), upper=float(self.upper), stiffness=float(self.stiffness),
            damping=float(self.damping), target=float(self.target), friction=float(self.friction),
            initial=float(self.initial), mass=float(self.mass), parent=str(self.parent),
        )

    @property
    def joint_name(self) -> str:
        return f"{self.name}_joint"


@dataclass(frozen=True)
class Part(Box):
    """One rigid body of many shapes, optionally with joints. It subclasses Box only so yam_arena's task-file checks
    accept it; ``size`` is its rest bounding box and ``pos`` its body origin. ``mass`` is None for a fixed part."""

    boxes: tuple = ()
    cylinders: tuple = ()
    joints: tuple = ()
    fixed: bool = False
    meshes: tuple = field(default=(), compare=False, hash=False)
    self_collide: bool = False
    friction: float = 0.8
    meta: dict = field(default_factory=dict, compare=False, hash=False)

    def _check(self) -> None:
        if not self.boxes and not self.cylinders and not self.meshes:
            raise ValueError(f"{self.name}: a Part needs at least one shape")
        if self.fixed == (self.mass is not None):
            raise ValueError(f"{self.name}: give a loose part a mass, and a fixed part none")

    @property
    def free(self) -> bool:  # yam_arena treats free objects as jittered on reset; fixed parts are not
        return not self.fixed


def _extent(boxes, cyls, meshes=()):
    lo, hi = [], []
    for m in meshes:
        a, b = m.bounds()
        lo.append(a)
        hi.append(b)
    for c, s, _, (yaw, pitch) in boxes:
        if pitch:  # a wall tipped about y: bound it by its turned corners in x and z
            hxp = abs(s[0] / 2 * math.cos(pitch)) + abs(s[2] / 2 * math.sin(pitch))
            hzp = abs(s[0] / 2 * math.sin(pitch)) + abs(s[2] / 2 * math.cos(pitch))
            s = (2 * hxp, s[1], 2 * hzp)
        hx, hy = s[0] / 2, s[1] / 2
        ex = abs(hx * math.cos(yaw)) + abs(hy * math.sin(yaw))
        ey = abs(hx * math.sin(yaw)) + abs(hy * math.cos(yaw))
        half = np.array([ex, ey, s[2] / 2])
        lo.append(np.array(c) - half)
        hi.append(np.array(c) + half)
    for c, r, h, _, axis in cyls:
        half = np.full(3, r)
        half[_AXES.index(axis)] = h / 2
        lo.append(np.array(c) - half)
        hi.append(np.array(c) + half)
    return np.min(lo, axis=0), np.max(hi, axis=0)


def part(name, pos, boxes=(), cylinders=(), mass=None, fixed=False, yaw=0.0, joints=(), color=(0.5, 0.5, 0.5),
         friction=0.8, meshes=(), self_collide=False) -> Part:
    """Build a Part; its ``size`` is the rest extent of all its shapes (joints included). ``self_collide``: the part's
    joint bodies collide with its other bodies (PhysX still ignores a body and the one it hangs on), e.g. a flexible
    cable of hinged segments lying on the part's own board."""
    b, c = _boxes(boxes, name), _cyls(cylinders, name)
    js = tuple(j.checked() for j in joints)
    names = [j.name for j in js]
    for i, j in enumerate(js):
        if j.parent and j.parent not in names[:i]:
            raise ValueError(f"{name}: joint {j.name} rides on {j.parent!r}, which must be listed before it")
    all_b = list(b) + [x for j in js for x in j.boxes]
    all_c = list(c) + [x for j in js for x in j.cylinders]
    m = _meshes(meshes, name)
    all_m = list(m) + [x for j in js for x in j.meshes]
    lo, hi = _extent(all_b, all_c, all_m)
    return Part(name=name, pos=_vec3(pos, name), size=tuple((hi - lo).tolist()), mass=None if fixed else mass,
                color=_vec3(color, name), yaw=float(yaw), boxes=b, cylinders=c, joints=js, fixed=bool(fixed),
                friction=float(friction), meshes=m, self_collide=bool(self_collide))



@dataclass(frozen=True)
class JointSet:
    """Settings for one joint of a ``Model`` (by its short name in the model's model.json). ``None`` keeps the
    model's own value. Positions in m (slides) or rad (hinges)."""

    name: str
    stiffness: float | None = None
    damping: float | None = None
    target: float = 0.0
    initial: float = 0.0
    friction: float | None = None
    lower: float | None = None
    upper: float | None = None


@dataclass(frozen=True)
class Model(Box):
    """A real USD model placed on the table: ``usd`` is its root file, ``info`` its model.json (extent, bodies, joints).
    ``fixed`` pins the body named ``base`` to the world; otherwise the model is loose. ``pos`` is where its origin
    (the bottom centre of a SimReady model) goes. ``mimic`` couples joints: (follower, leader, gearing, offset) makes
    follower = gearing * leader + offset."""

    usd: str = ""
    info: dict = field(default_factory=dict, compare=False, hash=False)
    fixed: bool = True
    base: str = "body_01"
    joint_sets: tuple = ()
    mimic: tuple = ()
    friction: float | None = None

    def _check(self) -> None:
        if not pathlib.Path(self.usd).is_file():
            raise ValueError(f"{self.name}: no USD file at {self.usd}")

    @property
    def free(self) -> bool:
        return False  # never jittered: the layout is fixed

    @property
    def joint_names(self) -> list[str]:
        return [j["name"] for j in self.info.get("joints", [])]

    def joint_prim(self, short: str) -> str:
        for j in self.info.get("joints", []):
            if j["name"] == short:
                return j["prim"]
        raise KeyError(f"{self.name}: no joint {short!r}; it has {self.joint_names}")


def model(name, folder, pos, yaw=0.0, fixed=True, base="body_01", joints=(), mimic=(), friction=None) -> Model:
    """Load a model folder holding model.json and simready_usd/ (see make_model_json.py)."""
    import json

    folder = pathlib.Path(folder).resolve()
    info = json.loads((folder / "model.json").read_text())
    lo, hi = np.array(info["min"]), np.array(info["max"])
    js = tuple(joints)
    known = [j["name"] for j in info["joints"]]
    for j in js:
        if j.name not in known:
            raise ValueError(f"{name}: no joint {j.name!r}; it has {known}")
    for m in mimic:
        if m[0] not in known or m[1] not in known:
            raise ValueError(f"{name}: mimic joints must be among {known}")
    return Model(name=name, pos=_vec3(pos, name), size=tuple((hi - lo).tolist()), mass=None if fixed else info["mass"],
                 color=(0.5, 0.5, 0.5), yaw=float(yaw), usd=str(folder / info["usd"]), info=info, fixed=bool(fixed),
                 base=base, joint_sets=js, mimic=tuple(mimic), friction=friction)


@dataclass(frozen=True)
class PartState(ObjectState):
    """A part's body-origin pose, velocities, world-aligned bounds, and joint positions by joint name."""

    joints: dict = field(default_factory=dict)

    def rotation(self) -> np.ndarray:
        x, y, z, w = self.quat_xyzw
        return np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ])

    def to_world(self, local) -> np.ndarray:
        return self.pos + self.rotation() @ np.asarray(local, float)

    def to_local(self, world) -> np.ndarray:
        return self.rotation().T @ (np.asarray(world, float) - self.pos)

    def tilt_deg(self) -> float:
        return float(np.degrees(np.arccos(np.clip(self.rotation()[2, 2], -1.0, 1.0))))


# -- authoring the USD ------------------------------------------------------------------------------------------------

_SPECS: dict[str, Part] = {}
# Own stool support: real maximal-coordinate closing joints, excluded from articulation tree.
LOOP_JOINTS: dict[str, list] = {}


def _author(prim_path: str, cfg, translation=None, orientation=None, **kwargs):
    """Write the part under ``prim_path``: an Xform holding the main body and one body per joint as siblings."""
    import isaaclab.sim as sim_utils
    from isaaclab.sim.utils import bind_physics_material, bind_visual_material
    from isaaclab.sim.utils.stage import get_current_stage
    from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdPhysics
    from yam_arena.station import linear_rgb

    spec: Part = _SPECS[cfg.part_name]
    stage = get_current_stage()
    root = UsdGeom.Xform.Define(stage, prim_path)
    root.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*(translation or (0.0, 0.0, 0.0))))
    x, y, z, w = orientation if orientation is not None else (0.0, 0.0, 0.0, 1.0)
    root.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(w, x, y, z))

    phys = f"{prim_path}/Looks/physics"
    pm = sim_utils.RigidBodyMaterialCfg(static_friction=spec.friction, dynamic_friction=spec.friction)
    pm.func(phys, pm)
    looks: dict = {}
    surface_materials = {}

    def look(color):
        if color not in looks:
            path = f"{prim_path}/Looks/c{len(looks)}"
            made = False
            if color in GLASS_COLORS:
                # Omniverse's glass material; failing that, a translucent surface; failing that, a plain one below.
                makers = (
                    lambda: sim_utils.GlassMdlCfg(glass_color=linear_rgb(color), thin_walled=True),
                    lambda: sim_utils.PreviewSurfaceCfg(diffuse_color=linear_rgb(color), roughness=0.05, opacity=0.3),
                )
                for n, make in enumerate(makers):
                    try:
                        vis = make()
                        vis.func(f"{path}_g{n}", vis)
                        path, made = f"{path}_g{n}", True
                        break
                    except Exception:
                        continue
            if not made:
                vis = sim_utils.PreviewSurfaceCfg(diffuse_color=linear_rgb(color), roughness=MATTE_COLORS.get(color, 0.5))
                vis.func(path, vis)
            looks[color] = path
        return looks[color]

    def body(path, pos, mass, kinematic=False):
        xf = UsdGeom.Xform.Define(stage, path)
        xf.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*pos))
        xf.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(1, 0, 0, 0))
        prim = xf.GetPrim()
        rb = UsdPhysics.RigidBodyAPI.Apply(prim)
        if kinematic:
            rb.CreateKinematicEnabledAttr(True)
        UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(mass)
        prb = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        prb.CreateMaxDepenetrationVelocityAttr(MAX_DEPEN_VEL)
        prb.CreateSolverPositionIterationCountAttr(SOLVER_POS_ITERS)
        prb.CreateAngularDampingAttr(ANGULAR_DAMPING)
        if MAX_LIN_VEL is not None:
            prb.CreateMaxLinearVelocityAttr(MAX_LIN_VEL)
        if MAX_ANG_VEL is not None:
            prb.CreateMaxAngularVelocityAttr(MAX_ANG_VEL)
        if LINEAR_DAMPING is not None:
            prb.CreateLinearDampingAttr(LINEAR_DAMPING)
        return prim

    def collide(prim, color):
        UsdPhysics.CollisionAPI.Apply(prim)
        pc = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        pc.CreateContactOffsetAttr(CONTACT_OFFSET)
        pc.CreateRestOffsetAttr(0.0)
        coeffs = SURFACE_FRICTION.get((spec.name, color))
        if coeffs and coeffs not in surface_materials:
            material = sim_utils.RigidBodyMaterialCfg(static_friction=coeffs[0], dynamic_friction=coeffs[1])
            path = f"{prim_path}/Looks/surface{len(surface_materials)}"
            material.func(path, material)
            surface_materials[coeffs] = path
        bind_physics_material(str(prim.GetPath()), surface_materials.get(coeffs,phys), stage=stage)
        bind_visual_material(str(prim.GetPath()), look(color), stage=stage)

    def mesh_prims(path, meshes, offset):
        from pxr import Vt

        off = np.array(offset)
        for i, m in enumerate(meshes):
            mesh = UsdGeom.Mesh.Define(stage, f"{path}/mesh_{i}")
            pts = m.points - off
            mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(pts.astype(np.float32)))
            mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(m.tris), 3, dtype=np.int32)))
            mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(m.tris.reshape(-1).astype(np.int32)))
            tri = pts[m.tris]
            n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
            n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
            mesh.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(np.repeat(n, 3, axis=0).astype(np.float32)))
            mesh.SetNormalsInterpolation(UsdGeom.Tokens.faceVarying)
            mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
            mesh.CreateDoubleSidedAttr(True)
            prim = mesh.GetPrim()
            if m.collide == "none":
                bind_visual_material(str(prim.GetPath()), look(m.color), stage=stage)
                continue
            collide(prim, m.color)
            mc = UsdPhysics.MeshCollisionAPI.Apply(prim)
            mc.CreateApproximationAttr("convexHull" if m.collide == "hull" else "convexDecomposition")

    def shapes(path, boxes, cyls, offset, meshes=()):
        mesh_prims(path, meshes, offset)
        off = np.array(offset)
        for i, (c, s, k, (yaw, pitch)) in enumerate(boxes):
            cube = UsdGeom.Cube.Define(stage, f"{path}/box_{i}")
            cube.CreateSizeAttr(1.0)
            cube.AddTranslateOp().Set(Gf.Vec3d(*(np.array(c) - off)))
            if yaw:
                cube.AddRotateZOp().Set(math.degrees(yaw))
            if pitch:
                cube.AddRotateYOp().Set(math.degrees(pitch))
            cube.AddScaleOp().Set(Gf.Vec3f(*s))
            collide(cube.GetPrim(), k)
        for i, (c, r, h, k, axis) in enumerate(cyls):
            cyl = UsdGeom.Cylinder.Define(stage, f"{path}/cyl_{i}")
            cyl.CreateAxisAttr(axis)
            cyl.CreateRadiusAttr(r)
            cyl.CreateHeightAttr(h)
            cyl.AddTranslateOp().Set(Gf.Vec3d(*(np.array(c) - off)))
            collide(cyl.GetPrim(), k)

    main = f"{prim_path}/body"
    articulated = bool(spec.joints)
    body(main, (0.0, 0.0, 0.0), spec.mass or 1.0, kinematic=spec.fixed and not articulated)
    if spec.name in CONTACT_REPORT:
        PhysxSchema.PhysxContactReportAPI.Apply(stage.GetPrimAtPath(main)).CreateThresholdAttr(0.)
    if spec.name in MASS_PROPERTIES:
        com, inertia = MASS_PROPERTIES[spec.name]
        mp = UsdPhysics.MassAPI.Get(stage, main)
        mp.CreateCenterOfMassAttr(Gf.Vec3f(*com))
        mp.CreateDiagonalInertiaAttr(Gf.Vec3f(*inertia))
    shapes(main, spec.boxes, spec.cylinders, (0.0, 0.0, 0.0), spec.meshes)
    for i, t in enumerate(TEXTURES.get(spec.name, ())):
        _textured_quad(stage, f"{main}/tex_{i}", f"{prim_path}/Looks/tex_{i}", *t)

    for j in spec.joints:
        path = f"{prim_path}/{j.name}"
        body(path, j.anchor, j.mass)
        shapes(path, j.boxes, j.cylinders, j.anchor, j.meshes)
        if j.kind == "slide":
            jp = UsdPhysics.PrismaticJoint.Define(stage, f"{path}/{j.joint_name}")
            scale, drive_type = 1.0, "linear"
        else:
            jp = UsdPhysics.RevoluteJoint.Define(stage, f"{path}/{j.joint_name}")
            scale, drive_type = 180.0 / math.pi, "angular"  # USD revolute limits and drives are in degrees
        jp.CreateAxisAttr(j.axis)
        parent = next((p for p in spec.joints if p.name == j.parent), None)
        jp.CreateBody0Rel().SetTargets([Sdf.Path(f"{prim_path}/{parent.name}" if parent else main)])
        jp.CreateBody1Rel().SetTargets([Sdf.Path(path)])
        jp.CreateLocalPos0Attr(Gf.Vec3f(*(np.array(j.anchor) - (np.array(parent.anchor) if parent else 0.0))))
        _fr=JOINT_FRAME_ROT.get(j.name,(1.,0.,0.,0.))
        jp.CreateLocalRot0Attr(Gf.Quatf(*_fr))
        jp.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
        jp.CreateLocalRot1Attr(Gf.Quatf(*_fr))
        jp.CreateLowerLimitAttr(j.lower * scale)
        jp.CreateUpperLimitAttr(j.upper * scale)
        drive = UsdPhysics.DriveAPI.Apply(jp.GetPrim(), drive_type)
        drive.CreateTypeAttr("force")
        drive.CreateStiffnessAttr(j.stiffness / scale)
        drive.CreateDampingAttr(j.damping / scale)
        drive.CreateTargetPositionAttr(j.target * scale)
        if j.friction > 0:
            PhysxSchema.PhysxJointAPI.Apply(jp.GetPrim()).CreateJointFrictionAttr(j.friction)
        # The joint starts at ``initial``: PhysX reads this state when the simulation starts.
        state = PhysxSchema.JointStateAPI.Apply(jp.GetPrim(), drive_type)
        state.CreatePositionAttr(j.initial * scale)
        state.CreateVelocityAttr(0.0)

    # Loop entries: name, body0, body1, common anchor in the part zero frame, axis.
    for lname, b0, b1, common, axis in LOOP_JOINTS.get(spec.name, []):
        anchors = {"body": np.zeros(3), **{j.name: np.asarray(j.anchor) for j in spec.joints}}
        loop = UsdPhysics.RevoluteJoint.Define(stage, f"{prim_path}/{lname}")
        loop.CreateBody0Rel().SetTargets([Sdf.Path(f"{prim_path}/{b0}")])
        loop.CreateBody1Rel().SetTargets([Sdf.Path(f"{prim_path}/{b1}")])
        loop.CreateAxisAttr(axis)
        loop.CreateLocalPos0Attr(Gf.Vec3f(*(np.asarray(common)-anchors[b0])))
        loop.CreateLocalPos1Attr(Gf.Vec3f(*(np.asarray(common)-anchors[b1])))
        loop.CreateLocalRot0Attr(Gf.Quatf(1, 0, 0, 0))
        loop.CreateLocalRot1Attr(Gf.Quatf(1, 0, 0, 0))
        loop.CreateExcludeFromArticulationAttr(True)
        loop.CreateCollisionEnabledAttr(False)
        print(f"STOOLLOOP {lname} body0={b0} body1={b1} excluded={loop.GetExcludeFromArticulationAttr().Get()}",flush=True)

    if articulated:
        if spec.fixed:
            anchor = UsdPhysics.FixedJoint.Define(stage, f"{prim_path}/fixed_to_world")
            anchor.CreateBody1Rel().SetTargets([Sdf.Path(main)])
            art_prim = root.GetPrim()
        else:
            art_prim = stage.GetPrimAtPath(main)
        UsdPhysics.ArticulationRootAPI.Apply(art_prim)
        art = PhysxSchema.PhysxArticulationAPI.Apply(art_prim)
        art.CreateEnabledSelfCollisionsAttr(bool(spec.self_collide))
        art.CreateSolverPositionIterationCountAttr(SOLVER_POS_ITERS)
        art.CreateSolverVelocityIterationCountAttr(SOLVER_VEL_ITERS)
        if ARTICULATION_SLEEP is not None:  # tray-sort: a sleeping GPU articulation ignores root-pose writes
            art.CreateSleepThresholdAttr(ARTICULATION_SLEEP)
            art.CreateStabilizationThresholdAttr(0.0)
    return root.GetPrim()


def _textured_quad(stage, path, mat_path, c, w, h, yaw, png, pitch=0.0, pivot=None):
    """A visual-only rectangle (no collider) facing ``yaw`` with an image on it (UsdPreviewSurface + UsdUVTexture);
    then rotated by ``pitch`` about the part's y axis through ``pivot`` (to match a box tilted the same way)."""
    from pxr import Gf, Sdf, UsdGeom, UsdShade, Vt

    u = np.array([-math.sin(yaw), math.cos(yaw), 0.0]); v = np.array([0.0, 0.0, 1.0]); n = np.cross(u, v)
    c = np.array(c, float)
    pts = [c - w / 2 * u - h / 2 * v, c + w / 2 * u - h / 2 * v, c + w / 2 * u + h / 2 * v, c - w / 2 * u + h / 2 * v]
    if pitch:
        cp, sp = math.cos(pitch), math.sin(pitch)
        Ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
        pv = np.array(pivot if pivot is not None else c, float)
        pts = [Ry @ (p - pv) + pv for p in pts]
        n = Ry @ n
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*p) for p in pts]))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray([4])); mesh.CreateFaceVertexIndicesAttr(Vt.IntArray([0, 1, 2, 3]))
    mesh.CreateNormalsAttr(Vt.Vec3fArray([Gf.Vec3f(*n)] * 4)); mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
    st.Set(Vt.Vec2fArray([Gf.Vec2f(0, 0), Gf.Vec2f(1, 0), Gf.Vec2f(1, 1), Gf.Vec2f(0, 1)]))
    mat = UsdShade.Material.Define(stage, mat_path)
    pbr = UsdShade.Shader.Define(stage, f"{mat_path}/pbr"); pbr.CreateIdAttr("UsdPreviewSurface")
    pbr.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.55)
    rd = UsdShade.Shader.Define(stage, f"{mat_path}/st"); rd.CreateIdAttr("UsdPrimvarReader_float2")
    rd.CreateInput("varname", Sdf.ValueTypeNames.String).Set("st")
    tex = UsdShade.Shader.Define(stage, f"{mat_path}/tex"); tex.CreateIdAttr("UsdUVTexture")
    tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(str(png))
    tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
    tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(rd.CreateOutput("result", Sdf.ValueTypeNames.Float2))
    pbr.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3))
    mat.CreateSurfaceOutput().ConnectToSource(pbr.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat)


def _holed_table(scene, table):
    """The station table with TABLE_CUTS: static slabs round the holes (cuts share one x band), plus a trim round
    each hole and a dark lining down its cut edge."""
    from yam_arena import station

    x0 = station.TABLE_CENTER[0] - station.TABLE_SIZE[0] / 2; x1 = x0 + station.TABLE_SIZE[0]
    y0 = station.TABLE_CENTER[1] - station.TABLE_SIZE[1] / 2; y1 = y0 + station.TABLE_SIZE[1]
    zc, th = station.TABLE_CENTER[2], station.TABLE_SIZE[2]
    colour = scene.STATION_TABLE_COLOR
    hx0 = min(c[0] - c[2] / 2 for c in TABLE_CUTS); hx1 = max(c[0] + c[2] / 2 for c in TABLE_CUTS)
    out = [scene._box("table", (hx0 - x0, y1 - y0, th), ((x0 + hx0) / 2, (y0 + y1) / 2, zc), colour, cls=scene._Table),
           scene._box("table_far", (x1 - hx1, y1 - y0, th), ((hx1 + x1) / 2, (y0 + y1) / 2, zc), colour)]
    edges = sorted([y0, y1] + [c[1] + s * c[3] / 2 for c in TABLE_CUTS for s in (-1, 1)])
    for k in range(0, len(edges), 2):
        a, b = edges[k], edges[k + 1]
        if b - a > 1e-6:
            out.append(scene._box(f"table_mid{k // 2}", (hx1 - hx0, b - a, th), ((hx0 + hx1) / 2, (a + b) / 2, zc), colour))
    tw, tht, tc = TABLE_TRIM["width"], TABLE_TRIM["height"], TABLE_TRIM["color"]
    top = zc + th / 2
    for i, (cx, cy, sx, sy) in enumerate(TABLE_CUTS):
        for j, (px, py, lx, ly) in enumerate((
                (cx + sx / 2 + tw / 2, cy, tw, sy + 2 * tw), (cx - sx / 2 - tw / 2, cy, tw, sy + 2 * tw),
                (cx, cy + sy / 2 + tw / 2, sx, tw), (cx, cy - sy / 2 - tw / 2, sx, tw))):
            out.append(scene._box(f"hole{i}_trim{j}", (lx, ly, tht), (px, py, top + tht / 2), tc))
        lt = 0.003  # lining down the cut edge (inside the hole)
        for j, (px, py, lx, ly) in enumerate((
                (cx + sx / 2 - lt / 2, cy, lt, sy), (cx - sx / 2 + lt / 2, cy, lt, sy),
                (cx, cy + sy / 2 - lt / 2, sx - 2 * lt, lt), (cx, cy - sy / 2 + lt / 2, sx - 2 * lt, lt))):
            out.append(scene._box(f"hole{i}_lining{j}", (lx, ly, th + 0.006), (px, py, zc - 0.003), tc))
    return out


def _arena_object(spec: Part):
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import ArticulationCfg
    from isaaclab.sim.spawners.spawner_cfg import SpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.utils.pose import Pose

    @configclass
    class LabPartSpawnerCfg(SpawnerCfg):
        func: object = clone(_author)
        part_name: str = ""

    class LabPartObject(Object):
        def _generate_articulation_cfg(self):
            # Implicit actuators carry each joint's spring and damping; the target is the joint's ``target``.
            actuators = {
                j.name: ImplicitActuatorCfg(joint_names_expr=[j.joint_name], stiffness=j.stiffness, damping=j.damping,
                                            friction=j.friction or None)
                for j in spec.joints
            }
            # Isaac Lab aims each implicit actuator at the default joint position, so the default is the spring
            # target; the starting position comes from the USD joint state authored in ``_author``.
            init = ArticulationCfg.InitialStateCfg(joint_pos={j.joint_name: j.target for j in spec.joints})
            cfg = ArticulationCfg(prim_path=self.prim_path, spawn=self.spawner_cfg, actuators=actuators,
                                  init_state=init)
            return self._add_initial_pose_to_cfg(cfg)

    kind = ObjectType.ARTICULATION if spec.joints else ObjectType.RIGID
    yaw = spec.yaw
    return LabPartObject(
        name=spec.name,
        object_type=kind,
        spawner_cfg=LabPartSpawnerCfg(part_name=spec.name),
        initial_pose=Pose(position_xyz=tuple(spec.pos), rotation_xyzw=(0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2))),
    )



def _author_model(prim_path: str, cfg, translation=None, orientation=None, **kwargs):
    """Reference the model's USD under ``prim_path``, pin its base to the world if fixed, make it one articulation, and
    apply the joint settings and couplings."""
    import isaaclab.sim as sim_utils
    from isaaclab.sim.utils import bind_physics_material
    from isaaclab.sim.utils.stage import get_current_stage
    from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics

    spec: Model = _SPECS[cfg.part_name]
    stage = get_current_stage()
    root = UsdGeom.Xform.Define(stage, prim_path)
    root.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*(translation or (0.0, 0.0, 0.0))))
    x, y, z, w = orientation if orientation is not None else (0.0, 0.0, 0.0, 1.0)
    root.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(w, x, y, z))
    mpath = f"{prim_path}/model"
    mprim = stage.DefinePrim(mpath, "Xform")
    mprim.GetReferences().AddReference(spec.usd)
    vs = mprim.GetVariantSets()
    for name, choice in (("PhysX", "Enabled"), ("Newton", "Disabled"), ("MuJoCo", "Disabled")):
        if vs.HasVariantSet(name):
            vs.GetVariantSet(name).SetVariantSelection(choice)

    if spec.friction is not None:
        phys = f"{prim_path}/Looks/physics"
        pm = sim_utils.RigidBodyMaterialCfg(static_friction=spec.friction, dynamic_friction=spec.friction)
        pm.func(phys, pm)
        for p in Usd.PrimRange(mprim):
            if p.HasAPI(UsdPhysics.CollisionAPI):
                bind_physics_material(str(p.GetPath()), phys, stage=stage)

    joints = {j["name"]: j for j in spec.info["joints"]}
    for js in spec.joint_sets:
        j = joints[js.name]
        jp = stage.GetPrimAtPath(f"{mpath}/Joints/{j['prim']}")
        drive_type, scale = ("angular", 180.0 / math.pi) if j["kind"] == "hinge" else ("linear", 1.0)
        joint = UsdPhysics.PrismaticJoint(jp) if j["kind"] == "slide" else UsdPhysics.RevoluteJoint(jp)
        if js.lower is not None:
            joint.GetLowerLimitAttr().Set(js.lower * scale)
        if js.upper is not None:
            joint.GetUpperLimitAttr().Set(js.upper * scale)
        drive = UsdPhysics.DriveAPI.Apply(jp, drive_type)
        drive.CreateTypeAttr("force")
        if js.stiffness is not None:
            drive.CreateStiffnessAttr().Set(js.stiffness / scale)
        if js.damping is not None:
            drive.CreateDampingAttr().Set(js.damping / scale)
        drive.CreateTargetPositionAttr().Set(js.target * scale)
        if js.friction is not None:
            PhysxSchema.PhysxJointAPI.Apply(jp).CreateJointFrictionAttr().Set(js.friction)
        state = PhysxSchema.JointStateAPI.Apply(jp, drive_type)
        state.CreatePositionAttr().Set(js.initial * scale)
        state.CreateVelocityAttr().Set(0.0)

    for follower, leader, gearing, offset in spec.mimic:
        fj, lj = joints[follower], joints[leader]
        fp = stage.GetPrimAtPath(f"{mpath}/Joints/{fj['prim']}")
        axis = ("rot" if fj["kind"] == "hinge" else "trans") + fj["axis"]
        mj = PhysxSchema.PhysxMimicJointAPI.Apply(fp, axis)
        mj.CreateReferenceJointRel().SetTargets([Sdf.Path(f"{mpath}/Joints/{lj['prim']}")])
        mj.CreateReferenceJointAxisAttr().Set(("rot" if lj["kind"] == "hinge" else "trans") + lj["axis"])
        mj.CreateGearingAttr().Set(float(gearing))
        mj.CreateOffsetAttr().Set(float(offset))

    if not spec.info["joints"]:
        # No moving joints: make the whole model one rigid body. The pieces' own bodies and the fixed joints between
        # them go; their colliders and masses then belong to the model's root body.
        for p in Usd.PrimRange(mprim):
            if p != mprim and p.HasAPI(UsdPhysics.RigidBodyAPI):
                p.RemoveAPI(UsdPhysics.RigidBodyAPI)
            if p.IsA(UsdPhysics.Joint):
                p.SetActive(False)
        rb = UsdPhysics.RigidBodyAPI.Apply(mprim)
        if spec.fixed:
            rb.CreateKinematicEnabledAttr(True)
        prb = PhysxSchema.PhysxRigidBodyAPI.Apply(mprim)
        prb.CreateMaxDepenetrationVelocityAttr(MAX_DEPEN_VEL)
        prb.CreateSolverPositionIterationCountAttr(SOLVER_POS_ITERS)
        return root.GetPrim()

    if spec.fixed:
        anchor = UsdPhysics.FixedJoint.Define(stage, f"{prim_path}/fixed_to_world")
        anchor.CreateBody1Rel().SetTargets([Sdf.Path(f"{mpath}/Geometry/{spec.base}_obj_00")])
    UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())
    art = PhysxSchema.PhysxArticulationAPI.Apply(root.GetPrim())
    art.CreateEnabledSelfCollisionsAttr(False)
    art.CreateSolverPositionIterationCountAttr(SOLVER_POS_ITERS)
    art.CreateSolverVelocityIterationCountAttr(SOLVER_VEL_ITERS)
    return root.GetPrim()


def _arena_model(spec: Model):
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import ArticulationCfg
    from isaaclab.sim.spawners.spawner_cfg import SpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.utils.pose import Pose

    @configclass
    class ModelSpawnerCfg(SpawnerCfg):
        func: object = clone(_author_model)
        part_name: str = ""

    sets = {j.name: j for j in spec.joint_sets}

    class ModelObject(Object):
        def _generate_articulation_cfg(self):
            actuators, init = {}, {}
            for j in spec.info["joints"]:
                js = sets.get(j["name"])
                actuators[j["name"]] = ImplicitActuatorCfg(
                    joint_names_expr=[j["prim"]],
                    stiffness=None if js is None else js.stiffness,
                    damping=None if js is None else js.damping,
                    friction=None if js is None else js.friction,
                )
                lo = j["lower"] if j["lower"] is not None else -1e9
                hi = j["upper"] if j["upper"] is not None else 1e9
                if js is not None:
                    lo = js.lower if js.lower is not None else lo
                    hi = js.upper if js.upper is not None else hi
                init[j["prim"]] = min(max(js.target if js else 0.0, lo), hi)
            cfg = ArticulationCfg(prim_path=self.prim_path, spawn=self.spawner_cfg, actuators=actuators,
                                  init_state=ArticulationCfg.InitialStateCfg(joint_pos=init))
            return self._add_initial_pose_to_cfg(cfg)

    yaw = spec.yaw
    return ModelObject(
        name=spec.name,
        object_type=ObjectType.ARTICULATION if spec.info["joints"] else ObjectType.RIGID,
        spawner_cfg=ModelSpawnerCfg(part_name=spec.name),
        initial_pose=Pose(position_xyz=tuple(spec.pos), rotation_xyzw=(0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2))),
    )


# -- reading state ----------------------------------------------------------------------------------------------------


def _np(x) -> np.ndarray:
    return getattr(x, "torch", x).detach().cpu().numpy().astype(np.float64)


def _part_state(environment, spec: Part) -> PartState:
    base = environment.unwrapped
    asset = base.scene[spec.name]
    origin = _np(base.scene.env_origins[0])
    pose = _np(asset.data.root_link_pose_w)[0]
    vel = _np(asset.data.root_vel_w)[0] if hasattr(asset.data, "root_vel_w") else np.zeros(6)
    probe = PartState(pos=pose[:3] - origin, quat_xyzw=pose[3:7], lin_vel=vel[:3], ang_vel=vel[3:6],
                      low=np.zeros(3), high=np.zeros(3))
    if isinstance(spec, Model):
        lo, hi = np.array(spec.info["min"]), np.array(spec.info["max"])
    else:
        lo, hi = _extent(spec.boxes, spec.cylinders, spec.meshes)
    corners = np.array([[a, b, c] for a in (lo[0], hi[0]) for b in (lo[1], hi[1]) for c in (lo[2], hi[2])])
    pts = np.array([probe.to_world(p) for p in corners])
    joints = {}
    if isinstance(spec, Model) and spec.info["joints"]:
        names = list(asset.joint_names)
        q = _np(asset.data.joint_pos)[0]
        joints = {j["name"]: float(q[names.index(j["prim"])]) for j in spec.info["joints"] if j["prim"] in names}
    elif isinstance(spec, Part) and spec.joints:
        names = list(asset.joint_names)
        q = _np(asset.data.joint_pos)[0]
        joints = {j.name: float(q[names.index(j.joint_name)]) for j in spec.joints}
    return PartState(pos=probe.pos, quat_xyzw=probe.quat_xyzw, lin_vel=probe.lin_vel, ang_vel=probe.ang_vel,
                     low=pts.min(0), high=pts.max(0), joints=joints)


# -- hooks into yam_arena ---------------------------------------------------------------------------------------------

_STEP: list = []
_RESET: list = []
_ENV: dict = {"env": None}


def on_step(fn):
    """Register ``fn(env)`` to run after every control step, before scoring."""
    _STEP.append(fn)
    return fn


def on_reset(fn):
    """Register ``fn(env)`` to run after every reset."""
    _RESET.append(fn)
    return fn


def env():
    return _ENV["env"]


def _patch_scene(scene) -> None:
    if getattr(scene, "_labkit", False):
        return
    base_object, base_states = scene.task_object, scene.object_states

    def task_object(spec):
        if isinstance(spec, Part):
            _SPECS[spec.name] = spec
            return _arena_object(spec)
        if isinstance(spec, Model):
            _SPECS[spec.name] = spec
            return _arena_model(spec)
        return base_object(spec)

    def object_states(environment, names):
        plain = [n for n in names if n not in _SPECS]
        out = base_states(environment, plain) if plain else {}
        for n in names:
            if n in _SPECS:
                out[n] = _part_state(environment, _SPECS[n])
        return {n: out[n] for n in names}

    base_room = scene._station_room

    def _station_room():
        objs = base_room()
        if not TABLE_CUTS:
            return objs
        out = []
        for o in objs:
            out.extend(_holed_table(scene, o) if o.name == "table" else [o])
        return out

    scene.task_object, scene.object_states = task_object, object_states
    scene._station_room = _station_room
    scene._labkit = True


def _patch_embodiment(module) -> None:
    cls = module.YamArenaEmbodiment
    if getattr(cls, "_labkit", False):
        return
    base_step, base_reset = cls._task_step, cls.reset

    def _task_step(self, *a, **k):
        _ENV["env"] = self._env
        for fn in _STEP:
            fn(self._env)
        return base_step(self, *a, **k)

    def reset(self, *a, **k):
        obs = base_reset(self, *a, **k)
        _ENV["env"] = self._env
        for fn in _RESET:
            fn(self._env)
        return obs

    cls._task_step, cls.reset = _task_step, reset
    cls._labkit = True


_TARGETS = {"yam_arena.scene": _patch_scene, "yam_arena.inspect_embodiment": _patch_embodiment}


class _Loader(importlib.abc.Loader):
    def __init__(self, inner, patch):
        self.inner, self.patch = inner, patch

    def create_module(self, spec):
        return self.inner.create_module(spec)

    def exec_module(self, module):
        self.inner.exec_module(module)
        self.patch(module)


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname not in _TARGETS:
            return None
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(fullname)
        finally:
            sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return None
        spec.loader = _Loader(spec.loader, _TARGETS[fullname])
        return spec


def install() -> None:
    """Hook yam_arena now for modules already imported, otherwise when they are. Idempotent."""
    for name, patch in _TARGETS.items():
        if name in sys.modules:
            patch(sys.modules[name])
    if not any(isinstance(f, _Finder) for f in sys.meta_path):
        sys.meta_path.insert(0, _Finder())


install()
