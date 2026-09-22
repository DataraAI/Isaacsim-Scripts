"""USD adapter and Isaac Sim debug markers for network-cable insertion features.

Loads the cable by reference. Debug prims are authored on the live stage only;
the original cable USD is never written.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

from insertion_features.cache import (
    DEFAULT_CACHE_PATH,
    save_local_cable_features,
    world_features_for_head,
)
from insertion_features.geometry import (
    ConnectorFeatures,
    extract_connector_features_from_mesh,
    transform_connector_features,
)

NETWORK_CABLE_USD = (
    Path.home() / "isaacsim_assets" / "Network cable 001" / "model_Networkcable1_69323.usd"
)
NETWORK_CABLE_ROOT_PATH = "/World/NetworkCable"
DEBUG_MARKER_ROOT = "/World/CableFeatureDebug"
CRYSTAL_HEAD_PREFIX = "E_crystal_head"
LINE_PREFIX = "E_line"

_AXIS_LENGTH_M = 0.030
_AXIS_RADIUS_M = 0.00045
_CENTER_RADIUS_M = 0.0014
_KEYPOINT_RADIUS_M = 0.0012
_LOWER_CORNER_RADIUS_M = 0.00085
_PLANE_OFFSET_M = 0.00018

_COLOR_MATING_PLANE = (0.10, 0.82, 0.92)
_COLOR_MATING_CENTER = (0.95, 0.12, 0.12)
_COLOR_INSERTION_AXIS = (1.00, 0.92, 0.12)
_COLOR_LATCH_PLANE = (0.85, 0.20, 0.85)
_COLOR_LATCH_LOWER = (1.00, 0.55, 0.12)
_COLOR_LATCH_KEYPOINT = (0.15, 0.85, 0.25)


@dataclass(frozen=True)
class CrystalHeadFeatures:
    """Named crystal-head features in stage world coordinates."""

    name: str
    prim_path: str
    features: ConnectorFeatures
    meters_per_unit: float

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self.features)
        payload["name"] = self.name
        payload["prim_path"] = self.prim_path
        payload["meters_per_unit"] = float(self.meters_per_unit)
        for key, value in list(payload.items()):
            payload[key] = _jsonable(value)
        return payload


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return np.round(value, 6).tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def _world_transform(prim) -> np.ndarray:
    matrix = UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(prim)
    return np.asarray(matrix, dtype=np.float64).T


def _transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack(
        (points, np.ones(points.shape[0], dtype=np.float64))
    )
    return (transform @ homogeneous.T).T[:, :3]


def _world_mesh_arrays(prim) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mesh = UsdGeom.Mesh(prim)
    points_value = mesh.GetPointsAttr().Get()
    counts_value = mesh.GetFaceVertexCountsAttr().Get()
    indices_value = mesh.GetFaceVertexIndicesAttr().Get()
    if points_value is None or counts_value is None or indices_value is None:
        raise RuntimeError(f"Mesh is missing topology: {prim.GetPath()}")
    local_points = np.asarray(points_value, dtype=np.float64)
    counts = np.asarray(counts_value, dtype=np.int64)
    indices = np.asarray(indices_value, dtype=np.int64)
    world_points = _transform_points(_world_transform(prim), local_points)
    return world_points, counts, indices


def _merge_meshes(meshes: list[tuple[np.ndarray, np.ndarray, np.ndarray]]):
    points = []
    counts = []
    indices = []
    offset = 0
    for mesh_points, mesh_counts, mesh_indices in meshes:
        points.append(mesh_points)
        counts.append(mesh_counts)
        indices.append(mesh_indices + offset)
        offset += len(mesh_points)
    return np.vstack(points), np.concatenate(counts), np.concatenate(indices)


def _find_named_prims(root, prefix: str) -> list:
    found = []
    for prim in Usd.PrimRange(root):
        if prim.GetName().startswith(prefix):
            found.append(prim)
    return found


def _bbox_center(stage: Usd.Stage, prim) -> np.ndarray:
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        useExtentsHint=True,
    )
    box = cache.ComputeWorldBound(prim).ComputeAlignedBox()
    minimum = np.array(box.GetMin(), dtype=np.float64)
    maximum = np.array(box.GetMax(), dtype=np.float64)
    return 0.5 * (minimum + maximum)


def _cable_center(stage: Usd.Stage, root) -> np.ndarray:
    lines = _find_named_prims(root, LINE_PREFIX)
    if lines:
        return _bbox_center(stage, lines[0])
    meshes = [prim for prim in Usd.PrimRange(root) if prim.IsA(UsdGeom.Mesh)]
    heads = {str(head.GetPath()) for head in _find_named_prims(root, CRYSTAL_HEAD_PREFIX)}
    body_meshes = [
        prim
        for prim in meshes
        if not any(str(prim.GetPath()).startswith(head_path) for head_path in heads)
    ]
    if not body_meshes:
        return _bbox_center(stage, root)
    points = [_world_mesh_arrays(prim)[0] for prim in body_meshes]
    return np.vstack(points).mean(axis=0)


def _head_mesh_arrays(head) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    meshes = [prim for prim in Usd.PrimRange(head) if prim.IsA(UsdGeom.Mesh)]
    if not meshes:
        raise RuntimeError(f"Crystal head has no meshes: {head.GetPath()}")
    return _merge_meshes([_world_mesh_arrays(prim) for prim in meshes])


def reference_network_cable(
    stage: Usd.Stage,
    *,
    usd_path: Path | str = NETWORK_CABLE_USD,
    prim_path: str = NETWORK_CABLE_ROOT_PATH,
) -> str:
    """Reference the cable USD onto the live stage without editing the asset file."""

    usd = Path(usd_path).expanduser().resolve()
    if not usd.is_file():
        raise FileNotFoundError(f"Network cable USD is missing: {usd}")
    if stage.GetPrimAtPath(prim_path).IsValid():
        stage.RemovePrim(Sdf.Path(prim_path))
    prim = stage.DefinePrim(prim_path, "Xform")
    prim.GetReferences().AddReference(str(usd))
    if not stage.GetPrimAtPath(prim_path).IsValid():
        raise RuntimeError(f"Failed to reference network cable at {prim_path}")
    return prim_path


def extract_crystal_head_features(
    stage: Usd.Stage,
    cable_root_path: str = NETWORK_CABLE_ROOT_PATH,
) -> list[CrystalHeadFeatures]:
    """Extract mating/latch features for every crystal head under the cable root."""

    root = stage.GetPrimAtPath(cable_root_path)
    if not root or not root.IsValid():
        raise RuntimeError(f"Missing cable root: {cable_root_path}")
    heads = _find_named_prims(root, CRYSTAL_HEAD_PREFIX)
    if not heads:
        raise RuntimeError(f"No crystal heads under {cable_root_path}")
    cable_center = _cable_center(stage, root)
    meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage))
    results: list[CrystalHeadFeatures] = []
    for head in heads:
        points, counts, indices = _head_mesh_arrays(head)
        features = extract_connector_features_from_mesh(
            points=points,
            face_vertex_counts=counts,
            face_vertex_indices=indices,
            cable_center_world=cable_center,
        )
        results.append(
            CrystalHeadFeatures(
                name=head.GetName(),
                prim_path=str(head.GetPath()),
                features=features,
                meters_per_unit=meters_per_unit,
            )
        )
    results.sort(key=lambda item: item.name)
    return results


def local_features_from_world_heads(
    stage: Usd.Stage,
    heads: list[CrystalHeadFeatures],
) -> dict[str, ConnectorFeatures]:
    """Convert extracted world features into each crystal head's local frame."""

    local: dict[str, ConnectorFeatures] = {}
    for head in heads:
        prim = stage.GetPrimAtPath(head.prim_path)
        if not prim or not prim.IsValid():
            raise RuntimeError(f"Missing crystal head: {head.prim_path}")
        world_from_head = _world_transform(prim)
        local[head.name] = transform_connector_features(
            head.features,
            np.linalg.inv(world_from_head),
        )
    return local


def write_local_feature_cache(
    stage: Usd.Stage,
    heads: list[CrystalHeadFeatures],
    path: Path | str = DEFAULT_CACHE_PATH,
    *,
    source_usd: Path | str = NETWORK_CABLE_USD,
) -> Path:
    """Cache head-local features for later simulations to import as variables."""

    return save_local_cable_features(
        local_features_from_world_heads(stage, heads),
        path,
        source_usd=str(source_usd),
    )


def world_from_crystal_head(stage: Usd.Stage, head_prim_path: str) -> np.ndarray:
    """Return the 4x4 column-vector world pose of a crystal-head prim."""

    prim = stage.GetPrimAtPath(head_prim_path)
    if not prim or not prim.IsValid():
        raise RuntimeError(f"Missing crystal head: {head_prim_path}")
    return _world_transform(prim)


def world_features_from_crystal_head(
    stage: Usd.Stage,
    head_prim_path: str,
    *,
    cache_path: Path | str | None = None,
) -> CrystalHeadFeatures:
    """Load cached local features and re-pose them with the live head transform."""

    prim = stage.GetPrimAtPath(head_prim_path)
    if not prim or not prim.IsValid():
        raise RuntimeError(f"Missing crystal head: {head_prim_path}")
    meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage))
    world = world_features_for_head(
        prim.GetName(),
        _world_transform(prim),
        cache_path=cache_path,
        meters_per_unit=meters_per_unit,
    )
    return CrystalHeadFeatures(
        name=prim.GetName(),
        prim_path=str(prim.GetPath()),
        features=world,
        meters_per_unit=meters_per_unit,
    )


def _strip_physics(prim) -> None:
    try:
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False).Set(False)
    except Exception:
        pass
    try:
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            UsdPhysics.RigidBodyAPI(prim).CreateRigidBodyEnabledAttr(False).Set(False)
    except Exception:
        pass


def _set_color(prim, color_rgb: tuple[float, float, float]) -> None:
    imageable = UsdGeom.Gprim(prim)
    imageable.CreateDisplayColorAttr([Gf.Vec3f(*color_rgb)])


def _define_visible(stage: Usd.Stage, prim_path: str, type_name: str):
    if stage.GetPrimAtPath(prim_path).IsValid():
        stage.RemovePrim(Sdf.Path(prim_path))
    if type_name == "Sphere":
        geom = UsdGeom.Sphere.Define(stage, Sdf.Path(prim_path))
    elif type_name == "Cylinder":
        geom = UsdGeom.Cylinder.Define(stage, Sdf.Path(prim_path))
    elif type_name == "Cone":
        geom = UsdGeom.Cone.Define(stage, Sdf.Path(prim_path))
    elif type_name == "Mesh":
        geom = UsdGeom.Mesh.Define(stage, Sdf.Path(prim_path))
    else:
        raise ValueError(f"Unsupported debug primitive: {type_name}")
    UsdGeom.Imageable(geom.GetPrim()).MakeVisible()
    _strip_physics(geom.GetPrim())
    return geom


def _spawn_sphere(
    stage: Usd.Stage,
    prim_path: str,
    center: np.ndarray,
    radius: float,
    color_rgb: tuple[float, float, float],
) -> None:
    sphere = _define_visible(stage, prim_path, "Sphere")
    sphere.CreateRadiusAttr(float(radius))
    _set_color(sphere.GetPrim(), color_rgb)
    xform = UsdGeom.Xformable(sphere.GetPrim())
    xform.ClearXformOpOrder()
    xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(
        Gf.Vec3d(float(center[0]), float(center[1]), float(center[2]))
    )


def _rotation_from_z(direction: np.ndarray) -> Gf.Quatd:
    rotation = Gf.Rotation(Gf.Vec3d(0.0, 0.0, 1.0), Gf.Vec3d(*[float(v) for v in direction]))
    quat = rotation.GetQuat()
    imaginary = quat.GetImaginary()
    return Gf.Quatd(
        float(quat.GetReal()),
        Gf.Vec3d(float(imaginary[0]), float(imaginary[1]), float(imaginary[2])),
    )


def _spawn_cylinder(
    stage: Usd.Stage,
    prim_path: str,
    start: np.ndarray,
    end: np.ndarray,
    radius: float,
    color_rgb: tuple[float, float, float],
) -> None:
    start = np.asarray(start, dtype=np.float64).reshape(3)
    end = np.asarray(end, dtype=np.float64).reshape(3)
    delta = end - start
    length = float(np.linalg.norm(delta))
    if length <= 1.0e-9:
        raise ValueError("Cylinder length must be positive")
    axis = delta / length
    center = 0.5 * (start + end)
    cylinder = _define_visible(stage, prim_path, "Cylinder")
    cylinder.CreateRadiusAttr(float(radius))
    cylinder.CreateHeightAttr(float(length))
    cylinder.CreateAxisAttr("Z")
    _set_color(cylinder.GetPrim(), color_rgb)
    xform = UsdGeom.Xformable(cylinder.GetPrim())
    xform.ClearXformOpOrder()
    xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(
        Gf.Vec3d(float(center[0]), float(center[1]), float(center[2]))
    )
    xform.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(_rotation_from_z(axis))


def _spawn_cone(
    stage: Usd.Stage,
    prim_path: str,
    tip: np.ndarray,
    direction: np.ndarray,
    height: float,
    radius: float,
    color_rgb: tuple[float, float, float],
) -> None:
    axis = _unit_or_raise(direction)
    center = tip - 0.5 * height * axis
    cone = _define_visible(stage, prim_path, "Cone")
    cone.CreateRadiusAttr(float(radius))
    cone.CreateHeightAttr(float(height))
    cone.CreateAxisAttr("Z")
    _set_color(cone.GetPrim(), color_rgb)
    xform = UsdGeom.Xformable(cone.GetPrim())
    xform.ClearXformOpOrder()
    xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(
        Gf.Vec3d(float(center[0]), float(center[1]), float(center[2]))
    )
    xform.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(_rotation_from_z(axis))


def _unit_or_raise(direction: np.ndarray) -> np.ndarray:
    value = np.asarray(direction, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(value))
    if norm <= 1.0e-12:
        raise ValueError("direction must be nonzero")
    return value / norm


def _spawn_quad(
    stage: Usd.Stage,
    prim_path: str,
    corners: np.ndarray,
    normal: np.ndarray,
    color_rgb: tuple[float, float, float],
    *,
    offset: float,
) -> None:
    corners = np.asarray(corners, dtype=np.float64).reshape(4, 3)
    shifted = corners + float(offset) * np.asarray(normal, dtype=np.float64).reshape(3)
    mesh = _define_visible(stage, prim_path, "Mesh")
    mesh.CreatePointsAttr([Gf.Vec3f(*[float(v) for v in point]) for point in shifted])
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    mesh.CreateDoubleSidedAttr(True)
    _set_color(mesh.GetPrim(), color_rgb)


def spawn_cable_feature_markers(
    stage: Usd.Stage,
    heads: list[CrystalHeadFeatures],
    *,
    marker_root: str = DEBUG_MARKER_ROOT,
) -> str:
    """Spawn visible, non-colliding debug prims for extracted cable features."""

    if stage.GetPrimAtPath(marker_root).IsValid():
        stage.RemovePrim(Sdf.Path(marker_root))
    UsdGeom.Xform.Define(stage, Sdf.Path(marker_root))
    for head in heads:
        geom = head.features
        root = f"{marker_root}/{head.name}"
        UsdGeom.Xform.Define(stage, Sdf.Path(root))
        _spawn_quad(
            stage,
            f"{root}/MatingPlane",
            geom.mating_corners,
            geom.insertion_axis,
            _COLOR_MATING_PLANE,
            offset=_PLANE_OFFSET_M,
        )
        _spawn_sphere(
            stage,
            f"{root}/MatingCenter",
            geom.mating_center,
            _CENTER_RADIUS_M,
            _COLOR_MATING_CENTER,
        )
        axis_end = geom.mating_center + _AXIS_LENGTH_M * geom.insertion_axis
        shaft_end = geom.mating_center + 0.85 * _AXIS_LENGTH_M * geom.insertion_axis
        _spawn_cylinder(
            stage,
            f"{root}/InsertionAxis",
            geom.mating_center,
            shaft_end,
            _AXIS_RADIUS_M,
            _COLOR_INSERTION_AXIS,
        )
        _spawn_cone(
            stage,
            f"{root}/InsertionAxisTip",
            axis_end,
            geom.insertion_axis,
            0.15 * _AXIS_LENGTH_M,
            1.8 * _AXIS_RADIUS_M,
            _COLOR_INSERTION_AXIS,
        )
        _spawn_quad(
            stage,
            f"{root}/LatchPlane",
            geom.latch_corners,
            geom.insertion_axis,
            _COLOR_LATCH_PLANE,
            offset=_PLANE_OFFSET_M,
        )
        lower = geom.latch_corners[np.argsort(geom.latch_corners @ geom.up_axis)[:2]]
        lower = lower[np.argsort(lower @ geom.width_axis)]
        for index, corner in enumerate(lower):
            _spawn_sphere(
                stage,
                f"{root}/LatchLowerCorner{index}",
                corner,
                _LOWER_CORNER_RADIUS_M,
                _COLOR_LATCH_LOWER,
            )
        for index, corner in enumerate(geom.latch_keypoints):
            _spawn_sphere(
                stage,
                f"{root}/LatchKeypoint{index}",
                corner,
                _KEYPOINT_RADIUS_M,
                _COLOR_LATCH_KEYPOINT,
            )
    return marker_root


def format_feature_report(heads: list[CrystalHeadFeatures]) -> str:
    lines = [
        "[CABLE FEATURES] Crystal-head insertion geometry",
        "  cyan quad     = connector mating plane (housing front, latch excluded)",
        "  red sphere    = mating-face center",
        "  yellow arrow  = insertion axis (into the port, away from the cable)",
        "  magenta quad  = latch face parallel to the mating plane",
        "  orange spheres= lower latch-face corners",
        "  green spheres = latch keypoints (the two higher latch-face corners)",
    ]
    for head in heads:
        geom = head.features
        scale_mm = 1000.0 * head.meters_per_unit
        lines.extend(
            [
                f"  {head.name}  ({head.prim_path})",
                f"    insertion_axis     {np.round(geom.insertion_axis, 4).tolist()}",
                f"    mating_plane.point {np.round(geom.mating_plane.point * scale_mm, 3).tolist()} mm",
                f"    mating_center      {np.round(geom.mating_center * scale_mm, 3).tolist()} mm",
                f"    latch_keypoints    {np.round(geom.latch_keypoints * scale_mm, 3).tolist()} mm",
            ]
        )
    return "\n".join(lines)
