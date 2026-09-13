"""USD adapter and Isaac Sim debug markers for RJ45 jack insertion features.

Opens DataHall (or any stage) read-only. Debug prims are authored on the live
session only; the original USD is never written.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pxr import Sdf, Usd, UsdGeom

from insertion_features.cache import (
    DEFAULT_PORT_CACHE_PATH,
    length_scale_transform,
    meters_transform_from_stage,
    save_local_port_features,
    world_features_for_jack,
)
from insertion_features.cable_features import (
    _spawn_cone,
    _spawn_cylinder,
    _spawn_quad,
    _spawn_sphere,
)
from insertion_features.geometry import (
    ConnectorFeatures,
    transform_connector_features,
)
from insertion_features.port_geometry import (
    extract_port_features_from_pack_mesh,
    infer_port_insertion_axis,
)

DATAHALL_USD = Path.home() / "Desktop" / "Aayush_ws" / "DataHall_6r_ur5e.usd"
EXAMPLE_PACK_PATH = (
    "/World/Network_Switches/AS4610_Ethernet_Row_Middle_1x_Grid/Upper_Left/"
    "AS4610_inst/AS4610_01/Switch/Net_12_Pack_no_LED_Component_04/RJ45_Group01"
)
INNER_MESH_NAME = "Mesh133"
OUTER_MESH_NAME = "Mesh134"
DEBUG_MARKER_ROOT = "/World/PortFeatureDebug"
WORLD_UP = np.array([0.0, 0.0, 1.0], dtype=np.float64)

_AXIS_LENGTH_M = 0.020
_AXIS_RADIUS_M = 0.00035
_CENTER_RADIUS_M = 0.0010
_KEYPOINT_RADIUS_M = 0.00090
_LOWER_CORNER_RADIUS_M = 0.00065
_PLANE_OFFSET_M = 0.00015

_COLOR_MATING_PLANE = (0.10, 0.82, 0.92)
_COLOR_MATING_CENTER = (0.95, 0.12, 0.12)
_COLOR_INSERTION_AXIS = (1.00, 0.92, 0.12)
_COLOR_LATCH_PLANE = (0.85, 0.20, 0.85)
_COLOR_LATCH_LOWER = (1.00, 0.55, 0.12)
_COLOR_LATCH_KEYPOINT = (0.15, 0.85, 0.25)


@dataclass(frozen=True)
class JackFeatures:
    """Named RJ45 jack features in metres, world coordinates."""

    name: str
    pack_prim_path: str
    row: str
    col: int
    copper_group: str
    features: ConnectorFeatures
    meters_per_unit: float


def _world_transform(prim) -> np.ndarray:
    matrix = UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(prim)
    return np.asarray(matrix, dtype=np.float64).T


def _transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack((points, np.ones(points.shape[0], dtype=np.float64)))
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


def _child_mesh(pack, name: str):
    prim = pack.GetChild(name)
    if prim and prim.IsValid() and prim.IsA(UsdGeom.Mesh):
        return prim
    for child in Usd.PrimRange(pack):
        if child.GetName() == name and child.IsA(UsdGeom.Mesh):
            return child
    raise RuntimeError(f"Missing {name} under {pack.GetPath()}")


def _bbox_center_m(stage: Usd.Stage, prim, meters_per_unit: float) -> np.ndarray:
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        useExtentsHint=True,
    )
    box = cache.ComputeWorldBound(prim).ComputeAlignedBox()
    minimum = np.array(box.GetMin(), dtype=np.float64) * meters_per_unit
    maximum = np.array(box.GetMax(), dtype=np.float64) * meters_per_unit
    return 0.5 * (minimum + maximum)


def _copper_centers(stage: Usd.Stage, pack, meters_per_unit: float) -> list[tuple[str, np.ndarray]]:
    centers = []
    for prim in Usd.PrimRange(pack):
        if not prim.GetName().startswith("Group_"):
            continue
        if prim.GetTypeName() != "Xform":
            continue
        has_mesh = any(child.IsA(UsdGeom.Mesh) for child in Usd.PrimRange(prim))
        if not has_mesh:
            continue
        centers.append((prim.GetName(), _bbox_center_m(stage, prim, meters_per_unit)))
    return centers


def _nearest_copper(center: np.ndarray, coppers: list[tuple[str, np.ndarray]]) -> str:
    if not coppers:
        return ""
    plane = center[1:3]
    return min(coppers, key=lambda item: float(np.linalg.norm(item[1][1:3] - plane)))[0]


def _name_jacks(ports: list[ConnectorFeatures]) -> list[tuple[str, str, int]]:
    labeled = []
    for port in ports:
        row = "upper" if float(np.dot(port.up_axis, WORLD_UP)) >= 0.0 else "lower"
        labeled.append((row, port))
    named: list[tuple[str, str, int]] = []
    for row in ("upper", "lower"):
        row_ports = [(index, port) for index, (label, port) in enumerate(labeled) if label == row]
        row_ports.sort(key=lambda item: float(item[1].mating_center[1]))
        for col, (index, _port) in enumerate(row_ports):
            named.append((index, f"jack_{row}_c{col}", row, col))
    named.sort(key=lambda item: item[0])
    return [(name, row, col) for _index, name, row, col in named]


def extract_rj45_group_features(
    stage: Usd.Stage,
    pack_prim_path: str = EXAMPLE_PACK_PATH,
) -> list[JackFeatures]:
    """Extract per-jack mating/latch features from one 12-pack RJ45 group."""

    pack = stage.GetPrimAtPath(pack_prim_path)
    if not pack or not pack.IsValid():
        raise RuntimeError(f"Missing RJ45 group: {pack_prim_path}")
    meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage))
    inner_points, inner_counts, inner_indices = _world_mesh_arrays(_child_mesh(pack, INNER_MESH_NAME))
    outer_points, _outer_counts, _outer_indices = _world_mesh_arrays(_child_mesh(pack, OUTER_MESH_NAME))
    inner_m = inner_points * meters_per_unit
    outer_m = outer_points * meters_per_unit
    insertion = infer_port_insertion_axis(inner_points=inner_m, outer_points=outer_m)
    ports = extract_port_features_from_pack_mesh(
        points=inner_m,
        face_vertex_counts=inner_counts,
        face_vertex_indices=inner_indices,
        insertion_axis=insertion,
    )
    names = _name_jacks(ports)
    coppers = _copper_centers(stage, pack, meters_per_unit)
    results = []
    for port, (name, row, col) in zip(ports, names):
        results.append(
            JackFeatures(
                name=name,
                pack_prim_path=str(pack.GetPath()),
                row=row,
                col=col,
                copper_group=_nearest_copper(port.mating_center, coppers),
                features=port,
                meters_per_unit=meters_per_unit,
            )
        )
    results.sort(key=lambda item: (0 if item.row == "upper" else 1, item.col))
    return results


def local_features_from_world_jacks(
    stage: Usd.Stage,
    jacks: list[JackFeatures],
) -> tuple[dict[str, ConnectorFeatures], dict[str, dict]]:
    """Convert extracted world-metre features into the pack prim's local frame."""

    local: dict[str, ConnectorFeatures] = {}
    metadata: dict[str, dict] = {}
    for jack in jacks:
        pack = stage.GetPrimAtPath(jack.pack_prim_path)
        if not pack or not pack.IsValid():
            raise RuntimeError(f"Missing RJ45 group: {jack.pack_prim_path}")
        world_from_pack_m = meters_transform_from_stage(
            _world_transform(pack),
            jack.meters_per_unit,
        )
        local[jack.name] = transform_connector_features(
            jack.features,
            np.linalg.inv(world_from_pack_m),
        )
        metadata[jack.name] = {
            "row": jack.row,
            "col": jack.col,
            "copper_group": jack.copper_group,
        }
    return local, metadata


def write_local_port_feature_cache(
    stage: Usd.Stage,
    jacks: list[JackFeatures],
    path: Path | str = DEFAULT_PORT_CACHE_PATH,
    *,
    source_usd: Path | str = DATAHALL_USD,
) -> Path:
    """Cache pack-local jack features for later simulations."""

    local, metadata = local_features_from_world_jacks(stage, jacks)
    pack_path = jacks[0].pack_prim_path if jacks else ""
    return save_local_port_features(
        local,
        path,
        source_usd=str(source_usd),
        pack_prim_path=pack_path,
        metadata=metadata,
    )


def world_from_rj45_group(stage: Usd.Stage, pack_prim_path: str) -> np.ndarray:
    """Return the 4x4 column-vector world pose of an RJ45 group prim (stage units)."""

    prim = stage.GetPrimAtPath(pack_prim_path)
    if not prim or not prim.IsValid():
        raise RuntimeError(f"Missing RJ45 group: {pack_prim_path}")
    return _world_transform(prim)


def world_features_from_rj45_group(
    stage: Usd.Stage,
    jack_name: str,
    pack_prim_path: str = EXAMPLE_PACK_PATH,
    *,
    cache_path: Path | str | None = None,
) -> JackFeatures:
    """Load cached pack-local features and re-pose them with the live group transform."""

    prim = stage.GetPrimAtPath(pack_prim_path)
    if not prim or not prim.IsValid():
        raise RuntimeError(f"Missing RJ45 group: {pack_prim_path}")
    meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage))
    world = world_features_for_jack(
        jack_name,
        _world_transform(prim),
        cache_path=cache_path,
        meters_per_unit=meters_per_unit,
    )
    return JackFeatures(
        name=jack_name,
        pack_prim_path=str(prim.GetPath()),
        row="upper" if jack_name.startswith("jack_upper") else "lower",
        col=-1,
        copper_group="",
        features=world,
        meters_per_unit=meters_per_unit,
    )


def spawn_port_feature_markers(
    stage: Usd.Stage,
    jacks: list[JackFeatures],
    *,
    marker_root: str = DEBUG_MARKER_ROOT,
) -> str:
    """Spawn visible, non-colliding debug prims for extracted jack features."""

    if stage.GetPrimAtPath(marker_root).IsValid():
        stage.RemovePrim(Sdf.Path(marker_root))
    UsdGeom.Xform.Define(stage, Sdf.Path(marker_root))
    meters_per_unit = (
        float(jacks[0].meters_per_unit) if jacks else float(UsdGeom.GetStageMetersPerUnit(stage))
    )
    to_stage = length_scale_transform(1.0 / meters_per_unit)
    size = 1.0 / meters_per_unit
    for jack in jacks:
        geom = transform_connector_features(jack.features, to_stage)
        root = f"{marker_root}/{jack.name}"
        UsdGeom.Xform.Define(stage, Sdf.Path(root))
        _spawn_quad(
            stage,
            f"{root}/MatingPlane",
            geom.mating_corners,
            geom.insertion_axis,
            _COLOR_MATING_PLANE,
            offset=_PLANE_OFFSET_M * size,
        )
        _spawn_sphere(
            stage,
            f"{root}/MatingCenter",
            geom.mating_center,
            _CENTER_RADIUS_M * size,
            _COLOR_MATING_CENTER,
        )
        axis_end = geom.mating_center + _AXIS_LENGTH_M * size * geom.insertion_axis
        shaft_end = geom.mating_center + 0.85 * _AXIS_LENGTH_M * size * geom.insertion_axis
        _spawn_cylinder(
            stage,
            f"{root}/InsertionAxis",
            geom.mating_center,
            shaft_end,
            _AXIS_RADIUS_M * size,
            _COLOR_INSERTION_AXIS,
        )
        _spawn_cone(
            stage,
            f"{root}/InsertionAxisTip",
            axis_end,
            geom.insertion_axis,
            0.15 * _AXIS_LENGTH_M * size,
            1.8 * _AXIS_RADIUS_M * size,
            _COLOR_INSERTION_AXIS,
        )
        _spawn_quad(
            stage,
            f"{root}/LatchPlane",
            geom.latch_corners,
            geom.insertion_axis,
            _COLOR_LATCH_PLANE,
            offset=_PLANE_OFFSET_M * size,
        )
        lower = geom.latch_corners[np.argsort(geom.latch_corners @ geom.up_axis)[:2]]
        lower = lower[np.argsort(lower @ geom.width_axis)]
        for index, corner in enumerate(lower):
            _spawn_sphere(
                stage,
                f"{root}/LatchLowerCorner{index}",
                corner,
                _LOWER_CORNER_RADIUS_M * size,
                _COLOR_LATCH_LOWER,
            )
        for index, corner in enumerate(geom.latch_keypoints):
            _spawn_sphere(
                stage,
                f"{root}/LatchKeypoint{index}",
                corner,
                _KEYPOINT_RADIUS_M * size,
                _COLOR_LATCH_KEYPOINT,
            )
    return marker_root


def format_port_feature_report(jacks: list[JackFeatures]) -> str:
    lines = [
        "[PORT FEATURES] RJ45 jack insertion geometry (Mesh133 opening silhouette)",
        "  cyan quad     = port mating plane (big entry rectangle, not the full stepped outline)",
        "  red sphere    = opening-face center",
        "  yellow arrow  = insertion axis (into the jack)",
        "  magenta quad  = latch-channel rectangle on top of the big opening",
        "  orange spheres= latch-channel corners on the mating-plane side",
        "  green spheres = latch keypoints (top two corners of that smaller rectangle)",
    ]
    for jack in jacks:
        geom = jack.features
        lines.extend(
            [
                f"  {jack.name}  row={jack.row} col={jack.col} copper={jack.copper_group}",
                f"    insertion_axis     {np.round(geom.insertion_axis, 4).tolist()}",
                f"    mating_center      {np.round(geom.mating_center * 1000.0, 3).tolist()} mm",
                f"    latch_keypoints    {np.round(geom.latch_keypoints * 1000.0, 3).tolist()} mm",
            ]
        )
    return "\n".join(lines)
