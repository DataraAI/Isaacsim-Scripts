#!/usr/bin/env python3
"""OpenUSD adapter for mesh-derived RJ45 insertion TCP calibration."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from pxr import Gf, Usd, UsdGeom

from cable_geometry import PlugFrame, validate_transform
from cable_mount import _world_transform
from connector_tcp import (
    InsertionTcpDerivation,
    MeshComponentBounds,
    connected_component_bounds,
    derive_insertion_tcp,
)


TCP_PROBE_ONLY = True
LEGACY_TCP_MARKER_PATH = "LegacyPlugTipProbe"
DERIVED_TCP_MARKER_PATH = "DerivedInsertionTcpProbe"
TCP_MARKER_RADIUS_M = 0.00125


def _transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    matrix = np.asarray(transform, dtype=np.float64).reshape(4, 4)
    values = np.asarray(points, dtype=np.float64)
    homogeneous = np.column_stack(
        (values, np.ones(values.shape[0], dtype=np.float64))
    )
    transformed = (matrix @ homogeneous.T).T
    return transformed[:, :3] / transformed[:, 3, None]


def mesh_components_in_plug_local(
    stage: Usd.Stage,
    tracked_plug_path: str,
) -> tuple[MeshComponentBounds, ...]:
    """Extract connected descendant mesh bounds in tracked-plug local space."""

    plug = stage.GetPrimAtPath(tracked_plug_path)
    if not plug.IsValid():
        raise RuntimeError(f"Tracked plug is invalid: {tracked_plug_path}")

    world_from_plug = _world_transform(stage, tracked_plug_path)
    plug_from_world = np.linalg.inv(world_from_plug)
    components: list[MeshComponentBounds] = []
    inspected: list[str] = []

    for prim in Usd.PrimRange(plug):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        points_value = mesh.GetPointsAttr().Get()
        counts_value = mesh.GetFaceVertexCountsAttr().Get()
        indices_value = mesh.GetFaceVertexIndicesAttr().Get()
        inspected.append(str(prim.GetPath()))
        if points_value is None or counts_value is None or indices_value is None:
            continue

        points = np.asarray(points_value, dtype=np.float64)
        counts = np.asarray(counts_value, dtype=np.int64)
        indices = np.asarray(indices_value, dtype=np.int64)
        if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] < 4:
            continue

        world_from_mesh = _world_transform(stage, str(prim.GetPath()))
        plug_from_mesh = plug_from_world @ world_from_mesh
        plug_points = _transform_points(plug_from_mesh, points)
        try:
            mesh_components = connected_component_bounds(
                points=plug_points,
                face_vertex_counts=counts,
                face_vertex_indices=indices,
                label_prefix=str(prim.GetPath()),
            )
        except (ValueError, RuntimeError):
            continue
        components.extend(mesh_components)

    if not components:
        raise RuntimeError(
            "No usable connected mesh components were found below the tracked plug. "
            f"Inspected meshes: {inspected}"
        )
    return tuple(components)


def derive_plug_frame_from_mesh(
    *,
    stage: Usd.Stage,
    tracked_plug_path: str,
    legacy_frame: PlugFrame,
    aperture_width_m: float,
    aperture_height_m: float,
) -> tuple[PlugFrame, InsertionTcpDerivation, tuple[MeshComponentBounds, ...]]:
    """Replace only the transverse legacy tip center using descendant geometry."""

    world_from_plug = _world_transform(stage, tracked_plug_path)
    axis_scale_m_per_local_unit = np.linalg.norm(
        world_from_plug[:3, :3],
        axis=0,
    )
    if (
        axis_scale_m_per_local_unit.shape != (3,)
        or not np.all(np.isfinite(axis_scale_m_per_local_unit))
        or np.any(axis_scale_m_per_local_unit <= 1.0e-12)
    ):
        raise RuntimeError(
            "Tracked plug has invalid physical axis scale: "
            f"{axis_scale_m_per_local_unit}"
        )

    components = mesh_components_in_plug_local(stage, tracked_plug_path)
    derivation = derive_insertion_tcp(
        legacy_tip_local=legacy_frame.tip_local_m,
        longitudinal_axis_index=legacy_frame.longitudinal_axis_index,
        nose_axis_local=legacy_frame.nose_axis_local,
        axis_scale_m_per_local_unit=axis_scale_m_per_local_unit,
        components=components,
        aperture_width_m=aperture_width_m,
        aperture_height_m=aperture_height_m,
    )

    plug_from_tip = np.asarray(
        legacy_frame.plug_from_tip,
        dtype=np.float64,
    ).copy()
    plug_from_tip[:3, 3] = derivation.tip_local
    validate_transform(plug_from_tip, "mesh_derived_plug_from_tip")
    frame = replace(
        legacy_frame,
        tip_local_m=derivation.tip_local.copy(),
        plug_from_tip=plug_from_tip,
    )
    return frame, derivation, components


def _define_hand_marker(
    *,
    stage: Usd.Stage,
    hand_path: str,
    name: str,
    hand_local_position: np.ndarray,
    color: tuple[float, float, float],
) -> str:
    path = f"{hand_path}/{name}"
    sphere = UsdGeom.Sphere.Define(stage, path)
    sphere.CreateRadiusAttr(float(TCP_MARKER_RADIUS_M))
    sphere.CreateDisplayColorAttr(
        [Gf.Vec3f(float(color[0]), float(color[1]), float(color[2]))]
    )
    xform = UsdGeom.XformCommonAPI(sphere.GetPrim())
    position = np.asarray(hand_local_position, dtype=np.float64).reshape(3)
    xform.SetTranslate(
        Gf.Vec3d(
            float(position[0]),
            float(position[1]),
            float(position[2]),
        )
    )
    return path


def author_tcp_probe_markers(
    *,
    stage: Usd.Stage,
    hand_path: str,
    tracked_plug_path: str,
    derivation: InsertionTcpDerivation,
) -> tuple[str, str]:
    """Attach legacy and derived TCP markers to the unscaled Franka hand."""

    world_from_hand = _world_transform(stage, hand_path)
    hand_from_world = np.linalg.inv(world_from_hand)
    world_from_plug = _world_transform(stage, tracked_plug_path)

    legacy_world = _transform_points(
        world_from_plug,
        derivation.legacy_tip_local.reshape(1, 3),
    )[0]
    derived_world = _transform_points(
        world_from_plug,
        derivation.tip_local.reshape(1, 3),
    )[0]
    legacy_hand = _transform_points(
        hand_from_world,
        legacy_world.reshape(1, 3),
    )[0]
    derived_hand = _transform_points(
        hand_from_world,
        derived_world.reshape(1, 3),
    )[0]

    legacy_path = _define_hand_marker(
        stage=stage,
        hand_path=hand_path,
        name=LEGACY_TCP_MARKER_PATH,
        hand_local_position=legacy_hand,
        color=(1.0, 0.1, 0.1),
    )
    derived_path = _define_hand_marker(
        stage=stage,
        hand_path=hand_path,
        name=DERIVED_TCP_MARKER_PATH,
        hand_local_position=derived_hand,
        color=(0.0, 1.0, 1.0),
    )
    return legacy_path, derived_path


def log_tcp_derivation(
    derivation: InsertionTcpDerivation,
    component_count: int,
    legacy_marker_path: str,
    derived_marker_path: str,
) -> None:
    shift_mm = np.asarray(derivation.shift_physical_m) * 1000.0
    cross_section_mm = np.asarray(derivation.cross_section_m) * 1000.0
    print(
        "[CONNECTOR TCP] MESH-DERIVED INSERTION CENTER\n"
        f"  selected component: {derivation.selected_label}\n"
        f"  connected components inspected: {component_count}\n"
        f"  qualified body candidates: {derivation.candidate_count}\n"
        f"  body cross-section mm: "
        f"{np.round(cross_section_mm, 3).tolist()}\n"
        f"  legacy tip local: "
        f"{np.round(derivation.legacy_tip_local, 6).tolist()}\n"
        f"  derived tip local: "
        f"{np.round(derivation.tip_local, 6).tolist()}\n"
        f"  physical TCP shift mm: {np.round(shift_mm, 3).tolist()}\n"
        f"  nose gap mm: {derivation.nose_gap_m * 1000.0:.3f}\n"
        f"  legacy marker: {legacy_marker_path} (red)\n"
        f"  derived marker: {derived_marker_path} (cyan)\n"
        "  port perception and marker positions: unchanged\n"
        "  insertion motion: locked for this probe run",
        flush=True,
    )
