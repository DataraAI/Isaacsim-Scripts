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
_MAXIMUM_PROFILE_SETBACK_M = 0.020


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
        whole_minimum = np.min(plug_points, axis=0)
        whole_maximum = np.max(plug_points, axis=0)
        if np.all(whole_maximum > whole_minimum):
            components.append(
                MeshComponentBounds(
                    label=f"{prim.GetPath()}#whole",
                    local_min=whole_minimum,
                    local_max=whole_maximum,
                    vertex_count=plug_points.shape[0],
                )
            )
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


def _component_rejection_report(
    *,
    components: tuple[MeshComponentBounds, ...],
    legacy_frame: PlugFrame,
    axis_scale_m_per_local_unit: np.ndarray,
    aperture_width_m: float,
    aperture_height_m: float,
) -> str:
    """Explain exactly why each real USD component passed or failed the gate."""

    scale = np.asarray(axis_scale_m_per_local_unit, dtype=np.float64).reshape(3)
    longitudinal = int(legacy_frame.longitudinal_axis_index)
    transverse = [axis for axis in range(3) if axis != longitudinal]
    nose_sign = float(legacy_frame.nose_axis_local[longitudinal])
    target_mm = np.sort(
        np.array([aperture_width_m, aperture_height_m], dtype=np.float64)
        * 1000.0
    )

    records: list[tuple[float, float, str]] = []
    for component in components:
        component_nose = (
            component.local_max[longitudinal]
            if nose_sign > 0.0
            else component.local_min[longitudinal]
        )
        profile_setback_mm = (
            nose_sign
            * (legacy_frame.tip_local_m[longitudinal] - component_nose)
            * scale[longitudinal]
            * 1000.0
        )
        physical_extent_mm = (
            (component.local_max - component.local_min) * scale * 1000.0
        )
        cross_section_mm = physical_extent_mm[transverse]
        sorted_cross_mm = np.sort(cross_section_mm)
        ratios = sorted_cross_mm / target_mm

        reasons: list[str] = []
        if profile_setback_mm < -1.0e-6:
            reasons.append(
                f"profile extends {-profile_setback_mm:.3f} mm ahead of nose"
            )
        if profile_setback_mm > _MAXIMUM_PROFILE_SETBACK_M * 1000.0:
            reasons.append(
                f"profile setback {profile_setback_mm:.3f} > "
                f"{_MAXIMUM_PROFILE_SETBACK_M * 1000.0:.3f} mm"
            )
        if np.any(ratios < 0.55) or np.any(ratios > 1.25):
            reasons.append("cross-section ratios outside [0.55, 1.25]")
        status = "QUALIFIED" if not reasons else "; ".join(reasons)
        score_hint = float(
            np.sum(np.abs(np.log(np.maximum(ratios, 1.0e-12))))
        )
        line = (
            f"  {component.label}\n"
            f"    vertices={component.vertex_count} "
            f"physical_extent_mm={np.round(physical_extent_mm, 3).tolist()}\n"
            f"    transverse_mm={np.round(cross_section_mm, 3).tolist()} "
            f"sorted_ratios={np.round(ratios, 3).tolist()} "
            f"profile_setback_mm={profile_setback_mm:.3f}\n"
            f"    result={status}"
        )
        records.append((abs(profile_setback_mm), score_hint, line))

    records.sort(key=lambda item: (item[0], item[1], item[2]))
    displayed = records[:40]
    lines = [
        "[CONNECTOR TCP] REAL USD COMPONENT DIAGNOSTICS",
        f"  port target cross-section mm: {np.round(target_mm, 3).tolist()}",
        f"  longitudinal axis: {longitudinal}",
        f"  transverse axes: {transverse}",
        f"  maximum profile setback mm: "
        f"{_MAXIMUM_PROFILE_SETBACK_M * 1000.0:.3f}",
        f"  axis scale m/local-unit: {np.round(scale, 9).tolist()}",
        f"  components inspected: {len(records)}",
        f"  components displayed: {len(displayed)}",
    ]
    lines.extend(item[2] for item in displayed)
    if len(records) > len(displayed):
        lines.append(
            f"  ... {len(records) - len(displayed)} additional components omitted"
        )
    return "\n".join(lines)


def _legacy_probe_derivation(
    *,
    legacy_frame: PlugFrame,
    axis_scale_m_per_local_unit: np.ndarray,
) -> InsertionTcpDerivation:
    """Represent a rejected probe without changing the connector frame."""

    scale = np.asarray(axis_scale_m_per_local_unit, dtype=np.float64).reshape(3)
    longitudinal = int(legacy_frame.longitudinal_axis_index)
    transverse = [axis for axis in range(3) if axis != longitudinal]
    local_extent = legacy_frame.local_max_m - legacy_frame.local_min_m
    return InsertionTcpDerivation(
        tip_local=np.asarray(legacy_frame.tip_local_m, dtype=np.float64).copy(),
        legacy_tip_local=np.asarray(
            legacy_frame.tip_local_m,
            dtype=np.float64,
        ).copy(),
        shift_physical_m=np.zeros(3, dtype=np.float64),
        selected_label="UNQUALIFIED_LEGACY_FRAME_RETAINED",
        selected_local_min=np.asarray(
            legacy_frame.local_min_m,
            dtype=np.float64,
        ).copy(),
        selected_local_max=np.asarray(
            legacy_frame.local_max_m,
            dtype=np.float64,
        ).copy(),
        cross_section_m=local_extent[transverse] * scale[transverse],
        nose_gap_m=0.0,
        score=float("inf"),
        candidate_count=0,
    )


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
    try:
        derivation = derive_insertion_tcp(
            legacy_tip_local=legacy_frame.tip_local_m,
            longitudinal_axis_index=legacy_frame.longitudinal_axis_index,
            nose_axis_local=legacy_frame.nose_axis_local,
            axis_scale_m_per_local_unit=axis_scale_m_per_local_unit,
            components=components,
            aperture_width_m=aperture_width_m,
            aperture_height_m=aperture_height_m,
            maximum_profile_setback_m=0.020,
        )
    except RuntimeError as error:
        print(
            f"[CONNECTOR TCP] DERIVATION REJECTED\n  reason: {error}\n"
            + _component_rejection_report(
                components=components,
                legacy_frame=legacy_frame,
                axis_scale_m_per_local_unit=axis_scale_m_per_local_unit,
                aperture_width_m=aperture_width_m,
                aperture_height_m=aperture_height_m,
            )
            + "\n  action: legacy connector frame retained for probe only\n"
            "  visual servo and insertion remain locked",
            flush=True,
        )
        return (
            legacy_frame,
            _legacy_probe_derivation(
                legacy_frame=legacy_frame,
                axis_scale_m_per_local_unit=axis_scale_m_per_local_unit,
            ),
            components,
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
    if derivation.candidate_count == 0:
        print(
            "[CONNECTOR TCP] PROBE ACTIVE — NO DERIVED TCP ACCEPTED\n"
            f"  connected components inspected: {component_count}\n"
            "  connector frame: legacy full-bounds frame retained\n"
            f"  legacy marker: {legacy_marker_path} (red)\n"
            f"  derived marker: {derived_marker_path} (cyan, overlaps red)\n"
            f"  full-bounds transverse cross-section mm: "
            f"{np.round(cross_section_mm, 3).tolist()}\n"
            "  port perception and marker positions: unchanged\n"
            "  YOLOE, visual servo, handoff, and insertion: locked",
            flush=True,
        )
        return

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
        f"  profile setback mm: {derivation.nose_gap_m * 1000.0:.3f}\n"
        f"  legacy marker: {legacy_marker_path} (red)\n"
        f"  derived marker: {derived_marker_path} (cyan)\n"
        "  port perception and marker positions: unchanged\n"
        "  insertion motion: locked for this probe run",
        flush=True,
    )
