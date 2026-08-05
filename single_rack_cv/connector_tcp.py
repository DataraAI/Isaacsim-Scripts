#!/usr/bin/env python3
"""Derive a functional connector insertion TCP from nose-reaching mesh bounds."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class MeshComponentBounds:
    """One connected mesh component expressed in tracked-plug local coordinates."""

    label: str
    local_min: np.ndarray
    local_max: np.ndarray
    vertex_count: int

    def __post_init__(self) -> None:
        minimum = np.asarray(self.local_min, dtype=np.float64).reshape(3)
        maximum = np.asarray(self.local_max, dtype=np.float64).reshape(3)
        if not np.all(np.isfinite(minimum)) or not np.all(np.isfinite(maximum)):
            raise ValueError("Mesh component bounds must be finite.")
        if np.any(maximum <= minimum):
            raise ValueError("Mesh component bounds must have positive extent.")
        if int(self.vertex_count) < 4:
            raise ValueError("Mesh component must contain at least four vertices.")
        object.__setattr__(self, "local_min", minimum.copy())
        object.__setattr__(self, "local_max", maximum.copy())
        object.__setattr__(self, "vertex_count", int(self.vertex_count))


def connected_component_bounds(
    *,
    points: np.ndarray,
    face_vertex_counts: np.ndarray,
    face_vertex_indices: np.ndarray,
    label_prefix: str,
) -> tuple[MeshComponentBounds, ...]:
    """Return one axis-aligned bound per disconnected face component."""

    values = np.asarray(points, dtype=np.float64)
    counts = np.asarray(face_vertex_counts, dtype=np.int64).reshape(-1)
    indices = np.asarray(face_vertex_indices, dtype=np.int64).reshape(-1)
    if values.ndim != 2 or values.shape[1] != 3 or values.shape[0] < 4:
        raise ValueError("points must have shape (N, 3) with N >= 4.")
    if not np.all(np.isfinite(values)):
        raise ValueError("points must be finite.")
    if counts.size == 0 or np.any(counts < 3) or int(np.sum(counts)) != indices.size:
        raise ValueError("face topology is invalid.")
    if np.any(indices < 0) or np.any(indices >= values.shape[0]):
        raise ValueError("face indices are out of range.")

    parent = np.arange(values.shape[0], dtype=np.int64)

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return index

    def union(first: int, second: int) -> None:
        root_first = find(first)
        root_second = find(second)
        if root_first != root_second:
            parent[root_second] = root_first

    cursor = 0
    used = np.zeros(values.shape[0], dtype=bool)
    for count in counts:
        face = indices[cursor : cursor + int(count)]
        cursor += int(count)
        used[face] = True
        anchor = int(face[0])
        for vertex in face[1:]:
            union(anchor, int(vertex))

    groups: dict[int, list[int]] = {}
    for index in np.flatnonzero(used):
        groups.setdefault(find(int(index)), []).append(int(index))

    components: list[MeshComponentBounds] = []
    for component_index, vertex_indices in enumerate(
        sorted(groups.values(), key=lambda group: min(group))
    ):
        if len(vertex_indices) < 4:
            continue
        component_points = values[np.asarray(vertex_indices, dtype=np.int64)]
        components.append(
            MeshComponentBounds(
                label=f"{label_prefix}#{component_index}",
                local_min=np.min(component_points, axis=0),
                local_max=np.max(component_points, axis=0),
                vertex_count=len(vertex_indices),
            )
        )
    if not components:
        raise RuntimeError("Mesh topology produced no usable connected components.")
    return tuple(components)


@dataclass(frozen=True)
class InsertionTcpDerivation:
    """A mesh-derived tip center that preserves the legacy nose depth and axes."""

    tip_local: np.ndarray
    legacy_tip_local: np.ndarray
    shift_physical_m: np.ndarray
    selected_label: str
    selected_local_min: np.ndarray
    selected_local_max: np.ndarray
    cross_section_m: np.ndarray
    nose_gap_m: float
    score: float
    candidate_count: int


@dataclass(frozen=True)
class _Candidate:
    component: MeshComponentBounds
    center_local: np.ndarray
    cross_section_m: np.ndarray
    nose_gap_m: float
    score: float


def _finite_vector(value, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label} must be finite with shape (3,).")
    return array


def _positive(value: float, label: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{label} must be finite and positive.")
    return number


def _dimension_score(dimensions: np.ndarray, aperture: np.ndarray) -> float:
    values = np.sort(np.asarray(dimensions, dtype=np.float64))
    target = np.sort(np.asarray(aperture, dtype=np.float64))
    ratios = values / target
    if np.any(ratios < 0.55) or np.any(ratios > 1.25):
        return float("inf")
    return float(np.sum(np.abs(np.log(ratios))))


def derive_insertion_tcp(
    *,
    legacy_tip_local: np.ndarray,
    longitudinal_axis_index: int,
    nose_axis_local: np.ndarray,
    axis_scale_m_per_local_unit: np.ndarray,
    components: Iterable[MeshComponentBounds],
    aperture_width_m: float,
    aperture_height_m: float,
    nose_reach_tolerance_m: float = 0.0015,
    ambiguity_score_margin: float = 0.08,
    ambiguity_center_tolerance_m: float = 0.00035,
    maximum_transverse_shift_m: float = 0.0030,
) -> InsertionTcpDerivation:
    """Select the insertable nose body and derive its transverse center.

    The longitudinal tip coordinate and orientation are preserved. Only the two
    transverse coordinates are replaced by the center of a nose-reaching mesh
    component whose physical cross-section plausibly fits the measured port.
    """

    legacy = _finite_vector(legacy_tip_local, "legacy_tip_local")
    scale = _finite_vector(
        axis_scale_m_per_local_unit,
        "axis_scale_m_per_local_unit",
    )
    if np.any(scale <= 0.0):
        raise ValueError("axis_scale_m_per_local_unit must be positive.")
    nose_axis = _finite_vector(nose_axis_local, "nose_axis_local")
    longitudinal = int(longitudinal_axis_index)
    if longitudinal not in (0, 1, 2):
        raise ValueError("longitudinal_axis_index must be 0, 1, or 2.")
    nose_sign = float(nose_axis[longitudinal])
    if not math.isclose(abs(nose_sign), 1.0, abs_tol=1.0e-9):
        raise ValueError("nose_axis_local must align with the longitudinal axis.")
    if np.count_nonzero(np.abs(nose_axis) > 1.0e-9) != 1:
        raise ValueError("nose_axis_local must contain exactly one nonzero axis.")

    aperture = np.array(
        [
            _positive(aperture_width_m, "aperture_width_m"),
            _positive(aperture_height_m, "aperture_height_m"),
        ],
        dtype=np.float64,
    )
    nose_tolerance = _positive(
        nose_reach_tolerance_m,
        "nose_reach_tolerance_m",
    )
    score_margin = _positive(ambiguity_score_margin, "ambiguity_score_margin")
    center_tolerance = _positive(
        ambiguity_center_tolerance_m,
        "ambiguity_center_tolerance_m",
    )
    maximum_shift = _positive(
        maximum_transverse_shift_m,
        "maximum_transverse_shift_m",
    )
    transverse = [axis for axis in range(3) if axis != longitudinal]

    candidates: list[_Candidate] = []
    for component in components:
        if not isinstance(component, MeshComponentBounds):
            raise TypeError("components must contain MeshComponentBounds values.")
        component_nose = (
            component.local_max[longitudinal]
            if nose_sign > 0.0
            else component.local_min[longitudinal]
        )
        nose_gap_m = (
            abs(component_nose - legacy[longitudinal]) * scale[longitudinal]
        )
        if nose_gap_m > nose_tolerance:
            continue

        local_extent = component.local_max - component.local_min
        cross_section_m = local_extent[transverse] * scale[transverse]
        shape_score = _dimension_score(cross_section_m, aperture)
        if not math.isfinite(shape_score):
            continue

        center_local = legacy.copy()
        component_center = 0.5 * (component.local_min + component.local_max)
        center_local[transverse] = component_center[transverse]
        score = shape_score + 20.0 * nose_gap_m
        candidates.append(
            _Candidate(
                component=component,
                center_local=center_local,
                cross_section_m=cross_section_m,
                nose_gap_m=float(nose_gap_m),
                score=float(score),
            )
        )

    if not candidates:
        raise RuntimeError(
            "No qualified nose-reaching connector body matched the port cross-section."
        )

    candidates.sort(
        key=lambda item: (
            item.score,
            -item.component.vertex_count,
            item.component.label,
        )
    )
    selected = candidates[0]
    if len(candidates) > 1:
        runner_up = candidates[1]
        if runner_up.score - selected.score <= score_margin:
            separation_m = float(
                np.linalg.norm(
                    (runner_up.center_local - selected.center_local) * scale
                )
            )
            if separation_m > center_tolerance:
                raise RuntimeError(
                    "Mesh-derived connector insertion TCP is ambiguous: "
                    f"{selected.component.label} and {runner_up.component.label} "
                    f"differ by {separation_m * 1000.0:.3f} mm."
                )

    shift_physical = (selected.center_local - legacy) * scale
    if abs(float(shift_physical[longitudinal])) > 1.0e-12:
        raise RuntimeError("Derived connector TCP changed the nose depth.")
    shift_magnitude = float(np.linalg.norm(shift_physical))
    if shift_magnitude > maximum_shift:
        raise RuntimeError(
            "Mesh-derived connector TCP transverse shift is implausible: "
            f"{shift_magnitude * 1000.0:.3f} mm."
        )

    return InsertionTcpDerivation(
        tip_local=selected.center_local.copy(),
        legacy_tip_local=legacy.copy(),
        shift_physical_m=shift_physical.copy(),
        selected_label=selected.component.label,
        selected_local_min=selected.component.local_min.copy(),
        selected_local_max=selected.component.local_max.copy(),
        cross_section_m=selected.cross_section_m.copy(),
        nose_gap_m=selected.nose_gap_m,
        score=selected.score,
        candidate_count=len(candidates),
    )
