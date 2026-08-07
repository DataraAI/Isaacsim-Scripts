#!/usr/bin/env python3
"""Pure geometry for pre-shaping a free-hanging deformable cable tail."""

from __future__ import annotations

import math

import numpy as np


def _finite_vector(value, label: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{label} must be finite with shape (3,)")
    return vector


def _finite_nonnegative(value: float, label: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{label} must be finite and nonnegative")
    return number


def _finite_positive(value: float, label: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{label} must be finite and positive")
    return number


def preshape_free_hanging_tail(
    points_world_m: np.ndarray,
    *,
    plug_world_m: np.ndarray,
    down_world_axis: np.ndarray,
    anchor_length_m: float,
    bend_length_m: float,
    far_anchor_length_m: float,
    drop_m: float,
) -> np.ndarray:
    """Curve a straight cable downward while leaving both ends fixed."""

    points = np.asarray(points_world_m, dtype=np.float64)
    if (
        points.ndim != 2
        or points.shape[1] != 3
        or points.shape[0] < 4
        or not np.all(np.isfinite(points))
    ):
        raise ValueError(
            "points_world_m must be a finite array with shape (N, 3)"
        )

    plug = _finite_vector(plug_world_m, "plug_world_m")
    down = _finite_vector(down_world_axis, "down_world_axis")
    down_norm = float(np.linalg.norm(down))
    if down_norm <= 1.0e-12:
        raise ValueError("down_world_axis must be nonzero")
    down = down / down_norm

    anchor = _finite_nonnegative(anchor_length_m, "anchor_length_m")
    bend = _finite_positive(bend_length_m, "bend_length_m")
    far_anchor = _finite_nonnegative(
        far_anchor_length_m,
        "far_anchor_length_m",
    )
    drop = _finite_nonnegative(drop_m, "drop_m")

    centered = points - np.mean(points, axis=0)
    _, singular_values, right_t = np.linalg.svd(
        centered,
        full_matrices=False,
    )
    if (
        singular_values.shape != (3,)
        or not np.all(np.isfinite(singular_values))
        or singular_values[0] <= 1.0e-9
    ):
        raise ValueError("tail point cloud is degenerate")

    axis = right_t[0]
    projections = points @ axis
    endpoint_count = max(
        2,
        min(
            points.shape[0] // 4,
            int(np.ceil(points.shape[0] * 0.05)),
        ),
    )
    order = np.argsort(projections, kind="stable")
    low_center = np.mean(points[order[:endpoint_count]], axis=0)
    high_center = np.mean(points[order[-endpoint_count:]], axis=0)
    if np.linalg.norm(high_center - plug) < np.linalg.norm(low_center - plug):
        axis = -axis

    progress = points @ axis
    progress = progress - float(np.min(progress))
    cable_length = float(np.max(progress))
    minimum_length = anchor + far_anchor + 2.0 * bend
    if not math.isfinite(cable_length) or cable_length <= minimum_length:
        raise ValueError(
            "tail point cloud is too short for the requested hanging profile"
        )

    def smoothstep(value: np.ndarray) -> np.ndarray:
        clipped = np.clip(value, 0.0, 1.0)
        return clipped * clipped * (3.0 - 2.0 * clipped)

    rise = smoothstep((progress - anchor) / bend)
    fall = smoothstep((cable_length - far_anchor - progress) / bend)
    weights = rise * fall
    return points + weights[:, None] * drop * down[None, :]
