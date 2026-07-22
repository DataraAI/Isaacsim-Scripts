#!/usr/bin/env python3
"""Refined local SGBM front-plane geometry with strict final gates."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

import front_rim_sgbm as base


STRICT_RAY_GAP_M = 0.0005


def _unit(value: np.ndarray, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(vector))
    if not np.all(np.isfinite(vector)) or norm <= 1.0e-12:
        raise ValueError(f"{name} must be finite and nonzero.")
    return vector / norm


def _fit_plane_stable(
    points_world_m: np.ndarray,
    cfg: base.LocalSGBMConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Fit a plane with monotonic trimming and a final hard residual pass."""
    points = np.asarray(points_world_m, dtype=np.float64).reshape(-1, 3)
    if points.shape[0] < cfg.min_cluster_points:
        raise RuntimeError("Too few points reached refined SGBM plane fitting.")

    inliers = np.ones(points.shape[0], dtype=bool)
    max_iterations = max(8, 2 * int(cfg.plane_fit_iterations))

    for _ in range(max_iterations):
        active = points[inliers]
        center = np.mean(active, axis=0)
        _, _, vh = np.linalg.svd(active - center, full_matrices=False)
        normal = _unit(vh[-1], "refined SGBM plane normal")
        residuals = np.abs((points - center) @ normal)
        active_residuals = residuals[inliers]

        median = float(np.median(active_residuals))
        mad = float(np.median(np.abs(active_residuals - median)))
        robust_limit = median + cfg.plane_mad_scale * 1.4826 * mad
        threshold = min(
            cfg.plane_max_residual_m,
            max(1.0e-8, robust_limit),
        )

        next_inliers = inliers & (residuals <= threshold)
        if int(np.count_nonzero(next_inliers)) < cfg.min_cluster_points:
            raise RuntimeError(
                "Refined SGBM plane fit rejected too many depth points."
            )

        if np.array_equal(next_inliers, inliers):
            hard_inliers = inliers & (
                residuals <= cfg.plane_max_residual_m
            )
            if int(np.count_nonzero(hard_inliers)) < cfg.min_cluster_points:
                raise RuntimeError(
                    "Refined SGBM hard residual pass rejected too many points."
                )
            if np.array_equal(hard_inliers, inliers):
                residual = float(np.max(residuals[inliers]))
                return center, normal, inliers, residual
            next_inliers = hard_inliers

        inliers = next_inliers

    for _ in range(max_iterations):
        active = points[inliers]
        center = np.mean(active, axis=0)
        _, _, vh = np.linalg.svd(active - center, full_matrices=False)
        normal = _unit(vh[-1], "refined SGBM plane normal")
        residuals = np.abs((points - center) @ normal)
        next_inliers = inliers & (
            residuals <= cfg.plane_max_residual_m
        )
        if int(np.count_nonzero(next_inliers)) < cfg.min_cluster_points:
            raise RuntimeError(
                "Refined SGBM final residual pass rejected too many points."
            )
        if np.array_equal(next_inliers, inliers):
            residual = float(np.max(residuals[inliers]))
            return center, normal, inliers, residual
        inliers = next_inliers

    raise RuntimeError("Refined SGBM plane inliers did not stabilize.")


def _intersect_midpoint_ray_with_plane(
    left_camera,
    right_camera,
    left_center_uv: tuple[float, float] | np.ndarray,
    right_center_uv: tuple[float, float] | np.ndarray,
    plane_center_world_m: np.ndarray,
    plane_normal_world: np.ndarray,
) -> np.ndarray:
    """Intersect one fused midpoint-camera ray with the fitted front plane."""
    left_origin, left_direction = left_camera.pixel_to_world_ray(left_center_uv)
    right_origin, right_direction = right_camera.pixel_to_world_ray(right_center_uv)
    origin = 0.5 * (
        np.asarray(left_origin, dtype=np.float64).reshape(3)
        + np.asarray(right_origin, dtype=np.float64).reshape(3)
    )
    direction = _unit(
        np.asarray(left_direction, dtype=np.float64).reshape(3)
        + np.asarray(right_direction, dtype=np.float64).reshape(3),
        "fused cavity-center direction",
    )
    normal = _unit(plane_normal_world, "front-plane normal")
    denominator = float(np.dot(direction, normal))
    if abs(denominator) <= 1.0e-9:
        raise RuntimeError("Fused cavity-center ray is parallel to front plane.")
    distance = float(
        np.dot(
            np.asarray(plane_center_world_m, dtype=np.float64).reshape(3)
            - origin,
            normal,
        )
        / denominator
    )
    if distance <= 0.0:
        raise RuntimeError("Front plane lies behind fused cavity-center ray.")
    return origin + distance * direction


def estimate_front_plane_sgbm_refined(
    left_rgb: np.ndarray,
    right_rgb: np.ndarray,
    left_bbox_xywh: tuple[int, int, int, int],
    left_center_uv: tuple[float, float] | np.ndarray,
    right_bbox_xywh: tuple[int, int, int, int],
    right_center_uv: tuple[float, float] | np.ndarray,
    left_camera,
    right_camera,
    cfg: base.LocalSGBMConfig = base.DEFAULT_SGBM_CONFIG,
    disparity: base.LocalDisparityResult | None = None,
) -> base.SGBMFrontPlaneResult:
    """Estimate the front plane without conflating box corners with ray gaps."""
    strict_cfg = replace(
        cfg,
        max_triangulation_ray_gap_m=min(
            float(cfg.max_triangulation_ray_gap_m),
            STRICT_RAY_GAP_M,
        ),
        # The legacy estimator compares two rays aimed at a recessed cavity point
        # on the front plane. Their separation is expected parallax, so the final
        # center is replaced by one fused midpoint ray below.
        center_max_gap_m=max(float(cfg.center_max_gap_m), 0.010),
    )

    original_fit = base._fit_plane
    original_triangulate = base.triangulate_pixel_pair
    recorded_pairs: list[tuple[np.ndarray, float]] = []

    def recording_triangulate(*args, **kwargs):
        point, gap = original_triangulate(*args, **kwargs)
        recorded_pairs.append(
            (np.asarray(point, dtype=np.float64).reshape(3), float(gap))
        )
        return point, gap

    base._fit_plane = _fit_plane_stable
    base.triangulate_pixel_pair = recording_triangulate
    try:
        result = base.estimate_front_plane_sgbm(
            left_rgb=left_rgb,
            right_rgb=right_rgb,
            left_bbox_xywh=left_bbox_xywh,
            left_center_uv=left_center_uv,
            right_bbox_xywh=right_bbox_xywh,
            right_center_uv=right_center_uv,
            left_camera=left_camera,
            right_camera=right_camera,
            cfg=strict_cfg,
            disparity=disparity,
        )
    finally:
        base._fit_plane = original_fit
        base.triangulate_pixel_pair = original_triangulate

    center = _intersect_midpoint_ray_with_plane(
        left_camera=left_camera,
        right_camera=right_camera,
        left_center_uv=left_center_uv,
        right_center_uv=right_center_uv,
        plane_center_world_m=result.center_world_m,
        plane_normal_world=result.normal_world,
    )
    corners = np.asarray(result.corners_world_m, dtype=np.float64).copy()
    corners += center - np.asarray(result.center_world_m, dtype=np.float64)

    dense_gaps: list[float] = []
    for point, gap in recorded_pairs:
        plane_distance = abs(
            float(
                np.dot(
                    point - np.asarray(result.center_world_m, dtype=np.float64),
                    np.asarray(result.normal_world, dtype=np.float64),
                )
            )
        )
        if (
            plane_distance <= cfg.plane_max_residual_m + 1.0e-12
            and gap <= STRICT_RAY_GAP_M + 1.0e-12
        ):
            dense_gaps.append(gap)
    if not dense_gaps:
        raise RuntimeError(
            "No refined SGBM plane correspondences survived strict ray-gap filtering."
        )

    return replace(
        result,
        center_world_m=center,
        corners_world_m=corners,
        max_ray_gap_m=float(max(dense_gaps)),
    )
