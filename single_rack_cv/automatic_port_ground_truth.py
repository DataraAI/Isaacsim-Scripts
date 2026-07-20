#!/usr/bin/env python3
"""Automatic USD-raycast reference for a visually detected port opening."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RaycastGroundTruthConfig:
    """Validation-only settings for fitting the front bezel plane."""

    rack_path_prefix: str = "/World/ServerRack"
    rim_outward_offset_px: float = 3.0
    max_raycast_distance_m: float = 2.0
    min_plane_hits: int = 12
    depth_cluster_tolerance_m: float = 0.004
    plane_fit_iterations: int = 4
    plane_mad_scale: float = 2.5
    plane_max_residual_m: float = 0.0005
    min_surface_normal_cosine: float = 0.80


@dataclass(frozen=True)
class RaycastHit:
    position_world_m: np.ndarray
    normal_world: np.ndarray
    prim_path: str
    distance_m: float

    def __post_init__(self) -> None:
        position = np.asarray(self.position_world_m, dtype=np.float64).reshape(3)
        normal = np.asarray(self.normal_world, dtype=np.float64).reshape(3)
        normal_norm = float(np.linalg.norm(normal))
        if not np.all(np.isfinite(position)):
            raise ValueError("Raycast hit position must be finite.")
        if not np.isfinite(normal_norm) or normal_norm <= 1.0e-12:
            raise ValueError("Raycast hit normal must be finite and nonzero.")
        if not np.isfinite(self.distance_m) or self.distance_m <= 0.0:
            raise ValueError("Raycast hit distance must be finite and positive.")
        object.__setattr__(self, "position_world_m", position.copy())
        object.__setattr__(self, "normal_world", normal / normal_norm)


@dataclass(frozen=True)
class AutomaticGroundTruth:
    center_world_m: np.ndarray
    normal_world: np.ndarray
    plane_residual_m: float
    valid_hit_count: int
    used_hit_count: int
    used_prim_paths: tuple[str, ...]

    def __post_init__(self) -> None:
        center = np.asarray(self.center_world_m, dtype=np.float64).reshape(3)
        normal = np.asarray(self.normal_world, dtype=np.float64).reshape(3)
        normal_norm = float(np.linalg.norm(normal))
        if not np.all(np.isfinite(center)):
            raise ValueError("Automatic ground-truth center must be finite.")
        if not np.isfinite(normal_norm) or normal_norm <= 1.0e-12:
            raise ValueError("Automatic ground-truth normal must be nonzero.")
        object.__setattr__(self, "center_world_m", center.copy())
        object.__setattr__(self, "normal_world", normal / normal_norm)


def _unit(vector: np.ndarray, name: str) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(value))
    if not np.all(np.isfinite(value)) or norm <= 1.0e-12:
        raise ValueError(f"{name} must be finite and nonzero.")
    return value / norm


def offset_rim_samples_outward(
    side_samples_uv: np.ndarray,
    center_uv: tuple[float, float] | np.ndarray,
    offset_px: float,
) -> np.ndarray:
    """Shift each complete rim side away from the fitted opening center."""
    samples = np.asarray(side_samples_uv, dtype=np.float64)
    if samples.ndim != 3 or samples.shape[0] != 4 or samples.shape[2] != 2:
        raise ValueError("side_samples_uv must have shape (4,N,2).")
    if not np.isfinite(offset_px) or offset_px <= 0.0:
        raise ValueError("offset_px must be finite and positive.")
    center = np.asarray(center_uv, dtype=np.float64).reshape(2)
    shifted = samples.copy()
    for side_index in range(4):
        side_center = np.mean(samples[side_index], axis=0)
        outward = side_center - center
        norm = float(np.linalg.norm(outward))
        if norm <= 1.0e-12:
            raise RuntimeError(f"Rim side {side_index} has no outward direction.")
        shifted[side_index] += offset_px * outward / norm
    return shifted


def intersect_ray_with_plane(
    ray_origin_world_m: np.ndarray,
    ray_direction_world: np.ndarray,
    plane_center_world_m: np.ndarray,
    plane_normal_world: np.ndarray,
) -> np.ndarray:
    origin = np.asarray(ray_origin_world_m, dtype=np.float64).reshape(3)
    direction = _unit(ray_direction_world, "ray_direction_world")
    plane_center = np.asarray(plane_center_world_m, dtype=np.float64).reshape(3)
    normal = _unit(plane_normal_world, "plane_normal_world")
    denominator = float(np.dot(direction, normal))
    if abs(denominator) <= 1.0e-9:
        raise RuntimeError("Opening-center ray is parallel to the fitted front plane.")
    distance = float(np.dot(plane_center - origin, normal) / denominator)
    if distance <= 0.0:
        raise RuntimeError("Fitted front plane is behind the opening-center ray.")
    return origin + distance * direction


def _densest_depth_cluster_mask(
    distances_m: np.ndarray,
    tolerance_m: float,
) -> np.ndarray:
    distances = np.asarray(distances_m, dtype=np.float64).reshape(-1)
    order = np.argsort(distances)
    sorted_distances = distances[order]
    best_start = 0
    best_end = 0
    end = 0
    for start in range(sorted_distances.size):
        end = max(end, start)
        while (
            end + 1 < sorted_distances.size
            and sorted_distances[end + 1] - sorted_distances[start] <= tolerance_m
        ):
            end += 1
        if end - start > best_end - best_start:
            best_start, best_end = start, end
    mask = np.zeros(distances.size, dtype=bool)
    mask[order[best_start : best_end + 1]] = True
    return mask


def _fit_plane_robust(
    points_world_m: np.ndarray,
    max_residual_m: float,
    iterations: int,
    mad_scale: float,
    min_inliers: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    points = np.asarray(points_world_m, dtype=np.float64).reshape(-1, 3)
    inlier_mask = np.ones(points.shape[0], dtype=bool)
    for _ in range(iterations):
        inliers = points[inlier_mask]
        center = np.mean(inliers, axis=0)
        _, _, vh = np.linalg.svd(inliers - center)
        normal = _unit(vh[-1], "fitted plane normal")
        residuals = np.abs((points - center) @ normal)
        active = residuals[inlier_mask]
        median = float(np.median(active))
        mad = float(np.median(np.abs(active - median)))
        robust_threshold = median + mad_scale * 1.4826 * mad
        threshold = min(max_residual_m, max(1.0e-8, robust_threshold))
        next_mask = residuals <= threshold
        if int(np.count_nonzero(next_mask)) < min_inliers:
            raise RuntimeError("Automatic front-plane fit rejected too many ray hits.")
        if np.array_equal(next_mask, inlier_mask):
            inlier_mask = next_mask
            break
        inlier_mask = next_mask

    inliers = points[inlier_mask]
    center = np.mean(inliers, axis=0)
    _, _, vh = np.linalg.svd(inliers - center)
    normal = _unit(vh[-1], "fitted plane normal")
    residuals = np.abs((inliers - center) @ normal)
    maximum = float(np.max(residuals))
    if maximum > max_residual_m:
        raise RuntimeError(
            f"Automatic front-plane residual {maximum:.6f} m exceeds gate."
        )
    return center, normal, inlier_mask, maximum


def build_automatic_ground_truth(
    hits: list[RaycastHit],
    camera_center_world_m: np.ndarray,
    center_ray_direction_world: np.ndarray,
    cfg: RaycastGroundTruthConfig,
) -> AutomaticGroundTruth:
    """Fit the dominant front bezel plane and intersect the opening-center ray."""
    camera = np.asarray(camera_center_world_m, dtype=np.float64).reshape(3)
    valid_hits = [
        hit
        for hit in hits
        if hit.prim_path.startswith(cfg.rack_path_prefix)
    ]
    if len(valid_hits) < cfg.min_plane_hits:
        raise RuntimeError(
            f"Only {len(valid_hits)} valid rack ray hits were available; "
            f"need {cfg.min_plane_hits}."
        )

    positions = np.vstack([hit.position_world_m for hit in valid_hits])
    distances = np.asarray([hit.distance_m for hit in valid_hits], dtype=np.float64)
    cluster_mask = _densest_depth_cluster_mask(
        distances,
        cfg.depth_cluster_tolerance_m,
    )
    if int(np.count_nonzero(cluster_mask)) < cfg.min_plane_hits:
        raise RuntimeError("No dominant front-depth ray-hit cluster was found.")

    cluster_positions = positions[cluster_mask]
    cluster_hits = [
        hit for hit, keep in zip(valid_hits, cluster_mask, strict=True) if keep
    ]
    plane_center, normal, plane_mask, residual = _fit_plane_robust(
        cluster_positions,
        max_residual_m=cfg.plane_max_residual_m,
        iterations=cfg.plane_fit_iterations,
        mad_scale=cfg.plane_mad_scale,
        min_inliers=cfg.min_plane_hits,
    )

    to_camera = _unit(camera - plane_center, "camera-facing direction")
    if float(np.dot(normal, to_camera)) < 0.0:
        normal *= -1.0
    if float(np.dot(normal, to_camera)) < cfg.min_surface_normal_cosine:
        raise RuntimeError("Automatic front-plane normal does not face the camera.")

    normal_cosines = np.asarray(
        [
            abs(float(np.dot(hit.normal_world, normal)))
            for hit in cluster_hits
        ],
        dtype=np.float64,
    )
    final_mask = plane_mask & (
        normal_cosines >= cfg.min_surface_normal_cosine
    )
    if int(np.count_nonzero(final_mask)) < cfg.min_plane_hits:
        raise RuntimeError("Too few ray-hit normals agree with the front plane.")

    final_positions = cluster_positions[final_mask]
    plane_center, normal, _, residual = _fit_plane_robust(
        final_positions,
        max_residual_m=cfg.plane_max_residual_m,
        iterations=cfg.plane_fit_iterations,
        mad_scale=cfg.plane_mad_scale,
        min_inliers=cfg.min_plane_hits,
    )
    to_camera = _unit(camera - plane_center, "camera-facing direction")
    if float(np.dot(normal, to_camera)) < 0.0:
        normal *= -1.0

    center = intersect_ray_with_plane(
        ray_origin_world_m=camera,
        ray_direction_world=center_ray_direction_world,
        plane_center_world_m=plane_center,
        plane_normal_world=normal,
    )
    used_hits = [
        hit for hit, keep in zip(cluster_hits, final_mask, strict=True) if keep
    ]
    return AutomaticGroundTruth(
        center_world_m=center,
        normal_world=normal,
        plane_residual_m=residual,
        valid_hit_count=len(valid_hits),
        used_hit_count=len(used_hits),
        used_prim_paths=tuple(sorted({hit.prim_path for hit in used_hits})),
    )
