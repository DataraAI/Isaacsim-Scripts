#!/usr/bin/env python3
"""Dense stereo triangulation and 3D fitting for cavity-anchored bezel samples."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from config import FrontRimConfig
from front_rim import FrontRim2D


class StereoCamera(Protocol):
    @property
    def camera_center_world_m(self) -> np.ndarray: ...

    def pixel_to_world_ray(
        self,
        pixel_uv: np.ndarray | tuple[float, float],
    ) -> tuple[np.ndarray, np.ndarray]: ...

    def project_world(self, point_world_m: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class FrontRim3D:
    corners_world_m: np.ndarray
    center_world_m: np.ndarray
    normal_world: np.ndarray
    horizontal_world: np.ndarray
    vertical_world: np.ndarray
    width_m: float
    height_m: float
    reprojection_rms_px: float
    max_reprojection_px: float
    max_ray_gap_m: float
    plane_residual_m: float
    sample_points_world_m: np.ndarray
    sample_inlier_mask: np.ndarray

    def __post_init__(self) -> None:
        corners = np.asarray(self.corners_world_m, dtype=np.float64)
        center = np.asarray(self.center_world_m, dtype=np.float64).reshape(3)
        normal = _unit_vector(self.normal_world, "normal_world")
        horizontal = _unit_vector(self.horizontal_world, "horizontal_world")
        vertical = _unit_vector(self.vertical_world, "vertical_world")
        points = np.asarray(self.sample_points_world_m, dtype=np.float64)
        mask = np.asarray(self.sample_inlier_mask, dtype=bool)

        if corners.shape != (4, 3):
            raise ValueError(
                f"corners_world_m must have shape (4,3), got {corners.shape}."
            )
        if points.ndim != 3 or points.shape[0] != 4 or points.shape[2] != 3:
            raise ValueError(
                "sample_points_world_m must have shape "
                "(4,samples_per_side,3)."
            )
        if mask.shape != points.shape[:2]:
            raise ValueError(
                "sample_inlier_mask must match the first two sample dimensions."
            )
        if float(self.width_m) <= 0.0 or float(self.height_m) <= 0.0:
            raise ValueError("Front-rim width and height must be positive.")

        object.__setattr__(self, "corners_world_m", corners.copy())
        object.__setattr__(self, "center_world_m", center.copy())
        object.__setattr__(self, "normal_world", normal)
        object.__setattr__(self, "horizontal_world", horizontal)
        object.__setattr__(self, "vertical_world", vertical)
        object.__setattr__(self, "sample_points_world_m", points.copy())
        object.__setattr__(self, "sample_inlier_mask", mask.copy())

    @property
    def accepted_sample_count(self) -> int:
        return int(np.count_nonzero(self.sample_inlier_mask))


def _unit_vector(value: np.ndarray, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(vector))
    if not np.all(np.isfinite(vector)) or norm <= 1.0e-12:
        raise ValueError(f"{name} must be finite and nonzero.")
    return vector / norm


def triangulate_pixel_pair(
    left_uv: np.ndarray,
    right_uv: np.ndarray,
    left_camera: StereoCamera,
    right_camera: StereoCamera,
) -> tuple[np.ndarray, float]:
    left_origin, left_direction = left_camera.pixel_to_world_ray(left_uv)
    right_origin, right_direction = right_camera.pixel_to_world_ray(right_uv)
    left_origin = np.asarray(left_origin, dtype=np.float64).reshape(3)
    right_origin = np.asarray(right_origin, dtype=np.float64).reshape(3)
    left_direction = _unit_vector(left_direction, "left stereo ray")
    right_direction = _unit_vector(right_direction, "right stereo ray")

    system = np.column_stack((left_direction, -right_direction))
    values, _, rank, _ = np.linalg.lstsq(
        system,
        right_origin - left_origin,
        rcond=None,
    )
    if rank < 2:
        raise RuntimeError("Stereo rays are parallel or numerically singular.")
    left_distance, right_distance = map(float, values)
    if left_distance <= 0.0 or right_distance <= 0.0:
        raise RuntimeError("Triangulated rim point lies behind a camera.")

    left_point = left_origin + left_distance * left_direction
    right_point = right_origin + right_distance * right_direction
    gap = float(np.linalg.norm(left_point - right_point))
    return (left_point + right_point) / 2.0, gap


def _robust_plane_fit(
    points_world_m: np.ndarray,
    cfg: FrontRimConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    points = np.asarray(points_world_m, dtype=np.float64).reshape(-1, 3)
    if points.shape[0] < cfg.min_plane_inliers:
        raise RuntimeError(
            f"Only {points.shape[0]} stereo bezel points reached plane fitting."
        )

    inlier_mask = np.ones(points.shape[0], dtype=bool)
    residuals = np.zeros(points.shape[0], dtype=np.float64)

    for _ in range(cfg.plane_fit_iterations):
        inliers = points[inlier_mask]
        center = np.mean(inliers, axis=0)
        _, _, vh = np.linalg.svd(inliers - center, full_matrices=False)
        normal = _unit_vector(vh[-1], "plane normal")
        residuals = np.abs((points - center) @ normal)

        current = residuals[inlier_mask]
        median = float(np.median(current))
        mad = float(np.median(np.abs(current - median)))
        robust_limit = median + cfg.plane_mad_scale * 1.4826 * mad
        threshold = min(
            cfg.plane_max_residual_m,
            max(1.0e-8, robust_limit),
        )
        next_mask = residuals <= threshold
        if int(np.count_nonzero(next_mask)) < cfg.min_plane_inliers:
            raise RuntimeError(
                "Robust plane fit rejected too many stereo bezel points."
            )
        if np.array_equal(next_mask, inlier_mask):
            inlier_mask = next_mask
            break
        inlier_mask = next_mask

    inliers = points[inlier_mask]
    center = np.mean(inliers, axis=0)
    _, _, vh = np.linalg.svd(inliers - center, full_matrices=False)
    normal = _unit_vector(vh[-1], "plane normal")
    residuals = np.abs((points - center) @ normal)
    return center, normal, inlier_mask, residuals


def _camera_facing_normal(
    normal_world: np.ndarray,
    center_world_m: np.ndarray,
    left_camera: StereoCamera,
    right_camera: StereoCamera,
    cfg: FrontRimConfig,
) -> np.ndarray:
    normal = _unit_vector(normal_world, "plane normal")
    camera_midpoint = 0.5 * (
        np.asarray(left_camera.camera_center_world_m, dtype=np.float64)
        + np.asarray(right_camera.camera_center_world_m, dtype=np.float64)
    )
    toward_cameras = _unit_vector(
        camera_midpoint - center_world_m,
        "opening-to-camera direction",
    )
    if float(np.dot(normal, toward_cameras)) < 0.0:
        normal = -normal
    cosine = float(np.dot(normal, toward_cameras))
    if cosine < cfg.normal_min_camera_cosine:
        raise RuntimeError(
            "Fitted front-bezel normal is not sufficiently camera-facing: "
            f"cosine={cosine:.3f}."
        )
    return normal


def _intersect_pixel_with_plane(
    camera: StereoCamera,
    pixel_uv: np.ndarray | tuple[float, float],
    plane_center_world_m: np.ndarray,
    plane_normal_world: np.ndarray,
) -> np.ndarray:
    origin, direction = camera.pixel_to_world_ray(pixel_uv)
    origin = np.asarray(origin, dtype=np.float64).reshape(3)
    direction = _unit_vector(direction, "pixel ray")
    normal = _unit_vector(plane_normal_world, "plane normal")
    denominator = float(np.dot(direction, normal))
    if abs(denominator) <= 1.0e-9:
        raise RuntimeError("Opening ray is parallel to the fitted front plane.")
    distance = float(
        np.dot(
            np.asarray(plane_center_world_m, dtype=np.float64) - origin,
            normal,
        )
        / denominator
    )
    if distance <= 0.0:
        raise RuntimeError("Fitted front plane lies behind an opening ray.")
    return origin + distance * direction


def _rectangle_axes(
    corners_world_m: np.ndarray,
    normal_world: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    corners = np.asarray(corners_world_m, dtype=np.float64).reshape(4, 3)
    normal = _unit_vector(normal_world, "plane normal")

    horizontal_vector = 0.5 * (
        (corners[1] - corners[0])
        + (corners[2] - corners[3])
    )
    horizontal_vector -= normal * float(np.dot(horizontal_vector, normal))
    horizontal = _unit_vector(horizontal_vector, "horizontal front-rim axis")

    vertical_vector = 0.5 * (
        (corners[3] - corners[0])
        + (corners[2] - corners[1])
    )
    vertical_vector -= normal * float(np.dot(vertical_vector, normal))
    vertical = _unit_vector(vertical_vector, "vertical front-rim axis")
    if float(np.dot(np.cross(horizontal, vertical), normal)) < 0.0:
        vertical = -vertical

    width_m = 0.5 * (
        float(np.linalg.norm(corners[1] - corners[0]))
        + float(np.linalg.norm(corners[2] - corners[3]))
    )
    height_m = 0.5 * (
        float(np.linalg.norm(corners[3] - corners[0]))
        + float(np.linalg.norm(corners[2] - corners[1]))
    )
    if width_m <= 0.0 or height_m <= 0.0:
        raise RuntimeError("Fitted front-rim rectangle has non-positive size.")
    return horizontal, vertical, width_m, height_m


def triangulate_front_rims(
    left_rim: FrontRim2D,
    right_rim: FrontRim2D,
    left_camera: StereoCamera,
    right_camera: StereoCamera,
    cfg: FrontRimConfig,
) -> FrontRim3D:
    """Fit the front bezel, then intersect opening rays with that plane."""
    left_samples = np.asarray(left_rim.side_samples_uv, dtype=np.float64)
    right_samples = np.asarray(right_rim.side_samples_uv, dtype=np.float64)
    if left_samples.shape != right_samples.shape:
        raise ValueError("Left/right front-bezel sample arrays must match.")
    if left_samples.ndim != 3 or left_samples.shape[0] != 4:
        raise ValueError("Front-bezel samples must have shape (4,N,2).")

    sample_count = left_samples.shape[1]
    points = np.full((4, sample_count, 3), np.nan, dtype=np.float64)
    accepted = np.zeros((4, sample_count), dtype=bool)
    ray_gaps = np.full((4, sample_count), np.nan, dtype=np.float64)
    reprojection_max = np.full((4, sample_count), np.nan, dtype=np.float64)
    reprojection_values: list[float] = []

    for side_index in range(4):
        for sample_index in range(sample_count):
            left_uv = left_samples[side_index, sample_index]
            right_uv = right_samples[side_index, sample_index]
            if abs(float(left_uv[1] - right_uv[1])) > (
                cfg.sample_max_epipolar_error_px
            ):
                continue
            try:
                point, gap = triangulate_pixel_pair(
                    left_uv,
                    right_uv,
                    left_camera,
                    right_camera,
                )
                left_error = float(
                    np.linalg.norm(left_camera.project_world(point) - left_uv)
                )
                right_error = float(
                    np.linalg.norm(right_camera.project_world(point) - right_uv)
                )
            except RuntimeError:
                continue

            max_error = max(left_error, right_error)
            if (
                gap > cfg.sample_max_ray_gap_m
                or max_error > cfg.sample_max_reprojection_px
            ):
                continue

            points[side_index, sample_index] = point
            accepted[side_index, sample_index] = True
            ray_gaps[side_index, sample_index] = gap
            reprojection_max[side_index, sample_index] = max_error
            reprojection_values.extend((left_error, right_error))

    accepted_count = int(np.count_nonzero(accepted))
    if accepted_count < cfg.min_accepted_sample_pairs:
        raise RuntimeError(
            f"Only {accepted_count}/{accepted.size} dense rim pairs passed "
            "stereo validation."
        )
    for side_index, side_name in enumerate(("top", "right", "bottom", "left")):
        if int(np.count_nonzero(accepted[side_index])) < 2:
            raise RuntimeError(
                f"The {side_name} rim has fewer than two accepted stereo samples."
            )

    flat_points = points[accepted]
    plane_center, normal, plane_inliers, _ = _robust_plane_fit(
        flat_points,
        cfg,
    )

    accepted_indices = np.argwhere(accepted)
    final_mask = np.zeros_like(accepted)
    for original_index, keep in zip(accepted_indices, plane_inliers, strict=True):
        if keep:
            final_mask[tuple(original_index)] = True

    if int(np.count_nonzero(final_mask)) < cfg.min_plane_inliers:
        raise RuntimeError("Too few front-bezel samples survived plane fitting.")
    for side_index, side_name in enumerate(("top", "right", "bottom", "left")):
        if int(np.count_nonzero(final_mask[side_index])) < 2:
            raise RuntimeError(
                f"The {side_name} bezel side lost too many plane inliers."
            )

    normal = _camera_facing_normal(
        normal,
        plane_center,
        left_camera,
        right_camera,
        cfg,
    )

    left_center = _intersect_pixel_with_plane(
        left_camera,
        left_rim.center_uv,
        plane_center,
        normal,
    )
    right_center = _intersect_pixel_with_plane(
        right_camera,
        right_rim.center_uv,
        plane_center,
        normal,
    )
    center_gap = float(np.linalg.norm(left_center - right_center))
    if center_gap > cfg.sample_max_ray_gap_m:
        raise RuntimeError(
            "Opening-center plane intersections disagree by "
            f"{center_gap * 1000.0:.3f} mm."
        )
    center = 0.5 * (left_center + right_center)

    corners: list[np.ndarray] = []
    corner_gaps: list[float] = []
    for corner_index in range(4):
        left_corner = _intersect_pixel_with_plane(
            left_camera,
            left_rim.corners_uv[corner_index],
            plane_center,
            normal,
        )
        right_corner = _intersect_pixel_with_plane(
            right_camera,
            right_rim.corners_uv[corner_index],
            plane_center,
            normal,
        )
        corners.append(0.5 * (left_corner + right_corner))
        corner_gaps.append(float(np.linalg.norm(left_corner - right_corner)))
    corners_world = np.vstack(corners)
    corners_world += center - np.mean(corners_world, axis=0)

    horizontal, vertical, width_m, height_m = _rectangle_axes(
        corners_world,
        normal,
    )

    reprojection = np.asarray(reprojection_values, dtype=np.float64)
    if reprojection.size == 0:
        raise RuntimeError("No front-bezel reprojection measurements were recorded.")
    final_ray_gaps = ray_gaps[final_mask]
    final_plane_residuals = np.abs(
        (points[final_mask] - plane_center) @ normal
    )
    maximum_gap = max(
        float(np.nanmax(final_ray_gaps)),
        center_gap,
        max(corner_gaps),
    )

    return FrontRim3D(
        corners_world_m=corners_world,
        center_world_m=center,
        normal_world=normal,
        horizontal_world=horizontal,
        vertical_world=vertical,
        width_m=float(width_m),
        height_m=float(height_m),
        reprojection_rms_px=float(
            np.sqrt(np.mean(reprojection * reprojection))
        ),
        max_reprojection_px=float(np.nanmax(reprojection_max[final_mask])),
        max_ray_gap_m=float(maximum_gap),
        plane_residual_m=float(np.max(final_plane_residuals)),
        sample_points_world_m=points,
        sample_inlier_mask=final_mask,
    )
