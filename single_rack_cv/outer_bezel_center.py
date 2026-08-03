#!/usr/bin/env python3
"""Physical RJ45 center reconstructed on a dense outer-bezel plane."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math

import numpy as np

from aperture_center import estimate_planar_aperture_center
from front_plane import (
    DEFAULT_FRONT_PLANE_CONFIG,
    FrontPlaneConfig,
    build_bezel_ring_pixels,
    compute_local_disparity,
    fit_plane_stable,
    intersect_midpoint_ray_with_plane,
    intersect_pixel_with_plane,
)
from stereo_geometry import triangulate_pixel_pair, unit_vector


_SIDE_COUNT = 4
_MIN_SUPPORTED_REGIONS = 2
_MIN_SUPPORT_SPAN_U_PX = 12.0
_MIN_SUPPORT_SPAN_V_PX = 12.0
_MIN_SUPPORT_MINOR_STD_PX = 3.0

OUTER_BEZEL_CONFIG = replace(
    DEFAULT_FRONT_PLANE_CONFIG,
    roi_margin_px=48,
    ring_inner_offset_px=6,
    ring_outer_offset_px=36,
    depth_cluster_tolerance_m=0.0020,
    min_cluster_points=20,
    min_points_per_side=1,
)


@dataclass(frozen=True)
class BezelSupportDiagnostics:
    region_count: int
    span_u_px: float
    span_v_px: float
    major_std_px: float
    minor_std_px: float
    side_counts: tuple[int, int, int, int]


@dataclass(frozen=True)
class OuterBezelPlaneResult:
    center_world_m: np.ndarray
    normal_world: np.ndarray
    corners_world_m: np.ndarray
    width_m: float
    height_m: float
    max_ray_gap_m: float
    reprojection_rms_px: float
    max_reprojection_px: float
    plane_residual_m: float
    valid_disparity_count: int
    consistent_disparity_count: int
    ring_candidate_count: int
    triangulated_count: int
    cluster_count: int
    side_support_counts: tuple[int, int, int, int]
    support_region_count: int
    support_span_u_px: float
    support_span_v_px: float
    support_minor_std_px: float
    median_disparity_px: float
    disparity: object

    def __post_init__(self) -> None:
        center = np.asarray(self.center_world_m, dtype=np.float64).reshape(3)
        normal = unit_vector(self.normal_world, "outer-bezel normal")
        corners = np.asarray(self.corners_world_m, dtype=np.float64).reshape(4, 3)
        if not np.all(np.isfinite(center)) or not np.all(np.isfinite(corners)):
            raise ValueError("Outer-bezel plane geometry must be finite.")
        if float(self.width_m) <= 0.0 or float(self.height_m) <= 0.0:
            raise ValueError("Outer-bezel projected dimensions must be positive.")
        if not 1 <= int(self.support_region_count) <= _SIDE_COUNT:
            raise ValueError("support_region_count must be between 1 and 4.")
        object.__setattr__(self, "center_world_m", center.copy())
        object.__setattr__(self, "normal_world", normal)
        object.__setattr__(self, "corners_world_m", corners.copy())
        object.__setattr__(
            self,
            "side_support_counts",
            tuple(int(value) for value in self.side_support_counts),
        )


@dataclass(frozen=True)
class OuterBezelApertureResult:
    center_world_m: np.ndarray
    left_center_world_m: np.ndarray
    right_center_world_m: np.ndarray
    left_center_uv: np.ndarray
    right_center_uv: np.ndarray
    eye_disagreement_m: float
    plane_origin_world_m: np.ndarray
    plane_normal_world: np.ndarray
    corners_world_m: np.ndarray
    width_m: float
    height_m: float
    plane_residual_m: float
    max_ray_gap_m: float
    reprojection_rms_px: float
    max_reprojection_px: float
    valid_disparity_count: int
    consistent_disparity_count: int
    ring_candidate_count: int
    triangulated_count: int
    cluster_count: int
    side_support_counts: tuple[int, int, int, int]
    support_region_count: int
    support_span_u_px: float
    support_span_v_px: float
    support_minor_std_px: float
    median_disparity_px: float
    disparity: object
    front_plane_config: FrontPlaneConfig

    def __post_init__(self) -> None:
        vector_fields = (
            "center_world_m",
            "left_center_world_m",
            "right_center_world_m",
            "plane_origin_world_m",
            "plane_normal_world",
        )
        for name in vector_fields:
            value = np.asarray(getattr(self, name), dtype=np.float64).reshape(3)
            if not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must be finite.")
            object.__setattr__(self, name, value.copy())
        for name in ("left_center_uv", "right_center_uv"):
            value = np.asarray(getattr(self, name), dtype=np.float64).reshape(2)
            if not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must be finite.")
            object.__setattr__(self, name, value.copy())
        corners = np.asarray(self.corners_world_m, dtype=np.float64).reshape(4, 3)
        if not np.all(np.isfinite(corners)):
            raise ValueError("corners_world_m must be finite.")
        object.__setattr__(self, "corners_world_m", corners.copy())
        disagreement = float(self.eye_disagreement_m)
        if not math.isfinite(disagreement) or disagreement < 0.0:
            raise ValueError("eye_disagreement_m must be finite and nonnegative.")
        object.__setattr__(
            self,
            "side_support_counts",
            tuple(int(value) for value in self.side_support_counts),
        )

    @property
    def normal_world(self) -> np.ndarray:
        return self.plane_normal_world.copy()


def support_diagnostics(
    pixels_uv: np.ndarray,
    side_labels: np.ndarray,
) -> BezelSupportDiagnostics:
    pixels = np.asarray(pixels_uv, dtype=np.float64).reshape(-1, 2)
    labels = np.asarray(side_labels, dtype=np.int64).reshape(-1)
    if pixels.shape[0] != labels.shape[0] or pixels.shape[0] < 3:
        raise RuntimeError(
            "Outer-bezel support needs at least three labeled pixels."
        )
    if not np.all(np.isfinite(pixels)):
        raise RuntimeError("Outer-bezel support pixels must be finite.")
    if np.any(labels < 0) or np.any(labels >= _SIDE_COUNT):
        raise RuntimeError("Outer-bezel side labels must be in [0, 3].")

    centered = pixels - np.mean(pixels, axis=0)
    covariance = centered.T @ centered / float(pixels.shape[0])
    eigenvalues = np.maximum(np.linalg.eigvalsh(covariance), 0.0)
    minor_std, major_std = np.sqrt(eigenvalues)
    side_counts = tuple(
        int(np.count_nonzero(labels == side_index))
        for side_index in range(_SIDE_COUNT)
    )
    return BezelSupportDiagnostics(
        region_count=int(sum(count > 0 for count in side_counts)),
        span_u_px=float(np.ptp(pixels[:, 0])),
        span_v_px=float(np.ptp(pixels[:, 1])),
        major_std_px=float(major_std),
        minor_std_px=float(minor_std),
        side_counts=side_counts,
    )


def _support_is_qualified(
    diagnostics: BezelSupportDiagnostics,
    *,
    min_supported_regions: int,
    min_span_u_px: float,
    min_span_v_px: float,
    min_minor_std_px: float,
) -> bool:
    return bool(
        diagnostics.region_count >= int(min_supported_regions)
        and diagnostics.span_u_px >= float(min_span_u_px)
        and diagnostics.span_v_px >= float(min_span_v_px)
        and diagnostics.minor_std_px >= float(min_minor_std_px)
    )


def select_nearest_supported_range_cluster(
    *,
    ranges_m: np.ndarray,
    pixels_uv: np.ndarray,
    side_labels: np.ndarray,
    tolerance_m: float,
    min_points: int,
    min_supported_regions: int = _MIN_SUPPORTED_REGIONS,
    min_span_u_px: float = _MIN_SUPPORT_SPAN_U_PX,
    min_span_v_px: float = _MIN_SUPPORT_SPAN_V_PX,
    min_minor_std_px: float = _MIN_SUPPORT_MINOR_STD_PX,
) -> tuple[np.ndarray, BezelSupportDiagnostics]:
    ranges = np.asarray(ranges_m, dtype=np.float64).reshape(-1)
    pixels = np.asarray(pixels_uv, dtype=np.float64).reshape(-1, 2)
    labels = np.asarray(side_labels, dtype=np.int64).reshape(-1)
    if ranges.shape[0] != pixels.shape[0] or ranges.shape[0] != labels.shape[0]:
        raise ValueError("Ranges, pixels, and side labels must have equal length.")
    if not np.all(np.isfinite(ranges)):
        raise ValueError("Outer-bezel ranges must be finite.")
    if not math.isfinite(float(tolerance_m)) or float(tolerance_m) <= 0.0:
        raise ValueError("tolerance_m must be finite and positive.")
    if int(min_points) < 3:
        raise ValueError("min_points must be at least three.")
    if not 1 <= int(min_supported_regions) <= _SIDE_COUNT:
        raise ValueError("min_supported_regions must be between 1 and 4.")

    order = np.argsort(ranges)
    values = ranges[order]
    for start in range(values.size):
        end = int(
            np.searchsorted(
                values,
                values[start] + float(tolerance_m),
                side="right",
            )
        )
        candidate_indices = order[start:end]
        if candidate_indices.size < int(min_points):
            continue
        diagnostics = support_diagnostics(
            pixels[candidate_indices],
            labels[candidate_indices],
        )
        if not _support_is_qualified(
            diagnostics,
            min_supported_regions=min_supported_regions,
            min_span_u_px=min_span_u_px,
            min_span_v_px=min_span_v_px,
            min_minor_std_px=min_minor_std_px,
        ):
            continue
        selected = np.zeros(ranges.shape[0], dtype=bool)
        selected[candidate_indices] = True
        return selected, diagnostics

    raise RuntimeError(
        "No qualified outer-bezel depth cluster had enough spatially "
        "distributed support."
    )


def _bbox_corners(bbox_xywh: tuple[int, int, int, int]) -> np.ndarray:
    x, y, width, height = map(float, bbox_xywh)
    return np.array(
        [
            [x, y],
            [x + width, y],
            [x + width, y + height],
            [x, y + height],
        ],
        dtype=np.float64,
    )


def estimate_outer_bezel_plane(
    *,
    left_rgb: np.ndarray,
    right_rgb: np.ndarray,
    left_bbox_xywh: tuple[int, int, int, int],
    left_detection_center_uv: tuple[float, float] | np.ndarray,
    right_bbox_xywh: tuple[int, int, int, int],
    right_detection_center_uv: tuple[float, float] | np.ndarray,
    left_camera,
    right_camera,
    front_plane_config: FrontPlaneConfig = OUTER_BEZEL_CONFIG,
    disparity=None,
) -> OuterBezelPlaneResult:
    cfg = front_plane_config
    if disparity is None:
        disparity = compute_local_disparity(
            left_rgb,
            right_rgb,
            left_bbox_xywh,
            left_detection_center_uv,
            right_bbox_xywh,
            right_detection_center_uv,
            cfg,
        )

    ring_uv, ring_sides = build_bezel_ring_pixels(
        left_bbox_xywh,
        np.asarray(left_rgb).shape[:2],
        cfg,
    )
    x0, y0, x1, y1 = disparity.crop_xyxy
    local_u = ring_uv[:, 0] - x0
    local_v = ring_uv[:, 1] - y0
    inside = (
        (local_u >= 0)
        & (local_u < x1 - x0)
        & (local_v >= 0)
        & (local_v < y1 - y0)
    )
    local_u = local_u[inside].astype(np.int64)
    local_v = local_v[inside].astype(np.int64)
    ring_uv = ring_uv[inside]
    ring_sides = ring_sides[inside]
    consistent = disparity.consistent_mask[local_v, local_u]
    ring_uv = ring_uv[consistent]
    ring_sides = ring_sides[consistent]
    local_u = local_u[consistent]
    local_v = local_v[consistent]
    disparities = disparity.disparity_crop_px[local_v, local_u]
    ring_candidate_count = int(ring_uv.shape[0])
    if ring_candidate_count < cfg.min_cluster_points:
        raise RuntimeError(
            f"Only {ring_candidate_count} consistent outer-bezel stereo pixels "
            "were available."
        )

    points: list[np.ndarray] = []
    used_left_uv: list[np.ndarray] = []
    gaps: list[float] = []
    depths: list[float] = []
    labels: list[int] = []
    used_disparities: list[float] = []
    reprojection_errors: list[float] = []

    camera_midpoint = 0.5 * (
        np.asarray(left_camera.camera_center_world_m, dtype=np.float64)
        + np.asarray(right_camera.camera_center_world_m, dtype=np.float64)
    )
    _, left_direction = left_camera.pixel_to_world_ray(
        left_detection_center_uv
    )
    _, right_direction = right_camera.pixel_to_world_ray(
        right_detection_center_uv
    )
    view_direction = unit_vector(
        np.asarray(left_direction, dtype=np.float64)
        + np.asarray(right_direction, dtype=np.float64),
        "outer-bezel stereo viewing direction",
    )
    vertical_shift = float(disparity.right_vertical_shift_px)

    for left_uv, side, value in zip(
        ring_uv,
        ring_sides,
        disparities,
        strict=True,
    ):
        right_uv = np.array(
            [
                float(left_uv[0]) - float(value),
                float(left_uv[1]) - vertical_shift,
            ],
            dtype=np.float64,
        )
        try:
            point, gap = triangulate_pixel_pair(
                np.asarray(left_uv, dtype=np.float64),
                right_uv,
                left_camera,
                right_camera,
            )
        except RuntimeError:
            continue
        if float(gap) > cfg.max_triangulation_ray_gap_m:
            continue

        left_error = float(
            np.linalg.norm(left_camera.project_world(point) - left_uv)
        )
        right_error = float(
            np.linalg.norm(right_camera.project_world(point) - right_uv)
        )
        points.append(np.asarray(point, dtype=np.float64))
        used_left_uv.append(np.asarray(left_uv, dtype=np.float64))
        gaps.append(float(gap))
        depths.append(float(np.dot(point - camera_midpoint, view_direction)))
        labels.append(int(side))
        used_disparities.append(float(value))
        reprojection_errors.extend((left_error, right_error))

    if len(points) < cfg.min_cluster_points:
        raise RuntimeError(
            f"Only {len(points)} outer-bezel pixels triangulated cleanly."
        )

    point_array = np.vstack(points)
    pixel_array = np.vstack(used_left_uv)
    depth_array = np.asarray(depths, dtype=np.float64)
    label_array = np.asarray(labels, dtype=np.int64)
    gap_array = np.asarray(gaps, dtype=np.float64)
    disparity_array = np.asarray(used_disparities, dtype=np.float64)

    cluster, support = select_nearest_supported_range_cluster(
        ranges_m=depth_array,
        pixels_uv=pixel_array,
        side_labels=label_array,
        tolerance_m=cfg.depth_cluster_tolerance_m,
        min_points=cfg.min_cluster_points,
    )
    cluster_points = point_array[cluster]
    cluster_pixels = pixel_array[cluster]
    cluster_labels = label_array[cluster]
    cluster_gaps = gap_array[cluster]
    cluster_disparities = disparity_array[cluster]

    plane_center, normal, plane_inliers, residual = fit_plane_stable(
        cluster_points,
        cfg,
    )
    final_support = support_diagnostics(
        cluster_pixels[plane_inliers],
        cluster_labels[plane_inliers],
    )
    if not _support_is_qualified(
        final_support,
        min_supported_regions=_MIN_SUPPORTED_REGIONS,
        min_span_u_px=_MIN_SUPPORT_SPAN_U_PX,
        min_span_v_px=_MIN_SUPPORT_SPAN_V_PX,
        min_minor_std_px=_MIN_SUPPORT_MINOR_STD_PX,
    ):
        raise RuntimeError(
            "Outer-bezel plane inliers lost spatially distributed support."
        )

    toward_cameras = unit_vector(
        camera_midpoint - plane_center,
        "outer-bezel camera direction",
    )
    if float(np.dot(normal, toward_cameras)) < 0.0:
        normal = -normal
    if float(np.dot(normal, toward_cameras)) < cfg.normal_min_camera_cosine:
        raise RuntimeError("Outer-bezel plane normal is not camera-facing.")

    center = intersect_midpoint_ray_with_plane(
        left_camera,
        right_camera,
        left_detection_center_uv,
        right_detection_center_uv,
        plane_center,
        normal,
    )

    left_corners = _bbox_corners(left_bbox_xywh)
    right_corners = _bbox_corners(right_bbox_xywh)
    corners: list[np.ndarray] = []
    for left_uv, right_uv in zip(
        left_corners,
        right_corners,
        strict=True,
    ):
        left_point = intersect_pixel_with_plane(
            left_camera,
            left_uv,
            plane_center,
            normal,
        )
        right_point = intersect_pixel_with_plane(
            right_camera,
            right_uv,
            plane_center,
            normal,
        )
        corners.append(0.5 * (left_point + right_point))
    corners_world = np.vstack(corners)
    corners_world += center - np.mean(corners_world, axis=0)
    width_m = 0.5 * (
        float(np.linalg.norm(corners_world[1] - corners_world[0]))
        + float(np.linalg.norm(corners_world[2] - corners_world[3]))
    )
    height_m = 0.5 * (
        float(np.linalg.norm(corners_world[3] - corners_world[0]))
        + float(np.linalg.norm(corners_world[2] - corners_world[1]))
    )

    reprojection = np.asarray(reprojection_errors, dtype=np.float64)
    return OuterBezelPlaneResult(
        center_world_m=center,
        normal_world=normal,
        corners_world_m=corners_world,
        width_m=width_m,
        height_m=height_m,
        max_ray_gap_m=float(np.max(cluster_gaps[plane_inliers])),
        reprojection_rms_px=float(np.sqrt(np.mean(reprojection**2))),
        max_reprojection_px=float(np.max(reprojection)),
        plane_residual_m=float(residual),
        valid_disparity_count=int(disparity.valid_count),
        consistent_disparity_count=int(disparity.consistent_count),
        ring_candidate_count=ring_candidate_count,
        triangulated_count=len(points),
        cluster_count=int(np.count_nonzero(cluster)),
        side_support_counts=final_support.side_counts,
        support_region_count=final_support.region_count,
        support_span_u_px=final_support.span_u_px,
        support_span_v_px=final_support.span_v_px,
        support_minor_std_px=final_support.minor_std_px,
        median_disparity_px=float(
            np.median(cluster_disparities[plane_inliers])
        ),
        disparity=disparity,
    )


def estimate_outer_bezel_aperture_center(
    *,
    left_rgb: np.ndarray,
    right_rgb: np.ndarray,
    left_mask: np.ndarray,
    right_mask: np.ndarray,
    left_bbox_xywh: tuple[int, int, int, int],
    right_bbox_xywh: tuple[int, int, int, int],
    left_detection_center_uv: tuple[float, float],
    right_detection_center_uv: tuple[float, float],
    left_camera,
    right_camera,
    aperture_width_m: float = 0.0114,
    aperture_height_m: float = 0.0070,
    front_plane_config: FrontPlaneConfig = OUTER_BEZEL_CONFIG,
) -> OuterBezelApertureResult:
    plane = estimate_outer_bezel_plane(
        left_rgb=left_rgb,
        right_rgb=right_rgb,
        left_bbox_xywh=left_bbox_xywh,
        left_detection_center_uv=left_detection_center_uv,
        right_bbox_xywh=right_bbox_xywh,
        right_detection_center_uv=right_detection_center_uv,
        left_camera=left_camera,
        right_camera=right_camera,
        front_plane_config=front_plane_config,
    )
    center = estimate_planar_aperture_center(
        left_mask=left_mask,
        right_mask=right_mask,
        left_camera=left_camera,
        right_camera=right_camera,
        plane_origin_world_m=plane.center_world_m,
        plane_normal_world=plane.normal_world,
        aperture_width_m=aperture_width_m,
        aperture_height_m=aperture_height_m,
    )
    left_uv = left_camera.project_world(center.left_center_world_m)
    right_uv = right_camera.project_world(center.right_center_world_m)
    return OuterBezelApertureResult(
        center_world_m=center.center_world_m,
        left_center_world_m=center.left_center_world_m,
        right_center_world_m=center.right_center_world_m,
        left_center_uv=left_uv,
        right_center_uv=right_uv,
        eye_disagreement_m=center.left_right_disagreement_m,
        plane_origin_world_m=plane.center_world_m,
        plane_normal_world=plane.normal_world,
        corners_world_m=plane.corners_world_m,
        width_m=plane.width_m,
        height_m=plane.height_m,
        plane_residual_m=plane.plane_residual_m,
        max_ray_gap_m=plane.max_ray_gap_m,
        reprojection_rms_px=plane.reprojection_rms_px,
        max_reprojection_px=plane.max_reprojection_px,
        valid_disparity_count=plane.valid_disparity_count,
        consistent_disparity_count=plane.consistent_disparity_count,
        ring_candidate_count=plane.ring_candidate_count,
        triangulated_count=plane.triangulated_count,
        cluster_count=plane.cluster_count,
        side_support_counts=plane.side_support_counts,
        support_region_count=plane.support_region_count,
        support_span_u_px=plane.support_span_u_px,
        support_span_v_px=plane.support_span_v_px,
        support_minor_std_px=plane.support_minor_std_px,
        median_disparity_px=plane.median_disparity_px,
        disparity=plane.disparity,
        front_plane_config=front_plane_config,
    )
