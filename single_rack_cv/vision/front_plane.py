#!/usr/bin/env python3
"""Qualified local SGBM front-opening plane estimation."""

from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np

from vision.stereo_geometry import triangulate_pixel_pair, unit_vector


SIDE_NAMES = ("top", "right", "bottom", "left")
STRICT_RAY_GAP_M = 0.0005


@dataclass(frozen=True)
class FrontPlaneConfig:
    """Dense stereo settings constrained by the detected cavity disparity."""

    roi_margin_px: int = 24
    disparity_half_range_px: float = 24.0
    block_size: int = 5
    uniqueness_ratio: int = 8
    speckle_window_size: int = 50
    speckle_range: int = 2
    disp12_max_diff: int = 1
    pre_filter_cap: int = 31
    lr_consistency_px: float = 1.0

    ring_inner_offset_px: int = 2
    ring_outer_offset_px: int = 10
    ring_sample_stride_px: int = 1

    max_triangulation_ray_gap_m: float = STRICT_RAY_GAP_M
    depth_cluster_tolerance_m: float = 0.0040
    min_cluster_points: int = 24
    min_points_per_side: int = 4

    plane_fit_iterations: int = 4
    plane_mad_scale: float = 2.5
    plane_max_residual_m: float = 0.0005
    normal_min_camera_cosine: float = 0.20


DEFAULT_FRONT_PLANE_CONFIG = FrontPlaneConfig()


@dataclass(frozen=True)
class LocalDisparityResult:
    disparity_crop_px: np.ndarray
    valid_mask: np.ndarray
    consistent_mask: np.ndarray
    crop_xyxy: tuple[int, int, int, int]
    right_vertical_shift_px: float
    center_disparity_px: float
    reverse_min_disparity_px: int

    def __post_init__(self) -> None:
        disparity = np.asarray(self.disparity_crop_px, dtype=np.float32)
        valid = np.asarray(self.valid_mask, dtype=bool)
        consistent = np.asarray(self.consistent_mask, dtype=bool)
        if disparity.ndim != 2:
            raise ValueError("disparity_crop_px must be a 2D array.")
        if valid.shape != disparity.shape or consistent.shape != disparity.shape:
            raise ValueError("Disparity masks must match disparity shape.")
        x0, y0, x1, y1 = map(int, self.crop_xyxy)
        if disparity.shape != (y1 - y0, x1 - x0):
            raise ValueError("Disparity shape does not match crop_xyxy.")
        object.__setattr__(self, "disparity_crop_px", disparity.copy())
        object.__setattr__(self, "valid_mask", valid.copy())
        object.__setattr__(self, "consistent_mask", consistent.copy())

    @property
    def valid_count(self) -> int:
        return int(np.count_nonzero(self.valid_mask))

    @property
    def consistent_count(self) -> int:
        return int(np.count_nonzero(self.consistent_mask))


@dataclass(frozen=True)
class FrontPlaneResult:
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
    median_disparity_px: float
    disparity: LocalDisparityResult

    def __post_init__(self) -> None:
        center = np.asarray(self.center_world_m, dtype=np.float64).reshape(3)
        normal = unit_vector(self.normal_world, "normal_world")
        corners = np.asarray(self.corners_world_m, dtype=np.float64).reshape(4, 3)
        if self.width_m <= 0.0 or self.height_m <= 0.0:
            raise ValueError("Estimated opening dimensions must be positive.")
        object.__setattr__(self, "center_world_m", center.copy())
        object.__setattr__(self, "normal_world", normal)
        object.__setattr__(self, "corners_world_m", corners.copy())


def _validate_config(cfg: FrontPlaneConfig) -> None:
    if cfg.roi_margin_px < 0:
        raise ValueError("roi_margin_px must be non-negative.")
    if cfg.disparity_half_range_px <= 0.0:
        raise ValueError("disparity_half_range_px must be positive.")
    if cfg.block_size < 3 or cfg.block_size % 2 == 0:
        raise ValueError("block_size must be odd and at least 3.")
    if cfg.lr_consistency_px <= 0.0:
        raise ValueError("lr_consistency_px must be positive.")
    if not 0 <= cfg.ring_inner_offset_px < cfg.ring_outer_offset_px:
        raise ValueError("Bezel ring offsets are invalid.")
    if cfg.ring_sample_stride_px < 1:
        raise ValueError("ring_sample_stride_px must be positive.")
    if cfg.min_cluster_points < 3 or cfg.min_points_per_side < 1:
        raise ValueError("SGBM support requirements are invalid.")
    if cfg.max_triangulation_ray_gap_m > STRICT_RAY_GAP_M:
        raise ValueError("Front-plane ray-gap gate may not exceed 0.5 mm.")


def _bbox_edges(
    bbox_xywh: tuple[int, int, int, int],
) -> tuple[float, float, float, float]:
    x, y, width, height = map(float, bbox_xywh)
    if width <= 1.0 or height <= 1.0:
        raise ValueError("Cavity boxes must have positive area.")
    return x, y, x + width, y + height


def _local_crop(
    image_shape_hw: tuple[int, int],
    left_bbox_xywh: tuple[int, int, int, int],
    right_bbox_xywh: tuple[int, int, int, int],
    right_vertical_shift_px: float,
    margin_px: int,
) -> tuple[int, int, int, int]:
    height, width = map(int, image_shape_hw)
    lx0, ly0, lx1, ly1 = _bbox_edges(left_bbox_xywh)
    rx0, ry0, rx1, ry1 = _bbox_edges(right_bbox_xywh)
    ry0 += right_vertical_shift_px
    ry1 += right_vertical_shift_px
    x0 = max(0, int(math.floor(min(lx0, rx0) - margin_px)))
    y0 = max(0, int(math.floor(min(ly0, ry0) - margin_px)))
    x1 = min(width, int(math.ceil(max(lx1, rx1) + margin_px)))
    y1 = min(height, int(math.ceil(max(ly1, ry1) + margin_px)))
    if x1 - x0 < 32 or y1 - y0 < 16:
        raise RuntimeError("Local SGBM crop is too small.")
    return x0, y0, x1, y1


def _num_disparities(half_range_px: float) -> int:
    span = 2.0 * float(half_range_px) + 2.0
    return max(16, int(math.ceil(span / 16.0)) * 16)


def _create_matcher(
    min_disparity_px: int,
    num_disparities: int,
    cfg: FrontPlaneConfig,
):
    block = int(cfg.block_size)
    return cv2.StereoSGBM_create(
        minDisparity=int(min_disparity_px),
        numDisparities=int(num_disparities),
        blockSize=block,
        P1=8 * block * block,
        P2=32 * block * block,
        disp12MaxDiff=int(cfg.disp12_max_diff),
        preFilterCap=int(cfg.pre_filter_cap),
        uniquenessRatio=int(cfg.uniqueness_ratio),
        speckleWindowSize=int(cfg.speckle_window_size),
        speckleRange=int(cfg.speckle_range),
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )


def compute_local_disparity(
    left_rgb: np.ndarray,
    right_rgb: np.ndarray,
    left_bbox_xywh: tuple[int, int, int, int],
    left_center_uv: tuple[float, float] | np.ndarray,
    right_bbox_xywh: tuple[int, int, int, int],
    right_center_uv: tuple[float, float] | np.ndarray,
    cfg: FrontPlaneConfig = DEFAULT_FRONT_PLANE_CONFIG,
) -> LocalDisparityResult:
    """Compute local left disparity and explicit left-right consistency."""
    _validate_config(cfg)
    left = np.asarray(left_rgb)
    right = np.asarray(right_rgb)
    if left.shape != right.shape or left.ndim != 3 or left.shape[2] != 3:
        raise ValueError("Stereo RGB images must have the same HxWx3 shape.")

    left_center = np.asarray(left_center_uv, dtype=np.float64).reshape(2)
    right_center = np.asarray(right_center_uv, dtype=np.float64).reshape(2)
    vertical_shift = float(left_center[1] - right_center[1])
    transform = np.array(
        [[1.0, 0.0, 0.0], [0.0, 1.0, vertical_shift]],
        dtype=np.float32,
    )
    right_aligned = cv2.warpAffine(
        right,
        transform,
        (right.shape[1], right.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT101,
    )

    center_disparity = float(left_center[0] - right_center[0])
    num_disparities = _num_disparities(cfg.disparity_half_range_px)
    horizontal_margin = max(
        cfg.roi_margin_px,
        num_disparities + cfg.block_size,
    )
    crop = _local_crop(
        left.shape[:2],
        left_bbox_xywh,
        right_bbox_xywh,
        vertical_shift,
        horizontal_margin,
    )
    x0, y0, x1, y1 = crop
    left_gray = cv2.cvtColor(left[y0:y1, x0:x1], cv2.COLOR_RGB2GRAY)
    right_gray = cv2.cvtColor(
        right_aligned[y0:y1, x0:x1],
        cv2.COLOR_RGB2GRAY,
    )
    if left_gray.shape[1] <= num_disparities + cfg.block_size + 2:
        raise RuntimeError(
            "Local SGBM crop is narrower than its disparity search."
        )

    min_left = int(
        math.floor(center_disparity - cfg.disparity_half_range_px)
    )
    left_disparity = (
        _create_matcher(min_left, num_disparities, cfg)
        .compute(left_gray, right_gray)
        .astype(np.float32)
        / 16.0
    )
    valid_left = left_disparity > float(min_left) - 0.5

    min_right = int(
        math.floor(-center_disparity - cfg.disparity_half_range_px)
    )
    right_disparity = (
        _create_matcher(min_right, num_disparities, cfg)
        .compute(right_gray, left_gray)
        .astype(np.float32)
        / 16.0
    )
    valid_right = right_disparity > float(min_right) - 0.5

    rows, columns = np.indices(left_disparity.shape)
    right_columns = np.rint(columns - left_disparity).astype(np.int64)
    inside = (
        (right_columns >= 0)
        & (right_columns < left_disparity.shape[1])
    )
    consistent = np.zeros_like(valid_left)
    candidate_rows, candidate_columns = np.where(valid_left & inside)
    mapped_columns = right_columns[candidate_rows, candidate_columns]
    reverse_values = right_disparity[candidate_rows, mapped_columns]
    consistent[candidate_rows, candidate_columns] = (
        valid_right[candidate_rows, mapped_columns]
        & (
            np.abs(
                left_disparity[candidate_rows, candidate_columns]
                + reverse_values
            )
            <= cfg.lr_consistency_px
        )
    )
    return LocalDisparityResult(
        disparity_crop_px=left_disparity,
        valid_mask=valid_left,
        consistent_mask=consistent,
        crop_xyxy=crop,
        right_vertical_shift_px=vertical_shift,
        center_disparity_px=center_disparity,
        reverse_min_disparity_px=min_right,
    )


def build_bezel_ring_pixels(
    bbox_xywh: tuple[int, int, int, int],
    image_shape_hw: tuple[int, int],
    cfg: FrontPlaneConfig = DEFAULT_FRONT_PLANE_CONFIG,
) -> tuple[np.ndarray, np.ndarray]:
    """Return disjoint full-image pixels for top/right/bottom/left bands."""
    _validate_config(cfg)
    image_height, image_width = map(int, image_shape_hw)
    x0f, y0f, x1f, y1f = _bbox_edges(bbox_xywh)
    x0, y0, x1, y1 = map(
        int,
        (round(x0f), round(y0f), round(x1f), round(y1f)),
    )
    inner = int(cfg.ring_inner_offset_px)
    outer = int(cfg.ring_outer_offset_px)
    stride = int(cfg.ring_sample_stride_px)
    rectangles = (
        (x0 - outer, y0 - outer, x1 + outer, y0 - inner),
        (x1 + inner, y0 - inner, x1 + outer, y1 + inner),
        (x0 - outer, y1 + inner, x1 + outer, y1 + outer),
        (x0 - outer, y0 - inner, x0 - inner, y1 + inner),
    )
    points: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for side_index, (rx0, ry0, rx1, ry1) in enumerate(rectangles):
        rx0 = max(0, rx0)
        ry0 = max(0, ry0)
        rx1 = min(image_width, rx1)
        ry1 = min(image_height, ry1)
        if rx1 <= rx0 or ry1 <= ry0:
            continue
        xs = np.arange(rx0, rx1, stride, dtype=np.int64)
        ys = np.arange(ry0, ry1, stride, dtype=np.int64)
        grid_x, grid_y = np.meshgrid(xs, ys)
        side_points = np.column_stack(
            (grid_x.reshape(-1), grid_y.reshape(-1))
        )
        points.append(side_points)
        labels.append(
            np.full(side_points.shape[0], side_index, dtype=np.int64)
        )
    if not points:
        raise RuntimeError("No front-bezel ring pixels fit inside the image.")
    return np.vstack(points), np.concatenate(labels)


def select_nearest_range_cluster(
    ranges_m: np.ndarray,
    tolerance_m: float,
    min_points: int,
) -> np.ndarray:
    """Select densest range window, breaking ties toward the camera."""
    ranges = np.asarray(ranges_m, dtype=np.float64).reshape(-1)
    if ranges.size < min_points:
        raise RuntimeError(
            f"Only {ranges.size} triangulated bezel points were available."
        )
    order = np.argsort(ranges)
    values = ranges[order]
    best_start = 0
    best_end = -1
    end = 0
    for start in range(values.size):
        end = max(end, start)
        while (
            end + 1 < values.size
            and values[end + 1] - values[start] <= tolerance_m
        ):
            end += 1
        best_count = best_end - best_start + 1
        count = end - start + 1
        if count > best_count:
            best_start, best_end = start, end
        elif count == best_count and count > 0:
            candidate_median = float(
                np.median(values[start : end + 1])
            )
            best_median = float(
                np.median(values[best_start : best_end + 1])
            )
            if candidate_median < best_median:
                best_start, best_end = start, end
    count = best_end - best_start + 1
    if count < min_points:
        raise RuntimeError(
            f"Nearest coherent depth cluster has only {count} points; "
            f"need {min_points}."
        )
    mask = np.zeros(ranges.size, dtype=bool)
    mask[order[best_start : best_end + 1]] = True
    return mask


def fit_plane_stable(
    points_world_m: np.ndarray,
    cfg: FrontPlaneConfig = DEFAULT_FRONT_PLANE_CONFIG,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Fit with monotonic robust trimming and final hard stabilization."""
    points = np.asarray(points_world_m, dtype=np.float64).reshape(-1, 3)
    if points.shape[0] < cfg.min_cluster_points:
        raise RuntimeError("Too few points reached front-plane fitting.")
    inliers = np.ones(points.shape[0], dtype=bool)
    max_iterations = max(8, 2 * int(cfg.plane_fit_iterations))

    for _ in range(max_iterations):
        active = points[inliers]
        center = np.mean(active, axis=0)
        _, _, vh = np.linalg.svd(active - center, full_matrices=False)
        normal = unit_vector(vh[-1], "front-plane normal")
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
                "Front-plane fit rejected too many depth points."
            )
        if np.array_equal(next_inliers, inliers):
            hard = inliers & (residuals <= cfg.plane_max_residual_m)
            if int(np.count_nonzero(hard)) < cfg.min_cluster_points:
                raise RuntimeError(
                    "Front-plane hard residual pass rejected too many points."
                )
            if np.array_equal(hard, inliers):
                return (
                    center,
                    normal,
                    inliers,
                    float(np.max(residuals[inliers])),
                )
            next_inliers = hard
        inliers = next_inliers

    for _ in range(max_iterations):
        active = points[inliers]
        center = np.mean(active, axis=0)
        _, _, vh = np.linalg.svd(active - center, full_matrices=False)
        normal = unit_vector(vh[-1], "front-plane normal")
        residuals = np.abs((points - center) @ normal)
        next_inliers = inliers & (
            residuals <= cfg.plane_max_residual_m
        )
        if int(np.count_nonzero(next_inliers)) < cfg.min_cluster_points:
            raise RuntimeError(
                "Final residual pass rejected too many points."
            )
        if np.array_equal(next_inliers, inliers):
            return (
                center,
                normal,
                inliers,
                float(np.max(residuals[inliers])),
            )
        inliers = next_inliers
    raise RuntimeError("Front-plane inliers did not stabilize.")


def intersect_pixel_with_plane(
    camera,
    pixel_uv,
    plane_center_world_m,
    plane_normal_world,
):
    origin, direction = camera.pixel_to_world_ray(pixel_uv)
    origin = np.asarray(origin, dtype=np.float64).reshape(3)
    direction = unit_vector(direction, "image ray")
    normal = unit_vector(plane_normal_world, "plane normal")
    denominator = float(np.dot(direction, normal))
    if abs(denominator) <= 1.0e-9:
        raise RuntimeError("Image ray is parallel to the front plane.")
    distance = float(
        np.dot(
            np.asarray(plane_center_world_m, dtype=np.float64) - origin,
            normal,
        )
        / denominator
    )
    if distance <= 0.0:
        raise RuntimeError("Front plane lies behind an image ray.")
    return origin + distance * direction


def intersect_midpoint_ray_with_plane(
    left_camera,
    right_camera,
    left_center_uv,
    right_center_uv,
    plane_center_world_m,
    plane_normal_world,
) -> np.ndarray:
    left_origin, left_direction = left_camera.pixel_to_world_ray(
        left_center_uv
    )
    right_origin, right_direction = right_camera.pixel_to_world_ray(
        right_center_uv
    )
    origin = 0.5 * (
        np.asarray(left_origin, dtype=np.float64).reshape(3)
        + np.asarray(right_origin, dtype=np.float64).reshape(3)
    )
    direction = unit_vector(
        np.asarray(left_direction, dtype=np.float64).reshape(3)
        + np.asarray(right_direction, dtype=np.float64).reshape(3),
        "fused cavity-center direction",
    )
    normal = unit_vector(plane_normal_world, "front-plane normal")
    denominator = float(np.dot(direction, normal))
    if abs(denominator) <= 1.0e-9:
        raise RuntimeError(
            "Fused cavity-center ray is parallel to front plane."
        )
    distance = float(
        np.dot(
            np.asarray(plane_center_world_m, dtype=np.float64) - origin,
            normal,
        )
        / denominator
    )
    if distance <= 0.0:
        raise RuntimeError(
            "Front plane lies behind fused cavity-center ray."
        )
    return origin + distance * direction


def _bbox_corners(
    bbox_xywh: tuple[int, int, int, int],
) -> np.ndarray:
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


def estimate_front_plane(
    left_rgb: np.ndarray,
    right_rgb: np.ndarray,
    left_bbox_xywh: tuple[int, int, int, int],
    left_center_uv: tuple[float, float] | np.ndarray,
    right_bbox_xywh: tuple[int, int, int, int],
    right_center_uv: tuple[float, float] | np.ndarray,
    left_camera,
    right_camera,
    cfg: FrontPlaneConfig = DEFAULT_FRONT_PLANE_CONFIG,
    disparity: LocalDisparityResult | None = None,
) -> FrontPlaneResult:
    """Estimate the physical front-opening plane from a cavity-anchored ROI."""
    _validate_config(cfg)
    if disparity is None:
        disparity = compute_local_disparity(
            left_rgb,
            right_rgb,
            left_bbox_xywh,
            left_center_uv,
            right_bbox_xywh,
            right_center_uv,
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
            f"Only {ring_candidate_count} consistent SGBM bezel pixels "
            "were available."
        )

    points: list[np.ndarray] = []
    gaps: list[float] = []
    depths: list[float] = []
    labels: list[int] = []
    used_disparities: list[float] = []
    reprojection_errors: list[float] = []
    camera_midpoint = 0.5 * (
        np.asarray(left_camera.camera_center_world_m, dtype=np.float64)
        + np.asarray(right_camera.camera_center_world_m, dtype=np.float64)
    )
    _, left_center_direction = left_camera.pixel_to_world_ray(
        left_center_uv
    )
    _, right_center_direction = right_camera.pixel_to_world_ray(
        right_center_uv
    )
    view_direction = unit_vector(
        np.asarray(left_center_direction, dtype=np.float64)
        + np.asarray(right_center_direction, dtype=np.float64),
        "stereo center viewing direction",
    )
    vertical_shift = disparity.right_vertical_shift_px

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
        if gap > cfg.max_triangulation_ray_gap_m:
            continue
        left_error = float(
            np.linalg.norm(left_camera.project_world(point) - left_uv)
        )
        right_error = float(
            np.linalg.norm(right_camera.project_world(point) - right_uv)
        )
        points.append(point)
        gaps.append(float(gap))
        depths.append(
            float(np.dot(point - camera_midpoint, view_direction))
        )
        labels.append(int(side))
        used_disparities.append(float(value))
        reprojection_errors.extend((left_error, right_error))

    if len(points) < cfg.min_cluster_points:
        raise RuntimeError(
            f"Only {len(points)} SGBM bezel pixels triangulated cleanly."
        )
    point_array = np.vstack(points)
    depth_array = np.asarray(depths, dtype=np.float64)
    label_array = np.asarray(labels, dtype=np.int64)
    gap_array = np.asarray(gaps, dtype=np.float64)
    disparity_array = np.asarray(used_disparities, dtype=np.float64)

    cluster = select_nearest_range_cluster(
        depth_array,
        cfg.depth_cluster_tolerance_m,
        cfg.min_cluster_points,
    )
    cluster_points = point_array[cluster]
    cluster_labels = label_array[cluster]
    cluster_gaps = gap_array[cluster]
    cluster_disparities = disparity_array[cluster]
    side_counts = tuple(
        int(np.count_nonzero(cluster_labels == side_index))
        for side_index in range(4)
    )
    if min(side_counts) < cfg.min_points_per_side:
        raise RuntimeError(
            "Nearest SGBM depth cluster lacks four-sided bezel support: "
            + ", ".join(
                f"{name}={count}"
                for name, count in zip(
                    SIDE_NAMES,
                    side_counts,
                    strict=True,
                )
            )
        )

    plane_center, normal, plane_inliers, residual = fit_plane_stable(
        cluster_points,
        cfg,
    )
    final_labels = cluster_labels[plane_inliers]
    final_side_counts = tuple(
        int(np.count_nonzero(final_labels == side_index))
        for side_index in range(4)
    )
    if min(final_side_counts) < cfg.min_points_per_side:
        raise RuntimeError(
            "Front-plane inliers lost four-sided bezel support: "
            + ", ".join(
                f"{name}={count}"
                for name, count in zip(
                    SIDE_NAMES,
                    final_side_counts,
                    strict=True,
                )
            )
        )

    toward_cameras = unit_vector(
        camera_midpoint - plane_center,
        "front-plane camera direction",
    )
    if float(np.dot(normal, toward_cameras)) < 0.0:
        normal = -normal
    if (
        float(np.dot(normal, toward_cameras))
        < cfg.normal_min_camera_cosine
    ):
        raise RuntimeError("Front-plane normal is not camera-facing.")

    center = intersect_midpoint_ray_with_plane(
        left_camera,
        right_camera,
        left_center_uv,
        right_center_uv,
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
    return FrontPlaneResult(
        center_world_m=center,
        normal_world=normal,
        corners_world_m=corners_world,
        width_m=width_m,
        height_m=height_m,
        max_ray_gap_m=float(np.max(cluster_gaps[plane_inliers])),
        reprojection_rms_px=float(
            np.sqrt(np.mean(reprojection * reprojection))
        ),
        max_reprojection_px=float(np.max(reprojection)),
        plane_residual_m=residual,
        valid_disparity_count=disparity.valid_count,
        consistent_disparity_count=disparity.consistent_count,
        ring_candidate_count=ring_candidate_count,
        triangulated_count=len(points),
        cluster_count=int(np.count_nonzero(cluster)),
        side_support_counts=final_side_counts,
        median_disparity_px=float(
            np.median(cluster_disparities[plane_inliers])
        ),
        disparity=disparity,
    )
