#!/usr/bin/env python3
"""Pure NumPy/OpenCV stereo RGB port perception and servo geometry."""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from config import PerceptionConfig


@dataclass(frozen=True)
class CameraModel:
    """Pinhole calibration plus the current USD camera-to-world matrix."""

    image_height_px: int
    image_width_px: int
    focal_length_mm: float
    horizontal_aperture_mm: float
    vertical_aperture_mm: float
    world_from_camera: np.ndarray

    def __post_init__(self) -> None:
        matrix = np.asarray(self.world_from_camera, dtype=np.float64)
        if matrix.shape != (4, 4):
            raise ValueError(
                f"world_from_camera must be 4x4, got {matrix.shape}."
            )
        object.__setattr__(self, "world_from_camera", matrix.copy())

    @property
    def fx_px(self) -> float:
        return (
            self.focal_length_mm
            * self.image_width_px
            / self.horizontal_aperture_mm
        )

    @property
    def fy_px(self) -> float:
        return (
            self.focal_length_mm
            * self.image_height_px
            / self.vertical_aperture_mm
        )

    @property
    def cx_px(self) -> float:
        return (self.image_width_px - 1) / 2.0

    @property
    def cy_px(self) -> float:
        return (self.image_height_px - 1) / 2.0

    @property
    def camera_from_world(self) -> np.ndarray:
        return np.linalg.inv(self.world_from_camera)

    @property
    def camera_center_world_m(self) -> np.ndarray:
        return transform_point_to_world(
            np.zeros(3, dtype=np.float64),
            self.world_from_camera,
        )

    def camera_point_from_world(self, point_world_m: np.ndarray) -> np.ndarray:
        point = np.append(
            np.asarray(point_world_m, dtype=np.float64).reshape(3),
            1.0,
        )
        local = point @ self.camera_from_world
        if abs(local[3]) > 1.0e-12:
            local = local / local[3]
        return local[:3]

    def project_world(self, point_world_m: np.ndarray) -> np.ndarray:
        point = self.camera_point_from_world(point_world_m)
        range_m = -float(point[2])
        if range_m <= 0.0:
            raise RuntimeError("World point is behind the camera.")
        u = self.cx_px + self.fx_px * float(point[0]) / range_m
        v = self.cy_px + self.fy_px * (-float(point[1])) / range_m
        return np.array([u, v], dtype=np.float64)

    def pixel_to_world_ray(
        self,
        pixel_uv: np.ndarray | tuple[float, float],
    ) -> tuple[np.ndarray, np.ndarray]:
        u, v = np.asarray(pixel_uv, dtype=np.float64).reshape(2)
        x = (u - self.cx_px) / self.fx_px
        y = -(v - self.cy_px) / self.fy_px
        local_direction = np.array([x, y, -1.0], dtype=np.float64)
        local_direction /= np.linalg.norm(local_direction)
        world_direction = (
            np.append(local_direction, 0.0) @ self.world_from_camera
        )[:3]
        world_direction /= np.linalg.norm(world_direction)
        return self.camera_center_world_m, world_direction


@dataclass(frozen=True)
class CameraFrame:
    rgb: np.ndarray
    camera: CameraModel


@dataclass(frozen=True)
class StereoFrame:
    left: CameraFrame
    right: CameraFrame
    virtual_camera: CameraModel


@dataclass(frozen=True)
class PortDetection:
    bbox_xywh: tuple[int, int, int, int]
    center_uv: tuple[float, float]
    shape_score: float
    roi_uv: tuple[int, int, int, int]
    mask: np.ndarray

    @property
    def scale_px(self) -> float:
        _, _, width, height = self.bbox_xywh
        return math.sqrt(float(width * height))


@dataclass(frozen=True)
class PortCorners:
    detection: PortDetection
    corners_uv: np.ndarray

    def __post_init__(self) -> None:
        corners = np.asarray(self.corners_uv, dtype=np.float64)
        if corners.shape != (4, 2):
            raise ValueError(f"corners_uv must be (4,2), got {corners.shape}.")
        object.__setattr__(self, "corners_uv", corners.copy())


@dataclass(frozen=True)
class StereoTriangulation:
    corners_world_m: np.ndarray
    center_world_m: np.ndarray
    normal_world: np.ndarray
    width_m: float
    height_m: float
    reprojection_rms_px: float
    max_reprojection_px: float
    max_ray_gap_m: float
    plane_residual_m: float
    opposite_edge_ratio: float


@dataclass(frozen=True)
class StereoPortObservation:
    left: PortCorners
    right: PortCorners
    corners_world_m: np.ndarray
    center_world_xyz_m: np.ndarray
    center_virtual_camera_usd_m: np.ndarray
    normal_world: np.ndarray
    projected_virtual_center_uv: tuple[float, float]
    desired_center_uv: tuple[float, float]
    desired_size_wh_px: tuple[float, float]
    desired_left_center_uv: tuple[float, float]
    desired_right_center_uv: tuple[float, float]
    center_error_px: np.ndarray
    estimated_range_m: float
    range_error_m: float
    correction_world_m: np.ndarray
    width_m: float
    height_m: float
    mean_disparity_px: float
    reprojection_rms_px: float
    max_reprojection_px: float
    max_ray_gap_m: float
    plane_residual_m: float


# ---------------------------------------------------------------------------
# Image normalization and strict per-eye candidate detection
# ---------------------------------------------------------------------------


def normalize_rgb(
    rgb: np.ndarray,
    resolution_hw: tuple[int, int],
) -> np.ndarray:
    """Return contiguous HxWx3 uint8 RGB data."""
    height, width = resolution_hw
    rgb = np.asarray(rgb)

    if rgb.ndim == 1:
        pixels = height * width
        channels = 4 if rgb.size == pixels * 4 else 3
        if rgb.size != pixels * channels:
            raise ValueError(
                f"Cannot reshape flat RGB array of size {rgb.size}."
            )
        rgb = rgb.reshape(height, width, channels)

    if rgb.ndim != 3 or rgb.shape[:2] != (height, width):
        raise ValueError(
            f"RGB shape {rgb.shape} does not match {(height, width)}."
        )

    if rgb.shape[2] == 4:
        rgb = rgb[:, :, :3]
    elif rgb.shape[2] != 3:
        raise ValueError(f"RGB must have 3 or 4 channels, got {rgb.shape}.")

    if rgb.dtype != np.uint8:
        rgb = rgb.astype(np.float32, copy=False)
        if np.nanmax(rgb) <= 1.0:
            rgb = rgb * 255.0
        rgb = np.clip(rgb, 0.0, 255.0).astype(np.uint8)

    return np.ascontiguousarray(rgb)


def score_port_shape(
    aspect_ratio: float,
    fill_ratio: float,
    cfg: PerceptionConfig,
) -> float:
    aspect_error = (
        abs(aspect_ratio - cfg.target_aspect_ratio)
        / cfg.target_aspect_ratio
    )
    fill_error = abs(fill_ratio - cfg.target_fill_ratio)
    aspect_score = max(
        0.0,
        1.0 - aspect_error / cfg.aspect_score_tolerance,
    )
    fill_score = max(
        0.0,
        1.0 - fill_error / cfg.fill_score_tolerance,
    )
    weight_sum = cfg.aspect_score_weight + cfg.fill_score_weight
    if weight_sum <= 0.0:
        raise ValueError("Shape-score weights must sum to a positive value.")
    return float(
        (
            cfg.aspect_score_weight * aspect_score
            + cfg.fill_score_weight * fill_score
        )
        / weight_sum
    )


def _has_bright_surround(
    gray_roi: np.ndarray,
    bbox_xywh: tuple[int, int, int, int],
    cfg: PerceptionConfig,
) -> bool:
    x, y, width, height = bbox_xywh
    ring_width = cfg.surround_ring_px
    image_height, image_width = gray_roi.shape
    x0 = max(0, x - ring_width)
    y0 = max(0, y - ring_width)
    x1 = min(image_width, x + width + ring_width)
    y1 = min(image_height, y + height + ring_width)
    patch = gray_roi[y0:y1, x0:x1]
    ring_mask = np.ones(patch.shape, dtype=bool)
    inner_x0 = x - x0
    inner_y0 = y - y0
    ring_mask[
        inner_y0:inner_y0 + height,
        inner_x0:inner_x0 + width,
    ] = False
    surround = patch[ring_mask]
    cavity = gray_roi[y:y + height, x:x + width]
    if surround.size == 0 or cavity.size == 0:
        return False
    surround_mean = float(np.mean(surround))
    cavity_mean = float(np.mean(cavity))
    return (
        surround_mean >= cfg.min_surround_mean_gray
        and surround_mean - cavity_mean
        >= cfg.min_surround_contrast_gray
    )


def detect_port_candidates(
    rgb: np.ndarray,
    cfg: PerceptionConfig,
) -> list[PortDetection]:
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 RGB, got {rgb.shape}.")
    image_height, image_width = rgb.shape[:2]
    if cfg.roi_uv is None:
        u0, v0, u1, v1 = 0, 0, image_width, image_height
    else:
        u0, v0, u1, v1 = cfg.roi_uv
    if not (
        0 <= u0 < u1 <= image_width
        and 0 <= v0 < v1 <= image_height
    ):
        raise ValueError(
            f"Search area {(u0, v0, u1, v1)} is outside "
            f"image {image_width}x{image_height}."
        )
    roi = rgb[v0:v1, u0:u1]
    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    binary = cv2.inRange(gray, 0, cfg.max_gray)
    count, _, stats, centroids = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8,
    )
    full_mask = np.zeros((image_height, image_width), dtype=np.uint8)
    full_mask[v0:v1, u0:u1] = binary
    candidates: list[PortDetection] = []
    for index in range(1, count):
        x, y, width, height, area = map(int, stats[index])
        if width <= 0 or height <= 0:
            continue
        global_x = u0 + x
        global_y = v0 + y
        aspect_ratio = width / height
        fill_ratio = area / (width * height)
        touches_edge = (
            global_x < cfg.edge_margin_px
            or global_y < cfg.edge_margin_px
            or global_x + width > image_width - cfg.edge_margin_px
            or global_y + height > image_height - cfg.edge_margin_px
        )
        accepted = (
            not touches_edge
            and cfg.min_width_px <= width <= cfg.max_width_px
            and cfg.min_height_px <= height <= cfg.max_height_px
            and cfg.min_aspect_ratio <= aspect_ratio <= cfg.max_aspect_ratio
            and cfg.min_area_px <= area <= cfg.max_area_px
            and fill_ratio >= cfg.min_fill_ratio
            and _has_bright_surround(gray, (x, y, width, height), cfg)
        )
        if not accepted:
            continue
        shape_score = score_port_shape(aspect_ratio, fill_ratio, cfg)
        if shape_score < cfg.min_shape_score:
            continue
        local_center_u, local_center_v = centroids[index]
        candidates.append(
            PortDetection(
                bbox_xywh=(global_x, global_y, width, height),
                center_uv=(
                    float(u0 + local_center_u),
                    float(v0 + local_center_v),
                ),
                shape_score=shape_score,
                roi_uv=(u0, v0, u1, v1),
                mask=full_mask,
            )
        )
    candidates.sort(key=lambda item: item.shape_score, reverse=True)
    return candidates


def select_port_candidate(
    candidates: list[PortDetection],
    previous: PortDetection | None,
    cfg: PerceptionConfig,
) -> PortDetection:
    if not candidates:
        raise RuntimeError("No RGB port candidate passed the shape filters.")
    if previous is None:
        return candidates[0]
    previous_center = np.asarray(previous.center_uv, dtype=np.float64)
    previous_scale = previous.scale_px
    max_log_scale = math.log(cfg.tracking_max_scale_ratio)
    tracked: list[tuple[float, PortDetection]] = []
    for candidate in candidates:
        center_distance = float(
            np.linalg.norm(
                np.asarray(candidate.center_uv, dtype=np.float64)
                - previous_center
            )
        )
        scale_ratio = max(
            candidate.scale_px / previous_scale,
            previous_scale / candidate.scale_px,
        )
        if center_distance > cfg.tracking_max_center_jump_px:
            continue
        if scale_ratio > cfg.tracking_max_scale_ratio:
            continue
        center_penalty = (
            cfg.tracking_center_penalty
            * center_distance
            / cfg.tracking_max_center_jump_px
        )
        scale_penalty = (
            cfg.tracking_scale_penalty
            * abs(math.log(scale_ratio))
            / max_log_scale
        )
        tracked.append(
            (candidate.shape_score - center_penalty - scale_penalty, candidate)
        )
    if not tracked:
        raise RuntimeError(
            "RGB port track was lost: no candidate passed the "
            "center/scale continuity gates."
        )
    return max(tracked, key=lambda item: item[0])[1]


# ---------------------------------------------------------------------------
# Camera and servo geometry
# ---------------------------------------------------------------------------


def compute_desired_port_camera_usd(
    camera_position_hand_m: np.ndarray,
    hand_from_camera: np.ndarray,
    tool_center_position_hand_m: np.ndarray,
    hand_from_tool: np.ndarray,
    preinsert_standoff_m: float,
) -> np.ndarray:
    camera_position = np.asarray(
        camera_position_hand_m,
        dtype=np.float64,
    ).reshape(3)
    tool_position = np.asarray(
        tool_center_position_hand_m,
        dtype=np.float64,
    ).reshape(3)
    hand_from_camera = np.asarray(
        hand_from_camera,
        dtype=np.float64,
    ).reshape(3, 3)
    hand_from_tool = np.asarray(
        hand_from_tool,
        dtype=np.float64,
    ).reshape(3, 3)
    if not math.isfinite(preinsert_standoff_m) or preinsert_standoff_m <= 0.0:
        raise ValueError("preinsert_standoff_m must be finite and positive.")
    port_in_tool = np.array([0.0, 0.0, preinsert_standoff_m])
    port_in_hand = tool_position + hand_from_tool @ port_in_tool
    port_in_camera = hand_from_camera.T @ (port_in_hand - camera_position)
    if port_in_camera[2] >= 0.0:
        raise RuntimeError(
            "Configured pre-insert point is not in front of the camera: "
            f"{np.round(port_in_camera, 6).tolist()}"
        )
    return port_in_camera


def project_port_feature(
    point_camera_usd_m: np.ndarray,
    camera: CameraModel,
    cfg: PerceptionConfig,
) -> tuple[tuple[float, float], tuple[float, float]]:
    point = np.asarray(point_camera_usd_m, dtype=np.float64).reshape(3)
    range_m = -float(point[2])
    if range_m <= 0.0:
        raise ValueError("Desired port point must be in front of the camera.")
    u = camera.cx_px + camera.fx_px * float(point[0]) / range_m
    v = camera.cy_px + camera.fy_px * (-float(point[1])) / range_m
    width_px = camera.fx_px * cfg.port_width_m / range_m
    height_px = camera.fy_px * cfg.port_height_m / range_m
    return (float(u), float(v)), (float(width_px), float(height_px))


def transform_point_to_world(
    point_usd_local: np.ndarray,
    world_from_camera: np.ndarray,
) -> np.ndarray:
    matrix = np.asarray(world_from_camera, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"Expected a 4x4 camera matrix, got {matrix.shape}.")
    homogeneous = np.append(
        np.asarray(point_usd_local, dtype=np.float64),
        1.0,
    )
    world = homogeneous @ matrix
    if abs(world[3]) > 1.0e-12:
        world = world / world[3]
    return world[:3]


def camera_point_error_to_world(
    current_point_usd: np.ndarray,
    desired_point_usd: np.ndarray,
    world_from_camera: np.ndarray,
) -> np.ndarray:
    current = np.asarray(current_point_usd, dtype=np.float64).reshape(3)
    desired = np.asarray(desired_point_usd, dtype=np.float64).reshape(3)
    matrix = np.asarray(world_from_camera, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"Expected a 4x4 camera matrix, got {matrix.shape}.")
    local_motion = current - desired
    return np.asarray((np.append(local_motion, 0.0) @ matrix)[:3])


def compute_bounded_step(
    correction_world_m: np.ndarray,
    gain: float,
    max_step_m: float,
) -> np.ndarray:
    correction = np.asarray(correction_world_m, dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(correction)):
        raise ValueError("Visual correction must be finite.")
    if not math.isfinite(gain) or gain <= 0.0:
        raise ValueError("Control gain must be finite and positive.")
    if not math.isfinite(max_step_m) or max_step_m <= 0.0:
        raise ValueError("Maximum target step must be finite and positive.")
    step = gain * correction
    norm = float(np.linalg.norm(step))
    if norm > max_step_m:
        step *= max_step_m / norm
    return step


def build_virtual_camera_model(
    left: CameraModel,
    right: CameraModel,
) -> CameraModel:
    """Return a mathematical midpoint eye; no rendered sensor is created."""
    scalar_pairs = (
        (left.image_height_px, right.image_height_px),
        (left.image_width_px, right.image_width_px),
        (left.focal_length_mm, right.focal_length_mm),
        (left.horizontal_aperture_mm, right.horizontal_aperture_mm),
        (left.vertical_aperture_mm, right.vertical_aperture_mm),
    )
    for a, b in scalar_pairs:
        if not np.isclose(a, b, atol=1.0e-9):
            raise ValueError("Stereo cameras must have identical intrinsics.")
    left_rotation = left.world_from_camera[:3, :3]
    right_rotation = right.world_from_camera[:3, :3]
    if not np.allclose(left_rotation, right_rotation, atol=1.0e-8):
        raise ValueError("Stereo cameras must have parallel optical axes.")
    matrix = left.world_from_camera.copy()
    matrix[3, :3] = (
        left.camera_center_world_m + right.camera_center_world_m
    ) / 2.0
    return CameraModel(
        image_height_px=left.image_height_px,
        image_width_px=left.image_width_px,
        focal_length_mm=left.focal_length_mm,
        horizontal_aperture_mm=left.horizontal_aperture_mm,
        vertical_aperture_mm=left.vertical_aperture_mm,
        world_from_camera=matrix,
    )


def triangulate_pixel_pair(
    left_uv: np.ndarray | tuple[float, float],
    right_uv: np.ndarray | tuple[float, float],
    left_camera: CameraModel,
    right_camera: CameraModel,
) -> tuple[np.ndarray, float]:
    left_origin, left_direction = left_camera.pixel_to_world_ray(left_uv)
    right_origin, right_direction = right_camera.pixel_to_world_ray(right_uv)
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
        raise RuntimeError("Triangulated point lies behind a camera.")
    left_point = left_origin + left_distance * left_direction
    right_point = right_origin + right_distance * right_direction
    gap = float(np.linalg.norm(left_point - right_point))
    return (left_point + right_point) / 2.0, gap


def triangulate_detection_centers(
    left_detection: PortDetection,
    right_detection: PortDetection,
    left_camera: CameraModel,
    right_camera: CameraModel,
    virtual_camera: CameraModel,
) -> StereoTriangulation:
    """Triangulate the matched dark-port center and build a diagnostic plane."""
    center_world, ray_gap = triangulate_pixel_pair(
        left_detection.center_uv,
        right_detection.center_uv,
        left_camera,
        right_camera,
    )
    reprojection_errors = np.asarray(
        [
            np.linalg.norm(
                left_camera.project_world(center_world)
                - np.asarray(left_detection.center_uv, dtype=np.float64)
            ),
            np.linalg.norm(
                right_camera.project_world(center_world)
                - np.asarray(right_detection.center_uv, dtype=np.float64)
            ),
        ],
        dtype=np.float64,
    )
    center_virtual = virtual_camera.camera_point_from_world(center_world)
    range_m = -float(center_virtual[2])
    if range_m <= 0.0:
        raise RuntimeError("Triangulated center lies behind the virtual camera.")

    _, _, left_width_px, left_height_px = left_detection.bbox_xywh
    _, _, right_width_px, right_height_px = right_detection.bbox_xywh
    width_m = range_m * 0.5 * (
        float(left_width_px) / left_camera.fx_px
        + float(right_width_px) / right_camera.fx_px
    )
    height_m = range_m * 0.5 * (
        float(left_height_px) / left_camera.fy_px
        + float(right_height_px) / right_camera.fy_px
    )

    half_width = width_m / 2.0
    half_height = height_m / 2.0
    local_corners = np.asarray(
        [
            center_virtual + [-half_width, +half_height, 0.0],
            center_virtual + [+half_width, +half_height, 0.0],
            center_virtual + [+half_width, -half_height, 0.0],
            center_virtual + [-half_width, -half_height, 0.0],
        ],
        dtype=np.float64,
    )
    corners_world = np.vstack(
        [
            transform_point_to_world(point, virtual_camera.world_from_camera)
            for point in local_corners
        ]
    )
    normal_world = (
        np.asarray([0.0, 0.0, 1.0, 0.0])
        @ virtual_camera.world_from_camera
    )[:3]
    normal_world /= np.linalg.norm(normal_world)

    rms = float(np.sqrt(np.mean(reprojection_errors * reprojection_errors)))
    maximum = float(np.max(reprojection_errors))
    return StereoTriangulation(
        corners_world_m=corners_world,
        center_world_m=center_world,
        normal_world=normal_world,
        width_m=float(width_m),
        height_m=float(height_m),
        reprojection_rms_px=rms,
        max_reprojection_px=maximum,
        max_ray_gap_m=float(ray_gap),
        plane_residual_m=0.0,
        opposite_edge_ratio=1.0,
    )


def _edge_ratio(a: float, b: float) -> float:
    minimum = min(a, b)
    if minimum <= 1.0e-12:
        return math.inf
    return max(a, b) / minimum


def triangulate_port_corners(
    left_corners_uv: np.ndarray,
    right_corners_uv: np.ndarray,
    left_camera: CameraModel,
    right_camera: CameraModel,
) -> StereoTriangulation:
    left_uv = np.asarray(left_corners_uv, dtype=np.float64)
    right_uv = np.asarray(right_corners_uv, dtype=np.float64)
    if left_uv.shape != (4, 2) or right_uv.shape != (4, 2):
        raise ValueError("Stereo corner arrays must both have shape (4,2).")
    points: list[np.ndarray] = []
    ray_gaps: list[float] = []
    reprojection_errors: list[float] = []
    for left_pixel, right_pixel in zip(left_uv, right_uv, strict=True):
        point, gap = triangulate_pixel_pair(
            left_pixel,
            right_pixel,
            left_camera,
            right_camera,
        )
        points.append(point)
        ray_gaps.append(gap)
        reprojection_errors.extend(
            [
                float(np.linalg.norm(left_camera.project_world(point) - left_pixel)),
                float(np.linalg.norm(right_camera.project_world(point) - right_pixel)),
            ]
        )
    corners = np.vstack(points)
    center = np.mean(corners, axis=0)
    _, _, vh = np.linalg.svd(corners - center)
    normal = vh[-1]
    normal /= np.linalg.norm(normal)
    camera_midpoint = (
        left_camera.camera_center_world_m
        + right_camera.camera_center_world_m
    ) / 2.0
    if float(np.dot(normal, camera_midpoint - center)) < 0.0:
        normal *= -1.0
    residuals = np.abs((corners - center) @ normal)
    top = float(np.linalg.norm(corners[1] - corners[0]))
    bottom = float(np.linalg.norm(corners[2] - corners[3]))
    left_edge = float(np.linalg.norm(corners[3] - corners[0]))
    right_edge = float(np.linalg.norm(corners[2] - corners[1]))
    width = (top + bottom) / 2.0
    height = (left_edge + right_edge) / 2.0
    opposite_ratio = max(
        _edge_ratio(top, bottom),
        _edge_ratio(left_edge, right_edge),
    )
    reprojection = np.asarray(reprojection_errors, dtype=np.float64)
    rms = float(np.sqrt(np.mean(reprojection * reprojection)))
    maximum = float(np.max(reprojection))
    max_gap = float(np.max(ray_gaps))
    plane_residual = float(np.max(residuals))

    # Broad structural gates belong here so obviously wrong corner ordering
    # never reaches application-specific dimension checks.
    if not np.all(np.isfinite(corners)):
        raise RuntimeError("Triangulation produced non-finite corners.")
    if width <= 0.0 or height <= 0.0:
        raise RuntimeError("Triangulated rectangle has zero-sized edges.")
    if max_gap > 0.010 or maximum > 10.0:
        raise RuntimeError("Corner correspondence has excessive ray/reprojection error.")
    if opposite_ratio > 3.0:
        raise RuntimeError("Corner correspondence does not form a rectangle.")

    return StereoTriangulation(
        corners_world_m=corners,
        center_world_m=center,
        normal_world=normal,
        width_m=width,
        height_m=height,
        reprojection_rms_px=rms,
        max_reprojection_px=maximum,
        max_ray_gap_m=max_gap,
        plane_residual_m=plane_residual,
        opposite_edge_ratio=opposite_ratio,
    )


# ---------------------------------------------------------------------------
# Four-corner extraction and stereo processing
# ---------------------------------------------------------------------------


def refine_port_corners(
    rgb: np.ndarray,
    detection: PortDetection,
    cfg: PerceptionConfig,
) -> PortCorners:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    x, y, width, height = detection.bbox_xywh
    seeds = np.array(
        [
            [x, y],
            [x + width - 1, y],
            [x + width - 1, y + height - 1],
            [x, y + height - 1],
        ],
        dtype=np.float32,
    )
    refined = seeds.reshape(-1, 1, 2).copy()
    window = int(cfg.stereo_corner_refine_window_px)
    cv2.cornerSubPix(
        gray,
        refined,
        (window, window),
        (-1, -1),
        (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            40,
            0.01,
        ),
    )
    corners = refined.reshape(4, 2).astype(np.float64)
    shifts = np.linalg.norm(corners - seeds.astype(np.float64), axis=1)
    if float(np.max(shifts)) > cfg.stereo_max_corner_refine_shift_px:
        raise RuntimeError("corner refinement moved outside the port boundary")
    image_height, image_width = gray.shape
    if (
        np.any(corners[:, 0] < 0.0)
        or np.any(corners[:, 0] >= image_width)
        or np.any(corners[:, 1] < 0.0)
        or np.any(corners[:, 1] >= image_height)
    ):
        raise RuntimeError("refined corner left the image")
    contour = np.round(corners).astype(np.int32).reshape(-1, 1, 2)
    if not cv2.isContourConvex(contour):
        raise RuntimeError("refined port corners are not convex")
    area = abs(float(cv2.contourArea(contour)))
    if area < cfg.min_area_px * 0.5:
        raise RuntimeError("refined port quadrilateral is too small")
    return PortCorners(detection=detection, corners_uv=corners)


def _candidate_continuity_ok(
    candidate: PortDetection,
    previous: PortDetection | None,
    cfg: PerceptionConfig,
) -> bool:
    if previous is None:
        return True
    center_distance = float(
        np.linalg.norm(
            np.asarray(candidate.center_uv) - np.asarray(previous.center_uv)
        )
    )
    scale_ratio = max(
        candidate.scale_px / previous.scale_px,
        previous.scale_px / candidate.scale_px,
    )
    return (
        center_distance <= cfg.tracking_max_center_jump_px
        and scale_ratio <= cfg.tracking_max_scale_ratio
    )


def _validate_stereo_result(
    result: StereoTriangulation,
    virtual_camera: CameraModel,
    cfg: PerceptionConfig,
) -> None:
    center_virtual = virtual_camera.camera_point_from_world(
        result.center_world_m
    )
    range_m = -float(center_virtual[2])
    checks = (
        (
            cfg.min_estimated_range_m <= range_m <= cfg.max_estimated_range_m,
            "stereo range is outside the configured working distance",
        ),
        (
            result.reprojection_rms_px
            <= cfg.stereo_max_reprojection_rms_px,
            "stereo reprojection RMS is too high",
        ),
        (
            result.max_reprojection_px <= cfg.stereo_max_reprojection_px,
            "stereo corner reprojection error is too high",
        ),
        (
            result.max_ray_gap_m <= cfg.stereo_max_ray_gap_m,
            "stereo rays do not intersect closely enough",
        ),
        (
            result.plane_residual_m <= cfg.stereo_max_plane_residual_m,
            "triangulated corners are not coplanar",
        ),
        (
            cfg.stereo_min_width_m
            <= result.width_m
            <= cfg.stereo_max_width_m,
            "triangulated port width is implausible",
        ),
        (
            cfg.stereo_min_height_m
            <= result.height_m
            <= cfg.stereo_max_height_m,
            "triangulated port height is implausible",
        ),
        (
            result.opposite_edge_ratio
            <= cfg.stereo_max_opposite_edge_ratio,
            "triangulated opposite edges disagree",
        ),
    )
    for accepted, reason in checks:
        if not accepted:
            raise RuntimeError(reason)


def process_stereo_port(
    frame: StereoFrame,
    cfg: PerceptionConfig,
    desired_port_virtual_camera_usd: np.ndarray,
    previous_left: PortDetection | None,
    previous_right: PortDetection | None,
) -> StereoPortObservation:
    """Require both eyes, triangulate the matched port center, and compute one correction."""
    left_candidates = detect_port_candidates(frame.left.rgb, cfg)
    if not left_candidates:
        raise RuntimeError("left eye did not detect a valid RGB port")
    right_candidates = detect_port_candidates(frame.right.rgb, cfg)
    if not right_candidates:
        raise RuntimeError("right eye did not detect a valid RGB port")

    left_corners: list[PortCorners] = []
    right_corners: list[PortCorners] = []
    left_corner_errors: list[str] = []
    right_corner_errors: list[str] = []
    for candidate in left_candidates:
        if not _candidate_continuity_ok(candidate, previous_left, cfg):
            continue
        try:
            left_corners.append(refine_port_corners(frame.left.rgb, candidate, cfg))
        except (RuntimeError, cv2.error) as exc:
            left_corner_errors.append(str(exc))
            continue
    for candidate in right_candidates:
        if not _candidate_continuity_ok(candidate, previous_right, cfg):
            continue
        try:
            right_corners.append(refine_port_corners(frame.right.rgb, candidate, cfg))
        except (RuntimeError, cv2.error) as exc:
            right_corner_errors.append(str(exc))
            continue
    if not left_corners:
        detail = left_corner_errors[-1] if left_corner_errors else "no candidate"
        raise RuntimeError(f"left eye corner refinement failed: {detail}")
    if not right_corners:
        detail = right_corner_errors[-1] if right_corner_errors else "no candidate"
        raise RuntimeError(f"right eye corner refinement failed: {detail}")

    pair_results: list[
        tuple[float, PortCorners, PortCorners, StereoTriangulation]
    ] = []
    pair_rejections: list[str] = []
    for left in left_corners:
        for right in right_corners:
            vertical_error = abs(
                left.detection.center_uv[1] - right.detection.center_uv[1]
            )
            if vertical_error > cfg.stereo_max_epipolar_error_px:
                continue
            scale_ratio = max(
                left.detection.scale_px / right.detection.scale_px,
                right.detection.scale_px / left.detection.scale_px,
            )
            if scale_ratio > cfg.stereo_max_scale_ratio:
                continue
            disparity = abs(
                left.detection.center_uv[0] - right.detection.center_uv[0]
            )
            if disparity < cfg.stereo_min_abs_disparity_px:
                continue
            try:
                result = triangulate_detection_centers(
                    left.detection,
                    right.detection,
                    frame.left.camera,
                    frame.right.camera,
                    frame.virtual_camera,
                )
                _validate_stereo_result(result, frame.virtual_camera, cfg)
            except RuntimeError as exc:
                pair_rejections.append(str(exc))
                continue
            dimension_error = (
                abs(result.width_m - cfg.port_width_m) / cfg.port_width_m
                + abs(result.height_m - cfg.port_height_m) / cfg.port_height_m
            )
            score = (
                left.detection.shape_score
                + right.detection.shape_score
                - 0.05 * vertical_error
                - 0.75 * dimension_error
                - result.reprojection_rms_px
            )
            pair_results.append((score, left, right, result))

    if not pair_results:
        detail = pair_rejections[-1] if pair_rejections else (
            "epipolar, scale, or disparity gate rejected every pair"
        )
        raise RuntimeError(
            "no left/right port pair passed stereo correspondence and "
            f"geometry checks: {detail}"
        )
    _, left, right, result = max(pair_results, key=lambda item: item[0])

    virtual_camera = frame.virtual_camera
    center_virtual = virtual_camera.camera_point_from_world(
        result.center_world_m
    )
    desired = np.asarray(
        desired_port_virtual_camera_usd,
        dtype=np.float64,
    ).reshape(3)
    desired_center, desired_size = project_port_feature(
        desired,
        virtual_camera,
        cfg,
    )
    projected_center = virtual_camera.project_world(result.center_world_m)
    center_error = projected_center - np.asarray(desired_center)
    estimated_range_m = -float(center_virtual[2])
    range_error_m = estimated_range_m - (-float(desired[2]))
    correction_world = camera_point_error_to_world(
        center_virtual,
        desired,
        virtual_camera.world_from_camera,
    )
    desired_world = transform_point_to_world(
        desired,
        virtual_camera.world_from_camera,
    )
    desired_left = frame.left.camera.project_world(desired_world)
    desired_right = frame.right.camera.project_world(desired_world)
    center_disparity = abs(
        left.detection.center_uv[0] - right.detection.center_uv[0]
    )

    return StereoPortObservation(
        left=left,
        right=right,
        corners_world_m=result.corners_world_m,
        center_world_xyz_m=result.center_world_m,
        center_virtual_camera_usd_m=center_virtual,
        normal_world=result.normal_world,
        projected_virtual_center_uv=(
            float(projected_center[0]),
            float(projected_center[1]),
        ),
        desired_center_uv=desired_center,
        desired_size_wh_px=desired_size,
        desired_left_center_uv=(float(desired_left[0]), float(desired_left[1])),
        desired_right_center_uv=(float(desired_right[0]), float(desired_right[1])),
        center_error_px=center_error,
        estimated_range_m=estimated_range_m,
        range_error_m=float(range_error_m),
        correction_world_m=correction_world,
        width_m=result.width_m,
        height_m=result.height_m,
        mean_disparity_px=float(center_disparity),
        reprojection_rms_px=result.reprojection_rms_px,
        max_reprojection_px=result.max_reprojection_px,
        max_ray_gap_m=result.max_ray_gap_m,
        plane_residual_m=result.plane_residual_m,
    )
