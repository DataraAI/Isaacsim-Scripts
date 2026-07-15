#!/usr/bin/env python3
"""Pure NumPy/OpenCV RGB port detection and visual-servo geometry."""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from config import PerceptionConfig


@dataclass(frozen=True)
class CameraModel:
    """Camera calibration plus the current USD camera-to-world matrix."""

    image_height_px: int
    image_width_px: int
    focal_length_mm: float
    horizontal_aperture_mm: float
    vertical_aperture_mm: float
    world_from_camera: np.ndarray

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


@dataclass(frozen=True)
class CameraFrame:
    rgb: np.ndarray
    camera: CameraModel


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
class PortObservation:
    detection: PortDetection
    estimated_range_m: float
    port_camera_usd_m: np.ndarray
    port_world_xyz_m: np.ndarray
    desired_port_camera_usd_m: np.ndarray
    desired_center_uv: tuple[float, float]
    desired_size_wh_px: tuple[float, float]
    center_error_px: np.ndarray
    range_error_m: float
    correction_world_m: np.ndarray


# ---------------------------------------------------------------------------
# Image normalization and candidate detection
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
    """Score RJ45 silhouette quality without using image position."""
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
    """Reject dark grille holes that lack the RJ45 port's bright bezel."""
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
    """Find all dark RJ45-like components across the configured image area."""
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
            and _has_bright_surround(
                gray,
                (x, y, width, height),
                cfg,
            )
        )
        if not accepted:
            continue

        shape_score = score_port_shape(
            aspect_ratio=aspect_ratio,
            fill_ratio=fill_ratio,
            cfg=cfg,
        )
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

    candidates.sort(
        key=lambda candidate: candidate.shape_score,
        reverse=True,
    )
    return candidates


def select_port_candidate(
    candidates: list[PortDetection],
    previous: PortDetection | None,
    cfg: PerceptionConfig,
) -> PortDetection:
    """Select by shape initially, then keep the same image-space track."""
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
            (
                candidate.shape_score
                - center_penalty
                - scale_penalty,
                candidate,
            )
        )

    if not tracked:
        raise RuntimeError(
            "RGB port track was lost: no candidate passed the "
            "center/scale continuity gates."
        )

    return max(tracked, key=lambda item: item[0])[1]


# ---------------------------------------------------------------------------
# Known-size monocular geometry
# ---------------------------------------------------------------------------


def compute_desired_port_camera_usd(
    camera_position_hand_m: np.ndarray,
    hand_from_camera: np.ndarray,
    tool_center_position_hand_m: np.ndarray,
    hand_from_tool: np.ndarray,
    preinsert_standoff_m: float,
) -> np.ndarray:
    """Compute where the port should appear in the camera at pre-insert."""
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

    port_in_tool = np.array(
        [0.0, 0.0, preinsert_standoff_m],
        dtype=np.float64,
    )
    port_in_hand = tool_position + hand_from_tool @ port_in_tool
    port_in_camera = (
        hand_from_camera.T @ (port_in_hand - camera_position)
    )

    if port_in_camera[2] >= 0.0:
        raise RuntimeError(
            "Configured pre-insert point is not in front of the camera: "
            f"{np.round(port_in_camera, 6).tolist()}"
        )

    return port_in_camera


def estimate_port_point_camera_usd(
    detection: PortDetection,
    camera: CameraModel,
    cfg: PerceptionConfig,
) -> tuple[np.ndarray, float]:
    """Estimate the port point from its known size and observed pixel box."""
    _, _, width_px, height_px = detection.bbox_xywh

    if width_px <= 0 or height_px <= 0:
        raise ValueError("Detected port size must be positive.")
    if cfg.port_width_m <= 0.0 or cfg.port_height_m <= 0.0:
        raise ValueError("Configured port dimensions must be positive.")

    range_from_width = camera.fx_px * cfg.port_width_m / width_px
    range_from_height = camera.fy_px * cfg.port_height_m / height_px
    range_m = float(np.median([range_from_width, range_from_height]))

    if not (
        cfg.min_estimated_range_m
        <= range_m
        <= cfg.max_estimated_range_m
    ):
        raise RuntimeError(
            f"Implausible RGB range estimate {range_m:.4f} m; expected "
            f"[{cfg.min_estimated_range_m:.3f}, "
            f"{cfg.max_estimated_range_m:.3f}] m."
        )

    u, v = detection.center_uv
    x_cv = (u - camera.cx_px) * range_m / camera.fx_px
    y_cv = (v - camera.cy_px) * range_m / camera.fy_px

    point_usd = np.array(
        [x_cv, -y_cv, -range_m],
        dtype=np.float64,
    )
    return point_usd, range_m


def project_port_feature(
    point_camera_usd_m: np.ndarray,
    camera: CameraModel,
    cfg: PerceptionConfig,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Project a camera-local USD point and known port size into the image."""
    point = np.asarray(point_camera_usd_m, dtype=np.float64).reshape(3)
    range_m = -float(point[2])

    if range_m <= 0.0:
        raise ValueError("Desired port point must be in front of the camera.")

    x_cv = float(point[0])
    y_cv = -float(point[1])

    u = camera.cx_px + camera.fx_px * x_cv / range_m
    v = camera.cy_px + camera.fy_px * y_cv / range_m
    width_px = camera.fx_px * cfg.port_width_m / range_m
    height_px = camera.fy_px * cfg.port_height_m / range_m

    return (float(u), float(v)), (float(width_px), float(height_px))


def transform_point_to_world(
    point_usd_local: np.ndarray,
    world_from_camera: np.ndarray,
) -> np.ndarray:
    """Apply USD/Gf row-vector matrix convention to a local point."""
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
    """Convert current-minus-desired camera point error into camera motion."""
    current = np.asarray(current_point_usd, dtype=np.float64).reshape(3)
    desired = np.asarray(desired_point_usd, dtype=np.float64).reshape(3)
    matrix = np.asarray(world_from_camera, dtype=np.float64)

    if matrix.shape != (4, 4):
        raise ValueError(f"Expected a 4x4 camera matrix, got {matrix.shape}.")

    # Moving the eye-in-hand camera by this vector makes the observed point
    # move toward the desired camera-frame coordinates.
    local_motion = current - desired
    world_motion = (np.append(local_motion, 0.0) @ matrix)[:3]
    return np.asarray(world_motion, dtype=np.float64)


def compute_bounded_step(
    correction_world_m: np.ndarray,
    gain: float,
    max_step_m: float,
) -> np.ndarray:
    """Scale one visual correction and cap its Euclidean step length."""
    correction = np.asarray(
        correction_world_m,
        dtype=np.float64,
    ).reshape(3)

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


def process_port(
    frame: CameraFrame,
    cfg: PerceptionConfig,
    desired_port_camera_usd: np.ndarray,
    previous_detection: PortDetection | None,
) -> PortObservation:
    """Detect, track, range, and convert one RGB observation to a correction."""
    candidates = detect_port_candidates(frame.rgb, cfg)

    plausible: list[PortDetection] = []
    for candidate in candidates:
        try:
            estimate_port_point_camera_usd(candidate, frame.camera, cfg)
        except (RuntimeError, ValueError):
            continue
        plausible.append(candidate)

    if not plausible:
        raise RuntimeError(
            "No RGB port candidate passed the shape and known-size range "
            "checks."
        )

    detection = select_port_candidate(
        plausible,
        previous_detection,
        cfg,
    )
    port_camera_usd, range_m = estimate_port_point_camera_usd(
        detection,
        frame.camera,
        cfg,
    )

    desired = np.asarray(
        desired_port_camera_usd,
        dtype=np.float64,
    ).reshape(3)
    desired_center, desired_size = project_port_feature(
        desired,
        frame.camera,
        cfg,
    )

    center_error = (
        np.asarray(detection.center_uv, dtype=np.float64)
        - np.asarray(desired_center, dtype=np.float64)
    )
    desired_range_m = -float(desired[2])
    range_error_m = range_m - desired_range_m

    correction_world = camera_point_error_to_world(
        current_point_usd=port_camera_usd,
        desired_point_usd=desired,
        world_from_camera=frame.camera.world_from_camera,
    )

    return PortObservation(
        detection=detection,
        estimated_range_m=range_m,
        port_camera_usd_m=port_camera_usd,
        port_world_xyz_m=transform_point_to_world(
            port_camera_usd,
            frame.camera.world_from_camera,
        ),
        desired_port_camera_usd_m=desired,
        desired_center_uv=desired_center,
        desired_size_wh_px=desired_size,
        center_error_px=center_error,
        range_error_m=float(range_error_m),
        correction_world_m=correction_world,
    )
