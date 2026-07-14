#!/usr/bin/env python3
"""Pure NumPy/OpenCV Ethernet-port perception and 3D geometry."""

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
    depth_m: np.ndarray
    camera: CameraModel


@dataclass(frozen=True)
class PortDetection:
    """2D candidate data used by geometry, ranking, and debug output."""

    bbox_xywh: tuple[int, int, int, int]
    center_uv: tuple[int, int]
    shape_score: float
    roi_uv: tuple[int, int, int, int]
    mask: np.ndarray


@dataclass(frozen=True)
class DepthSample:
    median_depth_m: float
    patch_bounds_uv: tuple[int, int, int, int]


@dataclass(frozen=True)
class OpeningPlane:
    depth_m: float
    recess_depth_m: float
    ring_bounds_xyxy: tuple[int, int, int, int]


@dataclass(frozen=True)
class PlaneFit:
    normal_usd_local: np.ndarray
    rms_residual_m: float
    camera_angle_deg: float


@dataclass(frozen=True)
class PortEstimate:
    detection: PortDetection
    cavity: DepthSample
    opening: OpeningPlane
    plane: PlaneFit

    cavity_world_xyz_m: np.ndarray
    opening_world_xyz_m: np.ndarray
    outward_world_normal: np.ndarray
    preinsert_world_xyz_m: np.ndarray


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


def normalize_depth(
    depth: np.ndarray,
    resolution_hw: tuple[int, int],
) -> np.ndarray:
    """Return contiguous HxW float32 metric depth."""
    height, width = resolution_hw
    depth = np.squeeze(np.asarray(depth))

    if depth.ndim == 1:
        if depth.size != height * width:
            raise ValueError(
                f"Cannot reshape flat depth array of size {depth.size}."
            )
        depth = depth.reshape(height, width)

    if depth.shape != (height, width):
        raise ValueError(
            f"Depth shape {depth.shape} does not match {(height, width)}."
        )

    return np.ascontiguousarray(depth.astype(np.float32, copy=False))


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

    weight_sum = (
        cfg.aspect_score_weight
        + cfg.fill_score_weight
    )
    if weight_sum <= 0.0:
        raise ValueError("Shape-score weights must sum to a positive value.")

    return float(
        (
            cfg.aspect_score_weight * aspect_score
            + cfg.fill_score_weight * fill_score
        )
        / weight_sum
    )


def detect_port_candidates(
    rgb: np.ndarray,
    cfg: PerceptionConfig,
) -> list[PortDetection]:
    """Find all RJ45-like dark components across the configured search area."""
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 RGB, got {rgb.shape}.")

    height, width = rgb.shape[:2]

    if cfg.roi_uv is None:
        u0, v0, u1, v1 = 0, 0, width, height
    else:
        u0, v0, u1, v1 = cfg.roi_uv

    if not (0 <= u0 < u1 <= width and 0 <= v0 < v1 <= height):
        raise ValueError(
            f"Search area {(u0, v0, u1, v1)} is outside "
            f"image {width}x{height}."
        )

    roi = rgb[v0:v1, u0:u1]
    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    binary = cv2.inRange(gray, 0, cfg.max_gray)

    count, _, stats, _ = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8,
    )

    full_mask = np.zeros((height, width), dtype=np.uint8)
    full_mask[v0:v1, u0:u1] = binary

    candidates: list[PortDetection] = []

    for index in range(1, count):
        x, y, w, h, area = map(int, stats[index])
        if w <= 0 or h <= 0:
            continue

        gx, gy = u0 + x, v0 + y
        aspect = w / h
        fill = area / (w * h)

        touches_edge = (
            gx < cfg.edge_margin_px
            or gy < cfg.edge_margin_px
            or gx + w > width - cfg.edge_margin_px
            or gy + h > height - cfg.edge_margin_px
        )

        accepted = (
            not touches_edge
            and cfg.min_width_px <= w <= cfg.max_width_px
            and cfg.min_height_px <= h <= cfg.max_height_px
            and cfg.min_aspect_ratio <= aspect <= cfg.max_aspect_ratio
            and cfg.min_area_px <= area <= cfg.max_area_px
            and fill >= cfg.min_fill_ratio
        )
        if not accepted:
            continue

        shape_score = score_port_shape(
            aspect_ratio=aspect,
            fill_ratio=fill,
            cfg=cfg,
        )
        if shape_score < cfg.min_shape_score:
            continue

        candidates.append(
            PortDetection(
                bbox_xywh=(gx, gy, w, h),
                center_uv=(gx + w // 2, gy + h // 2),
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


def sample_center_depth(
    depth_m: np.ndarray,
    center_uv: tuple[int, int],
    patch_size_px: int,
) -> DepthSample:
    """Use the median of an odd square patch around the detected center."""
    if patch_size_px <= 0 or patch_size_px % 2 == 0:
        raise ValueError("Depth patch size must be a positive odd integer.")

    height, width = depth_m.shape
    u, v = center_uv

    if not (0 <= u < width and 0 <= v < height):
        raise ValueError(f"Center {center_uv} is outside depth image.")

    half = patch_size_px // 2
    u0, u1 = max(0, u - half), min(width, u + half + 1)
    v0, v1 = max(0, v - half), min(height, v + half + 1)

    patch = depth_m[v0:v1, u0:u1]
    valid = patch[np.isfinite(patch) & (patch > 0.0)]

    if valid.size == 0:
        raise RuntimeError("Port-center depth patch contains no valid values.")

    return DepthSample(
        median_depth_m=float(np.median(valid)),
        patch_bounds_uv=(u0, v0, u1, v1),
    )


def _ring_samples(
    depth_m: np.ndarray,
    bbox_xywh: tuple[int, int, int, int],
    ring_width_px: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int, int, int]]:
    """Return U, V, depth samples from a rectangular ring around a box."""
    x, y, w, h = bbox_xywh
    image_h, image_w = depth_m.shape

    x0, y0 = max(0, x - ring_width_px), max(0, y - ring_width_px)
    x1 = min(image_w, x + w + ring_width_px)
    y1 = min(image_h, y + h + ring_width_px)

    patch = depth_m[y0:y1, x0:x1]
    ring = np.ones(patch.shape, dtype=bool)

    inner_x0, inner_y0 = x - x0, y - y0
    ring[
        inner_y0:inner_y0 + h,
        inner_x0:inner_x0 + w,
    ] = False

    local_v, local_u = np.indices(patch.shape)
    valid = ring & np.isfinite(patch) & (patch > 0.0)

    return (
        (local_u + x0)[valid].astype(np.float64),
        (local_v + y0)[valid].astype(np.float64),
        patch[valid].astype(np.float64),
        (x0, y0, x1, y1),
    )


def estimate_opening_plane(
    depth_m: np.ndarray,
    detection: PortDetection,
    cavity_depth_m: float,
    cfg: PerceptionConfig,
) -> OpeningPlane:
    """Estimate the front opening depth from the ring around the dark cavity."""
    _, _, values, bounds = _ring_samples(
        depth_m,
        detection.bbox_xywh,
        cfg.opening_ring_width_px,
    )

    if values.size < cfg.min_valid_ring_pixels:
        raise RuntimeError(
            f"Only {values.size} valid ring pixels; "
            f"need {cfg.min_valid_ring_pixels}."
        )

    depth = float(np.median(values))
    recess = float(cavity_depth_m - depth)

    if not cfg.min_recess_depth_m <= recess <= cfg.max_recess_depth_m:
        raise RuntimeError(
            "Implausible port recess: "
            f"opening={depth:.6f} m, cavity={cavity_depth_m:.6f} m, "
            f"recess={recess:.6f} m."
        )

    return OpeningPlane(
        depth_m=depth,
        recess_depth_m=recess,
        ring_bounds_xyxy=bounds,
    )


def deproject_pixel(
    center_uv: tuple[int, int],
    depth_m: float,
    camera: CameraModel,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the point in OpenCV and USD camera-local coordinates."""
    u, v = map(float, center_uv)

    x = (u - camera.cx_px) * depth_m / camera.fx_px
    y = (v - camera.cy_px) * depth_m / camera.fy_px

    point_cv = np.array([x, y, depth_m], dtype=np.float64)
    point_usd = np.array([x, -y, -depth_m], dtype=np.float64)
    return point_cv, point_usd


def transform_point_to_world(
    point_usd_local: np.ndarray,
    world_from_camera: np.ndarray,
) -> np.ndarray:
    """Apply USD/Gf row-vector matrix convention to a local point."""
    matrix = np.asarray(world_from_camera, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"Expected a 4x4 camera matrix, got {matrix.shape}.")

    homogeneous = np.append(np.asarray(point_usd_local, dtype=np.float64), 1.0)
    world = homogeneous @ matrix

    if abs(world[3]) > 1.0e-12:
        world = world / world[3]

    return world[:3]


def transform_direction_to_world(
    direction_usd_local: np.ndarray,
    world_from_camera: np.ndarray,
) -> np.ndarray:
    """Rotate a camera-local direction into world coordinates."""
    matrix = np.asarray(world_from_camera, dtype=np.float64)
    homogeneous = np.append(
        np.asarray(direction_usd_local, dtype=np.float64),
        0.0,
    )
    world = (homogeneous @ matrix)[:3]
    norm = np.linalg.norm(world)

    if norm <= 1.0e-12:
        raise RuntimeError("Camera direction transformed to zero length.")

    return world / norm


def fit_opening_normal(
    depth_m: np.ndarray,
    detection: PortDetection,
    camera: CameraModel,
    cfg: PerceptionConfig,
) -> PlaneFit:
    """Robustly fit the front-face ring and orient its normal toward camera."""
    u, v, z, _ = _ring_samples(
        depth_m,
        detection.bbox_xywh,
        cfg.opening_ring_width_px,
    )

    if z.size < cfg.plane_min_inlier_points:
        raise RuntimeError("Too few valid ring samples for a plane fit.")

    median = float(np.median(z))
    mad = float(np.median(np.abs(z - median)))
    tolerance = max(
        cfg.plane_min_depth_tolerance_m,
        cfg.plane_mad_scale * 1.4826 * mad,
    )

    keep = np.abs(z - median) <= tolerance
    u, v, z = u[keep], v[keep], z[keep]

    if z.size < cfg.plane_min_inlier_points:
        raise RuntimeError("Too few robust depth inliers for a plane fit.")

    x = (u - camera.cx_px) * z / camera.fx_px
    y = (v - camera.cy_px) * z / camera.fy_px
    points = np.column_stack((x, y, z))

    centroid = np.mean(points, axis=0)
    centered = points - centroid

    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal_cv = vh[-1].astype(np.float64)
    normal_cv /= np.linalg.norm(normal_cv)

    # Outward/toward-camera is -Z in OpenCV coordinates.
    if normal_cv[2] > 0.0:
        normal_cv *= -1.0

    residuals = centered @ normal_cv
    rms = float(np.sqrt(np.mean(residuals**2)))

    if rms > cfg.plane_max_rms_residual_m:
        raise RuntimeError(
            f"Plane RMS {rms * 1000.0:.3f} mm exceeds limit."
        )

    alignment = float(np.clip(np.dot(normal_cv, [0.0, 0.0, -1.0]), -1.0, 1.0))
    angle = math.degrees(math.acos(alignment))

    if angle > cfg.plane_max_camera_angle_deg:
        raise RuntimeError(
            f"Plane normal angle {angle:.2f}° exceeds limit."
        )

    normal_usd = np.array(
        [normal_cv[0], -normal_cv[1], -normal_cv[2]],
        dtype=np.float64,
    )

    return PlaneFit(
        normal_usd_local=normal_usd,
        rms_residual_m=rms,
        camera_angle_deg=angle,
    )


def _estimate_candidate_geometry(
    frame: CameraFrame,
    detection: PortDetection,
    cfg: PerceptionConfig,
) -> PortEstimate:
    """Build one complete 3D estimate or raise when geometry is implausible."""
    cavity = sample_center_depth(
        frame.depth_m,
        detection.center_uv,
        cfg.depth_patch_size_px,
    )

    opening = estimate_opening_plane(
        frame.depth_m,
        detection,
        cavity.median_depth_m,
        cfg,
    )

    plane = fit_opening_normal(
        frame.depth_m,
        detection,
        frame.camera,
        cfg,
    )

    _, cavity_usd = deproject_pixel(
        detection.center_uv,
        cavity.median_depth_m,
        frame.camera,
    )
    _, opening_usd = deproject_pixel(
        detection.center_uv,
        opening.depth_m,
        frame.camera,
    )

    cavity_world = transform_point_to_world(
        cavity_usd,
        frame.camera.world_from_camera,
    )
    opening_world = transform_point_to_world(
        opening_usd,
        frame.camera.world_from_camera,
    )
    outward_world = transform_direction_to_world(
        plane.normal_usd_local,
        frame.camera.world_from_camera,
    )

    preinsert = (
        opening_world
        + outward_world * cfg.preinsert_standoff_m
    )

    return PortEstimate(
        detection=detection,
        cavity=cavity,
        opening=opening,
        plane=plane,
        cavity_world_xyz_m=cavity_world,
        opening_world_xyz_m=opening_world,
        outward_world_normal=outward_world,
        preinsert_world_xyz_m=preinsert,
    )


def process_port(
    frame: CameraFrame,
    cfg: PerceptionConfig,
) -> PortEstimate:
    """
    Scan the full frame, validate every candidate in 3D, then choose shape.

    Image position is intentionally not part of the ranking. A candidate must
    first pass cavity-recess and front-plane checks. Among those physically
    plausible candidates, the highest RJ45 shape score wins.
    """
    candidates = detect_port_candidates(
        frame.rgb,
        cfg,
    )

    if not candidates:
        raise RuntimeError(
            "No port candidate passed the full-screen shape filters."
        )

    valid: list[PortEstimate] = []
    rejection_reasons: list[str] = []

    for candidate in candidates:
        try:
            valid.append(
                _estimate_candidate_geometry(
                    frame=frame,
                    detection=candidate,
                    cfg=cfg,
                )
            )
        except (RuntimeError, ValueError) as exc:
            rejection_reasons.append(
                f"{candidate.center_uv}: {exc}"
            )

    if not valid:
        details = "; ".join(rejection_reasons[:4])
        raise RuntimeError(
            f"Found {len(candidates)} full-screen shape candidate(s), "
            "but none passed depth/geometry validation. "
            f"{details}"
        )

    return max(
        valid,
        key=lambda estimate: estimate.detection.shape_score,
    )

