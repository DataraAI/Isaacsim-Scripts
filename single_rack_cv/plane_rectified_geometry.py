#!/usr/bin/env python3
"""Camera-derived metric plane frame and RGB rectification."""

from __future__ import annotations

import math
import cv2
import numpy as np

from plane_rectified_types import (
    DEFAULT_RECTIFIED_RESOLUTION_M,
    PlaneFrame,
    RectifiedEye,
    _unit,
)


def _camera_axes_world(camera) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(camera.world_from_camera, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError("Camera transform must be 4x4.")
    image_right = _unit(matrix[0, :3], "camera image-right axis")
    image_down = _unit(-matrix[1, :3], "camera image-down axis")
    return image_right, image_down


def build_plane_frame(
    left_camera,
    right_camera,
    plane_origin_world_m: np.ndarray,
    plane_normal_world: np.ndarray,
) -> PlaneFrame:
    """Build image-aligned metric axes on a camera-derived physical plane."""

    supplied_normal = _unit(plane_normal_world, "measured plane normal")
    left_right, left_down = _camera_axes_world(left_camera)
    right_right, right_down = _camera_axes_world(right_camera)
    average_right = _unit(left_right + right_right, "average image-right axis")
    average_down = _unit(left_down + right_down, "average image-down axis")

    axis_u = average_right - supplied_normal * float(average_right @ supplied_normal)
    axis_u = _unit(axis_u, "projected image-right axis")
    axis_v = average_down - supplied_normal * float(average_down @ supplied_normal)
    axis_v = axis_v - axis_u * float(axis_v @ axis_u)
    axis_v = _unit(axis_v, "projected image-down axis")
    normal = _unit(np.cross(axis_u, axis_v), "right-handed plane normal")
    if abs(float(normal @ supplied_normal)) < 0.95:
        raise RuntimeError("Camera-derived plane axes disagree with the measured plane.")

    return PlaneFrame(
        origin_world_m=np.asarray(plane_origin_world_m, dtype=np.float64),
        axis_u_world=axis_u,
        axis_v_world=axis_v,
        normal_world=normal,
    )


def _intersect_pixel_with_frame(camera, pixel_uv, frame: PlaneFrame) -> np.ndarray:
    origin, direction = camera.pixel_to_world_ray(pixel_uv)
    origin = np.asarray(origin, dtype=np.float64).reshape(3)
    direction = _unit(direction, "camera ray")
    denominator = float(direction @ frame.normal_world)
    if abs(denominator) <= 1.0e-10:
        raise RuntimeError("Camera ray is parallel to the measured front plane.")
    distance = float(
        (frame.origin_world_m - origin) @ frame.normal_world / denominator
    )
    if distance <= 0.0:
        raise RuntimeError("Measured front plane lies behind a camera.")
    return origin + distance * direction


def _largest_contour(mask: np.ndarray) -> np.ndarray:
    binary = np.where(np.asarray(mask) > 0, 255, 0).astype(np.uint8)
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if not contours:
        raise RuntimeError("Aperture mask contains no external contour.")
    contour = max(contours, key=cv2.contourArea)
    if float(cv2.contourArea(contour)) < 8.0:
        raise RuntimeError("Aperture contour is too small.")
    return contour.reshape(-1, 2)


def project_mask_bounds_to_plane(
    mask,
    camera,
    frame: PlaneFrame,
) -> tuple[np.ndarray, np.ndarray]:
    contour = _largest_contour(mask)
    world = np.asarray(
        [_intersect_pixel_with_frame(camera, pixel, frame) for pixel in contour],
        dtype=np.float64,
    )
    metric = frame.world_to_metric(world)
    return np.min(metric, axis=0), np.max(metric, axis=0)


def _camera_projection_map(
    camera,
    world_points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(camera.world_from_camera, dtype=np.float64)
    camera_from_world = np.linalg.inv(matrix)
    homogeneous = np.concatenate(
        (world_points, np.ones((*world_points.shape[:2], 1), dtype=np.float64)),
        axis=2,
    )
    local = homogeneous @ camera_from_world
    local = local[..., :3] / local[..., 3, None]
    range_m = -local[..., 2]
    map_u = camera.cx_px + camera.fx_px * local[..., 0] / range_m
    map_v = camera.cy_px + camera.fy_px * (-local[..., 1]) / range_m
    return map_u, map_v


def rectify_eye_to_plane(
    rgb: np.ndarray,
    mask: np.ndarray,
    camera,
    plane_frame: PlaneFrame,
    bounds_uv_m: tuple[np.ndarray, np.ndarray],
    *,
    resolution_m: float = DEFAULT_RECTIFIED_RESOLUTION_M,
) -> RectifiedEye:
    minimum = np.asarray(bounds_uv_m[0], dtype=np.float64).reshape(2)
    maximum = np.asarray(bounds_uv_m[1], dtype=np.float64).reshape(2)
    resolution = float(resolution_m)
    if np.any(maximum <= minimum):
        raise ValueError("Rectified bounds must have positive extent.")
    width = int(math.ceil((maximum[0] - minimum[0]) / resolution)) + 1
    height = int(math.ceil((maximum[1] - minimum[1]) / resolution)) + 1
    if width < 16 or height < 16 or width * height > 4_000_000:
        raise RuntimeError(f"Rectified patch dimensions are unsafe: {width}x{height}.")

    u_values = minimum[0] + np.arange(width, dtype=np.float64) * resolution
    v_values = minimum[1] + np.arange(height, dtype=np.float64) * resolution
    grid_u, grid_v = np.meshgrid(u_values, v_values)
    metric = np.stack((grid_u, grid_v), axis=2)
    world = plane_frame.metric_to_world(metric)
    map_u, map_v = _camera_projection_map(camera, world)
    visible = (
        np.isfinite(map_u)
        & np.isfinite(map_v)
        & (map_u >= 0.0)
        & (map_v >= 0.0)
        & (map_u < float(camera.image_width_px - 1))
        & (map_v < float(camera.image_height_px - 1))
    )
    rectified_rgb = cv2.remap(
        np.asarray(rgb, dtype=np.uint8),
        map_u.astype(np.float32),
        map_v.astype(np.float32),
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    rectified_mask = cv2.remap(
        np.where(np.asarray(mask) > 0, 255, 0).astype(np.uint8),
        map_u.astype(np.float32),
        map_v.astype(np.float32),
        cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
    ) > 0
    return RectifiedEye(
        rgb=rectified_rgb,
        mask=rectified_mask,
        visible=visible,
        map_u_px=map_u,
        map_v_px=map_v,
        minimum_uv_m=minimum,
        resolution_m=resolution,
        plane_frame=plane_frame,
        camera=camera,
    )
