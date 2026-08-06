#!/usr/bin/env python3
"""Front-rim-plane stereo center for the physical RJ45 mouth."""

from __future__ import annotations

import math

import numpy as np

from stereo_center import (
    MAX_RAY_GAP_M,
    MAX_REPROJECTION_PX,
    StereoApertureCenter,
    _front_rim_lines,
)
from stereo_geometry import triangulate_pixel_pair, unit_vector

_MAX_CORNER_RAY_GAP_M = 0.0020
_MAX_PLANE_RESIDUAL_M = 0.0010


def _intersect_side_and_horizontal(
    side: tuple[float, float],
    horizontal: tuple[float, float],
) -> np.ndarray:
    """Intersect x=a*y+b with y=c*x+d."""
    a, b = map(float, side)
    c, d = map(float, horizontal)
    denominator = 1.0 - a * c
    if abs(denominator) <= 1.0e-9:
        raise RuntimeError("Front-rim boundary lines are parallel.")
    x = (a * d + b) / denominator
    y = c * x + d
    point = np.array([x, y], dtype=np.float64)
    if not np.all(np.isfinite(point)):
        raise RuntimeError("Front-rim corner is not finite.")
    return point


def front_rim_corners_pixel(rgb, mask, camera) -> np.ndarray:
    """Return TL, TR, BR, BL corners of the RGB front mouth."""
    left, right, top, bottom = _front_rim_lines(rgb, mask, camera)
    corners = np.vstack(
        (
            _intersect_side_and_horizontal(left, top),
            _intersect_side_and_horizontal(right, top),
            _intersect_side_and_horizontal(right, bottom),
            _intersect_side_and_horizontal(left, bottom),
        )
    )
    if np.any(corners[:, 0] < 0.0) or np.any(
        corners[:, 0] >= float(camera.image_width_px)
    ):
        raise RuntimeError("RGB front-rim corner is outside the image width.")
    if np.any(corners[:, 1] < 0.0) or np.any(
        corners[:, 1] >= float(camera.image_height_px)
    ):
        raise RuntimeError("RGB front-rim corner is outside the image height.")
    return corners


def _line_intersection(p0, p1, q0, q1) -> np.ndarray:
    p0 = np.asarray(p0, dtype=np.float64).reshape(2)
    p1 = np.asarray(p1, dtype=np.float64).reshape(2)
    q0 = np.asarray(q0, dtype=np.float64).reshape(2)
    q1 = np.asarray(q1, dtype=np.float64).reshape(2)
    matrix = np.column_stack((p1 - p0, -(q1 - q0)))
    values, _, rank, _ = np.linalg.lstsq(matrix, q0 - p0, rcond=None)
    if rank < 2:
        raise RuntimeError("Front-rim diagonals are parallel.")
    center = p0 + float(values[0]) * (p1 - p0)
    if not np.all(np.isfinite(center)):
        raise RuntimeError("Projective front-rim center is not finite.")
    return center


def _fit_plane(points_world_m: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    points = np.asarray(points_world_m, dtype=np.float64)
    if points.shape != (4, 3) or not np.all(np.isfinite(points)):
        raise ValueError("Four finite 3D front-rim corners are required.")
    origin = np.mean(points, axis=0)
    _, _, vh = np.linalg.svd(points - origin, full_matrices=False)
    normal = unit_vector(vh[-1], "front-rim plane normal")
    residuals = np.abs((points - origin) @ normal)
    residual = float(np.max(residuals))
    if residual > _MAX_PLANE_RESIDUAL_M:
        raise RuntimeError(
            "Triangulated RGB front-rim corners are not planar: "
            f"residual={residual * 1000.0:.3f} mm; "
            f"limit={_MAX_PLANE_RESIDUAL_M * 1000.0:.3f} mm."
        )
    return origin, normal, residual


def _ray_plane_intersection(camera, pixel_uv, plane_origin, plane_normal):
    origin, direction = camera.pixel_to_world_ray(pixel_uv)
    origin = np.asarray(origin, dtype=np.float64).reshape(3)
    direction = unit_vector(direction, "front-rim center ray")
    denominator = float(np.dot(direction, plane_normal))
    if abs(denominator) <= 1.0e-9:
        raise RuntimeError("Front-rim center ray is parallel to the rim plane.")
    distance = float(np.dot(plane_origin - origin, plane_normal) / denominator)
    if distance <= 0.0:
        raise RuntimeError("Front-rim plane lies behind a camera.")
    return origin + distance * direction


def estimate_stereo_aperture_center(
    *,
    left_rgb: np.ndarray,
    right_rgb: np.ndarray,
    left_mask: np.ndarray,
    right_mask: np.ndarray,
    left_camera,
    right_camera,
    max_ray_gap_m: float = MAX_RAY_GAP_M,
    max_reprojection_px: float = MAX_REPROJECTION_PX,
) -> StereoApertureCenter:
    """Estimate the centered point on the physical front-rim plane."""
    maximum_gap = float(max_ray_gap_m)
    maximum_reprojection = float(max_reprojection_px)
    if not math.isfinite(maximum_gap) or not 0.0 < maximum_gap <= MAX_RAY_GAP_M:
        raise ValueError("Stereo center ray-gap gate must be in (0, 0.5 mm].")
    if not math.isfinite(maximum_reprojection) or maximum_reprojection <= 0.0:
        raise ValueError("Stereo center reprojection gate must be positive.")

    left_corners = front_rim_corners_pixel(left_rgb, left_mask, left_camera)
    right_corners = front_rim_corners_pixel(right_rgb, right_mask, right_camera)
    left_uv = _line_intersection(
        left_corners[0], left_corners[2], left_corners[1], left_corners[3]
    )
    right_uv = _line_intersection(
        right_corners[0], right_corners[2], right_corners[1], right_corners[3]
    )

    corners_world = []
    corner_gaps = []
    for left_corner, right_corner in zip(left_corners, right_corners):
        corner_world, corner_gap = triangulate_pixel_pair(
            left_corner,
            right_corner,
            left_camera,
            right_camera,
        )
        corners_world.append(corner_world)
        corner_gaps.append(float(corner_gap))
    maximum_corner_gap = max(corner_gaps)
    if maximum_corner_gap > _MAX_CORNER_RAY_GAP_M:
        raise RuntimeError(
            "Stereo RGB front-rim corner ray gap is "
            f"{maximum_corner_gap * 1000.0:.3f} mm; "
            f"limit is {_MAX_CORNER_RAY_GAP_M * 1000.0:.3f} mm."
        )

    plane_origin, plane_normal, _ = _fit_plane(np.asarray(corners_world))
    left_point = _ray_plane_intersection(
        left_camera, left_uv, plane_origin, plane_normal
    )
    right_point = _ray_plane_intersection(
        right_camera, right_uv, plane_origin, plane_normal
    )
    ray_gap = float(np.linalg.norm(left_point - right_point))
    if ray_gap > maximum_gap:
        raise RuntimeError(
            "Stereo RGB front-rim plane-center disagreement is "
            f"{ray_gap * 1000.0:.3f} mm; "
            f"limit is {maximum_gap * 1000.0:.3f} mm."
        )
    center_world = 0.5 * (left_point + right_point)

    reprojection_errors = np.asarray(
        (
            np.linalg.norm(left_camera.project_world(center_world) - left_uv),
            np.linalg.norm(right_camera.project_world(center_world) - right_uv),
        ),
        dtype=np.float64,
    )
    maximum_error = float(np.max(reprojection_errors))
    rms_error = float(np.sqrt(np.mean(reprojection_errors**2)))
    if maximum_error > maximum_reprojection:
        raise RuntimeError(
            "Stereo RGB front-rim plane-center reprojection error is "
            f"{maximum_error:.3f} px; limit is {maximum_reprojection:.3f} px."
        )

    return StereoApertureCenter(
        center_world_m=center_world,
        left_center_uv=left_uv,
        right_center_uv=right_uv,
        ray_gap_m=ray_gap,
        reprojection_rms_px=rms_error,
        max_reprojection_px=maximum_error,
    )
