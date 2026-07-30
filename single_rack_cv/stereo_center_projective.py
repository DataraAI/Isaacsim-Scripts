#!/usr/bin/env python3
"""Perspective-correct stereo reconstruction of the RJ45 front-rim center."""

from __future__ import annotations

import math

import numpy as np

from stereo_center import (
    MAX_RAY_GAP_M,
    MAX_REPROJECTION_PX,
    StereoApertureCenter,
    _front_rim_lines,
)
from stereo_geometry import triangulate_pixel_pair


def _intersect_side_and_horizontal(
    side: tuple[float, float],
    horizontal: tuple[float, float],
) -> np.ndarray:
    """Intersect x = a*y+b with y = c*x+d."""

    side_slope, side_intercept = map(float, side)
    horizontal_slope, horizontal_intercept = map(float, horizontal)
    denominator = 1.0 - side_slope * horizontal_slope
    if abs(denominator) <= 1.0e-9:
        raise RuntimeError("Front-rim boundary lines are parallel.")

    u = (
        side_slope * horizontal_intercept + side_intercept
    ) / denominator
    v = horizontal_slope * u + horizontal_intercept
    point = np.array([u, v], dtype=np.float64)
    if not np.all(np.isfinite(point)):
        raise RuntimeError("Front-rim corner is not finite.")
    return point


def _line_through(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    first_h = np.array([float(first[0]), float(first[1]), 1.0])
    second_h = np.array([float(second[0]), float(second[1]), 1.0])
    line = np.cross(first_h, second_h)
    if float(np.linalg.norm(line[:2])) <= 1.0e-12:
        raise RuntimeError("Front-rim diagonal is degenerate.")
    return line


def _projective_center_from_lines(
    left: tuple[float, float],
    right: tuple[float, float],
    top: tuple[float, float],
    bottom: tuple[float, float],
) -> np.ndarray:
    """Return the projective center of the fitted front-rim quadrilateral."""

    top_left = _intersect_side_and_horizontal(left, top)
    top_right = _intersect_side_and_horizontal(right, top)
    bottom_right = _intersect_side_and_horizontal(right, bottom)
    bottom_left = _intersect_side_and_horizontal(left, bottom)

    first_diagonal = _line_through(top_left, bottom_right)
    second_diagonal = _line_through(top_right, bottom_left)
    center_h = np.cross(first_diagonal, second_diagonal)
    if abs(float(center_h[2])) <= 1.0e-12:
        raise RuntimeError("Front-rim diagonals do not meet at a finite point.")

    center = center_h[:2] / center_h[2]
    if not np.all(np.isfinite(center)):
        raise RuntimeError("Projective front-rim center is not finite.")
    return center.astype(np.float64)


def aperture_center_pixel(
    rgb: np.ndarray,
    mask: np.ndarray,
    camera,
) -> np.ndarray:
    """Return the perspective-correct image center of the physical front rim."""

    left, right, top, bottom = _front_rim_lines(rgb, mask, camera)
    center = _projective_center_from_lines(left, right, top, bottom)
    if not (
        0.0 <= center[0] < float(camera.image_width_px)
        and 0.0 <= center[1] < float(camera.image_height_px)
    ):
        raise RuntimeError("Projective RGB front-rim center is outside the image.")
    return center


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
    """Triangulate corresponding perspective-correct RGB front-rim centers."""

    maximum_gap = float(max_ray_gap_m)
    maximum_reprojection = float(max_reprojection_px)
    if (
        not math.isfinite(maximum_gap)
        or not 0.0 < maximum_gap <= MAX_RAY_GAP_M
    ):
        raise ValueError("Stereo center ray-gap gate must be in (0, 0.5 mm].")
    if not math.isfinite(maximum_reprojection) or maximum_reprojection <= 0.0:
        raise ValueError("Stereo center reprojection gate must be positive.")

    left_uv = aperture_center_pixel(left_rgb, left_mask, left_camera)
    right_uv = aperture_center_pixel(right_rgb, right_mask, right_camera)
    center_world, ray_gap = triangulate_pixel_pair(
        left_uv,
        right_uv,
        left_camera,
        right_camera,
    )
    if ray_gap > maximum_gap:
        raise RuntimeError(
            "Stereo projective front-rim center ray gap is "
            f"{ray_gap * 1000.0:.3f} mm; "
            f"limit is {maximum_gap * 1000.0:.3f} mm."
        )

    reprojection_errors = np.asarray(
        [
            np.linalg.norm(left_camera.project_world(center_world) - left_uv),
            np.linalg.norm(right_camera.project_world(center_world) - right_uv),
        ],
        dtype=np.float64,
    )
    maximum_error = float(np.max(reprojection_errors))
    rms_error = float(np.sqrt(np.mean(reprojection_errors * reprojection_errors)))
    if maximum_error > maximum_reprojection:
        raise RuntimeError(
            "Stereo projective front-rim center reprojection error is "
            f"{maximum_error:.3f} px; "
            f"limit is {maximum_reprojection:.3f} px."
        )

    return StereoApertureCenter(
        center_world_m=center_world,
        left_center_uv=left_uv,
        right_center_uv=right_uv,
        ray_gap_m=ray_gap,
        reprojection_rms_px=rms_error,
        max_reprojection_px=maximum_error,
    )
