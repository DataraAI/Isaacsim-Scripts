#!/usr/bin/env python3
"""Physical RGB front-lip per-eye and joint fitters."""

from __future__ import annotations

import math
import cv2
import numpy as np

from plane_rectified_types import (
    FrontLipFit,
    MAX_EDGE_REPROJECTION_PX,
    MAX_OPPOSITE_EDGE_ANGLE_DEG,
    RectifiedEye,
)
from plane_rectified_fit_utils import (
    _edge_reprojection_residual,
    _fit_parallel_pair,
    _line_angle_deg,
    _mask_lower_mouth_geometry,
    _projective_center,
    _quad_from_lines,
    _qualified_signed_index,
    _robust_line,
    point_in_convex_quad,
)


def fit_rectified_front_lip(
    rectified: RectifiedEye,
    *,
    aperture_width_m: float = 0.0114,
    aperture_height_m: float = 0.0070,
    max_edge_reprojection_px: float = MAX_EDGE_REPROJECTION_PX,
) -> FrontLipFit:
    """Fit one independent physical front-lip quadrilateral from RGB."""

    resolution = rectified.resolution_m
    broad_start, broad_end, mask_left, mask_right = _mask_lower_mouth_geometry(
        rectified.mask
    )
    gray = cv2.cvtColor(rectified.rgb, cv2.COLOR_RGB2GRAY)
    normalized = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    smooth = cv2.GaussianBlur(normalized.astype(np.float64), (5, 5), 0)
    gradient_x = cv2.Sobel(smooth, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(smooth, cv2.CV_64F, 0, 1, ksize=3)

    exterior_span = int(round(0.45 * float(aperture_width_m) / resolution))
    interior_span = int(round(0.0010 / resolution))
    left_start = max(1, int(round(mask_left)) - exterior_span)
    left_end = min(rectified.rgb.shape[1] - 2, int(round(mask_left)) + interior_span)
    right_start = max(1, int(round(mask_right)) - interior_span)
    right_end = min(rectified.rgb.shape[1] - 2, int(round(mask_right)) + exterior_span)
    side_row_start = max(1, broad_start - 2)
    side_row_end = min(
        rectified.rgb.shape[0] - 2,
        broad_end + int(round(0.0020 / resolution)),
    )

    left_samples = []
    right_samples = []
    for row in range(side_row_start, side_row_end + 1):
        left_column = _qualified_signed_index(
            gradient_x[row, left_start : left_end + 1],
            start=left_start,
            negative=True,
            minimum_strength=6.0,
            relative_strength=0.14,
            choose_last=False,
        )
        right_column = _qualified_signed_index(
            gradient_x[row, right_start : right_end + 1],
            start=right_start,
            negative=False,
            minimum_strength=8.0,
            relative_strength=0.14,
            choose_last=True,
        )
        if left_column is not None:
            left_samples.append((float(left_column), float(row)))
        if right_column is not None:
            right_samples.append((float(right_column), float(row)))

    left_raw, left_keep, _ = _robust_line(
        np.asarray(left_samples), x_from_y=True, residual_floor=2.0
    )
    right_raw, right_keep, _ = _robust_line(
        np.asarray(right_samples), x_from_y=True, residual_floor=2.0
    )
    left_inliers = np.asarray(left_samples, dtype=np.float64)[left_keep]
    right_inliers = np.asarray(right_samples, dtype=np.float64)[right_keep]

    reference_y = 0.5 * float(broad_start + broad_end)
    left_at_reference = left_raw[0] * reference_y + left_raw[1]
    right_at_reference = right_raw[0] * reference_y + right_raw[1]
    opening_width_px = float(right_at_reference - left_at_reference)
    if opening_width_px <= 0.0:
        raise RuntimeError("RGB front-lip side walls cross.")

    top_samples = []
    top_start = max(1, broad_start - int(round(0.0015 / resolution)))
    top_end = min(
        rectified.rgb.shape[0] - 2,
        broad_start + int(round(0.0008 / resolution)),
    )
    first_column = max(1, int(math.floor(left_at_reference)))
    last_column = min(
        rectified.rgb.shape[1] - 2,
        int(math.ceil(right_at_reference)),
    )
    for column in range(first_column, last_column + 1):
        fraction = (float(column) - left_at_reference) / opening_width_px
        if not (fraction <= 0.32 or fraction >= 0.68):
            continue
        values = gradient_y[top_start : top_end + 1, column]
        strongest = max(0.0, -float(np.min(values)))
        threshold = max(8.0, 0.12 * strongest)
        candidates = np.flatnonzero(values <= -threshold) + top_start
        if candidates.size:
            row = int(candidates[np.argmin(np.abs(candidates - broad_start))])
            top_samples.append((float(column), float(row)))

    bottom_samples = []
    bottom_start = max(1, broad_end - int(round(0.0007 / resolution)))
    bottom_end = min(
        rectified.rgb.shape[0] - 2,
        broad_end + int(round(0.0032 / resolution)),
    )
    for column in range(first_column + 3, last_column - 2):
        values = gradient_y[bottom_start : bottom_end + 1, column]
        strongest = max(0.0, float(np.max(values)))
        threshold = max(7.0, 0.12 * strongest)
        candidates = np.flatnonzero(values >= threshold) + bottom_start
        if candidates.size:
            bottom_samples.append((float(column), float(candidates[-1])))

    top_raw, top_keep, _ = _robust_line(
        np.asarray(top_samples), x_from_y=False, residual_floor=2.0
    )
    bottom_raw, bottom_keep, _ = _robust_line(
        np.asarray(bottom_samples), x_from_y=False, residual_floor=2.5
    )
    top_inliers = np.asarray(top_samples, dtype=np.float64)[top_keep]
    bottom_inliers = np.asarray(bottom_samples, dtype=np.float64)[bottom_keep]

    side_angle = _line_angle_deg(left_raw, right_raw, side=True)
    horizontal_angle = _line_angle_deg(top_raw, bottom_raw, side=False)
    if max(side_angle, horizontal_angle) > MAX_OPPOSITE_EDGE_ANGLE_DEG:
        raise RuntimeError(
            "Rectified RGB front-lip opposite edges are not parallel enough: "
            f"side={side_angle:.3f}deg horizontal={horizontal_angle:.3f}deg."
        )

    left, right, left_parallel_keep, right_parallel_keep = _fit_parallel_pair(
        left_inliers,
        right_inliers,
        x_from_y=True,
        residual_floor=2.0,
    )
    top, bottom, top_parallel_keep, bottom_parallel_keep = _fit_parallel_pair(
        top_inliers,
        bottom_inliers,
        x_from_y=False,
        residual_floor=2.0,
    )
    left_inliers = left_inliers[left_parallel_keep]
    right_inliers = right_inliers[right_parallel_keep]
    top_inliers = top_inliers[top_parallel_keep]
    bottom_inliers = bottom_inliers[bottom_parallel_keep]
    corners_px = _quad_from_lines(left, right, top, bottom)
    center_px = _projective_center(corners_px)
    if not point_in_convex_quad(center_px, corners_px, tolerance=1.0e-6):
        raise RuntimeError("RGB front-lip center lies outside its quadrilateral.")

    named_pixel_samples = {
        "left": left_inliers,
        "right": right_inliers,
        "top": top_inliers,
        "bottom": bottom_inliers,
    }
    lines = {"left": left, "right": right, "top": top, "bottom": bottom}
    residuals = {
        name: _edge_reprojection_residual(rectified, name, samples, lines[name])
        for name, samples in named_pixel_samples.items()
    }
    maximum_residual = max(residuals.values())
    if maximum_residual > float(max_edge_reprojection_px):
        raise RuntimeError(
            "Rectified RGB front-lip edge reprojection residual is "
            f"{maximum_residual:.3f}px; limit is "
            f"{float(max_edge_reprojection_px):.3f}px."
        )

    corners_metric = rectified.pixel_to_metric(corners_px)
    center_metric = rectified.pixel_to_metric(center_px)
    edge_samples_metric = {
        name: rectified.pixel_to_metric(samples)
        for name, samples in named_pixel_samples.items()
    }
    top_width = float(np.linalg.norm(corners_metric[1] - corners_metric[0]))
    bottom_width = float(np.linalg.norm(corners_metric[2] - corners_metric[3]))
    left_height = float(np.linalg.norm(corners_metric[3] - corners_metric[0]))
    right_height = float(np.linalg.norm(corners_metric[2] - corners_metric[1]))
    width_m = 0.5 * (top_width + bottom_width)
    height_m = 0.5 * (left_height + right_height)
    minimum_width = 0.70 * float(aperture_width_m)
    maximum_width = 1.30 * float(aperture_width_m)
    minimum_height = 0.70 * float(aperture_height_m)
    maximum_height = 1.30 * float(aperture_height_m)
    if not minimum_width <= width_m <= maximum_width:
        raise RuntimeError(
            f"RGB front-lip width {width_m * 1000.0:.3f}mm is implausible."
        )
    if not minimum_height <= height_m <= maximum_height:
        raise RuntimeError(
            f"RGB front-lip height {height_m * 1000.0:.3f}mm is implausible."
        )

    return FrontLipFit(
        left_line=left,
        right_line=right,
        top_line=top,
        bottom_line=bottom,
        corners_uv_m=corners_metric,
        center_uv_m=center_metric,
        edge_samples_uv_m=edge_samples_metric,
        support_counts=(
            left_inliers.shape[0],
            right_inliers.shape[0],
            top_inliers.shape[0],
            bottom_inliers.shape[0],
        ),
        residual_px=maximum_residual,
        width_m=width_m,
        height_m=height_m,
    )


def _fit_joint_front_lip(
    left_fit: FrontLipFit,
    right_fit: FrontLipFit,
    *,
    aperture_width_m: float,
    aperture_height_m: float,
    resolution_m: float,
) -> FrontLipFit:
    samples = {
        name: np.vstack(
            (left_fit.edge_samples_uv_m[name], right_fit.edge_samples_uv_m[name])
        )
        for name in ("left", "right", "top", "bottom")
    }
    _, left_keep, _ = _robust_line(
        samples["left"], x_from_y=True, residual_floor=2.0 * resolution_m
    )
    _, right_keep, _ = _robust_line(
        samples["right"], x_from_y=True, residual_floor=2.0 * resolution_m
    )
    _, top_keep, _ = _robust_line(
        samples["top"], x_from_y=False, residual_floor=2.0 * resolution_m
    )
    _, bottom_keep, _ = _robust_line(
        samples["bottom"], x_from_y=False, residual_floor=2.5 * resolution_m
    )
    kept = {
        "left": samples["left"][left_keep],
        "right": samples["right"][right_keep],
        "top": samples["top"][top_keep],
        "bottom": samples["bottom"][bottom_keep],
    }
    left, right, left_parallel_keep, right_parallel_keep = _fit_parallel_pair(
        kept["left"],
        kept["right"],
        x_from_y=True,
        residual_floor=2.0 * resolution_m,
    )
    top, bottom, top_parallel_keep, bottom_parallel_keep = _fit_parallel_pair(
        kept["top"],
        kept["bottom"],
        x_from_y=False,
        residual_floor=2.0 * resolution_m,
    )
    kept = {
        "left": kept["left"][left_parallel_keep],
        "right": kept["right"][right_parallel_keep],
        "top": kept["top"][top_parallel_keep],
        "bottom": kept["bottom"][bottom_parallel_keep],
    }
    corners = _quad_from_lines(left, right, top, bottom)
    center = _projective_center(corners)
    width_m = 0.5 * (
        np.linalg.norm(corners[1] - corners[0])
        + np.linalg.norm(corners[2] - corners[3])
    )
    height_m = 0.5 * (
        np.linalg.norm(corners[3] - corners[0])
        + np.linalg.norm(corners[2] - corners[1])
    )
    if not 0.70 * aperture_width_m <= width_m <= 1.30 * aperture_width_m:
        raise RuntimeError("Joint RGB front-lip width is implausible.")
    if not 0.70 * aperture_height_m <= height_m <= 1.30 * aperture_height_m:
        raise RuntimeError("Joint RGB front-lip height is implausible.")
    return FrontLipFit(
        left_line=left,
        right_line=right,
        top_line=top,
        bottom_line=bottom,
        corners_uv_m=corners,
        center_uv_m=center,
        edge_samples_uv_m=kept,
        support_counts=tuple(
            kept[name].shape[0] for name in ("left", "right", "top", "bottom")
        ),
        residual_px=max(left_fit.residual_px, right_fit.residual_px),
        width_m=float(width_m),
        height_m=float(height_m),
    )
