#!/usr/bin/env python3
"""Qualified plane-rectified RGB front-lip center for angled stereo views."""

from __future__ import annotations

import math
import numpy as np

from plane_rectified_types import (
    DEFAULT_RECTIFIED_PADDING_M,
    DEFAULT_RECTIFIED_RESOLUTION_M,
    MAX_CENTER_DISAGREEMENT_M,
    MAX_EDGE_REPROJECTION_PX,
    PlaneRectifiedFrontLipDebug,
    PlaneRectifiedFrontLipResult,
)
from plane_rectified_geometry import (
    build_plane_frame,
    project_mask_bounds_to_plane,
    rectify_eye_to_plane,
)
from plane_rectified_fit_utils import (
    _draw_fit,
    _draw_reprojection,
    point_in_convex_quad,
)
from plane_rectified_fitting import (
    _fit_joint_front_lip,
    fit_rectified_front_lip,
)


_LATEST_DEBUG: PlaneRectifiedFrontLipDebug | None = None


def get_latest_plane_rectified_debug() -> PlaneRectifiedFrontLipDebug | None:
    return _LATEST_DEBUG


def estimate_plane_rectified_front_lip_center(
    *,
    left_rgb: np.ndarray,
    right_rgb: np.ndarray,
    left_mask: np.ndarray,
    right_mask: np.ndarray,
    left_camera,
    right_camera,
    plane_origin_world_m: np.ndarray,
    plane_normal_world: np.ndarray,
    aperture_width_m: float = 0.0114,
    aperture_height_m: float = 0.0070,
    padding_m: float = DEFAULT_RECTIFIED_PADDING_M,
    resolution_m: float = DEFAULT_RECTIFIED_RESOLUTION_M,
    max_center_disagreement_m: float = MAX_CENTER_DISAGREEMENT_M,
    max_edge_reprojection_px: float = MAX_EDGE_REPROJECTION_PX,
) -> PlaneRectifiedFrontLipResult:
    """Return one qualified physical center from two independent RGB fits."""

    global _LATEST_DEBUG
    _LATEST_DEBUG = None

    maximum = float(max_center_disagreement_m)
    if not math.isfinite(maximum) or not 0.0 < maximum <= MAX_CENTER_DISAGREEMENT_M:
        raise ValueError("Front-lip center-disagreement gate must be in (0, 0.5 mm].")

    frame = build_plane_frame(
        left_camera,
        right_camera,
        plane_origin_world_m,
        plane_normal_world,
    )
    left_minimum, left_maximum = project_mask_bounds_to_plane(
        left_mask, left_camera, frame
    )
    right_minimum, right_maximum = project_mask_bounds_to_plane(
        right_mask, right_camera, frame
    )
    minimum = np.minimum(left_minimum, right_minimum) - float(padding_m)
    maximum_bounds = np.maximum(left_maximum, right_maximum) + float(padding_m)
    bounds = (minimum, maximum_bounds)

    left_rectified = rectify_eye_to_plane(
        left_rgb,
        left_mask,
        left_camera,
        frame,
        bounds,
        resolution_m=resolution_m,
    )
    right_rectified = rectify_eye_to_plane(
        right_rgb,
        right_mask,
        right_camera,
        frame,
        bounds,
        resolution_m=resolution_m,
    )
    left_fit = fit_rectified_front_lip(
        left_rectified,
        aperture_width_m=aperture_width_m,
        aperture_height_m=aperture_height_m,
        max_edge_reprojection_px=max_edge_reprojection_px,
    )
    right_fit = fit_rectified_front_lip(
        right_rectified,
        aperture_width_m=aperture_width_m,
        aperture_height_m=aperture_height_m,
        max_edge_reprojection_px=max_edge_reprojection_px,
    )
    disagreement = float(
        np.linalg.norm(left_fit.center_uv_m - right_fit.center_uv_m)
    )
    if disagreement > maximum:
        raise RuntimeError(
            "Plane-rectified RGB front-lip centers disagree by "
            f"{disagreement * 1000.0:.3f} mm; "
            f"limit is {maximum * 1000.0:.3f} mm."
        )

    joint_fit = _fit_joint_front_lip(
        left_fit,
        right_fit,
        aperture_width_m=float(aperture_width_m),
        aperture_height_m=float(aperture_height_m),
        resolution_m=float(resolution_m),
    )
    if not point_in_convex_quad(
        joint_fit.center_uv_m,
        left_fit.corners_uv_m,
        tolerance=resolution_m,
    ):
        raise RuntimeError("Joint RGB center lies outside the left-eye quadrilateral.")
    if not point_in_convex_quad(
        joint_fit.center_uv_m,
        right_fit.corners_uv_m,
        tolerance=resolution_m,
    ):
        raise RuntimeError("Joint RGB center lies outside the right-eye quadrilateral.")

    center_world = frame.metric_to_world(joint_fit.center_uv_m)
    left_center_world = frame.metric_to_world(left_fit.center_uv_m)
    right_center_world = frame.metric_to_world(right_fit.center_uv_m)
    left_center_uv = left_camera.project_world(left_center_world)
    right_center_uv = right_camera.project_world(right_center_world)

    debug = PlaneRectifiedFrontLipDebug(
        left_rectified_rgb=left_rectified.rgb.copy(),
        right_rectified_rgb=right_rectified.rgb.copy(),
        left_overlay=_draw_fit(left_rectified, left_fit),
        right_overlay=_draw_fit(right_rectified, right_fit),
        joint_overlay=_draw_fit(left_rectified, joint_fit),
        left_reprojection=_draw_reprojection(
            left_rgb, left_camera, joint_fit, frame
        ),
        right_reprojection=_draw_reprojection(
            right_rgb, right_camera, joint_fit, frame
        ),
    )
    _LATEST_DEBUG = debug
    return PlaneRectifiedFrontLipResult(
        center_world_m=np.asarray(center_world, dtype=np.float64),
        left_center_world_m=np.asarray(left_center_world, dtype=np.float64),
        right_center_world_m=np.asarray(right_center_world, dtype=np.float64),
        left_center_uv=np.asarray(left_center_uv, dtype=np.float64),
        right_center_uv=np.asarray(right_center_uv, dtype=np.float64),
        center_disagreement_m=disagreement,
        left_fit=left_fit,
        right_fit=right_fit,
        joint_fit=joint_fit,
        plane_frame=frame,
        debug=debug,
    )
