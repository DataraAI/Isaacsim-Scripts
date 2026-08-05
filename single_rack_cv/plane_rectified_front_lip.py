#!/usr/bin/env python3
"""Qualified plane-rectified RGB front-lip center for angled stereo views."""

from __future__ import annotations

import math
from pathlib import Path

import cv2
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


def _write_debug_image(filename: str, image: np.ndarray) -> None:
    output_dir = Path(__file__).resolve().parent / "camera_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    success = cv2.imwrite(
        str(output_dir / filename),
        cv2.cvtColor(
            np.asarray(image, dtype=np.uint8),
            cv2.COLOR_RGB2BGR,
        ),
    )
    if not success:
        print(
            f"[RGB FRONT LIP] WARNING: could not save debug image {filename}",
            flush=True,
        )


def _save_debug_images(debug: PlaneRectifiedFrontLipDebug) -> None:
    images = {
        "front_lip_rectified_left.png": debug.left_rectified_rgb,
        "front_lip_rectified_right.png": debug.right_rectified_rgb,
        "front_lip_fit_left.png": debug.left_overlay,
        "front_lip_fit_right.png": debug.right_overlay,
        "front_lip_fit_joint.png": debug.joint_overlay,
        "front_lip_reprojection_left.png": debug.left_reprojection,
        "front_lip_reprojection_right.png": debug.right_reprojection,
    }
    for filename, image in images.items():
        _write_debug_image(filename, image)


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
    search_width_m: float | None = None,
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
    _write_debug_image("front_lip_rectified_left.png", left_rectified.rgb)
    _write_debug_image("front_lip_rectified_right.png", right_rectified.rgb)

    left_fit = fit_rectified_front_lip(
        left_rectified,
        aperture_width_m=aperture_width_m,
        aperture_height_m=aperture_height_m,
        search_width_m=search_width_m,
        max_edge_reprojection_px=max_edge_reprojection_px,
    )
    _write_debug_image(
        "front_lip_fit_left.png",
        _draw_fit(left_rectified, left_fit),
    )
    _write_debug_image(
        "front_lip_reprojection_left_eye_fit.png",
        _draw_reprojection(left_rgb, left_camera, left_fit, frame),
    )

    right_fit = fit_rectified_front_lip(
        right_rectified,
        aperture_width_m=aperture_width_m,
        aperture_height_m=aperture_height_m,
        search_width_m=search_width_m,
        max_edge_reprojection_px=max_edge_reprojection_px,
    )
    _write_debug_image(
        "front_lip_fit_right.png",
        _draw_fit(right_rectified, right_fit),
    )
    _write_debug_image(
        "front_lip_reprojection_right_eye_fit.png",
        _draw_reprojection(right_rgb, right_camera, right_fit, frame),
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
    _save_debug_images(debug)
    print(
        "[RGB FRONT LIP] "
        f"pair={disagreement * 1000.0:.3f}mm "
        f"residual={left_fit.residual_px:.3f}/"
        f"{right_fit.residual_px:.3f}px "
        f"left_size={left_fit.width_m * 1000.0:.3f}x"
        f"{left_fit.height_m * 1000.0:.3f}mm "
        f"right_size={right_fit.width_m * 1000.0:.3f}x"
        f"{right_fit.height_m * 1000.0:.3f}mm "
        f"joint_size={joint_fit.width_m * 1000.0:.3f}x"
        f"{joint_fit.height_m * 1000.0:.3f}mm "
        f"supports={left_fit.support_counts}/{right_fit.support_counts}",
        flush=True,
    )
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
