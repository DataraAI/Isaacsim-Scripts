#!/usr/bin/env python3
"""Live-control adapter for dense stereo outer-bezel plane depth."""

from __future__ import annotations

from live_control import apply_front_plane_result
from outer_bezel_center import estimate_outer_bezel_aperture_center


def refine_live_observation(
    frame,
    observation,
    desired_port_virtual_camera_usd,
    *,
    aperture_width_m: float = 0.0114,
    aperture_height_m: float = 0.0070,
):
    """Estimate the physical opening center on the outer rack-face plane."""

    outer_result = estimate_outer_bezel_aperture_center(
        left_rgb=frame.left.rgb,
        right_rgb=frame.right.rgb,
        left_mask=observation.left.detection.mask,
        right_mask=observation.right.detection.mask,
        left_bbox_xywh=observation.left.detection.bbox_xywh,
        right_bbox_xywh=observation.right.detection.bbox_xywh,
        left_detection_center_uv=observation.left.detection.center_uv,
        right_detection_center_uv=observation.right.detection.center_uv,
        left_camera=frame.left.camera,
        right_camera=frame.right.camera,
        aperture_width_m=aperture_width_m,
        aperture_height_m=aperture_height_m,
    )
    return apply_front_plane_result(
        frame=frame,
        observation=observation,
        desired_port_virtual_camera_usd=desired_port_virtual_camera_usd,
        front_plane_result=outer_result,
        aperture_center_disagreement_m=outer_result.eye_disagreement_m,
    )
