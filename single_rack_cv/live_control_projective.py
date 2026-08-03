#!/usr/bin/env python3
"""Live-control adapter for the physical RGB front-rim plane estimator."""

from __future__ import annotations

from live_control import apply_stereo_center_result
from stereo_front_rim_plane import estimate_stereo_aperture_center


def refine_live_observation(
    frame,
    observation,
    desired_port_virtual_camera_usd,
    *,
    aperture_width_m: float = 0.0114,
    aperture_height_m: float = 0.0070,
):
    """Estimate the centered point on the triangulated physical mouth plane."""

    # Kept only for API compatibility. No empirical metric or image offset is
    # applied. Four corresponding RGB front-rim corners define the mouth plane,
    # and each eye's projective center ray is intersected with that plane.
    del aperture_width_m, aperture_height_m

    stereo_center = estimate_stereo_aperture_center(
        left_rgb=frame.left.rgb,
        right_rgb=frame.right.rgb,
        left_mask=observation.left.detection.mask,
        right_mask=observation.right.detection.mask,
        left_camera=frame.left.camera,
        right_camera=frame.right.camera,
    )
    return apply_stereo_center_result(
        frame=frame,
        observation=observation,
        desired_port_virtual_camera_usd=desired_port_virtual_camera_usd,
        stereo_center_result=stereo_center,
    )
