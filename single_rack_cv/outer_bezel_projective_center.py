#!/usr/bin/env python3
"""Projective RJ45 image centerline reconstructed at outer-bezel depth."""

from __future__ import annotations

import math

import numpy as np

from front_plane import FrontPlaneConfig, intersect_midpoint_ray_with_plane
from outer_bezel_center import (
    OUTER_BEZEL_CONFIG,
    OuterBezelApertureResult,
    estimate_outer_bezel_plane,
)
from stereo_center_projective import (
    estimate_stereo_aperture_center as estimate_projective_stereo_center,
)


MAX_OUTER_PLANE_CENTER_DISAGREEMENT_M = 0.0005


def estimate_outer_bezel_projective_center(
    *,
    left_rgb: np.ndarray,
    right_rgb: np.ndarray,
    left_mask: np.ndarray,
    right_mask: np.ndarray,
    left_bbox_xywh: tuple[int, int, int, int],
    right_bbox_xywh: tuple[int, int, int, int],
    left_detection_center_uv: tuple[float, float],
    right_detection_center_uv: tuple[float, float],
    left_camera,
    right_camera,
    aperture_width_m: float = 0.0114,
    aperture_height_m: float = 0.0070,
    front_plane_config: FrontPlaneConfig = OUTER_BEZEL_CONFIG,
    max_center_disagreement_m: float = MAX_OUTER_PLANE_CENTER_DISAGREEMENT_M,
) -> OuterBezelApertureResult:
    """Place the proven stereo centerline on the measured outer rack plane.

    The RGB front-rim estimator supplies one corresponding center pixel per
    eye. Direct stereo triangulation qualifies that correspondence and fixes
    lateral/vertical position. Dense stereo support outside the opening
    supplies only the physical front depth. The recessed mask contour is never
    interpreted as a metric 11.4 x 7.0 mm mouth contour, and independently
    intersected eye rays are never averaged through a weakly supported plane.
    """

    # Kept for API compatibility with the existing runtime configuration. The
    # physical dimensions are not used to turn the recessed semantic contour
    # into metric geometry.
    del aperture_width_m, aperture_height_m

    maximum = float(max_center_disagreement_m)
    if (
        not math.isfinite(maximum)
        or maximum <= 0.0
        or maximum > MAX_OUTER_PLANE_CENTER_DISAGREEMENT_M
    ):
        raise ValueError(
            "Projective stereo ray-gap gate must be in (0, 0.5 mm]."
        )

    plane = estimate_outer_bezel_plane(
        left_rgb=left_rgb,
        right_rgb=right_rgb,
        left_bbox_xywh=left_bbox_xywh,
        left_detection_center_uv=left_detection_center_uv,
        right_bbox_xywh=right_bbox_xywh,
        right_detection_center_uv=right_detection_center_uv,
        left_camera=left_camera,
        right_camera=right_camera,
        front_plane_config=front_plane_config,
    )
    stereo_center = estimate_projective_stereo_center(
        left_rgb=left_rgb,
        right_rgb=right_rgb,
        left_mask=left_mask,
        right_mask=right_mask,
        left_camera=left_camera,
        right_camera=right_camera,
        max_ray_gap_m=maximum,
    )
    center_world = intersect_midpoint_ray_with_plane(
        left_camera,
        right_camera,
        stereo_center.left_center_uv,
        stereo_center.right_center_uv,
        plane.center_world_m,
        plane.normal_world,
    )

    return OuterBezelApertureResult(
        center_world_m=center_world,
        left_center_world_m=center_world,
        right_center_world_m=center_world,
        left_center_uv=stereo_center.left_center_uv,
        right_center_uv=stereo_center.right_center_uv,
        eye_disagreement_m=stereo_center.ray_gap_m,
        plane_origin_world_m=plane.center_world_m,
        plane_normal_world=plane.normal_world,
        corners_world_m=plane.corners_world_m,
        width_m=plane.width_m,
        height_m=plane.height_m,
        plane_residual_m=plane.plane_residual_m,
        max_ray_gap_m=plane.max_ray_gap_m,
        reprojection_rms_px=plane.reprojection_rms_px,
        max_reprojection_px=plane.max_reprojection_px,
        valid_disparity_count=plane.valid_disparity_count,
        consistent_disparity_count=plane.consistent_disparity_count,
        ring_candidate_count=plane.ring_candidate_count,
        triangulated_count=plane.triangulated_count,
        cluster_count=plane.cluster_count,
        side_support_counts=plane.side_support_counts,
        support_region_count=plane.support_region_count,
        spatial_region_counts=plane.spatial_region_counts,
        support_span_u_px=plane.support_span_u_px,
        support_span_v_px=plane.support_span_v_px,
        support_minor_std_px=plane.support_minor_std_px,
        median_disparity_px=plane.median_disparity_px,
        disparity=plane.disparity,
        front_plane_config=front_plane_config,
    )
