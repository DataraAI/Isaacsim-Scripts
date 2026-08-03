#!/usr/bin/env python3
"""Projective RJ45 image centers reconstructed on the outer-bezel plane."""

from __future__ import annotations

import math

import numpy as np

from front_mouth_projective_center import aperture_center_pixel
from front_plane import FrontPlaneConfig, intersect_pixel_with_plane
from outer_bezel_center import (
    OUTER_BEZEL_CONFIG,
    OuterBezelApertureResult,
    estimate_outer_bezel_plane,
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
    """Fuse outer-mouth image centers on the measured rack-face plane.

    The dark masks localize the port. RGB contrast polarity selects the
    physical outer left/right mouth edges instead of stronger recessed edges.
    Dense stereo support outside the opening determines the physical depth
    plane. No metric contour scaling or world-space correction is used.
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
            "Outer-plane center disagreement gate must be in (0, 0.5 mm]."
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

    left_uv = aperture_center_pixel(left_rgb, left_mask, left_camera)
    right_uv = aperture_center_pixel(right_rgb, right_mask, right_camera)
    left_point = intersect_pixel_with_plane(
        left_camera,
        left_uv,
        plane.center_world_m,
        plane.normal_world,
    )
    right_point = intersect_pixel_with_plane(
        right_camera,
        right_uv,
        plane.center_world_m,
        plane.normal_world,
    )
    disagreement = float(np.linalg.norm(left_point - right_point))
    if disagreement > maximum:
        raise RuntimeError(
            "Outer-bezel projective centers disagree by "
            f"{disagreement * 1000.0:.3f} mm; "
            f"limit is {maximum * 1000.0:.3f} mm."
        )

    return OuterBezelApertureResult(
        center_world_m=0.5 * (left_point + right_point),
        left_center_world_m=left_point,
        right_center_world_m=right_point,
        left_center_uv=left_uv,
        right_center_uv=right_uv,
        eye_disagreement_m=disagreement,
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
