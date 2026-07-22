#!/usr/bin/env python3
"""Live image-only front-opening control geometry."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from front_plane import estimate_front_plane


@dataclass(frozen=True)
class LiveFrontPlaneDiagnostics:
    cavity_range_m: float
    opening_range_m: float
    recess_depth_m: float
    plane_residual_m: float
    max_ray_gap_m: float
    valid_disparity_count: int
    consistent_disparity_count: int
    ring_candidate_count: int
    triangulated_count: int
    cluster_count: int
    side_support_counts: tuple[int, int, int, int]


def _project_camera_local(point_camera_usd_m, camera) -> np.ndarray:
    point = np.asarray(point_camera_usd_m, dtype=np.float64).reshape(3)
    range_m = -float(point[2])
    if range_m <= 0.0:
        raise RuntimeError("Desired point is behind the virtual camera.")
    return np.array(
        [
            camera.cx_px + camera.fx_px * float(point[0]) / range_m,
            camera.cy_px + camera.fy_px * (-float(point[1])) / range_m,
        ],
        dtype=np.float64,
    )


def _camera_error_to_world(
    current_camera_usd_m,
    desired_camera_usd_m,
    camera,
) -> np.ndarray:
    current = np.asarray(
        current_camera_usd_m,
        dtype=np.float64,
    ).reshape(3)
    desired = np.asarray(
        desired_camera_usd_m,
        dtype=np.float64,
    ).reshape(3)
    matrix = np.asarray(camera.world_from_camera, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError("virtual camera transform must be 4x4.")
    return np.asarray((np.append(current - desired, 0.0) @ matrix)[:3])


def apply_front_plane_result(
    frame,
    observation,
    desired_port_virtual_camera_usd,
    front_plane_result,
):
    """Replace recessed cavity geometry with fitted front-opening geometry."""
    virtual_camera = frame.virtual_camera
    center_world = np.asarray(
        front_plane_result.center_world_m,
        dtype=np.float64,
    ).reshape(3)
    center_virtual = virtual_camera.camera_point_from_world(center_world)
    opening_range_m = -float(center_virtual[2])
    if opening_range_m <= 0.0:
        raise RuntimeError(
            "Estimated front opening lies behind the virtual camera."
        )

    desired = np.asarray(
        desired_port_virtual_camera_usd,
        dtype=np.float64,
    ).reshape(3)
    desired_center_uv = _project_camera_local(desired, virtual_camera)
    projected_center_uv = virtual_camera.project_world(center_world)
    center_error_px = projected_center_uv - desired_center_uv
    range_error_m = opening_range_m - (-float(desired[2]))
    correction_world_m = _camera_error_to_world(
        center_virtual,
        desired,
        virtual_camera,
    )

    refined = replace(
        observation,
        corners_world_m=np.asarray(
            front_plane_result.corners_world_m,
            dtype=np.float64,
        ),
        center_world_xyz_m=center_world,
        center_virtual_camera_usd_m=np.asarray(
            center_virtual,
            dtype=np.float64,
        ),
        normal_world=np.asarray(
            front_plane_result.normal_world,
            dtype=np.float64,
        ),
        projected_virtual_center_uv=(
            float(projected_center_uv[0]),
            float(projected_center_uv[1]),
        ),
        center_error_px=np.asarray(center_error_px, dtype=np.float64),
        estimated_range_m=float(opening_range_m),
        range_error_m=float(range_error_m),
        correction_world_m=np.asarray(
            correction_world_m,
            dtype=np.float64,
        ),
        width_m=float(front_plane_result.width_m),
        height_m=float(front_plane_result.height_m),
        mean_disparity_px=float(front_plane_result.median_disparity_px),
        reprojection_rms_px=float(front_plane_result.reprojection_rms_px),
        max_reprojection_px=float(front_plane_result.max_reprojection_px),
        max_ray_gap_m=float(front_plane_result.max_ray_gap_m),
        plane_residual_m=float(front_plane_result.plane_residual_m),
    )

    cavity_range_m = float(observation.estimated_range_m)
    diagnostics = LiveFrontPlaneDiagnostics(
        cavity_range_m=cavity_range_m,
        opening_range_m=float(opening_range_m),
        recess_depth_m=float(cavity_range_m - opening_range_m),
        plane_residual_m=float(front_plane_result.plane_residual_m),
        max_ray_gap_m=float(front_plane_result.max_ray_gap_m),
        valid_disparity_count=int(
            front_plane_result.valid_disparity_count
        ),
        consistent_disparity_count=int(
            front_plane_result.consistent_disparity_count
        ),
        ring_candidate_count=int(front_plane_result.ring_candidate_count),
        triangulated_count=int(front_plane_result.triangulated_count),
        cluster_count=int(front_plane_result.cluster_count),
        side_support_counts=tuple(
            int(value) for value in front_plane_result.side_support_counts
        ),
    )
    return refined, diagnostics


def refine_live_observation(
    frame,
    observation,
    desired_port_virtual_camera_usd,
):
    """Run qualified SGBM and return front-opening control geometry."""
    result = estimate_front_plane(
        left_rgb=frame.left.rgb,
        right_rgb=frame.right.rgb,
        left_bbox_xywh=observation.left.detection.bbox_xywh,
        left_center_uv=observation.left.detection.center_uv,
        right_bbox_xywh=observation.right.detection.bbox_xywh,
        right_center_uv=observation.right.detection.center_uv,
        left_camera=frame.left.camera,
        right_camera=frame.right.camera,
    )
    return apply_front_plane_result(
        frame=frame,
        observation=observation,
        desired_port_virtual_camera_usd=desired_port_virtual_camera_usd,
        front_plane_result=result,
    )
