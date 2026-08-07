#!/usr/bin/env python3
"""Scale-preserving root placement for the pregrasped cable asset."""

from __future__ import annotations

import numpy as np

from cable.cable_geometry import (
    PlugFrame,
    rigid_pose_from_affine,
    validate_affine_transform,
    validate_transform,
)


def compute_world_from_root_for_tip_preserving_affine(
    world_from_root: np.ndarray,
    world_from_plug: np.ndarray,
    frame: PlugFrame,
    desired_world_from_tip: np.ndarray,
) -> np.ndarray:
    """Apply only a rigid world correction while preserving authored root scale."""

    root_affine = validate_affine_transform(
        world_from_root,
        "world_from_root",
    )
    plug_affine = validate_affine_transform(
        world_from_plug,
        "world_from_plug",
    )
    desired_tip_pose = validate_transform(
        desired_world_from_tip,
        "desired_world_from_tip",
    )

    current_tip_pose = rigid_pose_from_affine(
        plug_affine @ frame.plug_from_tip,
        "current_world_from_tip",
    )
    rigid_world_correction = (
        desired_tip_pose @ np.linalg.inv(current_tip_pose)
    )
    mounted_root = rigid_world_correction @ root_affine
    return validate_affine_transform(
        mounted_root,
        "mounted_world_from_root",
    )
