#!/usr/bin/env python3
"""Pure geometry for mounting an RJ45 connector tip onto ToolCenter."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class PlugFrame:
    """Detected connector geometry in the tracked plug's local frame."""

    local_min_m: np.ndarray
    local_max_m: np.ndarray
    dimensions_m: np.ndarray
    longitudinal_axis_index: int
    wide_transverse_axis_index: int
    cable_side_sign: int
    tip_local_m: np.ndarray
    nose_axis_local: np.ndarray
    wide_axis_local: np.ndarray
    narrow_axis_local: np.ndarray
    plug_from_tip: np.ndarray


@dataclass(frozen=True)
class AttachmentBounds:
    """Plug-local mask volume used to attach only the connector region."""

    local_min_m: np.ndarray
    local_max_m: np.ndarray
    center_local_m: np.ndarray
    size_m: np.ndarray


@dataclass(frozen=True)
class CableMountValidation:
    """Maximum errors observed across the complete startup window."""

    frame_count: int
    maximum_tip_error_m: float
    maximum_axis_error_deg: float


def _finite(value, shape: tuple[int, ...], label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{label} must be finite with shape {shape}")
    return array


def _finite_nonnegative(value: float, label: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{label} must be finite and nonnegative")
    return number


def _finite_positive(value: float, label: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{label} must be finite and positive")
    return number


def _validate_homogeneous(matrix: np.ndarray, label: str) -> np.ndarray:
    transform = _finite(matrix, (4, 4), label)
    if not np.allclose(
        transform[3],
        [0.0, 0.0, 0.0, 1.0],
        atol=1e-9,
        rtol=0.0,
    ):
        raise ValueError(f"{label} must be homogeneous")
    return transform


def validate_affine_transform(matrix: np.ndarray, label: str) -> np.ndarray:
    """Validate a finite, nonsingular, right-handed affine transform."""

    transform = _validate_homogeneous(matrix, label)
    determinant = float(np.linalg.det(transform[:3, :3]))
    if not math.isfinite(determinant) or determinant <= 1.0e-12:
        raise ValueError(
            f"{label} linear transform must be nonsingular and right handed"
        )
    return transform


def validate_transform(matrix: np.ndarray, label: str) -> np.ndarray:
    """Validate and return a rigid right-handed homogeneous transform."""

    transform = _validate_homogeneous(matrix, label)
    rotation = transform[:3, :3]
    if not np.allclose(
        rotation.T @ rotation,
        np.eye(3),
        atol=1e-7,
        rtol=0.0,
    ):
        raise ValueError(f"{label} rotation must be orthonormal")
    if not math.isclose(
        float(np.linalg.det(rotation)),
        1.0,
        abs_tol=1e-7,
    ):
        raise ValueError(f"{label} rotation must be right handed")
    return transform


def rigid_pose_from_affine(matrix: np.ndarray, label: str) -> np.ndarray:
    """Return translation plus the nearest proper rotation, discarding scale/shear."""

    affine = validate_affine_transform(matrix, label)
    left, singular_values, right_t = np.linalg.svd(affine[:3, :3])
    if (
        singular_values.shape != (3,)
        or not np.all(np.isfinite(singular_values))
        or float(np.min(singular_values)) <= 1.0e-12
    ):
        raise ValueError(f"{label} linear transform is singular")

    rotation = left @ right_t
    if float(np.linalg.det(rotation)) < 0.0:
        left[:, -1] *= -1.0
        rotation = left @ right_t

    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = rotation
    pose[:3, 3] = affine[:3, 3]
    return validate_transform(pose, f"{label}_rigid_pose")


def matrix_to_quaternion_wxyz(rotation_matrix: np.ndarray) -> np.ndarray:
    """Convert a proper column-vector rotation matrix to scalar-first quaternion."""

    rotation = _finite(rotation_matrix, (3, 3), "rotation_matrix")
    candidate = np.eye(4, dtype=np.float64)
    candidate[:3, :3] = rotation
    validate_transform(candidate, "rotation_matrix")

    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (rotation[2, 1] - rotation[1, 2]) / scale
        y = (rotation[0, 2] - rotation[2, 0]) / scale
        z = (rotation[1, 0] - rotation[0, 1]) / scale
    elif rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
        scale = math.sqrt(
            1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]
        ) * 2.0
        w = (rotation[2, 1] - rotation[1, 2]) / scale
        x = 0.25 * scale
        y = (rotation[0, 1] + rotation[1, 0]) / scale
        z = (rotation[0, 2] + rotation[2, 0]) / scale
    elif rotation[1, 1] > rotation[2, 2]:
        scale = math.sqrt(
            1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]
        ) * 2.0
        w = (rotation[0, 2] - rotation[2, 0]) / scale
        x = (rotation[0, 1] + rotation[1, 0]) / scale
        y = 0.25 * scale
        z = (rotation[1, 2] + rotation[2, 1]) / scale
    else:
        scale = math.sqrt(
            1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]
        ) * 2.0
        w = (rotation[1, 0] - rotation[0, 1]) / scale
        x = (rotation[0, 2] + rotation[2, 0]) / scale
        y = (rotation[1, 2] + rotation[2, 1]) / scale
        z = 0.25 * scale

    quaternion = np.array([w, x, y, z], dtype=np.float64)
    quaternion /= np.linalg.norm(quaternion)
    if quaternion[0] < 0.0:
        quaternion *= -1.0
    return quaternion


def detect_plug_frame(
    local_min_m: np.ndarray,
    local_max_m: np.ndarray,
    world_from_plug: np.ndarray,
    cable_center_world_m: np.ndarray,
    *,
    axis_ratio_min: float,
    cable_projection_min_m: float,
) -> PlugFrame:
    """Detect the insertion tip, longitudinal direction, and deterministic roll."""

    local_min = _finite(local_min_m, (3,), "local_min_m")
    local_max = _finite(local_max_m, (3,), "local_max_m")
    world_from_plug = validate_affine_transform(
        world_from_plug,
        "world_from_plug",
    )
    cable_center_world = _finite(
        cable_center_world_m,
        (3,),
        "cable_center_world_m",
    )
    axis_ratio_min = _finite_positive(axis_ratio_min, "axis_ratio_min")
    cable_projection_min_m = _finite_nonnegative(
        cable_projection_min_m,
        "cable_projection_min_m",
    )

    if np.any(local_max <= local_min):
        raise ValueError("plug bounds must have positive dimensions")

    dimensions = local_max - local_min
    order = np.argsort(dimensions, kind="stable")
    longitudinal = int(order[-1])
    second = int(order[-2])
    if dimensions[longitudinal] / dimensions[second] < axis_ratio_min:
        raise ValueError("ambiguous longitudinal axis")

    plug_center = 0.5 * (local_min + local_max)
    plug_from_world = np.linalg.inv(world_from_plug)
    cable_local = (
        plug_from_world @ np.r_[cable_center_world, 1.0]
    )[:3]
    projection = float(
        cable_local[longitudinal] - plug_center[longitudinal]
    )
    if abs(projection) < cable_projection_min_m:
        raise ValueError("ambiguous cable-side projection")

    cable_side_sign = 1 if projection > 0.0 else -1
    nose_sign = -cable_side_sign

    transverse = [axis for axis in range(3) if axis != longitudinal]
    wide = max(transverse, key=lambda axis: (dimensions[axis], -axis))

    nose_axis = np.zeros(3, dtype=np.float64)
    nose_axis[longitudinal] = float(nose_sign)
    wide_axis = np.zeros(3, dtype=np.float64)
    wide_axis[wide] = 1.0
    narrow_axis = np.cross(wide_axis, nose_axis)
    narrow_norm = float(np.linalg.norm(narrow_axis))
    if narrow_norm <= 0.0 or not math.isfinite(narrow_norm):
        raise ValueError("connector transverse frame is degenerate")
    narrow_axis /= narrow_norm

    tip = plug_center.copy()
    tip[longitudinal] = (
        local_max[longitudinal]
        if nose_sign > 0
        else local_min[longitudinal]
    )

    plug_from_tip = np.eye(4, dtype=np.float64)
    plug_from_tip[:3, 0] = narrow_axis
    plug_from_tip[:3, 1] = wide_axis
    plug_from_tip[:3, 2] = nose_axis
    plug_from_tip[:3, 3] = tip
    validate_transform(plug_from_tip, "plug_from_tip")

    return PlugFrame(
        local_min_m=local_min.copy(),
        local_max_m=local_max.copy(),
        dimensions_m=dimensions.copy(),
        longitudinal_axis_index=longitudinal,
        wide_transverse_axis_index=wide,
        cable_side_sign=cable_side_sign,
        tip_local_m=tip,
        nose_axis_local=nose_axis,
        wide_axis_local=wide_axis,
        narrow_axis_local=narrow_axis,
        plug_from_tip=plug_from_tip,
    )


def compute_attachment_bounds(
    frame: PlugFrame,
    padding_m: float,
) -> AttachmentBounds:
    """Return plug-local mask bounds without extending into the cable tail."""

    padding = _finite_nonnegative(padding_m, "padding_m")
    local_min = np.asarray(frame.local_min_m, dtype=np.float64).copy()
    local_max = np.asarray(frame.local_max_m, dtype=np.float64).copy()
    longitudinal = frame.longitudinal_axis_index

    for axis in range(3):
        if axis != longitudinal:
            local_min[axis] -= padding
            local_max[axis] += padding

    if frame.nose_axis_local[longitudinal] > 0.0:
        local_max[longitudinal] += padding
    else:
        local_min[longitudinal] -= padding

    center = 0.5 * (local_min + local_max)
    size = local_max - local_min
    if np.any(size <= 0.0) or not np.all(np.isfinite(size)):
        raise ValueError("attachment bounds must have positive finite size")

    return AttachmentBounds(
        local_min_m=local_min,
        local_max_m=local_max,
        center_local_m=center,
        size_m=size,
    )


def compute_world_from_root_for_tip(
    world_from_root: np.ndarray,
    world_from_plug: np.ndarray,
    frame: PlugFrame,
    desired_world_from_tip: np.ndarray,
) -> np.ndarray:
    """Return a rigid root correction that preserves the plug's authored scale."""

    world_from_root = validate_transform(world_from_root, "world_from_root")
    world_from_plug = validate_affine_transform(
        world_from_plug,
        "world_from_plug",
    )
    desired = validate_transform(
        desired_world_from_tip,
        "desired_world_from_tip",
    )

    current_world_from_tip_affine = world_from_plug @ frame.plug_from_tip
    current_world_from_tip_pose = rigid_pose_from_affine(
        current_world_from_tip_affine,
        "current_world_from_tip",
    )
    rigid_world_correction = desired @ np.linalg.inv(
        current_world_from_tip_pose
    )
    mounted = rigid_world_correction @ world_from_root
    return validate_transform(mounted, "mounted_world_from_root")


def angular_error_deg(axis_a: np.ndarray, axis_b: np.ndarray) -> float:
    """Return the unsigned angle between two nonzero 3D axes in degrees."""

    first = _finite(axis_a, (3,), "axis_a")
    second = _finite(axis_b, (3,), "axis_b")
    first_norm = float(np.linalg.norm(first))
    second_norm = float(np.linalg.norm(second))
    if first_norm <= 0.0 or second_norm <= 0.0:
        raise ValueError("axes must be nonzero")
    first = first / first_norm
    second = second / second_norm
    cosine = float(np.clip(np.dot(first, second), -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def validate_mount_window(
    samples: Iterable[tuple[float, float]],
    required_frames: int,
    max_tip_error_m: float,
    max_axis_error_deg: float,
) -> CableMountValidation:
    """Fail if any frame in the complete startup window exceeds a gate."""

    if isinstance(required_frames, bool) or not isinstance(required_frames, int):
        raise ValueError("required_frames must be a positive integer")
    if required_frames <= 0:
        raise ValueError("required_frames must be a positive integer")
    tip_limit = _finite_nonnegative(max_tip_error_m, "max_tip_error_m")
    axis_limit = _finite_nonnegative(
        max_axis_error_deg,
        "max_axis_error_deg",
    )

    sample_list = list(samples)
    if len(sample_list) != required_frames:
        raise ValueError("mount validation requires the complete frame window")

    tip_errors: list[float] = []
    axis_errors: list[float] = []
    for sample in sample_list:
        if not isinstance(sample, (tuple, list)) or len(sample) != 2:
            raise ValueError("each mount sample must contain tip and axis error")
        tip_errors.append(_finite_nonnegative(sample[0], "tip errors"))
        axis_errors.append(_finite_nonnegative(sample[1], "axis errors"))

    maximum_tip = max(tip_errors)
    maximum_axis = max(axis_errors)
    if maximum_tip > tip_limit:
        raise RuntimeError("RJ45 tip mount error exceeds limit")
    if maximum_axis > axis_limit:
        raise RuntimeError("RJ45 axis error exceeds limit")

    return CableMountValidation(
        frame_count=required_frames,
        maximum_tip_error_m=maximum_tip,
        maximum_axis_error_deg=maximum_axis,
    )
