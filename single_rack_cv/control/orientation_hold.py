#!/usr/bin/env python3
"""Pure bounded quaternion feedback for holding insertion orientation."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

_EPS = 1.0e-12


def _quaternion_wxyz(value, *, label: str) -> np.ndarray:
    quaternion = np.asarray(value, dtype=np.float64).reshape(-1)
    if quaternion.shape != (4,):
        raise ValueError(f"{label} must have shape (4,), got {quaternion.shape}")
    if not np.all(np.isfinite(quaternion)):
        raise ValueError(f"{label} must contain only finite values")
    norm = float(np.linalg.norm(quaternion))
    if norm <= _EPS:
        raise ValueError(f"{label} cannot have zero length")
    return quaternion / norm


def quaternion_multiply_wxyz(left, right) -> np.ndarray:
    """Compose two scalar-first quaternions: world_delta * orientation."""

    w1, x1, y1, z1 = _quaternion_wxyz(left, label="left")
    w2, x2, y2, z2 = _quaternion_wxyz(right, label="right")
    return _quaternion_wxyz(
        np.array(
            [
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ],
            dtype=np.float64,
        ),
        label="product",
    )


def quaternion_inverse_wxyz(value) -> np.ndarray:
    quaternion = _quaternion_wxyz(value, label="quaternion")
    return np.array(
        [
            quaternion[0],
            -quaternion[1],
            -quaternion[2],
            -quaternion[3],
        ],
        dtype=np.float64,
    )


def quaternion_error_deg(reference_wxyz, actual_wxyz) -> float:
    error = quaternion_multiply_wxyz(
        reference_wxyz,
        quaternion_inverse_wxyz(actual_wxyz),
    )
    if error[0] < 0.0:
        error = -error
    return math.degrees(
        2.0 * math.acos(float(np.clip(error[0], -1.0, 1.0)))
    )


def _scaled_shortest_error_quaternion(
    reference_wxyz,
    actual_wxyz,
    *,
    gain: float,
    maximum_step_deg: float,
) -> tuple[np.ndarray, float]:
    gain = float(gain)
    maximum_step_deg = float(maximum_step_deg)
    if not math.isfinite(gain) or gain <= 0.0:
        raise ValueError("gain must be finite and positive")
    if not math.isfinite(maximum_step_deg) or maximum_step_deg <= 0.0:
        raise ValueError("maximum_step_deg must be finite and positive")

    error = quaternion_multiply_wxyz(
        reference_wxyz,
        quaternion_inverse_wxyz(actual_wxyz),
    )
    if error[0] < 0.0:
        error = -error

    angle_rad = 2.0 * math.acos(float(np.clip(error[0], -1.0, 1.0)))
    error_deg = math.degrees(angle_rad)
    if angle_rad <= _EPS:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64), error_deg

    sine_half = math.sin(0.5 * angle_rad)
    if abs(sine_half) <= _EPS:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64), error_deg

    axis = error[1:] / sine_half
    axis /= float(np.linalg.norm(axis))
    step_rad = min(
        gain * angle_rad,
        math.radians(maximum_step_deg),
    )
    return (
        _quaternion_wxyz(
            np.r_[
                math.cos(0.5 * step_rad),
                axis * math.sin(0.5 * step_rad),
            ],
            label="scaled_error",
        ),
        error_deg,
    )


def _cap_command_bias(
    command_wxyz,
    reference_wxyz,
    *,
    maximum_bias_deg: float,
) -> tuple[np.ndarray, float, bool]:
    maximum_bias_deg = float(maximum_bias_deg)
    if not math.isfinite(maximum_bias_deg) or maximum_bias_deg <= 0.0:
        raise ValueError("maximum_bias_deg must be finite and positive")

    bias = quaternion_multiply_wxyz(
        command_wxyz,
        quaternion_inverse_wxyz(reference_wxyz),
    )
    if bias[0] < 0.0:
        bias = -bias
    angle_rad = 2.0 * math.acos(float(np.clip(bias[0], -1.0, 1.0)))
    bias_deg = math.degrees(angle_rad)
    if bias_deg <= maximum_bias_deg + 1.0e-12:
        return _quaternion_wxyz(command_wxyz, label="command"), bias_deg, False

    sine_half = math.sin(0.5 * angle_rad)
    if abs(sine_half) <= _EPS:
        return _quaternion_wxyz(reference_wxyz, label="reference"), 0.0, True

    axis = bias[1:] / sine_half
    axis /= float(np.linalg.norm(axis))
    capped_rad = math.radians(maximum_bias_deg)
    capped_bias = _quaternion_wxyz(
        np.r_[
            math.cos(0.5 * capped_rad),
            axis * math.sin(0.5 * capped_rad),
        ],
        label="capped_bias",
    )
    return (
        quaternion_multiply_wxyz(capped_bias, reference_wxyz),
        maximum_bias_deg,
        True,
    )


@dataclass(frozen=True)
class OrientationHoldUpdate:
    command_wxyz: np.ndarray
    actual_error_deg: float
    command_bias_deg: float
    bias_saturated: bool


def update_orientation_hold_command(
    *,
    reference_wxyz,
    actual_wxyz,
    current_command_wxyz,
    gain: float = 0.35,
    maximum_step_deg: float = 0.15,
    maximum_bias_deg: float = 3.0,
) -> OrientationHoldUpdate:
    """Integrate bounded measured-orientation feedback into the IK command."""

    reference = _quaternion_wxyz(reference_wxyz, label="reference_wxyz")
    actual = _quaternion_wxyz(actual_wxyz, label="actual_wxyz")
    current = _quaternion_wxyz(
        current_command_wxyz,
        label="current_command_wxyz",
    )
    delta, actual_error_deg = _scaled_shortest_error_quaternion(
        reference,
        actual,
        gain=gain,
        maximum_step_deg=maximum_step_deg,
    )
    candidate = quaternion_multiply_wxyz(delta, current)
    command, command_bias_deg, saturated = _cap_command_bias(
        candidate,
        reference,
        maximum_bias_deg=maximum_bias_deg,
    )
    return OrientationHoldUpdate(
        command_wxyz=command,
        actual_error_deg=actual_error_deg,
        command_bias_deg=command_bias_deg,
        bias_saturated=saturated,
    )
