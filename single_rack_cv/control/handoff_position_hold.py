#!/usr/bin/env python3
"""Bounded position-command compensation for frozen ToolCenter handoff goals."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HandoffPositionHoldUpdate:
    command_position_m: np.ndarray
    step_world_m: np.ndarray
    physical_error_m: float
    command_bias_m: float
    bias_saturated: bool


def _vector3(value, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (3,):
        raise ValueError(f"{name} must be three-dimensional.")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values.")
    return vector


def update_handoff_position_command(
    *,
    goal_position_m,
    actual_position_m,
    current_command_position_m,
    gain: float,
    maximum_step_m: float,
    maximum_bias_m: float,
) -> HandoffPositionHoldUpdate:
    """Move the IK command enough to remove static bias at a frozen physical goal."""

    goal = _vector3(goal_position_m, "goal_position_m")
    actual = _vector3(actual_position_m, "actual_position_m")
    current = _vector3(
        current_command_position_m,
        "current_command_position_m",
    )
    gain_value = float(gain)
    maximum_step = float(maximum_step_m)
    maximum_bias = float(maximum_bias_m)
    if not np.isfinite(gain_value) or gain_value < 0.0:
        raise ValueError("gain must be finite and nonnegative.")
    if not np.isfinite(maximum_step) or maximum_step <= 0.0:
        raise ValueError("maximum_step_m must be finite and positive.")
    if not np.isfinite(maximum_bias) or maximum_bias <= 0.0:
        raise ValueError("maximum_bias_m must be finite and positive.")

    physical_error_world_m = goal - actual
    physical_error_m = float(np.linalg.norm(physical_error_world_m))
    requested_step = gain_value * physical_error_world_m
    requested_step_norm = float(np.linalg.norm(requested_step))
    if requested_step_norm > maximum_step:
        requested_step = requested_step * (maximum_step / requested_step_norm)

    requested_command = current + requested_step
    requested_bias = requested_command - goal
    requested_bias_m = float(np.linalg.norm(requested_bias))
    bias_saturated = requested_bias_m > maximum_bias
    if bias_saturated:
        requested_command = goal + requested_bias * (
            maximum_bias / requested_bias_m
        )

    step_world_m = requested_command - current
    return HandoffPositionHoldUpdate(
        command_position_m=requested_command.copy(),
        step_world_m=step_world_m.copy(),
        physical_error_m=physical_error_m,
        command_bias_m=float(np.linalg.norm(requested_command - goal)),
        bias_saturated=bias_saturated,
    )


__all__ = [
    "HandoffPositionHoldUpdate",
    "update_handoff_position_command",
]
