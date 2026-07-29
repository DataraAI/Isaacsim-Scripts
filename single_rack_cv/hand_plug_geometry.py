#!/usr/bin/env python3
"""Pure geometry for a pitched Franka hand carrying a horizontal RJ45 plug."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

_EPS = 1.0e-12


def _rotation3(value, *, label: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{label} must be a finite 3x3 matrix")
    if not np.allclose(matrix.T @ matrix, np.eye(3), atol=1.0e-9):
        raise ValueError(f"{label} must be orthonormal")
    if not math.isclose(float(np.linalg.det(matrix)), 1.0, abs_tol=1.0e-9):
        raise ValueError(f"{label} must have determinant +1")
    return matrix.copy()


def _axis3(value, *, label: str) -> np.ndarray:
    axis = np.asarray(value, dtype=np.float64).reshape(-1)
    if axis.shape != (3,) or not np.all(np.isfinite(axis)):
        raise ValueError(f"{label} must be a finite length-3 vector")
    norm = float(np.linalg.norm(axis))
    if norm <= _EPS:
        raise ValueError(f"{label} cannot be zero")
    return axis / norm


def validate_downward_hand_pitch_deg(
    value: float,
    maximum_deg: float = 45.0,
) -> float:
    pitch = float(value)
    maximum = float(maximum_deg)
    if not math.isfinite(maximum) or maximum <= 0.0:
        raise ValueError("maximum_deg must be finite and positive")
    if not math.isfinite(pitch) or pitch < 0.0 or pitch > maximum:
        raise ValueError(
            f"downward hand pitch must be finite in [0, {maximum}], got {pitch}"
        )
    return pitch


def validate_palm_roll_deg(value: float) -> float:
    roll = float(value)
    if not math.isfinite(roll):
        raise ValueError("palm roll must be finite")
    return roll


def _rotation_y(angle_rad: float) -> np.ndarray:
    cosine = math.cos(angle_rad)
    sine = math.sin(angle_rad)
    return np.array(
        [
            [cosine, 0.0, sine],
            [0.0, 1.0, 0.0],
            [-sine, 0.0, cosine],
        ],
        dtype=np.float64,
    )


def _rotation_z(angle_rad: float) -> np.ndarray:
    cosine = math.cos(angle_rad)
    sine = math.sin(angle_rad)
    return np.array(
        [
            [cosine, -sine, 0.0],
            [sine, cosine, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def compute_pitched_hand_from_tool_rotation(
    base_hand_from_tool: np.ndarray,
    downward_pitch_deg: float,
    palm_roll_deg: float = 0.0,
) -> np.ndarray:
    """
    Build hand_T_tool for a downward-pitched hand and a level plug frame.

    The pitch is applied about tool-local +Y. The palm roll is applied about
    panda_hand local +Z. A 180-degree palm roll reproduces the previously
    working Franka presentation while leaving the hand forward axis and the
    horizontal plug insertion axis unchanged.
    """

    base = _rotation3(base_hand_from_tool, label="base_hand_from_tool")
    pitch_rad = math.radians(
        validate_downward_hand_pitch_deg(downward_pitch_deg)
    )
    roll_rad = math.radians(validate_palm_roll_deg(palm_roll_deg))
    return _rotation3(
        _rotation_z(roll_rad) @ base @ _rotation_y(pitch_rad),
        label="pitched_hand_from_tool",
    )


def horizontal_axis_error_deg(axis_world: np.ndarray) -> float:
    axis = _axis3(axis_world, label="axis_world")
    return math.degrees(
        math.asin(float(np.clip(abs(axis[2]), 0.0, 1.0)))
    )


def _directional_axis_error_deg(
    actual_axis_world: np.ndarray,
    expected_axis_world: np.ndarray,
) -> float:
    actual = _axis3(actual_axis_world, label="actual_axis_world")
    expected = _axis3(expected_axis_world, label="expected_axis_world")
    dot = float(np.clip(np.dot(actual, expected), -1.0, 1.0))
    return math.degrees(math.acos(dot))


def expected_palm_side_axis_world(plug_axis_world: np.ndarray) -> np.ndarray:
    """Return the old working palm-side direction for a horizontal plug."""

    plug_axis = _axis3(plug_axis_world, label="plug_axis_world")
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    horizontal_plug = plug_axis - float(np.dot(plug_axis, world_up)) * world_up
    horizontal_norm = float(np.linalg.norm(horizontal_plug))
    if horizontal_norm <= _EPS:
        raise ValueError("plug axis cannot be vertical when checking palm roll")
    horizontal_plug /= horizontal_norm
    return _axis3(
        np.cross(world_up, horizontal_plug),
        label="expected_palm_side_axis_world",
    )


@dataclass(frozen=True)
class HandPlugGeometryMetrics:
    relative_pitch_deg: float
    wrist_above_tip_m: float
    plug_horizontal_error_deg: float
    palm_roll_error_deg: float
    wrist_higher_fingertips_lower: bool


def measure_hand_plug_geometry(
    *,
    hand_position_m: np.ndarray,
    hand_rotation_world: np.ndarray,
    plug_tip_position_m: np.ndarray,
    plug_axis_world: np.ndarray,
) -> HandPlugGeometryMetrics:
    hand_position = np.asarray(hand_position_m, dtype=np.float64).reshape(-1)
    tip_position = np.asarray(plug_tip_position_m, dtype=np.float64).reshape(-1)
    if hand_position.shape != (3,) or not np.all(np.isfinite(hand_position)):
        raise ValueError("hand_position_m must be a finite length-3 vector")
    if tip_position.shape != (3,) or not np.all(np.isfinite(tip_position)):
        raise ValueError("plug_tip_position_m must be a finite length-3 vector")

    hand_rotation = _rotation3(
        hand_rotation_world,
        label="hand_rotation_world",
    )
    plug_axis = _axis3(plug_axis_world, label="plug_axis_world")
    hand_forward = _axis3(
        hand_rotation[:, 2],
        label="hand_forward_axis",
    )
    hand_side = _axis3(
        hand_rotation[:, 0],
        label="hand_side_axis",
    )

    dot = float(np.clip(np.dot(hand_forward, plug_axis), -1.0, 1.0))
    relative_pitch_deg = math.degrees(math.acos(dot))
    wrist_above_tip_m = float(hand_position[2] - tip_position[2])
    direction_ok = (
        wrist_above_tip_m > 0.0
        and hand_forward[2] < plug_axis[2]
    )
    palm_roll_error_deg = _directional_axis_error_deg(
        hand_side,
        expected_palm_side_axis_world(plug_axis),
    )

    return HandPlugGeometryMetrics(
        relative_pitch_deg=relative_pitch_deg,
        wrist_above_tip_m=wrist_above_tip_m,
        plug_horizontal_error_deg=horizontal_axis_error_deg(plug_axis),
        palm_roll_error_deg=palm_roll_error_deg,
        wrist_higher_fingertips_lower=direction_ok,
    )
