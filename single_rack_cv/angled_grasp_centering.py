#!/usr/bin/env python3
"""Calibrate a pitched hand so the RJ45 rear sits between the fingers."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from hand_plug_geometry import validate_downward_hand_pitch_deg


@dataclass(frozen=True)
class RearCenteredGraspCalibration:
    """Equivalent hand/tool/camera calibration with a centered plug rear."""

    base_hand_position_world_m: np.ndarray
    tool_position_hand_m: np.ndarray
    camera_positions_hand_m: np.ndarray
    local_shift_hand_m: np.ndarray

    def __post_init__(self) -> None:
        for name in (
            "base_hand_position_world_m",
            "tool_position_hand_m",
            "local_shift_hand_m",
        ):
            value = np.asarray(getattr(self, name), dtype=np.float64).reshape(3)
            if not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value.copy())

        cameras = np.asarray(
            self.camera_positions_hand_m,
            dtype=np.float64,
        )
        if cameras.ndim != 2 or cameras.shape[1] != 3:
            raise ValueError("camera_positions_hand_m must have shape (N, 3)")
        if not np.all(np.isfinite(cameras)):
            raise ValueError("camera_positions_hand_m must be finite")
        object.__setattr__(
            self,
            "camera_positions_hand_m",
            cameras.copy(),
        )


def _vector3(value, *, label: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64).reshape(-1)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{label} must be a finite length-3 vector")
    return vector.copy()


def _rotation3(value, *, label: str) -> np.ndarray:
    rotation = np.asarray(value, dtype=np.float64)
    if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
        raise ValueError(f"{label} must be a finite 3x3 matrix")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-9):
        raise ValueError(f"{label} must be orthonormal")
    if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1.0e-9):
        raise ValueError(f"{label} must have determinant +1")
    return rotation.copy()


def recenter_horizontal_plug_rear_in_pitched_hand(
    *,
    base_hand_position_m,
    base_hand_rotation_world,
    tool_position_hand_m,
    camera_positions_hand_m,
    plug_body_length_m: float,
    downward_pitch_deg: float,
) -> RearCenteredGraspCalibration:
    """
    Move the pitched hand around an unchanged plug-tip and camera calibration.

    With a horizontal connector and a hand pitched downward by ``pitch``, the
    connector rear is displaced from the hand centerline by
    ``plug_length * sin(pitch)`` along hand-local -X. Shifting ToolCenter and
    every camera mount by the opposite local amount centers that rear section
    between the fingers. The base hand position is compensated so the original
    world ToolCenter and all original world camera positions remain exact.
    """

    base_hand_position = _vector3(
        base_hand_position_m,
        label="base_hand_position_m",
    )
    base_hand_rotation = _rotation3(
        base_hand_rotation_world,
        label="base_hand_rotation_world",
    )
    tool_position_hand = _vector3(
        tool_position_hand_m,
        label="tool_position_hand_m",
    )
    cameras = np.asarray(camera_positions_hand_m, dtype=np.float64)
    if cameras.ndim != 2 or cameras.shape[1] != 3:
        raise ValueError("camera_positions_hand_m must have shape (N, 3)")
    if not np.all(np.isfinite(cameras)):
        raise ValueError("camera_positions_hand_m must be finite")

    plug_length = float(plug_body_length_m)
    if not math.isfinite(plug_length) or plug_length <= 0.0:
        raise ValueError("plug_body_length_m must be finite and positive")
    pitch_deg = validate_downward_hand_pitch_deg(downward_pitch_deg)
    pitch_rad = math.radians(pitch_deg)

    local_shift = np.array(
        [plug_length * math.sin(pitch_rad), 0.0, 0.0],
        dtype=np.float64,
    )
    shifted_tool = tool_position_hand + local_shift
    shifted_cameras = cameras + local_shift[None, :]
    shifted_base_hand = (
        base_hand_position - base_hand_rotation @ local_shift
    )

    old_tool_world = (
        base_hand_position + base_hand_rotation @ tool_position_hand
    )
    new_tool_world = (
        shifted_base_hand + base_hand_rotation @ shifted_tool
    )
    if not np.allclose(new_tool_world, old_tool_world, atol=1.0e-12):
        raise RuntimeError("rear-centering changed the world ToolCenter")

    old_cameras_world = (
        base_hand_position[None, :]
        + (base_hand_rotation @ cameras.T).T
    )
    new_cameras_world = (
        shifted_base_hand[None, :]
        + (base_hand_rotation @ shifted_cameras.T).T
    )
    if not np.allclose(
        new_cameras_world,
        old_cameras_world,
        atol=1.0e-12,
    ):
        raise RuntimeError("rear-centering changed camera-to-tool calibration")

    return RearCenteredGraspCalibration(
        base_hand_position_world_m=shifted_base_hand,
        tool_position_hand_m=shifted_tool,
        camera_positions_hand_m=shifted_cameras,
        local_shift_hand_m=local_shift,
    )
