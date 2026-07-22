"""Pure helpers for selecting grasp-control commands."""

from __future__ import annotations

import numpy as np


def apply_grasp_x_offset(
    estimated_cable_point_world_m: np.ndarray,
    *,
    x_offset_m: float,
) -> np.ndarray:
    """Shift only grasp X while preserving the estimated cable Y and Z."""
    grasp_point = np.asarray(
        estimated_cable_point_world_m,
        dtype=np.float64,
    ).reshape(3).copy()
    grasp_point[0] += float(x_offset_m)
    return grasp_point


def select_open_half_gap(
    *,
    cable_half_width_m: float,
    side_allowance_m: float,
    minimum_half_gap_m: float,
    maximum_half_gap_m: float,
) -> float:
    """Choose a measured opening with a safe minimum and hardware maximum."""
    requested = max(
        float(cable_half_width_m) + float(side_allowance_m),
        float(minimum_half_gap_m),
    )
    return min(requested, float(maximum_half_gap_m))


def bounded_linear_step(
    current_position_m: np.ndarray,
    target_position_m: np.ndarray,
    *,
    max_step_m: float,
) -> np.ndarray:
    """Advance toward a target along a straight line by at most max_step_m."""
    current = np.asarray(
        current_position_m,
        dtype=np.float64,
    ).reshape(3)
    target = np.asarray(
        target_position_m,
        dtype=np.float64,
    ).reshape(3)
    delta = target - current
    distance = float(np.linalg.norm(delta))
    if distance <= float(max_step_m):
        return target.copy()
    if distance <= 1.0e-12:
        return current.copy()
    return current + (float(max_step_m) / distance) * delta


def finger_target_reached(
    finger_positions_m: np.ndarray,
    *,
    target_position_m: float,
    tolerance_m: float,
) -> bool:
    """Return whether both finger joints reached the commanded position."""
    positions = np.asarray(finger_positions_m, dtype=np.float64).reshape(-1)
    return bool(
        positions.size == 2
        and np.all(
            np.abs(positions - float(target_position_m))
            <= float(tolerance_m)
        )
    )


def fingers_moved_toward_closed(
    finger_positions_m: np.ndarray,
    open_positions_m: np.ndarray,
    *,
    minimum_travel_m: float,
) -> bool:
    """Require both fingers to have moved inward before accepting contact."""
    positions = np.abs(
        np.asarray(finger_positions_m, dtype=np.float64).reshape(-1)
    )
    open_positions = np.abs(
        np.asarray(open_positions_m, dtype=np.float64).reshape(-1)
    )
    return bool(
        positions.size == 2
        and open_positions.size == 2
        and np.all(
            open_positions - positions >= float(minimum_travel_m)
        )
    )


def clearance_target_position(
    cable_point_world_m: np.ndarray,
    approach_direction: np.ndarray,
    *,
    clearance_m: float,
    minimum_z_m: float,
) -> np.ndarray:
    """Place ToolCenter on the configured approach line with a fixed Z floor."""
    cable_point = np.asarray(
        cable_point_world_m,
        dtype=np.float64,
    ).reshape(3)
    approach = np.asarray(
        approach_direction,
        dtype=np.float64,
    ).reshape(3)
    norm = float(np.linalg.norm(approach))
    if norm <= 1.0e-12:
        raise ValueError("Approach direction cannot have zero length.")
    target = cable_point - float(clearance_m) * (approach / norm)
    target[2] = max(float(target[2]), float(minimum_z_m))
    return target


def grasp_orientation_active(phase: str) -> bool:
    """Return whether IK should enforce the stored angled grasp orientation."""
    return phase != "idle"


def resolve_tool_orientation(
    marker_orientation_wxyz: np.ndarray,
    grasp_orientation_wxyz: np.ndarray | None,
    *,
    grasp_active: bool,
) -> np.ndarray:
    """
    Return the exact stored grasp orientation while grasp control is active.

    The USD target marker remains the position source, but its transformed
    orientation can differ from the quaternion originally assigned to it.
    """
    selected = (
        grasp_orientation_wxyz
        if grasp_active and grasp_orientation_wxyz is not None
        else marker_orientation_wxyz
    )
    quaternion = np.asarray(selected, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1.0e-12:
        raise ValueError("Quaternion cannot have zero length.")
    return quaternion / norm
