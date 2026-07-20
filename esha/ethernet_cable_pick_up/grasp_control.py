"""Pure helpers for selecting grasp-control commands."""

from __future__ import annotations

import numpy as np


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
