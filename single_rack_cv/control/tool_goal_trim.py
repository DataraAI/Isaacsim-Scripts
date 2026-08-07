#!/usr/bin/env python3
"""Pure helpers for explicit final ToolCenter goal trimming."""

from __future__ import annotations

import math

import numpy as np


def apply_tool_goal_trim(
    tool_goal_position_m,
    *,
    left_trim_m: float,
    downward_trim_m: float,
) -> np.ndarray:
    """Return a copied ToolCenter goal shifted left in view and downward.

    For the current rack-facing camera convention, image-left maps to world
    negative Y. Down maps to world negative Z. The measured port point is not
    passed to or changed by this helper.
    """

    goal = np.asarray(tool_goal_position_m, dtype=np.float64)
    if goal.shape != (3,) or not np.all(np.isfinite(goal)):
        raise ValueError(
            "tool_goal_position_m must be a finite vector with shape (3,)."
        )

    left_trim_m = float(left_trim_m)
    if not math.isfinite(left_trim_m) or left_trim_m < 0.0:
        raise ValueError("left_trim_m must be finite and nonnegative.")

    downward_trim_m = float(downward_trim_m)
    if not math.isfinite(downward_trim_m) or downward_trim_m < 0.0:
        raise ValueError("downward_trim_m must be finite and nonnegative.")

    trimmed = goal.copy()
    trimmed[1] -= left_trim_m
    trimmed[2] -= downward_trim_m
    return trimmed
