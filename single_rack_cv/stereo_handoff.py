#!/usr/bin/env python3
"""Pure helpers for a bounded stereo-to-kinematic handoff."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np


_EPS = 1.0e-12


def _vector3(value, *, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)

    if array.shape != (3,):
        raise ValueError(
            f"{label} must have shape (3,), got {array.shape}."
        )

    if not np.all(np.isfinite(array)):
        raise ValueError(
            f"{label} must contain only finite values."
        )

    return array


@dataclass(frozen=True)
class StableGoalEstimate:
    """Robust world-space goal inferred from recent stereo observations."""

    position_m: np.ndarray
    spread_m: float
    sample_count: int


@dataclass(frozen=True)
class HandoffDecision:
    """A stable stereo-derived goal that lies inside the bounded finish region."""

    estimate: StableGoalEstimate
    remaining_m: float


def estimate_stable_goal(
    goal_candidates_m: Iterable[object],
    *,
    minimum_samples: int,
    maximum_spread_m: float,
) -> StableGoalEstimate | None:
    """Return the median goal only when all supplied candidates agree."""

    if minimum_samples < 1:
        raise ValueError(
            "minimum_samples must be positive."
        )

    if (
        not math.isfinite(maximum_spread_m)
        or maximum_spread_m <= 0.0
    ):
        raise ValueError(
            "maximum_spread_m must be finite and positive."
        )

    candidates = [
        _vector3(value, label="goal_candidate")
        for value in goal_candidates_m
    ]

    if len(candidates) < minimum_samples:
        return None

    stacked = np.vstack(candidates)
    median = np.median(stacked, axis=0)
    spread_m = float(
        np.max(
            np.linalg.norm(
                stacked - median,
                axis=1,
            )
        )
    )

    if spread_m > maximum_spread_m:
        return None

    return StableGoalEstimate(
        position_m=median,
        spread_m=spread_m,
        sample_count=len(candidates),
    )


def select_recent_bounded_goal(
    goal_candidates_m: Iterable[object],
    current_target_position_m,
    *,
    minimum_samples: int,
    recent_sample_count: int,
    maximum_spread_m: float,
    maximum_distance_m: float,
) -> HandoffDecision | None:
    """
    Select a handoff goal from only the newest observations.

    Older observations are deliberately excluded because the apparent RJ45
    geometry changes as the angled wrist approaches the rack. The handoff is
    allowed only when the newest observations agree and the remaining target
    translation lies inside the bounded finish region.
    """

    if recent_sample_count < minimum_samples:
        raise ValueError(
            "recent_sample_count must be at least minimum_samples."
        )

    if (
        not math.isfinite(maximum_distance_m)
        or maximum_distance_m <= 0.0
    ):
        raise ValueError(
            "maximum_distance_m must be finite and positive."
        )

    candidates = list(goal_candidates_m)
    recent = candidates[-recent_sample_count:]

    estimate = estimate_stable_goal(
        recent,
        minimum_samples=minimum_samples,
        maximum_spread_m=maximum_spread_m,
    )

    if estimate is None:
        return None

    current_target = _vector3(
        current_target_position_m,
        label="current_target_position_m",
    )
    remaining_m = float(
        np.linalg.norm(
            estimate.position_m - current_target
        )
    )

    if remaining_m > maximum_distance_m:
        return None

    return HandoffDecision(
        estimate=estimate,
        remaining_m=remaining_m,
    )


def bounded_step_to_goal(
    current_position_m,
    goal_position_m,
    *,
    maximum_step_m: float,
) -> tuple[np.ndarray, float]:
    """Return one bounded world step and pre-step remaining distance."""

    if (
        not math.isfinite(maximum_step_m)
        or maximum_step_m <= 0.0
    ):
        raise ValueError(
            "maximum_step_m must be finite and positive."
        )

    current = _vector3(
        current_position_m,
        label="current_position_m",
    )
    goal = _vector3(
        goal_position_m,
        label="goal_position_m",
    )

    delta = goal - current
    distance_m = float(np.linalg.norm(delta))

    if distance_m <= _EPS:
        return (
            np.zeros(3, dtype=np.float64),
            0.0,
        )

    if distance_m <= maximum_step_m:
        return delta, distance_m

    return (
        delta * (maximum_step_m / distance_m),
        distance_m,
    )
