#!/usr/bin/env python3
"""Pure policy for stopping the guarded cable approach before contact."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math

from insertion import InsertionLimits


@dataclass(frozen=True)
class PrecontactAlignmentPolicy:
    """Cap frozen-axis motion at a fixed distance before the opening plane."""

    hold_offset_m: float = 0.002

    def __post_init__(self) -> None:
        value = float(self.hold_offset_m)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("hold_offset_m must be finite and positive")
        object.__setattr__(self, "hold_offset_m", value)


def build_precontact_limits(
    base_limits: InsertionLimits,
    policy: PrecontactAlignmentPolicy,
) -> InsertionLimits:
    """Return limits whose terminal command remains before the opening plane.

    Every existing safety threshold, opening depth, stage boundary, and step
    size is preserved. Only total travel is shortened so the final commanded
    port depth is exactly ``-policy.hold_offset_m``.
    """

    if not isinstance(base_limits, InsertionLimits):
        raise TypeError("base_limits must be an InsertionLimits instance")
    if not isinstance(policy, PrecontactAlignmentPolicy):
        raise TypeError("policy must be a PrecontactAlignmentPolicy instance")

    opening_depth_m = float(base_limits.opening_depth_m)
    if opening_depth_m <= 0.0:
        raise ValueError("precontact alignment requires a positive opening depth")
    if policy.hold_offset_m >= opening_depth_m:
        raise ValueError("hold_offset_m must be smaller than opening_depth_m")

    capped_total_depth_m = opening_depth_m - policy.hold_offset_m
    if capped_total_depth_m <= float(base_limits.coarse_approach_depth_m):
        raise ValueError(
            "precontact total depth must extend beyond the coarse approach stage"
        )
    if capped_total_depth_m >= opening_depth_m:
        raise RuntimeError("precontact policy produced a penetrating command range")

    return replace(
        base_limits,
        total_depth_m=capped_total_depth_m,
    )
