#!/usr/bin/env python3
"""Pure policy for stopping the guarded cable approach before contact."""

from __future__ import annotations

from dataclasses import dataclass
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


class PrecontactInsertionLimits(InsertionLimits):
    """Insertion-compatible limits whose terminal target precedes the opening."""

    def __post_init__(self) -> None:
        positive_float_fields = (
            "total_depth_m",
            "step_size_m",
            "settle_tolerance_m",
            "max_lateral_drift_m",
            "max_orientation_error_deg",
            "max_mount_tip_error_m",
            "max_mount_axis_error_deg",
        )
        for name in positive_float_fields:
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")

        coarse_depth = float(self.coarse_approach_depth_m)
        coarse_step = float(self.coarse_step_size_m)
        opening_depth = float(self.opening_depth_m)
        for name, value in (
            ("coarse_approach_depth_m", coarse_depth),
            ("coarse_step_size_m", coarse_step),
            ("opening_depth_m", opening_depth),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")

        if (coarse_depth <= 1.0e-12) != (coarse_step <= 1.0e-12):
            raise ValueError(
                "coarse_approach_depth_m and coarse_step_size_m must both be zero or positive"
            )
        if coarse_depth > self.total_depth_m + 1.0e-12:
            raise ValueError("coarse_approach_depth_m cannot exceed total_depth_m")
        if coarse_step > coarse_depth + 1.0e-12 and coarse_depth > 1.0e-12:
            raise ValueError("coarse_step_size_m cannot exceed coarse_approach_depth_m")
        if self.step_size_m > self.total_depth_m:
            raise ValueError("step_size_m cannot exceed total_depth_m")
        if opening_depth <= self.total_depth_m + 1.0e-12:
            raise ValueError(
                "precontact opening_depth_m must exceed terminal total_depth_m"
            )
        if self.required_settled_frames <= 0:
            raise ValueError("required_settled_frames must be positive")
        if self.step_timeout_frames <= 0:
            raise ValueError("step_timeout_frames must be positive")


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

    return PrecontactInsertionLimits(
        total_depth_m=capped_total_depth_m,
        step_size_m=float(base_limits.step_size_m),
        settle_tolerance_m=float(base_limits.settle_tolerance_m),
        required_settled_frames=int(base_limits.required_settled_frames),
        step_timeout_frames=int(base_limits.step_timeout_frames),
        max_lateral_drift_m=float(base_limits.max_lateral_drift_m),
        max_orientation_error_deg=float(
            base_limits.max_orientation_error_deg
        ),
        max_mount_tip_error_m=float(base_limits.max_mount_tip_error_m),
        max_mount_axis_error_deg=float(base_limits.max_mount_axis_error_deg),
        coarse_approach_depth_m=float(
            base_limits.coarse_approach_depth_m
        ),
        coarse_step_size_m=float(base_limits.coarse_step_size_m),
        opening_depth_m=opening_depth_m,
    )
