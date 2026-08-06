#!/usr/bin/env python3
"""Production full insertion with an insertion-only target-line trim."""

from __future__ import annotations

import numpy as np

from full_insertion_base_runtime import (
    AngledHandStereoHandoffRuntime as _BaseAngledHandStereoHandoffRuntime,
)
from insertion_target_trim import TrimmedConsecutivePoseInsertionController
from plug_axis_insertion import ExplicitInsertionAxisAdapter
from sim import log


_INSERTION_TARGET_OFFSET_WORLD_M = np.array(
    [0.0, -0.00015, -0.00025],
    dtype=np.float64,
)


class AngledHandStereoHandoffRuntime(
    _BaseAngledHandStereoHandoffRuntime
):
    """Move only the guarded insertion line; preserve perception and handoff."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        mount = self.cable_mount
        if mount is not None and getattr(mount, "tcp_probe_only", False):
            return

        limits = self.partial_insertion.limits
        self.partial_insertion = TrimmedConsecutivePoseInsertionController(
            limits,
            target_offset_world_m=_INSERTION_TARGET_OFFSET_WORLD_M,
        )
        self._insertion_axis_adapter = ExplicitInsertionAxisAdapter(
            self.partial_insertion
        )

        trim_magnitude_m = float(
            np.linalg.norm(_INSERTION_TARGET_OFFSET_WORLD_M)
        )
        remaining_lateral_budget_m = (
            float(limits.max_lateral_drift_m) - trim_magnitude_m
        )
        log(
            "INSERTION TARGET TRIM ACTIVE\n"
            "  perception-derived port point: unchanged\n"
            "  50 mm handoff goal: unchanged\n"
            "  insertion depth schedule: unchanged at 48 commands\n"
            "  insertion target line world Y: -0.150 mm\n"
            "  insertion target line world Z: -0.250 mm\n"
            f"  trim magnitude: {trim_magnitude_m * 1000.0:.3f} mm\n"
            f"  lateral abort limit: "
            f"{limits.max_lateral_drift_m * 1000.0:.3f} mm\n"
            f"  conservative remaining lateral budget: "
            f"{remaining_lateral_budget_m * 1000.0:.3f} mm\n"
            "  trim remains visible to the existing drift guard"
        )


__all__ = ["AngledHandStereoHandoffRuntime"]
