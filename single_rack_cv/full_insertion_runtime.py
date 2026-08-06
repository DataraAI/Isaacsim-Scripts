#!/usr/bin/env python3
"""Production full insertion with a calibrated insertion-only centerline."""

from __future__ import annotations

import numpy as np

from full_insertion_base_runtime import (
    AngledHandStereoHandoffRuntime as _BaseAngledHandStereoHandoffRuntime,
)
from insertion_target_trim import TrimmedConsecutivePoseInsertionController
from plug_axis_insertion import ExplicitInsertionAxisAdapter
from sim import log


_INSERTION_TARGET_OFFSET_WORLD_M = np.array(
    [0.0, -0.00030, -0.00045],
    dtype=np.float64,
)


class AngledHandStereoHandoffRuntime(
    _BaseAngledHandStereoHandoffRuntime
):
    """Calibrate only the guarded insertion line; preserve perception/handoff."""

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

        calibration_magnitude_m = float(
            np.linalg.norm(_INSERTION_TARGET_OFFSET_WORLD_M)
        )
        log(
            "INSERTION TARGET CALIBRATION ACTIVE\n"
            "  perception-derived port point: unchanged\n"
            "  50 mm handoff goal: unchanged\n"
            "  insertion depth schedule: unchanged at 48 commands\n"
            "  insertion target line world Y: -0.300 mm\n"
            "  insertion target line world Z: -0.450 mm\n"
            f"  calibration magnitude: "
            f"{calibration_magnitude_m * 1000.0:.3f} mm\n"
            "  lateral drift reference: calibrated insertion line\n"
            f"  lateral deviation abort limit: "
            f"{limits.max_lateral_drift_m * 1000.0:.3f} mm\n"
            "  calibration is not counted as lateral deviation"
        )


__all__ = ["AngledHandStereoHandoffRuntime"]
