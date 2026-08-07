#!/usr/bin/env python3
"""Preserved base runtime for the full guarded RJ45 insertion sequence."""

from __future__ import annotations

import math

from cable import connector_tcp_usd as _connector_tcp_usd

_connector_tcp_usd.PRECONTACT_ALIGNMENT_ONLY = False

from cable import scale_aware_cable_mount as _scale_aware_cable_mount

_scale_aware_cable_mount.PRECONTACT_ALIGNMENT_ONLY = False

from runtime.settled_stereo_handoff_runtime import (
    AngledHandStereoHandoffRuntime as _BaseAngledHandStereoHandoffRuntime,
)
from sim import log


_EXPECTED_TOTAL_COMMANDS = 48
_EXPECTED_FINAL_PORT_DEPTH_M = 0.010
_EPS = 1.0e-12


class AngledHandStereoHandoffRuntime(
    _BaseAngledHandStereoHandoffRuntime
):
    """Use the derived connector TCP for the complete +10 mm insertion."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        mount = self.cable_mount
        if mount is not None and getattr(mount, "tcp_probe_only", False):
            return

        if mount is None:
            raise RuntimeError(
                "Full insertion requires the scale-aware cable mount."
            )
        if getattr(mount, "precontact_alignment_only", False):
            raise RuntimeError(
                "Full insertion runtime inherited the temporary precontact cap."
            )

        limits = self.partial_insertion.limits
        final_port_depth_m = (
            float(limits.total_depth_m) - float(limits.opening_depth_m)
        )
        total_commands = int(limits.total_step_count)

        if total_commands != _EXPECTED_TOTAL_COMMANDS:
            raise RuntimeError(
                "Full insertion command count changed: "
                f"expected {_EXPECTED_TOTAL_COMMANDS}, got {total_commands}."
            )
        if not math.isclose(
            final_port_depth_m,
            _EXPECTED_FINAL_PORT_DEPTH_M,
            rel_tol=0.0,
            abs_tol=_EPS,
        ):
            raise RuntimeError(
                "Full insertion terminal depth changed: "
                f"expected +{_EXPECTED_FINAL_PORT_DEPTH_M * 1000.0:.3f} mm, "
                f"got {final_port_depth_m * 1000.0:+.3f} mm."
            )

        fine_distance_m = (
            float(limits.total_depth_m)
            - float(limits.coarse_approach_depth_m)
        )
        log(
            "FULL GUARDED INSERTION MODE ACTIVE\n"
            "  mesh-derived connector TCP: required\n"
            f"  coarse approach: "
            f"{limits.coarse_approach_depth_m * 1000.0:.3f} mm "
            f"at {limits.coarse_step_size_m * 1000.0:.3f} mm/step\n"
            f"  fine motion: {fine_distance_m * 1000.0:.3f} mm "
            f"at {limits.step_size_m * 1000.0:.3f} mm/step\n"
            f"  final depth inside opening: "
            f"+{final_port_depth_m * 1000.0:.3f} mm\n"
            f"  total commands: {total_commands}\n"
            f"  lateral abort limit: "
            f"{limits.max_lateral_drift_m * 1000.0:.3f} mm\n"
            f"  orientation abort limit: "
            f"{limits.max_orientation_error_deg:.3f} deg\n"
            "  mount, attachment, IK, settle, and timeout gates: unchanged"
        )


__all__ = ["AngledHandStereoHandoffRuntime"]
