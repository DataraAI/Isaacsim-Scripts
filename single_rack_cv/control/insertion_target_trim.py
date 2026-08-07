#!/usr/bin/env python3
"""Insertion-only calibrated world-space centerline for guarded commands."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from control.settled_insertion import ConsecutivePoseInsertionController


_MAXIMUM_INSERTION_CALIBRATION_M = 0.001


class TrimmedConsecutivePoseInsertionController(
    ConsecutivePoseInsertionController
):
    """Shift every insertion command and measure drift from that shifted line."""

    def __init__(
        self,
        limits,
        *,
        target_offset_world_m,
        **kwargs,
    ):
        super().__init__(limits, **kwargs)

        offset = np.asarray(
            target_offset_world_m,
            dtype=np.float64,
        ).reshape(-1)
        if offset.shape != (3,):
            raise ValueError(
                "target_offset_world_m must have shape (3,), "
                f"got {offset.shape}"
            )
        if not np.all(np.isfinite(offset)):
            raise ValueError(
                "target_offset_world_m must contain only finite values"
            )

        offset_magnitude_m = float(np.linalg.norm(offset))
        if offset_magnitude_m > _MAXIMUM_INSERTION_CALIBRATION_M:
            raise ValueError(
                "Insertion target calibration exceeds the 1.0 mm hard cap: "
                f"calibration={offset_magnitude_m * 1000.0:.6f} mm"
            )

        self.target_offset_world_m = offset.copy()

    def _calibrated_lateral_drift_m(self, sample) -> float:
        if (
            self.frozen_start_position_m is None
            or self.axis_world is None
        ):
            raise RuntimeError("Insertion frame has not been frozen")

        calibrated_origin = (
            self.frozen_start_position_m
            + self.target_offset_world_m
        )
        actual_position_m = np.asarray(
            sample.actual_position_m,
            dtype=np.float64,
        ).reshape(3)
        relative_position_m = actual_position_m - calibrated_origin
        calibrated_axial_m = float(
            np.dot(relative_position_m, self.axis_world)
        )
        calibrated_lateral_vector_m = (
            relative_position_m
            - calibrated_axial_m * self.axis_world
        )
        return float(np.linalg.norm(calibrated_lateral_vector_m))

    def _metrics(self, sample):
        metrics = super()._metrics(sample)
        if metrics is None:
            return None
        return replace(
            metrics,
            lateral_drift_m=self._calibrated_lateral_drift_m(sample),
        )

    def _issue_next_command(self, frame_index: int):
        command = super()._issue_next_command(frame_index)
        trimmed_command = replace(
            command,
            target_position_m=(
                command.target_position_m
                + self.target_offset_world_m
            ),
        )
        self.last_command = trimmed_command
        return trimmed_command


__all__ = ["TrimmedConsecutivePoseInsertionController"]
