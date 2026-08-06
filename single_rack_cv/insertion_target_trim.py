#!/usr/bin/env python3
"""Insertion-only world-space target trim for the guarded command sequence."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from settled_insertion import ConsecutivePoseInsertionController


class TrimmedConsecutivePoseInsertionController(
    ConsecutivePoseInsertionController
):
    """Shift every insertion command by one fixed world-space vector."""

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
        if offset_magnitude_m >= float(limits.max_lateral_drift_m):
            raise ValueError(
                "Insertion target trim must remain below the existing lateral "
                "drift limit: "
                f"trim={offset_magnitude_m * 1000.0:.6f} mm, "
                f"limit={limits.max_lateral_drift_m * 1000.0:.6f} mm"
            )

        self.target_offset_world_m = offset.copy()

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
