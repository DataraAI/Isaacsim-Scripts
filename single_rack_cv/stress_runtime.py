#!/usr/bin/env python3
"""Stress-only passive instrumentation around the canonical Isaac runtime."""

from __future__ import annotations

import math

import numpy as np

from sim import SimulationRuntime
from stress_alignment import quaternion_angular_distance_deg


def _json_number_or_none(value: float) -> float | None:
    number = float(value)
    return number if math.isfinite(number) else None


class InstrumentedSimulationRuntime(SimulationRuntime):
    """Observe canonical runtime behavior without adding a control path."""

    def __init__(self, simulation_app, cfg):
        self._track_acquired_ever = False
        self._visual_alignment_locked_ever = False
        self._track_acquisition_count = 0
        self._perception_rejection_count = 0
        self._final_center_error_px = math.nan
        self._final_range_error_m = math.nan
        self._maximum_target_step_m = 0.0
        self._maximum_orientation_deviation_deg = 0.0
        super().__init__(simulation_app=simulation_app, cfg=cfg)
        if self.ik is None:
            raise RuntimeError("stress instrumentation requires initialized IK")
        _position, orientation = self.ik.target.get_world_pose()
        self._initial_target_orientation_wxyz = np.asarray(
            orientation,
            dtype=np.float64,
        ).reshape(4)

    def note_perception_failure(self) -> None:
        self._perception_rejection_count += 1
        super().note_perception_failure()

    def observe_visual_servo(self, observation) -> None:
        state = self.visual_servo
        acquired_before = bool(state.acquired)
        aligned_before = bool(state.visual_aligned)
        target_before = None
        if self.ik is not None:
            position, _orientation = self.ik.target.get_world_pose()
            target_before = np.asarray(position, dtype=np.float64).reshape(3)

        self._final_center_error_px = float(
            np.linalg.norm(observation.center_error_px)
        )
        self._final_range_error_m = float(observation.range_error_m)
        super().observe_visual_servo(observation)

        if not acquired_before and state.acquired:
            self._track_acquisition_count += 1
            self._track_acquired_ever = True
        if not aligned_before and state.visual_aligned:
            self._visual_alignment_locked_ever = True

        if self.ik is not None and target_before is not None:
            target_after, _orientation = self.ik.target.get_world_pose()
            step_m = float(
                np.linalg.norm(
                    np.asarray(target_after, dtype=np.float64).reshape(3)
                    - target_before
                )
            )
            self._maximum_target_step_m = max(
                self._maximum_target_step_m,
                step_m,
            )

    def update_ik(self) -> None:
        super().update_ik()
        if self.ik is None or not self.visual_servo.startup_ready:
            return
        self._update_actual_tool_frame(self.ik)
        _position, actual_orientation = self.ik.actual_tool.get_world_pose()
        deviation_deg = quaternion_angular_distance_deg(
            self._initial_target_orientation_wxyz,
            actual_orientation,
        )
        self._maximum_orientation_deviation_deg = max(
            self._maximum_orientation_deviation_deg,
            deviation_deg,
        )

    def stress_snapshot(self) -> dict[str, object]:
        if self.ik is None:
            raise RuntimeError("stress snapshot requires initialized IK")
        self._update_actual_tool_frame(self.ik)
        target_position, _target_orientation = self.ik.target.get_world_pose()
        actual_position, _actual_orientation = self.ik.actual_tool.get_world_pose()
        return {
            "completed": bool(self.visual_servo.complete),
            "track_acquired": bool(self._track_acquired_ever),
            "visual_alignment_locked": bool(
                self._visual_alignment_locked_ever
            ),
            "final_center_error_px": _json_number_or_none(
                self._final_center_error_px
            ),
            "final_range_error_mm": _json_number_or_none(
                1000.0 * self._final_range_error_m
            ),
            "final_tool_target_world_m": [
                float(value) for value in target_position
            ],
            "final_actual_tool_world_m": [
                float(value) for value in actual_position
            ],
            "final_physical_tracking_error_mm": _json_number_or_none(
                1000.0 * self._tool_target_position_error_m()
            ),
            "maximum_target_step_mm": 1000.0 * self._maximum_target_step_m,
            "maximum_orientation_deviation_deg": (
                self._maximum_orientation_deviation_deg
            ),
            "perception_rejection_count": self._perception_rejection_count,
            "track_reacquisition_count": max(
                0,
                self._track_acquisition_count - 1,
            ),
            "insertion_command_count": 0,
        }
