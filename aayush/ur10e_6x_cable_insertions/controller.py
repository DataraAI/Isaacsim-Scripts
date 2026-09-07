"""Unit-aware strict controller used only by the six-arm demo."""

from __future__ import annotations

import numpy as np

from franka_motion_controller import FrankaMotionController
from ur10e_6x_cable_insertions.runtime_support import (
    meters_to_stage,
    stage_to_meters,
    validate_meters_per_unit,
)


class SixArmMotionController(FrankaMotionController):
    def __init__(self, *args, meters_per_unit: float, **kwargs):
        self._meters_per_unit = validate_meters_per_unit(meters_per_unit)
        self._six_arm_failure_reason = ""
        super().__init__(*args, **kwargs)

    def clear_queue(self) -> None:
        super().clear_queue()
        self._six_arm_failure_reason = ""
        self._last_commanded_joint_positions = None

    def add_cartesian_waypoint(self, position, orientation, **kwargs):
        kwargs = dict(kwargs)
        if kwargs.get("linear") and "linear_step" in kwargs:
            kwargs["linear_step"] = float(kwargs["linear_step"]) / self._meters_per_unit
        return super().add_cartesian_waypoint(
            meters_to_stage(position, self._meters_per_unit),
            orientation,
            **kwargs,
        )

    def current_hand_pose_meters(self):
        position, orientation = super()._current_hand_pose()
        return stage_to_meters(position, self._meters_per_unit), orientation

    def forward(self, current_joint_positions):
        action = super().forward(current_joint_positions)
        commanded = None if action is None else getattr(action, "joint_positions", None)
        setter = getattr(getattr(self, "_robot", None), "set_joint_positions", None)
        if commanded is not None and callable(setter):
            current = np.asarray(current_joint_positions, dtype=np.float64)
            previous = getattr(self, "_last_commanded_joint_positions", None)
            positions = (
                current.copy()
                if previous is None or np.asarray(previous).shape != current.shape
                else np.asarray(previous, dtype=np.float64).copy()
            )
            for index, value in enumerate(list(commanded)[: positions.size]):
                if value is not None:
                    positions[index] = float(value)
            setter(positions)
            self._last_commanded_joint_positions = positions
        return action

    def _init_joint_interp_segment(self, cmd, current_joint_positions, n_dof):
        super()._init_joint_interp_segment(cmd, current_joint_positions, n_dof)
        if self._joint_interp_warned:
            self._segment_failed = True
            label = str(cmd.get("label", "unlabelled waypoint"))
            self._six_arm_failure_reason = f"IK failed for {label}"

    def has_failed(self) -> bool:
        return bool(self._segment_failed)

    def failure_reason(self) -> str:
        return self._six_arm_failure_reason
