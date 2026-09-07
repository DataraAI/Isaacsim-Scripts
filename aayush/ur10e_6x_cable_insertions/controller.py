"""Unit-aware strict controller used only by the six-arm demo."""

from __future__ import annotations

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

    def add_cartesian_waypoint(self, position, orientation, **kwargs):
        return super().add_cartesian_waypoint(
            meters_to_stage(position, self._meters_per_unit),
            orientation,
            **kwargs,
        )

    def current_hand_pose_meters(self):
        position, orientation = super()._current_hand_pose()
        return stage_to_meters(position, self._meters_per_unit), orientation

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
