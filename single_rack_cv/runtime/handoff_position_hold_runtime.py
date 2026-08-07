#!/usr/bin/env python3
"""Full insertion runtime with bounded final handoff position compensation."""

from __future__ import annotations

import numpy as np

from runtime.full_insertion_base_runtime import (
    AngledHandStereoHandoffRuntime as _BaseAngledHandStereoHandoffRuntime,
)
from control.handoff_position_hold import update_handoff_position_command
from sim import log, update_convergence_counter, warn


class AngledHandStereoHandoffRuntime(
    _BaseAngledHandStereoHandoffRuntime
):
    """Remove static ToolCenter tracking bias without moving the frozen goal."""

    _HANDOFF_POSITION_HOLD_GAIN = 0.35
    _HANDOFF_POSITION_HOLD_MAXIMUM_STEP_M = 0.00010
    _HANDOFF_POSITION_HOLD_MAXIMUM_BIAS_M = 0.00100
    _HANDOFF_POSITION_HOLD_LOG_INTERVAL_FRAMES = 30
    _HANDOFF_POSITION_HOLD_HARD_TIMEOUT_S = 10.0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_handoff_position_hold_log_frame = -1_000_000

        log(
            "FROZEN HANDOFF POSITION HOLD ACTIVE\n"
            "  frozen physical ToolCenter goal: unchanged\n"
            "  compensation target: IK command only\n"
            f"  compensation gain: {self._HANDOFF_POSITION_HOLD_GAIN:.3f}\n"
            f"  maximum command step: "
            f"{self._HANDOFF_POSITION_HOLD_MAXIMUM_STEP_M * 1000.0:.3f} mm/frame\n"
            f"  maximum total command bias: "
            f"{self._HANDOFF_POSITION_HOLD_MAXIMUM_BIAS_M * 1000.0:.3f} mm\n"
            "  physical completion tolerance: unchanged at 0.300 mm\n"
            f"  hard fail-closed timeout: "
            f"{self._HANDOFF_POSITION_HOLD_HARD_TIMEOUT_S:.1f} s"
        )

    def update_visual_servo_completion(self) -> None:
        """Drive the actual ToolCenter onto the frozen goal before insertion."""

        if not self._handoff_active:
            super().update_visual_servo_completion()
            return

        state = self.visual_servo
        cfg = self.cfg.visual_servo

        if (
            not state.visual_aligned
            or state.complete
            or self.ik is None
        ):
            return

        goal = self._handoff_goal_world_m
        if goal is None:
            raise RuntimeError(
                "Frozen handoff position hold requires a qualified ToolCenter goal."
            )
        goal = np.asarray(goal, dtype=np.float64)

        self._update_actual_tool_frame(self.ik)
        command_position, command_orientation = self.ik.target.get_world_pose()
        actual_position, _ = self.ik.actual_tool.get_world_pose()
        command_position = np.asarray(command_position, dtype=np.float64)
        actual_position = np.asarray(actual_position, dtype=np.float64)

        physical_error_world_m = goal - actual_position
        position_error_m = float(np.linalg.norm(physical_error_world_m))
        update = update_handoff_position_command(
            goal_position_m=goal,
            actual_position_m=actual_position,
            current_command_position_m=command_position,
            gain=self._HANDOFF_POSITION_HOLD_GAIN,
            maximum_step_m=self._HANDOFF_POSITION_HOLD_MAXIMUM_STEP_M,
            maximum_bias_m=self._HANDOFF_POSITION_HOLD_MAXIMUM_BIAS_M,
        )

        if position_error_m > cfg.settle_position_tolerance_m:
            self.ik.target.set_world_pose(
                position=update.command_position_m,
                orientation=command_orientation,
            )
            command_position = update.command_position_m

            if (
                self.frame_index - self._last_handoff_position_hold_log_frame
                >= self._HANDOFF_POSITION_HOLD_LOG_INTERVAL_FRAMES
            ):
                self._last_handoff_position_hold_log_frame = self.frame_index
                log(
                    "FROZEN HANDOFF POSITION HOLD UPDATE\n"
                    f"  physical error: {position_error_m * 1000.0:.6f} mm\n"
                    f"  command step: "
                    f"{np.linalg.norm(update.step_world_m) * 1000.0:.6f} mm\n"
                    f"  total command bias: "
                    f"{update.command_bias_m * 1000.0:.6f} mm\n"
                    f"  bias saturated: {update.bias_saturated}\n"
                    "  frozen physical goal: unchanged"
                )

        state.settled_frame_count = update_convergence_counter(
            position_error_m=position_error_m,
            tolerance_m=cfg.settle_position_tolerance_m,
            current_count=state.settled_frame_count,
        )

        if state.settled_frame_count >= cfg.required_settled_frames:
            applied_command_bias_m = update.command_bias_m
            self.ik.target.set_world_pose(
                position=goal,
                orientation=command_orientation,
            )
            command_position = goal
            state.complete = True
            log(
                "RGB QUALIFIED PORT-POSE ALIGNMENT COMPLETE\n"
                f"  frozen physical ToolCenter goal: "
                f"{np.round(goal, 6).tolist()}\n"
                f"  insertion-start IK command: "
                f"{np.round(command_position, 6).tolist()}\n"
                f"  actual ToolCenter: "
                f"{np.round(actual_position, 6).tolist()}\n"
                f"  physical goal error: "
                f"{position_error_m * 1000.0:.3f} mm\n"
                f"  applied command bias before reset: "
                f"{applied_command_bias_m * 1000.0:.3f} mm\n"
                f"  settled frames: {state.settled_frame_count}/"
                f"{cfg.required_settled_frames}\n"
                "  next action: begin guarded two-stage port entry"
            )
            return

        warning_timeout_frames = max(
            1,
            int(
                round(
                    cfg.settle_warning_timeout_s
                    / self.cfg.scene.physics_dt
                )
            ),
        )
        elapsed_frames = self.frame_index - state.settle_start_frame
        if (
            elapsed_frames >= warning_timeout_frames
            and not state.settle_timeout_reported
        ):
            state.settle_timeout_reported = True
            warn(
                "Qualified kinematic target has not physically settled; "
                "bounded position hold remains active.\n"
                f"  current physical goal error: "
                f"{position_error_m * 1000.0:.3f} mm\n"
                f"  required: "
                f"{cfg.settle_position_tolerance_m * 1000.0:.3f} mm\n"
                f"  current command bias: "
                f"{update.command_bias_m * 1000.0:.3f} mm\n"
                f"  maximum command bias: "
                f"{self._HANDOFF_POSITION_HOLD_MAXIMUM_BIAS_M * 1000.0:.3f} mm"
            )

        hard_timeout_frames = max(
            warning_timeout_frames + 1,
            int(
                round(
                    self._HANDOFF_POSITION_HOLD_HARD_TIMEOUT_S
                    / self.cfg.scene.physics_dt
                )
            ),
        )
        if elapsed_frames >= hard_timeout_frames:
            raise SystemExit(
                "Frozen handoff position hold failed closed after "
                f"{self._HANDOFF_POSITION_HOLD_HARD_TIMEOUT_S:.1f} s: "
                f"physical error={position_error_m * 1000.0:.3f} mm, "
                f"command bias={update.command_bias_m * 1000.0:.3f} mm, "
                f"bias saturated={update.bias_saturated}."
            )


__all__ = ["AngledHandStereoHandoffRuntime"]
