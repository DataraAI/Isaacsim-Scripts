#!/usr/bin/env python3
"""Angled stereo handoff runtime with guarded startup and pose settling."""

from __future__ import annotations

import numpy as np

from cable.cable_geometry import validate_mount_window
from runtime.cable_runtime import (
    CableMountedSimulationRuntime as _CableMountedSimulationRuntime,
)
from control.insertion import InsertionPhase
from control.orientation_hold import update_orientation_hold_command
from control.plug_axis_insertion import ExplicitInsertionAxisAdapter
from control.settled_insertion import ConsecutivePoseInsertionController
from sim import log
from runtime.stereo_handoff_runtime import (
    AngledHandStereoHandoffRuntime as _BaseAngledHandStereoHandoffRuntime,
)


class AngledHandStereoHandoffRuntime(
    _BaseAngledHandStereoHandoffRuntime
):
    """Require consecutive startup geometry and insertion pose settling."""

    _TRANSIENT_GEOMETRY_PREFIXES = (
        "hand-to-plug pitch error exceeded limit:",
        "wrong hand pitch sign:",
        "palm side does not match the previous working pose:",
        "plug horizontal error exceeded limit:",
    )
    _ORIENTATION_HOLD_GAIN = 0.35
    _ORIENTATION_HOLD_MAXIMUM_STEP_DEG = 0.15
    _ORIENTATION_HOLD_MAXIMUM_BIAS_DEG = 3.0
    _ORIENTATION_HOLD_LOG_INTERVAL_FRAMES = 30

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        limits = self.partial_insertion.limits
        self.partial_insertion = ConsecutivePoseInsertionController(limits)
        self._insertion_axis_adapter = ExplicitInsertionAxisAdapter(
            self.partial_insertion
        )
        self._insertion_orientation_command_wxyz: np.ndarray | None = None
        self._last_orientation_hold_log_frame = -1_000_000

        log(
            "PROVEN MAIN INSERTION SETTLING ACTIVE\n"
            "  position tolerance: 0.300 mm\n"
            "  required consecutive frames: 6\n"
            "  fine-stage motion gate: disabled\n"
            "  angled-hand orientation hold: active\n"
            "  stiffness, damping, force limits, and safety limits: unchanged"
        )

    @classmethod
    def _is_transient_geometry_error(cls, error: RuntimeError) -> bool:
        message = str(error)
        return any(
            message.startswith(prefix)
            for prefix in cls._TRANSIENT_GEOMETRY_PREFIXES
        )

    def update_ik(self) -> None:
        self._update_insertion_orientation_hold_target()
        super().update_ik()

    def _update_insertion_orientation_hold_target(self) -> None:
        controller = self.partial_insertion
        reference = controller.frozen_orientation_wxyz
        if (
            self.ik is None
            or controller.phase is not InsertionPhase.ADVANCING
            or reference is None
        ):
            self._insertion_orientation_command_wxyz = None
            return

        if self._insertion_orientation_command_wxyz is None:
            self._insertion_orientation_command_wxyz = np.asarray(
                reference,
                dtype=np.float64,
            ).copy()

        _, actual_orientation = self._tool_pose_from_articulation()
        update = update_orientation_hold_command(
            reference_wxyz=reference,
            actual_wxyz=actual_orientation,
            current_command_wxyz=(
                self._insertion_orientation_command_wxyz
            ),
            gain=self._ORIENTATION_HOLD_GAIN,
            maximum_step_deg=self._ORIENTATION_HOLD_MAXIMUM_STEP_DEG,
            maximum_bias_deg=self._ORIENTATION_HOLD_MAXIMUM_BIAS_DEG,
        )
        self._insertion_orientation_command_wxyz = (
            update.command_wxyz.copy()
        )

        target_position, _ = self.ik.target.get_world_pose()
        self.ik.target.set_world_pose(
            position=np.asarray(target_position, dtype=np.float64),
            orientation=self._insertion_orientation_command_wxyz,
        )

        if (
            update.actual_error_deg
            > controller.limits.max_orientation_error_deg
            and self.frame_index - self._last_orientation_hold_log_frame
            >= self._ORIENTATION_HOLD_LOG_INTERVAL_FRAMES
        ):
            self._last_orientation_hold_log_frame = self.frame_index
            log(
                "INSERTION ORIENTATION HOLD ACTIVE\n"
                f"  actual error: {update.actual_error_deg:.6f} deg\n"
                f"  command compensation: "
                f"{update.command_bias_deg:.6f} deg\n"
                f"  maximum command step: "
                f"{self._ORIENTATION_HOLD_MAXIMUM_STEP_DEG:.3f} deg/frame\n"
                f"  maximum command compensation: "
                f"{self._ORIENTATION_HOLD_MAXIMUM_BIAS_DEG:.3f} deg\n"
                f"  compensation saturated: {update.bias_saturated}\n"
                "  safety reference and 1 degree actual-pose limit: unchanged"
            )

    def prepare_for_perception(self) -> None:
        if self.cable_mount is None:
            return

        cfg = self.cfg.cable_mount
        samples: list[tuple[float, float]] = []
        max_prepare_frames = (
            cfg.initial_settle_frames
            + cfg.validation_frames
            + 600
        )
        minimum_tool_error_m = float("inf")
        maximum_tool_error_m = 0.0
        current_tool_error_m = float("inf")
        last_transient_error: RuntimeError | None = None

        for frame_count in range(max_prepare_frames):
            self.step()
            self.update_ik()
            self._update_startup_settle()
            current_tool_error_m = self._tool_target_position_error_m()
            minimum_tool_error_m = min(
                minimum_tool_error_m,
                current_tool_error_m,
            )
            maximum_tool_error_m = max(
                maximum_tool_error_m,
                current_tool_error_m,
            )

            if frame_count == 0 or (frame_count + 1) % 120 == 0:
                self._log_startup_diagnostics(
                    frame_count=frame_count + 1,
                    minimum_tool_error_m=minimum_tool_error_m,
                    maximum_tool_error_m=maximum_tool_error_m,
                    validation_sample_count=len(samples),
                )

            if frame_count < cfg.initial_settle_frames:
                continue
            if not self.visual_servo.startup_ready:
                samples.clear()
                continue

            try:
                sample = self.cable_mount.sample_validation(self)
            except RuntimeError as error:
                if not self._is_transient_geometry_error(error):
                    raise
                samples.clear()
                last_transient_error = error
                continue

            samples.append(sample)
            if len(samples) == cfg.validation_frames:
                break
        else:
            self._log_startup_diagnostics(
                frame_count=max_prepare_frames,
                minimum_tool_error_m=minimum_tool_error_m,
                maximum_tool_error_m=maximum_tool_error_m,
                validation_sample_count=len(samples),
            )
            transient_status = (
                "none"
                if last_transient_error is None
                else str(last_transient_error)
            )
            raise RuntimeError(
                "Cable mount startup gate timed out.\n"
                f"  startup ready: {self.visual_servo.startup_ready}\n"
                f"  settled frames: "
                f"{self.visual_servo.startup_settled_frame_count}/"
                f"{self.cfg.visual_servo.required_startup_settled_frames}\n"
                f"  ToolCenter error current/min/max mm: "
                f"{current_tool_error_m * 1000.0:.6f} / "
                f"{minimum_tool_error_m * 1000.0:.6f} / "
                f"{maximum_tool_error_m * 1000.0:.6f}\n"
                f"  consecutive validation samples: {len(samples)}/"
                f"{cfg.validation_frames}\n"
                f"  last transient geometry error: {transient_status}"
            )

        validation = validate_mount_window(
            samples,
            cfg.validation_frames,
            cfg.max_tip_error_m,
            cfg.max_axis_error_deg,
        )
        self.cable_mount.log_success(validation)

    def _sample_mount_validation_live(self, runtime) -> tuple[float, float]:
        if not self.visual_servo.complete:
            return super()._sample_mount_validation_live(runtime)
        return _CableMountedSimulationRuntime._sample_mount_validation_live(
            self,
            runtime,
        )
