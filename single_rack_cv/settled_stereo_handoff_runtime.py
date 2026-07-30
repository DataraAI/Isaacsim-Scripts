#!/usr/bin/env python3
"""Angled stereo handoff runtime with a consecutive startup geometry gate."""

from __future__ import annotations

from cable_geometry import validate_mount_window
from cable_runtime import (
    CableMountedSimulationRuntime as _CableMountedSimulationRuntime,
)
from stereo_handoff_runtime import (
    AngledHandStereoHandoffRuntime as _BaseAngledHandStereoHandoffRuntime,
)


class AngledHandStereoHandoffRuntime(
    _BaseAngledHandStereoHandoffRuntime
):
    """
    Require a complete consecutive window of valid startup hand/plug geometry.

    Position convergence can become ready a few frames before the hand/plug
    orientation finishes settling. Geometry-only misses reset the startup
    validation window and keep settling. Structural mount failures still raise
    immediately, and every existing numerical limit remains unchanged.

    After stereo handoff completes, insertion safety belongs to the guarded
    PartialInsertionController. The camera-baseline presentation check must not
    deadlock command one because it is not the plug insertion orientation.
    """

    _TRANSIENT_GEOMETRY_PREFIXES = (
        "hand-to-plug pitch error exceeded limit:",
        "wrong hand pitch sign:",
        "palm side does not match the previous working pose:",
        "plug horizontal error exceeded limit:",
    )

    @classmethod
    def _is_transient_geometry_error(cls, error: RuntimeError) -> bool:
        message = str(error)
        return any(
            message.startswith(prefix)
            for prefix in cls._TRANSIENT_GEOMETRY_PREFIXES
        )

    def prepare_for_perception(self) -> None:
        """Settle until the full strict startup geometry window is consecutive."""

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
        """
        Keep the strict hand/camera presentation gate only before perception.

        Once stereo handoff has completed, the frozen ToolCenter quaternion
        guard in PartialInsertionController owns orientation safety. It ignores
        brief non-contact motion transients, then enforces the unchanged one
        degree limit after settling or at the opening plane. Structural fixed-
        joint, attachment, plug-tip, and plug-axis checks remain active through
        the base cable runtime on every insertion frame.
        """

        if not self.visual_servo.complete:
            return super()._sample_mount_validation_live(runtime)
        return _CableMountedSimulationRuntime._sample_mount_validation_live(
            self,
            runtime,
        )
