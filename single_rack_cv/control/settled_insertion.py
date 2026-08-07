#!/usr/bin/env python3
"""Insertion controller that settles translation and orientation together."""

from __future__ import annotations

import math

import numpy as np

from control.insertion import (
    InsertionEvent,
    InsertionPhase,
    InsertionStage,
    PartialInsertionController,
)

_EPS = 1.0e-12
_LATERAL_ABORT_PREFIX = "lateral drift exceeded limit:"
_ORIENTATION_ABORT_PREFIX = "orientation error exceeded limit:"


class ConsecutivePoseInsertionController(PartialInsertionController):
    """
    Require consecutive position-and-orientation validity before each step settles.

    A lateral excursion measured while the ToolCenter is still outside the
    active translation settle radius is an in-flight tracking transient, not a
    settled centerline violation. Keep the unchanged centerline target and let
    IK recover. The unchanged 0.5 mm lateral limit is enforced once translation
    settles; failure to settle still reaches the existing step timeout.

    Before the port opening, a brief orientation excursion resets the settle
    window and lets the commanded frozen quaternion recover. At or inside the
    opening plane, the unchanged orientation limit remains an immediate abort.
    Every structural, mount, timeout, and terminal check remains owned by the
    base controller.

    The optional fine-stage policy can require a tighter position tolerance,
    a longer consecutive settle window, and a maximum measured ToolCenter
    displacement per simulation frame. The generic controller defaults remain
    unchanged unless a runtime explicitly opts into that policy.
    """

    def __init__(
        self,
        limits,
        *,
        fine_settle_tolerance_m: float | None = None,
        fine_required_settled_frames: int | None = None,
        fine_max_motion_per_frame_m: float | None = None,
    ):
        super().__init__(limits)

        self.fine_settle_tolerance_m = (
            float(limits.settle_tolerance_m)
            if fine_settle_tolerance_m is None
            else float(fine_settle_tolerance_m)
        )
        self.fine_required_settled_frames = (
            int(limits.required_settled_frames)
            if fine_required_settled_frames is None
            else int(fine_required_settled_frames)
        )
        self.fine_max_motion_per_frame_m = (
            math.inf
            if fine_max_motion_per_frame_m is None
            else float(fine_max_motion_per_frame_m)
        )

        if (
            not math.isfinite(self.fine_settle_tolerance_m)
            or self.fine_settle_tolerance_m <= 0.0
        ):
            raise ValueError(
                "fine_settle_tolerance_m must be finite and positive"
            )
        if self.fine_required_settled_frames <= 0:
            raise ValueError(
                "fine_required_settled_frames must be positive"
            )
        if (
            self.fine_max_motion_per_frame_m != math.inf
            and (
                not math.isfinite(self.fine_max_motion_per_frame_m)
                or self.fine_max_motion_per_frame_m <= 0.0
            )
        ):
            raise ValueError(
                "fine_max_motion_per_frame_m must be finite and positive"
            )

        self._previous_actual_position_m: np.ndarray | None = None
        self._previous_frame_index: int | None = None
        self.last_tool_motion_per_frame_m = math.inf

    def _record_tool_motion_per_frame(self, sample) -> float:
        current_position = np.asarray(
            sample.actual_position_m,
            dtype=np.float64,
        ).reshape(3)
        current_frame = int(sample.frame_index)

        if (
            self._previous_actual_position_m is None
            or self._previous_frame_index is None
            or current_frame <= self._previous_frame_index
        ):
            motion_per_frame_m = math.inf
        else:
            elapsed_frames = current_frame - self._previous_frame_index
            motion_per_frame_m = float(
                np.linalg.norm(
                    current_position - self._previous_actual_position_m
                )
                / float(elapsed_frames)
            )

        self._previous_actual_position_m = current_position.copy()
        self._previous_frame_index = current_frame
        self.last_tool_motion_per_frame_m = motion_per_frame_m
        return motion_per_frame_m

    def _active_settle_policy(
        self,
        stage: InsertionStage | None,
    ) -> tuple[float, int, float]:
        if stage is InsertionStage.FINE_INSERTION:
            return (
                self.fine_settle_tolerance_m,
                self.fine_required_settled_frames,
                self.fine_max_motion_per_frame_m,
            )
        return (
            float(self.limits.settle_tolerance_m),
            int(self.limits.required_settled_frames),
            math.inf,
        )

    def update(self, sample) -> InsertionEvent:
        if self.phase is not InsertionPhase.ADVANCING:
            event = super().update(sample)
            self._record_tool_motion_per_frame(sample)
            return event

        tool_motion_per_frame_m = self._record_tool_motion_per_frame(sample)
        metrics = self._metrics(sample)
        if metrics is None:
            return self.abort("insertion metrics are unavailable", sample)

        (
            active_position_tolerance_m,
            active_required_settled_frames,
            active_max_motion_per_frame_m,
        ) = self._active_settle_policy(metrics.stage)

        reason = self._abort_reason(
            sample,
            metrics=metrics,
            include_start_tracking_error=False,
        )
        if (
            reason is not None
            and reason.startswith(_LATERAL_ABORT_PREFIX)
            and sample.target_error_m > active_position_tolerance_m
        ):
            # The commanded target remains on the frozen insertion centerline.
            # While the physical ToolCenter is still converging, its lateral
            # displacement is part of the unresolved target-tracking error.
            # Do not certify the step and do not abort on one in-flight sample.
            reason = None
        if (
            reason is not None
            and reason.startswith(_ORIENTATION_ABORT_PREFIX)
            and metrics.actual_port_depth_m < -_EPS
        ):
            # Non-contact motion may briefly lag the frozen quaternion. Do not
            # certify the step; reset its consecutive settle window instead.
            reason = None
        if reason is not None:
            return self.abort(reason, sample)

        position_valid = (
            sample.target_error_m <= active_position_tolerance_m
        )
        orientation_valid = (
            metrics.orientation_error_deg
            <= self.limits.max_orientation_error_deg
        )
        lateral_valid = (
            metrics.lateral_drift_m
            <= self.limits.max_lateral_drift_m
        )
        motion_valid = (
            tool_motion_per_frame_m <= active_max_motion_per_frame_m
        )
        if (
            position_valid
            and orientation_valid
            and lateral_valid
            and motion_valid
        ):
            self.settled_frame_count += 1
        else:
            self.settled_frame_count = 0

        metrics = self._metrics(sample)
        if metrics is None:
            return self.abort("insertion metrics are unavailable", sample)

        if self.settled_frame_count < active_required_settled_frames:
            if (
                sample.frame_index - self.step_start_frame
                >= self.limits.step_timeout_frames
            ):
                if position_valid and not lateral_valid:
                    timeout_reason = (
                        "insertion step timeout before lateral drift recovered: "
                        f"{metrics.lateral_drift_m * 1000.0:.6f} mm"
                    )
                elif position_valid and not orientation_valid:
                    timeout_reason = (
                        "insertion step timeout before orientation recovered: "
                        f"{metrics.orientation_error_deg:.6f} deg"
                    )
                elif position_valid and not motion_valid:
                    timeout_reason = (
                        "insertion step timeout before ToolCenter became quiet: "
                        f"{tool_motion_per_frame_m * 1000.0:.6f} mm/frame"
                    )
                else:
                    timeout_reason = (
                        "insertion step timeout before physical pose settled"
                    )
                return self.abort(timeout_reason, sample)
            return InsertionEvent(
                kind="waiting_for_settle",
                phase=self.phase,
                command=None,
                metrics=metrics,
            )

        settled_step_index = self.commanded_step_index
        if self.commanded_depth_m >= self.limits.total_depth_m - _EPS:
            self.phase = InsertionPhase.COMPLETE
            return InsertionEvent(
                kind="complete",
                phase=self.phase,
                command=None,
                metrics=metrics,
                settled_step_index=settled_step_index,
            )

        command = self._issue_next_command(sample.frame_index)
        return InsertionEvent(
            kind="step_settled",
            phase=self.phase,
            command=command,
            metrics=metrics,
            settled_step_index=settled_step_index,
        )
