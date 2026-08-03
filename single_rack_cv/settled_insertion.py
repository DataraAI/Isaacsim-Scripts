#!/usr/bin/env python3
"""Insertion controller that settles translation and orientation together."""

from __future__ import annotations

from insertion import (
    InsertionEvent,
    InsertionPhase,
    PartialInsertionController,
)

_EPS = 1.0e-12
_LATERAL_ABORT_PREFIX = "lateral drift exceeded limit:"
_ORIENTATION_ABORT_PREFIX = "orientation error exceeded limit:"


class ConsecutivePoseInsertionController(PartialInsertionController):
    """
    Require consecutive position-and-orientation validity before each step settles.

    A lateral excursion measured while the ToolCenter is still outside the
    translation settle radius is an in-flight tracking transient, not a settled
    centerline violation. Keep the unchanged centerline target and let IK
    recover. The unchanged 0.5 mm lateral limit is enforced once translation
    settles; failure to settle still reaches the existing step timeout.

    Before the port opening, a brief orientation excursion resets the settle
    window and lets the commanded frozen quaternion recover. At or inside the
    opening plane, the unchanged orientation limit remains an immediate abort.
    Every structural, mount, timeout, and terminal check remains owned by the
    base controller.
    """

    def update(self, sample) -> InsertionEvent:
        if self.phase is not InsertionPhase.ADVANCING:
            return super().update(sample)

        metrics = self._metrics(sample)
        if metrics is None:
            return self.abort("insertion metrics are unavailable", sample)

        reason = self._abort_reason(
            sample,
            metrics=metrics,
            include_start_tracking_error=False,
        )
        if (
            reason is not None
            and reason.startswith(_LATERAL_ABORT_PREFIX)
            and sample.target_error_m > self.limits.settle_tolerance_m
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
            sample.target_error_m <= self.limits.settle_tolerance_m
        )
        orientation_valid = (
            metrics.orientation_error_deg
            <= self.limits.max_orientation_error_deg
        )
        lateral_valid = (
            metrics.lateral_drift_m
            <= self.limits.max_lateral_drift_m
        )
        if position_valid and orientation_valid and lateral_valid:
            self.settled_frame_count += 1
        else:
            self.settled_frame_count = 0

        metrics = self._metrics(sample)
        if metrics is None:
            return self.abort("insertion metrics are unavailable", sample)

        if self.settled_frame_count < self.limits.required_settled_frames:
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
