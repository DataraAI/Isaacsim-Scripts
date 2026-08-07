#!/usr/bin/env python3
"""Angled-hand runtime capped at a 2 mm no-contact port-plane hold."""

from __future__ import annotations

from control.insertion import InsertionEvent
from control.plug_axis_insertion import ExplicitInsertionAxisAdapter
from control.precontact_alignment import (
    PrecontactAlignmentPolicy,
    build_precontact_limits,
)
from control.settled_insertion import ConsecutivePoseInsertionController
from runtime.settled_stereo_handoff_runtime import (
    AngledHandStereoHandoffRuntime as _BaseAngledHandStereoHandoffRuntime,
)
from sim import log


class NonPenetratingConsecutivePoseInsertionController(
    ConsecutivePoseInsertionController
):
    """Reject any command at or beyond the opening before runtime publication."""

    def _issue_next_command(self, frame_index: int):
        command = super()._issue_next_command(frame_index)
        if command.commanded_port_depth_m >= 0.0:
            raise RuntimeError(
                "Precontact controller attempted a penetrating command."
            )
        return command


class AngledHandStereoHandoffRuntime(_BaseAngledHandStereoHandoffRuntime):
    """Run the qualified handoff, then stop 2 mm before the opening plane."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        mount = self.cable_mount
        self._precontact_alignment_only = bool(
            mount is not None
            and getattr(mount, "precontact_alignment_only", False)
        )
        if not self._precontact_alignment_only:
            return

        hold_offset_m = float(mount.precontact_hold_offset_m)
        policy = PrecontactAlignmentPolicy(hold_offset_m=hold_offset_m)
        capped_limits = build_precontact_limits(
            self.partial_insertion.limits,
            policy,
        )

        self.partial_insertion = (
            NonPenetratingConsecutivePoseInsertionController(capped_limits)
        )
        self._insertion_axis_adapter = ExplicitInsertionAxisAdapter(
            self.partial_insertion
        )
        self._insertion_orientation_command_wxyz = None
        self._insertion_total_steps = capped_limits.total_step_count

        terminal_port_depth_m = (
            capped_limits.total_depth_m - capped_limits.opening_depth_m
        )
        if terminal_port_depth_m >= 0.0:
            raise RuntimeError(
                "Precontact runtime produced a penetrating terminal target."
            )

        fine_travel_m = (
            capped_limits.total_depth_m
            - capped_limits.coarse_approach_depth_m
        )
        log(
            "PRECONTACT ALIGNMENT SAFETY MODE ACTIVE\n"
            f"  qualified mesh-derived TCP: required\n"
            f"  preinsert standoff: "
            f"{capped_limits.opening_depth_m * 1000.0:.3f} mm\n"
            f"  coarse approach: "
            f"{capped_limits.coarse_approach_depth_m * 1000.0:.3f} mm "
            f"at {capped_limits.coarse_step_size_m * 1000.0:.3f} mm/step\n"
            f"  fine approach: {fine_travel_m * 1000.0:.3f} mm "
            f"at {capped_limits.step_size_m * 1000.0:.3f} mm/step\n"
            f"  terminal depth relative to opening: "
            f"{terminal_port_depth_m * 1000.0:+.3f} mm\n"
            f"  total commands: {self._insertion_total_steps}\n"
            "  penetration commands: disabled\n"
            "  terminal action: hold current ToolCenter target"
        )

    def _log_partial_insertion_event(
        self,
        event: InsertionEvent,
    ) -> None:
        if not self._precontact_alignment_only:
            super()._log_partial_insertion_event(event)
            return

        labels = {
            "started": "PRECONTACT ALIGNMENT STARTED",
            "step_settled": "PRECONTACT ALIGNMENT STEP SETTLED",
            "complete": "PRECONTACT ALIGNMENT HOLD REACHED",
            "aborted": "PRECONTACT ALIGNMENT ABORTED",
        }
        lines = [labels[event.kind]]

        if event.settled_step_index is not None:
            lines.append(
                f"  settled command: {event.settled_step_index}/"
                f"{self._insertion_total_steps}"
            )
        if event.command is not None:
            lines.extend(
                [
                    f"  next command: {event.command.step_index}/"
                    f"{self._insertion_total_steps}",
                    f"  next stage: {event.command.stage.value}",
                    f"  next total travel: "
                    f"{event.command.commanded_depth_m * 1000.0:.3f} mm",
                    f"  next depth relative to opening: "
                    f"{event.command.commanded_port_depth_m * 1000.0:+.3f} mm",
                ]
            )
        if event.metrics is not None:
            metrics = event.metrics
            lines.extend(
                [
                    f"  active stage: "
                    f"{metrics.stage.value if metrics.stage is not None else 'none'}",
                    f"  commanded total travel: "
                    f"{metrics.commanded_depth_m * 1000.0:.3f} mm",
                    f"  commanded depth relative to opening: "
                    f"{metrics.commanded_port_depth_m * 1000.0:+.3f} mm",
                    f"  actual axial travel: "
                    f"{metrics.actual_axial_depth_m * 1000.0:.3f} mm",
                    f"  actual depth relative to opening: "
                    f"{metrics.actual_port_depth_m * 1000.0:+.3f} mm",
                    f"  lateral drift: "
                    f"{metrics.lateral_drift_m * 1000.0:.3f} mm",
                    f"  ToolCenter tracking error: "
                    f"{metrics.target_error_m * 1000.0:.3f} mm",
                    f"  orientation error: "
                    f"{metrics.orientation_error_deg:.6f} deg",
                    f"  plug-tip mount error: "
                    f"{metrics.mount_tip_error_m * 1000.0:.6f} mm",
                    f"  plug-axis error: "
                    f"{metrics.mount_axis_error_deg:.6f} deg",
                    f"  settled frames: "
                    f"{metrics.settled_frame_count}/"
                    f"{self.partial_insertion.limits.required_settled_frames}",
                    f"  elapsed step frames: "
                    f"{metrics.elapsed_step_frames}/"
                    f"{self.partial_insertion.limits.step_timeout_frames}",
                ]
            )
        if event.reason is not None:
            lines.append(f"  reason: {event.reason}")
        if event.kind == "complete":
            lines.extend(
                [
                    "  penetration commands: disabled",
                    "  next action: hold 2 mm before the opening plane",
                ]
            )
        elif event.kind == "aborted":
            lines.append("  next action: hold current ToolCenter target")

        log("\n".join(lines))
