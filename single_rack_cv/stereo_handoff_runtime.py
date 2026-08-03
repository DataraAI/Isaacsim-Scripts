#!/usr/bin/env python3
"""Angled-hand runtime with a proactive guarded stereo handoff."""

from __future__ import annotations

from collections import deque

import numpy as np

from angled_hand_runtime import AngledHandCableRuntime
from sim import (
    log,
    update_convergence_counter,
    warn,
)
from stereo_handoff import (
    bounded_step_to_goal,
    select_recent_bounded_goal,
)


class AngledHandStereoHandoffRuntime(
    AngledHandCableRuntime
):
    """
    Use stereo until a stable nearby world goal exists, then finish kinematically.

    Every valid stereo observation implies a final world-space ToolCenter goal:

        actual_tool_position + correction_world

    Only the newest three implied goals are considered. Once they agree within
    2 mm and the remaining translation is at most 35 mm, the runtime freezes
    their median goal and finishes in bounded 5 mm steps. This transition is
    proactive; it occurs before the angled wrist destroys the camera view.
    """

    _GOAL_HISTORY_LENGTH = 8
    _MINIMUM_GOAL_SAMPLES = 3
    _RECENT_GOAL_WINDOW = 3
    _MAXIMUM_GOAL_SPREAD_M = 0.002
    _MAXIMUM_HANDOFF_DISTANCE_M = 0.035
    _MAXIMUM_HANDOFF_STEP_M = 0.005

    def __init__(
        self,
        simulation_app,
        cfg,
    ):
        self._stereo_goal_candidates: deque[
            np.ndarray
        ] = deque(
            maxlen=self._GOAL_HISTORY_LENGTH
        )
        self._handoff_goal_world_m: (
            np.ndarray | None
        ) = None
        self._handoff_active = False
        self._handoff_step_index = 0
        self._sparse_acquisition_notice_logged = False

        super().__init__(
            simulation_app=simulation_app,
            cfg=cfg,
        )

        log(
            "PROACTIVE GUARDED STEREO-TO-KINEMATIC HANDOFF READY\n"
            f"  newest stable stereo goals: "
            f"{self._RECENT_GOAL_WINDOW}\n"
            f"  maximum goal spread: "
            f"{self._MAXIMUM_GOAL_SPREAD_M * 1000.0:.1f} mm\n"
            f"  maximum bounded finish distance: "
            f"{self._MAXIMUM_HANDOFF_DISTANCE_M * 1000.0:.1f} mm\n"
            f"  maximum handoff step: "
            f"{self._MAXIMUM_HANDOFF_STEP_M * 1000.0:.1f} mm"
        )

    def observe_visual_servo(
        self,
        observation,
    ) -> None:
        """Record the implied world goal and hand off before vision collapses."""

        if self._handoff_active:
            return

        if self.ik is None:
            super().observe_visual_servo(
                observation
            )
            return

        self._update_actual_tool_frame(
            self.ik
        )
        actual_position_before, _ = (
            self.ik.actual_tool.get_world_pose()
        )
        actual_position_before = np.asarray(
            actual_position_before,
            dtype=np.float64,
        )

        correction_world_m = np.asarray(
            observation.correction_world_m,
            dtype=np.float64,
        )

        super().observe_visual_servo(
            observation
        )

        state = self.visual_servo

        if (
            not state.acquired
            or state.visual_aligned
            or state.complete
            or correction_world_m.shape != (3,)
            or not np.all(
                np.isfinite(
                    correction_world_m
                )
            )
        ):
            return

        self._stereo_goal_candidates.append(
            actual_position_before
            + correction_world_m
        )

        self._try_start_handoff(
            reason=(
                "newest stable stereo goals entered "
                "the bounded finish region"
            )
        )

    def note_perception_failure(
        self,
    ) -> None:
        """Preserve accepted startup samples while the stationary view is sparse."""

        state = self.visual_servo

        if state.complete:
            return

        if self._handoff_active:
            return

        if self._try_start_handoff(
            reason=(
                "perception failed after a stable "
                "bounded stereo goal was established"
            )
        ):
            return

        retained_acquisition_features = None
        if not state.acquired and state.acquisition_features:
            retained_acquisition_features = [
                np.asarray(
                    feature,
                    dtype=np.float64,
                ).copy()
                for feature in state.acquisition_features
            ]

        super().note_perception_failure()

        if (
            retained_acquisition_features is not None
            and not state.acquired
            and not state.acquisition_features
        ):
            keep_count = int(
                self.cfg.visual_servo.required_acquisition_samples
            )
            state.acquisition_features.extend(
                retained_acquisition_features[-keep_count:]
            )

            if not self._sparse_acquisition_notice_logged:
                self._sparse_acquisition_notice_logged = True
                log(
                    "RGB stereo retaining accepted acquisition samples while the hand is stationary\n"
                    f"  retained samples: {len(state.acquisition_features)}/"
                    f"{keep_count}\n"
                    "  rejected frames clear detector references, not accepted 3D evidence"
                )

    def _try_start_handoff(
        self,
        *,
        reason: str,
    ) -> bool:
        """Start once the newest stable stereo goal is within 35 mm."""

        state = self.visual_servo

        if (
            self._handoff_active
            or state.complete
            or not state.acquired
            or self.ik is None
        ):
            return False

        target_position, _ = (
            self.ik.target.get_world_pose()
        )
        target_position = np.asarray(
            target_position,
            dtype=np.float64,
        )

        decision = select_recent_bounded_goal(
            self._stereo_goal_candidates,
            target_position,
            minimum_samples=(
                self._MINIMUM_GOAL_SAMPLES
            ),
            recent_sample_count=(
                self._RECENT_GOAL_WINDOW
            ),
            maximum_spread_m=(
                self._MAXIMUM_GOAL_SPREAD_M
            ),
            maximum_distance_m=(
                self._MAXIMUM_HANDOFF_DISTANCE_M
            ),
        )

        if decision is None:
            return False

        estimate = decision.estimate

        self._handoff_goal_world_m = (
            estimate.position_m.copy()
        )
        self._handoff_active = True
        self._handoff_step_index = 0

        state.consecutive_misses = 0
        state.aligned_capture_count = 0
        state.visual_aligned = False
        state.settled_frame_count = 0
        state.settle_timeout_reported = False

        log(
            "RGB STEREO-TO-KINEMATIC HANDOFF STARTED\n"
            f"  recent world-goal samples: "
            f"{estimate.sample_count}\n"
            f"  world-goal spread: "
            f"{estimate.spread_m * 1000.0:.3f} mm\n"
            f"  remaining translation: "
            f"{decision.remaining_m * 1000.0:.3f} mm\n"
            f"  maximum permitted: "
            f"{self._MAXIMUM_HANDOFF_DISTANCE_M * 1000.0:.3f} mm\n"
            f"  reason: {reason}\n"
            "  destination: stereo-derived 50 mm pre-insert standoff\n"
            "  orientation: unchanged horizontal plug"
        )

        self._advance_handoff_if_settled()
        return True

    def capture_due(
        self,
    ) -> bool:
        """Stop capturing after handoff and advance after each physical settle."""

        if (
            self._handoff_active
            and not self.visual_servo.visual_aligned
        ):
            self._advance_handoff_if_settled()
            return False

        return super().capture_due()

    def _advance_handoff_if_settled(
        self,
    ) -> None:
        """Issue one bounded translation toward the frozen world goal."""

        state = self.visual_servo
        cfg = self.cfg.visual_servo
        goal = self._handoff_goal_world_m

        if (
            not self._handoff_active
            or goal is None
            or self.ik is None
            or state.visual_aligned
            or state.complete
        ):
            return

        tracking_error_m = (
            self._tool_target_position_error_m()
        )

        if (
            tracking_error_m
            > cfg.target_settle_tolerance_m
        ):
            return

        target_position, target_orientation = (
            self.ik.target.get_world_pose()
        )
        target_position = np.asarray(
            target_position,
            dtype=np.float64,
        )

        maximum_step_m = min(
            float(
                cfg.max_target_step_m
            ),
            self._MAXIMUM_HANDOFF_STEP_M,
        )

        step_world_m, remaining_m = (
            bounded_step_to_goal(
                target_position,
                goal,
                maximum_step_m=maximum_step_m,
            )
        )

        completion_radius_m = min(
            float(
                cfg.settle_position_tolerance_m
            ),
            0.0003,
        )

        if (
            remaining_m
            <= completion_radius_m
        ):
            state.visual_aligned = True
            state.settled_frame_count = 0
            state.settle_start_frame = (
                self.frame_index
            )
            state.settle_timeout_reported = False

            log(
                "RGB STEREO-TO-KINEMATIC HANDOFF TARGET REACHED\n"
                f"  frozen world goal: "
                f"{np.round(goal, 6).tolist()}\n"
                f"  remaining target error: "
                f"{remaining_m * 1000.0:.3f} mm\n"
                "  next action: verify physical "
                "ToolCenter settling"
            )
            return

        self._handoff_step_index += 1

        self.ik.target.set_world_pose(
            position=(
                target_position
                + step_world_m
            ),
            orientation=target_orientation,
        )

        log(
            "RGB STEREO-TO-KINEMATIC HANDOFF STEP\n"
            f"  step: "
            f"{self._handoff_step_index}\n"
            f"  commanded translation: "
            f"{np.linalg.norm(step_world_m) * 1000.0:.3f} mm\n"
            f"  remaining before step: "
            f"{remaining_m * 1000.0:.3f} mm"
        )

    def update_visual_servo_completion(
        self,
    ) -> None:
        """Complete the hybrid alignment without claiming continuous vision."""

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

        self._update_actual_tool_frame(
            self.ik
        )

        target_position, _ = (
            self.ik.target.get_world_pose()
        )
        actual_position, _ = (
            self.ik.actual_tool.get_world_pose()
        )

        target_position = np.asarray(
            target_position,
            dtype=np.float64,
        )
        actual_position = np.asarray(
            actual_position,
            dtype=np.float64,
        )

        position_error_m = (
            self._tool_target_position_error_m()
        )

        state.settled_frame_count = (
            update_convergence_counter(
                position_error_m=position_error_m,
                tolerance_m=(
                    cfg.settle_position_tolerance_m
                ),
                current_count=(
                    state.settled_frame_count
                ),
            )
        )

        if (
            state.settled_frame_count
            >= cfg.required_settled_frames
        ):
            state.complete = True

            log(
                "RGB STEREO-TO-KINEMATIC ALIGNMENT COMPLETE\n"
                f"  final ToolCenter target: "
                f"{np.round(target_position, 6).tolist()}\n"
                f"  actual ToolCenter: "
                f"{np.round(actual_position, 6).tolist()}\n"
                f"  physical tracking error: "
                f"{position_error_m * 1000.0:.3f} mm\n"
                f"  settled frames: "
                f"{state.settled_frame_count}/"
                f"{cfg.required_settled_frames}\n"
                "  next action: begin guarded "
                "two-stage port entry"
            )
            return

        timeout_frames = max(
            1,
            int(
                round(
                    cfg.settle_warning_timeout_s
                    / self.cfg.scene.physics_dt
                )
            ),
        )

        if (
            self.frame_index
            - state.settle_start_frame
            >= timeout_frames
            and not state.settle_timeout_reported
        ):
            state.settle_timeout_reported = True

            warn(
                "Stereo-to-kinematic handoff target "
                "has not physically settled.\n"
                f"  current physical error: "
                f"{position_error_m * 1000.0:.3f} mm\n"
                f"  required: "
                f"{cfg.settle_position_tolerance_m * 1000.0:.3f} mm"
            )
