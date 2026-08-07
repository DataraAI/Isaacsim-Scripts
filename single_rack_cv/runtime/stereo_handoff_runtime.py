#!/usr/bin/env python3
"""Angled-hand runtime with one-shot qualified stereo pose freezing."""

from __future__ import annotations

from collections import deque

import numpy as np

from runtime.angled_hand_runtime import AngledHandCableRuntime
from sim import (
    log,
    update_convergence_counter,
    warn,
)
from control.stereo_handoff import (
    bounded_step_to_goal,
    qualify_stationary_port_goal,
)


class AngledHandStereoHandoffRuntime(
    AngledHandCableRuntime
):
    """
    Qualify the physical port while stationary, then finish kinematically.

    Continuous front-rim detection is intentionally not required after motion
    starts. Three accepted stationary observations must agree on both the RGB
    front-mouth point and the derived 50 mm ToolCenter preinsert destination.
    Their medians are frozen once. Camera capture then stops and the robot moves
    to the frozen goal in bounded steps before the existing guarded insertion.
    """

    _QUALIFICATION_HISTORY_LENGTH = 8
    _MINIMUM_QUALIFICATION_SAMPLES = 3
    _RECENT_QUALIFICATION_WINDOW = 3
    _MAXIMUM_OPENING_SPREAD_M = 0.001
    _MAXIMUM_GOAL_SPREAD_M = 0.002
    _STANDOFF_TOLERANCE_M = 0.003
    _MAXIMUM_QUALIFIED_TRAVEL_M = 0.120
    _MAXIMUM_HANDOFF_STEP_M = 0.005

    def __init__(
        self,
        simulation_app,
        cfg,
    ):
        self._stationary_opening_candidates: deque[np.ndarray] = deque(
            maxlen=self._QUALIFICATION_HISTORY_LENGTH
        )
        self._stationary_tool_goal_candidates: deque[np.ndarray] = deque(
            maxlen=self._QUALIFICATION_HISTORY_LENGTH
        )
        self._frozen_port_point_world_m: np.ndarray | None = None
        self._handoff_goal_world_m: np.ndarray | None = None
        self._handoff_active = False
        self._handoff_step_index = 0
        self._sparse_acquisition_notice_logged = False
        self._qualification_wait_notice_logged = False

        super().__init__(
            simulation_app=simulation_app,
            cfg=cfg,
        )

        log(
            "QUALIFIED PORT-POSE TO KINEMATIC HANDOFF READY\n"
            f"  stationary accepted samples: "
            f"{self._RECENT_QUALIFICATION_WINDOW}\n"
            f"  maximum opening spread: "
            f"{self._MAXIMUM_OPENING_SPREAD_M * 1000.0:.1f} mm\n"
            f"  maximum preinsert-goal spread: "
            f"{self._MAXIMUM_GOAL_SPREAD_M * 1000.0:.1f} mm\n"
            f"  required opening-to-goal standoff: "
            f"{self.cfg.visual_servo.preinsert_standoff_m * 1000.0:.1f} "
            f"+/- {self._STANDOFF_TOLERANCE_M * 1000.0:.1f} mm\n"
            f"  maximum qualified travel: "
            f"{self._MAXIMUM_QUALIFIED_TRAVEL_M * 1000.0:.1f} mm\n"
            f"  maximum kinematic step: "
            f"{self._MAXIMUM_HANDOFF_STEP_M * 1000.0:.1f} mm"
        )

    @property
    def frozen_port_point_world_m(self) -> np.ndarray | None:
        """Return a copy of the qualified physical mouth point for debugging."""

        if self._frozen_port_point_world_m is None:
            return None
        return self._frozen_port_point_world_m.copy()

    def observe_visual_servo(
        self,
        observation,
    ) -> None:
        """Qualify one stationary physical port pose before issuing any motion."""

        if self._handoff_active:
            return

        state = self.visual_servo
        cfg = self.cfg.visual_servo

        if not cfg.enabled or state.complete:
            return
        if self.ik is None:
            raise RuntimeError(
                "Stationary port qualification requires initialized IK."
            )

        state.left_reference = observation.left.detection
        state.right_reference = observation.right.detection
        state.consecutive_misses = 0

        self._update_actual_tool_frame(self.ik)
        actual_position_before, _ = self.ik.actual_tool.get_world_pose()
        actual_position_before = np.asarray(
            actual_position_before,
            dtype=np.float64,
        )
        opening_position_world_m = np.asarray(
            observation.center_world_xyz_m,
            dtype=np.float64,
        )
        correction_world_m = np.asarray(
            observation.correction_world_m,
            dtype=np.float64,
        )

        if (
            actual_position_before.shape != (3,)
            or opening_position_world_m.shape != (3,)
            or correction_world_m.shape != (3,)
            or not np.all(np.isfinite(actual_position_before))
            or not np.all(np.isfinite(opening_position_world_m))
            or not np.all(np.isfinite(correction_world_m))
        ):
            raise RuntimeError(
                "Stationary port qualification received invalid 3D geometry."
            )

        self._stationary_opening_candidates.append(
            opening_position_world_m.copy()
        )
        self._stationary_tool_goal_candidates.append(
            actual_position_before + correction_world_m
        )

        if not state.acquired:
            self._update_visual_acquisition(observation)
            if not state.acquired:
                return

        qualification = qualify_stationary_port_goal(
            self._stationary_opening_candidates,
            self._stationary_tool_goal_candidates,
            minimum_samples=self._MINIMUM_QUALIFICATION_SAMPLES,
            recent_sample_count=self._RECENT_QUALIFICATION_WINDOW,
            maximum_opening_spread_m=self._MAXIMUM_OPENING_SPREAD_M,
            maximum_goal_spread_m=self._MAXIMUM_GOAL_SPREAD_M,
            expected_standoff_m=float(cfg.preinsert_standoff_m),
            standoff_tolerance_m=self._STANDOFF_TOLERANCE_M,
        )

        if qualification is None:
            if not self._qualification_wait_notice_logged:
                self._qualification_wait_notice_logged = True
                log(
                    "RGB stereo track acquired; waiting for qualified "
                    "stationary physical port pose\n"
                    "  requirement: stable front-mouth point and stable "
                    "50 mm derived preinsert goal\n"
                    "  motion command: none"
                )
            return

        target_position, _ = self.ik.target.get_world_pose()
        target_position = np.asarray(target_position, dtype=np.float64)
        remaining_m = float(
            np.linalg.norm(
                qualification.tool_goal_position_m - target_position
            )
        )
        if remaining_m > self._MAXIMUM_QUALIFIED_TRAVEL_M:
            warn(
                "Qualified stationary port goal exceeds the travel kill switch.\n"
                f"  remaining translation: {remaining_m * 1000.0:.3f} mm\n"
                f"  maximum permitted: "
                f"{self._MAXIMUM_QUALIFIED_TRAVEL_M * 1000.0:.3f} mm"
            )
            return

        self._frozen_port_point_world_m = (
            qualification.opening_position_m.copy()
        )
        self._handoff_goal_world_m = (
            qualification.tool_goal_position_m.copy()
        )
        self._handoff_active = True
        self._handoff_step_index = 0

        state.consecutive_misses = 0
        state.aligned_capture_count = 0
        state.visual_aligned = False
        state.settled_frame_count = 0
        state.settle_timeout_reported = False

        log(
            "RGB STATIONARY PORT POSE QUALIFIED\n"
            f"  accepted samples: {qualification.sample_count}\n"
            f"  frozen physical opening: "
            f"{np.round(qualification.opening_position_m, 6).tolist()}\n"
            f"  opening spread: "
            f"{qualification.opening_spread_m * 1000.0:.3f} mm\n"
            f"  frozen ToolCenter preinsert goal: "
            f"{np.round(qualification.tool_goal_position_m, 6).tolist()}\n"
            f"  goal spread: "
            f"{qualification.goal_spread_m * 1000.0:.3f} mm\n"
            f"  opening-to-goal standoff: "
            f"{qualification.standoff_m * 1000.0:.3f} mm\n"
            f"  remaining translation: {remaining_m * 1000.0:.3f} mm\n"
            "  destination: frozen physical port pose\n"
            "  camera: disabled after qualification\n"
            "  orientation: unchanged horizontal plug"
        )

        self._advance_handoff_if_settled()

    def note_perception_failure(self) -> None:
        state = self.visual_servo
        cfg = self.cfg.visual_servo

        if state.complete or self._handoff_active:
            return

        state.consecutive_misses += 1
        state.aligned_capture_count = 0
        state.visual_aligned = False
        state.settled_frame_count = 0
        state.settle_timeout_reported = False

        if state.consecutive_misses < cfg.max_consecutive_misses:
            return

        state.left_reference = None
        state.right_reference = None
        state.consecutive_misses = 0

        if not self._sparse_acquisition_notice_logged:
            self._sparse_acquisition_notice_logged = True
            log(
                "RGB stereo detector references reset while preserving "
                "accepted stationary 3D evidence\n"
                f"  physical opening samples retained: "
                f"{len(self._stationary_opening_candidates)}\n"
                f"  preinsert-goal samples retained: "
                f"{len(self._stationary_tool_goal_candidates)}\n"
                "  robot target: unchanged"
            )

    def capture_due(self) -> bool:
        if (
            self._handoff_active
            and not self.visual_servo.visual_aligned
        ):
            self._advance_handoff_if_settled()
            return False

        return super().capture_due()

    def _advance_handoff_if_settled(self) -> None:
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

        tracking_error_m = self._tool_target_position_error_m()
        if tracking_error_m > cfg.target_settle_tolerance_m:
            return

        target_position, target_orientation = self.ik.target.get_world_pose()
        target_position = np.asarray(
            target_position,
            dtype=np.float64,
        )

        maximum_step_m = min(
            float(cfg.max_target_step_m),
            self._MAXIMUM_HANDOFF_STEP_M,
        )
        step_world_m, remaining_m = bounded_step_to_goal(
            target_position,
            goal,
            maximum_step_m=maximum_step_m,
        )

        completion_radius_m = min(
            float(cfg.settle_position_tolerance_m),
            0.0003,
        )
        if remaining_m <= completion_radius_m:
            state.visual_aligned = True
            state.settled_frame_count = 0
            state.settle_start_frame = self.frame_index
            state.settle_timeout_reported = False

            log(
                "RGB QUALIFIED KINEMATIC TARGET REACHED\n"
                f"  frozen physical opening: "
                f"{np.round(self._frozen_port_point_world_m, 6).tolist()}\n"
                f"  frozen ToolCenter goal: {np.round(goal, 6).tolist()}\n"
                f"  remaining target error: "
                f"{remaining_m * 1000.0:.3f} mm\n"
                "  next action: verify physical ToolCenter settling"
            )
            return

        self._handoff_step_index += 1
        self.ik.target.set_world_pose(
            position=target_position + step_world_m,
            orientation=target_orientation,
        )

        log(
            "RGB QUALIFIED KINEMATIC STEP\n"
            f"  step: {self._handoff_step_index}\n"
            f"  commanded translation: "
            f"{np.linalg.norm(step_world_m) * 1000.0:.3f} mm\n"
            f"  remaining before step: {remaining_m * 1000.0:.3f} mm"
        )

    def update_visual_servo_completion(self) -> None:
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

        self._update_actual_tool_frame(self.ik)
        target_position, _ = self.ik.target.get_world_pose()
        actual_position, _ = self.ik.actual_tool.get_world_pose()
        target_position = np.asarray(target_position, dtype=np.float64)
        actual_position = np.asarray(actual_position, dtype=np.float64)
        position_error_m = self._tool_target_position_error_m()

        state.settled_frame_count = update_convergence_counter(
            position_error_m=position_error_m,
            tolerance_m=cfg.settle_position_tolerance_m,
            current_count=state.settled_frame_count,
        )

        if state.settled_frame_count >= cfg.required_settled_frames:
            state.complete = True
            log(
                "RGB QUALIFIED PORT-POSE ALIGNMENT COMPLETE\n"
                f"  final ToolCenter target: "
                f"{np.round(target_position, 6).tolist()}\n"
                f"  actual ToolCenter: "
                f"{np.round(actual_position, 6).tolist()}\n"
                f"  physical tracking error: "
                f"{position_error_m * 1000.0:.3f} mm\n"
                f"  settled frames: {state.settled_frame_count}/"
                f"{cfg.required_settled_frames}\n"
                "  next action: begin guarded two-stage port entry"
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
            self.frame_index - state.settle_start_frame >= timeout_frames
            and not state.settle_timeout_reported
        ):
            state.settle_timeout_reported = True
            warn(
                "Qualified kinematic target has not physically settled.\n"
                f"  current physical error: "
                f"{position_error_m * 1000.0:.3f} mm\n"
                f"  required: "
                f"{cfg.settle_position_tolerance_m * 1000.0:.3f} mm"
            )
