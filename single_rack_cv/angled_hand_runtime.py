#!/usr/bin/env python3
"""Cable runtime with a pitched Franka hand and horizontal RJ45 plug."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from angled_hand_config import ANGLED_HAND_CONFIG, AngledHandConfig
from cable_runtime import CableMountedSimulationRuntime
from hand_plug_geometry import (
    HandPlugGeometryMetrics,
    compute_pitched_hand_from_tool_rotation,
    measure_hand_plug_geometry,
    validate_downward_hand_pitch_deg,
)
from host_array_bridge import to_numpy_cpu
from plug_axis_insertion import ExplicitInsertionAxisAdapter
from sim import (
    log,
    matrix_to_quaternion_wxyz,
    quaternion_wxyz_to_matrix,
)


class AngledHandCableRuntime(CableMountedSimulationRuntime):
    """
    Hold the plug-tip frame horizontal while pitching panda_hand downward.

    `/World/IK_Target` and `/World/ToolCenter` remain the plug-tip control
    frame. The changed hand_T_tool rotation makes Lula command the hand at the
    requested pitch. Insertion freezes the live PhysX plug nose axis rather
    than assuming ToolCenter local +Z is the insertion direction.
    """

    def __init__(
        self,
        simulation_app,
        cfg,
        angled_cfg: AngledHandConfig = ANGLED_HAND_CONFIG,
    ):
        pitch_deg = validate_downward_hand_pitch_deg(
            angled_cfg.hand_downward_pitch_deg,
            maximum_deg=angled_cfg.maximum_supported_pitch_deg,
        )
        base_hand_from_tool = quaternion_wxyz_to_matrix(
            np.asarray(
                cfg.ik.tool_center_local_orientation_wxyz,
                dtype=np.float64,
            )
        )
        pitched_hand_from_tool = compute_pitched_hand_from_tool_rotation(
            base_hand_from_tool,
            pitch_deg,
        )
        pitched_orientation = tuple(
            float(value)
            for value in matrix_to_quaternion_wxyz(
                pitched_hand_from_tool
            )
        )
        pitched_cfg = replace(
            cfg,
            ik=replace(
                cfg.ik,
                tool_center_local_orientation_wxyz=pitched_orientation,
            ),
        )

        self._angled_cfg = angled_cfg
        self._configured_hand_pitch_deg = pitch_deg
        super().__init__(simulation_app=simulation_app, cfg=pitched_cfg)

        self._insertion_axis_adapter = ExplicitInsertionAxisAdapter(
            self.partial_insertion
        )
        log(
            "ANGLED HAND RUNTIME ACTIVE\n"
            f"  configured hand-to-plug pitch: {pitch_deg:.3f} deg\n"
            "  view convention: robot right side\n"
            "  requested geometry: wrist higher, fingertips lower\n"
            "  control frame: horizontal RJ45 plug tip\n"
            "  insertion frame: live PhysX plug nose axis"
        )

    def _live_plug_tip_and_axis(self) -> tuple[np.ndarray, np.ndarray]:
        if self.cable_mount is None:
            raise RuntimeError("Cable mount is unavailable")
        plug_frame = self.cable_mount.plug_frame
        if plug_frame is None:
            raise RuntimeError("Tracked plug frame is unavailable")

        plug_position, plug_orientation = (
            self._tracked_plug_body.get_world_pose()
        )
        plug_scale = self._tracked_plug_body.get_world_scale()
        position = to_numpy_cpu(
            plug_position,
            shape=(3,),
            label="tracked RJ45 live position",
        )
        orientation = to_numpy_cpu(
            plug_orientation,
            shape=(4,),
            label="tracked RJ45 live orientation",
        )
        scale = to_numpy_cpu(
            plug_scale,
            shape=(3,),
            label="tracked RJ45 world scale",
        )

        world_from_plug = np.eye(4, dtype=np.float64)
        world_from_plug[:3, :3] = (
            quaternion_wxyz_to_matrix(orientation) @ np.diag(scale)
        )
        world_from_plug[:3, 3] = position
        tip_world = (
            world_from_plug @ np.r_[plug_frame.tip_local_m, 1.0]
        )[:3]
        nose_world = (
            world_from_plug[:3, :3] @ plug_frame.nose_axis_local
        )
        nose_norm = float(np.linalg.norm(nose_world))
        if nose_norm <= 1.0e-12:
            raise RuntimeError("Live plug nose axis has zero length")
        return tip_world, nose_world / nose_norm

    def _live_hand_plug_geometry(self) -> HandPlugGeometryMetrics:
        plug_tip, plug_axis = self._live_plug_tip_and_axis()
        hand_position, hand_orientation = (
            self._hand_pose_from_articulation()
        )
        return measure_hand_plug_geometry(
            hand_position_m=hand_position,
            hand_rotation_world=quaternion_wxyz_to_matrix(
                hand_orientation
            ),
            plug_tip_position_m=plug_tip,
            plug_axis_world=plug_axis,
        )

    def _validate_live_hand_plug_geometry(
        self,
    ) -> HandPlugGeometryMetrics:
        metrics = self._live_hand_plug_geometry()
        pitch_error_deg = abs(
            metrics.relative_pitch_deg
            - self._configured_hand_pitch_deg
        )
        if pitch_error_deg > self._angled_cfg.pitch_tolerance_deg:
            raise RuntimeError(
                "hand-to-plug pitch error exceeded limit: "
                f"measured={metrics.relative_pitch_deg:.6f} deg, "
                f"configured={self._configured_hand_pitch_deg:.6f} deg"
            )
        if (
            self._configured_hand_pitch_deg > 1.0e-9
            and not metrics.wrist_higher_fingertips_lower
        ):
            raise RuntimeError(
                "wrong hand pitch sign: wrist is not above the plug tip "
                "with fingertips directed downward toward the port"
            )
        if (
            metrics.plug_horizontal_error_deg
            > self.cfg.cable_mount.max_axis_error_deg
        ):
            raise RuntimeError(
                "plug horizontal error exceeded limit: "
                f"{metrics.plug_horizontal_error_deg:.6f} deg"
            )
        return metrics

    def _sample_mount_validation_live(self, runtime) -> tuple[float, float]:
        tip_error_m, axis_error_deg = (
            super()._sample_mount_validation_live(runtime)
        )
        self._validate_live_hand_plug_geometry()
        return tip_error_m, axis_error_deg

    def _partial_insertion_sample(self):
        _, plug_axis_world = self._live_plug_tip_and_axis()
        self._insertion_axis_adapter.set_axis_world(plug_axis_world)
        return super()._partial_insertion_sample()

    def _log_startup_diagnostics(
        self,
        *,
        frame_count: int,
        minimum_tool_error_m: float,
        maximum_tool_error_m: float,
        validation_sample_count: int,
    ) -> float:
        current_error_m = super()._log_startup_diagnostics(
            frame_count=frame_count,
            minimum_tool_error_m=minimum_tool_error_m,
            maximum_tool_error_m=maximum_tool_error_m,
            validation_sample_count=validation_sample_count,
        )
        try:
            metrics = self._live_hand_plug_geometry()
            _, plug_axis = self._live_plug_tip_and_axis()
            _, hand_orientation = self._hand_pose_from_articulation()
            hand_axis = quaternion_wxyz_to_matrix(
                hand_orientation
            )[:, 2]
            geometry_status = (
                "ANGLED HAND GEOMETRY\n"
                f"  configured hand pitch: "
                f"{self._configured_hand_pitch_deg:.3f} deg\n"
                f"  measured hand-to-plug pitch: "
                f"{metrics.relative_pitch_deg:.6f} deg\n"
                f"  wrist above plug tip: "
                f"{metrics.wrist_above_tip_m * 1000.0:.3f} mm\n"
                f"  requested pitch sign valid: "
                f"{metrics.wrist_higher_fingertips_lower}\n"
                f"  hand forward axis: "
                f"{np.round(hand_axis, 6).tolist()}\n"
                f"  plug insertion axis: "
                f"{np.round(plug_axis, 6).tolist()}\n"
                f"  plug horizontal error: "
                f"{metrics.plug_horizontal_error_deg:.6f} deg"
            )
        except Exception as error:
            geometry_status = (
                "ANGLED HAND GEOMETRY\n"
                f"  measurement pending/failed: "
                f"{type(error).__name__}: {error}"
            )
        log(geometry_status)
        return current_error_m
