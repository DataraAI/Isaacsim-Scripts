#!/usr/bin/env python3
"""Cable runtime with a pitched Franka hand and horizontal RJ45 plug."""

from __future__ import annotations

from dataclasses import replace
import math

import numpy as np
from pxr import UsdPhysics

from robot.angled_grasp_centering import (
    RearCenteredGraspCalibration,
    recenter_horizontal_plug_rear_in_pitched_hand,
)
from robot.angled_hand_config import ANGLED_HAND_CONFIG, AngledHandConfig
from runtime.cable_runtime import CableMountedSimulationRuntime
from robot.hand_plug_geometry import (
    AngledHandPose,
    HandPlugGeometryMetrics,
    compute_angled_hand_pose_from_fixed_grasp,
    compute_angled_hand_pose_preserving_tool,
    measure_hand_plug_geometry,
    validate_downward_hand_pitch_deg,
)
from robot.host_array_bridge import to_numpy_cpu
from control.plug_axis_insertion import ExplicitInsertionAxisAdapter
from sim import (
    log,
    matrix_to_quaternion_wxyz,
    _rotate_z,
    quaternion_wxyz_to_matrix,
)


class AngledHandCableRuntime(CableMountedSimulationRuntime):
    """Pitch panda_hand while preserving the horizontal RJ45 control frame."""

    def __init__(
        self,
        simulation_app,
        cfg,
        angled_cfg: AngledHandConfig = ANGLED_HAND_CONFIG,
    ):
        if not cfg.ik.use_fixed_start_pose:
            raise ValueError(
                "Angled hand geometry requires ik.use_fixed_start_pose=True"
            )

        pitch_deg = validate_downward_hand_pitch_deg(
            angled_cfg.hand_downward_pitch_deg,
            maximum_deg=angled_cfg.maximum_supported_pitch_deg,
        )
        original_base_hand_position = np.asarray(
            cfg.ik.initial_position,
            dtype=np.float64,
        )
        base_hand_rotation = quaternion_wxyz_to_matrix(
            np.asarray(
                cfg.ik.initial_orientation_wxyz,
                dtype=np.float64,
            )
        )

        robot_position = np.asarray(cfg.scene.franka_position, dtype=np.float64)
        robot_yaw_rad = math.radians(cfg.scene.franka_yaw_deg)
        reference_position = np.asarray(
            cfg.scene.reference_franka_position, dtype=np.float64
        )
        reference_yaw_rad = math.radians(cfg.scene.reference_franka_yaw_deg)
        delta_yaw_rad = robot_yaw_rad - reference_yaw_rad

        relative_position = original_base_hand_position - reference_position
        rotated_position = _rotate_z(relative_position, delta_yaw_rad)
        original_base_hand_position = robot_position + rotated_position

        cos_delta = math.cos(delta_yaw_rad)
        sin_delta = math.sin(delta_yaw_rad)
        yaw_correction_matrix = np.array(
            [
                [cos_delta, -sin_delta, 0.0],
                [sin_delta, cos_delta, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        base_hand_rotation = yaw_correction_matrix @ base_hand_rotation

        print(
            f"[DEBUG] yaw correction: robot_yaw_deg={cfg.scene.franka_yaw_deg} "
            f"reference_yaw_deg={cfg.scene.reference_franka_yaw_deg} "
            f"delta_yaw_rad={delta_yaw_rad:.6f} "
            f"robot_position={robot_position.tolist()} "
            f"reference_position={reference_position.tolist()} "
            f"corrected_base_hand_position={original_base_hand_position.tolist()} "
            f"corrected_base_hand_orientation_wxyz="
            f"{matrix_to_quaternion_wxyz(base_hand_rotation).tolist()}",
            flush=True,
        )

        base_hand_from_tool = quaternion_wxyz_to_matrix(
            np.asarray(
                cfg.ik.tool_center_local_orientation_wxyz,
                dtype=np.float64,
            )
        )
        camera_positions_hand = np.asarray(
            [
                cfg.camera.left_local_position,
                cfg.camera.right_local_position,
                cfg.camera.virtual_local_position,
            ],
            dtype=np.float64,
        )
        grasp_calibration = recenter_horizontal_plug_rear_in_pitched_hand(
            base_hand_position_m=original_base_hand_position,
            base_hand_rotation_world=base_hand_rotation,
            tool_position_hand_m=np.asarray(
                cfg.ik.tool_center_local_position_m,
                dtype=np.float64,
            ),
            camera_positions_hand_m=camera_positions_hand,
            plug_body_length_m=angled_cfg.plug_body_length_m,
            downward_pitch_deg=pitch_deg,
        )
        base_hand_position = grasp_calibration.base_hand_position_world_m
        tool_position_hand = grasp_calibration.tool_position_hand_m

        angled_pose = compute_angled_hand_pose_preserving_tool(
            base_hand_position_m=base_hand_position,
            base_hand_rotation_world=base_hand_rotation,
            base_hand_from_tool_rotation=base_hand_from_tool,
            tool_position_hand_m=tool_position_hand,
            downward_pitch_deg=pitch_deg,
        )
        angled_hand_orientation = tuple(
            float(value)
            for value in matrix_to_quaternion_wxyz(
                angled_pose.hand_rotation_world
            )
        )
        angled_hand_from_tool_orientation = tuple(
            float(value)
            for value in matrix_to_quaternion_wxyz(
                angled_pose.hand_from_tool_rotation
            )
        )
        left_camera_position = tuple(
            float(value)
            for value in grasp_calibration.camera_positions_hand_m[0]
        )
        right_camera_position = tuple(
            float(value)
            for value in grasp_calibration.camera_positions_hand_m[1]
        )
        virtual_camera_position = tuple(
            float(value)
            for value in grasp_calibration.camera_positions_hand_m[2]
        )
        angled_cfg_runtime = replace(
            cfg,
            camera=replace(
                cfg.camera,
                left_local_position=left_camera_position,
                right_local_position=right_camera_position,
                virtual_local_position=virtual_camera_position,
            ),
            ik=replace(
                cfg.ik,
                initial_position=tuple(
                    float(value)
                    for value in angled_pose.hand_position_world_m
                ),
                initial_orientation_wxyz=angled_hand_orientation,
                tool_center_local_position_m=tuple(
                    float(value) for value in tool_position_hand
                ),
                tool_center_local_orientation_wxyz=(
                    angled_hand_from_tool_orientation
                ),
            ),
        )

        self._angled_cfg = angled_cfg
        self._configured_hand_pitch_deg = pitch_deg
        self._ik_recompute_base_hand_rotation = base_hand_rotation
        self._angled_pose: AngledHandPose = angled_pose
        self._grasp_calibration: RearCenteredGraspCalibration = (
            grasp_calibration
        )
        self._geometry_success_logged = False
        super().__init__(
            simulation_app=simulation_app,
            cfg=angled_cfg_runtime,
        )

        self._insertion_axis_adapter = ExplicitInsertionAxisAdapter(
            self.partial_insertion
        )
        log(
            "ANGLED HAND RUNTIME ACTIVE\n"
            f"  configured hand-to-plug pitch: {pitch_deg:.3f} deg\n"
            "  view convention: robot right side\n"
            "  requested geometry: wrist higher, fingertips lower\n"
            "  grasp presentation: RJ45 rear centered between fingers\n"
            f"  local rear-centering shift: "
            f"{grasp_calibration.local_shift_hand_m[0] * 1000.0:.3f} mm\n"
            "  camera-to-plug calibration: preserved exactly\n"
            f"  preserved plug-tip target: "
            f"{np.round(angled_pose.tool_position_world_m, 6).tolist()}\n"
            f"  solved hand target: "
            f"{np.round(angled_pose.hand_position_world_m, 6).tolist()}\n"
            "  control frame: unchanged horizontal RJ45 plug tip\n"
            "  insertion frame: live PhysX plug nose axis"
        )
        print(
            f"[DEBUG] exact hand target for esha carry match: "
            f"position={np.round(angled_pose.hand_position_world_m, 6).tolist()} "
            f"orientation_wxyz={list(angled_hand_orientation)}",
            flush=True,
        )

    def _maybe_recompute_ik_target(self) -> None:
        if not self.cfg.cable_mount.already_grasped_by_pickup_pipeline:
            return
        if self.cable_mount is None or self.cable_mount.plug_frame is None:
            raise RuntimeError(
                "Merged IK target recompute requires cable_mount.plug_frame "
                "from author_from_existing_grasp()"
            )
        stage = self.cable_mount.stage
        if stage is None:
            raise RuntimeError(
                "Merged IK target recompute requires cable_mount.stage"
            )

        joint_prim = stage.GetPrimAtPath(
            self.cfg.cable_mount.fixed_joint_path
        )
        if not joint_prim.IsValid() or not joint_prim.IsA(
            UsdPhysics.FixedJoint
        ):
            raise RuntimeError(
                "Merged grasp fixed joint is missing or invalid: "
                f"{self.cfg.cable_mount.fixed_joint_path}"
            )

        joint = UsdPhysics.FixedJoint(joint_prim)
        local_pos0 = joint.GetLocalPos0Attr().Get()
        local_rot0 = joint.GetLocalRot0Attr().Get()
        local_pos1 = joint.GetLocalPos1Attr().Get()
        local_rot1 = joint.GetLocalRot1Attr().Get()
        if local_pos0 is None or local_rot0 is None:
            raise RuntimeError(
                "Merged grasp fixed joint is missing LocalPos0/LocalRot0: "
                f"{self.cfg.cable_mount.fixed_joint_path}"
            )

        def _quat_report(label: str, quat) -> str:
            if quat is None:
                return f"{label}=None (schema-default / unset)"
            imag = quat.GetImaginary()
            wxyz = [
                float(quat.GetReal()),
                float(imag[0]),
                float(imag[1]),
                float(imag[2]),
            ]
            # Identity in USD Quatf is (1, 0, 0, 0) real-first.
            is_identity = (
                abs(wxyz[0] - 1.0) < 1.0e-6
                and abs(wxyz[1]) < 1.0e-6
                and abs(wxyz[2]) < 1.0e-6
                and abs(wxyz[3]) < 1.0e-6
            )
            return (
                f"{label}_wxyz={np.round(wxyz, 8).tolist()} "
                f"identity={is_identity}"
            )

        pos1_report = (
            "None (unset)"
            if local_pos1 is None
            else np.round(
                [float(local_pos1[0]), float(local_pos1[1]), float(local_pos1[2])],
                8,
            ).tolist()
        )
        print(
            "[DEBUG] FixedJoint local frames (hypothesis LocalRot1):\n"
            f"  path={self.cfg.cable_mount.fixed_joint_path}\n"
            f"  body0={ [str(p) for p in joint.GetBody0Rel().GetTargets()] }\n"
            f"  body1={ [str(p) for p in joint.GetBody1Rel().GetTargets()] }\n"
            f"  LocalPos0={np.round([float(local_pos0[0]), float(local_pos0[1]), float(local_pos0[2])], 8).tolist()}\n"
            f"  {_quat_report('LocalRot0', local_rot0)}\n"
            f"  LocalPos1={pos1_report}\n"
            f"  {_quat_report('LocalRot1', local_rot1)}",
            flush=True,
        )

        rotation_imag = local_rot0.GetImaginary()
        hand_from_plug = np.eye(4, dtype=np.float64)
        hand_from_plug[:3, :3] = quaternion_wxyz_to_matrix(
            np.asarray(
                [
                    float(local_rot0.GetReal()),
                    float(rotation_imag[0]),
                    float(rotation_imag[1]),
                    float(rotation_imag[2]),
                ],
                dtype=np.float64,
            )
        )
        hand_from_plug[:3, 3] = np.asarray(
            [float(local_pos0[0]), float(local_pos0[1]), float(local_pos0[2])],
            dtype=np.float64,
        )

        plug_from_tip = np.asarray(
            self.cable_mount.plug_frame.plug_from_tip,
            dtype=np.float64,
        )
        nose_axis_local = np.asarray(
            self.cable_mount.plug_frame.nose_axis_local,
            dtype=np.float64,
        )
        tip_local = np.asarray(
            self.cable_mount.plug_frame.tip_local_m,
            dtype=np.float64,
        )
        plug_from_tip_z = plug_from_tip[:3, 2]
        print(
            "[DEBUG] plug_frame local-axis convention (hypothesis nose_axis):\n"
            f"  tracked_plug={self.cfg.cable_mount.tracked_plug_path}\n"
            f"  tip_local_m={np.round(tip_local, 8).tolist()}\n"
            f"  nose_axis_local={np.round(nose_axis_local, 8).tolist()}\n"
            f"  plug_from_tip[:,2] (hook tip/nose Z)="
            f"{np.round(plug_from_tip_z, 8).tolist()}\n"
            f"  nose_axis == plug_from_tip[:,2]? "
            f"{bool(np.allclose(nose_axis_local, plug_from_tip_z, atol=1e-9))}\n"
            f"  plug_from_tip translation={np.round(plug_from_tip[:3, 3], 8).tolist()}",
            flush=True,
        )

        # Freeze rigid grasp transforms for per-physics-step reorient curve.
        self._frozen_hand_from_plug = hand_from_plug.copy()
        self._frozen_plug_from_tip = plug_from_tip.copy()

        # already_grasped: FixedJoint relative is physical and immutable.
        # Invert through it instead of inventing a new hand_from_tool.
        angled_pose = compute_angled_hand_pose_from_fixed_grasp(
            base_hand_position_m=(
                self._grasp_calibration.base_hand_position_world_m
            ),
            base_hand_rotation_world=self._ik_recompute_base_hand_rotation,
            hand_from_plug_frozen=hand_from_plug,
            plug_from_tip=plug_from_tip,
            downward_pitch_deg=self._configured_hand_pitch_deg,
            pitch_tolerance_deg=self._angled_cfg.pitch_tolerance_deg,
        )
        corrected_orientation = tuple(
            float(value)
            for value in matrix_to_quaternion_wxyz(
                angled_pose.hand_rotation_world
            )
        )
        self.cfg = replace(
            self.cfg,
            ik=replace(
                self.cfg.ik,
                initial_position=tuple(
                    float(value)
                    for value in angled_pose.hand_position_world_m
                ),
                initial_orientation_wxyz=corrected_orientation,
            ),
        )
        self._angled_pose = angled_pose

        plug_axis = angled_pose.tool_rotation_world[:, 2]
        metrics = measure_hand_plug_geometry(
            hand_position_m=angled_pose.hand_position_world_m,
            hand_rotation_world=angled_pose.hand_rotation_world,
            plug_tip_position_m=angled_pose.tool_position_world_m,
            plug_axis_world=plug_axis,
        )
        print(
            "[DEBUG] merged IK target recomputed from live FixedJoint grasp\n"
            f"  fixed_joint_path={self.cfg.cable_mount.fixed_joint_path}\n"
            f"  corrected initial_position="
            f"{list(self.cfg.ik.initial_position)}\n"
            f"  corrected initial_orientation_wxyz="
            f"{list(self.cfg.ik.initial_orientation_wxyz)}\n"
            f"  recomputed plug_axis={np.round(plug_axis, 6).tolist()}\n"
            f"  plug_axis[2]={plug_axis[2]:.6f}\n"
            f"  wrist_higher_fingertips_lower="
            f"{metrics.wrist_higher_fingertips_lower}\n"
            f"  relative_pitch_deg={metrics.relative_pitch_deg:.6f}",
            flush=True,
        )

    def _live_plug_tip_and_axis(self) -> tuple[np.ndarray, np.ndarray]:
        tip_world, plug_axis = self._physx_plug_tip_and_axis()
        self._plug_axis_sample_count = (
            getattr(self, "_plug_axis_sample_count", 0) + 1
        )
        count = self._plug_axis_sample_count
        if count <= 12 or count % 5 == 0:
            usd_sample = self._sample_tracked_plug_tip_and_axis_from_stage()
            usd_axis = (
                None
                if usd_sample is None
                else np.round(usd_sample[1], 6).tolist()
            )
            print(
                f"[DEBUG] live _live_plug_tip_and_axis call={count}: "
                f"tip={np.round(tip_world, 6).tolist()} "
                f"plug_axis={np.round(plug_axis, 6).tolist()} "
                f"plug_axis[2]={plug_axis[2]:.6f} "
                f"usd_plug_axis={usd_axis}",
                flush=True,
            )
        return tip_world, plug_axis

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
            metrics.palm_roll_error_deg
            > self._angled_cfg.palm_side_tolerance_deg
        ):
            raise RuntimeError(
                "palm side does not match the previous working pose: "
                f"error={metrics.palm_roll_error_deg:.6f} deg"
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
        metrics = self._validate_live_hand_plug_geometry()
        if not self._geometry_success_logged:
            _, plug_axis = self._live_plug_tip_and_axis()
            hand_position, hand_orientation = (
                self._hand_pose_from_articulation()
            )
            hand_rotation = quaternion_wxyz_to_matrix(hand_orientation)
            log(
                "ANGLED HAND GEOMETRY VALIDATED\n"
                f"  measured hand-to-plug pitch: "
                f"{metrics.relative_pitch_deg:.6f} deg\n"
                f"  wrist above plug tip: "
                f"{metrics.wrist_above_tip_m * 1000.0:.3f} mm\n"
                f"  requested pitch sign valid: "
                f"{metrics.wrist_higher_fingertips_lower}\n"
                f"  palm side error: "
                f"{metrics.palm_roll_error_deg:.6f} deg\n"
                f"  hand forward axis: "
                f"{np.round(hand_rotation[:, 2], 6).tolist()}\n"
                f"  palm side axis: "
                f"{np.round(hand_rotation[:, 0], 6).tolist()}\n"
                f"  plug insertion axis: "
                f"{np.round(plug_axis, 6).tolist()}\n"
                f"  plug horizontal error: "
                f"{metrics.plug_horizontal_error_deg:.6f} deg\n"
                f"  plug-tip error: {tip_error_m * 1000.0:.6f} mm\n"
                f"  plug-axis error: {axis_error_deg:.6f} deg"
            )
            if self.run_logger is not None:
                self.run_logger.log_event(
                    t=self.frame_index,
                    event="ANGLED_HAND_GEOMETRY_VALIDATED",
                    relative_pitch_deg=metrics.relative_pitch_deg,
                    wrist_above_tip_mm=metrics.wrist_above_tip_m * 1000.0,
                    palm_roll_error_deg=metrics.palm_roll_error_deg,
                    plug_horizontal_error_deg=(
                        metrics.plug_horizontal_error_deg
                    ),
                    plug_tip_error_mm=tip_error_m * 1000.0,
                    plug_axis_error_deg=axis_error_deg,
                )
            self._geometry_success_logged = True
        return tip_error_m, axis_error_deg

    @staticmethod
    def _rotation_angle_deg(rotation_a: np.ndarray, rotation_b: np.ndarray) -> float:
        """Geodesic angle (deg) between two SO(3) matrices."""

        relative = np.asarray(rotation_a, dtype=np.float64).T @ np.asarray(
            rotation_b, dtype=np.float64
        )
        cos_theta = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
        return float(math.degrees(math.acos(cos_theta)))

    def _sample_reorient_orient_error(self) -> None:
        """One physics-step sample: FixedJoint prediction vs live PhysX plug."""

        if (
            getattr(self, "_frozen_hand_from_plug", None) is None
            or getattr(self, "_frozen_plug_from_tip", None) is None
            or self.ik is None
            or not hasattr(self, "_tracked_plug_body")
        ):
            return

        hand_position, hand_orientation = self._hand_pose_from_articulation()
        hand_rotation = quaternion_wxyz_to_matrix(hand_orientation)
        plug_position, plug_orientation = self._tracked_plug_body.get_world_pose()
        plug_quat = to_numpy_cpu(
            plug_orientation,
            shape=(4,),
            label="reorient curve plug quat",
        )
        plug_pos = to_numpy_cpu(
            plug_position,
            shape=(3,),
            label="reorient curve plug pos",
        )
        actual_plug_rotation = quaternion_wxyz_to_matrix(plug_quat)

        hand_from_plug_R = self._frozen_hand_from_plug[:3, :3]
        plug_from_tip_R = self._frozen_plug_from_tip[:3, :3]
        # Body-frame FixedJoint prediction (what the weld should enforce).
        predicted_plug_rotation = hand_rotation @ hand_from_plug_R
        # Tip-frame prediction/actual (user-requested composition).
        predicted_tip_rotation = predicted_plug_rotation @ plug_from_tip_R
        actual_tip_rotation = actual_plug_rotation @ plug_from_tip_R

        body_err_deg = self._rotation_angle_deg(
            predicted_plug_rotation, actual_plug_rotation
        )
        tip_err_deg = self._rotation_angle_deg(
            predicted_tip_rotation, actual_tip_rotation
        )
        nose_local = self._frozen_plug_from_tip[:3, 2]
        pred_nose = predicted_plug_rotation @ nose_local
        act_nose = actual_plug_rotation @ nose_local
        pred_nose = pred_nose / max(float(np.linalg.norm(pred_nose)), 1e-12)
        act_nose = act_nose / max(float(np.linalg.norm(act_nose)), 1e-12)
        nose_err_deg = float(
            math.degrees(
                math.acos(
                    float(np.clip(np.dot(pred_nose, act_nose), -1.0, 1.0))
                )
            )
        )

        sample = {
            "frame": int(self.frame_index),
            "body_err_deg": body_err_deg,
            "tip_err_deg": tip_err_deg,
            "nose_err_deg": nose_err_deg,
            "hand_forward": hand_rotation[:, 2].copy(),
            "pred_nose": pred_nose.copy(),
            "act_nose": act_nose.copy(),
            "plug_pos": plug_pos.copy(),
        }
        self._reorient_curve_samples.append(sample)
        n = len(self._reorient_curve_samples)
        # Log densely for first 40 steps, then every 10th, always last-ish via summary.
        if n <= 40 or n % 10 == 0:
            print(
                f"[DEBUG] reorient curve step={n} frame={self.frame_index}: "
                f"body_err_deg={body_err_deg:.3f} "
                f"tip_err_deg={tip_err_deg:.3f} "
                f"nose_err_deg={nose_err_deg:.3f} "
                f"pred_nose={np.round(pred_nose, 4).tolist()} "
                f"act_nose={np.round(act_nose, 4).tolist()} "
                f"plug_pos={np.round(plug_pos, 4).tolist()}",
                flush=True,
            )

    def _summarize_reorient_curve(self) -> None:
        samples = getattr(self, "_reorient_curve_samples", None) or []
        if not samples:
            print("[DEBUG] reorient curve: no samples collected", flush=True)
            return
        errs = np.asarray([s["body_err_deg"] for s in samples], dtype=np.float64)
        n = len(errs)
        # Shape heuristics for the report.
        early = float(np.mean(errs[: min(10, n)]))
        late = float(np.mean(errs[max(0, n - 10) :]))
        mid = float(np.mean(errs[n // 3 : (2 * n) // 3])) if n >= 9 else float(np.mean(errs))
        # Monotonic decrease: allow small noise via overall early->late drop.
        dropped = early - late
        flat_late = float(np.std(errs[max(0, n - 20) :])) if n >= 5 else float(np.std(errs))
        if dropped > 5.0 and late < 5.0:
            shape = "MONOTONICALLY_DECREASING_TOWARD_ZERO"
        elif late > 15.0 and abs(mid - late) < 5.0 and flat_late < 5.0:
            shape = "FLAT_NONZERO_STEADY_STATE"
        elif float(np.std(np.diff(errs))) > 10.0 and late > 10.0:
            shape = "OSCILLATING_OR_NOISY"
        else:
            shape = "OTHER"
        print(
            "[DEBUG] reorient curve SUMMARY:\n"
            f"  samples={n}\n"
            f"  body_err_deg first={errs[0]:.3f} min={errs.min():.3f} "
            f"max={errs.max():.3f} last={errs[-1]:.3f}\n"
            f"  early_mean10={early:.3f} mid_mean={mid:.3f} late_mean10={late:.3f}\n"
            f"  early_minus_late={dropped:.3f} late_std20={flat_late:.3f}\n"
            f"  shape_class={shape}\n"
            f"  tip_err first/last="
            f"{samples[0]['tip_err_deg']:.3f}/{samples[-1]['tip_err_deg']:.3f} "
            f"(should match body_err if same PfT on both sides)\n"
            f"  nose_err first/last="
            f"{samples[0]['nose_err_deg']:.3f}/{samples[-1]['nose_err_deg']:.3f}",
            flush=True,
        )
        # Compact sparkline of every 5th sample for the log.
        stride = max(1, n // 40)
        series = [
            f"{errs[i]:.1f}" for i in range(0, n, stride)
        ]
        print(
            f"[DEBUG] reorient curve body_err_deg series (stride={stride}): "
            f"{series}",
            flush=True,
        )

    def step(self) -> None:
        super().step()
        if not self.cfg.cable_mount.already_grasped_by_pickup_pipeline:
            return
        if getattr(self, "_frozen_hand_from_plug", None) is None:
            return
        if self.ik is None or not hasattr(self, "_tracked_plug_body"):
            return
        if getattr(self, "_reorient_curve_done", False):
            return
        if not hasattr(self, "_reorient_curve_samples") or self._reorient_curve_samples is None:
            self._reorient_curve_samples = []
            print(
                "[DEBUG] reorient curve: logging predicted-vs-actual plug "
                "orientation on EVERY physics step() during IK reorient "
                "(prepare_for_perception window)",
                flush=True,
            )
        self._sample_reorient_orient_error()
        # Cover initial settle + validation budget (+buffer). Settled handoff
        # overrides prepare_for_perception, so summarize from step() itself.
        max_n = (
            int(self.cfg.cable_mount.initial_settle_frames)
            + int(self.cfg.cable_mount.validation_frames)
            + 50
        )
        if len(self._reorient_curve_samples) >= max_n:
            self._summarize_reorient_curve()
            self._reorient_curve_done = True

    def prepare_for_perception(self) -> None:
        # Note: production MRO uses settled_stereo_handoff_runtime's override,
        # so this may not run. Per-step logging is armed from step() instead.
        super().prepare_for_perception()

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
            hand_position, hand_orientation = (
                self._hand_pose_from_articulation()
            )
            hand_rotation = quaternion_wxyz_to_matrix(hand_orientation)
            hand_axis = hand_rotation[:, 2]
            palm_side_axis = hand_rotation[:, 0]
            geometry_status = (
                "ANGLED HAND GEOMETRY\n"
                f"  configured hand pitch: "
                f"{self._configured_hand_pitch_deg:.3f} deg\n"
                f"  solved hand target: "
                f"{np.round(self._angled_pose.hand_position_world_m, 6).tolist()}\n"
                f"  live hand position: "
                f"{np.round(hand_position, 6).tolist()}\n"
                f"  measured hand-to-plug pitch: "
                f"{metrics.relative_pitch_deg:.6f} deg\n"
                f"  wrist above plug tip: "
                f"{metrics.wrist_above_tip_m * 1000.0:.3f} mm\n"
                f"  requested pitch sign valid: "
                f"{metrics.wrist_higher_fingertips_lower}\n"
                f"  palm side error: "
                f"{metrics.palm_roll_error_deg:.6f} deg\n"
                f"  hand forward axis: "
                f"{np.round(hand_axis, 6).tolist()}\n"
                f"  palm side axis: "
                f"{np.round(palm_side_axis, 6).tolist()}\n"
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
