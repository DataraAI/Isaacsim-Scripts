#!/usr/bin/env python3
"""GPU-safe cable runtime facade for Isaac Sim CUDA dynamics."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import omni.usd
from pxr import Gf, UsdGeom

from isaacsim.core.prims import SingleRigidPrim

from cable.cable_geometry import angular_error_deg
from robot.host_array_bridge import to_numpy_cpu
from control.insertion import (
    InsertionEvent,
    InsertionLimits,
    InsertionPhase,
    InsertionSample,
    PartialInsertionController,
)
from vision.perception import CameraModel
from runtime.cable_runtime_base import (
    CableMountedSimulationRuntime as _BaseCableMountedSimulationRuntime,
)
from sim import (
    hand_pose_to_tool_pose,
    log,
    matrix_to_quaternion_wxyz,
    quaternion_wxyz_to_matrix,
    tool_pose_to_hand_pose,
    update_convergence_counter,
    warn,
)


class CableMountedSimulationRuntime(_BaseCableMountedSimulationRuntime):
    """Use live PhysX/FK poses and guarded frozen-axis port entry."""

    def __init__(self, simulation_app, cfg):
        cfg = replace(
            cfg,
            visual_servo=replace(
                cfg.visual_servo,
                max_target_step_m=max(
                    float(cfg.visual_servo.max_target_step_m),
                    0.005,
                ),
                target_settle_tolerance_m=max(
                    float(cfg.visual_servo.target_settle_tolerance_m),
                    0.001,
                ),
            ),
        )
        self._last_capture_wait_log_frame = -1_000_000

        super().__init__(simulation_app=simulation_app, cfg=cfg)
        if self.cable_mount is None:
            raise RuntimeError("Cable mount was not created")

        self._tracked_plug_body = SingleRigidPrim(
            prim_path=self.cfg.cable_mount.tracked_plug_path,
            name="tracked_cable_plug_live",
        )
        self._tracked_plug_body.initialize()

        self._base_mount_sample_validation = self.cable_mount.sample_validation
        self.cable_mount.sample_validation = self._sample_mount_validation_live

        insertion_cfg = self.cfg.insertion
        step_timeout_frames = max(
            1,
            int(
                round(
                    float(insertion_cfg.step_timeout_s)
                    / float(self.cfg.scene.physics_dt)
                )
            ),
        )
        self.partial_insertion = PartialInsertionController(
            InsertionLimits(
                total_depth_m=float(insertion_cfg.total_depth_m),
                step_size_m=float(insertion_cfg.step_size_m),
                coarse_approach_depth_m=float(
                    insertion_cfg.coarse_approach_depth_m
                ),
                coarse_step_size_m=float(
                    insertion_cfg.coarse_step_size_m
                ),
                opening_depth_m=float(insertion_cfg.opening_depth_m),
                settle_tolerance_m=float(
                    insertion_cfg.settle_position_tolerance_m
                ),
                required_settled_frames=int(
                    insertion_cfg.required_settled_frames
                ),
                step_timeout_frames=step_timeout_frames,
                max_lateral_drift_m=float(
                    insertion_cfg.max_lateral_drift_m
                ),
                max_orientation_error_deg=float(
                    insertion_cfg.max_orientation_error_deg
                ),
                max_mount_tip_error_m=float(
                    self.cfg.cable_mount.max_tip_error_m
                ),
                max_mount_axis_error_deg=float(
                    self.cfg.cable_mount.max_axis_error_deg
                ),
            )
        )
        self._insertion_total_steps = (
            self.partial_insertion.limits.total_step_count
        )
        fine_travel_m = (
            float(insertion_cfg.total_depth_m)
            - float(insertion_cfg.coarse_approach_depth_m)
        )
        final_port_depth_m = (
            float(insertion_cfg.total_depth_m)
            - float(insertion_cfg.opening_depth_m)
        )

        log(
            "GPU FK + LIVE PLUG POSE BRIDGE ACTIVE\n"
            f"  visual-servo max step: "
            f"{self.cfg.visual_servo.max_target_step_m * 1000.0:.3f} mm\n"
            f"  between-capture settle tolerance: "
            f"{self.cfg.visual_servo.target_settle_tolerance_m * 1000.0:.3f} mm\n"
            "  guarded two-stage port entry:\n"
            f"    coarse approach: "
            f"{insertion_cfg.coarse_approach_depth_m * 1000.0:.1f} mm "
            f"at {insertion_cfg.coarse_step_size_m * 1000.0:.1f} mm/step\n"
            f"    fine motion: {fine_travel_m * 1000.0:.1f} mm "
            f"at {insertion_cfg.step_size_m * 1000.0:.1f} mm/step\n"
            f"    final depth inside opening: "
            f"{final_port_depth_m * 1000.0:.1f} mm\n"
            f"    total commands: {self._insertion_total_steps}"
        )

    def capture_due(self) -> bool:
        due = super().capture_due()
        if due:
            return True

        state = self.visual_servo
        cfg = self.cfg.visual_servo
        waiting_for_target = (
            state.startup_ready
            and state.acquired
            and not state.visual_aligned
            and not state.complete
        )
        log_interval_frames = 120

        if (
            waiting_for_target
            and self.frame_index - self._last_capture_wait_log_frame
            >= log_interval_frames
        ):
            self._last_capture_wait_log_frame = self.frame_index
            error_m = self._tool_target_position_error_m()
            if error_m > cfg.target_settle_tolerance_m:
                log(
                    "RGB SERVO WAITING FOR TARGET SETTLE\n"
                    f"  physical ToolCenter error: {error_m * 1000.0:.3f} mm\n"
                    f"  capture gate: "
                    f"{cfg.target_settle_tolerance_m * 1000.0:.3f} mm\n"
                    f"  commanded step: {cfg.max_target_step_m * 1000.0:.3f} mm"
                )

        return False

    def update_visual_servo_completion(self) -> None:
        state = self.visual_servo
        cfg = self.cfg.visual_servo

        if not state.visual_aligned or state.complete:
            return
        if self.ik is None:
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
                "RGB STEREO VISUAL SERVO COMPLETE\n"
                f"  final ToolCenter target: "
                f"{np.round(target_position, 6).tolist()}\n"
                f"  actual ToolCenter: "
                f"{np.round(actual_position, 6).tolist()}\n"
                f"  physical tracking error: "
                f"{position_error_m * 1000.0:.3f} mm\n"
                f"  settled frames: "
                f"{state.settled_frame_count}/"
                f"{cfg.required_settled_frames}\n"
                "  next action: begin 40 mm coarse approach, "
                "then 20 mm fine entry."
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
            self.frame_index - state.settle_start_frame
            >= timeout_frames
            and not state.settle_timeout_reported
        ):
            state.settle_timeout_reported = True
            warn(
                "RGB stereo alignment is stable, but ToolCenter has not settled "
                "within the warning timeout.\n"
                f"  current physical error: "
                f"{position_error_m * 1000.0:.3f} mm\n"
                f"  required: "
                f"{cfg.settle_position_tolerance_m * 1000.0:.3f} mm"
            )

    def update_partial_insertion(self) -> None:
        if not self.cfg.insertion.enabled:
            return
        if self.ik is None or self.cable_mount is None:
            return
        if (
            not self.visual_servo.complete
            and self.partial_insertion.phase
            is InsertionPhase.WAITING_FOR_ALIGNMENT
        ):
            return
        if self.partial_insertion.phase in (
            InsertionPhase.COMPLETE,
            InsertionPhase.ABORTED,
        ):
            return

        sample: InsertionSample | None = None
        try:
            sample = self._partial_insertion_sample()
            event = self.partial_insertion.update(sample)
        except Exception as exc:
            event = self.partial_insertion.abort(
                f"live insertion validation failed: {exc}",
                sample,
            )

        if event.command is not None:
            try:
                reachable = self._insertion_target_is_ik_reachable(
                    event.command.target_position_m,
                    event.command.target_orientation_wxyz,
                )
            except Exception as exc:
                reachable = False
                ik_reason = f"Lula IK preflight raised: {exc}"
            else:
                ik_reason = "Lula IK rejected insertion target"

            if not reachable:
                event = self.partial_insertion.abort(ik_reason, sample)
            else:
                try:
                    self.ik.target.set_world_pose(
                        position=event.command.target_position_m,
                        orientation=event.command.target_orientation_wxyz,
                    )
                except Exception as exc:
                    event = self.partial_insertion.abort(
                        f"could not publish insertion target: {exc}",
                        sample,
                    )

        if event.kind in (
            "started",
            "step_settled",
            "complete",
            "aborted",
        ):
            self._log_partial_insertion_event(event)

    def _partial_insertion_sample(self) -> InsertionSample:
        if self.ik is None or self.cable_mount is None:
            raise RuntimeError("Insertion runtime is not initialized")

        actual_position, actual_orientation = (
            self._tool_pose_from_articulation()
        )
        target_position, _ = self.ik.target.get_world_pose()
        target = np.asarray(target_position, dtype=np.float64).reshape(3)
        target_error_m = float(
            np.linalg.norm(actual_position - target)
        )
        mount_tip_error_m, mount_axis_error_deg = (
            self._sample_mount_validation_live(self)
        )

        return InsertionSample(
            frame_index=int(self.frame_index),
            alignment_complete=bool(self.visual_servo.complete),
            actual_position_m=actual_position,
            actual_orientation_wxyz=actual_orientation,
            target_error_m=target_error_m,
            mount_tip_error_m=mount_tip_error_m,
            mount_axis_error_deg=mount_axis_error_deg,
            fixed_joint_valid=self.cable_mount.fixed_joint_is_valid(),
            attachment_preserved=(
                self.cable_mount.built_in_attachment_is_preserved()
            ),
        )

    def _insertion_target_is_ik_reachable(
        self,
        tool_position_m: np.ndarray,
        tool_orientation_wxyz: np.ndarray,
    ) -> bool:
        if self.ik is None:
            return False

        hand_target_position, hand_target_orientation = (
            tool_pose_to_hand_pose(
                tool_position_m=np.asarray(
                    tool_position_m,
                    dtype=np.float64,
                ),
                tool_orientation_wxyz=np.asarray(
                    tool_orientation_wxyz,
                    dtype=np.float64,
                ),
                tool_local_position_m=np.asarray(
                    self.cfg.ik.tool_center_local_position_m,
                    dtype=np.float64,
                ),
                tool_local_orientation_wxyz=np.asarray(
                    self.cfg.ik.tool_center_local_orientation_wxyz,
                    dtype=np.float64,
                ),
            )
        )
        base_position, base_orientation = (
            self.ik.articulation.get_world_pose()
        )
        self.ik.kinematics_solver.set_robot_base_pose(
            base_position,
            base_orientation,
        )
        _, success = (
            self.ik.articulation_solver.compute_inverse_kinematics(
                target_position=hand_target_position,
                target_orientation=hand_target_orientation,
                position_tolerance=self.cfg.ik.position_tolerance_m,
                orientation_tolerance=self.cfg.ik.orientation_tolerance_rad,
            )
        )
        return bool(success)

    def _log_partial_insertion_event(
        self,
        event: InsertionEvent,
    ) -> None:
        labels = {
            "started": "TWO-STAGE PORT ENTRY STARTED",
            "step_settled": "TWO-STAGE PORT ENTRY STEP SETTLED",
            "complete": "PARTIAL INSERTION COMPLETE",
            "aborted": "PARTIAL INSERTION ABORTED",
        }
        label = labels[event.kind]
        lines = [label]

        if event.settled_step_index is not None:
            lines.append(
                f"  settled command: {event.settled_step_index}/"
                f"{self._insertion_total_steps}"
            )
        if event.command is not None:
            lines.append(
                f"  next command: {event.command.step_index}/"
                f"{self._insertion_total_steps}"
            )
            lines.append(f"  next stage: {event.command.stage.value}")
            lines.append(
                f"  next total travel: "
                f"{event.command.commanded_depth_m * 1000.0:.3f} mm"
            )
            lines.append(
                f"  next depth relative to opening: "
                f"{event.command.commanded_port_depth_m * 1000.0:+.3f} mm"
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
        if event.kind in ("complete", "aborted"):
            lines.append("  next action: hold current ToolCenter target")

        log("\n".join(lines))

        if self.run_logger is not None:
            event_name = label.replace(" ", "_").replace("-", "_")
            fields: dict[str, object] = {}
            if event.settled_step_index is not None:
                fields["settled_command"] = event.settled_step_index
                fields["total_steps"] = self._insertion_total_steps
            if event.command is not None:
                fields["next_command"] = event.command.step_index
                fields["next_stage"] = event.command.stage.value
            if event.metrics is not None:
                fields["lateral_drift_mm"] = (
                    event.metrics.lateral_drift_m * 1000.0
                )
                fields["tool_center_tracking_error_mm"] = (
                    event.metrics.target_error_m * 1000.0
                )
                fields["orientation_error_deg"] = (
                    event.metrics.orientation_error_deg
                )
            if event.reason is not None:
                fields["reason"] = event.reason
            self.run_logger.log_event(
                t=self.frame_index,
                event=event_name,
                **fields,
            )

            # Frame-stream coverage during coarse_approach/fine_insertion:
            # the perception-loop log_frame() stops firing once the camera
            # is disabled at qualification, so mirror the settlement metrics
            # into frames.jsonl here.
            if event.metrics is not None:
                self.run_logger.log_frame(
                    t=self.frame_index,
                    phase=self.current_phase(),
                    lateral_drift_mm=event.metrics.lateral_drift_m * 1000.0,
                    tool_center_tracking_error_mm=(
                        event.metrics.target_error_m * 1000.0
                    ),
                    orientation_error_deg=event.metrics.orientation_error_deg,
                )

    def _hand_pose_from_articulation(self) -> tuple[np.ndarray, np.ndarray]:
        if self.ik is None:
            raise RuntimeError("IK runtime is not initialized")

        base_position, base_orientation = self.ik.articulation.get_world_pose()
        self.ik.kinematics_solver.set_robot_base_pose(
            np.asarray(base_position, dtype=np.float64).reshape(3),
            np.asarray(base_orientation, dtype=np.float64).reshape(4),
        )

        translation, rotation = (
            self.ik.articulation_solver.compute_end_effector_pose()
        )
        if translation is None or rotation is None:
            raise RuntimeError("Lula forward kinematics failed for panda_hand")

        return (
            np.asarray(translation, dtype=np.float64).reshape(3),
            matrix_to_quaternion_wxyz(
                np.asarray(rotation, dtype=np.float64).reshape(3, 3)
            ),
        )

    def _tool_pose_from_articulation(self) -> tuple[np.ndarray, np.ndarray]:
        hand_position, hand_orientation = self._hand_pose_from_articulation()
        return hand_pose_to_tool_pose(
            hand_position_m=hand_position,
            hand_orientation_wxyz=hand_orientation,
            tool_local_position_m=np.asarray(
                self.cfg.ik.tool_center_local_position_m,
                dtype=np.float64,
            ),
            tool_local_orientation_wxyz=np.asarray(
                self.cfg.ik.tool_center_local_orientation_wxyz,
                dtype=np.float64,
            ),
        )

    def _sample_mount_validation_live(self, runtime) -> tuple[float, float]:
        self._base_mount_sample_validation(runtime)

        plug_frame = self.cable_mount.plug_frame
        if plug_frame is None:
            raise RuntimeError("Tracked plug frame is unavailable")

        plug_position, plug_orientation = self._tracked_plug_body.get_world_pose()
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

        tool_position, tool_orientation = self._tool_pose_from_articulation()
        tool_axis = quaternion_wxyz_to_matrix(tool_orientation)[:, 2]

        return (
            float(np.linalg.norm(tip_world - tool_position)),
            angular_error_deg(nose_world, tool_axis),
        )

    def _get_world_pose(self, path: str) -> tuple[np.ndarray, np.ndarray]:
        if self.ik is not None and path == self.ik.hand_path:
            return self._hand_pose_from_articulation()
        return super()._get_world_pose(path)

    def _update_actual_tool_frame(self, runtime) -> None:
        tool_position, tool_orientation = self._tool_pose_from_articulation()
        runtime.actual_tool.set_world_pose(
            position=tool_position,
            orientation=tool_orientation,
        )

    def _hand_camera_local_matrix(
        self,
        local_position: tuple[float, float, float],
    ) -> np.ndarray:
        camera_cfg = self.cfg.camera
        y_quat = Gf.Rotation(
            Gf.Vec3d(0.0, 1.0, 0.0),
            camera_cfg.local_y_rotation_deg,
        ).GetQuat()
        roll_quat = Gf.Rotation(
            Gf.Vec3d(0.0, 0.0, 1.0),
            camera_cfg.local_roll_deg,
        ).GetQuat()
        local_quat = y_quat * roll_quat
        imaginary = local_quat.GetImaginary()
        rotation = quaternion_wxyz_to_matrix(
            np.asarray(
                [
                    local_quat.GetReal(),
                    imaginary[0],
                    imaginary[1],
                    imaginary[2],
                ],
                dtype=np.float64,
            )
        )

        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :3] = rotation.T
        matrix[3, :3] = np.asarray(local_position, dtype=np.float64)
        return matrix

    def _world_from_hand_matrix(self) -> np.ndarray:
        position, orientation = self._hand_pose_from_articulation()
        rotation = quaternion_wxyz_to_matrix(orientation)
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :3] = rotation.T
        matrix[3, :3] = position
        return matrix

    def _camera_model(self, camera_path: str, rgb: np.ndarray) -> CameraModel:
        stage = omni.usd.get_context().get_stage()
        camera_prim = stage.GetPrimAtPath(camera_path)
        camera = UsdGeom.Camera(camera_prim)

        if camera_path == self.left_camera_path:
            local_position = self.cfg.camera.left_local_position
        elif camera_path == self.right_camera_path:
            local_position = self.cfg.camera.right_local_position
        else:
            raise RuntimeError(f"Unknown hand camera path: {camera_path}")

        world_from_camera = (
            self._hand_camera_local_matrix(local_position)
            @ self._world_from_hand_matrix()
        )

        return CameraModel(
            image_height_px=rgb.shape[0],
            image_width_px=rgb.shape[1],
            focal_length_mm=float(camera.GetFocalLengthAttr().Get()),
            horizontal_aperture_mm=float(
                camera.GetHorizontalApertureAttr().Get()
            ),
            vertical_aperture_mm=float(
                camera.GetVerticalApertureAttr().Get()
            ),
            world_from_camera=world_from_camera,
        )
