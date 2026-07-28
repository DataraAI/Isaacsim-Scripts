#!/usr/bin/env python3
"""GPU-safe cable runtime facade for Isaac Sim CUDA dynamics."""

from __future__ import annotations

from dataclasses import replace
import importlib.util
import sys
from pathlib import Path

import numpy as np
import omni.usd
from pxr import Gf, UsdGeom

from isaacsim.core.prims import SingleRigidPrim

from cable_geometry import angular_error_deg
from host_array_bridge import to_numpy_cpu
from perception import CameraModel
from sim import (
    hand_pose_to_tool_pose,
    log,
    matrix_to_quaternion_wxyz,
    quaternion_wxyz_to_matrix,
)


_BASE_PATH = Path(__file__).resolve().parents[1] / "cable_runtime.py"
_SPEC = importlib.util.spec_from_file_location(
    "_single_rack_cable_runtime_base",
    _BASE_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"Could not load cable runtime base from {_BASE_PATH}")
_BASE_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _BASE_MODULE
_SPEC.loader.exec_module(_BASE_MODULE)
_BaseCableMountedSimulationRuntime = _BASE_MODULE.CableMountedSimulationRuntime


class CableMountedSimulationRuntime(_BaseCableMountedSimulationRuntime):
    """Use live PhysX/FK poses instead of stale USD child transforms."""

    def __init__(self, simulation_app, cfg):
        # CUDA cable dynamics settles above the CPU-only 0.5 mm tracking floor.
        # Use 5 mm coarse visual-servo steps while retaining the 1 mm capture
        # settle gate and the original final alignment limits.
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

        # Keep every structural/topology check in CableMount.sample_validation,
        # but replace its stale USD pose result with a live PhysX rigid pose.
        self._base_mount_sample_validation = self.cable_mount.sample_validation
        self.cable_mount.sample_validation = self._sample_mount_validation_live

        log(
            "GPU FK + LIVE PLUG POSE BRIDGE ACTIVE\n"
            f"  visual-servo max step: "
            f"{self.cfg.visual_servo.max_target_step_m * 1000.0:.3f} mm\n"
            f"  between-capture settle tolerance: "
            f"{self.cfg.visual_servo.target_settle_tolerance_m * 1000.0:.3f} mm"
        )

    def capture_due(self) -> bool:
        """Expose a prolonged stop-and-look wait instead of silently freezing."""

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
        """Validate the physical plug against ToolCenter using live PhysX pose."""

        # This retains all existing validity, GPU, fixed-joint, and attachment
        # checks. Its returned pose error is discarded because it reads USD.
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
