#!/usr/bin/env python3
"""GPU-safe cable runtime facade for Isaac Sim CUDA dynamics."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import omni.usd
from pxr import Gf, UsdGeom

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
    """Use live articulation FK instead of stale USD child transforms."""

    def __init__(self, simulation_app, cfg):
        super().__init__(simulation_app=simulation_app, cfg=cfg)
        log("GPU FK POSE BRIDGE ACTIVE")

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
