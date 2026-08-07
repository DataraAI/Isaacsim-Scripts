from __future__ import annotations

import math
import unittest

import numpy as np

from robot.angled_grasp_centering import (
    recenter_horizontal_plug_rear_in_pitched_hand,
)
from robot.hand_plug_geometry import compute_angled_hand_pose_preserving_tool


def rotation_y(angle_deg: float) -> np.ndarray:
    angle = math.radians(angle_deg)
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return np.array(
        [
            [cosine, 0.0, sine],
            [0.0, 1.0, 0.0],
            [-sine, 0.0, cosine],
        ],
        dtype=np.float64,
    )


class AngledGraspCenteringTests(unittest.TestCase):
    def setUp(self):
        self.base_hand_position = np.array(
            [0.9000, -0.1375, 1.3000],
            dtype=np.float64,
        )
        self.base_hand_rotation = rotation_y(-90.0)
        self.tool_position_hand = np.array(
            [0.0, 0.0, 0.1334],
            dtype=np.float64,
        )
        self.camera_positions_hand = np.array(
            [
                [0.04, -0.020, 0.025],
                [0.04, +0.020, 0.025],
                [0.04, 0.0, 0.025],
            ],
            dtype=np.float64,
        )
        self.plug_length_m = 0.036152
        self.pitch_deg = 30.0

    def test_rear_centering_shift_is_derived_from_plug_length_and_pitch(self):
        calibration = recenter_horizontal_plug_rear_in_pitched_hand(
            base_hand_position_m=self.base_hand_position,
            base_hand_rotation_world=self.base_hand_rotation,
            tool_position_hand_m=self.tool_position_hand,
            camera_positions_hand_m=self.camera_positions_hand,
            plug_body_length_m=self.plug_length_m,
            downward_pitch_deg=self.pitch_deg,
        )

        np.testing.assert_allclose(
            calibration.local_shift_hand_m,
            np.array([0.018076, 0.0, 0.0]),
            atol=1.0e-12,
        )

    def test_tool_and_camera_world_poses_are_preserved_exactly(self):
        calibration = recenter_horizontal_plug_rear_in_pitched_hand(
            base_hand_position_m=self.base_hand_position,
            base_hand_rotation_world=self.base_hand_rotation,
            tool_position_hand_m=self.tool_position_hand,
            camera_positions_hand_m=self.camera_positions_hand,
            plug_body_length_m=self.plug_length_m,
            downward_pitch_deg=self.pitch_deg,
        )

        old_tool_world = (
            self.base_hand_position
            + self.base_hand_rotation @ self.tool_position_hand
        )
        new_tool_world = (
            calibration.base_hand_position_world_m
            + self.base_hand_rotation @ calibration.tool_position_hand_m
        )
        np.testing.assert_allclose(new_tool_world, old_tool_world, atol=1.0e-12)

        old_cameras_world = (
            self.base_hand_position[None, :]
            + (self.base_hand_rotation @ self.camera_positions_hand.T).T
        )
        new_cameras_world = (
            calibration.base_hand_position_world_m[None, :]
            + (
                self.base_hand_rotation
                @ calibration.camera_positions_hand_m.T
            ).T
        )
        np.testing.assert_allclose(
            new_cameras_world,
            old_cameras_world,
            atol=1.0e-12,
        )

    def test_plug_rear_lands_on_pitched_hand_centerline(self):
        calibration = recenter_horizontal_plug_rear_in_pitched_hand(
            base_hand_position_m=self.base_hand_position,
            base_hand_rotation_world=self.base_hand_rotation,
            tool_position_hand_m=self.tool_position_hand,
            camera_positions_hand_m=self.camera_positions_hand,
            plug_body_length_m=self.plug_length_m,
            downward_pitch_deg=self.pitch_deg,
        )
        base_hand_from_tool = np.eye(3, dtype=np.float64)
        pose = compute_angled_hand_pose_preserving_tool(
            base_hand_position_m=calibration.base_hand_position_world_m,
            base_hand_rotation_world=self.base_hand_rotation,
            base_hand_from_tool_rotation=base_hand_from_tool,
            tool_position_hand_m=calibration.tool_position_hand_m,
            downward_pitch_deg=self.pitch_deg,
        )
        plug_axis_hand = pose.hand_from_tool_rotation[:, 2]
        plug_rear_hand = (
            calibration.tool_position_hand_m
            - self.plug_length_m * plug_axis_hand
        )

        self.assertAlmostEqual(float(plug_rear_hand[0]), 0.0, places=12)


if __name__ == "__main__":
    unittest.main()
