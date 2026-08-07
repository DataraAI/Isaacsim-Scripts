from __future__ import annotations

import math
import unittest

import numpy as np

from robot.hand_plug_geometry import (
    compute_angled_hand_pose_preserving_tool,
    expected_camera_baseline_axis_world,
    horizontal_axis_error_deg,
    measure_hand_plug_geometry,
    validate_downward_hand_pitch_deg,
)


def quaternion_wxyz_to_matrix(quaternion) -> np.ndarray:
    w, x, y, z = np.asarray(quaternion, dtype=np.float64)
    return np.array(
        [
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - z * w),
                2.0 * (x * z + y * w),
            ],
            [
                2.0 * (x * y + z * w),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - x * w),
            ],
            [
                2.0 * (x * z - y * w),
                2.0 * (y * z + x * w),
                1.0 - 2.0 * (x * x + y * y),
            ],
        ],
        dtype=np.float64,
    )


class PitchValidationTests(unittest.TestCase):
    def test_accepts_supported_range(self):
        self.assertEqual(validate_downward_hand_pitch_deg(0.0), 0.0)
        self.assertEqual(validate_downward_hand_pitch_deg(30.0), 30.0)
        self.assertEqual(validate_downward_hand_pitch_deg(45.0), 45.0)

    def test_rejects_invalid_values(self):
        for value in (-0.001, 45.001, float("nan"), float("inf")):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    validate_downward_hand_pitch_deg(value)


class PreservedToolPoseTests(unittest.TestCase):
    def setUp(self):
        self.base_hand_position = np.array(
            [0.9000, -0.1375, 1.3000],
            dtype=np.float64,
        )
        self.base_hand_rotation = quaternion_wxyz_to_matrix(
            [0.7071067811865476, 0.0, -0.7071067811865475, 0.0]
        )
        self.base_hand_from_tool = quaternion_wxyz_to_matrix(
            [0.7071067811865476, 0.0, 0.0, -0.7071067811865475]
        )
        self.tool_offset_hand = np.array(
            [0.0, 0.0, 0.1334],
            dtype=np.float64,
        )
        self.pose = compute_angled_hand_pose_preserving_tool(
            base_hand_position_m=self.base_hand_position,
            base_hand_rotation_world=self.base_hand_rotation,
            base_hand_from_tool_rotation=self.base_hand_from_tool,
            tool_position_hand_m=self.tool_offset_hand,
            downward_pitch_deg=30.0,
        )

    def test_preserves_exact_validated_tool_pose(self):
        expected_tool_position = (
            self.base_hand_position
            + self.base_hand_rotation @ self.tool_offset_hand
        )
        expected_tool_rotation = (
            self.base_hand_rotation @ self.base_hand_from_tool
        )
        np.testing.assert_allclose(
            self.pose.tool_position_world_m,
            expected_tool_position,
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            self.pose.tool_rotation_world,
            expected_tool_rotation,
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            self.pose.hand_position_world_m
            + self.pose.hand_rotation_world @ self.tool_offset_hand,
            expected_tool_position,
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            self.pose.hand_rotation_world
            @ self.pose.hand_from_tool_rotation,
            expected_tool_rotation,
            atol=1.0e-12,
        )

    def test_matches_horizontal_stereo_baseline_presentation(self):
        np.testing.assert_allclose(
            self.pose.hand_rotation_world[:, 2],
            np.array(
                [-math.cos(math.radians(30.0)), 0.0, -0.5],
                dtype=np.float64,
            ),
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            self.pose.hand_rotation_world[:, 1],
            np.array([0.0, 1.0, 0.0], dtype=np.float64),
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            self.pose.hand_position_world_m,
            np.array([0.88212779, -0.1375, 1.3667]),
            atol=1.0e-8,
        )

    def test_relative_hand_to_tool_rotation_is_downward_pitch_not_yaw(self):
        pitch = math.radians(30.0)
        expected = np.array(
            [
                [0.0, math.cos(pitch), math.sin(pitch)],
                [-1.0, 0.0, 0.0],
                [0.0, -math.sin(pitch), math.cos(pitch)],
            ],
            dtype=np.float64,
        )
        np.testing.assert_allclose(
            self.pose.hand_from_tool_rotation,
            expected,
            atol=1.0e-12,
        )

    def test_old_local_roll_composition_rotates_plug_in_world_xy(self):
        pitch = math.radians(30.0)
        rotation_y = np.array(
            [
                [math.cos(pitch), 0.0, math.sin(pitch)],
                [0.0, 1.0, 0.0],
                [-math.sin(pitch), 0.0, math.cos(pitch)],
            ],
            dtype=np.float64,
        )
        rotation_z_180 = np.diag([-1.0, -1.0, 1.0])
        broken_hand_from_tool = (
            rotation_z_180
            @ self.base_hand_from_tool
            @ rotation_y
        )
        broken_tool_axis = (
            self.base_hand_rotation @ broken_hand_from_tool
        )[:, 2]
        np.testing.assert_allclose(
            broken_tool_axis,
            np.array([-math.cos(pitch), 0.5, 0.0]),
            atol=1.0e-12,
        )
        self.assertGreater(
            math.degrees(
                math.acos(
                    float(
                        np.clip(
                            np.dot(broken_tool_axis, np.array([-1.0, 0.0, 0.0])),
                            -1.0,
                            1.0,
                        )
                    )
                )
            ),
            29.9,
        )


class GeometryMeasurementTests(unittest.TestCase):
    @staticmethod
    def _hand_rotation(*, wrong_baseline: bool) -> np.ndarray:
        hand_forward = np.array(
            [-math.cos(math.radians(30.0)), 0.0, -0.5],
            dtype=np.float64,
        )
        camera_baseline = np.array(
            [0.0, -1.0 if wrong_baseline else 1.0, 0.0],
            dtype=np.float64,
        )
        hand_side = np.cross(camera_baseline, hand_forward)
        hand_side /= float(np.linalg.norm(hand_side))
        return np.column_stack((hand_side, camera_baseline, hand_forward))

    def test_requested_geometry_passes_all_measurements(self):
        metrics = measure_hand_plug_geometry(
            hand_position_m=np.array([0.88212779, -0.1375, 1.3667]),
            hand_rotation_world=self._hand_rotation(wrong_baseline=False),
            plug_tip_position_m=np.array([0.7666, -0.1375, 1.3]),
            plug_axis_world=np.array([-1.0, 0.0, 0.0]),
        )
        self.assertAlmostEqual(metrics.relative_pitch_deg, 30.0, places=9)
        self.assertGreater(metrics.wrist_above_tip_m, 0.0)
        self.assertTrue(metrics.wrist_higher_fingertips_lower)
        self.assertAlmostEqual(metrics.camera_baseline_error_deg, 0.0, places=9)
        self.assertAlmostEqual(metrics.plug_horizontal_error_deg, 0.0, places=12)

    def test_opposite_camera_baseline_is_detected(self):
        metrics = measure_hand_plug_geometry(
            hand_position_m=np.array([0.88212779, -0.1375, 1.3667]),
            hand_rotation_world=self._hand_rotation(wrong_baseline=True),
            plug_tip_position_m=np.array([0.7666, -0.1375, 1.3]),
            plug_axis_world=np.array([-1.0, 0.0, 0.0]),
        )
        self.assertAlmostEqual(
            metrics.camera_baseline_error_deg,
            180.0,
            places=9,
        )

    def test_expected_camera_baseline_is_world_positive_y(self):
        np.testing.assert_allclose(
            expected_camera_baseline_axis_world([-1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            atol=1.0e-12,
        )

    def test_horizontal_error_measures_only_vertical_axis_component(self):
        self.assertAlmostEqual(
            horizontal_axis_error_deg([-1.0, 0.0, 0.0]),
            0.0,
        )
        self.assertAlmostEqual(
            horizontal_axis_error_deg(
                [
                    -math.cos(math.radians(1.0)),
                    0.0,
                    math.sin(math.radians(1.0)),
                ]
            ),
            1.0,
            places=9,
        )


if __name__ == "__main__":
    unittest.main()
