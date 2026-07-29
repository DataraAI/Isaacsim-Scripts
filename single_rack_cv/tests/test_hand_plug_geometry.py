from __future__ import annotations

import math
import unittest

import numpy as np

from hand_plug_geometry import (
    compute_pitched_hand_from_tool_rotation,
    expected_palm_side_axis_world,
    horizontal_axis_error_deg,
    measure_hand_plug_geometry,
    validate_downward_hand_pitch_deg,
    validate_palm_roll_deg,
)


def rotation_y(angle_deg: float) -> np.ndarray:
    angle = math.radians(angle_deg)
    c = math.cos(angle)
    s = math.sin(angle)
    return np.array(
        [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]],
        dtype=np.float64,
    )


def rotation_z(angle_deg: float) -> np.ndarray:
    angle = math.radians(angle_deg)
    c = math.cos(angle)
    s = math.sin(angle)
    return np.array(
        [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
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

    def test_palm_roll_must_be_finite(self):
        self.assertEqual(validate_palm_roll_deg(180.0), 180.0)
        for value in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    validate_palm_roll_deg(value)


class PitchedTransformTests(unittest.TestCase):
    def setUp(self):
        self.base_hand_from_tool = np.array(
            [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        self.world_from_tool = np.array(
            [[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
            dtype=np.float64,
        )

    def test_zero_pitch_and_zero_roll_are_backward_compatible(self):
        actual = compute_pitched_hand_from_tool_rotation(
            self.base_hand_from_tool,
            0.0,
            palm_roll_deg=0.0,
        )
        np.testing.assert_allclose(actual, self.base_hand_from_tool, atol=1.0e-12)

    def test_thirty_degree_pitch_and_legacy_roll_use_exact_composition(self):
        actual = compute_pitched_hand_from_tool_rotation(
            self.base_hand_from_tool,
            30.0,
            palm_roll_deg=180.0,
        )
        np.testing.assert_allclose(
            actual,
            rotation_z(180.0)
            @ self.base_hand_from_tool
            @ rotation_y(30.0),
            atol=1.0e-12,
        )

    def test_result_matches_previous_palm_presentation(self):
        hand_from_tool = compute_pitched_hand_from_tool_rotation(
            self.base_hand_from_tool,
            30.0,
            palm_roll_deg=180.0,
        )
        world_from_hand = self.world_from_tool @ hand_from_tool.T
        hand_forward_world = world_from_hand[:, 2]
        palm_side_world = world_from_hand[:, 0]
        plug_axis_world = self.world_from_tool[:, 2]
        np.testing.assert_allclose(
            hand_forward_world,
            np.array([-math.cos(math.radians(30.0)), 0.0, -0.5]),
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            palm_side_world,
            np.array([0.0, -1.0, 0.0]),
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            plug_axis_world,
            np.array([-1.0, 0.0, 0.0]),
            atol=1.0e-12,
        )


class GeometryMeasurementTests(unittest.TestCase):
    @staticmethod
    def _hand_rotation(*, flipped_palm: bool) -> np.ndarray:
        hand_forward = np.array(
            [-math.cos(math.radians(30.0)), 0.0, -0.5],
            dtype=np.float64,
        )
        hand_side = np.array(
            [0.0, 1.0 if flipped_palm else -1.0, 0.0],
            dtype=np.float64,
        )
        hand_up = np.cross(hand_forward, hand_side)
        hand_up /= float(np.linalg.norm(hand_up))
        return np.column_stack((hand_side, hand_up, hand_forward))

    def test_requested_geometry_reports_zero_palm_roll_error(self):
        metrics = measure_hand_plug_geometry(
            hand_position_m=np.array([0.8, 0.0, 1.4]),
            hand_rotation_world=self._hand_rotation(flipped_palm=False),
            plug_tip_position_m=np.array([0.7, 0.0, 1.3333]),
            plug_axis_world=np.array([-1.0, 0.0, 0.0]),
        )
        self.assertAlmostEqual(metrics.relative_pitch_deg, 30.0, places=9)
        self.assertGreater(metrics.wrist_above_tip_m, 0.0)
        self.assertTrue(metrics.wrist_higher_fingertips_lower)
        self.assertAlmostEqual(metrics.palm_roll_error_deg, 0.0, places=9)
        self.assertAlmostEqual(metrics.plug_horizontal_error_deg, 0.0, places=12)

    def test_previous_bug_is_detected_as_180_degree_palm_roll_error(self):
        metrics = measure_hand_plug_geometry(
            hand_position_m=np.array([0.8, 0.0, 1.4]),
            hand_rotation_world=self._hand_rotation(flipped_palm=True),
            plug_tip_position_m=np.array([0.7, 0.0, 1.3333]),
            plug_axis_world=np.array([-1.0, 0.0, 0.0]),
        )
        self.assertAlmostEqual(metrics.palm_roll_error_deg, 180.0, places=9)

    def test_expected_palm_side_axis_matches_old_pose(self):
        np.testing.assert_allclose(
            expected_palm_side_axis_world([-1.0, 0.0, 0.0]),
            np.array([0.0, -1.0, 0.0]),
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
