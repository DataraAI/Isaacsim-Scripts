from __future__ import annotations

import math
import unittest

import numpy as np

from hand_plug_geometry import (
    compute_pitched_hand_from_tool_rotation,
    horizontal_axis_error_deg,
    measure_hand_plug_geometry,
    validate_downward_hand_pitch_deg,
)


def rotation_y(angle_deg: float) -> np.ndarray:
    angle = math.radians(angle_deg)
    c = math.cos(angle)
    s = math.sin(angle)
    return np.array(
        [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]],
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

    def test_zero_pitch_is_backward_compatible(self):
        actual = compute_pitched_hand_from_tool_rotation(
            self.base_hand_from_tool,
            0.0,
        )
        np.testing.assert_allclose(actual, self.base_hand_from_tool, atol=1.0e-12)

    def test_thirty_degree_pitch_uses_tool_local_positive_y(self):
        actual = compute_pitched_hand_from_tool_rotation(
            self.base_hand_from_tool,
            30.0,
        )
        np.testing.assert_allclose(
            actual,
            self.base_hand_from_tool @ rotation_y(30.0),
            atol=1.0e-12,
        )

    def test_result_places_hand_forward_downward_while_tool_axis_stays_horizontal(self):
        hand_from_tool = compute_pitched_hand_from_tool_rotation(
            self.base_hand_from_tool,
            30.0,
        )
        world_from_hand = self.world_from_tool @ hand_from_tool.T
        hand_forward_world = world_from_hand[:, 2]
        plug_axis_world = self.world_from_tool[:, 2]
        np.testing.assert_allclose(
            hand_forward_world,
            np.array([-math.cos(math.radians(30.0)), 0.0, -0.5]),
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            plug_axis_world,
            np.array([-1.0, 0.0, 0.0]),
            atol=1.0e-12,
        )


class GeometryMeasurementTests(unittest.TestCase):
    def test_requested_geometry_reports_thirty_degrees_and_wrist_above_tip(self):
        hand_forward = np.array(
            [-math.cos(math.radians(30.0)), 0.0, -0.5],
            dtype=np.float64,
        )
        hand_rotation = np.eye(3, dtype=np.float64)
        hand_rotation[:, 2] = hand_forward
        hand_rotation[:, 1] = np.array([0.0, 1.0, 0.0])
        hand_rotation[:, 0] = np.cross(hand_rotation[:, 1], hand_rotation[:, 2])
        metrics = measure_hand_plug_geometry(
            hand_position_m=np.array([0.8, 0.0, 1.4]),
            hand_rotation_world=hand_rotation,
            plug_tip_position_m=np.array([0.7, 0.0, 1.3333]),
            plug_axis_world=np.array([-1.0, 0.0, 0.0]),
        )
        self.assertAlmostEqual(metrics.relative_pitch_deg, 30.0, places=9)
        self.assertGreater(metrics.wrist_above_tip_m, 0.0)
        self.assertTrue(metrics.wrist_higher_fingertips_lower)
        self.assertAlmostEqual(metrics.plug_horizontal_error_deg, 0.0, places=12)

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
