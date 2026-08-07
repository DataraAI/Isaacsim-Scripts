from __future__ import annotations

import math
import unittest

import numpy as np

from control.orientation_hold import (
    quaternion_error_deg,
    quaternion_multiply_wxyz,
    update_orientation_hold_command,
)


def axis_angle_wxyz(axis, angle_deg: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    axis /= float(np.linalg.norm(axis))
    half = 0.5 * math.radians(float(angle_deg))
    return np.r_[math.cos(half), axis * math.sin(half)]


class OrientationHoldTests(unittest.TestCase):
    def test_static_axial_roll_bias_is_cancelled_without_relaxing_actual_limit(self):
        reference = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        plant_bias = axis_angle_wxyz([0.0, 0.0, 1.0], 1.655809)
        command = reference.copy()

        for _ in range(120):
            actual = quaternion_multiply_wxyz(plant_bias, command)
            update = update_orientation_hold_command(
                reference_wxyz=reference,
                actual_wxyz=actual,
                current_command_wxyz=command,
            )
            command = update.command_wxyz

        actual = quaternion_multiply_wxyz(plant_bias, command)
        self.assertLess(quaternion_error_deg(reference, actual), 0.05)
        self.assertLess(quaternion_error_deg(reference, command), 3.0)

    def test_one_update_is_bounded_to_point_one_five_degree(self):
        reference = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        actual = axis_angle_wxyz([0.0, 0.0, 1.0], 2.0)
        update = update_orientation_hold_command(
            reference_wxyz=reference,
            actual_wxyz=actual,
            current_command_wxyz=reference,
        )
        self.assertAlmostEqual(update.command_bias_deg, 0.15, places=9)

    def test_total_command_bias_is_capped(self):
        reference = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        actual = axis_angle_wxyz([1.0, 0.0, 0.0], 20.0)
        command = reference.copy()

        saturated = False
        for _ in range(100):
            update = update_orientation_hold_command(
                reference_wxyz=reference,
                actual_wxyz=actual,
                current_command_wxyz=command,
            )
            command = update.command_wxyz
            saturated = saturated or update.bias_saturated

        self.assertTrue(saturated)
        self.assertLessEqual(update.command_bias_deg, 3.0)
        self.assertAlmostEqual(
            quaternion_error_deg(reference, command),
            3.0,
            places=8,
        )

    def test_zero_error_does_not_move_command(self):
        reference = axis_angle_wxyz([1.0, 2.0, 3.0], 37.0)
        update = update_orientation_hold_command(
            reference_wxyz=reference,
            actual_wxyz=reference,
            current_command_wxyz=reference,
        )
        self.assertLess(update.actual_error_deg, 1.0e-9)
        self.assertLess(update.command_bias_deg, 1.0e-9)
        np.testing.assert_allclose(update.command_wxyz, reference, atol=1.0e-12)


if __name__ == "__main__":
    unittest.main()
