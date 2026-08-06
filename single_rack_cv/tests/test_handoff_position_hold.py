#!/usr/bin/env python3

from __future__ import annotations

import unittest

import numpy as np

from handoff_position_hold import update_handoff_position_command


class HandoffPositionHoldTests(unittest.TestCase):
    def test_static_tracking_error_advances_command_toward_frozen_goal(self):
        goal = np.array([1.0, 2.0, 3.0])
        actual = goal + np.array([-0.000455, 0.0, 0.0])

        update = update_handoff_position_command(
            goal_position_m=goal,
            actual_position_m=actual,
            current_command_position_m=goal,
            gain=0.35,
            maximum_step_m=0.0001,
            maximum_bias_m=0.001,
        )

        self.assertAlmostEqual(update.physical_error_m, 0.000455, places=12)
        self.assertAlmostEqual(
            float(np.linalg.norm(update.step_world_m)),
            0.0001,
            places=12,
        )
        np.testing.assert_allclose(
            update.command_position_m,
            goal + np.array([0.0001, 0.0, 0.0]),
            atol=1.0e-12,
        )
        self.assertFalse(update.bias_saturated)

    def test_observed_point_four_zero_seven_mm_deadlock_enters_original_gate(self):
        goal = np.zeros(3)
        command = goal.copy()
        static_bias = np.array([0.000407, 0.0, 0.0])
        physical_error_m = float("inf")

        for _ in range(20):
            actual = command - static_bias
            physical_error_m = float(np.linalg.norm(goal - actual))
            if physical_error_m <= 0.0003:
                break
            update = update_handoff_position_command(
                goal_position_m=goal,
                actual_position_m=actual,
                current_command_position_m=command,
                gain=0.35,
                maximum_step_m=0.0001,
                maximum_bias_m=0.001,
            )
            command = update.command_position_m

        self.assertLessEqual(physical_error_m, 0.0003)
        self.assertLessEqual(float(np.linalg.norm(command - goal)), 0.001)

    def test_constant_point_four_five_five_mm_bias_enters_original_gate(self):
        goal = np.zeros(3)
        command = goal.copy()
        static_bias = np.array([0.000455, 0.0, 0.0])
        physical_error_m = float("inf")

        for _ in range(20):
            actual = command - static_bias
            physical_error_m = float(np.linalg.norm(goal - actual))
            if physical_error_m <= 0.0003:
                break
            update = update_handoff_position_command(
                goal_position_m=goal,
                actual_position_m=actual,
                current_command_position_m=command,
                gain=0.35,
                maximum_step_m=0.0001,
                maximum_bias_m=0.001,
            )
            command = update.command_position_m

        self.assertLessEqual(physical_error_m, 0.0003)
        self.assertLessEqual(float(np.linalg.norm(command - goal)), 0.001)

    def test_total_command_bias_is_capped(self):
        goal = np.zeros(3)
        current = np.array([0.00095, 0.0, 0.0])
        actual = np.array([-0.000455, 0.0, 0.0])

        update = update_handoff_position_command(
            goal_position_m=goal,
            actual_position_m=actual,
            current_command_position_m=current,
            gain=1.0,
            maximum_step_m=0.0001,
            maximum_bias_m=0.001,
        )

        np.testing.assert_allclose(
            update.command_position_m,
            np.array([0.001, 0.0, 0.0]),
            atol=1.0e-12,
        )
        self.assertAlmostEqual(update.command_bias_m, 0.001, places=12)
        self.assertAlmostEqual(
            float(np.linalg.norm(update.step_world_m)),
            0.00005,
            places=12,
        )
        self.assertTrue(update.bias_saturated)

    def test_zero_physical_error_does_not_move_command(self):
        goal = np.array([0.5, -0.2, 1.3])
        current = goal + np.array([0.0002, 0.0, 0.0])

        update = update_handoff_position_command(
            goal_position_m=goal,
            actual_position_m=goal,
            current_command_position_m=current,
            gain=0.35,
            maximum_step_m=0.0001,
            maximum_bias_m=0.001,
        )

        np.testing.assert_allclose(
            update.command_position_m,
            current,
            atol=1.0e-12,
        )
        np.testing.assert_allclose(update.step_world_m, np.zeros(3), atol=1.0e-12)
        self.assertAlmostEqual(update.physical_error_m, 0.0, places=12)

    def test_invalid_vector_shape_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "three-dimensional"):
            update_handoff_position_command(
                goal_position_m=np.zeros(2),
                actual_position_m=np.zeros(3),
                current_command_position_m=np.zeros(3),
                gain=0.35,
                maximum_step_m=0.0001,
                maximum_bias_m=0.001,
            )


if __name__ == "__main__":
    unittest.main()
