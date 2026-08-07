#!/usr/bin/env python3

from __future__ import annotations

import unittest

import numpy as np

from control.handoff_position_hold import update_handoff_position_command


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
        self.assertGreater(update.command_position_m[0], goal[0])
        self.assertAlmostEqual(update.command_position_m[1], goal[1], places=12)
        self.assertAlmostEqual(update.command_position_m[2], goal[2], places=12)

    def test_command_bias_is_bounded(self):
        goal = np.array([0.0, 0.0, 0.0])
        update = update_handoff_position_command(
            goal_position_m=goal,
            actual_position_m=np.array([-0.010, 0.0, 0.0]),
            current_command_position_m=np.array([0.00095, 0.0, 0.0]),
            gain=1.0,
            maximum_step_m=0.0002,
            maximum_bias_m=0.001,
        )

        self.assertAlmostEqual(update.command_bias_m, 0.001, places=12)
        self.assertTrue(update.bias_saturated)
        np.testing.assert_allclose(
            update.command_position_m,
            np.array([0.001, 0.0, 0.0]),
            atol=1.0e-12,
        )

    def test_invalid_bounds_fail_closed(self):
        with self.assertRaises(ValueError):
            update_handoff_position_command(
                goal_position_m=np.zeros(3),
                actual_position_m=np.zeros(3),
                current_command_position_m=np.zeros(3),
                gain=0.35,
                maximum_step_m=0.0,
                maximum_bias_m=0.001,
            )


if __name__ == "__main__":
    unittest.main()
