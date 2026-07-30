#!/usr/bin/env python3

import unittest

import numpy as np

from stereo_handoff import (
    bounded_step_to_goal,
    estimate_stable_goal,
    select_recent_bounded_goal,
)


class StableGoalTests(unittest.TestCase):
    def test_accepts_consistent_world_goals(self):
        estimate = estimate_stable_goal(
            [
                [0.7000, -0.1900, 1.3200],
                [0.7004, -0.1901, 1.3202],
                [0.6997, -0.1899, 1.3198],
            ],
            minimum_samples=3,
            maximum_spread_m=0.002,
        )

        self.assertIsNotNone(estimate)
        self.assertLess(
            estimate.spread_m,
            0.001,
        )

    def test_rejects_unstable_world_goals(self):
        estimate = estimate_stable_goal(
            [
                [0.7000, -0.1900, 1.3200],
                [0.7100, -0.1900, 1.3200],
                [0.6900, -0.1900, 1.3200],
            ],
            minimum_samples=3,
            maximum_spread_m=0.002,
        )

        self.assertIsNone(estimate)

    def test_requires_enough_samples(self):
        estimate = estimate_stable_goal(
            [
                [0.7000, -0.1900, 1.3200],
                [0.7001, -0.1900, 1.3200],
            ],
            minimum_samples=3,
            maximum_spread_m=0.002,
        )

        self.assertIsNone(estimate)


class HandoffDecisionTests(unittest.TestCase):
    def test_accepts_log_observed_27_point_6_mm_remaining_distance(self):
        decision = select_recent_bounded_goal(
            [
                [0.0278, 0.0002, 0.0],
                [0.0275, -0.0001, 0.0001],
                [0.0276, 0.0, -0.0001],
            ],
            [0.0, 0.0, 0.0],
            minimum_samples=3,
            recent_sample_count=3,
            maximum_spread_m=0.002,
            maximum_distance_m=0.035,
        )

        self.assertIsNotNone(decision)
        self.assertLess(
            decision.remaining_m,
            0.028,
        )

    def test_ignores_old_oblique_view_outliers(self):
        decision = select_recent_bounded_goal(
            [
                [0.0500, 0.0100, 0.0],
                [0.0450, -0.0080, 0.0],
                [0.0302, 0.0002, 0.0],
                [0.0300, -0.0001, 0.0001],
                [0.0298, 0.0, -0.0001],
            ],
            [0.0, 0.0, 0.0],
            minimum_samples=3,
            recent_sample_count=3,
            maximum_spread_m=0.002,
            maximum_distance_m=0.035,
        )

        self.assertIsNotNone(decision)
        self.assertEqual(
            decision.estimate.sample_count,
            3,
        )
        self.assertLess(
            decision.estimate.spread_m,
            0.001,
        )

    def test_rejects_goal_outside_bounded_finish_region(self):
        decision = select_recent_bounded_goal(
            [
                [0.0360, 0.0, 0.0],
                [0.0361, 0.0, 0.0],
                [0.0359, 0.0, 0.0],
            ],
            [0.0, 0.0, 0.0],
            minimum_samples=3,
            recent_sample_count=3,
            maximum_spread_m=0.002,
            maximum_distance_m=0.035,
        )

        self.assertIsNone(decision)


class BoundedStepTests(unittest.TestCase):
    def test_caps_long_step(self):
        step, remaining = bounded_step_to_goal(
            [0.0, 0.0, 0.0],
            [0.012, 0.0, 0.0],
            maximum_step_m=0.005,
        )

        np.testing.assert_allclose(
            step,
            [0.005, 0.0, 0.0],
            atol=1.0e-12,
        )
        self.assertAlmostEqual(
            remaining,
            0.012,
            places=12,
        )

    def test_uses_exact_short_step(self):
        step, remaining = bounded_step_to_goal(
            [0.0, 0.0, 0.0],
            [0.002, -0.001, 0.0],
            maximum_step_m=0.005,
        )

        np.testing.assert_allclose(
            step,
            [0.002, -0.001, 0.0],
            atol=1.0e-12,
        )
        self.assertAlmostEqual(
            remaining,
            np.sqrt(5.0) * 0.001,
            places=12,
        )


if __name__ == "__main__":
    unittest.main()
