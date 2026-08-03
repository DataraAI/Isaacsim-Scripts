#!/usr/bin/env python3

import unittest

import numpy as np

from stereo_handoff import (
    bounded_step_to_goal,
    estimate_stable_goal,
    qualify_stationary_port_goal,
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


class StationaryPortQualificationTests(unittest.TestCase):
    def test_freezes_median_physical_opening_and_preinsert_goal(self):
        openings = [
            [0.652428, -0.191790, 1.323579],
            [0.652331, -0.192126, 1.323409],
            [0.652328, -0.192235, 1.323591],
        ]
        goals = [
            [0.702428, -0.191790, 1.323579],
            [0.702331, -0.192126, 1.323409],
            [0.702328, -0.192235, 1.323591],
        ]

        result = qualify_stationary_port_goal(
            openings,
            goals,
            minimum_samples=3,
            recent_sample_count=3,
            maximum_opening_spread_m=0.001,
            maximum_goal_spread_m=0.001,
            expected_standoff_m=0.050,
            standoff_tolerance_m=0.003,
        )

        self.assertIsNotNone(result)
        np.testing.assert_allclose(
            result.opening_position_m,
            np.median(np.asarray(openings), axis=0),
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            result.tool_goal_position_m,
            np.median(np.asarray(goals), axis=0),
            atol=1.0e-12,
        )
        self.assertAlmostEqual(result.standoff_m, 0.050, places=9)

    def test_rejects_three_consistent_cavity_points_with_wrong_standoff(self):
        openings = [
            [0.64710, -0.1920, 1.3235],
            [0.64715, -0.1921, 1.3235],
            [0.64705, -0.1919, 1.3234],
        ]
        goals = [
            [0.70240, -0.1920, 1.3235],
            [0.70245, -0.1921, 1.3235],
            [0.70235, -0.1919, 1.3234],
        ]

        result = qualify_stationary_port_goal(
            openings,
            goals,
            minimum_samples=3,
            recent_sample_count=3,
            maximum_opening_spread_m=0.001,
            maximum_goal_spread_m=0.001,
            expected_standoff_m=0.050,
            standoff_tolerance_m=0.003,
        )

        self.assertIsNone(result)

    def test_rejects_unstable_physical_opening_even_when_goals_agree(self):
        result = qualify_stationary_port_goal(
            [
                [0.6520, -0.1920, 1.3235],
                [0.6570, -0.1920, 1.3235],
                [0.6470, -0.1920, 1.3235],
            ],
            [
                [0.7020, -0.1920, 1.3235],
                [0.7021, -0.1920, 1.3235],
                [0.7019, -0.1920, 1.3235],
            ],
            minimum_samples=3,
            recent_sample_count=3,
            maximum_opening_spread_m=0.001,
            maximum_goal_spread_m=0.001,
            expected_standoff_m=0.050,
            standoff_tolerance_m=0.003,
        )

        self.assertIsNone(result)


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
