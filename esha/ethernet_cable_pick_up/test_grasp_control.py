import unittest

import numpy as np

from grasp_control import (
    apply_grasp_x_offset,
    bounded_linear_step,
    clearance_target_position,
    finger_target_reached,
    fingers_moved_toward_closed,
    grasp_orientation_active,
    select_open_half_gap,
    resolve_tool_orientation,
)


class ResolveToolOrientationTests(unittest.TestCase):
    def test_active_grasp_uses_stored_orientation(self) -> None:
        marker_orientation = np.array([0.14, 0.0, 0.99, 0.0])
        grasp_orientation = np.array([0.50, 0.0, 0.866, 0.0])

        result = resolve_tool_orientation(
            marker_orientation,
            grasp_orientation,
            grasp_active=True,
        )

        expected = grasp_orientation / np.linalg.norm(grasp_orientation)
        np.testing.assert_allclose(result, expected)

    def test_inactive_grasp_uses_marker_orientation(self) -> None:
        marker_orientation = np.array([2.0, 0.0, 0.0, 0.0])

        result = resolve_tool_orientation(
            marker_orientation,
            None,
            grasp_active=False,
        )

        np.testing.assert_allclose(result, [1.0, 0.0, 0.0, 0.0])

    def test_grasp_orientation_starts_at_single_angle_waypoint(self) -> None:
        self.assertFalse(grasp_orientation_active("idle"))
        self.assertTrue(grasp_orientation_active("angle"))

    def test_angled_standoff_stays_on_configured_approach_line(self) -> None:
        cable_point = np.array([0.7557, 0.0001, 0.0506])
        approach = np.array([0.8660254, 0.0, -0.5])

        result = clearance_target_position(
            cable_point,
            approach,
            clearance_m=0.120,
            minimum_z_m=0.042,
        )

        expected = cable_point - 0.120 * approach
        np.testing.assert_allclose(result, expected, atol=1.0e-8)

    def test_open_requires_both_fingers_at_commanded_gap(self) -> None:
        self.assertTrue(
            finger_target_reached(
                np.array([0.0125, 0.0127]),
                target_position_m=0.0126,
                tolerance_m=0.0002,
            )
        )
        self.assertFalse(
            finger_target_reached(
                np.array([0.0080, 0.0126]),
                target_position_m=0.0126,
                tolerance_m=0.0002,
            )
        )

    def test_close_requires_real_travel_from_open_position(self) -> None:
        open_positions = np.array([0.0126, 0.0126])
        self.assertTrue(
            fingers_moved_toward_closed(
                np.array([0.0105, 0.0104]),
                open_positions,
                minimum_travel_m=0.001,
            )
        )
        self.assertFalse(
            fingers_moved_toward_closed(
                np.array([0.0125, 0.0125]),
                open_positions,
                minimum_travel_m=0.001,
            )
        )

    def test_open_gap_has_safe_minimum(self) -> None:
        self.assertAlmostEqual(
            select_open_half_gap(
                cable_half_width_m=0.0106,
                side_allowance_m=0.002,
                minimum_half_gap_m=0.018,
                maximum_half_gap_m=0.040,
            ),
            0.018,
        )

    def test_grasp_approach_step_is_straight_and_bounded(self) -> None:
        current = np.array([0.6519, 0.0001, 0.1106])
        target = np.array([0.7536, 0.0001, 0.0518])

        result = bounded_linear_step(current, target, max_step_m=0.003)

        step = result - current
        remaining = target - current
        self.assertAlmostEqual(float(np.linalg.norm(step)), 0.003)
        np.testing.assert_allclose(
            np.cross(step, remaining),
            np.zeros(3),
            atol=1.0e-10,
        )

    def test_grasp_x_offset_does_not_change_y_or_z(self) -> None:
        estimated = np.array([0.7556, 0.0001, 0.0508])

        result = apply_grasp_x_offset(estimated, x_offset_m=-0.002)

        np.testing.assert_allclose(result, [0.7536, 0.0001, 0.0508])


if __name__ == "__main__":
    unittest.main()
