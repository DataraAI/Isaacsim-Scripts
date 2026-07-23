from __future__ import annotations

import unittest
import numpy as np

from cable_geometry import (
    angular_error_deg,
    compute_attachment_bounds,
    compute_world_from_root_for_tip,
    detect_plug_frame,
    validate_mount_window,
    validate_transform,
)


def matrix(rotation=None, translation=(0.0, 0.0, 0.0)):
    result = np.eye(4, dtype=np.float64)
    if rotation is not None:
        result[:3, :3] = np.asarray(rotation, dtype=np.float64)
    result[:3, 3] = np.asarray(translation, dtype=np.float64)
    return result


class CableGeometryTests(unittest.TestCase):
    def test_x_axis_with_cable_on_negative_side_selects_positive_nose(self):
        frame = detect_plug_frame(
            np.array([-0.018, -0.005, -0.006]),
            np.array([+0.018, +0.005, +0.006]),
            matrix(),
            np.array([-0.20, 0.0, 0.0]),
            axis_ratio_min=1.5,
            cable_projection_min_m=0.002,
        )
        np.testing.assert_allclose(frame.tip_local_m, [0.018, 0.0, 0.0])
        np.testing.assert_allclose(frame.nose_axis_local, [1.0, 0.0, 0.0])
        np.testing.assert_allclose(frame.wide_axis_local, [0.0, 0.0, 1.0])
        self.assertEqual(frame.cable_side_sign, -1)

    def test_y_axis_with_cable_on_positive_side_selects_negative_nose(self):
        frame = detect_plug_frame(
            np.array([-0.005, -0.018, -0.006]),
            np.array([+0.005, +0.018, +0.006]),
            matrix(),
            np.array([0.0, +0.20, 0.0]),
            axis_ratio_min=1.5,
            cable_projection_min_m=0.002,
        )
        np.testing.assert_allclose(frame.tip_local_m, [0.0, -0.018, 0.0])
        np.testing.assert_allclose(frame.nose_axis_local, [0.0, -1.0, 0.0])
        self.assertEqual(frame.cable_side_sign, 1)

    def test_z_axis_is_supported(self):
        frame = detect_plug_frame(
            np.array([-0.005, -0.006, -0.018]),
            np.array([+0.005, +0.006, +0.018]),
            matrix(),
            np.array([0.0, 0.0, -0.20]),
            axis_ratio_min=1.5,
            cable_projection_min_m=0.002,
        )
        np.testing.assert_allclose(frame.nose_axis_local, [0.0, 0.0, 1.0])

    def test_world_rotation_is_used_when_classifying_cable_side(self):
        rotation = np.array([
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        frame = detect_plug_frame(
            np.array([-0.018, -0.005, -0.006]),
            np.array([+0.018, +0.005, +0.006]),
            matrix(rotation, (1.0, 2.0, 3.0)),
            np.array([1.0, 1.8, 3.0]),
            axis_ratio_min=1.5,
            cable_projection_min_m=0.002,
        )
        np.testing.assert_allclose(frame.nose_axis_local, [1.0, 0.0, 0.0])

    def test_ambiguous_axis_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "ambiguous longitudinal axis"):
            detect_plug_frame(
                np.array([-0.010, -0.009, -0.006]),
                np.array([+0.010, +0.009, +0.006]),
                matrix(),
                np.array([-0.1, 0.0, 0.0]),
                axis_ratio_min=1.5,
                cable_projection_min_m=0.002,
            )

    def test_ambiguous_cable_projection_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "ambiguous cable-side projection"):
            detect_plug_frame(
                np.array([-0.018, -0.005, -0.006]),
                np.array([+0.018, +0.005, +0.006]),
                matrix(),
                np.array([0.0, 0.1, 0.0]),
                axis_ratio_min=1.5,
                cable_projection_min_m=0.002,
            )

    def test_attachment_does_not_extend_past_cable_side(self):
        frame = detect_plug_frame(
            np.array([-0.018, -0.005, -0.006]),
            np.array([+0.018, +0.005, +0.006]),
            matrix(),
            np.array([-0.2, 0.0, 0.0]),
            axis_ratio_min=1.5,
            cable_projection_min_m=0.002,
        )
        bounds = compute_attachment_bounds(frame, padding_m=0.0005)
        self.assertAlmostEqual(bounds.local_min_m[0], -0.018)
        self.assertAlmostEqual(bounds.local_max_m[0], +0.0185)
        self.assertAlmostEqual(bounds.local_min_m[1], -0.0055)
        self.assertAlmostEqual(bounds.local_max_m[1], +0.0055)
        self.assertAlmostEqual(bounds.local_min_m[2], -0.0065)
        self.assertAlmostEqual(bounds.local_max_m[2], +0.0065)

    def test_attachment_padding_on_negative_nose_does_not_extend_cable_side(self):
        frame = detect_plug_frame(
            np.array([-0.018, -0.005, -0.006]),
            np.array([+0.018, +0.005, +0.006]),
            matrix(),
            np.array([+0.2, 0.0, 0.0]),
            axis_ratio_min=1.5,
            cable_projection_min_m=0.002,
        )
        bounds = compute_attachment_bounds(frame, padding_m=0.0005)
        self.assertAlmostEqual(bounds.local_min_m[0], -0.0185)
        self.assertAlmostEqual(bounds.local_max_m[0], +0.018)

    def test_root_mapping_puts_tip_frame_on_toolcenter(self):
        world_from_root = matrix(translation=(0.2, -0.1, 0.4))
        world_from_plug = matrix(translation=(0.5, 0.0, 0.2))
        desired = matrix(translation=(0.7, -0.2, 1.3))
        frame = detect_plug_frame(
            np.array([-0.018, -0.005, -0.006]),
            np.array([+0.018, +0.005, +0.006]),
            world_from_plug,
            np.array([0.1, 0.0, 0.2]),
            axis_ratio_min=1.5,
            cable_projection_min_m=0.002,
        )
        mounted = compute_world_from_root_for_tip(
            world_from_root,
            world_from_plug,
            frame,
            desired,
        )
        root_from_plug = np.linalg.inv(world_from_root) @ world_from_plug
        actual = mounted @ root_from_plug @ frame.plug_from_tip
        np.testing.assert_allclose(actual, desired, atol=1e-9)

    def test_angular_error_is_zero_for_parallel_axes(self):
        self.assertAlmostEqual(
            angular_error_deg(
                np.array([2.0, 0.0, 0.0]),
                np.array([1.0, 0.0, 0.0]),
            ),
            0.0,
        )

    def test_angular_error_rejects_zero_axis(self):
        with self.assertRaisesRegex(ValueError, "nonzero"):
            angular_error_deg(np.zeros(3), np.array([1.0, 0.0, 0.0]))

    def test_one_bad_validation_frame_fails_window(self):
        samples = [(0.0001, 0.1)] * 29 + [(0.0006, 0.1)]
        with self.assertRaisesRegex(RuntimeError, "tip mount error"):
            validate_mount_window(samples, 30, 0.0005, 1.0)

    def test_complete_valid_window_returns_maxima(self):
        samples = [(0.0001, 0.1)] * 29 + [(0.0004, 0.9)]
        result = validate_mount_window(samples, 30, 0.0005, 1.0)
        self.assertEqual(result.frame_count, 30)
        self.assertAlmostEqual(result.maximum_tip_error_m, 0.0004)
        self.assertAlmostEqual(result.maximum_axis_error_deg, 0.9)

    def test_validation_window_rejects_negative_nonfinite_or_wrong_count(self):
        with self.assertRaisesRegex(ValueError, "complete frame window"):
            validate_mount_window([(0.0, 0.0)], 30, 0.0005, 1.0)
        with self.assertRaisesRegex(ValueError, "finite and nonnegative"):
            validate_mount_window(
                [(float("nan"), 0.0)] * 30,
                30,
                0.0005,
                1.0,
            )
        with self.assertRaisesRegex(ValueError, "finite and nonnegative"):
            validate_mount_window([(-0.1, 0.0)] * 30, 30, 0.0005, 1.0)

    def test_transform_validation_rejects_reflection(self):
        reflected = np.eye(4)
        reflected[0, 0] = -1.0
        with self.assertRaisesRegex(ValueError, "right handed"):
            validate_transform(reflected, "reflected")


if __name__ == "__main__":
    unittest.main()
