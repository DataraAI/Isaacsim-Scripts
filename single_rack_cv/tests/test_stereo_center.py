from __future__ import annotations

import inspect
import unittest

import numpy as np

from stereo_center import estimate_stereo_aperture_center
from tests.test_rgb_front_rim_center import (
    SyntheticCamera,
    render_front_rim,
)


class StereoApertureCenterTests(unittest.TestCase):
    def test_direct_center_survives_oblique_view_and_recessed_mask_parallax(self):
        left = SyntheticCamera((-0.02, 0.0, 0.0))
        right = SyntheticCamera((0.02, 0.0, 0.0))
        center = np.array([0.004, -0.003, -0.16], dtype=np.float64)
        normal = np.array([0.20, -0.08, 0.976], dtype=np.float64)
        normal /= np.linalg.norm(normal)
        baseline = right.camera_center_world_m - left.camera_center_world_m
        horizontal = baseline - float(np.dot(baseline, normal)) * normal
        horizontal /= np.linalg.norm(horizontal)
        vertical = np.cross(normal, horizontal)
        vertical /= np.linalg.norm(vertical)

        left_rgb, left_mask = render_front_rim(
            left,
            center,
            horizontal,
            vertical,
            mask_shift_px=-3.0,
        )
        right_rgb, right_mask = render_front_rim(
            right,
            center,
            horizontal,
            vertical,
            mask_shift_px=4.0,
        )

        result = estimate_stereo_aperture_center(
            left_rgb=left_rgb,
            right_rgb=right_rgb,
            left_mask=left_mask,
            right_mask=right_mask,
            left_camera=left,
            right_camera=right,
        )

        self.assertLess(
            float(np.linalg.norm(result.center_world_m - center)),
            0.00075,
        )
        self.assertLess(result.ray_gap_m, 0.0005)

    def test_bad_vertical_front_rim_correspondence_fails_closed(self):
        left = SyntheticCamera((-0.02, 0.0, 0.0))
        right = SyntheticCamera((0.02, 0.0, 0.0))
        center = np.array([0.0, 0.0, -0.16], dtype=np.float64)
        horizontal = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        vertical = np.array([0.0, 1.0, 0.0], dtype=np.float64)

        left_rgb, left_mask = render_front_rim(
            left,
            center,
            horizontal,
            vertical,
            mask_shift_px=0.0,
        )
        right_rgb, right_mask = render_front_rim(
            right,
            center,
            horizontal,
            vertical,
            mask_shift_px=0.0,
        )
        right_rgb = np.roll(right_rgb, shift=8, axis=0)
        right_mask = np.roll(right_mask, shift=8, axis=0)

        with self.assertRaisesRegex(RuntimeError, "ray gap"):
            estimate_stereo_aperture_center(
                left_rgb=left_rgb,
                right_rgb=right_rgb,
                left_mask=left_mask,
                right_mask=right_mask,
                left_camera=left,
                right_camera=right,
            )

    def test_rgb_is_required_and_no_manual_offset_parameter_exists(self):
        parameters = inspect.signature(
            estimate_stereo_aperture_center
        ).parameters
        self.assertIn("left_rgb", parameters)
        self.assertIn("right_rgb", parameters)
        forbidden = {"offset", "world_offset", "center_offset", "bias"}
        self.assertTrue(forbidden.isdisjoint(parameters))


if __name__ == "__main__":
    unittest.main()
