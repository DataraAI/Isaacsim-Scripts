from __future__ import annotations

import unittest

import numpy as np

from stereo_front_rim_plane import estimate_stereo_aperture_center
from tests.test_rgb_front_rim_center import SyntheticCamera, render_front_rim


class StereoFrontRimPlaneTests(unittest.TestCase):
    def test_center_is_reconstructed_on_oblique_front_plane(self):
        left = SyntheticCamera((-0.02, 0.0, 0.0))
        right = SyntheticCamera((0.02, 0.0, 0.0))
        expected = np.array([0.004, -0.003, -0.16], dtype=np.float64)
        normal = np.array([0.20, -0.08, 0.976], dtype=np.float64)
        normal /= np.linalg.norm(normal)
        baseline = right.camera_center_world_m - left.camera_center_world_m
        horizontal = baseline - float(np.dot(baseline, normal)) * normal
        horizontal /= np.linalg.norm(horizontal)
        vertical = np.cross(normal, horizontal)
        vertical /= np.linalg.norm(vertical)

        left_rgb, left_mask = render_front_rim(
            left,
            expected,
            horizontal,
            vertical,
            mask_shift_px=-3.0,
        )
        right_rgb, right_mask = render_front_rim(
            right,
            expected,
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
            float(np.linalg.norm(result.center_world_m - expected)),
            0.00075,
        )
        self.assertLess(result.ray_gap_m, 0.0005)

    def test_recessed_semantic_masks_do_not_define_depth(self):
        left = SyntheticCamera((-0.02, 0.0, 0.0))
        right = SyntheticCamera((0.02, 0.0, 0.0))
        expected = np.array([0.0, 0.0, -0.16], dtype=np.float64)
        horizontal = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        vertical = np.array([0.0, 1.0, 0.0], dtype=np.float64)

        left_rgb, left_mask = render_front_rim(
            left, expected, horizontal, vertical, mask_shift_px=-5.0
        )
        right_rgb, right_mask = render_front_rim(
            right, expected, horizontal, vertical, mask_shift_px=5.0
        )

        result = estimate_stereo_aperture_center(
            left_rgb=left_rgb,
            right_rgb=right_rgb,
            left_mask=left_mask,
            right_mask=right_mask,
            left_camera=left,
            right_camera=right,
        )
        self.assertLess(abs(float(result.center_world_m[2] - expected[2])), 0.00075)


if __name__ == "__main__":
    unittest.main()
