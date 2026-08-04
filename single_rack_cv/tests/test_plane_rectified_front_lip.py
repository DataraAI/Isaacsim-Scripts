#!/usr/bin/env python3

from __future__ import annotations

import unittest

import cv2
import numpy as np

from plane_rectified_fitting import fit_rectified_front_lip
from plane_rectified_geometry import build_plane_frame
from plane_rectified_types import PlaneFrame, RectifiedEye


class _Camera:
    def __init__(self, world_from_camera):
        self.world_from_camera = np.asarray(world_from_camera, dtype=np.float64)


class PlaneRectifiedFrontLipTests(unittest.TestCase):
    @staticmethod
    def _rectified(mask_notch_shift: int = 0) -> RectifiedEye:
        height, width = 280, 360
        rgb = np.full((height, width, 3), 210, dtype=np.uint8)
        cv2.rectangle(rgb, (66, 70), (294, 210), (25, 25, 25), -1)
        # A stronger recessed edge must not replace the exterior physical lip.
        cv2.rectangle(rgb, (90, 95), (270, 190), (0, 0, 0), 3)

        mask = np.zeros((height, width), dtype=np.uint8)
        notch_left = 120 + int(mask_notch_shift)
        notch_right = 220 + int(mask_notch_shift)
        polygon = np.array(
            [
                [66, 90],
                [notch_left, 90],
                [notch_left, 55],
                [notch_right, 55],
                [notch_right, 90],
                [294, 90],
                [294, 210],
                [66, 210],
            ],
            dtype=np.int32,
        )
        cv2.fillPoly(mask, [polygon], 255)
        return PlaneRectifiedFrontLipTests._with_rgb_and_mask(rgb, mask)

    @staticmethod
    def _with_rgb_and_mask(rgb: np.ndarray, mask: np.ndarray) -> RectifiedEye:
        height, width = rgb.shape[:2]
        y, x = np.indices((height, width))
        frame = PlaneFrame(
            origin_world_m=np.zeros(3),
            axis_u_world=np.array([1.0, 0.0, 0.0]),
            axis_v_world=np.array([0.0, 1.0, 0.0]),
            normal_world=np.array([0.0, 0.0, 1.0]),
        )
        return RectifiedEye(
            rgb=rgb,
            mask=np.asarray(mask) > 0,
            visible=np.ones((height, width), dtype=bool),
            map_u_px=x.astype(np.float64),
            map_v_px=y.astype(np.float64),
            minimum_uv_m=np.array([-0.009, -0.007]),
            resolution_m=0.00005,
            plane_frame=frame,
            camera=None,
        )

    def test_camera_derived_plane_frame_is_orthonormal(self):
        left = _Camera(np.eye(4))
        right = _Camera(np.eye(4))
        frame = build_plane_frame(
            left,
            right,
            np.zeros(3),
            np.array([0.0, 0.0, 1.0]),
        )
        self.assertAlmostEqual(np.linalg.norm(frame.axis_u_world), 1.0, places=12)
        self.assertAlmostEqual(np.linalg.norm(frame.axis_v_world), 1.0, places=12)
        self.assertAlmostEqual(float(frame.axis_u_world @ frame.axis_v_world), 0.0, places=12)

    def test_stronger_recessed_edge_does_not_replace_front_lip(self):
        fit = fit_rectified_front_lip(self._rectified())
        self.assertLessEqual(fit.residual_px, 1.5)
        self.assertGreaterEqual(fit.width_m, 0.70 * 0.0114)
        self.assertLessEqual(fit.width_m, 1.30 * 0.0114)
        self.assertGreaterEqual(fit.height_m, 0.70 * 0.0070)
        self.assertLessEqual(fit.height_m, 1.30 * 0.0070)
        self.assertLess(abs(float(fit.center_uv_m[0])), 0.0001)

    def test_latch_notch_mask_shift_does_not_move_rgb_center(self):
        baseline = fit_rectified_front_lip(self._rectified(0))
        shifted = fit_rectified_front_lip(self._rectified(18))
        self.assertLess(
            float(np.linalg.norm(baseline.center_uv_m - shifted.center_uv_m)),
            0.0001,
        )

    def test_moderate_brightness_and_contrast_change_preserves_center(self):
        baseline_input = self._rectified()
        baseline = fit_rectified_front_lip(baseline_input)
        changed_rgb = np.clip(
            baseline_input.rgb.astype(np.float64) * 0.80 + 15.0,
            0.0,
            255.0,
        ).astype(np.uint8)
        changed = fit_rectified_front_lip(
            self._with_rgb_and_mask(changed_rgb, baseline_input.mask)
        )
        self.assertLess(
            float(np.linalg.norm(baseline.center_uv_m - changed.center_uv_m)),
            0.0001,
        )

    def test_missing_physical_boundary_fails_closed(self):
        rectified = self._rectified()
        damaged = rectified.rgb.copy()
        damaged[:, 260:, :] = 210
        with self.assertRaises(RuntimeError):
            fit_rectified_front_lip(
                self._with_rgb_and_mask(damaged, rectified.mask)
            )

    def test_nonparallel_physical_edges_fail_closed(self):
        height, width = 280, 360
        rgb = np.full((height, width, 3), 210, dtype=np.uint8)
        cv2.fillPoly(
            rgb,
            [np.array([[66, 70], [294, 70], [260, 210], [66, 210]])],
            (25, 25, 25),
        )
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(
            mask,
            [
                np.array(
                    [
                        [66, 90],
                        [120, 90],
                        [120, 55],
                        [220, 55],
                        [220, 90],
                        [294, 90],
                        [260, 210],
                        [66, 210],
                    ],
                    dtype=np.int32,
                )
            ],
            255,
        )
        with self.assertRaisesRegex(RuntimeError, "parallel"):
            fit_rectified_front_lip(self._with_rgb_and_mask(rgb, mask))

    def test_implausibly_narrow_rgb_mouth_fails_closed(self):
        height, width = 280, 360
        rgb = np.full((height, width, 3), 210, dtype=np.uint8)
        cv2.rectangle(rgb, (105, 70), (255, 210), (25, 25, 25), -1)
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(
            mask,
            [
                np.array(
                    [
                        [105, 90],
                        [140, 90],
                        [140, 55],
                        [220, 55],
                        [220, 90],
                        [255, 90],
                        [255, 210],
                        [105, 210],
                    ],
                    dtype=np.int32,
                )
            ],
            255,
        )
        with self.assertRaisesRegex(RuntimeError, "width"):
            fit_rectified_front_lip(self._with_rgb_and_mask(rgb, mask))


if __name__ == "__main__":
    unittest.main()
