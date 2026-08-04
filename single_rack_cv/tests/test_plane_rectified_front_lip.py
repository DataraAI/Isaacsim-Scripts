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
        y, x = np.indices((height, width))
        frame = PlaneFrame(
            origin_world_m=np.zeros(3),
            axis_u_world=np.array([1.0, 0.0, 0.0]),
            axis_v_world=np.array([0.0, 1.0, 0.0]),
            normal_world=np.array([0.0, 0.0, 1.0]),
        )
        return RectifiedEye(
            rgb=rgb,
            mask=mask > 0,
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

    def test_missing_physical_boundary_fails_closed(self):
        rectified = self._rectified()
        damaged = rectified.rgb.copy()
        damaged[:, 280:, :] = 25
        invalid = RectifiedEye(
            rgb=damaged,
            mask=rectified.mask,
            visible=rectified.visible,
            map_u_px=rectified.map_u_px,
            map_v_px=rectified.map_v_px,
            minimum_uv_m=rectified.minimum_uv_m,
            resolution_m=rectified.resolution_m,
            plane_frame=rectified.plane_frame,
            camera=None,
        )
        with self.assertRaises(RuntimeError):
            fit_rectified_front_lip(invalid)


if __name__ == "__main__":
    unittest.main()
