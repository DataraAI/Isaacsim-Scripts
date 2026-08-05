from __future__ import annotations

import unittest

import cv2
import numpy as np

from front_lip_calibration import (
    VISIBLE_FRONT_LIP_HEIGHT_M,
    VISIBLE_FRONT_LIP_WIDTH_M,
)
from plane_rectified_fitting import fit_rectified_front_lip
from plane_rectified_types import PlaneFrame, RectifiedEye


class VisibleFrontLipGeometryTests(unittest.TestCase):
    @staticmethod
    def _visible_front_lip() -> RectifiedEye:
        height, width = 300, 440
        rgb = np.full((height, width, 3), 210, dtype=np.uint8)
        cv2.rectangle(rgb, (67, 70), (373, 210), (25, 25, 25), -1)
        cv2.rectangle(rgb, (110, 95), (330, 190), (0, 0, 0), 3)

        mask = np.zeros((height, width), dtype=np.uint8)
        polygon = np.array(
            [
                [67, 90],
                [140, 90],
                [140, 55],
                [300, 55],
                [300, 90],
                [373, 90],
                [373, 210],
                [67, 210],
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
            minimum_uv_m=np.array([-0.011, -0.007]),
            resolution_m=0.00005,
            plane_frame=frame,
            camera=None,
        )

    def test_visible_rectangle_passes_with_live_calibration(self):
        fit = fit_rectified_front_lip(
            self._visible_front_lip(),
            aperture_width_m=VISIBLE_FRONT_LIP_WIDTH_M,
            aperture_height_m=VISIBLE_FRONT_LIP_HEIGHT_M,
        )

        self.assertLess(abs(fit.width_m - VISIBLE_FRONT_LIP_WIDTH_M), 0.0002)
        # This regression is about the live 15.3 mm width calibration. The
        # synthetic top/bottom gradients produce an approximately 8.0 mm fit,
        # which is valid under the unchanged production 70%-130% height gate.
        self.assertGreaterEqual(fit.height_m, 0.70 * VISIBLE_FRONT_LIP_HEIGHT_M)
        self.assertLessEqual(fit.height_m, 1.30 * VISIBLE_FRONT_LIP_HEIGHT_M)
        self.assertLess(float(np.linalg.norm(fit.center_uv_m)), 0.0001)
        self.assertLessEqual(fit.residual_px, 1.5)

    def test_same_visible_rectangle_is_rejected_by_old_internal_width_prior(self):
        with self.assertRaisesRegex(RuntimeError, "width"):
            fit_rectified_front_lip(
                self._visible_front_lip(),
                aperture_width_m=0.0114,
                aperture_height_m=VISIBLE_FRONT_LIP_HEIGHT_M,
            )


if __name__ == "__main__":
    unittest.main()
