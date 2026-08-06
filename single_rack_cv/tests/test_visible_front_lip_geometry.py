from __future__ import annotations

import unittest

import cv2
import numpy as np

from front_lip_calibration import (
    REJECTED_OUTER_BEZEL_MEDIAN_M,
    VISIBLE_FRONT_LIP_HEIGHT_M,
    VISIBLE_FRONT_LIP_SEARCH_WIDTH_M,
    VISIBLE_FRONT_LIP_WIDTH_M,
)
from plane_rectified_types import PlaneFrame, RectifiedEye
from plane_rectified_width_hypotheses import fit_rectified_front_lip_width_prior


class VisibleFrontLipGeometryTests(unittest.TestCase):
    @staticmethod
    def _visible_front_lip() -> RectifiedEye:
        height, width = 300, 440
        rgb = np.full((height, width, 3), 210, dtype=np.uint8)
        # 258 px at 0.05 mm/px = 12.9 mm physical visible mouth.
        cv2.rectangle(rgb, (91, 70), (349, 210), (25, 25, 25), -1)
        cv2.rectangle(rgb, (125, 95), (315, 190), (0, 0, 0), 3)

        mask = np.zeros((height, width), dtype=np.uint8)
        polygon = np.array(
            [
                [106, 90],
                [150, 90],
                [150, 55],
                [290, 55],
                [290, 90],
                [334, 90],
                [334, 210],
                [106, 210],
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

    def test_physical_visible_rectangle_passes_width_prior(self):
        fit = fit_rectified_front_lip_width_prior(
            self._visible_front_lip(),
            aperture_width_m=VISIBLE_FRONT_LIP_WIDTH_M,
            aperture_height_m=VISIBLE_FRONT_LIP_HEIGHT_M,
            search_width_m=VISIBLE_FRONT_LIP_SEARCH_WIDTH_M,
        )

        self.assertLess(abs(fit.width_m - VISIBLE_FRONT_LIP_WIDTH_M), 0.0002)
        self.assertGreaterEqual(fit.height_m, 0.70 * VISIBLE_FRONT_LIP_HEIGHT_M)
        self.assertLessEqual(fit.height_m, 1.30 * VISIBLE_FRONT_LIP_HEIGHT_M)
        self.assertLess(abs(float(fit.center_uv_m[0])), 0.0001)
        self.assertLessEqual(fit.residual_px, 1.5)

    def test_contaminated_outer_bezel_prior_rejects_physical_mouth(self):
        with self.assertRaisesRegex(RuntimeError, "physical width prior"):
            fit_rectified_front_lip_width_prior(
                self._visible_front_lip(),
                aperture_width_m=REJECTED_OUTER_BEZEL_MEDIAN_M,
                aperture_height_m=VISIBLE_FRONT_LIP_HEIGHT_M,
                search_width_m=VISIBLE_FRONT_LIP_SEARCH_WIDTH_M,
            )


if __name__ == "__main__":
    unittest.main()
