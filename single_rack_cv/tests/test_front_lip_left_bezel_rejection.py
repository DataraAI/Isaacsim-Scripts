from __future__ import annotations

import unittest

import cv2
import numpy as np

from front_lip_calibration import (
    VISIBLE_FRONT_LIP_HEIGHT_M,
    VISIBLE_FRONT_LIP_SEARCH_WIDTH_M,
    VISIBLE_FRONT_LIP_WIDTH_M,
)
from plane_rectified_fitting import fit_rectified_front_lip
from plane_rectified_types import PlaneFrame, RectifiedEye


class FrontLipLeftBezelRejectionTests(unittest.TestCase):
    @staticmethod
    def _left_eye_outer_bezel_trap() -> RectifiedEye:
        """Synthetic version of the August 6 left-eye failure.

        The semantic mask is slightly narrower than the physical mouth. A
        second dark vertical edge sits farther left in the white bezel. The
        old 11.4 mm search span reached that false edge and selected it first.
        """

        height, width = 320, 460
        rgb = np.full((height, width, 3), 90, dtype=np.uint8)
        cv2.rectangle(rgb, (20, 30), (430, 290), (165, 165, 165), -1)

        # False outer-bezel shadow: approximately 3 mm farther left than the
        # physical lower-mouth wall at the production 0.05 mm/px resolution.
        cv2.rectangle(rgb, (35, 85), (50, 235), (70, 70, 70), -1)

        # Physical lower mouth. Its center is pixel (221, 160).
        cv2.rectangle(rgb, (96, 85), (346, 235), (35, 35, 35), -1)

        mask = np.zeros((height, width), dtype=np.uint8)
        mouth_mask = np.array(
            [
                [106, 100],
                [170, 100],
                [170, 70],
                [270, 70],
                [270, 100],
                [336, 100],
                [336, 230],
                [106, 230],
            ],
            dtype=np.int32,
        )
        cv2.fillPoly(mask, [mouth_mask], 255)

        y, x = np.indices((height, width))
        resolution_m = 0.00005
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
            minimum_uv_m=np.array(
                [-221.0 * resolution_m, -160.0 * resolution_m]
            ),
            resolution_m=resolution_m,
            plane_frame=frame,
            camera=None,
        )

    def test_production_search_excludes_farther_left_bezel_edge(self):
        fit = fit_rectified_front_lip(
            self._left_eye_outer_bezel_trap(),
            aperture_width_m=VISIBLE_FRONT_LIP_WIDTH_M,
            aperture_height_m=VISIBLE_FRONT_LIP_HEIGHT_M,
            search_width_m=VISIBLE_FRONT_LIP_SEARCH_WIDTH_M,
        )

        self.assertLess(
            abs(float(fit.center_uv_m[0])),
            0.0002,
            "The left-eye fit selected the farther outer-bezel edge.",
        )
        self.assertGreater(fit.width_m, 0.0120)
        self.assertLess(fit.width_m, 0.0135)


if __name__ == "__main__":
    unittest.main()
