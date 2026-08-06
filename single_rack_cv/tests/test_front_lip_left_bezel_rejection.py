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
    def _left_eye_three_edge_trap() -> RectifiedEye:
        """Model the outer-bezel, physical-mouth, and inner-cavity edges.

        The narrow 5 mm search cannot reach the physical left mouth wall and
        therefore pairs the inner cavity with the physical right wall. The
        old wide search reaches the physical wall but blindly chooses the
        farther outer-bezel edge first. Production must inspect all qualified
        side edges and select the pair consistent with the physical width.
        """

        height, width = 320, 440
        rgb = np.full((height, width, 3), 90, dtype=np.uint8)
        cv2.rectangle(rgb, (5, 30), (420, 290), (165, 165, 165), -1)

        # Farther outer-bezel shadow: false negative edge around x=20.
        cv2.rectangle(rgb, (20, 85), (34, 235), (70, 70, 70), -1)

        # Physical visible mouth: 258 px = 12.9 mm at 0.05 mm/px.
        cv2.rectangle(rgb, (55, 85), (313, 235), (35, 35, 35), -1)

        # Inner-cavity edge exposed by the too-narrow search. It has the same
        # gradient sign as the physical left wall but produces a ~9.7 mm pair.
        cv2.line(rgb, (120, 105), (120, 215), (5, 5, 5), 4)

        mask = np.zeros((height, width), dtype=np.uint8)
        mouth_mask = np.array(
            [
                [106, 100],
                [165, 100],
                [165, 70],
                [247, 70],
                [247, 100],
                [306, 100],
                [306, 230],
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
        physical_center_px = np.array([184.0, 160.0])
        return RectifiedEye(
            rgb=rgb,
            mask=mask > 0,
            visible=np.ones((height, width), dtype=bool),
            map_u_px=x.astype(np.float64),
            map_v_px=y.astype(np.float64),
            minimum_uv_m=-physical_center_px * resolution_m,
            resolution_m=resolution_m,
            plane_frame=frame,
            camera=None,
        )

    def test_production_selects_physical_pair_between_outer_and_inner_edges(self):
        fit = fit_rectified_front_lip(
            self._left_eye_three_edge_trap(),
            aperture_width_m=VISIBLE_FRONT_LIP_WIDTH_M,
            aperture_height_m=VISIBLE_FRONT_LIP_HEIGHT_M,
            search_width_m=VISIBLE_FRONT_LIP_SEARCH_WIDTH_M,
        )

        self.assertLess(
            abs(float(fit.center_uv_m[0])),
            0.0002,
            "The fit did not select the physical visible-mouth side pair.",
        )
        self.assertLessEqual(
            abs(fit.width_m - 0.0129),
            0.0003,
            "The fit selected the outer bezel or inner cavity width.",
        )


if __name__ == "__main__":
    unittest.main()
