from __future__ import annotations

import inspect
import unittest
from pathlib import Path

import cv2
import numpy as np

from front_lip_calibration import (
    VISIBLE_FRONT_LIP_HEIGHT_M,
    VISIBLE_FRONT_LIP_SEARCH_WIDTH_M,
    VISIBLE_FRONT_LIP_WIDTH_M,
)
from plane_rectified_fitting import fit_rectified_front_lip
from plane_rectified_types import PlaneFrame, RectifiedEye


ROOT = Path(__file__).resolve().parents[1]


class FrontLipSearchCalibrationTests(unittest.TestCase):
    @staticmethod
    def _mouth_with_competing_sloped_bezel() -> RectifiedEye:
        height, width = 320, 760
        rgb = np.full((height, width, 3), 220, dtype=np.uint8)

        # The farther bezel has strong, nonparallel side edges. The physical
        # visible mouth remains a 15.3 mm parallel rectangle. A search radius
        # based on 15.3 mm reaches the bezel; the old 11.4 mm localization
        # radius excludes it and recovers the mouth.
        outer = np.array(
            [
                [90, 50],
                [660, 50],
                [642, 270],
                [108, 270],
            ],
            dtype=np.int32,
        )
        cv2.fillPoly(rgb, [outer], (120, 120, 120))
        cv2.rectangle(rgb, (220, 90), (526, 230), (25, 25, 25), -1)

        mask = np.zeros((height, width), dtype=np.uint8)
        mouth_mask = np.array(
            [
                [220, 105],
                [300, 105],
                [300, 70],
                [446, 70],
                [446, 105],
                [526, 105],
                [526, 230],
                [220, 230],
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
                [-373.0 * resolution_m, -160.0 * resolution_m]
            ),
            resolution_m=resolution_m,
            plane_frame=frame,
            camera=None,
        )

    def test_localization_search_width_is_decoupled_from_visible_width_gate(self):
        self.assertIn(
            "search_width_m",
            inspect.signature(fit_rectified_front_lip).parameters,
            "The visible-width gate is still coupled to side-edge localization.",
        )

        rectified = self._mouth_with_competing_sloped_bezel()

        with self.assertRaises(RuntimeError):
            fit_rectified_front_lip(
                rectified,
                aperture_width_m=VISIBLE_FRONT_LIP_WIDTH_M,
                aperture_height_m=VISIBLE_FRONT_LIP_HEIGHT_M,
                search_width_m=VISIBLE_FRONT_LIP_WIDTH_M,
            )

        fit = fit_rectified_front_lip(
            rectified,
            aperture_width_m=VISIBLE_FRONT_LIP_WIDTH_M,
            aperture_height_m=VISIBLE_FRONT_LIP_HEIGHT_M,
            search_width_m=VISIBLE_FRONT_LIP_SEARCH_WIDTH_M,
        )

        self.assertLess(
            abs(fit.width_m - VISIBLE_FRONT_LIP_WIDTH_M),
            0.0002,
        )
        self.assertLess(abs(float(fit.center_uv_m[0])), 0.0001)

    def test_search_width_reaches_both_eye_fitters_in_production(self):
        live_source = (ROOT / "live_control_projective.py").read_text(
            encoding="utf-8"
        )
        projective_source = (
            ROOT / "outer_bezel_projective_center.py"
        ).read_text(encoding="utf-8")
        stereo_source = (ROOT / "plane_rectified_front_lip.py").read_text(
            encoding="utf-8"
        )

        self.assertIn("search_width_m=search_width_m", live_source)
        self.assertIn("search_width_m=search_width_m", projective_source)
        self.assertGreaterEqual(
            stereo_source.count("search_width_m=search_width_m"),
            2,
        )


if __name__ == "__main__":
    unittest.main()
