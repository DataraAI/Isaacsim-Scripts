from __future__ import annotations

import unittest
from pathlib import Path

from config import CONFIG
from vision.front_lip_calibration import (
    REJECTED_OUTER_BEZEL_MEDIAN_M,
    REJECTED_OUTER_BEZEL_POPULATION_STD_M,
    REJECTED_OUTER_BEZEL_SAMPLE_COUNT,
    VISIBLE_FRONT_LIP_HEIGHT_M,
    VISIBLE_FRONT_LIP_SEARCH_WIDTH_M,
    VISIBLE_FRONT_LIP_WIDTH_M,
)


ROOT = Path(__file__).resolve().parents[1]
VISION_ROOT = ROOT / "vision"


class VisibleFrontLipCalibrationTests(unittest.TestCase):
    def test_production_uses_physical_width_and_bounded_search_hypotheses(self):
        self.assertAlmostEqual(VISIBLE_FRONT_LIP_WIDTH_M, 0.0129, places=12)
        self.assertAlmostEqual(VISIBLE_FRONT_LIP_HEIGHT_M, 0.0070, places=12)
        self.assertAlmostEqual(
            VISIBLE_FRONT_LIP_SEARCH_WIDTH_M,
            0.0114,
            places=12,
        )

        self.assertEqual(REJECTED_OUTER_BEZEL_SAMPLE_COUNT, 91)
        self.assertAlmostEqual(REJECTED_OUTER_BEZEL_MEDIAN_M, 0.015287, places=12)
        self.assertAlmostEqual(
            REJECTED_OUTER_BEZEL_POPULATION_STD_M,
            0.0002513259929648458,
            places=15,
        )
        self.assertNotAlmostEqual(
            VISIBLE_FRONT_LIP_WIDTH_M,
            REJECTED_OUTER_BEZEL_MEDIAN_M,
            places=4,
        )

        source = (ROOT / "main.py").read_text(encoding="utf-8")
        self.assertIn("from vision.front_lip_calibration import (", source)
        self.assertIn(
            "aperture_width_m=VISIBLE_FRONT_LIP_WIDTH_M",
            source,
        )
        self.assertIn(
            "aperture_height_m=VISIBLE_FRONT_LIP_HEIGHT_M",
            source,
        )
        self.assertIn(
            "search_width_m=VISIBLE_FRONT_LIP_SEARCH_WIDTH_M",
            source,
        )
        self.assertNotIn(
            "aperture_width_m=CONFIG.perception.port_width_m",
            source,
        )
        self.assertNotIn(
            "aperture_height_m=CONFIG.perception.port_height_m",
            source,
        )

        stereo_source = (VISION_ROOT / "plane_rectified_front_lip.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("fit_rectified_front_lip_width_prior", stereo_source)

    def test_internal_port_model_is_not_silently_redefined(self):
        self.assertAlmostEqual(CONFIG.perception.port_width_m, 0.0114, places=12)
        self.assertAlmostEqual(CONFIG.perception.port_height_m, 0.0070, places=12)


if __name__ == "__main__":
    unittest.main()
