from __future__ import annotations

import inspect
import unittest
from pathlib import Path

from front_lip_calibration import VISIBLE_FRONT_LIP_SEARCH_WIDTH_M
from plane_rectified_width_hypotheses import (
    _search_width_hypotheses,
    fit_rectified_front_lip_width_prior,
)


ROOT = Path(__file__).resolve().parents[1]


class FrontLipSearchCalibrationTests(unittest.TestCase):
    def test_production_search_is_a_bounded_hypothesis_schedule(self):
        self.assertIn(
            "search_width_m",
            inspect.signature(fit_rectified_front_lip_width_prior).parameters,
        )
        widths = _search_width_hypotheses(VISIBLE_FRONT_LIP_SEARCH_WIDTH_M)
        self.assertEqual(len(widths), 5)
        self.assertAlmostEqual(widths[0], 0.0050, places=12)
        self.assertAlmostEqual(widths[-1], 0.0114, places=12)
        self.assertTrue(all(first < second for first, second in zip(widths, widths[1:])))

    def test_width_prior_reaches_both_independent_eye_fitters_in_production(self):
        live_source = (ROOT / "live_control_projective.py").read_text(
            encoding="utf-8"
        )
        projective_source = (
            ROOT / "outer_bezel_projective_center.py"
        ).read_text(encoding="utf-8")
        stereo_source = (ROOT / "plane_rectified_front_lip.py").read_text(
            encoding="utf-8"
        )
        hypothesis_source = (
            ROOT / "plane_rectified_width_hypotheses.py"
        ).read_text(encoding="utf-8")

        self.assertIn("search_width_m=search_width_m", live_source)
        self.assertIn("search_width_m=search_width_m", projective_source)
        self.assertGreaterEqual(
            stereo_source.count("search_width_m=search_width_m"),
            2,
        )
        self.assertIn("fit_rectified_front_lip_width_prior", stereo_source)
        self.assertIn("_fit_single_search", hypothesis_source)
        self.assertIn("_MAXIMUM_WIDTH_PRIOR_ERROR_M = 0.0010", hypothesis_source)


if __name__ == "__main__":
    unittest.main()
