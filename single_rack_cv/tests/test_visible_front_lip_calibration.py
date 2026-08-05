from __future__ import annotations

import unittest
from pathlib import Path

from config import CONFIG


ROOT = Path(__file__).resolve().parents[1]


class VisibleFrontLipCalibrationTests(unittest.TestCase):
    def test_production_uses_live_visible_front_lip_calibration(self):
        self.assertAlmostEqual(
            CONFIG.front_plane.visible_front_lip_width_m,
            0.0153,
            places=12,
        )
        self.assertAlmostEqual(
            CONFIG.front_plane.visible_front_lip_height_m,
            0.0070,
            places=12,
        )

        source = (ROOT / "main.py").read_text(encoding="utf-8")
        self.assertIn(
            "aperture_width_m=(\n"
            "                        CONFIG.front_plane.visible_front_lip_width_m\n"
            "                    )",
            source,
        )
        self.assertIn(
            "aperture_height_m=(\n"
            "                        CONFIG.front_plane.visible_front_lip_height_m\n"
            "                    )",
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

    def test_internal_port_model_is_not_silently_redefined(self):
        self.assertAlmostEqual(CONFIG.perception.port_width_m, 0.0114, places=12)
        self.assertAlmostEqual(CONFIG.perception.port_height_m, 0.0070, places=12)


if __name__ == "__main__":
    unittest.main()
