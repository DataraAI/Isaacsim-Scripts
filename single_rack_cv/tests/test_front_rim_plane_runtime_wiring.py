from __future__ import annotations

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class FrontRimPlaneRuntimeWiringTests(unittest.TestCase):
    def test_runtime_adapter_uses_outer_bezel_plane_estimator(self):
        source = (ROOT / "live_control_projective.py").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            "from outer_bezel_center import "
            "estimate_outer_bezel_aperture_center",
            source,
        )
        self.assertNotIn("from stereo_front_rim_plane import", source)
        self.assertNotIn("estimate_stereo_aperture_center(", source)

    def test_main_routes_live_observations_through_corrected_adapter(self):
        source = (ROOT / "main.py").read_text(encoding="utf-8")
        self.assertIn(
            "from live_control_projective import refine_live_observation",
            source,
        )
        refine = source.index("refine_live_observation(")
        observe = source.index("runtime.observe_visual_servo(observation)")
        self.assertLess(refine, observe)


if __name__ == "__main__":
    unittest.main()
