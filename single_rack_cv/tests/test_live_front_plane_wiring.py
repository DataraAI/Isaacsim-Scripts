from __future__ import annotations

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class LiveFrontPlaneWiringTests(unittest.TestCase):
    def test_highres_enables_front_plane_control(self):
        source = (ROOT / "highres_config.py").read_text(encoding="utf-8")
        self.assertIn("front_rim=replace(", source)
        self.assertIn("enabled=True", source)

    def test_main_refines_before_motion_and_debug(self):
        source = (ROOT / "main.py").read_text(encoding="utf-8")
        refine = source.index("refine_live_observation_to_front_plane(")
        observe = source.index("runtime.observe_visual_servo(observation)")
        debug = source.index("debug.handle(")
        self.assertLess(refine, observe)
        self.assertLess(refine, debug)

    def test_main_holds_on_refinement_failure(self):
        source = (ROOT / "main.py").read_text(encoding="utf-8")
        self.assertIn("runtime.note_perception_failure()", source)
        self.assertIn("RGB stereo capture", source)


if __name__ == "__main__":
    unittest.main()
