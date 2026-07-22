from __future__ import annotations

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class RuntimeWiringTests(unittest.TestCase):
    def test_canonical_config_is_high_resolution_front_plane(self):
        source = (ROOT / "config.py").read_text(encoding="utf-8")
        self.assertIn("resolution: tuple[int, int] = (960, 1280)", source)
        self.assertIn("class FrontPlaneRuntimeConfig", source)
        self.assertIn("enabled: bool = True", source)
        self.assertIn("front_plane: FrontPlaneRuntimeConfig", source)
        self.assertNotIn("class FrontRimConfig", source)

    def test_main_refines_before_motion_and_debug(self):
        source = (ROOT / "main.py").read_text(encoding="utf-8")
        self.assertIn("from live_control import refine_live_observation", source)
        refine = source.index("refine_live_observation(")
        observe = source.index("runtime.observe_visual_servo(observation)")
        debug = source.index("debug.handle(")
        self.assertLess(refine, observe)
        self.assertLess(refine, debug)
        self.assertIn("CONFIG.front_plane.enabled", source)

    def test_failure_path_holds_and_reacquires(self):
        source = (ROOT / "main.py").read_text(encoding="utf-8")
        self.assertIn("runtime.note_perception_failure()", source)
        self.assertIn("RGB stereo capture", source)
        self.assertIn("no manual depth offset", source)


if __name__ == "__main__":
    unittest.main()
