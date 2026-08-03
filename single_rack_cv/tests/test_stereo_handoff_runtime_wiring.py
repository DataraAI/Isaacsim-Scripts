#!/usr/bin/env python3

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
RUNTIME_PATH = ROOT / "stereo_handoff_runtime.py"
MAIN_PATH = ROOT / "main.py"
CONFIG_PATH = ROOT / "config.py"


class StereoHandoffWiringTests(unittest.TestCase):
    def test_main_selects_handoff_runtime(self):
        source = MAIN_PATH.read_text(
            encoding="utf-8"
        )

        self.assertIn(
            "AngledHandStereoHandoffRuntime "
            "as CableMountedSimulationRuntime",
            source,
        )

    def test_runtime_waits_for_acquired_nearby_goal_before_handoff(self):
        source = RUNTIME_PATH.read_text(
            encoding="utf-8"
        )

        self.assertIn(
            "_RECENT_GOAL_WINDOW = 3",
            source,
        )
        self.assertIn(
            "_MAXIMUM_HANDOFF_DISTANCE_M = 0.035",
            source,
        )
        self.assertIn(
            "select_recent_bounded_goal",
            source,
        )
        self.assertIn(
            "not state.acquired",
            source,
        )
        self.assertIn(
            "newest stable stereo goals entered",
            source,
        )

        correction = source.index("correction_world_m = np.asarray(")
        visual_step = source.index(
            "super().observe_visual_servo(",
            correction,
        )
        acquired_gate = source.index(
            "not state.acquired",
            visual_step,
        )
        append = source.index(
            "self._stereo_goal_candidates.append(",
            acquired_gate,
        )
        self.assertLess(
            visual_step,
            acquired_gate,
            "The base controller must establish acquisition state before "
            "handoff samples are accepted.",
        )
        self.assertLess(
            acquired_gate,
            append,
            "World-goal samples must be rejected until stereo acquisition "
            "has completed.",
        )

    def test_runtime_keeps_safety_limits(self):
        source = RUNTIME_PATH.read_text(
            encoding="utf-8"
        )

        self.assertIn(
            "_MAXIMUM_GOAL_SPREAD_M = 0.002",
            source,
        )
        self.assertIn(
            "_MAXIMUM_HANDOFF_STEP_M = 0.005",
            source,
        )
        self.assertIn(
            "destination: stereo-derived 50 mm pre-insert standoff",
            source,
        )
        self.assertIn(
            "orientation: unchanged horizontal plug",
            source,
        )

    def test_camera_remains_on_last_visible_mount(self):
        source = CONFIG_PATH.read_text(
            encoding="utf-8"
        )

        self.assertIn(
            "local_y_rotation_deg: float = 186.0248",
            source,
        )
        self.assertIn(
            "        0.04,\n"
            "        -0.020,\n"
            "        0.025,",
            source,
        )
        self.assertIn(
            "        0.04,\n"
            "        +0.020,\n"
            "        0.025,",
            source,
        )


if __name__ == "__main__":
    unittest.main()
