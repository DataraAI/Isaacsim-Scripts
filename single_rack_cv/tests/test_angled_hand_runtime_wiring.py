from __future__ import annotations

from pathlib import Path
import unittest

from angled_hand_config import ANGLED_HAND_CONFIG

ROOT = Path(__file__).resolve().parents[1]
RUNTIME_PATH = ROOT / "angled_hand_runtime.py"
MAIN_PATH = ROOT / "main.py"


class AngledHandRuntimeWiringTests(unittest.TestCase):
    def test_shared_geometry_defaults_match_requested_pose(self):
        self.assertEqual(
            ANGLED_HAND_CONFIG.hand_downward_pitch_deg,
            30.0,
        )
        self.assertEqual(ANGLED_HAND_CONFIG.palm_roll_deg, 180.0)
        self.assertEqual(ANGLED_HAND_CONFIG.palm_roll_tolerance_deg, 1.0)

    def test_main_selects_the_angled_runtime(self):
        source = MAIN_PATH.read_text(encoding="utf-8")
        self.assertIn(
            "AngledHandCableRuntime as CableMountedSimulationRuntime",
            source,
        )
        self.assertIn("runtime = CableMountedSimulationRuntime(", source)

    def test_runtime_applies_pitch_and_palm_roll_before_scene_construction(self):
        source = RUNTIME_PATH.read_text(encoding="utf-8")
        pitch_index = source.index("hand_downward_pitch_deg")
        roll_index = source.index("palm_roll_deg")
        super_index = source.index(
            "super().__init__(simulation_app=simulation_app, cfg=pitched_cfg)"
        )
        self.assertLess(pitch_index, super_index)
        self.assertLess(roll_index, super_index)
        self.assertIn("compute_pitched_hand_from_tool_rotation", source)
        self.assertIn("palm_roll_deg=palm_roll_deg", source)
        self.assertIn("ExplicitInsertionAxisAdapter", source)

    def test_runtime_rejects_the_flipped_palm_bug(self):
        source = RUNTIME_PATH.read_text(encoding="utf-8")
        self.assertIn("palm_roll_error_deg", source)
        self.assertIn("palm_roll_tolerance_deg", source)
        self.assertIn(
            "palm roll does not match the previous working pose",
            source,
        )

    def test_runtime_separates_live_plug_axis_from_tool_orientation(self):
        source = RUNTIME_PATH.read_text(encoding="utf-8")
        self.assertIn("def _live_plug_tip_and_axis", source)
        self.assertIn(
            "self._insertion_axis_adapter.set_axis_world",
            source,
        )
        self.assertIn("def _sample_mount_validation_live", source)
        self.assertIn("plug_horizontal_error_deg", source)

    def test_runtime_does_not_override_camera_poses(self):
        source = RUNTIME_PATH.read_text(encoding="utf-8")
        self.assertNotIn("left_camera_sensor.set", source)
        self.assertNotIn("right_camera_sensor.set", source)
        self.assertNotIn("camera_prim.Set", source)


if __name__ == "__main__":
    unittest.main()
