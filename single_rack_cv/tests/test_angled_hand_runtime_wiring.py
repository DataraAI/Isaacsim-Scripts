from __future__ import annotations

from pathlib import Path
import unittest

from angled_hand_config import ANGLED_HAND_CONFIG

ROOT = Path(__file__).resolve().parents[1]
RUNTIME_PATH = ROOT / "angled_hand_runtime.py"
MAIN_PATH = ROOT / "main.py"


class AngledHandRuntimeWiringTests(unittest.TestCase):
    def test_single_shared_pitch_defaults_to_thirty_degrees(self):
        self.assertEqual(
            ANGLED_HAND_CONFIG.hand_downward_pitch_deg,
            30.0,
        )

    def test_main_selects_the_angled_runtime(self):
        source = MAIN_PATH.read_text(encoding="utf-8")
        self.assertIn(
            "AngledHandCableRuntime as CableMountedSimulationRuntime",
            source,
        )
        self.assertIn("runtime = CableMountedSimulationRuntime(", source)

    def test_runtime_applies_pitch_before_base_scene_construction(self):
        source = RUNTIME_PATH.read_text(encoding="utf-8")
        pitch_index = source.index("hand_downward_pitch_deg")
        super_index = source.index(
            "super().__init__(simulation_app=simulation_app, cfg=pitched_cfg)"
        )
        self.assertLess(pitch_index, super_index)
        self.assertIn("compute_pitched_hand_from_tool_rotation", source)
        self.assertIn("ExplicitInsertionAxisAdapter", source)

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
