from __future__ import annotations

from pathlib import Path
import unittest

from config import CONFIG


ROOT = Path(__file__).resolve().parents[1]


class TwoStageRuntimeWiringTests(unittest.TestCase):
    def test_exact_two_stage_configuration(self):
        self.assertTrue(CONFIG.insertion.enabled)
        self.assertEqual(CONFIG.visual_servo.preinsert_standoff_m, 0.050)
        self.assertEqual(CONFIG.insertion.total_depth_m, 0.060)
        self.assertEqual(CONFIG.insertion.coarse_approach_depth_m, 0.040)
        self.assertEqual(CONFIG.insertion.coarse_step_size_m, 0.005)
        self.assertEqual(CONFIG.insertion.opening_depth_m, 0.050)
        self.assertEqual(CONFIG.insertion.step_size_m, 0.0005)
        self.assertEqual(CONFIG.insertion.settle_position_tolerance_m, 0.0003)
        self.assertEqual(CONFIG.insertion.required_settled_frames, 6)
        self.assertEqual(CONFIG.insertion.step_timeout_s, 2.0)

    def test_runtime_passes_both_stages_and_logs_port_depth(self):
        source = (ROOT / "cable_runtime" / "__init__.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("coarse_approach_depth_m", source)
        self.assertIn("coarse_step_size_m", source)
        self.assertIn("opening_depth_m", source)
        self.assertIn("total_step_count", source)
        self.assertIn("command.stage.value", source)
        self.assertIn("commanded_port_depth_m", source)
        self.assertIn("actual_port_depth_m", source)
        self.assertIn("40 mm coarse approach", source)
        self.assertIn("20 mm fine entry", source)

    def test_readme_states_actual_port_entry_geometry(self):
        source = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("40 mm coarse approach", source)
        self.assertIn("20 mm fine motion", source)
        self.assertIn("eight 5 mm", source)
        self.assertIn("forty 0.5 mm", source)
        self.assertIn("48 commands", source)
        self.assertIn("10 mm inside the opening", source)


if __name__ == "__main__":
    unittest.main()
