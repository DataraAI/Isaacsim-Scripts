#!/usr/bin/env python3

from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class InsertionOnlyRuntimeWiringTests(unittest.TestCase):
    def test_production_export_bypasses_handoff_position_wrapper(self):
        main_source = (ROOT / "main.py").read_text()
        export_source = (ROOT / "full_insertion_runtime.py").read_text()

        self.assertIn("from full_insertion_runtime import", main_source)
        self.assertIn("from full_insertion_base_runtime import", export_source)
        self.assertNotIn(
            "from handoff_position_hold_runtime import",
            export_source,
        )
        self.assertNotIn("apply_tool_goal_trim", export_source)
        self.assertNotIn("_handoff_goal_world_m", export_source)

    def test_exact_calibrated_offset_is_installed_on_insertion_controller(self):
        source = (ROOT / "full_insertion_runtime.py").read_text()

        self.assertIn("[0.0, -0.00030, -0.00045]", source)
        self.assertIn(
            "TrimmedConsecutivePoseInsertionController(",
            source,
        )
        self.assertIn(
            "target_offset_world_m=_INSERTION_TARGET_OFFSET_WORLD_M",
            source,
        )
        self.assertIn(
            "self._insertion_axis_adapter = ExplicitInsertionAxisAdapter(",
            source,
        )

    def test_log_states_perception_handoff_and_depth_are_unchanged(self):
        source = (ROOT / "full_insertion_runtime.py").read_text()

        self.assertIn("INSERTION TARGET CALIBRATION ACTIVE", source)
        self.assertIn("perception-derived port point: unchanged", source)
        self.assertIn("50 mm handoff goal: unchanged", source)
        self.assertIn("insertion depth schedule: unchanged at 48 commands", source)
        self.assertIn(
            "lateral drift reference: calibrated insertion line",
            source,
        )
        self.assertIn(
            "lateral deviation abort limit:",
            source,
        )
        self.assertNotIn("remaining lateral budget", source)

    def test_calibration_controller_references_calibrated_line_for_drift(self):
        source = (ROOT / "insertion_target_trim.py").read_text()

        self.assertIn("_MAXIMUM_INSERTION_CALIBRATION_M = 0.001", source)
        self.assertIn("def _calibrated_lateral_drift_m", source)
        self.assertIn("calibrated_origin", source)
        self.assertIn("def _metrics(self, sample)", source)
        self.assertIn("lateral_drift_m=", source)

    def test_old_handoff_experiment_is_not_the_production_export(self):
        export_source = (ROOT / "full_insertion_runtime.py").read_text()
        dormant_source = (ROOT / "handoff_position_hold_runtime.py").read_text()

        self.assertNotIn("FROZEN HANDOFF POSITION HOLD ACTIVE", export_source)
        self.assertIn("FROZEN HANDOFF POSITION HOLD ACTIVE", dormant_source)
        self.assertIn("_advance_handoff_if_settled", dormant_source)


if __name__ == "__main__":
    unittest.main()
