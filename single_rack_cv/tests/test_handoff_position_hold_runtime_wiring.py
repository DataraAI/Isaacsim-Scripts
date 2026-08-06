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

    def test_exact_trim_is_installed_on_insertion_controller(self):
        source = (ROOT / "full_insertion_runtime.py").read_text()

        self.assertIn("[0.0, -0.00015, -0.00025]", source)
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

    def test_log_states_perception_and_handoff_are_unchanged(self):
        source = (ROOT / "full_insertion_runtime.py").read_text()

        self.assertIn("INSERTION TARGET TRIM ACTIVE", source)
        self.assertIn("perception-derived port point: unchanged", source)
        self.assertIn("50 mm handoff goal: unchanged", source)
        self.assertIn("insertion depth schedule: unchanged at 48 commands", source)
        self.assertIn("trim remains visible to the existing drift guard", source)

    def test_old_handoff_experiment_is_not_the_production_export(self):
        export_source = (ROOT / "full_insertion_runtime.py").read_text()
        dormant_source = (ROOT / "handoff_position_hold_runtime.py").read_text()

        self.assertNotIn("FROZEN HANDOFF POSITION HOLD ACTIVE", export_source)
        self.assertIn("FROZEN HANDOFF POSITION HOLD ACTIVE", dormant_source)
        self.assertIn("_advance_handoff_if_settled", dormant_source)


if __name__ == "__main__":
    unittest.main()
