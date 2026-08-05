#!/usr/bin/env python3

from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class HandoffPositionHoldRuntimeWiringTests(unittest.TestCase):
    def test_production_export_selects_position_hold_wrapper(self):
        main_source = (ROOT / "main.py").read_text()
        export_source = (ROOT / "full_insertion_runtime.py").read_text()
        self.assertIn("from full_insertion_runtime import", main_source)
        self.assertIn("from handoff_position_hold_runtime import", export_source)

    def test_wrapper_keeps_preserved_full_insertion_as_base(self):
        source = (ROOT / "handoff_position_hold_runtime.py").read_text()
        self.assertIn("from full_insertion_base_runtime import", source)
        self.assertIn("class AngledHandStereoHandoffRuntime", source)

    def test_frozen_physical_goal_not_compensated_command_controls_completion(self):
        source = (ROOT / "handoff_position_hold_runtime.py").read_text()
        self.assertIn("physical_error_world_m = goal - actual_position", source)
        self.assertIn("position_error_m = float(np.linalg.norm(physical_error_world_m))", source)
        self.assertNotIn("position_error_m = self._tool_target_position_error_m()", source)

    def test_compensation_is_bounded_and_does_not_relax_settle_gate(self):
        source = (ROOT / "handoff_position_hold_runtime.py").read_text()
        self.assertIn("_HANDOFF_POSITION_HOLD_MAXIMUM_STEP_M = 0.00010", source)
        self.assertIn("_HANDOFF_POSITION_HOLD_MAXIMUM_BIAS_M = 0.00100", source)
        self.assertIn("tolerance_m=cfg.settle_position_tolerance_m", source)
        self.assertIn("update_handoff_position_command", source)
        self.assertIn("position=update.command_position_m", source)

    def test_deadlock_has_a_hard_fail_closed_timeout(self):
        source = (ROOT / "handoff_position_hold_runtime.py").read_text()
        self.assertIn("_HANDOFF_POSITION_HOLD_HARD_TIMEOUT_S = 10.0", source)
        self.assertIn("raise SystemExit", source)


if __name__ == "__main__":
    unittest.main()
