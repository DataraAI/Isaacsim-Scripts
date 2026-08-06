#!/usr/bin/env python3

from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ProductionRuntimeWiringTests(unittest.TestCase):
    def test_production_composes_position_hold_before_insertion_calibration(self):
        main_source = (ROOT / "main.py").read_text()
        export_source = (ROOT / "full_insertion_runtime.py").read_text()
        hold_source = (ROOT / "handoff_position_hold_runtime.py").read_text()

        self.assertIn("from full_insertion_runtime import", main_source)
        self.assertIn(
            "from handoff_position_hold_runtime import (",
            export_source,
        )
        self.assertNotIn(
            "from full_insertion_base_runtime import (",
            export_source,
        )
        self.assertIn(
            "from full_insertion_base_runtime import (",
            hold_source,
        )

    def test_position_hold_does_not_move_the_camera_derived_handoff_goal(self):
        source = (ROOT / "handoff_position_hold_runtime.py").read_text()

        self.assertIn("FROZEN HANDOFF POSITION HOLD ACTIVE", source)
        self.assertIn("frozen physical ToolCenter goal: unchanged", source)
        self.assertIn("def update_visual_servo_completion", source)
        self.assertNotIn("apply_tool_goal_trim", source)
        self.assertNotIn("_TOOL_GOAL_LEFT_TRIM_M", source)
        self.assertNotIn("_TOOL_GOAL_DOWNWARD_TRIM_M", source)
        self.assertNotIn("FINAL TOOLCENTER GOAL TRIM APPLIED", source)
        self.assertNotIn("def _advance_handoff_if_settled", source)

    def test_position_hold_keeps_original_completion_gate_and_resets_command(self):
        source = (ROOT / "handoff_position_hold_runtime.py").read_text()

        self.assertIn(
            "position_error_m > cfg.settle_position_tolerance_m",
            source,
        )
        self.assertIn(
            "state.settled_frame_count >= cfg.required_settled_frames",
            source,
        )
        self.assertIn(
            "self.ik.target.set_world_pose(\n"
            "                position=goal,",
            source,
        )
        self.assertIn("_HANDOFF_POSITION_HOLD_HARD_TIMEOUT_S = 10.0", source)
        self.assertIn("maximum command bias", source)

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
        self.assertIn("lateral deviation abort limit:", source)

    def test_calibration_controller_references_calibrated_line_for_drift(self):
        source = (ROOT / "insertion_target_trim.py").read_text()

        self.assertIn("_MAXIMUM_INSERTION_CALIBRATION_M = 0.001", source)
        self.assertIn("def _calibrated_lateral_drift_m", source)
        self.assertIn("calibrated_origin", source)
        self.assertIn("def _metrics(self, sample)", source)
        self.assertIn("lateral_drift_m=", source)


if __name__ == "__main__":
    unittest.main()
