from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from control.tool_goal_trim import apply_tool_goal_trim


ROOT = Path(__file__).resolve().parents[1]


class ToolGoalTrimTests(unittest.TestCase):
    def test_moves_only_tool_goal_point_one_five_mm_left_and_point_two_five_mm_down(self):
        original = np.array(
            [0.704262, -0.192331, 1.322690],
            dtype=np.float64,
        )

        trimmed = apply_tool_goal_trim(
            original,
            left_trim_m=0.00015,
            downward_trim_m=0.00025,
        )

        np.testing.assert_allclose(
            trimmed,
            [0.704262, -0.192481, 1.322440],
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            original,
            [0.704262, -0.192331, 1.322690],
            atol=1.0e-12,
        )

    def test_rejects_negative_or_nonfinite_trims(self):
        with self.assertRaisesRegex(ValueError, "left_trim_m"):
            apply_tool_goal_trim(
                [0.7, -0.19, 1.32],
                left_trim_m=-0.00015,
                downward_trim_m=0.00025,
            )
        with self.assertRaisesRegex(ValueError, "downward_trim_m"):
            apply_tool_goal_trim(
                [0.7, -0.19, 1.32],
                left_trim_m=0.00015,
                downward_trim_m=float("nan"),
            )

    def test_legacy_handoff_trim_helper_is_not_used_by_any_runtime(self):
        production_source = (
            ROOT / "runtime" / "full_insertion_runtime.py"
        ).read_text(encoding="utf-8")
        hold_source = (
            ROOT / "runtime" / "handoff_position_hold_runtime.py"
        ).read_text(encoding="utf-8")
        handoff_source = (
            ROOT / "runtime" / "stereo_handoff_runtime.py"
        ).read_text(encoding="utf-8")

        self.assertNotIn("apply_tool_goal_trim(", production_source)
        self.assertNotIn("apply_tool_goal_trim(", hold_source)
        self.assertNotIn("apply_tool_goal_trim(", handoff_source)
        self.assertNotIn("_TOOL_GOAL_LEFT_TRIM_M", hold_source)
        self.assertNotIn("_TOOL_GOAL_DOWNWARD_TRIM_M", hold_source)
        self.assertNotIn("def _advance_handoff_if_settled", hold_source)
        self.assertIn(
            "qualification.opening_position_m.copy()",
            handoff_source,
        )


if __name__ == "__main__":
    unittest.main()
