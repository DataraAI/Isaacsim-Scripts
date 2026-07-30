from __future__ import annotations

import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
RUNTIME_PATH = ROOT / "settled_stereo_handoff_runtime.py"
CONFIG_PATH = ROOT / "config.py"


class OrientationHoldRuntimeWiringTests(unittest.TestCase):
    def test_runtime_updates_orientation_target_before_lula_solve(self):
        source = RUNTIME_PATH.read_text(encoding="utf-8")
        tree = ast.parse(source)
        method = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
            and node.name == "update_ik"
        )
        calls = [
            ast.unparse(node)
            for node in ast.walk(method)
            if isinstance(node, ast.Call)
        ]
        self.assertIn(
            "self._update_insertion_orientation_hold_target()",
            calls,
        )
        self.assertIn("super().update_ik()", calls)
        self.assertLess(
            source.index("self._update_insertion_orientation_hold_target()"),
            source.index("super().update_ik()"),
        )

    def test_feedback_is_bounded_and_actual_one_degree_limit_is_unchanged(self):
        source = RUNTIME_PATH.read_text(encoding="utf-8")
        self.assertIn("_ORIENTATION_HOLD_MAXIMUM_STEP_DEG = 0.15", source)
        self.assertIn("_ORIENTATION_HOLD_MAXIMUM_BIAS_DEG = 3.0", source)
        self.assertIn("update_orientation_hold_command", source)

        config_source = CONFIG_PATH.read_text(encoding="utf-8")
        self.assertIn("max_orientation_error_deg: float = 1.0", config_source)

    def test_feedback_runs_only_during_advancing_insertion(self):
        source = RUNTIME_PATH.read_text(encoding="utf-8")
        tree = ast.parse(source)
        method = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
            and node.name == "_update_insertion_orientation_hold_target"
        )
        conditions = {
            ast.unparse(node.test)
            for node in ast.walk(method)
            if isinstance(node, ast.If)
        }
        self.assertTrue(
            any(
                "controller.phase is not InsertionPhase.ADVANCING" in condition
                for condition in conditions
            )
        )


if __name__ == "__main__":
    unittest.main()
