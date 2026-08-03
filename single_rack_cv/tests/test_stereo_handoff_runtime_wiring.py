#!/usr/bin/env python3

import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
RUNTIME_PATH = ROOT / "stereo_handoff_runtime.py"
MAIN_PATH = ROOT / "main.py"
CONFIG_PATH = ROOT / "config.py"


def _method_node(source: str, class_name: str, method_name: str) -> ast.FunctionDef:
    tree = ast.parse(source)
    runtime_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    return next(
        node
        for node in runtime_class.body
        if isinstance(node, ast.FunctionDef) and node.name == method_name
    )


def _statement_contains(statement: ast.stmt, expression: str) -> bool:
    return expression in ast.unparse(statement)


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

    def test_runtime_freezes_stable_goal_before_continuous_vision_collapses(self):
        source = RUNTIME_PATH.read_text(
            encoding="utf-8"
        )

        self.assertIn(
            "_RECENT_GOAL_WINDOW = 3",
            source,
        )
        self.assertIn(
            "_MAXIMUM_HANDOFF_DISTANCE_M = 0.100",
            source,
        )
        self.assertIn(
            "select_recent_bounded_goal",
            source,
        )
        self.assertIn(
            "newest stable stereo goals entered",
            source,
        )

        method = _method_node(
            source,
            "AngledHandStereoHandoffRuntime",
            "observe_visual_servo",
        )
        append_index = next(
            index
            for index, statement in enumerate(method.body)
            if _statement_contains(
                statement,
                "self._stereo_goal_candidates.append",
            )
        )
        valid_visual_step_index = next(
            index
            for index, statement in enumerate(method.body)
            if index > append_index
            and _statement_contains(
                statement,
                "super().observe_visual_servo",
            )
        )
        self.assertLess(
            append_index,
            valid_visual_step_index,
            "A valid world-goal sample must be recorded before the visual "
            "controller changes its target or loses the next camera view.",
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
