from __future__ import annotations

import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
RUNTIME_ROOT = ROOT / "runtime"
ROBOT_ROOT = ROOT / "robot"
MAIN_PATH = ROOT / "main.py"
RUNTIME_PATH = RUNTIME_ROOT / "settled_stereo_handoff_runtime.py"
FULL_RUNTIME_PATH = RUNTIME_ROOT / "full_insertion_runtime.py"
HANDOFF_HOLD_RUNTIME_PATH = RUNTIME_ROOT / "handoff_position_hold_runtime.py"
FULL_BASE_RUNTIME_PATH = RUNTIME_ROOT / "full_insertion_base_runtime.py"
ANGLED_CONFIG_PATH = ROBOT_ROOT / "angled_hand_config.py"


class StartupGeometrySettleTests(unittest.TestCase):
    def test_main_selects_full_wrapper_over_settling_runtime(self):
        source = MAIN_PATH.read_text(encoding="utf-8")
        export_source = FULL_RUNTIME_PATH.read_text(encoding="utf-8")
        hold_source = HANDOFF_HOLD_RUNTIME_PATH.read_text(encoding="utf-8")
        base_source = FULL_BASE_RUNTIME_PATH.read_text(encoding="utf-8")

        self.assertIn(
            "from runtime.full_insertion_runtime import (",
            source,
        )
        self.assertIn(
            "from runtime.handoff_position_hold_runtime import (",
            export_source,
        )
        self.assertNotIn(
            "from runtime.full_insertion_base_runtime import (",
            export_source,
        )
        self.assertIn(
            "from runtime.full_insertion_base_runtime import (",
            hold_source,
        )
        self.assertIn(
            "from runtime.settled_stereo_handoff_runtime import (",
            base_source,
        )
        self.assertIn(
            "AngledHandStereoHandoffRuntime as "
            "CableMountedSimulationRuntime",
            source,
        )
        self.assertNotIn("from runtime.precontact_runtime import (", source)

    def test_transient_geometry_miss_resets_consecutive_window(self):
        source = RUNTIME_PATH.read_text(encoding="utf-8")
        self.assertIn("_TRANSIENT_GEOMETRY_PREFIXES", source)
        self.assertIn("samples.clear()", source)
        self.assertIn("last_transient_error = error", source)
        self.assertIn(
            "if not self._is_transient_geometry_error(error):\n"
            "                    raise",
            source,
        )
        self.assertIn(
            "if len(samples) == cfg.validation_frames:",
            source,
        )

    def test_post_handoff_does_not_wait_on_camera_presentation(self):
        source = RUNTIME_PATH.read_text(encoding="utf-8")
        self.assertNotIn("def update_partial_insertion(self)", source)
        self.assertNotIn("INSERTION PRESENTATION WAITING", source)
        self.assertNotIn("INSERTION PRESENTATION VALIDATED", source)
        self.assertNotIn("ConsecutiveValidityWindow", source)

    def test_runtime_installs_consecutive_pose_insertion_controller(self):
        source = RUNTIME_PATH.read_text(encoding="utf-8")
        export_source = FULL_RUNTIME_PATH.read_text(encoding="utf-8")
        base_source = FULL_BASE_RUNTIME_PATH.read_text(encoding="utf-8")

        self.assertIn(
            "from control.settled_insertion import ConsecutivePoseInsertionController",
            source,
        )
        self.assertIn(
            "self.partial_insertion = ConsecutivePoseInsertionController(limits)",
            source,
        )
        self.assertIn(
            "self._insertion_axis_adapter = ExplicitInsertionAxisAdapter(",
            source,
        )
        self.assertIn(
            "TrimmedConsecutivePoseInsertionController(",
            export_source,
        )
        self.assertIn(
            "self._insertion_axis_adapter = ExplicitInsertionAxisAdapter(",
            export_source,
        )
        self.assertNotIn("ConsecutivePoseInsertionController(", base_source)
        self.assertIn("self.partial_insertion.limits", base_source)

    def test_strict_presentation_check_ends_when_visual_handoff_completes(self):
        source = RUNTIME_PATH.read_text(encoding="utf-8")
        tree = ast.parse(source)
        method = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
            and node.name == "_sample_mount_validation_live"
        )
        conditions = {
            ast.unparse(node.test)
            for node in ast.walk(method)
            if isinstance(node, ast.If)
        }

        self.assertIn("not self.visual_servo.complete", conditions)
        self.assertIn(
            "return super()._sample_mount_validation_live(runtime)",
            source,
        )
        self.assertIn(
            "_CableMountedSimulationRuntime._sample_mount_validation_live(",
            source,
        )

    def test_one_degree_palm_limit_is_not_relaxed(self):
        source = ANGLED_CONFIG_PATH.read_text(encoding="utf-8")
        self.assertIn("palm_side_tolerance_deg: float = 1.0", source)
        runtime_source = RUNTIME_PATH.read_text(encoding="utf-8")
        export_source = FULL_RUNTIME_PATH.read_text(encoding="utf-8")
        base_source = FULL_BASE_RUNTIME_PATH.read_text(encoding="utf-8")

        self.assertNotIn("1.01", runtime_source)
        self.assertNotIn("tolerance_deg =", runtime_source)
        self.assertNotIn("max_orientation_error_deg=", export_source)
        self.assertNotIn("max_orientation_error_deg=", base_source)
        self.assertIn("limits.max_orientation_error_deg", base_source)


if __name__ == "__main__":
    unittest.main()
