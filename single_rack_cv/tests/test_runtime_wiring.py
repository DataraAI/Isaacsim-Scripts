from __future__ import annotations

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class RuntimeWiringTests(unittest.TestCase):
    def test_canonical_config_is_high_resolution_front_plane(self):
        source = (ROOT / "config.py").read_text(encoding="utf-8")
        self.assertIn("resolution: tuple[int, int] = (960, 1280)", source)
        self.assertIn("class FrontPlaneRuntimeConfig", source)
        self.assertIn("enabled: bool = True", source)
        self.assertIn("front_plane: FrontPlaneRuntimeConfig", source)
        self.assertNotIn("class FrontRimConfig", source)

    def test_main_refines_before_motion_and_debug(self):
        source = (ROOT / "main.py").read_text(encoding="utf-8")
        self.assertIn("from live_control import refine_live_observation", source)
        refine = source.index("refine_live_observation(")
        observe = source.index("runtime.observe_visual_servo(observation)")
        debug = source.index("debug.handle(")
        self.assertLess(refine, observe)
        self.assertLess(refine, debug)
        self.assertIn("RUNTIME_CONFIG.front_plane.enabled", source)

    def test_failure_path_holds_and_reacquires(self):
        source = (ROOT / "main.py").read_text(encoding="utf-8")
        self.assertIn("runtime.note_perception_failure()", source)
        self.assertIn("RGB stereo capture", source)
        self.assertIn("no manual depth offset", source)

    def test_stress_mode_is_optional_and_uses_instrumented_subclass(self):
        source = (ROOT / "main.py").read_text(encoding="utf-8")
        self.assertIn("parse_stress_run_args", source)
        self.assertIn("stress_args is None", source)
        self.assertIn("derive_stress_config", source)
        self.assertIn("InstrumentedSimulationRuntime", source)
        self.assertIn("runtime.stress_snapshot()", source)
        self.assertIn("write_json_atomic", source)
        self.assertIn("raise SystemExit(child_exit_status)", source)

    def test_child_has_no_parent_scoring_or_truth_file_access(self):
        source = (ROOT / "main.py").read_text(encoding="utf-8").lower()
        self.assertNotIn("front_plane_ground_truth", source)
        self.assertNotIn("expected_preinsert_target", source)
        self.assertNotIn("ground_truth_target_error", source)
        self.assertNotIn("finalize_parent_result", source)
        self.assertNotIn('"qualified"', source)

    def test_default_runtime_class_remains_canonical(self):
        source = (ROOT / "main.py").read_text(encoding="utf-8")
        self.assertIn("runtime_class = SimulationRuntime", source)
        normal_assignment = source.index("runtime_class = SimulationRuntime")
        stress_guard = source.index("if stress_args is not None:", normal_assignment)
        stress_assignment = source.index(
            "runtime_class = InstrumentedSimulationRuntime",
            stress_guard,
        )
        self.assertLess(normal_assignment, stress_guard)
        self.assertLess(stress_guard, stress_assignment)

    def test_parent_scores_truth_only_after_child_returns(self):
        source = (ROOT / "tools" / "run_alignment_stress.py").read_text(
            encoding="utf-8"
        )
        main_loop = source.index("for index, case in enumerate(cases")
        child_call = source.index("= run_one_case(", main_loop)
        finalize = source.index("result = finalize_parent_result(", child_call)
        self.assertLess(child_call, finalize)
        self.assertIn("front_plane_ground_truth.json", source)
        command_block = source[
            source.index("def build_child_command"):
            source.index("def _finite_vector3")
        ]
        self.assertNotIn("front_plane_ground_truth", command_block)

    def test_no_insertion_path_and_step_limit_unchanged(self):
        main_source = (ROOT / "main.py").read_text(encoding="utf-8").lower()
        stress_source = (ROOT / "stress_runtime.py").read_text(encoding="utf-8")
        config_source = (ROOT / "config.py").read_text(encoding="utf-8")
        self.assertNotIn("insert_along", main_source)
        self.assertIn('"insertion_command_count": 0', stress_source)
        self.assertNotIn("insert_along", stress_source)
        self.assertIn("max_target_step_m: float = 0.001", config_source)

    def test_launcher_and_readme_lock_27_of_27(self):
        launcher = (ROOT / "tools" / "run_alignment_stress.sh").read_text(
            encoding="utf-8"
        )
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("qualification requires 27/27", launcher)
        self.assertIn("exec /usr/bin/python3 tools/run_alignment_stress.py", launcher)
        self.assertIn("passed_run_count=27", readme)
        self.assertIn("failed_run_count=0", readme)
        self.assertIn("QUALIFIED=True", readme)
        self.assertNotIn("recovery/pre-single-rack-cleanup", readme)


if __name__ == "__main__":
    unittest.main()
