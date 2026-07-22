from __future__ import annotations

import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from config import CONFIG
from stress_alignment import (
    ORIENTATION_LIMIT_DEG,
    STEP_LIMIT_MM,
    STRESS_SEED,
    StressCase,
    StressRunArgs,
    aggregate_suite,
    build_stress_cases,
    derive_stress_config,
    expected_preinsert_target_world_m,
    finalize_parent_result,
    new_child_result,
    parse_stress_run_args,
    quaternion_angular_distance_deg,
    write_json_atomic,
)


class AlignmentStressTests(unittest.TestCase):
    def test_matrix_is_nine_poses_three_repeats(self):
        cases = build_stress_cases()
        self.assertEqual(len(cases), 27)
        counts: dict[tuple[int, int], int] = {}
        for case in cases:
            key = (case.y_offset_mm, case.z_offset_mm)
            counts[key] = counts.get(key, 0) + 1
        self.assertEqual(
            counts,
            {(y, z): 3 for y in (-10, 0, 10) for z in (-10, 0, 10)},
        )
        self.assertEqual(len({case.run_id for case in cases}), 27)
        self.assertEqual(len({case.directory_name for case in cases}), 27)

    def test_seed_is_reproducible(self):
        first = [case.run_id for case in build_stress_cases(STRESS_SEED)]
        second = [case.run_id for case in build_stress_cases(STRESS_SEED)]
        other = [case.run_id for case in build_stress_cases(STRESS_SEED + 1)]
        self.assertEqual(first, second)
        self.assertNotEqual(first, other)

    def test_run_id_round_trip(self):
        case = StressCase(-10, 10, 2)
        self.assertEqual(case.run_id, "y-10_z+10_r2")
        self.assertEqual(case.directory_name, "y-10_z+10_repeat-2")
        self.assertEqual(StressCase.from_run_id(case.run_id), case)

    def test_invalid_run_id_rejected(self):
        for value in ("", "y-1_z+10_r2", "y-10_z+10_r4", "y+11_z+00_r1"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                StressCase.from_run_id(value)

    def test_cli_uses_approved_arguments_only(self):
        args = parse_stress_run_args([
            "--start-y-offset-mm", "-10",
            "--start-z-offset-mm", "10",
            "--stress-run-id", "y-10_z+10_r2",
            "--stress-result-json", "/tmp/child_result.json",
            "--stress-timeout-s", "240",
            "--exit-after-complete",
        ])
        self.assertIsNotNone(args)
        assert args is not None
        self.assertEqual(args.case, StressCase(-10, 10, 2))
        self.assertEqual(args.result_json, Path("/tmp/child_result.json"))
        self.assertTrue(args.exit_after_complete)

    def test_no_stress_flags_returns_none(self):
        self.assertIsNone(parse_stress_run_args([]))
        self.assertIsNone(parse_stress_run_args(["--some-unrelated-option"]))

    def test_partial_or_mismatched_stress_args_rejected(self):
        bad_argv = [
            ["--exit-after-complete"],
            [
                "--start-y-offset-mm", "0", "--start-z-offset-mm", "0",
                "--stress-run-id", "y+00_z+00_r1",
                "--stress-result-json", "/tmp/out.json",
                "--stress-timeout-s", "240",
            ],
            [
                "--start-y-offset-mm", "10", "--start-z-offset-mm", "0",
                "--stress-run-id", "y+00_z+00_r1",
                "--stress-result-json", "/tmp/out.json",
                "--stress-timeout-s", "240", "--exit-after-complete",
            ],
            [
                "--start-y-offset-mm", "0", "--start-z-offset-mm", "0",
                "--stress-run-id", "y+00_z+00_r1",
                "--stress-result-json", "/tmp/out.json",
                "--stress-timeout-s", "nan", "--exit-after-complete",
            ],
        ]
        for argv in bad_argv:
            with self.subTest(argv=argv), self.assertRaises(ValueError):
                parse_stress_run_args(argv)

    def test_stress_runtime_instrumentation_is_passive_and_complete(self):
        source = Path("stress_runtime.py").read_text(encoding="utf-8")
        for token in (
            "class InstrumentedSimulationRuntime",
            "track_acquired_ever",
            "visual_alignment_locked_ever",
            "perception_rejection_count",
            "maximum_target_step_m",
            "maximum_orientation_deviation_deg",
            "def stress_snapshot",
            '"insertion_command_count": 0',
        ):
            self.assertIn(token, source)
        self.assertIn("super().observe_visual_servo(observation)", source)
        self.assertIn("target_before", source)
        self.assertNotIn("compute_bounded_step(", source)
        self.assertNotIn("insert_along", source)

    def test_quaternion_sign_flip_is_zero(self):
        self.assertAlmostEqual(
            quaternion_angular_distance_deg(
                (1.0, 0.0, 0.0, 0.0),
                (-1.0, 0.0, 0.0, 0.0),
            ),
            0.0,
            places=12,
        )

    def test_known_ten_degree_rotation(self):
        half = math.radians(5.0)
        rotated = (math.cos(half), 0.0, math.sin(half), 0.0)
        self.assertAlmostEqual(
            quaternion_angular_distance_deg((1.0, 0.0, 0.0, 0.0), rotated),
            10.0,
            places=12,
        )

    def test_quaternion_distance_rejects_nonfinite_and_zero(self):
        with self.assertRaises(ValueError):
            quaternion_angular_distance_deg((0.0, 0.0, 0.0, 0.0), (1, 0, 0, 0))
        with self.assertRaises(ValueError):
            quaternion_angular_distance_deg((math.nan, 0, 0, 0), (1, 0, 0, 0))

    def test_offsets_change_only_world_y_and_z(self):
        args = StressRunArgs(
            case=StressCase(10, -10, 1),
            result_json=Path("/tmp/child_result.json"),
            timeout_s=240.0,
            exit_after_complete=True,
        )
        derived = derive_stress_config(CONFIG, args)
        x, y, z = CONFIG.ik.initial_position
        self.assertEqual(derived.ik.initial_position[0], x)
        self.assertAlmostEqual(derived.ik.initial_position[1], y + 0.010)
        self.assertAlmostEqual(derived.ik.initial_position[2], z - 0.010)
        self.assertEqual(
            derived.ik.initial_orientation_wxyz,
            CONFIG.ik.initial_orientation_wxyz,
        )
        self.assertEqual(CONFIG.ik.initial_position, (x, y, z))

    def passing_child(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "run_id": "y+00_z+00_r1",
            "start_y_offset_mm": 0,
            "start_z_offset_mm": 0,
            "repeat": 1,
            "started_at": "2026-07-22T00:00:00+00:00",
            "ended_at": "2026-07-22T00:01:00+00:00",
            "runtime_duration_s": 60.0,
            "completed": True,
            "internal_timed_out": False,
            "track_acquired": True,
            "visual_alignment_locked": True,
            "final_center_error_px": 2.0,
            "final_range_error_mm": -3.0,
            "final_tool_target_world_m": [1.05, 2.0, 3.0],
            "final_actual_tool_world_m": [1.0503, 2.0, 3.0],
            "final_physical_tracking_error_mm": 0.3,
            "maximum_target_step_mm": STEP_LIMIT_MM,
            "maximum_orientation_deviation_deg": ORIENTATION_LIMIT_DEG,
            "perception_rejection_count": 0,
            "track_reacquisition_count": 0,
            "fatal_error": "",
            "insertion_command_count": 0,
        }

    def finalize(self, child: dict[str, object] | None = None, **overrides):
        kwargs = dict(
            child_payload=self.passing_child() if child is None else child,
            subprocess_exit_status=0,
            parent_hard_timed_out=False,
            console_log_path="runs/y+00_z+00_repeat-1/console.log",
            child_result_parse_status="valid",
            truth_center_world_m=[1.0, 2.0, 3.0],
            truth_normal_world=[1.0, 0.0, 0.0],
            preinsert_standoff_m=0.05,
        )
        kwargs.update(overrides)
        return finalize_parent_result(**kwargs)

    def test_new_child_result_has_complete_child_schema(self):
        args = StressRunArgs(
            case=StressCase(0, 0, 1),
            result_json=Path("/tmp/result.json"),
            timeout_s=240.0,
            exit_after_complete=True,
        )
        payload = new_child_result(args, "start")
        self.assertEqual(payload["run_id"], "y+00_z+00_r1")
        self.assertIsNone(payload["final_tool_target_world_m"])
        self.assertNotIn("qualified", payload)
        self.assertNotIn("subprocess_exit_status", payload)
        self.assertNotIn("ground_truth_target_error_mm", payload)

    def test_expected_target_normalizes_normal(self):
        target = expected_preinsert_target_world_m(
            [1.0, 2.0, 3.0], [2.0, 0.0, 0.0], 0.05
        )
        self.assertTrue(np.allclose(target, [1.05, 2.0, 3.0]))

    def test_expected_target_rejects_bad_inputs(self):
        for center, normal, distance in (
            ([math.nan, 0, 0], [1, 0, 0], 0.05),
            ([0, 0, 0], [0, 0, 0], 0.05),
            ([0, 0, 0], [1, 0, 0], 0.0),
        ):
            with self.subTest(center=center, normal=normal, distance=distance):
                with self.assertRaises(ValueError):
                    expected_preinsert_target_world_m(center, normal, distance)

    def test_exact_boundaries_pass(self):
        result = self.finalize()
        self.assertTrue(result["qualified"])
        self.assertEqual(result["failed_gates"], [])
        self.assertAlmostEqual(result["ground_truth_target_error_mm"], 0.0)

    def test_each_boolean_gate_fails(self):
        cases = [
            ("subprocess_exit_status", dict(subprocess_exit_status=1)),
            ("parent_hard_timed_out", dict(parent_hard_timed_out=True)),
            ("child_result_parse_status", dict(child_result_parse_status="missing")),
        ]
        for gate, kwargs in cases:
            with self.subTest(gate=gate):
                result = self.finalize(**kwargs)
                self.assertFalse(result["qualified"])
                self.assertIn(gate, result["failed_gates"])

        child_mutations = [
            ("completed", False),
            ("internal_timed_out", True),
            ("track_acquired", False),
            ("visual_alignment_locked", False),
            ("fatal_error", "traceback"),
            ("insertion_command_count", 1),
        ]
        for gate, value in child_mutations:
            with self.subTest(gate=gate):
                child = self.passing_child()
                child[gate] = value
                result = self.finalize(child)
                self.assertFalse(result["qualified"])
                self.assertIn(gate, result["failed_gates"])

    def test_one_over_each_numeric_limit_fails(self):
        mutations = [
            ("runtime_duration_s", 240.000001),
            ("final_center_error_px", 2.000001),
            ("final_range_error_mm", -3.000001),
            ("final_physical_tracking_error_mm", 0.300001),
            ("maximum_target_step_mm", STEP_LIMIT_MM + 0.000001),
            ("maximum_orientation_deviation_deg", ORIENTATION_LIMIT_DEG + 0.000001),
        ]
        for gate, value in mutations:
            with self.subTest(gate=gate):
                child = self.passing_child()
                child[gate] = value
                result = self.finalize(child)
                self.assertFalse(result["qualified"])
                expected_gate = (
                    "absolute_final_range_error_mm"
                    if gate == "final_range_error_mm"
                    else gate
                )
                self.assertIn(expected_gate, result["failed_gates"])

        child = self.passing_child()
        child["final_tool_target_world_m"] = [1.051000001, 2.0, 3.0]
        result = self.finalize(child)
        self.assertFalse(result["qualified"])
        self.assertIn("ground_truth_target_error_mm", result["failed_gates"])

    def test_missing_and_nonfinite_metrics_fail_and_write_null(self):
        child = self.passing_child()
        child["final_center_error_px"] = math.nan
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "payload.json"
            write_json_atomic(path, child)
            text = path.read_text(encoding="utf-8")
            self.assertIn('"final_center_error_px": null', text)
            json.loads(text)
        result = self.finalize(child)
        self.assertFalse(result["qualified"])
        self.assertIn("final_center_error_px", result["failed_gates"])

        child = self.passing_child()
        del child["final_actual_tool_world_m"]
        result = self.finalize(child)
        self.assertFalse(result["qualified"])
        self.assertIn("required_fields", result["failed_gates"])

    def test_aggregate_requires_exactly_27_of_27(self):
        execution_order = [case.run_id for case in build_stress_cases()]
        passing = self.finalize()
        results = []
        for run_id in execution_order:
            case = StressCase.from_run_id(run_id)
            item = dict(passing)
            item.update(
                run_id=run_id,
                start_y_offset_mm=case.y_offset_mm,
                start_z_offset_mm=case.z_offset_mm,
                repeat=case.repeat,
            )
            results.append(item)
        summary = aggregate_suite(results, execution_order)
        self.assertTrue(summary["QUALIFIED"])
        self.assertEqual(summary["passed_run_count"], 27)
        self.assertEqual(summary["execution_order"], execution_order)

        summary = aggregate_suite(results[:-1], execution_order)
        self.assertFalse(summary["QUALIFIED"])
        self.assertEqual(summary["completed_run_count"], 26)

        failed = [dict(item) for item in results]
        failed[0]["qualified"] = False
        failed[0]["failed_gates"] = ["wrong_port"]
        summary = aggregate_suite(failed, execution_order)
        self.assertFalse(summary["QUALIFIED"])
        self.assertEqual(summary["failed_run_count"], 1)
        self.assertEqual(summary["failures_by_gate"]["wrong_port"], 1)


if __name__ == "__main__":
    unittest.main()
