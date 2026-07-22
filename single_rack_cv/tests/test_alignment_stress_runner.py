from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

from stress_alignment import StressCase

ROOT = Path(__file__).resolve().parents[1]


def load_runner():
    path = ROOT / "tools" / "run_alignment_stress.py"
    spec = importlib.util.spec_from_file_location("stress_runner_test", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class AlignmentStressRunnerTests(unittest.TestCase):
    def valid_truth(self):
        return {
            "schema_version": 4,
            "camera_resolution_height_width": [960, 1280],
            "source": "automatic_rtx_mesh_raycast_front_bezel_plane",
            "control_usage": "forbidden; benchmark scoring only",
            "center_world_m": [1.0, 2.0, 3.0],
            "normal_world": [2.0, 0.0, 0.0],
            "used_prim_paths": ["/World/ServerRack/Asset/Port"],
        }

    def test_child_command_uses_only_approved_arguments(self):
        runner = load_runner()
        case = StressCase(-10, 10, 2)
        command = runner.build_child_command(
            Path("/home/aayush/isaacsim/python.sh"),
            ROOT,
            case,
            Path("/tmp/child_result.json"),
        )
        self.assertEqual(command[0], "/home/aayush/isaacsim/python.sh")
        self.assertIn("--stress-run-id", command)
        self.assertIn(case.run_id, command)
        self.assertIn("--stress-result-json", command)
        self.assertIn("--exit-after-complete", command)
        self.assertNotIn("--stress-repeat", command)
        self.assertNotIn("front_plane_ground_truth", " ".join(command))

    def test_truth_validation_is_scene_specific(self):
        runner = load_runner()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "truth.json"
            path.write_text(json.dumps(self.valid_truth()), encoding="utf-8")
            payload = runner.load_valid_ground_truth(path)
            self.assertEqual(payload["center_world_m"], [1.0, 2.0, 3.0])

            mutations = [
                {"schema_version": 3},
                {"camera_resolution_height_width": [720, 1280]},
                {"source": "manual"},
                {"control_usage": "allowed"},
                {"normal_world": [0.0, 0.0, 0.0]},
                {"used_prim_paths": ["/World/Other/Port"]},
            ]
            for mutation in mutations:
                invalid = self.valid_truth()
                invalid.update(mutation)
                path.write_text(json.dumps(invalid), encoding="utf-8")
                with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                    runner.load_valid_ground_truth(path)

    def test_child_result_missing_malformed_and_mismatch_are_synthesized(self):
        runner = load_runner()
        case = StressCase(0, 0, 1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "child.json"
            payload, status = runner.load_child_result(case, path)
            self.assertEqual(status, "missing")
            self.assertEqual(payload["run_id"], case.run_id)

            path.write_text("{", encoding="utf-8")
            payload, status = runner.load_child_result(case, path)
            self.assertEqual(status, "malformed")

            payload = runner._synthesized_child(case, path, "")
            payload["run_id"] = "y+10_z+00_r1"
            path.write_text(json.dumps(payload), encoding="utf-8")
            _payload, status = runner.load_child_result(case, path)
            self.assertEqual(status, "mismatch")

    def test_unique_suite_directory_never_overwrites(self):
        runner = load_runner()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = runner.create_suite_directory(root)
            second = runner.create_suite_directory(root)
            self.assertNotEqual(first, second)
            self.assertTrue(first.is_dir())
            self.assertTrue(second.is_dir())

    def test_suite_outputs_write_strict_json_csv_and_report(self):
        runner = load_runner()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = {
                "run_id": "y+00_z+00_r1",
                "start_y_offset_mm": 0,
                "start_z_offset_mm": 0,
                "repeat": 1,
                "runtime_duration_s": 1.0,
                "qualified": False,
                "failed_gates": ["test_gate"],
                "final_center_error_px": 1.0,
                "final_range_error_mm": 0.0,
                "ground_truth_target_error_mm": 0.0,
                "final_physical_tracking_error_mm": 0.1,
                "maximum_target_step_mm": 1.0,
                "maximum_orientation_deviation_deg": 0.0,
                "perception_rejection_count": 0,
                "track_reacquisition_count": 0,
            }
            summary = runner.write_suite_outputs(
                root,
                [result],
                ["y+00_z+00_r1"],
            )
            self.assertFalse(summary["QUALIFIED"])
            self.assertTrue((root / "summary.csv").is_file())
            report = (root / "report.txt").read_text(encoding="utf-8")
            self.assertTrue(report.startswith(
                "ALIGNMENT STRESS QUALIFICATION\n"
                "passed_run_count=0\n"
                "failed_run_count=1\n"
                "QUALIFIED=False\n"
            ))
            json.loads((root / "summary.json").read_text(encoding="utf-8"))

    def test_run_one_case_uses_270_second_timeout_and_process_group_cleanup(self):
        runner = load_runner()
        process = mock.Mock()
        process.pid = 1234
        process.poll.return_value = None
        process.wait.side_effect = [subprocess.TimeoutExpired("cmd", 270), 143]
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(runner.subprocess, "Popen", return_value=process) as popen:
                with mock.patch.object(runner.os, "killpg") as killpg:
                    status, hard_timeout, _duration = runner.run_one_case(
                        Path("/isaac/python.sh"),
                        ROOT,
                        StressCase(0, 0, 1),
                        Path(directory) / "run",
                    )
        self.assertTrue(hard_timeout)
        self.assertEqual(status, 143)
        process.wait.assert_any_call(timeout=270.0)
        killpg.assert_called_once_with(1234, runner.signal.SIGTERM)
        self.assertTrue(popen.call_args.kwargs["start_new_session"])

    def test_keyboard_interrupt_terminates_child_group(self):
        runner = load_runner()
        process = mock.Mock()
        process.pid = 4321
        process.poll.return_value = None
        process.wait.side_effect = [KeyboardInterrupt(), 143]
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(runner.subprocess, "Popen", return_value=process):
                with mock.patch.object(runner.os, "killpg") as killpg:
                    with self.assertRaises(KeyboardInterrupt):
                        runner.run_one_case(
                            Path("/isaac/python.sh"),
                            ROOT,
                            StressCase(0, 0, 1),
                            Path(directory) / "run",
                        )
        killpg.assert_called_once_with(4321, runner.signal.SIGTERM)


if __name__ == "__main__":
    unittest.main()
