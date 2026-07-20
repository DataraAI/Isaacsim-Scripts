#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_ROOT = ROOT / "benchmarks"


class PromptBenchmarkStructureTests(unittest.TestCase):
    def read_required(self, name: str) -> str:
        path = BENCHMARK_ROOT / name
        self.assertTrue(path.exists(), f"{name} has not been implemented")
        return path.read_text(encoding="utf-8")

    def test_runner_uses_two_fresh_child_processes(self):
        source = self.read_required("run_prompt_ab_benchmark.py")
        self.assertIn("prompt_benchmark_capture.py", source)
        self.assertIn("prompt_benchmark_evaluate_isaac_bootstrap.py", source)
        self.assertIn("subprocess.run", source)
        self.assertNotIn("from config import", source)
        self.assertNotIn("from perception import", source)
        self.assertNotIn("from sim import", source)

    def test_capture_starts_simulation_before_importing_sim(self):
        source = self.read_required("prompt_benchmark_capture.py")
        app_index = source.index("from isaacsim import SimulationApp")
        sim_index = source.index("from sim import SimulationRuntime")
        self.assertLess(app_index, sim_index)
        self.assertNotIn("YOLOEPortDetector", source)
        self.assertNotIn("process_stereo_port", source)
        self.assertIn("BENCHMARK_FRAME_COUNT", source)

    def test_evaluation_is_offline_and_uses_last_atlas_box(self):
        source = self.read_required("prompt_benchmark_evaluate.py")
        self.assertNotIn("SimulationApp", source)
        self.assertIn("reference_boxes_xyxy[-1]", source)
        self.assertIn("previous_left=None", source)
        self.assertIn("previous_right=None", source)
        self.assertIn("CachedDetector", source)

    def test_existing_runtime_files_are_never_write_targets(self):
        for name in ("run_prompt_ab_benchmark.py", "prompt_benchmark_capture.py", "prompt_benchmark_evaluate.py"):
            source = self.read_required(name)
            self.assertNotIn('write_text("main.py"', source)
            self.assertNotIn('write_text("config.py"', source)
            self.assertNotIn('write_text("perception.py"', source)
            self.assertNotIn('write_text("sim.py"', source)


if __name__ == "__main__":
    unittest.main()
