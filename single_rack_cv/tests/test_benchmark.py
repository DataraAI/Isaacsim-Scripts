from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import ModuleType
import unittest


ROOT = Path(__file__).resolve().parents[1]


def _load_runner_module():
    path = ROOT / "tools" / "run_benchmark_isaac.py"
    spec = importlib.util.spec_from_file_location("benchmark_runner_test", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load benchmark runner: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BenchmarkStructureTests(unittest.TestCase):
    def test_one_public_high_resolution_benchmark(self):
        source = (ROOT / "benchmarks" / "front_plane_benchmark.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("EXPECTED_RESOLUTION = [960, 1280]", source)
        self.assertIn("from front_plane import", source)
        self.assertIn("plane_residual_p95_mm", source)
        self.assertIn('"mode": "front_plane_highres_v1"', source)
        self.assertNotIn("front_rim_sgbm_refined", source)
        self.assertNotIn("front_rim_sgbm_highres", source)

    def test_qualification_gates_are_unchanged(self):
        source = (ROOT / "benchmarks" / "front_plane_benchmark.py").read_text(
            encoding="utf-8"
        )
        for gate in (
            "pair_success_rate>=0.95",
            "track_switch_count=0",
            "radial_jitter_mm<=0.5",
            "ray_gap_p95_mm<=0.5",
            "plane_residual_p95_mm<=0.5",
            "plane_error_median_mm<=0.5",
            "plane_error_p95_mm<=1.0",
        ):
            self.assertIn(gate, source)

    def test_launcher_uses_summary_exit_status(self):
        source = (ROOT / "tools" / "run_benchmark.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("1280x960", source)
        self.assertIn('"QUALIFIED": true', source)
        self.assertIn("exit 0", source)
        self.assertIn("exit 2", source)
        self.assertIn("run_benchmark_isaac.py", source)

    def test_isaac_runner_replaces_shadowing_cv2_config_module(self):
        runner = _load_runner_module()
        previous = sys.modules.get("config")
        fake = ModuleType("config")
        fake.__file__ = "/tmp/cv2/config.py"
        sys.modules["config"] = fake
        try:
            loaded = runner._load_project_config()
            self.assertEqual(
                Path(loaded.__file__).resolve(),
                (ROOT / "config.py").resolve(),
            )
            self.assertIs(sys.modules["config"], loaded)
        finally:
            sys.modules.pop("config", None)
            if previous is not None:
                sys.modules["config"] = previous


if __name__ == "__main__":
    unittest.main()
