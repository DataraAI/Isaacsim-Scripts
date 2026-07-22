from __future__ import annotations

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


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


if __name__ == "__main__":
    unittest.main()
