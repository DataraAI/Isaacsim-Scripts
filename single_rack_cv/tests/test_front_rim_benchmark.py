from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np

from benchmarks.front_rim_benchmark import (
    point_to_plane_error_m,
    qualification_passes,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class FrontRimBenchmarkTests(unittest.TestCase):
    def test_launcher_generates_missing_ground_truth(self) -> None:
        source = (
            PROJECT_ROOT / "tools" / "run_front_rim_benchmark.sh"
        ).read_text(encoding="utf-8")
        self.assertIn('GROUND_TRUTH="benchmarks/front_rim_ground_truth.json"', source)
        self.assertIn('bash tools/run_front_rim_ground_truth.sh', source)
        self.assertIn('if [[ ! -s "$GROUND_TRUTH" ]]', source)

    def test_launcher_uses_summary_to_set_final_exit_status(self) -> None:
        source = (
            PROJECT_ROOT / "tools" / "run_front_rim_benchmark.sh"
        ).read_text(encoding="utf-8")
        self.assertIn('SUMMARY="camera_output/front_rim_benchmark_v1/summary.json"', source)
        self.assertIn('"QUALIFIED": true', source)
        self.assertIn("exit 2", source)
        self.assertNotIn(
            'exec "$HOME/isaacsim/python.sh" tools/run_front_rim_benchmark_isaac.py',
            source,
        )

    def test_refined_sgbm_benchmark_keeps_strict_gates(self) -> None:
        refined = (
            PROJECT_ROOT
            / "benchmarks"
            / "front_rim_sgbm_refined_benchmark.py"
        ).read_text(encoding="utf-8")
        self.assertIn("estimate_front_plane_sgbm_refined", refined)
        self.assertIn('"plane_residual_p95_mm"', refined)
        self.assertIn("STRICT_PLANE_RESIDUAL_P95_MM", refined)
        self.assertIn("and plane_residual_p95_mm <=", refined)
        self.assertIn('"mode": "local_sgbm_refined_v7"', refined)
        self.assertNotIn("center_max_gap_m=0.0020", refined)
        self.assertNotIn("plane_max_residual_m=0.0010", refined)

        bootstrap = (
            PROJECT_ROOT / "tools" / "run_front_rim_benchmark_isaac.py"
        ).read_text(encoding="utf-8")
        self.assertIn("front_rim_sgbm_refined_benchmark.py", bootstrap)

        launcher = (
            PROJECT_ROOT / "tools" / "run_front_rim_benchmark.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("mode=local-sgbm-refined-v7", launcher)

    def test_point_to_plane_error(self) -> None:
        error = point_to_plane_error_m(
            point_world_m=np.array([0.0, 0.0, 0.002]),
            plane_center_world_m=np.zeros(3),
            plane_normal_world=np.array([0.0, 0.0, 1.0]),
        )
        self.assertAlmostEqual(error, 0.002)

    def test_point_to_plane_error_normalizes_normal(self) -> None:
        error = point_to_plane_error_m(
            point_world_m=np.array([0.0, 0.0, -0.003]),
            plane_center_world_m=np.zeros(3),
            plane_normal_world=np.array([0.0, 0.0, 5.0]),
        )
        self.assertAlmostEqual(error, 0.003)

    def test_qualification_requires_every_gate(self) -> None:
        passing = dict(
            pair_success_rate=1.0,
            track_switch_count=0,
            radial_jitter_mm=0.4,
            ray_gap_p95_mm=0.3,
            plane_error_median_mm=0.4,
            plane_error_p95_mm=0.8,
        )
        self.assertTrue(qualification_passes(**passing))
        for key, value in {
            "pair_success_rate": 0.94,
            "track_switch_count": 1,
            "radial_jitter_mm": 0.51,
            "ray_gap_p95_mm": 0.51,
            "plane_error_median_mm": 0.51,
            "plane_error_p95_mm": 1.01,
        }.items():
            failing = dict(passing)
            failing[key] = value
            self.assertFalse(qualification_passes(**failing), key)


if __name__ == "__main__":
    unittest.main()
