from __future__ import annotations

import unittest

import numpy as np

from benchmarks.front_rim_benchmark import (
    point_to_plane_error_m,
    qualification_passes,
)


class FrontRimBenchmarkTests(unittest.TestCase):
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
