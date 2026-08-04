#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class PlaneRectifiedRuntimeWiringTests(unittest.TestCase):
    def test_outer_bezel_runtime_uses_rgb_plane_rectification(self):
        source = (ROOT / "outer_bezel_projective_center.py").read_text()
        self.assertIn(
            "from plane_rectified_front_lip import",
            source,
        )
        self.assertIn(
            "estimate_plane_rectified_front_lip_center(",
            source,
        )
        self.assertNotIn("lower_mouth_projective_center", source)

    def test_outer_bezel_depth_estimator_is_preserved(self):
        source = (ROOT / "outer_bezel_projective_center.py").read_text()
        self.assertIn("estimate_outer_bezel_plane(", source)
        self.assertIn("plane_origin_world_m=plane.center_world_m", source)
        self.assertIn("plane_normal_world=plane.normal_world", source)

    def test_safety_gate_is_not_relaxed(self):
        source = (ROOT / "plane_rectified_types.py").read_text()
        self.assertIn("MAX_CENTER_DISAGREEMENT_M = 0.0005", source)
        self.assertIn("MAX_EDGE_REPROJECTION_PX = 1.5", source)

    def test_no_runtime_truth_or_correction_source_is_introduced(self):
        sources = "\n".join(
            (ROOT / name).read_text()
            for name in (
                "plane_rectified_front_lip.py",
                "plane_rectified_geometry.py",
                "plane_rectified_fitting.py",
                "outer_bezel_projective_center.py",
            )
        ).lower()
        forbidden = (
            "rtx ray",
            "port prim",
            "rack transform",
            "world correction",
            "manual correction",
        )
        for phrase in forbidden:
            self.assertNotIn(phrase, sources)


if __name__ == "__main__":
    unittest.main()
