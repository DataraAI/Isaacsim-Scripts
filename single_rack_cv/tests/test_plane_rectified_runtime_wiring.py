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

    def test_independent_eye_gate_precedes_joint_refit(self):
        source = (ROOT / "plane_rectified_front_lip.py").read_text()
        gate = source.index("if disagreement > maximum:")
        joint = source.index("joint_fit = _fit_joint_front_lip(")
        self.assertLess(gate, joint)

    def test_runtime_saves_progressive_and_accepted_fit_diagnostics(self):
        source = (ROOT / "plane_rectified_front_lip.py").read_text()
        for filename in (
            "front_lip_rectified_left.png",
            "front_lip_rectified_right.png",
            "front_lip_fit_left.png",
            "front_lip_fit_right.png",
            "front_lip_fit_joint.png",
            "front_lip_reprojection_left_eye_fit.png",
            "front_lip_reprojection_right_eye_fit.png",
            "front_lip_reprojection_left.png",
            "front_lip_reprojection_right.png",
        ):
            self.assertIn(filename, source)
        self.assertIn('"[RGB FRONT LIP] "', source)
        self.assertIn("supports=", source)
        self.assertIn("joint_size=", source)

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
