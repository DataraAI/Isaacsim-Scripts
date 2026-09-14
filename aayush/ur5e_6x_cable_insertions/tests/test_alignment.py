from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
AAYUSH_DIR = REPO_ROOT / "aayush"
sys.path.insert(0, str(AAYUSH_DIR))

from insertion_features.geometry import ConnectorFeatures, Plane
from ur5e_6x_cable_insertions.alignment import (
    clamp_nudge,
    evaluate_alignment,
    insert_target_tip,
    mating_gap_along_axis,
)


def _box_features(*, origin, axis, width, up, half_w=0.005, half_u=0.003) -> ConnectorFeatures:
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    width = np.asarray(width, dtype=np.float64)
    width = width / np.linalg.norm(width)
    up = np.asarray(up, dtype=np.float64)
    up = up / np.linalg.norm(up)
    o = np.asarray(origin, dtype=np.float64)
    corners = np.array(
        [
            o - half_w * width - half_u * up,
            o + half_w * width - half_u * up,
            o + half_w * width + half_u * up,
            o - half_w * width + half_u * up,
        ]
    )
    latch = corners.copy()
    latch[:, 2] += 0.002
    return ConnectorFeatures(
        insertion_axis=axis,
        mating_plane=Plane(point=o.copy(), normal=axis.copy()),
        mating_center=o.copy(),
        mating_corners=corners,
        latch_plane=Plane(point=o + 0.002 * up, normal=axis.copy()),
        latch_corners=latch,
        latch_keypoints=np.array([latch[2], latch[3]]),
        width_axis=width,
        up_axis=up,
    )


class AlignmentTests(unittest.TestCase):
    def test_aligned_features_pass(self) -> None:
        port = _box_features(
            origin=[0.0, 0.0, 0.1],
            axis=[-1.0, 0.0, 0.0],
            width=[0.0, 1.0, 0.0],
            up=[0.0, 0.0, 1.0],
            half_w=0.006,
            half_u=0.004,
        )
        crystal = _box_features(
            origin=[0.01, 0.0, 0.1],
            axis=[-1.0, 0.0, 0.0],
            width=[0.0, 1.0, 0.0],
            up=[0.0, 0.0, 1.0],
            half_w=0.004,
            half_u=0.002,
        )
        # Latch keypoints on crystal must be lower Z than port latch keypoints.
        crystal_kp = crystal.latch_keypoints.copy()
        crystal_kp[:, 2] = 0.099
        port_kp = port.latch_keypoints.copy()
        port_kp[:, 2] = 0.105
        crystal = ConnectorFeatures(
            insertion_axis=crystal.insertion_axis,
            mating_plane=crystal.mating_plane,
            mating_center=crystal.mating_center,
            mating_corners=crystal.mating_corners,
            latch_plane=crystal.latch_plane,
            latch_corners=crystal.latch_corners,
            latch_keypoints=crystal_kp,
            width_axis=crystal.width_axis,
            up_axis=crystal.up_axis,
        )
        port = ConnectorFeatures(
            insertion_axis=port.insertion_axis,
            mating_plane=port.mating_plane,
            mating_center=port.mating_center,
            mating_corners=port.mating_corners,
            latch_plane=port.latch_plane,
            latch_corners=port.latch_corners,
            latch_keypoints=port_kp,
            width_axis=port.width_axis,
            up_axis=port.up_axis,
        )
        residual = evaluate_alignment(
            crystal,
            port,
            latch_z_margin_m=0.0005,
            mating_side_margin_m=0.0005,
            axis_dot_min=0.98,
        )
        self.assertTrue(residual.passed)

    def test_mating_gap_along_axis(self) -> None:
        port = _box_features(
            origin=[0.0, 0.0, 0.1],
            axis=[-1.0, 0.0, 0.0],
            width=[0.0, 1.0, 0.0],
            up=[0.0, 0.0, 1.0],
        )
        crystal = _box_features(
            origin=[0.01, 0.0, 0.1],
            axis=[-1.0, 0.0, 0.0],
            width=[0.0, 1.0, 0.0],
            up=[0.0, 0.0, 1.0],
        )
        gap = mating_gap_along_axis(crystal, port)
        self.assertAlmostEqual(gap, -0.01, places=9)

    def test_insert_step_moves_along_port_axis(self) -> None:
        tip = np.array([0.05, 0.0, 0.1])
        axis = np.array([-1.0, 0.0, 0.0])
        nxt = insert_target_tip(tip, axis, 0.002)
        np.testing.assert_allclose(nxt, [0.048, 0.0, 0.1], atol=1e-9)

    def test_clamp_nudge_limits_step(self) -> None:
        pos, rot = clamp_nudge(
            np.array([0.01, 0.0, 0.0]),
            np.array([1.0, 0.0, 0.0]),
            max_pos_m=0.002,
            max_rot_rad=0.05,
        )
        self.assertAlmostEqual(float(np.linalg.norm(pos)), 0.002, places=6)
        self.assertAlmostEqual(float(np.linalg.norm(rot)), 0.05, places=6)


if __name__ == "__main__":
    unittest.main()
