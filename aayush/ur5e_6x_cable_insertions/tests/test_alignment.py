from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
AAYUSH_DIR = REPO_ROOT / "aayush"
if str(AAYUSH_DIR) not in sys.path:
    sys.path.insert(0, str(AAYUSH_DIR))

from insertion_features.cache import local_features_for_head, local_features_for_jack
from insertion_features.geometry import ConnectorFeatures, Plane, transform_connector_features
from ur5e_6x_cable_insertions import alignment
from ur5e_6x_cable_insertions import config as cfg


def _box_features(*, origin, axis, width, up, half_w=0.005, half_u=0.003):
    axis = np.asarray(axis, dtype=np.float64) / np.linalg.norm(axis)
    width = np.asarray(width, dtype=np.float64) / np.linalg.norm(width)
    up = np.asarray(up, dtype=np.float64) / np.linalg.norm(up)
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


def _nominal_head_at_port():
    head = local_features_for_head(cfg.HEAD45_NAME)
    port = local_features_for_jack(cfg.DEFAULT_JACK_ID)
    head_basis = np.column_stack((head.insertion_axis, head.width_axis, head.up_axis))
    port_basis = np.column_stack((port.insertion_axis, port.width_axis, port.up_axis))
    rotation = port_basis @ np.linalg.inv(head_basis)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = port.mating_center - rotation @ head.mating_center
    return transform_connector_features(head, transform), port


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
        crystal_kp = crystal.latch_keypoints.copy()
        crystal_kp[:, 2] = 0.099
        port_kp = port.latch_keypoints.copy()
        port_kp[:, 2] = 0.105
        crystal = ConnectorFeatures(
            **{**crystal.__dict__, "latch_keypoints": crystal_kp}
        )
        port = ConnectorFeatures(**{**port.__dict__, "latch_keypoints": port_kp})
        residual = alignment.evaluate_alignment(
            crystal,
            port,
            latch_z_margin_m=0.0005,
            mating_side_margin_m=0.0005,
            axis_dot_min=0.98,
        )
        self.assertTrue(residual.passed)

    def test_mating_gap_along_axis(self) -> None:
        port = _box_features(
            origin=[0.0, 0.0, 0.1], axis=[-1, 0, 0], width=[0, 1, 0], up=[0, 0, 1]
        )
        crystal = _box_features(
            origin=[0.01, 0.0, 0.1], axis=[-1, 0, 0], width=[0, 1, 0], up=[0, 0, 1]
        )
        self.assertAlmostEqual(alignment.mating_gap_along_axis(crystal, port), -0.01)

    def test_insert_step_moves_along_port_axis(self) -> None:
        nxt = alignment.insert_target_tip(
            np.array([0.05, 0.0, 0.1]), np.array([-1.0, 0.0, 0.0]), 0.002
        )
        np.testing.assert_allclose(nxt, [0.048, 0.0, 0.1], atol=1e-9)

    def test_clamp_nudge_limits_step(self) -> None:
        pos, rot = alignment.clamp_nudge(
            np.array([0.01, 0.0, 0.0]),
            np.array([1.0, 0.0, 0.0]),
            max_pos_m=0.002,
            max_rot_rad=0.05,
        )
        self.assertAlmostEqual(float(np.linalg.norm(pos)), 0.002, places=6)
        self.assertAlmostEqual(float(np.linalg.norm(rot)), 0.05, places=6)

    def test_real_caches_pass_at_nominal_mate_with_shipped_constants(self) -> None:
        crystal, port = _nominal_head_at_port()
        residual = alignment.evaluate_alignment(
            crystal,
            port,
            latch_z_margin_m=cfg.LATCH_Z_MARGIN_M,
            mating_side_margin_m=cfg.MATING_SIDE_MARGIN_M,
            axis_dot_min=cfg.AXIS_DOT_MIN,
        )
        self.assertTrue(residual.passed, residual)

    def test_latch_z_does_not_nudge_after_clearance_passes(self) -> None:
        crystal, port = _nominal_head_at_port()
        residual = alignment.evaluate_alignment(
            crystal,
            port,
            latch_z_margin_m=cfg.LATCH_Z_MARGIN_M,
            mating_side_margin_m=0.0,
            axis_dot_min=cfg.AXIS_DOT_MIN,
        )
        self.assertTrue(residual.latch_z_ok)
        np.testing.assert_allclose(residual.pos_error_m, np.zeros(3), atol=1e-12)

    def test_port_standoff_follows_negative_insertion_axis(self) -> None:
        _crystal, port = _nominal_head_at_port()
        target = alignment.port_standoff_target(port, 0.02)
        axis = port.insertion_axis / np.linalg.norm(port.insertion_axis)
        np.testing.assert_allclose(target, port.mating_center - 0.02 * axis)
        self.assertAlmostEqual(float(target[0]), float(port.mating_center[0]), places=8)

    def test_scaled_linear_transform_fails_loudly(self) -> None:
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] *= 0.01
        with self.assertRaisesRegex(ValueError, "unit scale"):
            alignment.assert_unit_linear_scale(transform, label="head")


if __name__ == "__main__":
    unittest.main()
