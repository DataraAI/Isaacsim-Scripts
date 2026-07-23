from __future__ import annotations

from types import SimpleNamespace
import unittest

import numpy as np

from affine_root_geometry import (
    compute_world_from_root_for_tip_preserving_affine,
)
from cable_geometry import rigid_pose_from_affine


def _transform(linear=None, translation=(0.0, 0.0, 0.0)):
    result = np.eye(4, dtype=np.float64)
    if linear is not None:
        result[:3, :3] = np.asarray(linear, dtype=np.float64)
    result[:3, 3] = np.asarray(translation, dtype=np.float64)
    return result


class AffineRootGeometryTests(unittest.TestCase):
    def test_scaled_root_and_plug_align_tip_without_changing_root_scale(self):
        angle = np.deg2rad(35.0)
        rotation = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0.0],
                [np.sin(angle), np.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        world_from_root = _transform(
            rotation @ np.diag([0.01, 0.02, 0.03]),
            (0.4, -0.2, 0.8),
        )
        root_from_plug = _transform(
            np.diag([2.0, 1.0, 0.5]),
            (0.1, 0.2, -0.05),
        )
        world_from_plug = world_from_root @ root_from_plug
        plug_from_tip = _transform(translation=(0.018, 0.0, 0.0))
        frame = SimpleNamespace(plug_from_tip=plug_from_tip)
        desired_world_from_tip = _transform(
            translation=(0.9, 0.1, 1.3)
        )

        mounted_world_from_root = (
            compute_world_from_root_for_tip_preserving_affine(
                world_from_root,
                world_from_plug,
                frame,
                desired_world_from_tip,
            )
        )
        actual_world_from_tip = (
            mounted_world_from_root @ root_from_plug @ plug_from_tip
        )
        actual_tip_pose = rigid_pose_from_affine(
            actual_world_from_tip,
            "actual_world_from_tip",
        )

        np.testing.assert_allclose(
            actual_tip_pose,
            desired_world_from_tip,
            atol=1.0e-9,
        )
        np.testing.assert_allclose(
            np.linalg.svd(
                mounted_world_from_root[:3, :3],
                compute_uv=False,
            ),
            np.linalg.svd(
                world_from_root[:3, :3],
                compute_uv=False,
            ),
            atol=1.0e-12,
        )


if __name__ == "__main__":
    unittest.main()
