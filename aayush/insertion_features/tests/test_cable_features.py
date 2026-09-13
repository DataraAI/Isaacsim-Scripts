"""Synthetic RJ45 tests for crystal-head mating and latch features."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

AAYUSH_DIR = Path(__file__).resolve().parents[2]
if str(AAYUSH_DIR) not in sys.path:
    sys.path.insert(0, str(AAYUSH_DIR))

from insertion_features.geometry import extract_connector_features_from_mesh


def _box_mesh(minimum: np.ndarray, maximum: np.ndarray):
    mn = np.asarray(minimum, dtype=np.float64).reshape(3)
    mx = np.asarray(maximum, dtype=np.float64).reshape(3)
    xmin, ymin, zmin = mn
    xmax, ymax, zmax = mx
    points = np.array(
        [
            [xmin, ymin, zmin],
            [xmax, ymin, zmin],
            [xmax, ymax, zmin],
            [xmin, ymax, zmin],
            [xmin, ymin, zmax],
            [xmax, ymin, zmax],
            [xmax, ymax, zmax],
            [xmin, ymax, zmax],
        ],
        dtype=np.float64,
    )
    faces = np.array(
        [
            [0, 3, 2, 1],
            [4, 5, 6, 7],
            [0, 1, 5, 4],
            [3, 7, 6, 2],
            [1, 2, 6, 5],
            [0, 4, 7, 3],
        ],
        dtype=np.int64,
    )
    counts = np.full(6, 4, dtype=np.int64)
    indices = faces.reshape(-1)
    return points, counts, indices


def _merge_meshes(meshes):
    points = []
    counts = []
    indices = []
    offset = 0
    for mesh_points, mesh_counts, mesh_indices in meshes:
        points.append(mesh_points)
        counts.append(mesh_counts)
        indices.append(mesh_indices + offset)
        offset += len(mesh_points)
    return (
        np.vstack(points),
        np.concatenate(counts),
        np.concatenate(indices),
    )


def _rj45_housing(*, tip_sign: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Axis-aligned RJ45 housing with a proud latch tab on +Z."""

    length = 0.0205
    half_width = 0.00525
    body_z_min = -0.00591
    body_z_max = 0.00122
    tip = 0.5 * length * float(tip_sign)
    cable_end = -tip
    x_lo, x_hi = sorted((cable_end, tip))
    body = _box_mesh(
        [x_lo, -half_width, body_z_min],
        [x_hi, half_width, body_z_max],
    )
    latch_protrusion = 0.00050
    latch_y = 0.00273
    latch_z_min = 0.00000
    latch_z_max = 0.00225
    if tip_sign > 0.0:
        latch = _box_mesh(
            [tip - 0.00010, -latch_y, latch_z_min],
            [tip + latch_protrusion, latch_y, latch_z_max],
        )
    else:
        latch = _box_mesh(
            [tip - latch_protrusion, -latch_y, latch_z_min],
            [tip + 0.00010, latch_y, latch_z_max],
        )
    return _merge_meshes((body, latch))


class ConnectorFeatureTests(unittest.TestCase):
    def test_plus_x_head_mating_plane_excludes_latch_and_uses_face_center(self):
        points, counts, indices = _rj45_housing(tip_sign=1.0)
        features = extract_connector_features_from_mesh(
            points=points,
            face_vertex_counts=counts,
            face_vertex_indices=indices,
            cable_center_world=np.array([-0.12, 0.0, 0.0]),
        )

        np.testing.assert_allclose(features.insertion_axis, [1.0, 0.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(features.mating_plane.normal, [1.0, 0.0, 0.0], atol=1e-6)
        self.assertAlmostEqual(features.mating_plane.point[0], 0.01025, places=5)
        np.testing.assert_allclose(
            features.mating_center,
            [0.01025, 0.0, -0.002345],
            atol=2e-4,
        )
        mesh_centroid = points.mean(axis=0)
        self.assertGreater(
            abs(features.mating_center[2] - mesh_centroid[2]),
            0.001,
        )
        self.assertGreater(features.mating_center[0], mesh_centroid[0])

    def test_latch_keypoints_are_the_two_higher_corners_of_the_latch_face(self):
        points, counts, indices = _rj45_housing(tip_sign=1.0)
        features = extract_connector_features_from_mesh(
            points=points,
            face_vertex_counts=counts,
            face_vertex_indices=indices,
            cable_center_world=np.array([-0.12, 0.0, 0.0]),
        )

        self.assertEqual(features.latch_corners.shape, (4, 3))
        self.assertEqual(features.latch_keypoints.shape, (2, 3))
        np.testing.assert_allclose(features.latch_plane.normal, features.insertion_axis, atol=1e-6)
        self.assertGreater(features.latch_plane.point[0], features.mating_plane.point[0])
        for corner in features.latch_keypoints:
            self.assertAlmostEqual(corner[2], 0.00225, places=4)
            self.assertAlmostEqual(abs(corner[1]), 0.00273, places=4)
        lower_z = sorted(features.latch_corners[:, 2])[:2]
        self.assertTrue(np.all(np.asarray(lower_z) < 0.0005))

    def test_minus_x_head_inserts_away_from_the_cable(self):
        points, counts, indices = _rj45_housing(tip_sign=-1.0)
        features = extract_connector_features_from_mesh(
            points=points,
            face_vertex_counts=counts,
            face_vertex_indices=indices,
            cable_center_world=np.array([0.12, 0.0, 0.0]),
        )

        np.testing.assert_allclose(features.insertion_axis, [-1.0, 0.0, 0.0], atol=1e-6)
        self.assertLess(features.mating_center[0], 0.0)
        self.assertLess(features.latch_plane.point[0], features.mating_plane.point[0])
        for corner in features.latch_keypoints:
            self.assertAlmostEqual(corner[2], 0.00225, places=4)


if __name__ == "__main__":
    unittest.main()
