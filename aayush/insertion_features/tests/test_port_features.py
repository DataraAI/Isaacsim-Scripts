"""Synthetic stepped RJ45 jack-opening tests."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

AAYUSH_DIR = Path(__file__).resolve().parents[2]
if str(AAYUSH_DIR) not in sys.path:
    sys.path.insert(0, str(AAYUSH_DIR))

from insertion_features.port_geometry import (
    extract_port_features_from_aperture_points,
    extract_port_features_from_pack_mesh,
    infer_port_insertion_axis,
)


def _stepped_aperture(*, latch_up: bool) -> np.ndarray:
    """12-point stepped RJ45 opening in the YZ plane at X=0 (meters).

    Big rectangle under a smaller latch rectangle under a smallest tip,
    or the same stack mirrored when the lower-row jack is inverted.
    """

    x = 0.0
    y_outer = 0.005955
    y_small = 0.003150
    y_tip = 0.002030
    z_big_lo = 0.0
    z_big_hi = 0.006830
    z_mid = 0.008565
    z_tip = 0.009600
    if latch_up:
        points = [
            [-y_outer, z_big_lo],
            [y_outer, z_big_lo],
            [-y_outer, z_big_hi],
            [-y_small, z_big_hi],
            [y_small, z_big_hi],
            [y_outer, z_big_hi],
            [-y_small, z_mid],
            [y_small, z_mid],
            [-y_tip, z_mid],
            [y_tip, z_mid],
            [-y_tip, z_tip],
            [y_tip, z_tip],
        ]
    else:
        points = [
            [-y_outer, -z_big_lo],
            [y_outer, -z_big_lo],
            [-y_outer, -z_big_hi],
            [-y_small, -z_big_hi],
            [y_small, -z_big_hi],
            [y_outer, -z_big_hi],
            [-y_small, -z_mid],
            [y_small, -z_mid],
            [-y_tip, -z_mid],
            [y_tip, -z_mid],
            [-y_tip, -z_tip],
            [y_tip, -z_tip],
        ]
    yz = np.asarray(points, dtype=np.float64)
    return np.column_stack((np.full(len(yz), x), yz[:, 0], yz[:, 1]))


class PortApertureTests(unittest.TestCase):
    def test_upper_jack_uses_big_rectangle_center_not_full_silhouette(self) -> None:
        points = _stepped_aperture(latch_up=True)
        features = extract_port_features_from_aperture_points(
            points,
            insertion_axis=np.array([1.0, 0.0, 0.0]),
        )
        np.testing.assert_allclose(features.insertion_axis, [1.0, 0.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(
            features.mating_center,
            [0.0, 0.0, 0.003415],
            atol=2e-4,
        )
        silhouette_center_z = 0.5 * (points[:, 2].min() + points[:, 2].max())
        self.assertLess(features.mating_center[2], silhouette_center_z - 0.0005)
        height = features.mating_corners[:, 2].max() - features.mating_corners[:, 2].min()
        width = features.mating_corners[:, 1].max() - features.mating_corners[:, 1].min()
        self.assertAlmostEqual(width, 0.01191, places=5)
        self.assertAlmostEqual(height, 0.00683, places=5)

    def test_latch_keypoints_are_top_corners_of_rectangle_above_the_big_opening(self) -> None:
        points = _stepped_aperture(latch_up=True)
        features = extract_port_features_from_aperture_points(
            points,
            insertion_axis=np.array([1.0, 0.0, 0.0]),
        )
        self.assertEqual(features.latch_keypoints.shape, (2, 3))
        for corner in features.latch_keypoints:
            self.assertAlmostEqual(corner[2], 0.008565, places=4)
            self.assertAlmostEqual(abs(corner[1]), 0.003150, places=4)

    def test_inverted_lower_jack_latch_keypoints_are_on_the_latch_side(self) -> None:
        points = _stepped_aperture(latch_up=False)
        features = extract_port_features_from_aperture_points(
            points,
            insertion_axis=np.array([1.0, 0.0, 0.0]),
        )
        self.assertLess(features.mating_center[2], 0.0)
        for corner in features.latch_keypoints:
            self.assertAlmostEqual(corner[2], -0.008565, places=4)
            self.assertAlmostEqual(abs(corner[1]), 0.003150, places=4)


def _fan_mesh(points: np.ndarray, *, normal_x: float = 1.0):
    """Triangle fan covering an aperture, with faces pointing along +X."""

    values = np.asarray(points, dtype=np.float64)
    centroid = values.mean(axis=0)
    mesh_points = np.vstack((centroid.reshape(1, 3), values))
    n = values.shape[0]
    faces = []
    for index in range(n):
        a = 1 + index
        b = 1 + ((index + 1) % n)
        if normal_x >= 0.0:
            faces.append([0, a, b])
        else:
            faces.append([0, b, a])
    counts = np.full(n, 3, dtype=np.int64)
    indices = np.asarray(faces, dtype=np.int64).reshape(-1)
    return mesh_points, counts, indices


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
    return np.vstack(points), np.concatenate(counts), np.concatenate(indices)


class PackMeshTests(unittest.TestCase):
    def test_insertion_axis_points_from_the_proud_outer_shell_into_the_pack(self) -> None:
        inner = np.array(
            [
                [-0.1568, -0.041, 0.000],
                [-0.1568, 0.041, 0.024],
                [-0.1436, -0.041, 0.000],
                [-0.1436, 0.041, 0.024],
            ]
        )
        outer = np.array(
            [
                [-0.1653, -0.044, -0.001],
                [-0.1653, 0.044, 0.025],
                [-0.1434, -0.044, -0.001],
                [-0.1434, 0.044, 0.025],
            ]
        )
        axis = infer_port_insertion_axis(inner_points=inner, outer_points=outer)
        np.testing.assert_allclose(axis, [1.0, 0.0, 0.0], atol=1e-6)

    def test_pack_mesh_splits_upper_and_lower_jacks_and_keeps_latch_sides(self) -> None:
        upper = _stepped_aperture(latch_up=True) + np.array([0.0, 0.020, 0.012])
        lower = _stepped_aperture(latch_up=False) + np.array([0.0, 0.020, -0.012])
        other = _stepped_aperture(latch_up=True) + np.array([0.0, -0.020, 0.012])
        points, counts, indices = _merge_meshes(
            [
                _fan_mesh(upper),
                _fan_mesh(lower),
                _fan_mesh(other),
            ]
        )
        ports = extract_port_features_from_pack_mesh(
            points=points,
            face_vertex_counts=counts,
            face_vertex_indices=indices,
            insertion_axis=np.array([1.0, 0.0, 0.0]),
        )
        self.assertEqual(len(ports), 3)
        z_centers = sorted(float(port.mating_center[2]) for port in ports)
        self.assertLess(z_centers[0], 0.0)
        self.assertGreater(z_centers[1], 0.0)
        self.assertGreater(z_centers[2], 0.0)
        inverted = min(ports, key=lambda port: port.mating_center[2])
        self.assertLess(float(np.dot(inverted.up_axis, [0.0, 0.0, 1.0])), 0.0)
        for corner in inverted.latch_keypoints:
            self.assertLess(corner[2], inverted.mating_center[2])


if __name__ == "__main__":
    unittest.main()
