#!/usr/bin/env python3
"""Contract tests for the existing stereo camera geometry in perception.py."""

from __future__ import annotations

import unittest

import numpy as np

from perception import (
    CameraModel,
    build_virtual_camera_model,
    camera_point_error_to_world,
    transform_point_to_world,
    triangulate_pixel_pair,
)


IMAGE_HEIGHT = 480
IMAGE_WIDTH = 640
FOCAL_LENGTH_MM = 18.0
HORIZONTAL_APERTURE_MM = 20.955
VERTICAL_APERTURE_MM = 20.955 * 9.0 / 16.0
BASELINE_M = 0.040


def _world_from_camera(
    center_world_m: tuple[float, float, float],
    rotation_row: np.ndarray | None = None,
) -> np.ndarray:
    """Build the row-vector USD matrix used by the project."""
    matrix = np.eye(4, dtype=np.float64)
    if rotation_row is not None:
        rotation = np.asarray(rotation_row, dtype=np.float64)
        if rotation.shape != (3, 3):
            raise ValueError("rotation_row must have shape (3, 3)")
        matrix[:3, :3] = rotation
    matrix[3, :3] = np.asarray(center_world_m, dtype=np.float64)
    return matrix


def _camera(
    center_world_m: tuple[float, float, float],
    rotation_row: np.ndarray | None = None,
) -> CameraModel:
    return CameraModel(
        image_height_px=IMAGE_HEIGHT,
        image_width_px=IMAGE_WIDTH,
        focal_length_mm=FOCAL_LENGTH_MM,
        horizontal_aperture_mm=HORIZONTAL_APERTURE_MM,
        vertical_aperture_mm=VERTICAL_APERTURE_MM,
        world_from_camera=_world_from_camera(
            center_world_m,
            rotation_row=rotation_row,
        ),
    )


class CameraTransformTests(unittest.TestCase):
    def test_camera_world_round_trip_recovers_local_point(self) -> None:
        rotation_row = np.array(
            [
                [0.0, 1.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        camera = _camera(
            (1.25, -0.40, 0.85),
            rotation_row=rotation_row,
        )
        point_camera = np.array([0.025, -0.012, -0.240])

        point_world = transform_point_to_world(
            point_camera,
            camera.world_from_camera,
        )
        recovered_camera = camera.camera_point_from_world(point_world)

        np.testing.assert_allclose(
            recovered_camera,
            point_camera,
            atol=1.0e-12,
            rtol=0.0,
        )

    def test_center_pixel_ray_matches_camera_negative_z_axis(self) -> None:
        rotation_row = np.array(
            [
                [0.0, 1.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        camera = _camera(
            (0.3, -0.2, 1.1),
            rotation_row=rotation_row,
        )

        origin, direction = camera.pixel_to_world_ray(
            (camera.cx_px, camera.cy_px)
        )
        expected_direction = (
            np.array([0.0, 0.0, -1.0, 0.0])
            @ camera.world_from_camera
        )[:3]
        expected_direction /= np.linalg.norm(expected_direction)

        np.testing.assert_allclose(
            origin,
            camera.camera_center_world_m,
            atol=1.0e-12,
            rtol=0.0,
        )
        np.testing.assert_allclose(
            direction,
            expected_direction,
            atol=1.0e-12,
            rtol=0.0,
        )


class StereoGeometryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.left = _camera((-BASELINE_M / 2.0, 0.0, 0.0))
        self.right = _camera((+BASELINE_M / 2.0, 0.0, 0.0))
        self.virtual = build_virtual_camera_model(self.left, self.right)

    def test_virtual_camera_is_exact_stereo_midpoint(self) -> None:
        np.testing.assert_allclose(
            self.virtual.camera_center_world_m,
            np.zeros(3),
            atol=1.0e-12,
            rtol=0.0,
        )
        np.testing.assert_allclose(
            self.virtual.world_from_camera[:3, :3],
            self.left.world_from_camera[:3, :3],
            atol=1.0e-12,
            rtol=0.0,
        )

    def test_project_then_triangulate_recovers_known_world_point(self) -> None:
        point_world = np.array([0.008, -0.004, -0.200])

        left_uv = self.left.project_world(point_world)
        right_uv = self.right.project_world(point_world)
        reconstructed, ray_gap = triangulate_pixel_pair(
            left_uv,
            right_uv,
            self.left,
            self.right,
        )

        np.testing.assert_allclose(
            reconstructed,
            point_world,
            atol=1.0e-10,
            rtol=0.0,
        )
        self.assertLess(ray_gap, 1.0e-10)
        np.testing.assert_allclose(
            self.left.project_world(reconstructed),
            left_uv,
            atol=1.0e-9,
            rtol=0.0,
        )
        np.testing.assert_allclose(
            self.right.project_world(reconstructed),
            right_uv,
            atol=1.0e-9,
            rtol=0.0,
        )

    def test_disparity_matches_focal_baseline_depth_equation(self) -> None:
        depth_m = 0.200
        point_world = np.array([0.0, 0.0, -depth_m])

        left_uv = self.left.project_world(point_world)
        right_uv = self.right.project_world(point_world)
        disparity_px = float(left_uv[0] - right_uv[0])
        expected_disparity_px = (
            self.left.fx_px * BASELINE_M / depth_m
        )
        recovered_depth_m = (
            self.left.fx_px * BASELINE_M / abs(disparity_px)
        )

        self.assertGreater(disparity_px, 0.0)
        self.assertAlmostEqual(
            disparity_px,
            expected_disparity_px,
            places=10,
        )
        self.assertAlmostEqual(
            recovered_depth_m,
            depth_m,
            places=12,
        )

    def test_projected_stereo_pair_is_rectified(self) -> None:
        point_world = np.array([-0.010, 0.006, -0.240])

        left_uv = self.left.project_world(point_world)
        right_uv = self.right.project_world(point_world)

        self.assertAlmostEqual(left_uv[1], right_uv[1], places=10)
        self.assertGreater(left_uv[0] - right_uv[0], 0.0)


class CorrectionDirectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.world_from_camera = np.eye(4, dtype=np.float64)
        self.desired = np.array([0.0, 0.0, -0.200])

    def _correction(self, current: tuple[float, float, float]) -> np.ndarray:
        return camera_point_error_to_world(
            np.asarray(current, dtype=np.float64),
            self.desired,
            self.world_from_camera,
        )

    def test_left_right_corrections_have_opposite_x_signs(self) -> None:
        right_of_target = self._correction((+0.010, 0.0, -0.200))
        left_of_target = self._correction((-0.010, 0.0, -0.200))

        self.assertGreater(right_of_target[0], 0.0)
        self.assertLess(left_of_target[0], 0.0)

    def test_above_below_corrections_have_opposite_y_signs(self) -> None:
        above_target = self._correction((0.0, +0.010, -0.200))
        below_target = self._correction((0.0, -0.010, -0.200))

        self.assertGreater(above_target[1], 0.0)
        self.assertLess(below_target[1], 0.0)

    def test_far_near_corrections_have_expected_z_signs(self) -> None:
        too_far = self._correction((0.0, 0.0, -0.240))
        too_near = self._correction((0.0, 0.0, -0.160))

        self.assertLess(too_far[2], 0.0)
        self.assertGreater(too_near[2], 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
