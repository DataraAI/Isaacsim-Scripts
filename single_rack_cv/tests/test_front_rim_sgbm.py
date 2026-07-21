from __future__ import annotations

import unittest

import cv2
import numpy as np

from front_rim_sgbm import (
    build_bezel_ring_pixels,
    compute_local_disparity,
    estimate_front_plane_sgbm,
    select_nearest_range_cluster,
)


IMAGE_HEIGHT = 160
IMAGE_WIDTH = 260
FOCAL_PX = 550.0
LEFT_BBOX = (130, 60, 40, 24)
LEFT_CENTER = (150.0, 72.0)


class SyntheticCamera:
    def __init__(self, position_xyz: tuple[float, float, float]) -> None:
        self.position = np.asarray(position_xyz, dtype=np.float64)
        self.fx_px = FOCAL_PX
        self.fy_px = FOCAL_PX
        self.cx_px = (IMAGE_WIDTH - 1) / 2.0
        self.cy_px = (IMAGE_HEIGHT - 1) / 2.0

    @property
    def camera_center_world_m(self) -> np.ndarray:
        return self.position.copy()

    def project_world(self, point_world_m: np.ndarray) -> np.ndarray:
        local = np.asarray(point_world_m, dtype=np.float64) - self.position
        depth = -float(local[2])
        if depth <= 0.0:
            raise RuntimeError("Point is behind camera.")
        return np.array(
            [
                self.cx_px + self.fx_px * float(local[0]) / depth,
                self.cy_px - self.fy_px * float(local[1]) / depth,
            ],
            dtype=np.float64,
        )

    def pixel_to_world_ray(
        self,
        pixel_uv: np.ndarray | tuple[float, float],
    ) -> tuple[np.ndarray, np.ndarray]:
        u, v = np.asarray(pixel_uv, dtype=np.float64).reshape(2)
        direction = np.array(
            [
                (u - self.cx_px) / self.fx_px,
                -(v - self.cy_px) / self.fy_px,
                -1.0,
            ],
            dtype=np.float64,
        )
        direction /= np.linalg.norm(direction)
        return self.position.copy(), direction


def textured_pair(
    disparity_px: float,
    vertical_offset_px: float = 1.0,
) -> tuple[
    np.ndarray,
    np.ndarray,
    tuple[int, int, int, int],
    tuple[float, float],
]:
    rng = np.random.default_rng(7)
    gray = np.clip(
        rng.normal(127.0, 45.0, (IMAGE_HEIGHT, IMAGE_WIDTH)),
        0.0,
        255.0,
    ).astype(np.uint8)
    gray = cv2.GaussianBlur(gray, (3, 3), 0.0)
    left = np.repeat(gray[:, :, None], 3, axis=2)
    transform = np.array(
        [
            [1.0, 0.0, -float(disparity_px)],
            [0.0, 1.0, -float(vertical_offset_px)],
        ],
        dtype=np.float32,
    )
    right = cv2.warpAffine(
        left,
        transform,
        (IMAGE_WIDTH, IMAGE_HEIGHT),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT101,
    )
    right_bbox = (
        int(round(LEFT_BBOX[0] - disparity_px)),
        int(round(LEFT_BBOX[1] - vertical_offset_px)),
        LEFT_BBOX[2],
        LEFT_BBOX[3],
    )
    right_center = (
        LEFT_CENTER[0] - disparity_px,
        LEFT_CENTER[1] - vertical_offset_px,
    )
    return left, right, right_bbox, right_center


class LocalDisparityTests(unittest.TestCase):
    def test_recovers_positive_and_negative_disparity(self) -> None:
        for expected in (14.0, -14.0, 40.0):
            with self.subTest(disparity=expected):
                left, right, right_bbox, right_center = textured_pair(expected)
                result = compute_local_disparity(
                    left,
                    right,
                    LEFT_BBOX,
                    LEFT_CENTER,
                    right_bbox,
                    right_center,
                )
                self.assertGreater(result.consistent_count, 500)
                recovered = result.disparity_crop_px[result.consistent_mask]
                self.assertAlmostEqual(float(np.median(recovered)), expected, places=1)

    def test_flat_images_do_not_produce_dense_consistent_depth(self) -> None:
        left = np.full((IMAGE_HEIGHT, IMAGE_WIDTH, 3), 120, dtype=np.uint8)
        right = left.copy()
        result = compute_local_disparity(
            left,
            right,
            LEFT_BBOX,
            LEFT_CENTER,
            LEFT_BBOX,
            LEFT_CENTER,
        )
        self.assertLess(result.consistent_count, 100)


class BezelSelectionTests(unittest.TestCase):
    def test_ring_contains_all_four_sides(self) -> None:
        points, labels = build_bezel_ring_pixels(
            LEFT_BBOX,
            (IMAGE_HEIGHT, IMAGE_WIDTH),
        )
        self.assertGreater(points.shape[0], 100)
        self.assertEqual(set(labels.tolist()), {0, 1, 2, 3})
        for side in range(4):
            self.assertGreater(int(np.count_nonzero(labels == side)), 10)

    def test_nearest_cluster_wins_equal_size_tie(self) -> None:
        ranges = np.array(
            [0.100, 0.101, 0.102, 0.103] * 6
            + [0.130, 0.131, 0.132, 0.133] * 6,
            dtype=np.float64,
        )
        mask = select_nearest_range_cluster(ranges, 0.004, 20)
        self.assertEqual(int(np.count_nonzero(mask)), 24)
        self.assertLess(float(np.median(ranges[mask])), 0.110)


class FrontPlaneTests(unittest.TestCase):
    def test_recovers_frontoparallel_plane(self) -> None:
        disparity_px = 14.0
        depth_m = 0.13
        baseline_m = disparity_px * depth_m / FOCAL_PX
        left_camera = SyntheticCamera((-baseline_m / 2.0, 0.0, 0.0))
        right_camera = SyntheticCamera((+baseline_m / 2.0, 0.0, 0.0))
        left, right, right_bbox, right_center = textured_pair(
            disparity_px,
            vertical_offset_px=0.0,
        )
        result = estimate_front_plane_sgbm(
            left,
            right,
            LEFT_BBOX,
            LEFT_CENTER,
            right_bbox,
            right_center,
            left_camera,
            right_camera,
        )
        self.assertGreaterEqual(min(result.side_support_counts), 4)
        self.assertAlmostEqual(result.median_disparity_px, disparity_px, places=1)
        self.assertAlmostEqual(float(result.center_world_m[2]), -depth_m, places=4)
        self.assertLess(result.plane_residual_m, 1.0e-6)
        self.assertLess(result.max_ray_gap_m, 1.0e-6)


if __name__ == "__main__":
    unittest.main()
