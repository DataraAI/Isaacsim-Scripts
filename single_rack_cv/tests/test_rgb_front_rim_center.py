from __future__ import annotations

import inspect
import unittest

import cv2
import numpy as np

from stereo_center import (
    aperture_center_pixel,
    estimate_stereo_aperture_center,
)


IMAGE_HEIGHT = 240
IMAGE_WIDTH = 320
FOCAL_PX = 550.0


class SyntheticCamera:
    def __init__(self, position_xyz) -> None:
        self.position = np.asarray(position_xyz, dtype=np.float64)
        self.fx_px = FOCAL_PX
        self.fy_px = FOCAL_PX
        self.image_height_px = IMAGE_HEIGHT
        self.image_width_px = IMAGE_WIDTH
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

    def pixel_to_world_ray(self, pixel_uv):
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


def stepped_aperture() -> np.ndarray:
    return np.array(
        [
            [-0.50, -0.50],
            [+0.50, -0.50],
            [+0.50, +0.12],
            [+0.24, +0.12],
            [+0.24, +0.50],
            [-0.24, +0.50],
            [-0.24, +0.12],
            [-0.50, +0.12],
        ],
        dtype=np.float64,
    )


def render_front_rim(
    camera,
    center,
    horizontal,
    vertical,
    *,
    mask_shift_px: float,
) -> tuple[np.ndarray, np.ndarray]:
    width_m = 0.0114
    height_m = 0.0070
    template = stepped_aperture()
    polygon_world = (
        center[None, :]
        + template[:, :1] * width_m * horizontal[None, :]
        + template[:, 1:] * height_m * vertical[None, :]
    )
    polygon_uv = np.vstack(
        [camera.project_world(point) for point in polygon_world]
    )
    polygon_int = np.rint(polygon_uv).astype(np.int32)

    rgb = np.full(
        (IMAGE_HEIGHT, IMAGE_WIDTH, 3),
        180,
        dtype=np.uint8,
    )
    opening = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.uint8)
    cv2.fillPoly(opening, [polygon_int], 255)

    minimum_row = int(np.min(polygon_int[:, 1]))
    maximum_row = int(np.max(polygon_int[:, 1]))
    for row in range(minimum_row, maximum_row + 1):
        fraction = (row - minimum_row) / max(
            1,
            maximum_row - minimum_row,
        )
        value = int(round(20.0 + 110.0 * fraction))
        rgb[row][opening[row] > 0] = value

    recessed = polygon_uv.copy()
    recessed[:, 0] += float(mask_shift_px)
    shoulder_row = float(np.mean(recessed[[2, 3, 6, 7], 1]))
    bottom_row = float(np.max(recessed[:, 1]))
    cutoff_row = shoulder_row + 0.45 * (bottom_row - shoulder_row)
    recessed[0, 1] = cutoff_row
    recessed[1, 1] = cutoff_row

    mask = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.uint8)
    cv2.fillPoly(mask, [np.rint(recessed).astype(np.int32)], 255)
    return rgb, mask


class RGBFrontRimCenterTests(unittest.TestCase):
    def test_recessed_mask_parallax_does_not_define_stereo_depth(self):
        left = SyntheticCamera((-0.02, 0.0, 0.0))
        right = SyntheticCamera((+0.02, 0.0, 0.0))
        center = np.array([0.004, -0.003, -0.16], dtype=np.float64)
        normal = np.array([0.20, -0.08, 0.976], dtype=np.float64)
        normal /= np.linalg.norm(normal)
        baseline = right.camera_center_world_m - left.camera_center_world_m
        horizontal = baseline - float(np.dot(baseline, normal)) * normal
        horizontal /= np.linalg.norm(horizontal)
        vertical = np.cross(normal, horizontal)
        vertical /= np.linalg.norm(vertical)

        left_rgb, left_mask = render_front_rim(
            left,
            center,
            horizontal,
            vertical,
            mask_shift_px=-3.0,
        )
        right_rgb, right_mask = render_front_rim(
            right,
            center,
            horizontal,
            vertical,
            mask_shift_px=+4.0,
        )

        result = estimate_stereo_aperture_center(
            left_rgb=left_rgb,
            right_rgb=right_rgb,
            left_mask=left_mask,
            right_mask=right_mask,
            left_camera=left,
            right_camera=right,
        )

        self.assertLess(
            float(np.linalg.norm(result.center_world_m - center)),
            0.00075,
        )
        self.assertLess(result.ray_gap_m, 0.0005)

    def test_mask_shift_does_not_move_one_eye_front_rim_center(self):
        camera = SyntheticCamera((0.0, 0.0, 0.0))
        center = np.array([0.0, 0.0, -0.16], dtype=np.float64)
        horizontal = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        vertical = np.array([0.0, 1.0, 0.0], dtype=np.float64)

        rgb, centered_mask = render_front_rim(
            camera,
            center,
            horizontal,
            vertical,
            mask_shift_px=0.0,
        )
        _, shifted_mask = render_front_rim(
            camera,
            center,
            horizontal,
            vertical,
            mask_shift_px=5.0,
        )

        centered = aperture_center_pixel(rgb, centered_mask, camera)
        shifted = aperture_center_pixel(rgb, shifted_mask, camera)
        self.assertLess(float(np.linalg.norm(centered - shifted)), 1.0)

    def test_runtime_api_requires_rgb_and_has_no_manual_offset(self):
        parameters = inspect.signature(
            estimate_stereo_aperture_center
        ).parameters
        self.assertIn("left_rgb", parameters)
        self.assertIn("right_rgb", parameters)
        forbidden = {"offset", "world_offset", "center_offset", "bias"}
        self.assertTrue(forbidden.isdisjoint(parameters))


if __name__ == "__main__":
    unittest.main()
