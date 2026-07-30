from __future__ import annotations

import inspect
import unittest

import cv2
import numpy as np

from stereo_center import estimate_stereo_aperture_center


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
    width_m = 0.0114
    height_m = 0.0070
    return np.array(
        [
            [-0.50 * width_m, -0.50 * height_m],
            [+0.50 * width_m, -0.50 * height_m],
            [+0.50 * width_m, +0.12 * height_m],
            [+0.24 * width_m, +0.12 * height_m],
            [+0.24 * width_m, +0.50 * height_m],
            [-0.24 * width_m, +0.50 * height_m],
            [-0.24 * width_m, +0.12 * height_m],
            [-0.50 * width_m, +0.12 * height_m],
        ],
        dtype=np.float64,
    )


def render_mask(
    camera,
    center,
    horizontal,
    vertical,
    *,
    top_shift_px: float = 0.0,
) -> np.ndarray:
    polygon_xy = stepped_aperture()
    polygon_world = (
        center[None, :]
        + polygon_xy[:, :1] * horizontal[None, :]
        + polygon_xy[:, 1:] * vertical[None, :]
    )
    polygon_uv = np.vstack(
        [camera.project_world(point) for point in polygon_world]
    )
    polygon_uv[4:6, 0] += float(top_shift_px)
    mask = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.uint8)
    cv2.fillPoly(mask, [np.rint(polygon_uv).astype(np.int32)], 255)
    return mask


class StereoApertureCenterTests(unittest.TestCase):
    def test_direct_center_survives_oblique_view_and_top_asymmetry(self):
        left = SyntheticCamera((-0.02, 0.0, 0.0))
        right = SyntheticCamera((0.02, 0.0, 0.0))
        center = np.array([0.004, -0.003, -0.16], dtype=np.float64)
        normal = np.array([0.20, -0.08, 0.976], dtype=np.float64)
        normal /= np.linalg.norm(normal)
        baseline = right.camera_center_world_m - left.camera_center_world_m
        horizontal = baseline - float(np.dot(baseline, normal)) * normal
        horizontal /= np.linalg.norm(horizontal)
        vertical = np.cross(normal, horizontal)
        vertical /= np.linalg.norm(vertical)

        result = estimate_stereo_aperture_center(
            left_mask=render_mask(
                left,
                center,
                horizontal,
                vertical,
                top_shift_px=-3.0,
            ),
            right_mask=render_mask(
                right,
                center,
                horizontal,
                vertical,
                top_shift_px=4.0,
            ),
            left_camera=left,
            right_camera=right,
        )

        self.assertLess(
            float(np.linalg.norm(result.center_world_m - center)),
            0.00035,
        )
        self.assertLess(result.ray_gap_m, 0.0005)

    def test_bad_vertical_correspondence_fails_closed(self):
        left = SyntheticCamera((-0.02, 0.0, 0.0))
        right = SyntheticCamera((0.02, 0.0, 0.0))
        center = np.array([0.0, 0.0, -0.16], dtype=np.float64)
        horizontal = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        vertical = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        left_mask = render_mask(left, center, horizontal, vertical)
        right_mask = np.roll(
            render_mask(right, center, horizontal, vertical),
            shift=8,
            axis=0,
        )

        with self.assertRaisesRegex(RuntimeError, "ray gap"):
            estimate_stereo_aperture_center(
                left_mask=left_mask,
                right_mask=right_mask,
                left_camera=left,
                right_camera=right,
            )

    def test_no_manual_offset_parameter_exists(self):
        parameters = inspect.signature(
            estimate_stereo_aperture_center
        ).parameters
        forbidden = {"offset", "world_offset", "center_offset", "bias"}
        self.assertTrue(forbidden.isdisjoint(parameters))


if __name__ == "__main__":
    unittest.main()
