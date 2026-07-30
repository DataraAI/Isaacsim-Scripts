from __future__ import annotations

import unittest

import cv2
import numpy as np

from aperture_center import estimate_planar_aperture_center


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


def render_mask(camera, center, horizontal, vertical) -> np.ndarray:
    polygon_xy = stepped_aperture()
    polygon_world = (
        center[None, :]
        + polygon_xy[:, :1] * horizontal[None, :]
        + polygon_xy[:, 1:] * vertical[None, :]
    )
    polygon_uv = np.vstack(
        [camera.project_world(point) for point in polygon_world]
    )
    mask = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.uint8)
    cv2.fillPoly(mask, [np.rint(polygon_uv).astype(np.int32)], 255)
    return mask


class ApertureCenterLatchAsymmetryTests(unittest.TestCase):
    def test_eye_specific_top_latch_asymmetry_does_not_move_center(self):
        left = SyntheticCamera((-0.02, 0.0, 0.0))
        right = SyntheticCamera((0.02, 0.0, 0.0))
        center = np.array([0.0, 0.0, -0.16], dtype=np.float64)
        normal = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        horizontal = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        vertical = np.array([0.0, 1.0, 0.0], dtype=np.float64)

        left_mask = render_mask(left, center, horizontal, vertical)
        right_mask = render_mask(right, center, horizontal, vertical)

        # Reproduce the real stereo-mask failure: one eye loses part of the
        # upper latch silhouette while both full aperture side walls remain.
        x, y, width, height = cv2.boundingRect(
            (right_mask > 0).astype(np.uint8)
        )
        right_mask[
            y : y + max(1, int(0.35 * height)),
            x + int(0.65 * width) : x + int(0.80 * width),
        ] = 0

        result = estimate_planar_aperture_center(
            left_mask=left_mask,
            right_mask=right_mask,
            left_camera=left,
            right_camera=right,
            plane_origin_world_m=center,
            plane_normal_world=normal,
            aperture_width_m=0.0114,
            aperture_height_m=0.0070,
        )

        self.assertLess(
            float(np.linalg.norm(result.center_world_m - center)),
            0.00035,
        )
        self.assertLess(result.left_right_disagreement_m, 0.0005)


if __name__ == "__main__":
    unittest.main()
