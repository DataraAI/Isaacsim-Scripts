#!/usr/bin/env python3
"""Run the exact August 4 image pair when its fixture directory is supplied."""

from __future__ import annotations

import os
from pathlib import Path
import unittest

import cv2
import numpy as np

from vision.plane_rectified_front_lip import (
    estimate_plane_rectified_front_lip_center,
)


LEFT_WORLD_FROM_CAMERA = np.array(
    [
        [0.998710350188, 0.023015520754, 0.045253974756, 0.0],
        [0.010799136581, 0.774647641200, -0.632300886155, 0.0],
        [-0.049608618967, 0.631974143291, 0.773400069263, 0.0],
        [0.029440632334, 0.126198364667, 0.173421861340, 1.0],
    ],
    dtype=np.float64,
)
RIGHT_WORLD_FROM_CAMERA = np.array(
    [
        [0.995747599871, 0.044235612968, 0.080807969268, 0.0],
        [0.013301921138, 0.798939571861, -0.601264184373, 0.0],
        [-0.091157974121, 0.599782269712, 0.794953742487, 0.0],
        [0.064979950977, 0.125256809854, 0.187684311748, 1.0],
    ],
    dtype=np.float64,
)


class _FixtureCamera:
    image_width_px = 240
    image_height_px = 200

    def __init__(self, world_from_camera, crop_origin_xy):
        self.world_from_camera = np.asarray(world_from_camera, dtype=np.float64)
        self._crop_origin_xy = tuple(int(value) for value in crop_origin_xy)

    @property
    def fx_px(self) -> float:
        return 18.0 * 1280.0 / 20.955

    @property
    def fy_px(self) -> float:
        return 18.0 * 960.0 / (20.955 * 9.0 / 16.0)

    @property
    def cx_px(self) -> float:
        return (1280.0 - 1.0) / 2.0 - self._crop_origin_xy[0]

    @property
    def cy_px(self) -> float:
        return (960.0 - 1.0) / 2.0 - self._crop_origin_xy[1]

    @property
    def camera_from_world(self) -> np.ndarray:
        return np.linalg.inv(self.world_from_camera)

    @property
    def camera_center_world_m(self) -> np.ndarray:
        return self.world_from_camera[3, :3].copy()

    def camera_point_from_world(self, point_world_m) -> np.ndarray:
        point = np.append(np.asarray(point_world_m, dtype=np.float64).reshape(3), 1.0)
        local = point @ self.camera_from_world
        return local[:3] / local[3]

    def project_world(self, point_world_m) -> np.ndarray:
        point = self.camera_point_from_world(point_world_m)
        range_m = -float(point[2])
        if range_m <= 0.0:
            raise RuntimeError("Fixture point is behind the camera.")
        return np.array(
            [
                self.cx_px + self.fx_px * float(point[0]) / range_m,
                self.cy_px + self.fy_px * (-float(point[1])) / range_m,
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
        world = (np.append(direction, 0.0) @ self.world_from_camera)[:3]
        world /= np.linalg.norm(world)
        return self.camera_center_world_m, world


def _read_any(root: Path, names: tuple[str, ...]) -> np.ndarray:
    for name in names:
        path = root / name
        if not path.exists():
            continue
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise RuntimeError(f"Could not read fixture image: {path}")
        if image.ndim == 3:
            image = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2RGB)
        return image
    raise FileNotFoundError(f"Could not find any fixture file: {names}")


class PlaneRectifiedWorkstationFixtureTests(unittest.TestCase):
    def test_exact_august_4_pair_passes_without_correction(self):
        fixture_dir = os.environ.get("FRONT_LIP_FIXTURE_DIR")
        if not fixture_dir:
            self.skipTest("Set FRONT_LIP_FIXTURE_DIR to run the workstation regression.")
        root = Path(fixture_dir)
        left_rgb = _read_any(
            root, ("rgb_left_latest.png", "rgb_left_latest(7).png")
        )[290:490, 300:540]
        right_rgb = _read_any(
            root, ("rgb_right_latest.png", "rgb_right_latest(7).png")
        )[290:490, 70:310]
        left_mask = _read_any(
            root,
            ("port_detection_mask_left.png", "port_detection_mask_left(6).png"),
        )[290:490, 300:540]
        right_mask = _read_any(
            root,
            ("port_detection_mask_right.png", "port_detection_mask_right(6).png"),
        )[290:490, 70:310]

        result = estimate_plane_rectified_front_lip_center(
            left_rgb=left_rgb,
            right_rgb=right_rgb,
            left_mask=left_mask,
            right_mask=right_mask,
            left_camera=_FixtureCamera(LEFT_WORLD_FROM_CAMERA, (300, 290)),
            right_camera=_FixtureCamera(RIGHT_WORLD_FROM_CAMERA, (70, 290)),
            plane_origin_world_m=np.zeros(3),
            plane_normal_world=np.array([0.0, 0.0, 1.0]),
        )
        self.assertLessEqual(result.center_disagreement_m, 0.0005)
        self.assertLessEqual(result.left_fit.residual_px, 1.5)
        self.assertLessEqual(result.right_fit.residual_px, 1.5)
        print(
            "August 4 RGB front-lip regression: "
            f"disagreement={result.center_disagreement_m * 1000.0:.3f}mm "
            f"residuals={result.left_fit.residual_px:.3f}/"
            f"{result.right_fit.residual_px:.3f}px"
        )


if __name__ == "__main__":
    unittest.main()
