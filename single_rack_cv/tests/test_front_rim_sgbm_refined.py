from __future__ import annotations

import unittest

import numpy as np

from front_rim_sgbm import LocalSGBMConfig
from front_rim_sgbm_refined import (
    _fit_plane_stable,
    _intersect_midpoint_ray_with_plane,
)


class SyntheticCamera:
    def __init__(
        self,
        position_xyz: tuple[float, float, float],
        focal_px: float = 500.0,
    ) -> None:
        self.position = np.asarray(position_xyz, dtype=np.float64)
        self.focal_px = float(focal_px)
        self.cx_px = 320.0
        self.cy_px = 240.0

    def project_world(self, point_world_m: np.ndarray) -> np.ndarray:
        local = np.asarray(point_world_m, dtype=np.float64) - self.position
        depth = -float(local[2])
        if depth <= 0.0:
            raise RuntimeError("Point is behind camera.")
        return np.array(
            [
                self.cx_px + self.focal_px * float(local[0]) / depth,
                self.cy_px - self.focal_px * float(local[1]) / depth,
            ],
            dtype=np.float64,
        )

    def pixel_to_world_ray(
        self,
        pixel_uv: tuple[float, float] | np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        u, v = np.asarray(pixel_uv, dtype=np.float64).reshape(2)
        direction = np.array(
            [
                (u - self.cx_px) / self.focal_px,
                -(v - self.cy_px) / self.focal_px,
                -1.0,
            ],
            dtype=np.float64,
        )
        direction /= np.linalg.norm(direction)
        return self.position.copy(), direction


def intersect_ray_with_z_plane(
    camera: SyntheticCamera,
    pixel_uv: np.ndarray,
    plane_z: float,
) -> np.ndarray:
    origin, direction = camera.pixel_to_world_ray(pixel_uv)
    distance = (plane_z - float(origin[2])) / float(direction[2])
    return origin + distance * direction


class StablePlaneFitTests(unittest.TestCase):
    def test_final_refit_still_honors_hard_residual_gate(self) -> None:
        rng = np.random.default_rng(1)
        main_xy = rng.uniform(-0.02, 0.02, (75, 2))
        raised_xy = np.column_stack(
            (
                rng.uniform(0.0, 0.02, 25),
                rng.uniform(-0.02, 0.02, 25),
            )
        )
        points = np.column_stack(
            (
                np.vstack((main_xy, raised_xy)),
                np.concatenate(
                    (
                        np.zeros(75, dtype=np.float64),
                        np.full(25, 0.0007, dtype=np.float64),
                    )
                ),
            )
        )
        cfg = LocalSGBMConfig()
        _, _, inliers, residual = _fit_plane_stable(points, cfg)

        self.assertGreaterEqual(int(np.count_nonzero(inliers)), 24)
        self.assertLessEqual(residual, cfg.plane_max_residual_m)


class FusedCenterTests(unittest.TestCase):
    def test_recessed_cavity_uses_one_midpoint_ray_on_front_plane(self) -> None:
        left = SyntheticCamera((-0.02, 0.0, 0.0))
        right = SyntheticCamera((0.02, 0.0, 0.0))
        recessed_cavity = np.array([0.003, -0.002, -0.14])
        left_uv = left.project_world(recessed_cavity)
        right_uv = right.project_world(recessed_cavity)
        plane_center = np.array([0.0, 0.0, -0.13])
        plane_normal = np.array([0.0, 0.0, 1.0])

        left_hit = intersect_ray_with_z_plane(left, left_uv, -0.13)
        right_hit = intersect_ray_with_z_plane(right, right_uv, -0.13)
        self.assertGreater(float(np.linalg.norm(left_hit - right_hit)), 0.002)

        fused = _intersect_midpoint_ray_with_plane(
            left,
            right,
            left_uv,
            right_uv,
            plane_center,
            plane_normal,
        )
        self.assertAlmostEqual(float(fused[2]), -0.13, places=12)
        self.assertTrue(np.all(np.isfinite(fused)))
        self.assertGreater(float(fused[0]), float(left_hit[0]))
        self.assertLess(float(fused[0]), float(right_hit[0]))


if __name__ == "__main__":
    unittest.main()
