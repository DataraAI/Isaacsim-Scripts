from __future__ import annotations

import unittest

import numpy as np

from config import FrontRimConfig
from front_rim import FrontRim2D, RimLine2D
from front_rim_stereo import triangulate_front_rims


class SyntheticCamera:
    def __init__(self, position_xyz: tuple[float, float, float]) -> None:
        self.position = np.asarray(position_xyz, dtype=np.float64)
        self.fx_px = 550.0
        self.fy_px = 550.0
        self.cx_px = 319.5
        self.cy_px = 239.5

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


def empty_line() -> RimLine2D:
    support = np.zeros((12, 2), dtype=np.float64)
    return RimLine2D(
        point_uv=np.array([0.0, 0.0]),
        direction_uv=np.array([1.0, 0.0]),
        normal_uv=np.array([0.0, 1.0]),
        support_uv=support,
        inlier_uv=support,
    )


def make_rim(samples: np.ndarray) -> FrontRim2D:
    corners = np.array(
        [
            samples[0, 0],
            samples[0, -1],
            samples[2, -1],
            samples[2, 0],
        ],
        dtype=np.float64,
    )
    center = np.mean(corners, axis=0)
    return FrontRim2D(
        roi_uv=(0, 0, 640, 480),
        corners_uv=corners,
        center_uv=(float(center[0]), float(center[1])),
        side_lines=(empty_line(), empty_line(), empty_line(), empty_line()),
        side_samples_uv=samples,
    )


def rectangle_samples_world() -> np.ndarray:
    corners = np.array(
        [
            [-0.0057, +0.0035, -0.13],
            [+0.0057, +0.0035, -0.13],
            [+0.0057, -0.0035, -0.13],
            [-0.0057, -0.0035, -0.13],
        ],
        dtype=np.float64,
    )
    values = np.linspace(0.15, 0.85, 7)
    return np.stack(
        [
            corners[0] + values[:, None] * (corners[1] - corners[0]),
            corners[1] + values[:, None] * (corners[2] - corners[1]),
            corners[3] + values[:, None] * (corners[2] - corners[3]),
            corners[0] + values[:, None] * (corners[3] - corners[0]),
        ],
        axis=0,
    )


class FrontRimStereoTests(unittest.TestCase):
    def test_recovers_planar_rectangle_center(self) -> None:
        left = SyntheticCamera((-0.02, 0.0, 0.0))
        right = SyntheticCamera((+0.02, 0.0, 0.0))
        world_samples = rectangle_samples_world()
        left_samples = np.stack(
            [[left.project_world(point) for point in side] for side in world_samples]
        )
        right_samples = np.stack(
            [[right.project_world(point) for point in side] for side in world_samples]
        )

        result = triangulate_front_rims(
            left_rim=make_rim(left_samples),
            right_rim=make_rim(right_samples),
            left_camera=left,
            right_camera=right,
            cfg=FrontRimConfig(),
        )

        np.testing.assert_allclose(
            result.center_world_m,
            np.array([0.0, 0.0, -0.13]),
            atol=1.0e-8,
        )
        self.assertAlmostEqual(result.width_m, 0.0114, places=7)
        self.assertAlmostEqual(result.height_m, 0.0070, places=7)
        self.assertGreater(result.normal_world[2], 0.99)
        self.assertEqual(result.accepted_sample_count, 28)
        self.assertLess(result.max_ray_gap_m, 1.0e-9)
        self.assertLess(result.plane_residual_m, 1.0e-9)

    def test_rejects_too_few_stereo_pairs(self) -> None:
        left = SyntheticCamera((-0.02, 0.0, 0.0))
        right = SyntheticCamera((+0.02, 0.0, 0.0))
        world_samples = rectangle_samples_world()
        left_samples = np.stack(
            [[left.project_world(point) for point in side] for side in world_samples]
        )
        right_samples = np.stack(
            [[right.project_world(point) for point in side] for side in world_samples]
        )
        right_samples[:, :, 1] += 10.0

        with self.assertRaisesRegex(RuntimeError, "dense rim pairs"):
            triangulate_front_rims(
                left_rim=make_rim(left_samples),
                right_rim=make_rim(right_samples),
                left_camera=left,
                right_camera=right,
                cfg=FrontRimConfig(),
            )


if __name__ == "__main__":
    unittest.main()
