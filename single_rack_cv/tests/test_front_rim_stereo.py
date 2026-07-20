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


def make_rim(
    samples: np.ndarray,
    *,
    corners: np.ndarray,
    center_uv: tuple[float, float] | None = None,
) -> FrontRim2D:
    corners = np.asarray(corners, dtype=np.float64).reshape(4, 2)
    if center_uv is None:
        center = np.mean(corners, axis=0)
        center_uv = (float(center[0]), float(center[1]))
    return FrontRim2D(
        roi_uv=(0, 0, 640, 480),
        corners_uv=corners,
        center_uv=center_uv,
        side_lines=(empty_line(), empty_line(), empty_line(), empty_line()),
        side_samples_uv=samples,
    )


def rectangle_corners_world(
    *,
    center_x_m: float = 0.0,
    half_width_m: float = 0.0057,
    half_height_m: float = 0.0035,
) -> np.ndarray:
    return np.array(
        [
            [center_x_m - half_width_m, +half_height_m, -0.13],
            [center_x_m + half_width_m, +half_height_m, -0.13],
            [center_x_m + half_width_m, -half_height_m, -0.13],
            [center_x_m - half_width_m, -half_height_m, -0.13],
        ],
        dtype=np.float64,
    )


def rectangle_samples_world(
    *,
    center_x_m: float = 0.0,
    half_width_m: float = 0.0057,
    half_height_m: float = 0.0035,
) -> np.ndarray:
    corners = rectangle_corners_world(
        center_x_m=center_x_m,
        half_width_m=half_width_m,
        half_height_m=half_height_m,
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


def project_points(camera: SyntheticCamera, points: np.ndarray) -> np.ndarray:
    return np.stack([camera.project_world(point) for point in points])


class FrontRimStereoTests(unittest.TestCase):
    def test_recovers_planar_rectangle_center(self) -> None:
        left = SyntheticCamera((-0.02, 0.0, 0.0))
        right = SyntheticCamera((+0.02, 0.0, 0.0))
        world_corners = rectangle_corners_world()
        world_samples = rectangle_samples_world()
        left_samples = np.stack(
            [[left.project_world(point) for point in side] for side in world_samples]
        )
        right_samples = np.stack(
            [[right.project_world(point) for point in side] for side in world_samples]
        )

        result = triangulate_front_rims(
            left_rim=make_rim(
                left_samples,
                corners=project_points(left, world_corners),
            ),
            right_rim=make_rim(
                right_samples,
                corners=project_points(right, world_corners),
            ),
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

    def test_center_comes_from_opening_rays_not_shifted_bezel_ring(self) -> None:
        left = SyntheticCamera((-0.02, 0.0, 0.0))
        right = SyntheticCamera((+0.02, 0.0, 0.0))

        shifted_bezel = rectangle_samples_world(
            center_x_m=0.002,
            half_width_m=0.0065,
            half_height_m=0.0043,
        )
        opening_corners = rectangle_corners_world()
        opening_center = np.array([0.0, 0.0, -0.13])

        left_samples = np.stack(
            [[left.project_world(point) for point in side] for side in shifted_bezel]
        )
        right_samples = np.stack(
            [[right.project_world(point) for point in side] for side in shifted_bezel]
        )
        left_corners = project_points(left, opening_corners)
        right_corners = project_points(right, opening_corners)

        result = triangulate_front_rims(
            left_rim=make_rim(
                left_samples,
                corners=left_corners,
                center_uv=tuple(left.project_world(opening_center)),
            ),
            right_rim=make_rim(
                right_samples,
                corners=right_corners,
                center_uv=tuple(right.project_world(opening_center)),
            ),
            left_camera=left,
            right_camera=right,
            cfg=FrontRimConfig(),
        )

        np.testing.assert_allclose(
            result.center_world_m,
            opening_center,
            atol=1.0e-8,
        )
        self.assertAlmostEqual(result.width_m, 0.0114, places=7)
        self.assertAlmostEqual(result.height_m, 0.0070, places=7)

    def test_rejects_too_few_stereo_pairs(self) -> None:
        left = SyntheticCamera((-0.02, 0.0, 0.0))
        right = SyntheticCamera((+0.02, 0.0, 0.0))
        world_corners = rectangle_corners_world()
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
                left_rim=make_rim(
                    left_samples,
                    corners=project_points(left, world_corners),
                ),
                right_rim=make_rim(
                    right_samples,
                    corners=project_points(right, world_corners),
                ),
                left_camera=left,
                right_camera=right,
                cfg=FrontRimConfig(),
            )


if __name__ == "__main__":
    unittest.main()
