from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
import inspect
import unittest

import numpy as np

from live_control import (
    apply_front_plane_result,
    apply_stereo_center_result,
    refine_live_observation,
)


class FakeCamera:
    image_height_px = 960
    image_width_px = 1280
    fx_px = 1000.0
    fy_px = 1000.0
    cx_px = 639.5
    cy_px = 479.5
    world_from_camera = np.eye(4, dtype=np.float64)

    def camera_point_from_world(self, point_world_m):
        return np.asarray(point_world_m, dtype=np.float64).reshape(3)

    def project_world(self, point_world_m):
        x, y, z = np.asarray(point_world_m, dtype=np.float64).reshape(3)
        depth = -float(z)
        return np.array(
            [
                self.cx_px + self.fx_px * x / depth,
                self.cy_px + self.fy_px * (-y) / depth,
            ]
        )


@dataclass(frozen=True)
class Observation:
    center_world_xyz_m: np.ndarray
    center_virtual_camera_usd_m: np.ndarray
    normal_world: np.ndarray
    corners_world_m: np.ndarray
    projected_virtual_center_uv: tuple[float, float]
    center_error_px: np.ndarray
    estimated_range_m: float
    range_error_m: float
    correction_world_m: np.ndarray
    width_m: float
    height_m: float
    mean_disparity_px: float
    reprojection_rms_px: float
    max_reprojection_px: float
    max_ray_gap_m: float
    plane_residual_m: float


class LiveControlTests(unittest.TestCase):
    @staticmethod
    def _cavity() -> Observation:
        return Observation(
            center_world_xyz_m=np.array([0.0, 0.0, -0.14]),
            center_virtual_camera_usd_m=np.array([0.0, 0.0, -0.14]),
            normal_world=np.array([0.0, 0.0, 1.0]),
            corners_world_m=np.zeros((4, 3)),
            projected_virtual_center_uv=(639.5, 479.5),
            center_error_px=np.zeros(2),
            estimated_range_m=0.14,
            range_error_m=0.01,
            correction_world_m=np.array([0.0, 0.0, -0.01]),
            width_m=0.0114,
            height_m=0.007,
            mean_disparity_px=300.0,
            reprojection_rms_px=0.0,
            max_reprojection_px=0.0,
            max_ray_gap_m=0.0,
            plane_residual_m=0.0,
        )

    def test_front_plane_helper_remains_available_offline(self):
        camera = FakeCamera()
        frame = SimpleNamespace(virtual_camera=camera)
        cavity = self._cavity()
        opening = SimpleNamespace(
            center_world_m=np.array([0.0, 0.0, -0.13]),
            normal_world=np.array([0.0, 0.0, 1.0]),
            corners_world_m=np.array(
                [
                    [-0.0057, 0.0035, -0.13],
                    [0.0057, 0.0035, -0.13],
                    [0.0057, -0.0035, -0.13],
                    [-0.0057, -0.0035, -0.13],
                ]
            ),
            width_m=0.0114,
            height_m=0.007,
            median_disparity_px=320.0,
            reprojection_rms_px=0.1,
            max_reprojection_px=0.2,
            max_ray_gap_m=0.0002,
            plane_residual_m=0.0003,
            valid_disparity_count=1000,
            consistent_disparity_count=800,
            ring_candidate_count=200,
            triangulated_count=180,
            cluster_count=150,
            side_support_counts=(30, 40, 35, 45),
        )

        refined, diagnostics = apply_front_plane_result(
            frame=frame,
            observation=cavity,
            desired_port_virtual_camera_usd=np.array([0.0, 0.0, -0.13]),
            front_plane_result=opening,
        )

        self.assertAlmostEqual(refined.estimated_range_m, 0.13, places=12)
        self.assertTrue(np.allclose(refined.correction_world_m, np.zeros(3)))
        self.assertTrue(
            np.allclose(refined.center_world_xyz_m, opening.center_world_m)
        )
        self.assertAlmostEqual(diagnostics.cavity_range_m, 0.14, places=12)
        self.assertAlmostEqual(diagnostics.opening_range_m, 0.13, places=12)

    def test_direct_stereo_center_replaces_control_center(self):
        camera = FakeCamera()
        frame = SimpleNamespace(virtual_camera=camera)
        stereo_center = SimpleNamespace(
            center_world_m=np.array([0.0, 0.0, -0.13]),
            left_center_uv=np.array([700.0, 480.0]),
            right_center_uv=np.array([400.0, 480.0]),
            ray_gap_m=0.0002,
            reprojection_rms_px=0.2,
            max_reprojection_px=0.3,
        )

        refined, diagnostics = apply_stereo_center_result(
            frame=frame,
            observation=self._cavity(),
            desired_port_virtual_camera_usd=np.array([0.0, 0.0, -0.13]),
            stereo_center_result=stereo_center,
        )

        np.testing.assert_allclose(
            refined.center_world_xyz_m,
            stereo_center.center_world_m,
        )
        self.assertAlmostEqual(refined.estimated_range_m, 0.13, places=12)
        self.assertAlmostEqual(refined.max_ray_gap_m, 0.0002, places=12)
        self.assertAlmostEqual(
            diagnostics.aperture_center_disagreement_m,
            0.0002,
            places=12,
        )
        self.assertEqual(diagnostics.side_support_counts, (0, 0, 0, 0))

    def test_live_refinement_uses_direct_stereo_center_not_bezel_plane(self):
        source = inspect.getsource(refine_live_observation)
        self.assertIn("estimate_stereo_aperture_center", source)
        self.assertIn("left_mask=observation.left.detection.mask", source)
        self.assertIn("right_mask=observation.right.detection.mask", source)
        self.assertIn("apply_stereo_center_result", source)
        self.assertNotIn("estimate_front_plane", source)
        self.assertNotIn("estimate_planar_aperture_center", source)

    def test_public_apis_have_no_manual_offset_parameter(self):
        for function in (apply_front_plane_result, apply_stereo_center_result):
            parameters = inspect.signature(function).parameters
            self.assertFalse(
                any("offset" in name.lower() for name in parameters)
            )


if __name__ == "__main__":
    unittest.main()
