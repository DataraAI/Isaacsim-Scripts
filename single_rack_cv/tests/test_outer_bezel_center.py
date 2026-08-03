from __future__ import annotations

import inspect
import unittest
from unittest.mock import patch

import numpy as np

from aperture_center import PlanarApertureCenter
from outer_bezel_center import (
    OUTER_BEZEL_CONFIG,
    OuterBezelPlaneResult,
    estimate_outer_bezel_aperture_center,
    select_nearest_supported_range_cluster,
)


class DummyDisparity:
    valid_count = 300
    consistent_count = 200


class DummyCamera:
    image_height_px = 120
    image_width_px = 160

    def project_world(self, point):
        point = np.asarray(point, dtype=np.float64)
        return np.array([80.0 + point[1] * 100.0, 60.0 - point[2] * 100.0])


class OuterBezelSupportTests(unittest.TestCase):
    def test_nearer_supported_plane_beats_larger_recessed_cluster(self):
        near_uv = np.array(
            [[20.0 + x, 20.0] for x in range(0, 45, 4)]
            + [[20.0, 20.0 + y] for y in range(4, 49, 4)],
            dtype=np.float64,
        )
        near_ranges = np.linspace(0.1800, 0.1804, near_uv.shape[0])
        near_labels = np.array([0] * 12 + [3] * 12, dtype=np.int64)

        far_x, far_y = np.meshgrid(
            np.arange(40.0, 80.0, 4.0),
            np.arange(40.0, 72.0, 4.0),
        )
        far_uv = np.column_stack((far_x.reshape(-1), far_y.reshape(-1)))
        far_ranges = np.linspace(0.1900, 0.1908, far_uv.shape[0])
        far_labels = np.full(far_uv.shape[0], 1, dtype=np.int64)

        selected, diagnostics = select_nearest_supported_range_cluster(
            ranges_m=np.concatenate((near_ranges, far_ranges)),
            pixels_uv=np.vstack((near_uv, far_uv)),
            side_labels=np.concatenate((near_labels, far_labels)),
            tolerance_m=0.0010,
            min_points=10,
            min_supported_regions=2,
            min_span_u_px=16.0,
            min_span_v_px=16.0,
            min_minor_std_px=3.0,
            min_points_per_spatial_region=4,
        )

        self.assertEqual(int(np.count_nonzero(selected)), near_uv.shape[0])
        self.assertGreaterEqual(diagnostics.region_count, 2)
        self.assertGreaterEqual(diagnostics.span_u_px, 16.0)
        self.assertGreaterEqual(diagnostics.span_v_px, 16.0)
        self.assertGreaterEqual(diagnostics.minor_std_px, 3.0)

    def test_single_narrow_edge_is_rejected(self):
        pixels = np.column_stack(
            (np.full(24, 30.0), np.linspace(10.0, 70.0, 24))
        )
        ranges = np.linspace(0.1800, 0.1803, 24)
        labels = np.zeros(24, dtype=np.int64)

        with self.assertRaisesRegex(RuntimeError, "qualified outer-bezel"):
            select_nearest_supported_range_cluster(
                ranges_m=ranges,
                pixels_uv=pixels,
                side_labels=labels,
                tolerance_m=0.0010,
                min_points=12,
                min_supported_regions=2,
                min_span_u_px=12.0,
                min_span_v_px=12.0,
                min_minor_std_px=3.0,
            )

    def test_one_stray_pixel_does_not_fake_a_second_region(self):
        pixels = np.column_stack(
            (
                np.linspace(10.0, 90.0, 40),
                np.full(40, 20.0),
            )
        )
        labels = np.zeros(pixels.shape[0], dtype=np.int64)
        pixels = np.vstack((pixels, np.array([[5.0, 55.0]])))
        labels = np.concatenate((labels, np.array([3], dtype=np.int64)))
        ranges = np.linspace(0.1800, 0.1804, pixels.shape[0])

        with self.assertRaisesRegex(RuntimeError, "qualified outer-bezel"):
            select_nearest_supported_range_cluster(
                ranges_m=ranges,
                pixels_uv=pixels,
                side_labels=labels,
                tolerance_m=0.0010,
                min_points=20,
                min_supported_regions=2,
                min_span_u_px=12.0,
                min_span_v_px=8.0,
                min_minor_std_px=2.0,
                min_points_per_spatial_region=4,
            )

    def test_broad_single_side_patch_is_valid_two_dimensional_support(self):
        grid_u, grid_v = np.meshgrid(
            np.arange(10.0, 110.0, 4.0),
            np.arange(20.0, 32.0, 2.0),
        )
        pixels = np.column_stack((grid_u.reshape(-1), grid_v.reshape(-1)))
        labels = np.zeros(pixels.shape[0], dtype=np.int64)
        ranges = np.linspace(0.1800, 0.1804, pixels.shape[0])

        selected, diagnostics = select_nearest_supported_range_cluster(
            ranges_m=ranges,
            pixels_uv=pixels,
            side_labels=labels,
            tolerance_m=0.0010,
            min_points=20,
            min_supported_regions=2,
            min_span_u_px=12.0,
            min_span_v_px=8.0,
            min_minor_std_px=2.0,
            min_points_per_spatial_region=8,
        )

        self.assertEqual(int(np.count_nonzero(selected)), pixels.shape[0])
        self.assertGreaterEqual(diagnostics.region_count, 2)
        self.assertEqual(diagnostics.side_counts, (pixels.shape[0], 0, 0, 0))


class OuterBezelCenterTests(unittest.TestCase):
    def test_uses_outer_plane_and_plane_rectified_center(self):
        plane = OuterBezelPlaneResult(
            center_world_m=np.array([0.65, -0.19, 1.32]),
            normal_world=np.array([-1.0, 0.0, 0.0]),
            corners_world_m=np.array(
                [
                    [0.65, -0.20, 1.31],
                    [0.65, -0.18, 1.31],
                    [0.65, -0.18, 1.33],
                    [0.65, -0.20, 1.33],
                ]
            ),
            width_m=0.02,
            height_m=0.02,
            max_ray_gap_m=0.0002,
            reprojection_rms_px=0.3,
            max_reprojection_px=0.5,
            plane_residual_m=0.0002,
            valid_disparity_count=300,
            consistent_disparity_count=200,
            ring_candidate_count=80,
            triangulated_count=64,
            cluster_count=28,
            side_support_counts=(12, 0, 8, 8),
            support_region_count=3,
            spatial_region_counts=(12, 12, 12, 12),
            support_span_u_px=24.0,
            support_span_v_px=20.0,
            support_minor_std_px=4.0,
            median_disparity_px=220.0,
            disparity=DummyDisparity(),
        )
        center = PlanarApertureCenter(
            center_world_m=np.array([0.65, -0.192, 1.323]),
            left_center_world_m=np.array([0.65, -0.1921, 1.323]),
            right_center_world_m=np.array([0.65, -0.1919, 1.323]),
            left_right_disagreement_m=0.0002,
        )
        camera = DummyCamera()

        with patch(
            "outer_bezel_center.estimate_outer_bezel_plane",
            return_value=plane,
        ), patch(
            "outer_bezel_center.estimate_planar_aperture_center",
            return_value=center,
        ):
            result = estimate_outer_bezel_aperture_center(
                left_rgb=np.zeros((120, 160, 3), dtype=np.uint8),
                right_rgb=np.zeros((120, 160, 3), dtype=np.uint8),
                left_mask=np.zeros((120, 160), dtype=np.uint8),
                right_mask=np.zeros((120, 160), dtype=np.uint8),
                left_bbox_xywh=(40, 30, 40, 30),
                right_bbox_xywh=(20, 30, 40, 30),
                left_detection_center_uv=(60.0, 45.0),
                right_detection_center_uv=(40.0, 45.0),
                left_camera=camera,
                right_camera=camera,
                aperture_width_m=0.0114,
                aperture_height_m=0.0070,
            )

        np.testing.assert_allclose(result.center_world_m, center.center_world_m)
        np.testing.assert_allclose(result.plane_origin_world_m, plane.center_world_m)
        self.assertEqual(result.support_region_count, 3)
        self.assertAlmostEqual(result.eye_disagreement_m, 0.0002)
        self.assertIs(result.front_plane_config, OUTER_BEZEL_CONFIG)

    def test_public_api_has_no_manual_depth_offset(self):
        parameters = inspect.signature(
            estimate_outer_bezel_aperture_center
        ).parameters
        forbidden = {
            "offset",
            "depth_offset",
            "world_offset",
            "bias",
            "port_prim",
            "rack_transform",
        }
        self.assertTrue(forbidden.isdisjoint(parameters))


if __name__ == "__main__":
    unittest.main()
