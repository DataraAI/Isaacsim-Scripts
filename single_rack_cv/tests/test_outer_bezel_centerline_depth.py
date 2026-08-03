from __future__ import annotations

import inspect
import unittest
from unittest.mock import patch

import numpy as np

from outer_bezel_center import OUTER_BEZEL_CONFIG, OuterBezelPlaneResult
from outer_bezel_projective_center import estimate_outer_bezel_projective_center
from stereo_center import StereoApertureCenter


class DummyDisparity:
    valid_count = 300
    consistent_count = 200


class DummyCamera:
    image_height_px = 120
    image_width_px = 160

    def project_world(self, point):
        point = np.asarray(point, dtype=np.float64)
        return np.array(
            [80.0 + point[1] * 100.0, 60.0 - point[2] * 100.0]
        )


class OuterBezelCenterlineDepthTests(unittest.TestCase):
    @staticmethod
    def _plane() -> OuterBezelPlaneResult:
        return OuterBezelPlaneResult(
            center_world_m=np.array([0.65, -0.19, 1.32]),
            normal_world=np.array([-0.9998, 0.0200, 0.0]),
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
            side_support_counts=(28, 0, 0, 0),
            support_region_count=2,
            spatial_region_counts=(12, 12, 12, 12),
            support_span_u_px=24.0,
            support_span_v_px=8.0,
            support_minor_std_px=2.5,
            median_disparity_px=220.0,
            disparity=DummyDisparity(),
        )

    def test_stereo_centerline_sets_lateral_position_and_plane_sets_depth(self):
        plane = self._plane()
        left_uv = np.array([70.0, 45.0])
        right_uv = np.array([50.0, 45.0])
        direct_stereo = StereoApertureCenter(
            center_world_m=np.array([0.658, -0.1926, 1.3230]),
            left_center_uv=left_uv,
            right_center_uv=right_uv,
            ray_gap_m=0.00012,
            reprojection_rms_px=0.08,
            max_reprojection_px=0.11,
        )
        expected_front_center = np.array([0.6501, -0.19255, 1.32302])
        camera = DummyCamera()

        with patch(
            "outer_bezel_projective_center.estimate_outer_bezel_plane",
            return_value=plane,
        ), patch(
            "outer_bezel_projective_center.estimate_projective_stereo_center",
            return_value=direct_stereo,
            create=True,
        ) as direct_estimator, patch(
            "outer_bezel_projective_center.intersect_midpoint_ray_with_plane",
            return_value=expected_front_center,
            create=True,
        ) as centerline_intersection, patch(
            "outer_bezel_projective_center.projective_center_pixel",
            side_effect=(left_uv, right_uv),
        ), patch(
            "outer_bezel_projective_center.intersect_pixel_with_plane",
            side_effect=(
                np.array([0.65, -0.1937, 1.3230]),
                np.array([0.65, -0.1913, 1.3230]),
            ),
        ):
            result = estimate_outer_bezel_projective_center(
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
            )

        direct_estimator.assert_called_once()
        centerline_intersection.assert_called_once()
        np.testing.assert_allclose(result.center_world_m, expected_front_center)
        np.testing.assert_allclose(
            result.left_center_world_m,
            expected_front_center,
        )
        np.testing.assert_allclose(
            result.right_center_world_m,
            expected_front_center,
        )
        self.assertAlmostEqual(result.eye_disagreement_m, 0.00012)
        self.assertIs(result.front_plane_config, OUTER_BEZEL_CONFIG)

    def test_runtime_does_not_average_independent_plane_intersections(self):
        source = inspect.getsource(estimate_outer_bezel_projective_center)
        self.assertIn("estimate_projective_stereo_center", source)
        self.assertIn("intersect_midpoint_ray_with_plane", source)
        self.assertNotIn("intersect_pixel_with_plane", source)
        self.assertNotIn("0.5 * (left_point + right_point)", source)


if __name__ == "__main__":
    unittest.main()
