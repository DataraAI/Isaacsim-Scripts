from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np

from automatic_port_ground_truth import (
    RaycastGroundTruthConfig,
    RaycastHit,
    build_automatic_ground_truth,
    intersect_ray_with_plane,
    offset_rim_samples_outward,
)
from tools.extract_front_rim_ground_truth import _bbox_side_samples


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class AutomaticGroundTruthTests(unittest.TestCase):
    def test_validator_uses_rtx_mesh_backend_not_physx(self) -> None:
        source = (
            PROJECT_ROOT
            / "tools"
            / "extract_front_rim_ground_truth.py"
        ).read_text(encoding="utf-8")
        self.assertIn("omni.kit.raycast.query", source)
        self.assertNotIn(
            "from omni.physx import get_physx_scene_query_interface",
            source,
        )

    def test_highres_metadata_is_written_before_simulation_shutdown(self) -> None:
        source = (
            PROJECT_ROOT
            / "tools"
            / "generate_ground_truth.py"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "original_write_result = implementation._write_result",
            source,
        )
        self.assertIn(
            "implementation._write_result = write_result_with_resolution",
            source,
        )
        self.assertIn("_stamp_resolution_metadata()", source)
        self.assertNotIn("highres_config", source)

    def test_bbox_side_samples_follow_refined_cavity_box(self) -> None:
        samples = _bbox_side_samples(
            (10, 20, 21, 11),
            samples_per_side=3,
            corner_trim_fraction=0.0,
        )
        self.assertEqual(samples.shape, (4, 3, 2))
        np.testing.assert_allclose(
            samples[0],
            np.array([[10.0, 20.0], [20.0, 20.0], [30.0, 20.0]]),
        )
        np.testing.assert_allclose(
            samples[1],
            np.array([[30.0, 20.0], [30.0, 25.0], [30.0, 30.0]]),
        )
        np.testing.assert_allclose(
            samples[2],
            np.array([[10.0, 30.0], [20.0, 30.0], [30.0, 30.0]]),
        )
        np.testing.assert_allclose(
            samples[3],
            np.array([[10.0, 20.0], [10.0, 25.0], [10.0, 30.0]]),
        )

    def test_offsets_each_side_away_from_center(self) -> None:
        samples = np.array(
            [
                [[-1.0, 1.0], [0.0, 1.0], [1.0, 1.0]],
                [[1.0, 1.0], [1.0, 0.0], [1.0, -1.0]],
                [[-1.0, -1.0], [0.0, -1.0], [1.0, -1.0]],
                [[-1.0, 1.0], [-1.0, 0.0], [-1.0, -1.0]],
            ],
            dtype=np.float64,
        )
        shifted = offset_rim_samples_outward(
            side_samples_uv=samples,
            center_uv=(0.0, 0.0),
            offset_px=2.0,
        )
        np.testing.assert_allclose(shifted[0, :, 1], 3.0)
        np.testing.assert_allclose(shifted[1, :, 0], 3.0)
        np.testing.assert_allclose(shifted[2, :, 1], -3.0)
        np.testing.assert_allclose(shifted[3, :, 0], -3.0)

    def test_fits_front_plane_and_rejects_recessed_outliers(self) -> None:
        camera = np.array([0.0, 0.0, 0.0])
        hits: list[RaycastHit] = []
        for x in np.linspace(-0.01, 0.01, 6):
            for y in (-0.006, 0.006):
                hits.append(
                    RaycastHit(
                        position_world_m=np.array([x, y, -0.10]),
                        normal_world=np.array([0.0, 0.0, 1.0]),
                        prim_path="/World/ServerRack/Bezel",
                        distance_m=float(np.linalg.norm([x, y, -0.10])),
                    )
                )
        for x in (-0.005, 0.0, 0.005):
            hits.append(
                RaycastHit(
                    position_world_m=np.array([x, 0.0, -0.13]),
                    normal_world=np.array([0.0, 0.0, 1.0]),
                    prim_path="/World/ServerRack/Cavity",
                    distance_m=float(np.linalg.norm([x, 0.0, -0.13])),
                )
            )

        result = build_automatic_ground_truth(
            hits=hits,
            camera_center_world_m=camera,
            center_ray_direction_world=np.array([0.0, 0.0, -1.0]),
            cfg=RaycastGroundTruthConfig(
                rack_path_prefix="/World/ServerRack",
                min_plane_hits=8,
                depth_cluster_tolerance_m=0.004,
                plane_max_residual_m=0.0005,
            ),
        )

        np.testing.assert_allclose(
            result.center_world_m,
            np.array([0.0, 0.0, -0.10]),
            atol=1.0e-8,
        )
        self.assertGreater(result.normal_world[2], 0.99)
        self.assertEqual(result.used_hit_count, 12)
        self.assertLess(result.plane_residual_m, 1.0e-9)

    def test_intersects_forward_ray_with_plane(self) -> None:
        point = intersect_ray_with_plane(
            ray_origin_world_m=np.zeros(3),
            ray_direction_world=np.array([0.0, 0.0, -1.0]),
            plane_center_world_m=np.array([0.0, 0.0, -0.2]),
            plane_normal_world=np.array([0.0, 0.0, 1.0]),
        )
        np.testing.assert_allclose(point, np.array([0.0, 0.0, -0.2]))

    def test_rejects_too_few_valid_hits(self) -> None:
        hits = [
            RaycastHit(
                position_world_m=np.array([0.0, 0.0, -0.1]),
                normal_world=np.array([0.0, 0.0, 1.0]),
                prim_path="/World/ServerRack/Bezel",
                distance_m=0.1,
            )
        ]
        with self.assertRaisesRegex(RuntimeError, "valid rack ray hits"):
            build_automatic_ground_truth(
                hits=hits,
                camera_center_world_m=np.zeros(3),
                center_ray_direction_world=np.array([0.0, 0.0, -1.0]),
                cfg=RaycastGroundTruthConfig(min_plane_hits=4),
            )


if __name__ == "__main__":
    unittest.main()
