"""Cache local crystal-head features and re-pose them from a live head transform."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

AAYUSH_DIR = Path(__file__).resolve().parents[2]
if str(AAYUSH_DIR) not in sys.path:
    sys.path.insert(0, str(AAYUSH_DIR))

from insertion_features.geometry import (
    ConnectorFeatures,
    Plane,
    extract_connector_features_from_mesh,
    transform_connector_features,
)
from insertion_features.cache import (
    connector_features_from_dict,
    load_local_cable_features,
    load_local_port_features,
    meters_transform_from_stage,
    save_local_cable_features,
    save_local_port_features,
    world_features_for_head,
    world_features_for_jack,
)

from insertion_features.tests.test_cable_features import _rj45_housing


def _rz(degrees: float, translation: np.ndarray) -> np.ndarray:
    angle = np.deg2rad(degrees)
    cosine, sine = np.cos(angle), np.sin(angle)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.array(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    transform[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    return transform


def _sample_features() -> ConnectorFeatures:
    points, counts, indices = _rj45_housing(tip_sign=1.0)
    return extract_connector_features_from_mesh(
        points=points,
        face_vertex_counts=counts,
        face_vertex_indices=indices,
        cable_center_world=np.array([-0.12, 0.0, 0.0]),
    )


class FeatureTransformTests(unittest.TestCase):
    def test_identity_transform_preserves_features(self) -> None:
        features = _sample_features()
        replayed = transform_connector_features(features, np.eye(4))
        np.testing.assert_allclose(replayed.mating_center, features.mating_center)
        np.testing.assert_allclose(replayed.insertion_axis, features.insertion_axis)
        np.testing.assert_allclose(replayed.latch_keypoints, features.latch_keypoints)

    def test_translated_and_rotated_head_moves_mating_center_and_axis(self) -> None:
        local = _sample_features()
        world_from_head = _rz(90.0, [0.4, -0.2, 0.1])
        world = transform_connector_features(local, world_from_head)

        expected_center = world_from_head[:3, :3] @ local.mating_center + world_from_head[:3, 3]
        expected_axis = world_from_head[:3, :3] @ local.insertion_axis
        expected_axis /= np.linalg.norm(expected_axis)
        np.testing.assert_allclose(world.mating_center, expected_center, atol=1e-9)
        np.testing.assert_allclose(world.insertion_axis, expected_axis, atol=1e-9)
        np.testing.assert_allclose(
            world.latch_keypoints[0],
            world_from_head[:3, :3] @ local.latch_keypoints[0] + world_from_head[:3, 3],
            atol=1e-9,
        )
        # 90° about Z sends local +X insertion toward +Y.
        self.assertGreater(world.insertion_axis[1], 0.9)

    def test_world_to_local_roundtrip(self) -> None:
        world = _sample_features()
        world_from_head = _rz(-35.0, [0.05, 0.02, -0.01])
        local = transform_connector_features(world, np.linalg.inv(world_from_head))
        restored = transform_connector_features(local, world_from_head)
        np.testing.assert_allclose(restored.mating_center, world.mating_center, atol=1e-9)
        np.testing.assert_allclose(restored.insertion_axis, world.insertion_axis, atol=1e-9)
        np.testing.assert_allclose(restored.latch_keypoints, world.latch_keypoints, atol=1e-9)


class FeatureCacheTests(unittest.TestCase):
    def test_json_roundtrip_exposes_head_local_variables(self) -> None:
        local = _sample_features()
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "cable_features_local.json"
            save_local_cable_features(
                {"E_crystal_head1_45": local},
                path,
                source_usd="model_Networkcable1_69323.usd",
            )
            heads = load_local_cable_features(path)
            self.assertIn("E_crystal_head1_45", heads)
            loaded = heads["E_crystal_head1_45"]
            np.testing.assert_allclose(loaded.mating_center, local.mating_center, atol=1e-9)
            np.testing.assert_allclose(loaded.latch_keypoints, local.latch_keypoints, atol=1e-9)

            raw = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(raw["frame"], "crystal_head_local")
            rebuilt = connector_features_from_dict(raw["heads"]["E_crystal_head1_45"])
            np.testing.assert_allclose(rebuilt.insertion_axis, local.insertion_axis, atol=1e-9)

    def test_world_features_for_head_uses_cached_local_geometry(self) -> None:
        local = _sample_features()
        world_from_head = _rz(180.0, [1.0, 2.0, 3.0])
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "cable_features_local.json"
            save_local_cable_features({"E_crystal_head2_39": local}, path)
            world = world_features_for_head(
                "E_crystal_head2_39",
                world_from_head,
                cache_path=path,
            )
        expected = transform_connector_features(local, world_from_head)
        np.testing.assert_allclose(world.mating_center, expected.mating_center)
        np.testing.assert_allclose(world.insertion_axis, expected.insertion_axis)
        # 180° about Z flips local +X insertion to -X.
        self.assertLess(world.insertion_axis[0], -0.9)

    def test_committed_cache_exposes_both_crystal_heads_as_local_variables(self) -> None:
        import importlib

        import insertion_features.cache as cache

        importlib.reload(cache)
        heads = load_local_cable_features(cache.DEFAULT_CACHE_PATH)
        self.assertEqual(
            set(heads),
            {"E_crystal_head1_45", "E_crystal_head2_39"},
        )
        plus_x = heads["E_crystal_head1_45"].insertion_axis
        minus_x = heads["E_crystal_head2_39"].insertion_axis
        self.assertGreater(plus_x[0], 0.9)
        self.assertLess(minus_x[0], -0.9)
        self.assertGreater(
            heads["E_crystal_head1_45"].latch_keypoints[:, 2].min(),
            heads["E_crystal_head1_45"].mating_center[2],
        )
        np.testing.assert_allclose(
            cache.E_crystal_head1_45.mating_center,
            heads["E_crystal_head1_45"].mating_center,
        )
        np.testing.assert_allclose(
            cache.E_crystal_head2_39.latch_keypoints,
            heads["E_crystal_head2_39"].latch_keypoints,
        )


class PortFeatureCacheTests(unittest.TestCase):
    def test_port_cache_roundtrip_is_pack_local_metres(self) -> None:
        local = _sample_features()
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "port_features_local.json"
            save_local_port_features(
                {"jack_upper_c0": local},
                path,
                source_usd="DataHall_6r_ur5e.usd",
                pack_prim_path="/World/RJ45_Group01",
                metadata={"jack_upper_c0": {"row": "upper", "col": 0, "copper_group": "Group_14341"}},
            )
            jacks = load_local_port_features(path)
            self.assertIn("jack_upper_c0", jacks)
            np.testing.assert_allclose(jacks["jack_upper_c0"].mating_center, local.mating_center)
            raw = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(raw["frame"], "rj45_group_local")
            self.assertEqual(raw["length_unit"], "meters")
            self.assertEqual(raw["jacks"]["jack_upper_c0"]["copper_group"], "Group_14341")

    def test_centimetre_pack_pose_is_scaled_before_applying_metre_cache(self) -> None:
        local = _sample_features()
        world_from_pack_cm = np.eye(4, dtype=np.float64)
        world_from_pack_cm[:3, 3] = [10.0, 20.0, 30.0]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "port_features_local.json"
            save_local_port_features({"jack_lower_c1": local}, path)
            world = world_features_for_jack(
                "jack_lower_c1",
                world_from_pack_cm,
                cache_path=path,
                meters_per_unit=0.01,
            )
        expected_center = local.mating_center + np.array([0.10, 0.20, 0.30])
        np.testing.assert_allclose(world.mating_center, expected_center, atol=1e-9)
    def test_committed_port_cache_has_twelve_pack_local_jacks(self) -> None:
        import importlib

        import insertion_features.cache as cache

        importlib.reload(cache)
        jacks = load_local_port_features(cache.DEFAULT_PORT_CACHE_PATH)
        self.assertEqual(len(jacks), 12)
        self.assertEqual(
            set(jacks),
            {f"jack_{row}_c{col}" for row in ("upper", "lower") for col in range(6)},
        )
        upper = jacks["jack_upper_c0"]
        lower = jacks["jack_lower_c0"]
        # Pack-local -Y maps to world +X on this DataHall instance.
        self.assertLess(upper.insertion_axis[1], -0.9)
        self.assertGreater(upper.up_axis[2], 0.9)
        self.assertLess(lower.up_axis[2], -0.9)
        np.testing.assert_allclose(
            cache.jack_upper_c0.mating_center,
            upper.mating_center,
        )


if __name__ == "__main__":
    unittest.main()
