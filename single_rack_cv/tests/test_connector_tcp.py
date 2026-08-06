from __future__ import annotations

import unittest

import numpy as np

from connector_tcp import (
    MeshComponentBounds,
    connected_component_bounds,
    derive_insertion_tcp,
)


class ConnectorTcpTests(unittest.TestCase):
    def setUp(self):
        self.legacy_tip = np.array([18.0, 0.0, 0.0])
        self.scale = np.array([0.001, 0.001, 0.001])

    @staticmethod
    def component(name, minimum, maximum):
        return MeshComponentBounds(
            label=name,
            local_min=np.array(minimum, dtype=np.float64),
            local_max=np.array(maximum, dtype=np.float64),
            vertex_count=24,
        )

    def test_selects_nose_body_and_excludes_latch_from_vertical_center(self):
        result = derive_insertion_tcp(
            legacy_tip_local=self.legacy_tip,
            longitudinal_axis_index=0,
            nose_axis_local=np.array([1.0, 0.0, 0.0]),
            axis_scale_m_per_local_unit=self.scale,
            components=(
                self.component(
                    "body",
                    [13.0, -5.2, -4.5],
                    [18.0, 5.2, 2.5],
                ),
                self.component(
                    "latch",
                    [13.5, -2.0, 2.0],
                    [18.0, 2.0, 5.8],
                ),
                self.component(
                    "contacts",
                    [17.0, -4.5, -1.0],
                    [18.2, 4.5, 0.2],
                ),
            ),
            aperture_width_m=0.0114,
            aperture_height_m=0.0070,
        )
        np.testing.assert_allclose(result.tip_local, [18.0, 0.0, -1.0])
        np.testing.assert_allclose(
            result.shift_physical_m,
            [0.0, 0.0, -0.001],
        )
        self.assertEqual(result.selected_label, "body")
        np.testing.assert_allclose(
            result.cross_section_m,
            [0.0104, 0.0070],
        )
        self.assertAlmostEqual(result.nose_gap_m, 0.0)

    def test_selects_dimension_matching_rear_body_as_profile_donor(self):
        result = derive_insertion_tcp(
            legacy_tip_local=np.array([18.076, 0.0, 0.0]),
            longitudinal_axis_index=0,
            nose_axis_local=np.array([1.0, 0.0, 0.0]),
            axis_scale_m_per_local_unit=self.scale,
            components=(
                self.component(
                    "front_with_latch",
                    [-2.414, -5.246, -5.910],
                    [18.076, 5.246, 5.910],
                ),
                self.component(
                    "thin_insert",
                    [5.406, -3.704, -0.450],
                    [16.957, 3.704, 0.450],
                ),
                self.component(
                    "rear_body",
                    [-18.076, -5.2475, -4.135],
                    [5.406, 5.2475, 3.000],
                ),
            ),
            aperture_width_m=0.0114,
            aperture_height_m=0.0070,
        )
        self.assertEqual(result.selected_label, "rear_body")
        self.assertEqual(result.tip_local[0], 18.076)
        np.testing.assert_allclose(result.tip_local[1:], [0.0, -0.5675])
        np.testing.assert_allclose(
            result.shift_physical_m,
            [0.0, 0.0, -0.0005675],
        )
        np.testing.assert_allclose(
            result.cross_section_m,
            [0.010495, 0.007135],
        )
        self.assertAlmostEqual(result.nose_gap_m, 0.012670)

    def test_rejects_dimension_matching_profile_beyond_setback_limit(self):
        with self.assertRaisesRegex(RuntimeError, "setback"):
            derive_insertion_tcp(
                legacy_tip_local=self.legacy_tip,
                longitudinal_axis_index=0,
                nose_axis_local=np.array([1.0, 0.0, 0.0]),
                axis_scale_m_per_local_unit=self.scale,
                components=(
                    self.component(
                        "far_body",
                        [-20.0, -5.2, -4.0],
                        [-3.0, 5.2, 3.0],
                    ),
                ),
                aperture_width_m=0.0114,
                aperture_height_m=0.0070,
                maximum_profile_setback_m=0.010,
            )

    def test_rejects_two_equally_plausible_components_with_different_centers(self):
        with self.assertRaisesRegex(RuntimeError, "ambiguous"):
            derive_insertion_tcp(
                legacy_tip_local=self.legacy_tip,
                longitudinal_axis_index=0,
                nose_axis_local=np.array([1.0, 0.0, 0.0]),
                axis_scale_m_per_local_unit=self.scale,
                components=(
                    self.component(
                        "body_a",
                        [13.0, -5.2, -4.5],
                        [18.0, 5.2, 2.5],
                    ),
                    self.component(
                        "body_b",
                        [13.0, -5.2, -3.5],
                        [18.0, 5.2, 3.5],
                    ),
                ),
                aperture_width_m=0.0114,
                aperture_height_m=0.0070,
            )

    def test_rejects_full_bounds_that_include_latch(self):
        with self.assertRaisesRegex(RuntimeError, "profile"):
            derive_insertion_tcp(
                legacy_tip_local=self.legacy_tip,
                longitudinal_axis_index=0,
                nose_axis_local=np.array([1.0, 0.0, 0.0]),
                axis_scale_m_per_local_unit=self.scale,
                components=(
                    self.component(
                        "combined",
                        [13.0, -5.2, -5.9],
                        [18.0, 5.2, 5.9],
                    ),
                ),
                aperture_width_m=0.0114,
                aperture_height_m=0.0070,
            )

    def test_only_transverse_coordinates_change(self):
        result = derive_insertion_tcp(
            legacy_tip_local=np.array([-18.0, 0.5, 0.5]),
            longitudinal_axis_index=0,
            nose_axis_local=np.array([-1.0, 0.0, 0.0]),
            axis_scale_m_per_local_unit=self.scale,
            components=(
                self.component(
                    "body",
                    [-18.0, -5.0, -4.0],
                    [-13.0, 5.0, 3.0],
                ),
            ),
            aperture_width_m=0.0114,
            aperture_height_m=0.0070,
        )
        self.assertEqual(result.tip_local[0], -18.0)
        self.assertEqual(result.shift_physical_m[0], 0.0)

    def test_splits_disconnected_mesh_topology_into_component_bounds(self):
        points = np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
                [10, 0, 0],
                [11, 0, 0],
                [10, 1, 0],
                [10, 0, 1],
            ],
            dtype=np.float64,
        )
        components = connected_component_bounds(
            points=points,
            face_vertex_counts=np.array([3, 3, 3, 3]),
            face_vertex_indices=np.array(
                [0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7]
            ),
            label_prefix="mesh",
        )
        self.assertEqual(len(components), 2)
        np.testing.assert_allclose(components[0].local_min, [0, 0, 0])
        np.testing.assert_allclose(components[1].local_min, [10, 0, 0])


if __name__ == "__main__":
    unittest.main()
