#!/usr/bin/env python3

from __future__ import annotations

import inspect
import unittest

import numpy as np

from front_mouth_projective_center import (
    _outer_front_side_lines,
    _outermost_signed_edge_index,
    aperture_center_pixel,
)


class _Camera:
    image_width_px = 100
    image_height_px = 80


class FrontMouthOuterEdgeTests(unittest.TestCase):
    def test_selects_outermost_qualified_edge_not_strongest_inner_edge(self):
        negative_gradient = np.array(
            [0.0, -72.0, -18.0, -155.0, 0.0],
            dtype=np.float64,
        )
        positive_gradient = np.array(
            [0.0, 155.0, 18.0, 72.0, 0.0],
            dtype=np.float64,
        )

        self.assertEqual(
            _outermost_signed_edge_index(
                negative_gradient,
                start_index=30,
                polarity="negative",
            ),
            31,
        )
        self.assertEqual(
            _outermost_signed_edge_index(
                positive_gradient,
                start_index=60,
                polarity="positive",
            ),
            63,
        )

    def test_side_lines_recover_outer_mouth_when_inner_edges_are_stronger(self):
        camera = _Camera()
        gray = np.full((80, 100), 230, dtype=np.uint8)

        # The physical mouth is x=30..70. A darker recessed cavity at x=38..62
        # creates stronger same-polarity gradients that the old argmin/argmax
        # rule incorrectly preferred.
        gray[20:66, 30:71] = 150
        gray[20:66, 38:63] = 10
        rgb = np.repeat(gray[:, :, None], 3, axis=2)

        mask = np.zeros((80, 100), dtype=np.uint8)
        mask[12:21, 44:57] = 255
        mask[20:51, 38:63] = 255

        left, right = _outer_front_side_lines(rgb, mask, camera)
        row = 35.0
        left_x = left[0] * row + left[1]
        right_x = right[0] * row + right[1]

        self.assertLess(left_x, 34.0)
        self.assertGreater(right_x, 66.0)
        self.assertAlmostEqual(0.5 * (left_x + right_x), 50.0, delta=1.0)

    def test_public_center_has_no_manual_offset_or_world_bias(self):
        source = inspect.getsource(aperture_center_pixel)
        forbidden = {"offset", "world_offset", "center_offset", "bias"}
        self.assertTrue(
            forbidden.isdisjoint(
                inspect.signature(aperture_center_pixel).parameters
            )
        )
        self.assertNotIn("MANUAL", source.upper())


if __name__ == "__main__":
    unittest.main()
