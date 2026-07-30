from __future__ import annotations

import inspect
import unittest

import numpy as np

from stereo_center_projective import (
    _projective_center_from_lines,
    estimate_stereo_aperture_center,
)


def _line_x_from_y(first: np.ndarray, second: np.ndarray) -> tuple[float, float]:
    dy = float(second[1] - first[1])
    if abs(dy) <= 1.0e-12:
        raise ValueError("Side line must not be horizontal.")
    slope = float(second[0] - first[0]) / dy
    return slope, float(first[0] - slope * first[1])


def _line_y_from_x(first: np.ndarray, second: np.ndarray) -> tuple[float, float]:
    dx = float(second[0] - first[0])
    if abs(dx) <= 1.0e-12:
        raise ValueError("Horizontal rim line must not be vertical.")
    slope = float(second[1] - first[1]) / dx
    return slope, float(first[1] - slope * first[0])


def _independent_diagonal_intersection(corners: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack((corners, np.ones(4, dtype=np.float64)))
    first_diagonal = np.cross(homogeneous[0], homogeneous[2])
    second_diagonal = np.cross(homogeneous[1], homogeneous[3])
    point = np.cross(first_diagonal, second_diagonal)
    return point[:2] / point[2]


class ProjectiveFrontRimCenterTests(unittest.TestCase):
    def test_center_is_diagonal_intersection_of_oblique_rim(self):
        corners = np.array(
            [
                [112.0, 71.0],
                [231.0, 77.0],
                [219.0, 166.0],
                [121.0, 158.0],
            ],
            dtype=np.float64,
        )
        left = _line_x_from_y(corners[0], corners[3])
        right = _line_x_from_y(corners[1], corners[2])
        top = _line_y_from_x(corners[0], corners[1])
        bottom = _line_y_from_x(corners[3], corners[2])

        actual = _projective_center_from_lines(left, right, top, bottom)
        expected = _independent_diagonal_intersection(corners)

        np.testing.assert_allclose(actual, expected, atol=1.0e-9)

    def test_runtime_estimator_has_no_manual_center_fraction_or_offset(self):
        source = inspect.getsource(estimate_stereo_aperture_center)
        self.assertNotIn("CENTER_HORIZONTAL_FRACTION", source)
        self.assertNotIn("CENTER_VERTICAL_FRACTION", source)
        forbidden = {"offset", "world_offset", "center_offset", "bias"}
        self.assertTrue(
            forbidden.isdisjoint(
                inspect.signature(estimate_stereo_aperture_center).parameters
            )
        )


if __name__ == "__main__":
    unittest.main()
