#!/usr/bin/env python3

from __future__ import annotations

import unittest

import cv2
import numpy as np

from lower_mouth_projective_center import aperture_center_pixel


class _Camera:
    image_width_px = 120
    image_height_px = 90


def _mask_with_notch(notch_left: int, notch_right: int) -> np.ndarray:
    mask = np.zeros((90, 120), dtype=np.uint8)
    polygon = np.array(
        [
            [30, 30],
            [notch_left, 29],
            [notch_left - 1, 13],
            [notch_right - 1, 12],
            [notch_right, 28],
            [75, 27],
            [78, 60],
            [28, 62],
        ],
        dtype=np.int32,
    )
    cv2.fillPoly(mask, [polygon], 255)
    return mask


def _intersection(p1, p2, p3, p4):
    first = np.cross(
        [p1[0], p1[1], 1.0],
        [p2[0], p2[1], 1.0],
    )
    second = np.cross(
        [p3[0], p3[1], 1.0],
        [p4[0], p4[1], 1.0],
    )
    point = np.cross(first, second)
    return point[:2] / point[2]


class LowerMouthProjectiveCenterTests(unittest.TestCase):
    def test_upper_notch_shift_does_not_move_lower_mouth_center(self):
        camera = _Camera()
        rgb = np.zeros((90, 120, 3), dtype=np.uint8)

        first = aperture_center_pixel(
            rgb,
            _mask_with_notch(42, 59),
            camera,
        )
        second = aperture_center_pixel(
            rgb,
            _mask_with_notch(47, 64),
            camera,
        )

        self.assertLess(float(np.linalg.norm(first - second)), 0.35)

    def test_isolated_wide_upper_row_does_not_replace_lower_mouth(self):
        camera = _Camera()
        rgb = np.zeros((90, 120, 3), dtype=np.uint8)
        clean = _mask_with_notch(42, 59)
        contaminated = clean.copy()
        contaminated[24, 30:76] = 255

        expected = aperture_center_pixel(rgb, clean, camera)
        actual = aperture_center_pixel(rgb, contaminated, camera)

        self.assertLess(float(np.linalg.norm(actual - expected)), 0.35)

    def test_recovers_projective_center_of_lower_quadrilateral(self):
        camera = _Camera()
        rgb = np.zeros((90, 120, 3), dtype=np.uint8)
        mask = _mask_with_notch(42, 59)

        expected = _intersection(
            np.array([30.0, 30.0]),
            np.array([78.0, 60.0]),
            np.array([75.0, 27.0]),
            np.array([28.0, 62.0]),
        )
        actual = aperture_center_pixel(rgb, mask, camera)

        self.assertLess(float(np.linalg.norm(actual - expected)), 1.0)

    def test_uploaded_like_oblique_shape_centers_lower_mouth(self):
        camera = _Camera()
        rgb = np.zeros((90, 120, 3), dtype=np.uint8)
        mask = np.zeros((90, 120), dtype=np.uint8)
        polygon = np.array(
            [
                [25, 34],
                [40, 33],
                [40, 17],
                [58, 16],
                [61, 32],
                [84, 30],
                [87, 66],
                [22, 68],
            ],
            dtype=np.int32,
        )
        cv2.fillPoly(mask, [polygon], 255)

        expected = _intersection(
            np.array([25.0, 34.0]),
            np.array([87.0, 66.0]),
            np.array([84.0, 30.0]),
            np.array([22.0, 68.0]),
        )
        actual = aperture_center_pixel(rgb, mask, camera)

        self.assertLess(float(np.linalg.norm(actual - expected)), 1.0)


if __name__ == "__main__":
    unittest.main()
