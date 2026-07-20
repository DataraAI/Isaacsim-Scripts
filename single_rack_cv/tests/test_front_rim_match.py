from __future__ import annotations

import unittest

import cv2
import numpy as np

from config import FrontRimConfig
from front_rim import extract_front_rim
from front_rim_match import (
    EpipolarPatchConfig,
    build_matched_right_rim,
    match_front_bezel_samples,
)


IMAGE_HEIGHT = 120
IMAGE_WIDTH = 180
LEFT_BBOX = (70, 45, 40, 24)
ACTUAL_SHIFT = np.array([-12.5, 1.0], dtype=np.float64)


def _ring_center(bbox_xywh: tuple[int, int, int, int]) -> np.ndarray:
    x, y, width, height = bbox_xywh
    return np.array([x + 0.5 * width, y + 0.5 * height], dtype=np.float64)


def _make_rims(
    left_rgb: np.ndarray,
    right_rgb: np.ndarray,
    prediction_error_uv: tuple[float, float] = (1.0, -0.5),
):
    cfg = FrontRimConfig()
    left_center = _ring_center(LEFT_BBOX)
    predicted_right_center = (
        left_center
        + ACTUAL_SHIFT
        + np.asarray(prediction_error_uv, dtype=np.float64)
    )
    right_bbox = (
        int(round(LEFT_BBOX[0] + ACTUAL_SHIFT[0])),
        int(round(LEFT_BBOX[1] + ACTUAL_SHIFT[1])),
        LEFT_BBOX[2],
        LEFT_BBOX[3],
    )
    left_rim = extract_front_rim(
        left_rgb,
        LEFT_BBOX,
        cfg,
        center_uv=tuple(left_center),
    )
    right_rim = extract_front_rim(
        right_rgb,
        right_bbox,
        cfg,
        center_uv=tuple(predicted_right_center),
    )
    return left_rim, right_rim


def _translated_textured_pair() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(4)
    gray = rng.normal(127.0, 45.0, (IMAGE_HEIGHT, IMAGE_WIDTH))
    gray = np.clip(gray, 0.0, 255.0).astype(np.uint8)
    cv2.rectangle(gray, (50, 30), (130, 90), 220, 2)
    cv2.line(gray, (30, 60), (150, 60), 30, 2)
    left = np.repeat(gray[:, :, None], 3, axis=2)
    transform = np.array(
        [[1.0, 0.0, ACTUAL_SHIFT[0]], [0.0, 1.0, ACTUAL_SHIFT[1]]],
        dtype=np.float32,
    )
    right = cv2.warpAffine(
        left,
        transform,
        (IMAGE_WIDTH, IMAGE_HEIGHT),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )
    return left, right


class FrontRimEpipolarMatcherTests(unittest.TestCase):
    def test_recovers_known_subpixel_translation(self) -> None:
        left, right = _translated_textured_pair()
        left_rim, right_rim = _make_rims(left, right)
        result = match_front_bezel_samples(
            left,
            right,
            left_rim,
            right_rim,
        )

        self.assertGreaterEqual(result.accepted_count, 20)
        recovered = (
            result.right_samples_uv[result.valid_mask]
            - left_rim.side_samples_uv[result.valid_mask]
        )
        errors = np.linalg.norm(recovered - ACTUAL_SHIFT, axis=1)
        self.assertLessEqual(float(np.median(errors)), 0.51)
        self.assertLessEqual(result.roundtrip_p95_px(), 0.75)
        self.assertGreater(result.median_ncc(), 0.72)

    def test_rejects_flat_patches(self) -> None:
        left = np.full((IMAGE_HEIGHT, IMAGE_WIDTH, 3), 120, dtype=np.uint8)
        right = left.copy()
        left_rim, right_rim = _make_rims(left, right, (0.0, 0.0))
        result = match_front_bezel_samples(
            left,
            right,
            left_rim,
            right_rim,
        )
        self.assertEqual(result.accepted_count, 0)

    def test_rejects_repetitive_ambiguous_stripes(self) -> None:
        gray = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.uint8)
        for column in range(IMAGE_WIDTH):
            gray[:, column] = 40 if (column // 2) % 2 == 0 else 210
        left = np.repeat(gray[:, :, None], 3, axis=2)
        transform = np.array(
            [[1.0, 0.0, ACTUAL_SHIFT[0]], [0.0, 1.0, ACTUAL_SHIFT[1]]],
            dtype=np.float32,
        )
        right = cv2.warpAffine(
            left,
            transform,
            (IMAGE_WIDTH, IMAGE_HEIGHT),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )
        left_rim, right_rim = _make_rims(left, right)
        result = match_front_bezel_samples(
            left,
            right,
            left_rim,
            right_rim,
        )
        self.assertLess(result.accepted_count, 5)

    def test_invalid_matches_are_forced_outside_epipolar_gate(self) -> None:
        left, right = _translated_textured_pair()
        left_rim, right_rim = _make_rims(left, right)
        strict = EpipolarPatchConfig(min_ncc=0.99999)
        result = match_front_bezel_samples(
            left,
            right,
            left_rim,
            right_rim,
            strict,
        )
        matched_rim = build_matched_right_rim(
            left_rim,
            right_rim,
            result,
            max_epipolar_error_px=2.0,
        )
        invalid = ~result.valid_mask
        vertical_gap = np.abs(
            matched_rim.side_samples_uv[invalid, 1]
            - left_rim.side_samples_uv[invalid, 1]
        )
        self.assertTrue(np.all(vertical_gap > 2.0))


if __name__ == "__main__":
    unittest.main()
