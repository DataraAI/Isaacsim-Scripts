from __future__ import annotations

import unittest

import numpy as np

from config import FrontRimConfig
from front_rim import expand_detection_roi, extract_front_rim


def synthetic_port_image(
    *,
    height: int = 120,
    width: int = 160,
    opening_xyxy: tuple[int, int, int, int] = (55, 42, 105, 78),
) -> np.ndarray:
    image = np.full((height, width, 3), 190, dtype=np.uint8)
    x0, y0, x1, y1 = opening_xyxy
    image[y0:y1, x0:x1] = 20
    image[y0 - 2:y0, x0 - 2:x1 + 2] = 235
    image[y1:y1 + 2, x0 - 2:x1 + 2] = 235
    image[y0 - 2:y1 + 2, x0 - 2:x0] = 235
    image[y0 - 2:y1 + 2, x1:x1 + 2] = 235
    return image


class FrontRimConfigTests(unittest.TestCase):
    def test_front_rim_starts_disabled(self) -> None:
        self.assertFalse(FrontRimConfig().enabled)

    def test_expand_detection_roi_clamps_to_image(self) -> None:
        cfg = FrontRimConfig(
            roi_expand_ratio=0.50,
            roi_min_margin_px=8,
        )
        roi = expand_detection_roi(
            bbox_xywh=(2, 3, 20, 10),
            image_shape_hw=(40, 50),
            cfg=cfg,
        )
        self.assertEqual(roi, (0, 0, 32, 23))

    def test_expand_detection_roi_rejects_zero_area(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive width and height"):
            expand_detection_roi(
                bbox_xywh=(10, 10, 0, 8),
                image_shape_hw=(40, 50),
                cfg=FrontRimConfig(),
            )


class FrontRimExtractionTests(unittest.TestCase):
    def test_extracts_dense_front_rim(self) -> None:
        rgb = synthetic_port_image()
        rim = extract_front_rim(
            rgb=rgb,
            bbox_xywh=(55, 42, 50, 36),
            cfg=FrontRimConfig(),
        )
        np.testing.assert_allclose(
            rim.center_uv,
            (80.0, 60.0),
            atol=1.0,
        )
        self.assertEqual(rim.side_samples_uv.shape, (4, 7, 2))
        self.assertGreaterEqual(
            min(line.inlier_uv.shape[0] for line in rim.side_lines),
            12,
        )

    def test_rejects_missing_bottom_rim(self) -> None:
        rgb = synthetic_port_image()
        rgb[76:, :] = 20
        with self.assertRaisesRegex(RuntimeError, "bottom"):
            extract_front_rim(
                rgb=rgb,
                bbox_xywh=(55, 42, 50, 36),
                cfg=FrontRimConfig(),
            )


if __name__ == "__main__":
    unittest.main()
