from __future__ import annotations

import unittest

import numpy as np

from config import FrontRimConfig
from front_rim import (
    BEZEL_OUTWARD_OFFSET_PX,
    expand_detection_roi,
    extract_front_rim,
)


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
    def test_builds_bezel_ring_outside_cavity_box(self) -> None:
        rgb = np.zeros((120, 160, 3), dtype=np.uint8)
        rim = extract_front_rim(
            rgb=rgb,
            bbox_xywh=(55, 42, 50, 36),
            cfg=FrontRimConfig(),
        )

        np.testing.assert_allclose(rim.center_uv, (80.0, 60.0))
        np.testing.assert_allclose(
            rim.corners_uv,
            np.array(
                [
                    [55.0, 42.0],
                    [105.0, 42.0],
                    [105.0, 78.0],
                    [55.0, 78.0],
                ]
            ),
        )
        self.assertEqual(rim.side_samples_uv.shape, (4, 7, 2))
        np.testing.assert_allclose(
            rim.side_samples_uv[0, :, 1],
            42.0 - BEZEL_OUTWARD_OFFSET_PX,
        )
        np.testing.assert_allclose(
            rim.side_samples_uv[1, :, 0],
            105.0 + BEZEL_OUTWARD_OFFSET_PX,
        )
        np.testing.assert_allclose(
            rim.side_samples_uv[2, :, 1],
            78.0 + BEZEL_OUTWARD_OFFSET_PX,
        )
        np.testing.assert_allclose(
            rim.side_samples_uv[3, :, 0],
            55.0 - BEZEL_OUTWARD_OFFSET_PX,
        )

    def test_internal_image_edges_do_not_move_bezel_ring(self) -> None:
        plain = np.zeros((120, 160, 3), dtype=np.uint8)
        cluttered = plain.copy()
        cluttered[45:77, 60:63] = 255
        cluttered[50:53, 58:103] = 255
        cluttered[65:68, 58:103] = 255

        plain_rim = extract_front_rim(
            plain,
            (55, 42, 50, 36),
            FrontRimConfig(),
        )
        cluttered_rim = extract_front_rim(
            cluttered,
            (55, 42, 50, 36),
            FrontRimConfig(),
        )

        np.testing.assert_allclose(
            cluttered_rim.side_samples_uv,
            plain_rim.side_samples_uv,
        )
        np.testing.assert_allclose(
            cluttered_rim.corners_uv,
            plain_rim.corners_uv,
        )

    def test_rejects_box_too_small_for_bezel_sampling(self) -> None:
        rgb = np.zeros((40, 50, 3), dtype=np.uint8)
        with self.assertRaisesRegex(RuntimeError, "too small"):
            extract_front_rim(
                rgb=rgb,
                bbox_xywh=(10, 10, 1, 1),
                cfg=FrontRimConfig(),
            )


if __name__ == "__main__":
    unittest.main()
