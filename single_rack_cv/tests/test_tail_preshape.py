from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np

from cable.tail_preshape import preshape_free_hanging_tail


ROOT = Path(__file__).resolve().parents[1]


class TailPreshapeTests(unittest.TestCase):
    def test_keeps_both_ends_and_drops_middle(self):
        x = np.linspace(0.0, 1.0, 101)
        points = np.column_stack([x, np.zeros_like(x), np.zeros_like(x)])
        bent = preshape_free_hanging_tail(
            points,
            plug_world_m=np.array([0.0, 0.0, 0.0]),
            down_world_axis=np.array([0.0, 0.0, -1.0]),
            anchor_length_m=0.02,
            bend_length_m=0.10,
            far_anchor_length_m=0.02,
            drop_m=0.10,
        )
        np.testing.assert_allclose(bent[0], points[0], atol=1e-12)
        np.testing.assert_allclose(bent[-1], points[-1], atol=1e-12)
        self.assertAlmostEqual(bent[50, 2], -0.10, places=12)
        self.assertLess(bent[10, 2], 0.0)
        self.assertGreater(bent[10, 2], -0.10)

    def test_finds_plug_end_when_point_order_is_reversed(self):
        x = np.linspace(1.0, 0.0, 101)
        points = np.column_stack([x, np.zeros_like(x), np.zeros_like(x)])
        bent = preshape_free_hanging_tail(
            points,
            plug_world_m=np.array([0.0, 0.0, 0.0]),
            down_world_axis=np.array([0.0, 0.0, -1.0]),
            anchor_length_m=0.02,
            bend_length_m=0.10,
            far_anchor_length_m=0.02,
            drop_m=0.10,
        )
        np.testing.assert_allclose(bent[-1], points[-1], atol=1e-12)
        self.assertAlmostEqual(float(np.min(bent[:, 2])), -0.10, places=12)

    def test_rejects_degenerate_point_cloud(self):
        points = np.zeros((8, 3), dtype=np.float64)
        with self.assertRaisesRegex(ValueError, "tail point cloud is degenerate"):
            preshape_free_hanging_tail(
                points,
                plug_world_m=np.zeros(3),
                down_world_axis=np.array([0.0, 0.0, -1.0]),
                anchor_length_m=0.02,
                bend_length_m=0.10,
                far_anchor_length_m=0.02,
                drop_m=0.10,
            )

    def test_runtime_wires_preshape_before_play(self):
        source = (ROOT / "cable" / "scale_aware_cable_mount.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("_TAIL_PRESHAPE_DROP_M = 0.100", source)
        self.assertIn("preshape_free_hanging_tail", source)
        placement = source.index("super().author_before_play(")
        preshape = source.index("self._preshape_deformable_tail()")
        self.assertLess(placement, preshape)
        self.assertIn("omniphysics:restShapePoints", source)
        self.assertIn("primvars:normals", source)
        self.assertIn("built_in_attachment_is_preserved", source)
        self.assertNotIn("create_auto_deformable_attachment", source)


if __name__ == "__main__":
    unittest.main()
