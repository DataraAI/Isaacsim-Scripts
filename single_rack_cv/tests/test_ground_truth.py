from __future__ import annotations

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class GroundTruthStructureTests(unittest.TestCase):
    def test_generator_stamps_high_resolution_before_shutdown(self):
        source = (ROOT / "tools" / "generate_ground_truth.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("front_plane_ground_truth.json", source)
        self.assertIn('"camera_resolution_height_width"', source)
        self.assertIn("[960, 1280]", source)
        self.assertIn("write_result_with_resolution", source)
        self.assertNotIn("highres_config", source)

    def test_rtx_truth_is_benchmark_only(self):
        implementation = (
            ROOT / "tools" / "extract_front_rim_ground_truth.py"
        ).read_text(encoding="utf-8")
        self.assertIn("omni.kit.raycast.query", implementation)
        self.assertIn("control_usage", implementation)
        self.assertIn("forbidden", implementation.lower())


if __name__ == "__main__":
    unittest.main()
