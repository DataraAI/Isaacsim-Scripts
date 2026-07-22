from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types
import unittest


ROOT = Path(__file__).resolve().parents[1]


def _load_generator_module():
    path = ROOT / "tools" / "generate_ground_truth.py"
    spec = importlib.util.spec_from_file_location(
        "ground_truth_generator_test_module",
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class GroundTruthStructureTests(unittest.TestCase):
    def test_generator_replaces_shadowing_cv2_config_with_project_config(self):
        generator = _load_generator_module()
        original = sys.modules.get("config")
        fake = types.ModuleType("config")
        fake.__file__ = "/isaac/exts/omni.pip.compute/pip_prebundle/cv2/config.py"
        fake.CONFIG = object()
        sys.modules["config"] = fake
        try:
            loaded = generator._load_project_config()
            self.assertEqual(
                Path(loaded.__file__).resolve(),
                (ROOT / "config.py").resolve(),
            )
            self.assertIs(sys.modules["config"], loaded)
        finally:
            if original is None:
                sys.modules.pop("config", None)
            else:
                sys.modules["config"] = original

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
