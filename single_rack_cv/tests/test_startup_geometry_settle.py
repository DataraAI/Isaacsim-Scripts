from __future__ import annotations

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
MAIN_PATH = ROOT / "main.py"
RUNTIME_PATH = ROOT / "settled_stereo_handoff_runtime.py"
ANGLED_CONFIG_PATH = ROOT / "angled_hand_config.py"


class StartupGeometrySettleTests(unittest.TestCase):
    def test_main_selects_settling_tolerant_handoff_runtime(self):
        source = MAIN_PATH.read_text(encoding="utf-8")
        self.assertIn(
            "from settled_stereo_handoff_runtime import (",
            source,
        )
        self.assertIn(
            "AngledHandStereoHandoffRuntime as "
            "CableMountedSimulationRuntime",
            source,
        )

    def test_transient_geometry_miss_resets_consecutive_window(self):
        source = RUNTIME_PATH.read_text(encoding="utf-8")
        self.assertIn("_TRANSIENT_GEOMETRY_PREFIXES", source)
        self.assertIn("samples.clear()", source)
        self.assertIn("last_transient_error = error", source)
        self.assertIn(
            "if not self._is_transient_geometry_error(error):\n"
            "                    raise",
            source,
        )
        self.assertIn(
            "if len(samples) == cfg.validation_frames:",
            source,
        )

    def test_insertion_waits_for_consecutive_valid_presentation_samples(self):
        source = RUNTIME_PATH.read_text(encoding="utf-8")
        self.assertIn("ConsecutiveValidityWindow", source)
        self.assertIn("def update_partial_insertion(self)", source)
        self.assertIn("self._validate_live_hand_plug_geometry()", source)
        self.assertIn("self._insertion_presentation_window.observe(False)", source)
        self.assertIn("self._insertion_presentation_window.observe(True)", source)
        self.assertIn("INSERTION PRESENTATION VALIDATED", source)

    def test_advancing_insertion_uses_controller_orientation_guard(self):
        source = RUNTIME_PATH.read_text(encoding="utf-8")
        self.assertIn("def _sample_mount_validation_live", source)
        self.assertIn(
            "if self.partial_insertion.phase is InsertionPhase.WAITING_FOR_ALIGNMENT:",
            source,
        )
        self.assertIn(
            "_CableMountedSimulationRuntime._sample_mount_validation_live(",
            source,
        )

    def test_one_degree_palm_limit_is_not_relaxed(self):
        source = ANGLED_CONFIG_PATH.read_text(encoding="utf-8")
        self.assertIn("palm_side_tolerance_deg: float = 1.0", source)
        runtime_source = RUNTIME_PATH.read_text(encoding="utf-8")
        self.assertNotIn("1.01", runtime_source)
        self.assertNotIn("tolerance_deg =", runtime_source)


if __name__ == "__main__":
    unittest.main()
