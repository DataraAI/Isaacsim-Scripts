from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class PrecontactRuntimeWiringTests(unittest.TestCase):
    def test_main_selects_precontact_runtime(self):
        source = (ROOT / "main.py").read_text()
        self.assertIn("from precontact_runtime import (", source)
        self.assertNotIn(
            "from settled_stereo_handoff_runtime import (\n"
            "        AngledHandStereoHandoffRuntime as CableMountedSimulationRuntime",
            source,
        )

    def test_runtime_caps_existing_controller_without_relaxing_limits(self):
        source = (ROOT / "precontact_runtime.py").read_text()
        self.assertIn("build_precontact_limits", source)
        self.assertIn("ConsecutivePoseInsertionController", source)
        self.assertIn("ExplicitInsertionAxisAdapter", source)
        self.assertIn("terminal_port_depth_m >= 0.0", source)
        self.assertIn("commanded_port_depth_m >= 0.0", source)
        self.assertNotIn("max_lateral_drift_m=", source)
        self.assertNotIn("max_orientation_error_deg=", source)
        self.assertNotIn("EstimatedPortPoint", source)
        self.assertNotIn("FrozenPortPoint", source)

    def test_mode_is_exactly_two_mm_before_opening(self):
        source = (ROOT / "connector_tcp_usd.py").read_text()
        self.assertIn("TCP_PROBE_ONLY = False", source)
        self.assertIn("PRECONTACT_ALIGNMENT_ONLY = True", source)
        self.assertIn("PRECONTACT_HOLD_OFFSET_M = 0.002", source)
        self.assertIn("penetration commands: disabled", source)

    def test_rejected_tcp_still_forces_probe_lock(self):
        source = (ROOT / "scale_aware_cable_mount.py").read_text()
        self.assertIn("derivation_accepted", source)
        self.assertIn("TCP_PROBE_ONLY or not derivation_accepted", source)
        self.assertIn(
            "PRECONTACT_ALIGNMENT_ONLY and derivation_accepted",
            source,
        )

    def test_terminal_logs_are_no_contact_specific(self):
        source = (ROOT / "precontact_runtime.py").read_text()
        self.assertIn("PRECONTACT ALIGNMENT STARTED", source)
        self.assertIn("PRECONTACT ALIGNMENT STEP SETTLED", source)
        self.assertIn("PRECONTACT ALIGNMENT HOLD REACHED", source)
        self.assertIn("PRECONTACT ALIGNMENT ABORTED", source)
        self.assertIn("actual depth relative to opening", source)
        self.assertIn("lateral drift", source)
        self.assertIn("orientation error", source)
        self.assertIn("ToolCenter tracking error", source)
        self.assertIn("penetration commands: disabled", source)
        self.assertNotIn("PARTIAL INSERTION COMPLETE", source)


if __name__ == "__main__":
    unittest.main()
