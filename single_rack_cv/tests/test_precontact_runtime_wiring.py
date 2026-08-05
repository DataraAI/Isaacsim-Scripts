from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class FullInsertionRuntimeWiringTests(unittest.TestCase):
    def test_main_selects_explicit_full_insertion_runtime(self):
        source = (ROOT / "main.py").read_text()
        export_source = (ROOT / "full_insertion_runtime.py").read_text()
        self.assertIn("from full_insertion_runtime import (", source)
        self.assertIn(
            "AngledHandStereoHandoffRuntime as CableMountedSimulationRuntime",
            source,
        )
        self.assertNotIn("from precontact_runtime import (", source)
        self.assertIn(
            "from handoff_position_hold_runtime import",
            export_source,
        )

    def test_full_mode_disables_only_precontact_cap_before_base_import(self):
        source = (ROOT / "full_insertion_base_runtime.py").read_text()
        connector_patch = source.index(
            "_connector_tcp_usd.PRECONTACT_ALIGNMENT_ONLY = False"
        )
        mount_import = source.index("import scale_aware_cable_mount")
        base_import = source.index("from settled_stereo_handoff_runtime import")
        self.assertLess(connector_patch, mount_import)
        self.assertLess(mount_import, base_import)
        self.assertIn(
            "_scale_aware_cable_mount.PRECONTACT_ALIGNMENT_ONLY = False",
            source,
        )
        self.assertIn("FULL GUARDED INSERTION MODE ACTIVE", source)
        self.assertIn("_EXPECTED_TOTAL_COMMANDS = 48", source)
        self.assertIn("_EXPECTED_FINAL_PORT_DEPTH_M = 0.010", source)
        self.assertIn("final depth inside opening", source)

    def test_full_runtime_keeps_proven_two_stage_controller(self):
        source = (ROOT / "settled_stereo_handoff_runtime.py").read_text()
        self.assertIn("ConsecutivePoseInsertionController", source)
        self.assertIn("ExplicitInsertionAxisAdapter", source)
        self.assertIn("PROVEN MAIN INSERTION SETTLING ACTIVE", source)
        self.assertIn("position tolerance: 0.300 mm", source)
        self.assertIn("required consecutive frames: 6", source)
        self.assertNotIn("max_lateral_drift_m=", source)
        self.assertNotIn("max_orientation_error_deg=", source)

    def test_rejected_tcp_still_forces_probe_lock(self):
        source = (ROOT / "scale_aware_cable_mount.py").read_text()
        self.assertIn("derivation_accepted", source)
        self.assertIn("TCP_PROBE_ONLY or not derivation_accepted", source)
        self.assertIn("not derivation_accepted", source)

    def test_dormant_precontact_runtime_remains_nonpenetrating(self):
        source = (ROOT / "precontact_runtime.py").read_text()
        self.assertIn("build_precontact_limits", source)
        self.assertIn("commanded_port_depth_m >= 0.0", source)
        self.assertIn("PRECONTACT ALIGNMENT HOLD REACHED", source)
        self.assertIn("penetration commands: disabled", source)


if __name__ == "__main__":
    unittest.main()
