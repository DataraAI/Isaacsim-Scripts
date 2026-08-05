from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ConnectorTcpRuntimeWiringTests(unittest.TestCase):
    def test_scale_aware_mount_replaces_only_plug_tip_frame(self):
        source = (ROOT / "scale_aware_cable_mount.py").read_text()
        self.assertIn(
            "original_detect = cable_mount_module.detect_plug_frame",
            source,
        )
        self.assertIn("derive_plug_frame_from_mesh", source)
        self.assertIn(
            "cable_mount_module.detect_plug_frame = original_detect",
            source,
        )
        self.assertNotIn("EstimatedPortPoint", source)
        self.assertNotIn("FrozenPortPoint", source)

    def test_rejected_derivation_still_locks_before_detector_initialization(self):
        main_source = (ROOT / "main.py").read_text()
        mount_source = (ROOT / "scale_aware_cable_mount.py").read_text()
        probe = main_source.index("[CONNECTOR TCP PROBE] MOTION LOCKED")
        detector = main_source.index("detector.initialize()")
        self.assertLess(probe, detector)
        self.assertIn("runtime.step()", main_source[probe:detector])
        self.assertNotIn("update_partial_insertion", main_source[probe:detector])
        self.assertIn("not derivation_accepted", mount_source)

    def test_usd_adapter_uses_rear_profile_donor_and_two_markers(self):
        source = (ROOT / "connector_tcp_usd.py").read_text()
        self.assertIn("connected_component_bounds", source)
        self.assertIn("#whole", source)
        self.assertIn("maximum_profile_setback_m=0.020", source)
        self.assertIn("profile setback mm", source)
        self.assertIn("LegacyPlugTipProbe", source)
        self.assertIn("DerivedInsertionTcpProbe", source)
        self.assertIn("TCP_PROBE_ONLY = False", source)
        self.assertIn("PRECONTACT_HOLD_OFFSET_M = 0.002", source)
        self.assertNotIn("world_offset", source)
        self.assertNotIn("port_offset", source)

    def test_production_wrapper_disables_precontact_without_touching_tcp_geometry(self):
        source = (ROOT / "full_insertion_runtime.py").read_text()
        self.assertIn(
            "_connector_tcp_usd.PRECONTACT_ALIGNMENT_ONLY = False",
            source,
        )
        self.assertIn(
            "_scale_aware_cable_mount.PRECONTACT_ALIGNMENT_ONLY = False",
            source,
        )
        self.assertNotIn("tip_local", source)
        self.assertNotIn("world_offset", source)
        self.assertNotIn("port_offset", source)


if __name__ == "__main__":
    unittest.main()
