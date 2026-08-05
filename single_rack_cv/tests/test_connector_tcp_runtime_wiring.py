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

    def test_probe_locks_motion_before_detector_initialization(self):
        source = (ROOT / "main.py").read_text()
        probe = source.index("[CONNECTOR TCP PROBE] MOTION LOCKED")
        detector = source.index("detector.initialize()")
        self.assertLess(probe, detector)
        self.assertIn("runtime.step()", source[probe:detector])
        self.assertNotIn("update_partial_insertion", source[probe:detector])

    def test_usd_adapter_uses_mesh_fallback_and_two_markers(self):
        source = (ROOT / "connector_tcp_usd.py").read_text()
        self.assertIn("connected_component_bounds", source)
        self.assertIn("#whole", source)
        self.assertIn("LegacyPlugTipProbe", source)
        self.assertIn("DerivedInsertionTcpProbe", source)
        self.assertIn("TCP_PROBE_ONLY = True", source)
        self.assertNotIn("world_offset", source)
        self.assertNotIn("port_offset", source)


if __name__ == "__main__":
    unittest.main()
