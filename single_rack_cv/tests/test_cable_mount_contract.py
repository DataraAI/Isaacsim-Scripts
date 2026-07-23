from __future__ import annotations

from pathlib import Path
import unittest

from config import CONFIG


class CableMountContractTests(unittest.TestCase):
    def test_canonical_paths_and_limits(self):
        cfg = CONFIG.cable_mount
        self.assertTrue(cfg.enabled)
        self.assertEqual(
            cfg.usd_path,
            "/home/aayush/isaacsim_assets/Network cable 001/model_Networkcable1_69323.usd",
        )
        self.assertEqual(cfg.root_path, "/World/NetworkCable")
        self.assertEqual(
            cfg.tracked_plug_path,
            "/World/NetworkCable/E_crystal_head1_45",
        )
        self.assertEqual(cfg.proxy_path, "/World/CableMountProxy")
        self.assertEqual(cfg.fixed_joint_path, "/World/CableMountFixedJoint")
        self.assertEqual(cfg.attachment_path, "/World/CableMountAttachment")
        self.assertEqual(cfg.mask_path, "/World/CableMountAttachment/MaskShape")
        self.assertEqual(cfg.validation_frames, 30)
        self.assertAlmostEqual(cfg.attachment_padding_m, 0.0005)
        self.assertAlmostEqual(cfg.finger_total_clearance_m, 0.001)
        self.assertAlmostEqual(cfg.max_tip_error_m, 0.0005)
        self.assertAlmostEqual(cfg.max_axis_error_deg, 1.0)

    def test_feature_branch_uses_cuda_for_all_diagnostic_modes(self):
        self.assertEqual(CONFIG.scene.device, "cuda:0")

    def test_schema_probe_prohibits_legacy_fallback(self):
        source = Path("tools/inspect_cable_asset.py").read_text(encoding="utf-8")
        self.assertIn('HasAPI("OmniPhysicsDeformableBodyAPI")', source)
        self.assertIn("cable_asset_schema.json", source)
        self.assertNotIn("PhysxPhysicsAttachment", source)
        self.assertNotIn("PhysxAutoAttachmentAPI", source)


if __name__ == "__main__":
    unittest.main()
