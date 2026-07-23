from __future__ import annotations

from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ScaleAwareCableMountContractTests(unittest.TestCase):
    def test_adapter_extracts_rigid_joint_frames_from_affine_usd_poses(self):
        source = (ROOT / "scale_aware_cable_mount.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("class ScaleAwareCableMount(CableMount)", source)
        self.assertIn("rigid_pose_from_affine", source)
        self.assertIn("world_from_hand_pose", source)
        self.assertIn("world_from_plug_pose", source)
        self.assertIn("UsdPhysics.FixedJoint.Define", source)
        self.assertIn("axis_scale_m_per_local_unit", source)
        self.assertIn("physical_dimensions_m", source)
        self.assertIn("replace(", source)
        self.assertNotIn("CableMountProxy", source)
        self.assertNotIn("PhysxAutoDeformableAttachmentAPI", source)

    def test_runtime_uses_scale_aware_adapter(self):
        source = (ROOT / "cable_runtime.py").read_text(encoding="utf-8")
        self.assertIn(
            "from scale_aware_cable_mount import ScaleAwareCableMount",
            source,
        )
        self.assertIn(
            "self.cable_mount = ScaleAwareCableMount(self.cfg)",
            source,
        )


if __name__ == "__main__":
    unittest.main()
