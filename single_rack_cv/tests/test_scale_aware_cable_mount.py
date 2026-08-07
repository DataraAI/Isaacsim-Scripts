from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np

from cable.scale_aware_cable_mount import _matrix_to_gf_quatf_compatible

ROOT = Path(__file__).resolve().parents[1]


class ScaleAwareCableMountContractTests(unittest.TestCase):
    def test_adapter_extracts_rigid_joint_frames_from_affine_usd_poses(self):
        source = (ROOT / "cable" / "scale_aware_cable_mount.py").read_text(
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

    def test_matrix_to_quatf_uses_supported_openusd_binding(self):
        rotation = np.array(
            [
                [0.0, -1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        quaternion = _matrix_to_gf_quatf_compatible(rotation)
        imaginary = quaternion.GetImaginary()
        expected = np.sqrt(0.5)
        self.assertAlmostEqual(float(quaternion.GetReal()), expected, places=6)
        self.assertAlmostEqual(float(imaginary[0]), 0.0, places=6)
        self.assertAlmostEqual(float(imaginary[1]), 0.0, places=6)
        self.assertAlmostEqual(float(imaginary[2]), expected, places=6)

    def test_adapter_patches_and_restores_affine_root_helpers(self):
        source = (ROOT / "cable" / "scale_aware_cable_mount.py").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            "compute_world_from_root_for_tip_preserving_affine",
            source,
        )
        self.assertIn("_numpy_to_gf_matrix_affine", source)
        self.assertIn(
            "cable_mount_module.compute_world_from_root_for_tip = (",
            source,
        )
        self.assertIn(
            "cable_mount_module._numpy_to_gf_matrix = (",
            source,
        )
        self.assertIn("finally:", source)
        self.assertIn("original_compute", source)
        self.assertIn("original_converter", source)

    def test_runtime_uses_scale_aware_adapter(self):
        source = (ROOT / "runtime" / "cable_runtime_base.py").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            "from cable.scale_aware_cable_mount import ScaleAwareCableMount",
            source,
        )
        self.assertIn(
            "self.cable_mount = ScaleAwareCableMount(self.cfg)",
            source,
        )


if __name__ == "__main__":
    unittest.main()
