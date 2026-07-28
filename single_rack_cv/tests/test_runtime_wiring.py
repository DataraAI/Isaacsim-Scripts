from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np

from config import CONFIG


ROOT = Path(__file__).resolve().parents[1]


class RuntimeWiringTests(unittest.TestCase):
    def test_canonical_config_is_high_resolution_front_plane(self):
        source = (ROOT / "config.py").read_text(encoding="utf-8")
        self.assertIn("resolution: tuple[int, int] = (960, 1280)", source)
        self.assertIn("class FrontPlaneRuntimeConfig", source)
        self.assertIn("enabled: bool = True", source)
        self.assertIn("front_plane: FrontPlaneRuntimeConfig", source)
        self.assertNotIn("class FrontRimConfig", source)

    def test_main_refines_before_motion_and_debug(self):
        source = (ROOT / "main.py").read_text(encoding="utf-8")
        self.assertIn("from live_control import refine_live_observation", source)
        refine = source.index("refine_live_observation(")
        observe = source.index("runtime.observe_visual_servo(observation)")
        debug = source.index("debug.handle(")
        self.assertLess(refine, observe)
        self.assertLess(refine, debug)
        self.assertIn("CONFIG.front_plane.enabled", source)

    def test_failure_path_holds_and_reacquires(self):
        source = (ROOT / "main.py").read_text(encoding="utf-8")
        self.assertIn("runtime.note_perception_failure()", source)
        self.assertIn("RGB stereo capture", source)
        self.assertIn("no manual depth offset", source)

    def test_toolcenter_calibration_rolls_and_extends_pregrasped_plug(self):
        np.testing.assert_allclose(
            CONFIG.ik.tool_center_local_position_m,
            (0.0, 0.0, 0.1334),
            atol=0.0,
        )
        np.testing.assert_allclose(
            CONFIG.ik.tool_center_local_orientation_wxyz,
            (
                0.7071067811865476,
                0.0,
                0.0,
                -0.7071067811865475,
            ),
            atol=1.0e-15,
        )
        config_source = (ROOT / "config.py").read_text(encoding="utf-8")
        self.assertNotIn("presentation_roll_deg", config_source)
        self.assertNotIn("forward_tip_offset_m", config_source)

    def test_cable_mount_uses_existing_rigid_plug_topology(self):
        source = (ROOT / "cable_mount.py").read_text(encoding="utf-8")
        self.assertIn("UsdPhysics.RigidBodyAPI", source)
        self.assertIn('HasAPI("PhysxAutoDeformableAttachmentAPI")', source)
        self.assertIn("built_in_attachment_is_preserved", source)
        self.assertIn("compute_world_from_root_for_tip", source)
        self.assertNotIn("CableMountProxy", source)
        self.assertNotIn("create_auto_deformable_attachment", source)
        self.assertNotIn("maskShapes", source)
        self.assertNotIn("PhysxPhysicsAttachment", source)

    def test_direct_plug_joint_and_narrow_collision_filtering(self):
        source = (ROOT / "cable_mount.py").read_text(encoding="utf-8")
        self.assertIn("UsdPhysics.FixedJoint.Define", source)
        self.assertIn("CreateBody0Rel", source)
        self.assertIn("CreateBody1Rel", source)
        self.assertIn("fixed_joint_is_valid", source)
        self.assertIn("UsdPhysics.FilteredPairsAPI.Apply", source)
        self.assertIn("finger_link_names", source)
        self.assertNotIn("create_auto_deformable_attachment", source)

    def test_tracked_plug_is_forced_dynamic_before_fixed_joint(self):
        source = (ROOT / "scale_aware_cable_mount.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("def _ensure_tracked_plug_dynamic", source)
        self.assertIn("CreateKinematicEnabledAttr().Set(False)", source)
        self.assertIn("Tracked RJ45 plug remained kinematic", source)
        dynamic = source.index("self._ensure_tracked_plug_dynamic()")
        joint = source.index("UsdPhysics.FixedJoint.Define")
        self.assertLess(dynamic, joint)

    def test_cable_runtime_owns_gpu_startup_and_bounded_mount_gate(self):
        source = (ROOT / "cable_runtime.py").read_text(encoding="utf-8")
        self.assertIn(
            "class CableMountedSimulationRuntime(SimulationRuntime)",
            source,
        )
        self.assertIn("set_enabled_gpu_dynamics(True)", source)
        self.assertIn('set_broadphase_type("GPU")', source)
        self.assertIn('set_solver_type("TGS")', source)
        self.assertIn("CableMount(self.cfg)", source)
        self.assertIn("author_before_play", source)
        self.assertIn("configure_fingers", source)
        self.assertIn("def prepare_for_perception", source)
        self.assertIn("max_prepare_frames", source)
        self.assertIn("validate_mount_window", source)
        self.assertIn("self._get_world_pose(hand_path)", source)
        self.assertNotIn("while not", source)
        self.assertNotIn("CableMountProxy", source)
        self.assertNotIn("create_auto_deformable_attachment", source)

    def test_mount_validation_precedes_debug_yolo_and_control_loop(self):
        source = (ROOT / "main.py").read_text(encoding="utf-8")
        self.assertIn("CableMountedSimulationRuntime", source)
        prepare = source.index("runtime.prepare_for_perception()")
        debug = source.index("debug = DebugOutputs")
        yolo = source.index("detector.initialize()")
        loop = source.index("while runtime.is_running()")
        self.assertLess(prepare, debug)
        self.assertLess(prepare, yolo)
        self.assertLess(yolo, loop)

    def test_partial_insertion_config_is_exact(self):
        self.assertTrue(CONFIG.insertion.enabled)
        self.assertEqual(CONFIG.insertion.total_depth_m, 0.010)
        self.assertEqual(CONFIG.insertion.step_size_m, 0.0005)
        self.assertEqual(
            CONFIG.insertion.settle_position_tolerance_m,
            0.0003,
        )
        self.assertEqual(CONFIG.insertion.required_settled_frames, 6)
        self.assertEqual(CONFIG.insertion.step_timeout_s, 2.0)
        self.assertEqual(CONFIG.insertion.max_lateral_drift_m, 0.0005)
        self.assertEqual(CONFIG.insertion.max_orientation_error_deg, 1.0)

    def test_partial_insertion_runtime_is_wired_after_visual_completion(self):
        runtime_source = (
            ROOT / "cable_runtime" / "__init__.py"
        ).read_text(encoding="utf-8")
        main_source = (ROOT / "main.py").read_text(encoding="utf-8")
        self.assertIn("PartialInsertionController", runtime_source)
        self.assertIn("def update_partial_insertion", runtime_source)
        self.assertIn("tool_pose_to_hand_pose", runtime_source)
        self.assertIn("compute_inverse_kinematics", runtime_source)
        self.assertIn("could not publish insertion target", runtime_source)
        self.assertIn("runtime.update_partial_insertion()", main_source)
        self.assertLess(
            main_source.index("runtime.update_visual_servo_completion()"),
            main_source.index("runtime.update_partial_insertion()"),
        )

    def test_readme_documents_mount_and_partial_insertion_kill_switch(self):
        source = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("direct fixed joint", source)
        self.assertIn("built-in deformable attachment", source)
        self.assertIn("RJ45 insertion tip", source)
        self.assertIn("GPU dynamics", source)
        self.assertIn("30/30", source)
        self.assertIn("10 mm partial insertion", source)
        self.assertIn("0.5 mm steps", source)
        self.assertIn("frozen ToolCenter +Z", source)
        self.assertIn("holds on abort", source)
        self.assertNotIn("No insertion motion", source)
        self.assertNotIn("recovery/pre-single-rack-cleanup", source)


if __name__ == "__main__":
    unittest.main()
