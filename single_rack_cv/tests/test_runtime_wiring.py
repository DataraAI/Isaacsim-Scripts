from __future__ import annotations

from pathlib import Path
import unittest


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


if __name__ == "__main__":
    unittest.main()
