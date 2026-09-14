# aayush/ur5e_6x_cable_insertions/tests/test_runtime.py
from __future__ import annotations

import ast
import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
TANISH_DIR = REPO_ROOT / "tanish"
AAYUSH_DIR = REPO_ROOT / "aayush"
for path in (str(TANISH_DIR), str(AAYUSH_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from ur5e_6x_cable_insertions import config as cfg


class StationTableTests(unittest.TestCase):
    def test_six_stations_and_default_jack(self) -> None:
        self.assertEqual(len(cfg.STATIONS), 6)
        for spec in cfg.STATIONS:
            self.assertEqual(spec.jack_id, "jack_upper_c2")
            self.assertTrue(spec.robot_prim_path.startswith("/World/Robots/UR5e_"))
            self.assertIn("/RJ45_Group01", spec.port_pack_path)
            self.assertTrue(spec.port_contacts_path.endswith("/Group_14343"))

    def test_per_station_jack_override(self) -> None:
        spec = cfg.make_station("negative", "top", jack_id="jack_upper_c5")
        self.assertEqual(spec.jack_id, "jack_upper_c5")
        self.assertTrue(spec.port_contacts_path.endswith("/Group_14345"))

    def test_grasp_tilt_and_friction_match_1x(self) -> None:
        self.assertAlmostEqual(cfg.GRASP_TILT_FROM_DOWN_DEG, 30.0)
        self.assertAlmostEqual(cfg.GRASP_FRICTION_STATIC, 0.8)
        self.assertAlmostEqual(cfg.GRASP_FRICTION_DYNAMIC, 0.8)
        self.assertEqual(cfg.GRASP_FRICTION_COMBINE_MODE, "max")

    def test_home_arm_is_length_6(self) -> None:
        self.assertEqual(np.asarray(cfg.UR5E_HOME_ARM).shape, (6,))


class ControllerWiringTests(unittest.TestCase):
    def test_controller_uses_ur5e_arm_joint_names(self) -> None:
        path = Path(__file__).resolve().parents[1] / "controller.py"
        source = path.read_text(encoding="utf-8")
        self.assertIn("UR5E_ARM_JOINT_NAMES", source)
        self.assertNotIn("UR10E_ARM_JOINT_NAMES", source)

    def test_controller_supports_direct_joint_waypoints(self) -> None:
        path = Path(__file__).resolve().parents[1] / "controller.py"
        source = path.read_text(encoding="utf-8")
        self.assertIn("def add_joint_waypoint", source)
        self.assertIn('"joint_waypoint"', source)


class SceneWiringTests(unittest.TestCase):
    def test_scene_wires_ur5e_cable_physics_and_friction(self) -> None:
        path = Path(__file__).resolve().parents[1] / "scene.py"
        source = path.read_text(encoding="utf-8") if path.exists() else ""
        for needle in (
            "Ur5eSixArmMotionController",
            "GRASP_FRICTION_STATIC",
            "enable_crystal_head_physics",
            "configure_cable_deformable_for_stage",
        ):
            self.assertIn(needle, source)

    def test_scene_does_not_fake_grasp_with_fixed_joint(self) -> None:
        path = Path(__file__).resolve().parents[1] / "scene.py"
        source = path.read_text(encoding="utf-8") if path.exists() else ""
        self.assertNotIn("FixedJoint", source)
        self.assertNotIn("_attach_cable_head_to_gripper", source)

    def test_observe_hand_computed_from_path39(self) -> None:
        path = Path(__file__).resolve().parents[1] / "scene.py"
        source = path.read_text(encoding="utf-8") if path.exists() else ""
        start = source.find("prim_bbox(stage, spec.path39)")
        self.assertGreater(start, 0, "head39 bbox lookup missing from scene.py")
        end = source.find("observe_hand =", start)
        self.assertGreater(end, start, "observe_hand assignment missing from scene.py")
        window = source[start : end + 120]
        self.assertIn("spec.path39", window)
        self.assertIn("observation_hand_from_head39", window)
        self.assertNotIn("grasp_paths", window)


class PrimitiveSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.path = Path(__file__).resolve().parents[1] / "primitives.py"
        cls.source = cls.path.read_text(encoding="utf-8")
        cls.tree = ast.parse(cls.source)
        cls.functions = {
            node.name: node
            for node in ast.walk(cls.tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }

    def test_exports_all_task_6_primitives(self) -> None:
        expected = {
            "queue_move",
            "detect_grasp_part",
            "queue_grasp",
            "check_physical_grasp",
            "monitor_cable_hold",
            "queue_port_offset",
            "check_at_port_offset",
            "queue_align_and_insert",
            "check_at_port_insert",
            "tick_align_and_insert",
            "queue_release_gripper",
            "check_gripper_released",
            "queue_return_home",
            "check_at_home",
        }
        self.assertEqual(expected - self.functions.keys(), set())

    def test_no_fixed_joint_weld(self) -> None:
        self.assertNotIn("FixedJoint", self.source)
        self.assertNotIn("_attach_cable_head_to_gripper", self.source)

    def test_align_insert_uses_feature_cache_helpers(self) -> None:
        for needle in (
            "world_features_for_head",
            "world_features_for_jack",
            "evaluate_alignment",
            "clamp_nudge",
            "insert_target_tip",
            "mating_gap_along_axis",
            "INSERT_STEP_M",
            "MATING_TOUCH_GAP_M",
            "ALIGN_INSERT_MAX_FRAMES",
        ):
            self.assertIn(needle, self.source)

    def test_align_insert_interleaves_nudges_and_micro_steps(self) -> None:
        start = self.source.find("def tick_align_and_insert")
        self.assertGreaterEqual(start, 0, "tick_align_and_insert is missing")
        end = self.source.find("\ndef ", start + 1)
        body = self.source[start:] if end < 0 else self.source[start:end]
        self.assertLess(body.index("evaluate_alignment"), body.index("residual.passed"))
        self.assertLess(body.index("residual.passed"), body.index("clamp_nudge"))
        self.assertLess(body.index("MATING_TOUCH_GAP_M"), body.index("insert_target_tip"))
        self.assertIn("monitor_cable_hold(context)", body)

    def test_port_offset_uses_feature_mating_center_plus_world_x(self) -> None:
        start = self.source.find("def queue_port_offset")
        self.assertGreaterEqual(start, 0, "queue_port_offset is missing")
        end = self.source.find("\ndef ", start + 1)
        body = self.source[start:] if end < 0 else self.source[start:end]
        self.assertIn("port.mating_center", body)
        self.assertIn("PORT_APPROACH_X_OFFSET_M", body)
        self.assertIn("[1.0, 0.0, 0.0]", body)

    def test_release_opens_and_home_joint_interpolates(self) -> None:
        self.assertIn('action="open"', self.source)
        self.assertIn("UR5E_HOME_ARM", self.source)
        self.assertIn("add_joint_waypoint", self.source)


class TaskIntelligenceTests(unittest.TestCase):
    def test_tree_covers_align_insert_release_home(self) -> None:
        from behaviour_tree_insertion import BehaviourTreeRuntime, load_task_intelligence

        path = Path(__file__).resolve().parents[1] / "task_intelligence.json"
        payload = load_task_intelligence(path)
        rendered = BehaviourTreeRuntime(payload, {}).render_tree()
        for needle in (
            "Move to observation pose",
            "Grasp and lift",
            "Maneuver to port offset",
            "Align and insert",
            "Release cable",
            "Return home",
        ):
            self.assertIn(needle, rendered)


if __name__ == "__main__":
    unittest.main()
