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
            if spec.side == "negative":
                self.assertIn("/Lower_Left/", spec.port_pack_path)
            else:
                self.assertIn("/Upper_Right/", spec.port_pack_path)

    def test_per_station_jack_override(self) -> None:
        spec = cfg.make_station("negative", "top", jack_id="jack_upper_c5")
        self.assertEqual(spec.jack_id, "jack_upper_c5")
        self.assertTrue(spec.port_contacts_path.endswith("/Group_14345"))

    def test_grasp_z_bias_lowered_for_both_pads(self) -> None:
        self.assertLessEqual(float(cfg.GRASP_TIP_Z_BIAS_M), -0.020)
        self.assertTrue(cfg.GRASP_BOTH_FINGERS_CONTACT)
        self.assertGreater(float(cfg.GRASP_FINGER_CONTACT_MAX_GAP_M), 0.0)
        self.assertTrue(cfg.GRASP_CLAMP_TIP_Y_TO_BLOCK_GAP)
        self.assertTrue(cfg.GRASP_CENTER_PADS_ON_NECK)
        self.assertAlmostEqual(float(cfg.GRASP_TOWARD_NEG_X_FRAC), 0.0)
        self.assertAlmostEqual(float(cfg.GRASP_TIP_Y_STAGE_DELTA), 1.0)

    def test_primitives_wire_both_fingers_contact(self) -> None:
        path = Path(__file__).resolve().parents[1] / "primitives.py"
        source = path.read_text(encoding="utf-8")
        self.assertIn("def both_fingers_contact_neck", source)
        self.assertIn("GRASP_BOTH_FINGERS_CONTACT", source)
        self.assertIn("both_fingers_contact", source)

    def test_grasp_tilt_and_friction_match_1x(self) -> None:
        self.assertAlmostEqual(cfg.GRASP_TILT_FROM_DOWN_DEG, 60.0)
        self.assertAlmostEqual(cfg.FINGER_OPEN_YAW_ABOUT_TOOL_Z_DEG, 90.0)
        self.assertAlmostEqual(cfg.GRASP_FRICTION_STATIC, 3.0)
        self.assertAlmostEqual(cfg.GRASP_FRICTION_DYNAMIC, 3.0)
        self.assertEqual(cfg.GRASP_FRICTION_COMBINE_MODE, "max")
        # Side-keyed YZ lean: NegY −Z→−Y, PosY −Z→+Y.
        tilt = np.deg2rad(cfg.GRASP_TILT_FROM_DOWN_DEG)
        neg = np.array([0.0, -np.sin(tilt), -np.cos(tilt)], dtype=np.float64)
        pos = np.array([0.0, np.sin(tilt), -np.cos(tilt)], dtype=np.float64)
        np.testing.assert_allclose(cfg.grasp_approach_dir("negative"), neg, atol=1e-9)
        np.testing.assert_allclose(cfg.grasp_approach_dir("positive"), pos, atol=1e-9)
        self.assertLess(float(cfg.grasp_approach_dir("negative")[1]), 0.0)
        self.assertGreater(float(cfg.grasp_approach_dir("positive")[1]), 0.0)
        self.assertIsNone(cfg.GRASP_TIP_X_STAGE)

    def test_arm_gains_stiffer_than_soft_defaults(self) -> None:
        self.assertGreaterEqual(cfg.UR5E_LIVE_STIFFNESS_MULTIPLIER, 1.0)
        self.assertGreaterEqual(cfg.UR5E_LIVE_DAMPING_MULTIPLIER, 5.0)
        finger = cfg.ROBOTIQ_DRIVE_PARAMETERS["finger_joint"]
        self.assertGreaterEqual(float(finger[0]), 3.0)
        self.assertGreaterEqual(float(finger[2]), 50.0)
        # Gravity on requires stage-scaled drives (÷ mpu²) like ur10e_6x.
        self.assertFalse(cfg.LINK_DISABLE_GRAVITY)

    def test_scene_scales_angular_drives_for_stage(self) -> None:
        path = Path(__file__).resolve().parents[1] / "scene.py"
        source = path.read_text(encoding="utf-8")
        self.assertIn("angular_drive_value_for_stage", source)

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
        # Hover demo targets Network cable root XY (not head39 bbox) in primitives.
        path = Path(__file__).resolve().parents[1] / "primitives.py"
        source = path.read_text(encoding="utf-8")
        self.assertIn("def hover_tip_above_cable", source)
        self.assertIn("hover_z_stage_for", source)
        self.assertIn("cable_world_translate_meters", source)

    def test_end_effector_prefers_ur5e_gripper_mount(self) -> None:
        path = Path(__file__).resolve().parents[1] / "scene.py"
        source = path.read_text(encoding="utf-8")
        self.assertIn("def resolve_gripper_prim_path", source)
        self.assertIn("def resolve_end_effector_path", source)
        grip = source[
            source.find("def resolve_gripper_prim_path") : source.find(
                "\ndef ", source.find("def resolve_gripper_prim_path") + 1
            )
        ]
        ee = source[
            source.find("def resolve_end_effector_path") : source.find(
                "\ndef ", source.find("def resolve_end_effector_path") + 1
            )
        ]
        self.assertIn("Gripper/Robotiq_2F_85", grip)
        self.assertNotIn('find_descendant(stage, str(gripper.GetPath()), "base_link")', grip)
        self.assertIn("wrist_3_link", ee)
        self.assertNotIn('find_descendant(stage, str(gripper.GetPath()), "base_link")', ee)
        attach = source[
            source.find("def _attach_manipulator") : source.find(
                "\ndef ", source.find("def _attach_manipulator") + 1
            )
        ]
        # ParallelGripper must use wrist EE (rigid link), not Robotiq Xform.
        self.assertIn("end_effector_prim_path=ee_path", attach)
        self.assertNotIn("end_effector_prim_path=gripper_path", attach)

    def test_scene_uses_lula_wrist_fallback_like_1x(self) -> None:
        path = Path(__file__).resolve().parents[1] / "scene.py"
        source = path.read_text(encoding="utf-8")
        start = source.find("def _make_motion_controller")
        self.assertGreater(start, 0)
        end = source.find("\ndef ", start + 1)
        body = source[start:end]
        self.assertIn("UR5E_EE_FRAME", body)
        self.assertIn("UR5E_EE_FRAME_FALLBACK", body)
        self.assertIn("apply_live_arm_gains", source)
        self.assertIn("angular_drive_value_for_stage", source)
        config = (Path(__file__).resolve().parents[1] / "config.py").read_text()
        self.assertIn("OBSERVE_ORIENTATION = lula_orientation_from_tool", config)
        self.assertIn("GRASP_ORIENTATION = lula_orientation_from_tool", config)

    def test_scene_freezes_inactive_robots_when_station_filtered(self) -> None:
        path = Path(__file__).resolve().parents[1] / "scene.py"
        source = path.read_text(encoding="utf-8")
        self.assertIn("def freeze_inactive_ur5e_robots", source)
        self.assertIn("freeze_inactive_ur5e_robots(stage, selected)", source)
        freeze_start = source.find("def freeze_inactive_ur5e_robots")
        freeze_end = source.find("\ndef ", freeze_start + 1)
        body = source[freeze_start:freeze_end]
        # Idle arms are muted quietly; only --station robots should log.
        self.assertIn("quiet=True", body)
        self.assertIn("MakeInvisible", body)
        self.assertIn("CreateArticulationEnabledAttr(True)", body)


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

    def test_exports_hover_and_orient_primitives(self) -> None:
        expected = {"queue_move", "queue_orient_tilt", "hand_from_tip", "hover_tip_above_cable"}
        self.assertEqual(expected - self.functions.keys(), set())

    def test_orient_uses_grasp_orientation_at_hover_tip(self) -> None:
        start = self.source.find("def queue_orient_tilt")
        self.assertGreaterEqual(start, 0)
        end = self.source.find("\ndef ", start + 1)
        body = self.source[start:] if end < 0 else self.source[start:end]
        self.assertIn("GRASP_TOOL_ORIENTATION", body)
        self.assertIn("GRASP_ORIENTATION", body)
        self.assertIn("hover_tip", body)
        self.assertIn("hand_from_tip", body)
        self.assertIn("ORIENT_JOINT_STEPS", body)

    def test_no_fixed_joint_weld(self) -> None:
        self.assertNotIn("FixedJoint", self.source)
        self.assertNotIn("_attach_cable_head_to_gripper", self.source)


class MainSourceTests(unittest.TestCase):
    def test_registry_wires_hover_and_orient(self) -> None:
        root = Path(__file__).resolve().parents[1]
        main_source = (root / "main.py").read_text(encoding="utf-8")
        self.assertIn("queue_orient_tilt", main_source)
        self.assertIn('"orient_gripper": controller_primitive(queue_orient_tilt)', main_source)
        self.assertIn('"navigate_to_workspace": controller_primitive(queue_move)', main_source)


class InsertReachabilityTests(unittest.TestCase):
    def test_seat_tip_math_matches_rigid_grasp(self) -> None:
        """tip_seat = port_mc − R @ p_local (standoff 0) places crystal on port."""

        tip_now = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        tool = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)  # identity
        crystal = np.array([0.0, 0.0, 0.02], dtype=np.float64)
        port = np.array([-0.15, -1.0, 3.8], dtype=np.float64)
        r = cfg._quat_to_rot_matrix(tool)
        p_local = r.T @ (crystal - tip_now)
        desired = port.copy()  # standoff 0
        tip_seat = desired - (r @ p_local)
        np.testing.assert_allclose(tip_seat, port - np.array([0.0, 0.0, 0.02]), atol=1e-9)
        tip_off = (port + np.array([0.08, 0.0, 0.0])) - (r @ p_local)
        self.assertGreater(float(tip_off[0]), float(tip_seat[0]))

    def test_primitives_wire_insert_reach_gate(self) -> None:
        path = Path(__file__).resolve().parents[1] / "primitives.py"
        source = path.read_text(encoding="utf-8")
        self.assertIn("def _ensure_insert_reachability", source)
        self.assertIn("INSERT_REACH_CHECK", source)
        self.assertIn("insert-reach-reconfig", source)
        self.assertIn("_begin_align_translate_after_reach", source)
        self.assertIn("def check_at_port_offset", source)
        cfg_src = (Path(__file__).resolve().parents[1] / "config.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("INSERT_REACH_RECONFIG_DELTAS_DEG", cfg_src)
        self.assertIn("MANEUVER_AT_OFFSET_TIP_TOL_M", cfg_src)
        main_src = (Path(__file__).resolve().parents[1] / "main.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("validate=check_at_port_offset", main_src)
        self.assertIn("_segment_failed", source)


class TaskIntelligenceTests(unittest.TestCase):
    def test_tree_covers_hover_then_orient(self) -> None:
        from behaviour_tree_insertion import BehaviourTreeRuntime, load_task_intelligence

        path = Path(__file__).resolve().parents[1] / "task_intelligence.json"
        payload = load_task_intelligence(path)
        rendered = BehaviourTreeRuntime(payload, {}).render_tree()
        for needle in (
            "Move to cable hover",
            "Orient + tilt gripper",
        ):
            self.assertIn(needle, rendered)
        self.assertIn("at_workspace", str(payload))
        self.assertIn("gripper_oriented", str(payload))


if __name__ == "__main__":
    unittest.main()
