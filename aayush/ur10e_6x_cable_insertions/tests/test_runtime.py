"""Host-side tests for the 6× UR10e cable-insertion package (no Isaac Sim)."""

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

from behaviour_tree_insertion import BehaviourTreeRuntime, load_task_intelligence
from ur10e_6x_cable_insertions import config as cfg


class StationTableTests(unittest.TestCase):
    def test_six_stations_cover_left_right_and_three_heights(self) -> None:
        self.assertEqual(len(cfg.STATIONS), 6)
        sides = {spec.side for spec in cfg.STATIONS}
        heights = {spec.height for spec in cfg.STATIONS}
        self.assertEqual(sides, {"negative", "positive"})
        self.assertEqual(heights, {"top", "middle", "bottom"})

    def test_negative_is_upper_left_positive_is_upper_right(self) -> None:
        for spec in cfg.STATIONS:
            if spec.side == "negative":
                self.assertEqual(spec.robot_loc, "Upper_Left")
            else:
                self.assertEqual(spec.robot_loc, "Upper_Right")

    def test_height_maps_to_grid_option(self) -> None:
        by_height = {spec.height: spec.grid_option for spec in cfg.STATIONS}
        self.assertEqual(by_height["top"], cfg.GRID_TOP)
        self.assertEqual(by_height["middle"], cfg.GRID_MIDDLE)
        self.assertEqual(by_height["bottom"], cfg.GRID_BOTTOM)

    def test_port_contacts_follow_user_template(self) -> None:
        spec = cfg.make_station("negative", "top")
        self.assertEqual(
            spec.port_contacts_path,
            "/World/Network_Switches/AS4610_Ethernet_Row_Top_1x_Grid/Upper_Left/"
            "AS4610_inst/AS4610_01/Switch/Net_12_Pack_no_LED_Component_03/"
            "RJ45_Group01/CopperContacts/Group_14343",
        )
        spec = cfg.make_station("positive", "bottom")
        self.assertEqual(
            spec.port_contacts_path,
            "/World/Network_Switches/AS4610_01_1x_Grid/Upper_Right/"
            "AS4610_inst/AS4610_01/Switch/Net_12_Pack_no_LED_Component_03/"
            "RJ45_Group01/CopperContacts/Group_14343",
        )

    def test_robot_and_cable_prim_paths(self) -> None:
        spec = cfg.make_station("negative", "middle")
        self.assertEqual(spec.robot_prim_path, "/World/Robots/UR10e_NegativeY_Middle")
        self.assertEqual(spec.cable_root_path, "/World/NetworkCables/Cable_NegativeY_Middle")
        self.assertTrue(spec.grasp_part_path.endswith("/E_crystal_head1_45/E_part006_44"))

    def test_debug_markers_are_offset_insert_and_vias(self) -> None:
        names = cfg.debug_marker_names()
        self.assertIn("Offset", names)
        self.assertIn("Insert", names)
        self.assertTrue(any(n.startswith("Via_") for n in names))
        for frac in cfg.PORT_APPROACH_VIA_FRACTIONS:
            self.assertIn(f"Via_{int(round(frac * 100)):02d}", names)
        spec = cfg.STATIONS[0]
        self.assertEqual(spec.debug_marker_root, f"/World/DebugPortMarkers/{spec.station_id}")


class SceneWiringTests(unittest.TestCase):
    def test_scene_uses_six_arm_controller_and_raw_stage_base_pose(self) -> None:
        scene_path = Path(__file__).resolve().parents[1] / "scene.py"
        source = scene_path.read_text(encoding="utf-8")
        self.assertIn(
            "from ur10e_6x_cable_insertions.controller import SixArmMotionController",
            source,
        )
        self.assertIn("SixArmMotionController(", source)
        self.assertIn("meters_per_unit=meters_per_unit(stage)", source)
        self.assertIn("kinematics.set_robot_base_pose(base_position, base_orientation)", source)
        self.assertNotIn(
            "kinematics.set_robot_base_pose(pose_to_meters(base_position, stage), base_orientation)",
            source,
        )

    def test_scene_enforces_one_arm_ur10e_drive_parameters(self) -> None:
        scene_path = Path(__file__).resolve().parents[1] / "scene.py"
        source = scene_path.read_text(encoding="utf-8")
        self.assertIn("apply_ur10e_drive_parameters(", source)
        self.assertIn("robot.set_world_pose(", source)
        self.assertIn("angular_drive_value_for_stage", source)
        self.assertNotIn("**cfg.ROBOTIQ_DRIVE_PARAMETERS", source)
        self.assertEqual(
            cfg.UR10E_ARM_DRIVE_PARAMETERS["shoulder_pan_joint"],
            (3271.49169921875, 13.085969924926758, 330.0),
        )
        self.assertEqual(
            cfg.UR10E_ARM_DRIVE_PARAMETERS["wrist_3_joint"],
            (1268.18603515625, 5.070000171661377, 56.0),
        )


class GraspWiringTests(unittest.TestCase):
    @staticmethod
    def _function_body(source: str, name: str) -> str:
        tree = ast.parse(source)
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == name:
                return ast.get_source_segment(source, node) or ""
        return ""

    def test_grasp_records_initial_position_and_uses_metre_pose(self) -> None:
        primitives_path = Path(__file__).resolve().parents[1] / "primitives.py"
        source = primitives_path.read_text(encoding="utf-8")
        self.assertIn('context.services["initial_grasp_point"] = center.copy()', source)
        self.assertIn("physical_grasp_is_valid(", source)
        self.assertNotIn("lifted = float(center[2]) >= block_top + 0.04", source)

    def test_check_physical_grasp_calls_motion_controller_metre_pose(self) -> None:
        primitives_path = Path(__file__).resolve().parents[1] / "primitives.py"
        source = primitives_path.read_text(encoding="utf-8")
        grasp_check = self._function_body(source, "check_physical_grasp")
        self.assertTrue(grasp_check, "check_physical_grasp not found")
        self.assertIn(
            'context.services["motion_controller"].current_hand_pose_meters()',
            grasp_check,
        )

    def test_check_physical_grasp_uses_grasp_tip_with_x_offset(self) -> None:
        primitives_path = Path(__file__).resolve().parents[1] / "primitives.py"
        source = primitives_path.read_text(encoding="utf-8")
        grasp_check = self._function_body(source, "check_physical_grasp")
        hold_check = self._function_body(source, "cable_still_in_gripper")
        self.assertIn("grasp_tip_from_part(", grasp_check)
        self.assertIn("grasp_tip_from_part(", hold_check)
        self.assertIn("current_part=grasp_tip", grasp_check)
        self.assertNotIn("grasp_joint_created", grasp_check)

    def test_regular_grasp_requires_finger_contact_without_fixed_joint(self) -> None:
        primitives_path = Path(__file__).resolve().parents[1] / "primitives.py"
        source = primitives_path.read_text(encoding="utf-8")
        hold_check = self._function_body(source, "monitor_cable_hold")
        cable_check = self._function_body(source, "cable_still_in_gripper")
        self.assertNotIn("FixedJoint", source)
        self.assertNotIn("_attach_cable_head_to_gripper", source)
        self.assertNotIn("squeeze-hold", self._function_body(source, "queue_grasp"))
        self.assertIn('cmd.get("action") == "close"', hold_check)
        self.assertIn("closed_enough", cable_check)
        self.assertIn("and closed_enough", cable_check)

    def test_grasp_uses_smooth_joint_interpolation_toward_positive_x(self) -> None:
        primitives_path = Path(__file__).resolve().parents[1] / "primitives.py"
        source = primitives_path.read_text(encoding="utf-8")
        queue_grasp = self._function_body(source, "queue_grasp")
        self.assertGreater(float(cfg.GRASP_APPROACH_DIR[0]), 0.0)
        self.assertIn("tip_hover = tip_grasp - approach", queue_grasp)
        hover = queue_grasp[queue_grasp.index("hover-tilted") :]
        hover = hover[: hover.index("add_gripper_command")]
        self.assertIn("joint_interp=True", hover)
        self.assertIn("joint_steps=240", hover)
        descend = queue_grasp[queue_grasp.index("descend-tilted") :]
        descend = descend[: descend.index("add_gripper_command")]
        self.assertIn("joint_interp=True", descend)
        self.assertIn("joint_steps=200", descend)
        lift = queue_grasp[queue_grasp.index('label=f"{context.step.name}: lift"') :]
        self.assertIn("joint_interp=True", lift)
        self.assertIn("joint_steps=900", lift)

    def test_grasp_places_head_inside_fingertips(self) -> None:
        self.assertAlmostEqual(cfg.TOOL_OFFSET_M, 0.16)
        self.assertAlmostEqual(cfg.ROBOTIQ_CLOSED_RAD, float(np.deg2rad(70.0)))
        self.assertAlmostEqual(cfg.ROBOTIQ_CONTACT_RAD, float(np.deg2rad(12.0)))
        self.assertEqual(cfg.GRASP_FRICTION_STATIC, 0.8)
        self.assertEqual(cfg.GRASP_FRICTION_DYNAMIC, 0.8)


class CableInsertionTreeTests(unittest.TestCase):
    def test_task_intelligence_renders_grasp_lift_tree(self) -> None:
        json_path = Path(__file__).resolve().parents[1] / "task_intelligence.json"
        payload = load_task_intelligence(json_path)
        tree = BehaviourTreeRuntime(payload, {}, logger=lambda _: None)
        rendered = tree.render_tree()
        self.assertIn("Grasp, lift, and approach ethernet port", rendered)
        self.assertIn("Detect E_part006_44", rendered)
        self.assertIn("Grasp and lift E_part006_44", rendered)
        self.assertIn("Move held cable to port offset then insert", rendered)


if __name__ == "__main__":
    unittest.main()
