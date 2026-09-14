# aayush/ur5e_6x_cable_insertions/tests/test_runtime.py
from __future__ import annotations

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
