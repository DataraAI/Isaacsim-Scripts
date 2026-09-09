"""Host-side smoke tests for the cable-insertion behaviour tree JSON."""

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

from behaviour_tree_insertion import BehaviourTreeRuntime, load_task_intelligence
from ur10e_1x_cable_insertion import config as cfg


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

    def test_port_approach_joint_steps_slow_from_via_60_through_95(self) -> None:
        select_steps = getattr(
            cfg,
            "port_approach_joint_steps",
            lambda _source, _destination, *, is_final: 160 if is_final else 120,
        )

        self.assertEqual(select_steps(0.82, 0.95, is_final=False), 480)
        self.assertEqual(select_steps(0.60, 0.82, is_final=False), 480)
        self.assertEqual(select_steps(0.95, 1.0, is_final=True), 160)

    def test_observation_pose_uses_head39_xy_and_safe_clearance(self) -> None:
        compute_pose = getattr(
            cfg,
            "observation_hand_from_head39",
            lambda _center, _support_y, _block_top: np.zeros(3),
        )

        pose = compute_pose([0.277, -0.598, 1.131], -0.600, 1.125)

        np.testing.assert_allclose(pose, [0.277, -0.600, 1.451])
        np.testing.assert_allclose(cfg.OBSERVE_ORIENTATION, cfg.GRASP_ORIENTATION)

    def test_joint_interpolation_ik_failure_holds_command_queue(self) -> None:
        controller_path = (
            REPO_ROOT / "detailedInsertion" / "cable" / "franka_motion_controller.py"
        )
        source = controller_path.read_text()
        execution = source.split('if cmd.get("joint_interp"):', 2)[2].split(
            'if cmd.get("linear"):', 1
        )[0]
        initialization = source.split(
            "def _init_joint_interp_segment", 1
        )[1].split("def _joint_interp_action", 1)[0]

        self.assertIn("if self._segment_failed:", execution)
        self.assertIn("self._segment_failed = True", initialization)


if __name__ == "__main__":
    unittest.main()
