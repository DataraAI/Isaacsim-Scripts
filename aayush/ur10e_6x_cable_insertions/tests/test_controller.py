from __future__ import annotations

import importlib
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np


class FakeSharedController:
    def __init__(self, *args, **kwargs):
        self.queued = None
        self.next_action = None
        self._joint_interp_warned = False
        self._segment_failed = False
        self._current_command_index = 0
        self._command_queue = []
        self._segment_ready = False
        self._joint_interp_step = 0
        self._joint_interp_steps = 1

    def clear_queue(self) -> None:
        self._command_queue = []
        self._current_command_index = 0
        self._clear_segment_playback()

    def add_cartesian_waypoint(self, position, orientation, **kwargs):
        self.queued = (np.asarray(position), np.asarray(orientation), kwargs)

    def _current_hand_pose(self):
        return np.array([57.83, -135.0, 313.66]), np.array([1.0, 0.0, 0.0, 0.0])

    def _init_joint_interp_segment(self, cmd, current_joint_positions, n_dof):
        self._joint_interp_warned = bool(cmd.get("force_ik_failure"))
        self._joint_interp_steps = max(1, int(cmd.get("joint_steps", 120)))
        self._joint_interp_step = 0

    def _clear_segment_playback(self) -> None:
        self._segment_ready = False
        self._segment_failed = False
        self._joint_interp_warned = False
        self._joint_interp_step = 0
        self._joint_interp_steps = 1

    def _advance_command(self) -> None:
        self._current_command_index += 1
        self._clear_segment_playback()

    def is_done(self) -> bool:
        return self._current_command_index >= len(self._command_queue)

    def forward(self, current_joint_positions):
        if self.is_done():
            return None

        cmd = self._command_queue[self._current_command_index]
        if cmd.get("type") != "cartesian" or not cmd.get("joint_interp"):
            return self.next_action

        if not self._segment_ready:
            self._init_joint_interp_segment(cmd, current_joint_positions, 7)
            self._segment_ready = True

        self._joint_interp_step += 1
        finished = self._joint_interp_step >= self._joint_interp_steps
        if finished:
            self._advance_command()
        return self.next_action


def load_adapter():
    fake_module = types.ModuleType("franka_motion_controller")
    fake_module.FrankaMotionController = FakeSharedController
    with patch.dict(sys.modules, {"franka_motion_controller": fake_module}):
        sys.modules.pop("ur10e_6x_cable_insertions.controller", None)
        try:
            return importlib.import_module("ur10e_6x_cable_insertions.controller")
        except ModuleNotFoundError as exc:
            raise AssertionError("six-arm controller module is missing") from exc


def bt_tick(controller, joint_positions):
    """Mirror behaviour_tree_insertion.isaac_adapters controller tick checks."""
    if controller.has_failed():
        return "FAILURE"
    if controller.is_done():
        return "SUCCESS"
    controller.forward(joint_positions)
    if controller.has_failed():
        return "FAILURE"
    return "RUNNING"


class SixArmControllerTests(unittest.TestCase):
    def test_cartesian_target_is_converted_to_stage_units_once(self):
        module = load_adapter()
        controller = module.SixArmMotionController(meters_per_unit=0.01)
        controller.add_cartesian_waypoint(
            np.array([0.5783, -1.35, 3.1366]),
            np.array([1.0, 0.0, 0.0, 0.0]),
            label="descend",
        )
        np.testing.assert_allclose(controller.queued[0], [57.83, -135.0, 313.66])

    def test_linear_step_is_converted_to_stage_units(self):
        module = load_adapter()
        controller = module.SixArmMotionController(meters_per_unit=0.01)
        controller.add_cartesian_waypoint(
            np.array([0.5783, -1.35, 3.1366]),
            np.array([1.0, 0.0, 0.0, 0.0]),
            linear=True,
            linear_step=0.002,
        )
        self.assertAlmostEqual(controller.queued[2]["linear_step"], 0.2)

    def test_forward_applies_commanded_joint_positions_immediately(self):
        class FakeRobot:
            def __init__(self):
                self.positions = None

            def set_joint_positions(self, positions):
                self.positions = np.asarray(positions, dtype=np.float64)

        class FakeAction:
            joint_positions = [0.1, None, 0.3]

        module = load_adapter()
        controller = module.SixArmMotionController(meters_per_unit=0.01)
        controller._robot = FakeRobot()
        controller._command_queue = [{"type": "gripper"}]
        controller.next_action = FakeAction()

        controller.forward(np.array([0.0, 0.2, 0.0]))

        self.assertIsNotNone(controller._robot.positions)
        np.testing.assert_allclose(controller._robot.positions, [0.1, 0.2, 0.3])

        controller.next_action.joint_positions = [None, 0.4, None]
        controller.forward(np.array([9.0, 9.0, 9.0]))

        np.testing.assert_allclose(controller._robot.positions, [0.1, 0.4, 0.3])

    def test_measured_hand_pose_is_returned_in_metres(self):
        module = load_adapter()
        controller = module.SixArmMotionController(meters_per_unit=0.01)
        position, _ = controller.current_hand_pose_meters()
        np.testing.assert_allclose(position, [0.5783, -1.35, 3.1366])

    def test_joint_interp_ik_failure_marks_controller_failed(self):
        module = load_adapter()
        controller = module.SixArmMotionController(meters_per_unit=0.01)
        controller._command_queue = [{
            "type": "cartesian",
            "label": "descend",
            "joint_interp": True,
            "force_ik_failure": True,
            "joint_steps": 2,
            "max_frames": 400,
            "frames_spent": 0,
        }]

        status = bt_tick(controller, np.zeros(7))

        self.assertEqual(status, "FAILURE")
        self.assertTrue(controller.has_failed())
        self.assertIn("descend", controller.failure_reason())
        self.assertEqual(controller._current_command_index, 0)

        healthy = module.SixArmMotionController(meters_per_unit=0.01)
        healthy._command_queue = [{
            "type": "cartesian",
            "label": "approach",
            "joint_interp": True,
            "joint_steps": 1,
            "max_frames": 400,
            "frames_spent": 0,
        }]
        self.assertEqual(healthy._current_command_index, 0)
        healthy.forward(np.zeros(7))
        self.assertEqual(healthy._current_command_index, 1)

    def test_clear_queue_resets_failure_reason(self):
        module = load_adapter()
        controller = module.SixArmMotionController(meters_per_unit=0.01)
        cmd = {"label": "descend", "force_ik_failure": True}
        controller._init_joint_interp_segment(cmd, np.zeros(7), 7)
        self.assertTrue(controller.has_failed())
        self.assertIn("descend", controller.failure_reason())

        controller.clear_queue()

        self.assertFalse(controller.has_failed())
        self.assertEqual(controller.failure_reason(), "")
