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
        self._joint_interp_warned = False
        self._segment_failed = False
        self._current_command_index = 0
        self._command_queue = []

    def add_cartesian_waypoint(self, position, orientation, **kwargs):
        self.queued = (np.asarray(position), np.asarray(orientation), kwargs)

    def _current_hand_pose(self):
        return np.array([57.83, -135.0, 313.66]), np.array([1.0, 0.0, 0.0, 0.0])

    def _init_joint_interp_segment(self, cmd, current_joint_positions, n_dof):
        self._joint_interp_warned = bool(cmd.get("force_ik_failure"))


def load_adapter():
    fake_module = types.ModuleType("franka_motion_controller")
    fake_module.FrankaMotionController = FakeSharedController
    with patch.dict(sys.modules, {"franka_motion_controller": fake_module}):
        sys.modules.pop("ur10e_6x_cable_insertions.controller", None)
        try:
            return importlib.import_module("ur10e_6x_cable_insertions.controller")
        except ModuleNotFoundError as exc:
            raise AssertionError("six-arm controller module is missing") from exc


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

    def test_measured_hand_pose_is_returned_in_metres(self):
        module = load_adapter()
        controller = module.SixArmMotionController(meters_per_unit=0.01)
        position, _ = controller.current_hand_pose_meters()
        np.testing.assert_allclose(position, [0.5783, -1.35, 3.1366])

    def test_joint_interp_ik_failure_marks_controller_failed(self):
        module = load_adapter()
        controller = module.SixArmMotionController(meters_per_unit=0.01)
        cmd = {"label": "descend", "force_ik_failure": True}
        controller._init_joint_interp_segment(cmd, np.zeros(7), 7)
        self.assertTrue(controller.has_failed())
        self.assertIn("descend", controller.failure_reason())
        self.assertEqual(controller._current_command_index, 0)
