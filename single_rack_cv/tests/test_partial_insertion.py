from __future__ import annotations

import math
import unittest

import numpy as np

from insertion import (
    InsertionLimits,
    InsertionPhase,
    InsertionSample,
    PartialInsertionController,
    decompose_axis_motion,
    quaternion_angular_error_deg,
)


def make_limits() -> InsertionLimits:
    return InsertionLimits(
        total_depth_m=0.010,
        step_size_m=0.0005,
        settle_tolerance_m=0.0003,
        required_settled_frames=6,
        step_timeout_frames=120,
        max_lateral_drift_m=0.0005,
        max_orientation_error_deg=1.0,
        max_mount_tip_error_m=0.0005,
        max_mount_axis_error_deg=1.0,
    )


def sample(
    *,
    frame_index: int = 0,
    alignment_complete: bool = True,
    position=(0.0, 0.0, 0.0),
    orientation=(1.0, 0.0, 0.0, 0.0),
    target_error_m: float = 0.0,
    mount_tip_error_m: float = 0.0,
    mount_axis_error_deg: float = 0.0,
    fixed_joint_valid: bool = True,
    attachment_preserved: bool = True,
) -> InsertionSample:
    return InsertionSample(
        frame_index=frame_index,
        alignment_complete=alignment_complete,
        actual_position_m=np.asarray(position, dtype=np.float64),
        actual_orientation_wxyz=np.asarray(orientation, dtype=np.float64),
        target_error_m=target_error_m,
        mount_tip_error_m=mount_tip_error_m,
        mount_axis_error_deg=mount_axis_error_deg,
        fixed_joint_valid=fixed_joint_valid,
        attachment_preserved=attachment_preserved,
    )


class PartialInsertionGeometryTests(unittest.TestCase):
    def test_axis_decomposition_reports_axial_and_lateral_motion(self):
        axial, lateral = decompose_axis_motion(
            start_position_m=np.array([1.0, 2.0, 3.0]),
            actual_position_m=np.array([1.010, 2.0003, 3.0004]),
            axis_world=np.array([1.0, 0.0, 0.0]),
        )
        self.assertAlmostEqual(axial, 0.010, places=12)
        self.assertAlmostEqual(lateral, 0.0005, places=12)

    def test_quaternion_error_is_sign_invariant(self):
        reference = np.array([1.0, 0.0, 0.0, 0.0])
        self.assertEqual(quaternion_angular_error_deg(reference, -reference), 0.0)


class PartialInsertionControllerTests(unittest.TestCase):
    def test_no_command_before_visual_alignment_completion(self):
        controller = PartialInsertionController(make_limits())
        event = controller.update(sample(alignment_complete=False))
        self.assertIs(controller.phase, InsertionPhase.WAITING_FOR_ALIGNMENT)
        self.assertIsNone(event.command)
        self.assertIsNone(controller.last_command)

    def test_first_command_freezes_pose_and_moves_half_millimeter_on_local_plus_z(self):
        controller = PartialInsertionController(make_limits())
        start = np.array([0.7, -0.2, 1.3])
        event = controller.update(sample(frame_index=10, position=start))
        self.assertIs(controller.phase, InsertionPhase.ADVANCING)
        self.assertEqual(event.kind, "started")
        self.assertIsNotNone(event.command)
        self.assertEqual(event.command.step_index, 1)
        self.assertAlmostEqual(event.command.commanded_depth_m, 0.0005, places=12)
        np.testing.assert_allclose(
            event.command.target_position_m,
            start + np.array([0.0, 0.0, 0.0005]),
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            event.command.target_orientation_wxyz,
            np.array([1.0, 0.0, 0.0, 0.0]),
            atol=1.0e-12,
        )

    def test_twenty_settled_steps_complete_at_exactly_ten_millimeters(self):
        controller = PartialInsertionController(make_limits())
        frame = 0
        event = controller.update(sample(frame_index=frame))
        command = event.command
        self.assertIsNotNone(command)

        for expected_step in range(1, 21):
            self.assertEqual(command.step_index, expected_step)
            expected_depth = expected_step * 0.0005
            self.assertAlmostEqual(command.commanded_depth_m, expected_depth, places=12)
            for _ in range(6):
                frame += 1
                event = controller.update(
                    sample(
                        frame_index=frame,
                        position=command.target_position_m,
                        orientation=command.target_orientation_wxyz,
                        target_error_m=0.0,
                    )
                )
            if expected_step < 20:
                self.assertEqual(event.kind, "step_settled")
                self.assertIsNotNone(event.command)
                command = event.command
            else:
                self.assertEqual(event.kind, "complete")
                self.assertIsNone(event.command)

        self.assertIs(controller.phase, InsertionPhase.COMPLETE)
        self.assertAlmostEqual(controller.commanded_depth_m, 0.010, places=12)
        terminal = controller.update(
            sample(
                frame_index=frame + 1,
                position=np.array([0.0, 0.0, 0.010]),
            )
        )
        self.assertIsNone(terminal.command)
        self.assertIs(controller.phase, InsertionPhase.COMPLETE)

    def test_lateral_drift_above_half_millimeter_aborts(self):
        controller = PartialInsertionController(make_limits())
        controller.update(sample(frame_index=0))
        event = controller.update(
            sample(
                frame_index=1,
                position=(0.000500001, 0.0, 0.0005),
                target_error_m=0.0,
            )
        )
        self.assertIs(controller.phase, InsertionPhase.ABORTED)
        self.assertEqual(event.kind, "aborted")
        self.assertIn("lateral drift", event.reason)
        self.assertIsNone(event.command)

    def test_orientation_error_above_one_degree_aborts(self):
        controller = PartialInsertionController(make_limits())
        controller.update(sample(frame_index=0))
        angle = math.radians(1.01)
        rotated = (
            math.cos(angle / 2.0),
            math.sin(angle / 2.0),
            0.0,
            0.0,
        )
        event = controller.update(
            sample(
                frame_index=1,
                position=(0.0, 0.0, 0.0005),
                orientation=rotated,
            )
        )
        self.assertIs(controller.phase, InsertionPhase.ABORTED)
        self.assertIn("orientation error", event.reason)

    def test_mount_and_structural_failures_abort(self):
        cases = (
            ({"mount_tip_error_m": 0.000500001}, "plug-tip mount error"),
            ({"mount_axis_error_deg": 1.000001}, "plug-axis error"),
            ({"fixed_joint_valid": False}, "fixed joint"),
            ({"attachment_preserved": False}, "attachment"),
        )
        for overrides, reason in cases:
            with self.subTest(reason=reason):
                controller = PartialInsertionController(make_limits())
                controller.update(sample(frame_index=0))
                event = controller.update(
                    sample(
                        frame_index=1,
                        position=(0.0, 0.0, 0.0005),
                        **overrides,
                    )
                )
                self.assertIs(controller.phase, InsertionPhase.ABORTED)
                self.assertIn(reason, event.reason)
                self.assertIsNone(event.command)

    def test_step_timeout_aborts_without_issuing_another_target(self):
        controller = PartialInsertionController(make_limits())
        first = controller.update(sample(frame_index=0))
        self.assertIsNotNone(first.command)
        event = controller.update(
            sample(
                frame_index=120,
                position=(0.0, 0.0, 0.0),
                target_error_m=0.00031,
            )
        )
        self.assertIs(controller.phase, InsertionPhase.ABORTED)
        self.assertIn("timeout", event.reason)
        self.assertIsNone(event.command)

    def test_explicit_ik_abort_is_terminal(self):
        controller = PartialInsertionController(make_limits())
        controller.update(sample(frame_index=0))
        event = controller.abort(
            "Lula IK rejected insertion target",
            sample(frame_index=1, position=(0.0, 0.0, 0.0)),
        )
        self.assertIs(controller.phase, InsertionPhase.ABORTED)
        self.assertIn("Lula IK", event.reason)
        self.assertIsNone(event.command)
        held = controller.update(sample(frame_index=2))
        self.assertIsNone(held.command)


if __name__ == "__main__":
    unittest.main()
