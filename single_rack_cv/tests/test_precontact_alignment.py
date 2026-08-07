from __future__ import annotations

import unittest

import numpy as np

from control.insertion import (
    InsertionLimits,
    InsertionPhase,
    InsertionSample,
    InsertionStage,
    PartialInsertionController,
)
from control.precontact_alignment import (
    PrecontactAlignmentPolicy,
    PrecontactInsertionLimits,
    build_precontact_limits,
)


def base_limits() -> InsertionLimits:
    return InsertionLimits(
        total_depth_m=0.060,
        step_size_m=0.0005,
        coarse_approach_depth_m=0.040,
        coarse_step_size_m=0.005,
        opening_depth_m=0.050,
        settle_tolerance_m=0.0003,
        required_settled_frames=6,
        step_timeout_frames=120,
        max_lateral_drift_m=0.0005,
        max_orientation_error_deg=1.0,
        max_mount_tip_error_m=0.0005,
        max_mount_axis_error_deg=1.0,
    )


def sample(frame: int, position=(0.0, 0.0, 0.0)) -> InsertionSample:
    return InsertionSample(
        frame_index=frame,
        alignment_complete=True,
        actual_position_m=np.asarray(position, dtype=np.float64),
        actual_orientation_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        target_error_m=0.0,
        mount_tip_error_m=0.0,
        mount_axis_error_deg=0.0,
        fixed_joint_valid=True,
        attachment_preserved=True,
    )


class PrecontactAlignmentTests(unittest.TestCase):
    def test_exact_2mm_hold_has_24_nonpenetrating_commands(self):
        limits = build_precontact_limits(
            base_limits(),
            PrecontactAlignmentPolicy(hold_offset_m=0.002),
        )
        controller = PartialInsertionController(limits)
        event = controller.update(sample(0))
        commands = [event.command]
        frame = 0

        while event.kind != "complete":
            command = commands[-1]
            for _ in range(limits.required_settled_frames):
                frame += 1
                event = controller.update(
                    sample(frame, command.target_position_m)
                )
            if event.command is not None:
                commands.append(event.command)

        self.assertIsInstance(limits, PrecontactInsertionLimits)
        self.assertIsInstance(limits, InsertionLimits)
        self.assertEqual(limits.total_step_count, 24)
        self.assertEqual(len(commands), 24)
        self.assertIs(commands[0].stage, InsertionStage.COARSE_APPROACH)
        self.assertAlmostEqual(commands[7].commanded_depth_m, 0.040)
        self.assertAlmostEqual(commands[7].commanded_port_depth_m, -0.010)
        self.assertIs(commands[8].stage, InsertionStage.FINE_INSERTION)
        self.assertAlmostEqual(commands[-1].commanded_depth_m, 0.048)
        self.assertAlmostEqual(commands[-1].commanded_port_depth_m, -0.002)
        self.assertTrue(
            all(command.commanded_port_depth_m < 0.0 for command in commands)
        )
        self.assertIs(controller.phase, InsertionPhase.COMPLETE)

    def test_policy_changes_only_total_depth(self):
        original = base_limits()
        capped = build_precontact_limits(
            original,
            PrecontactAlignmentPolicy(hold_offset_m=0.002),
        )
        self.assertAlmostEqual(capped.total_depth_m, 0.048)
        self.assertAlmostEqual(capped.opening_depth_m, original.opening_depth_m)
        self.assertAlmostEqual(
            capped.coarse_approach_depth_m,
            original.coarse_approach_depth_m,
        )
        self.assertAlmostEqual(capped.step_size_m, original.step_size_m)
        self.assertAlmostEqual(
            capped.coarse_step_size_m,
            original.coarse_step_size_m,
        )
        self.assertAlmostEqual(
            capped.max_lateral_drift_m,
            original.max_lateral_drift_m,
        )
        self.assertAlmostEqual(
            capped.max_orientation_error_deg,
            original.max_orientation_error_deg,
        )

    def test_standard_limits_still_reject_terminal_before_opening(self):
        with self.assertRaisesRegex(
            ValueError,
            "opening_depth_m cannot exceed total_depth_m",
        ):
            InsertionLimits(
                total_depth_m=0.048,
                step_size_m=0.0005,
                coarse_approach_depth_m=0.040,
                coarse_step_size_m=0.005,
                opening_depth_m=0.050,
                settle_tolerance_m=0.0003,
                required_settled_frames=6,
                step_timeout_frames=120,
                max_lateral_drift_m=0.0005,
                max_orientation_error_deg=1.0,
                max_mount_tip_error_m=0.0005,
                max_mount_axis_error_deg=1.0,
            )

    def test_rejects_nonpositive_hold_offset(self):
        for value in (0.0, -0.001):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "positive"):
                    PrecontactAlignmentPolicy(hold_offset_m=value)

    def test_rejects_hold_offset_at_or_beyond_opening_depth(self):
        with self.assertRaisesRegex(ValueError, "smaller"):
            build_precontact_limits(
                base_limits(),
                PrecontactAlignmentPolicy(hold_offset_m=0.050),
            )

    def test_rejects_cap_that_does_not_extend_beyond_coarse_stage(self):
        with self.assertRaisesRegex(ValueError, "coarse"):
            build_precontact_limits(
                base_limits(),
                PrecontactAlignmentPolicy(hold_offset_m=0.010),
            )


if __name__ == "__main__":
    unittest.main()
