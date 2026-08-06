from __future__ import annotations

import math
import unittest

import numpy as np

from insertion import (
    InsertionLimits,
    InsertionPhase,
    InsertionSample,
    InsertionStage,
    PartialInsertionController,
)
from insertion_target_trim import TrimmedConsecutivePoseInsertionController


def limits() -> InsertionLimits:
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


_HORIZONTAL_INSERTION_ORIENTATION_WXYZ = np.array(
    [math.sqrt(0.5), 0.0, -math.sqrt(0.5), 0.0],
    dtype=np.float64,
)


def sample(
    frame: int,
    position=(0.0, 0.0, 0.0),
    *,
    target_error_m: float = 0.0,
    orientation_wxyz=(1.0, 0.0, 0.0, 0.0),
) -> InsertionSample:
    return InsertionSample(
        frame_index=frame,
        alignment_complete=True,
        actual_position_m=np.asarray(position, dtype=np.float64),
        actual_orientation_wxyz=np.asarray(
            orientation_wxyz,
            dtype=np.float64,
        ),
        target_error_m=target_error_m,
        mount_tip_error_m=0.0,
        mount_axis_error_deg=0.0,
        fixed_joint_valid=True,
        attachment_preserved=True,
    )


def collect_commands(
    controller,
    *,
    orientation_wxyz=(1.0, 0.0, 0.0, 0.0),
) -> list:
    event = controller.update(
        sample(0, orientation_wxyz=orientation_wxyz)
    )
    commands = [event.command]
    frame = 0

    while event.kind != "complete":
        command = commands[-1]
        for _ in range(6):
            frame += 1
            event = controller.update(
                sample(
                    frame,
                    command.target_position_m,
                    orientation_wxyz=orientation_wxyz,
                )
            )
        if event.command is not None:
            commands.append(event.command)

    return commands


class TwoStageInsertionTests(unittest.TestCase):
    def test_exact_stage_boundaries_and_final_port_depth(self):
        controller = PartialInsertionController(limits())
        commands = collect_commands(controller)

        self.assertEqual(controller.limits.total_step_count, 48)
        self.assertEqual(len(commands), 48)

        self.assertIs(commands[0].stage, InsertionStage.COARSE_APPROACH)
        self.assertAlmostEqual(commands[0].commanded_depth_m, 0.005)

        self.assertIs(commands[7].stage, InsertionStage.COARSE_APPROACH)
        self.assertAlmostEqual(commands[7].commanded_depth_m, 0.040)
        self.assertAlmostEqual(commands[7].commanded_port_depth_m, -0.010)

        self.assertIs(commands[8].stage, InsertionStage.FINE_INSERTION)
        self.assertAlmostEqual(commands[8].commanded_depth_m, 0.0405)

        self.assertAlmostEqual(commands[27].commanded_depth_m, 0.050)
        self.assertAlmostEqual(commands[27].commanded_port_depth_m, 0.0)

        self.assertAlmostEqual(commands[47].commanded_depth_m, 0.060)
        self.assertAlmostEqual(commands[47].commanded_port_depth_m, 0.010)
        self.assertIs(controller.phase, InsertionPhase.COMPLETE)

    def test_configured_world_trim_shifts_every_target_without_changing_depths(self):
        baseline = PartialInsertionController(limits())
        baseline_commands = collect_commands(
            baseline,
            orientation_wxyz=_HORIZONTAL_INSERTION_ORIENTATION_WXYZ,
        )

        trim_world_m = np.array([0.0, -0.00030, -0.00045])
        trimmed = TrimmedConsecutivePoseInsertionController(
            limits(),
            target_offset_world_m=trim_world_m,
        )
        trimmed_commands = collect_commands(
            trimmed,
            orientation_wxyz=_HORIZONTAL_INSERTION_ORIENTATION_WXYZ,
        )

        self.assertEqual(len(trimmed_commands), 48)
        self.assertGreater(
            float(np.linalg.norm(trim_world_m)),
            trimmed.limits.max_lateral_drift_m,
        )

        for baseline_command, trimmed_command in zip(
            baseline_commands,
            trimmed_commands,
        ):
            np.testing.assert_allclose(
                trimmed_command.target_position_m
                - baseline_command.target_position_m,
                trim_world_m,
                atol=1.0e-12,
                rtol=0.0,
            )
            self.assertEqual(
                trimmed_command.stage,
                baseline_command.stage,
            )
            self.assertAlmostEqual(
                trimmed_command.commanded_depth_m,
                baseline_command.commanded_depth_m,
            )
            self.assertAlmostEqual(
                trimmed_command.commanded_port_depth_m,
                baseline_command.commanded_port_depth_m,
            )

        self.assertAlmostEqual(
            trimmed_commands[-1].commanded_port_depth_m,
            0.010,
        )

    def test_calibrated_line_is_zero_drift_and_real_deviation_still_aborts(self):
        trim_world_m = np.array([0.0, -0.00030, -0.00045])
        controller = TrimmedConsecutivePoseInsertionController(
            limits(),
            target_offset_world_m=trim_world_m,
        )
        started = controller.update(
            sample(
                0,
                orientation_wxyz=_HORIZONTAL_INSERTION_ORIENTATION_WXYZ,
            )
        )
        command = started.command
        self.assertIsNotNone(command)

        on_line = controller.update(
            sample(
                1,
                command.target_position_m,
                orientation_wxyz=_HORIZONTAL_INSERTION_ORIENTATION_WXYZ,
            )
        )
        self.assertEqual(on_line.kind, "waiting_for_settle")
        self.assertIsNotNone(on_line.metrics)
        self.assertLess(on_line.metrics.lateral_drift_m, 1.0e-12)

        off_line_position = command.target_position_m + np.array(
            [0.0, 0.00051, 0.0],
            dtype=np.float64,
        )
        aborted = controller.update(
            sample(
                2,
                off_line_position,
                orientation_wxyz=_HORIZONTAL_INSERTION_ORIENTATION_WXYZ,
            )
        )
        self.assertEqual(aborted.kind, "aborted")
        self.assertIn("lateral drift exceeded limit", aborted.reason)

    def test_first_shifted_command_can_converge_without_transient_abort(self):
        trim_world_m = np.array([0.0, -0.00030, -0.00045])
        controller = TrimmedConsecutivePoseInsertionController(
            limits(),
            target_offset_world_m=trim_world_m,
        )
        started = controller.update(
            sample(
                0,
                orientation_wxyz=_HORIZONTAL_INSERTION_ORIENTATION_WXYZ,
            )
        )
        command = started.command
        self.assertIsNotNone(command)

        in_flight = controller.update(
            sample(
                1,
                position=(0.0, 0.0, 0.0),
                target_error_m=float(
                    np.linalg.norm(command.target_position_m)
                ),
                orientation_wxyz=_HORIZONTAL_INSERTION_ORIENTATION_WXYZ,
            )
        )
        self.assertEqual(in_flight.kind, "waiting_for_settle")
        self.assertIsNone(in_flight.reason)
        self.assertEqual(controller.phase, InsertionPhase.ADVANCING)


if __name__ == "__main__":
    unittest.main()
