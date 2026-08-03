#!/usr/bin/env python3

from __future__ import annotations

import unittest

import numpy as np

from insertion import InsertionLimits, InsertionSample, InsertionStage
from settled_insertion import ConsecutivePoseInsertionController


def _limits() -> InsertionLimits:
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
        max_mount_tip_error_m=0.001,
        max_mount_axis_error_deg=1.0,
    )


def _sample(
    *,
    frame_index: int,
    position_m,
    target_error_m: float,
) -> InsertionSample:
    return InsertionSample(
        frame_index=frame_index,
        alignment_complete=True,
        actual_position_m=np.asarray(position_m, dtype=np.float64),
        actual_orientation_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        target_error_m=target_error_m,
        mount_tip_error_m=0.0,
        mount_axis_error_deg=0.0,
        fixed_joint_valid=True,
        attachment_preserved=True,
    )


def _advance_to_first_fine_command(
    controller: ConsecutivePoseInsertionController,
):
    frame = 0
    event = controller.update(
        _sample(
            frame_index=frame,
            position_m=(0.0, 0.0, 0.0),
            target_error_m=0.0,
        )
    )
    command = event.command
    assert command is not None

    for coarse_index in range(8):
        assert command.stage is InsertionStage.COARSE_APPROACH
        for _ in range(6):
            frame += 1
            event = controller.update(
                _sample(
                    frame_index=frame,
                    position_m=command.target_position_m,
                    target_error_m=0.0,
                )
            )
        if coarse_index < 7:
            command = event.command
            assert command is not None

    command = event.command
    assert command is not None
    assert command.stage is InsertionStage.FINE_INSERTION
    return frame, command


class FineInsertionSettlingTests(unittest.TestCase):
    def test_fine_step_does_not_accept_old_point_three_mm_tolerance(self):
        controller = ConsecutivePoseInsertionController(_limits())
        frame, command = _advance_to_first_fine_command(controller)

        for _ in range(10):
            frame += 1
            event = controller.update(
                _sample(
                    frame_index=frame,
                    position_m=command.target_position_m,
                    target_error_m=0.0002,
                )
            )
            self.assertEqual(event.kind, "waiting_for_settle")

        self.assertEqual(controller.settled_frame_count, 0)

    def test_fine_step_requires_ten_quiet_frames_after_arrival(self):
        controller = ConsecutivePoseInsertionController(_limits())
        frame, command = _advance_to_first_fine_command(controller)

        frame += 1
        arrival = controller.update(
            _sample(
                frame_index=frame,
                position_m=command.target_position_m,
                target_error_m=0.00008,
            )
        )
        self.assertEqual(arrival.kind, "waiting_for_settle")
        self.assertEqual(controller.settled_frame_count, 0)

        for _ in range(9):
            frame += 1
            event = controller.update(
                _sample(
                    frame_index=frame,
                    position_m=command.target_position_m,
                    target_error_m=0.00008,
                )
            )
            self.assertEqual(event.kind, "waiting_for_settle")

        frame += 1
        settled = controller.update(
            _sample(
                frame_index=frame,
                position_m=command.target_position_m,
                target_error_m=0.00008,
            )
        )
        self.assertEqual(settled.kind, "step_settled")
        self.assertEqual(settled.settled_step_index, 9)

    def test_fine_step_rejects_motion_through_position_window(self):
        controller = ConsecutivePoseInsertionController(_limits())
        frame, command = _advance_to_first_fine_command(controller)
        target = np.asarray(command.target_position_m, dtype=np.float64)

        for index in range(12):
            frame += 1
            offset_m = 0.00008 if index % 2 == 0 else 0.00004
            position = target.copy()
            position[2] -= offset_m
            event = controller.update(
                _sample(
                    frame_index=frame,
                    position_m=position,
                    target_error_m=offset_m,
                )
            )
            self.assertEqual(event.kind, "waiting_for_settle")

        self.assertEqual(controller.settled_frame_count, 0)


if __name__ == "__main__":
    unittest.main()
