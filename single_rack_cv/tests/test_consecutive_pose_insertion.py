#!/usr/bin/env python3

import math
import unittest

import numpy as np

from insertion import InsertionLimits, InsertionPhase, InsertionSample
from settled_insertion import ConsecutivePoseInsertionController


def _quat_x_deg(angle_deg: float) -> np.ndarray:
    half = math.radians(angle_deg) / 2.0
    return np.array([math.cos(half), math.sin(half), 0.0, 0.0])


def _limits(*, opening_depth_m: float = 0.050) -> InsertionLimits:
    return InsertionLimits(
        total_depth_m=0.060,
        step_size_m=0.005,
        settle_tolerance_m=0.0003,
        required_settled_frames=6,
        step_timeout_frames=120,
        max_lateral_drift_m=0.0005,
        max_orientation_error_deg=1.0,
        max_mount_tip_error_m=0.001,
        max_mount_axis_error_deg=1.0,
        opening_depth_m=opening_depth_m,
    )


def _sample(
    *,
    frame_index: int,
    z_m: float,
    target_error_m: float,
    orientation_deg: float,
    alignment_complete: bool = True,
) -> InsertionSample:
    return InsertionSample(
        frame_index=frame_index,
        alignment_complete=alignment_complete,
        actual_position_m=np.array([0.0, 0.0, z_m]),
        actual_orientation_wxyz=_quat_x_deg(orientation_deg),
        target_error_m=target_error_m,
        mount_tip_error_m=0.0,
        mount_axis_error_deg=0.0,
        fixed_joint_valid=True,
        attachment_preserved=True,
    )


class ConsecutivePoseInsertionTests(unittest.TestCase):
    def test_preopening_orientation_transient_resets_settle_instead_of_aborting(self):
        controller = ConsecutivePoseInsertionController(_limits())

        started = controller.update(
            _sample(
                frame_index=0,
                z_m=0.0,
                target_error_m=0.0,
                orientation_deg=0.0,
            )
        )
        self.assertEqual(started.kind, "started")
        self.assertEqual(controller.phase, InsertionPhase.ADVANCING)

        transient = controller.update(
            _sample(
                frame_index=20,
                z_m=0.004781,
                target_error_m=0.000283,
                orientation_deg=1.195336,
            )
        )

        self.assertEqual(transient.kind, "waiting_for_settle")
        self.assertEqual(controller.phase, InsertionPhase.ADVANCING)
        self.assertEqual(controller.settled_frame_count, 0)
        self.assertIsNone(controller.abort_reason)

    def test_step_requires_six_frames_with_position_and_orientation_valid(self):
        controller = ConsecutivePoseInsertionController(_limits())
        controller.update(
            _sample(
                frame_index=0,
                z_m=0.0,
                target_error_m=0.0,
                orientation_deg=0.0,
            )
        )

        for frame_index in range(1, 6):
            event = controller.update(
                _sample(
                    frame_index=frame_index,
                    z_m=0.005,
                    target_error_m=0.0002,
                    orientation_deg=0.8,
                )
            )
            self.assertEqual(event.kind, "waiting_for_settle")

        settled = controller.update(
            _sample(
                frame_index=6,
                z_m=0.005,
                target_error_m=0.0002,
                orientation_deg=0.8,
            )
        )
        self.assertEqual(settled.kind, "step_settled")
        self.assertEqual(settled.settled_step_index, 1)
        self.assertEqual(settled.command.step_index, 2)

    def test_orientation_over_limit_at_opening_aborts_immediately(self):
        controller = ConsecutivePoseInsertionController(
            _limits(opening_depth_m=0.005)
        )
        controller.update(
            _sample(
                frame_index=0,
                z_m=0.0,
                target_error_m=0.0,
                orientation_deg=0.0,
            )
        )

        aborted = controller.update(
            _sample(
                frame_index=1,
                z_m=0.005,
                target_error_m=0.0002,
                orientation_deg=1.2,
            )
        )

        self.assertEqual(aborted.kind, "aborted")
        self.assertIn("orientation error exceeded limit", aborted.reason)


if __name__ == "__main__":
    unittest.main()
