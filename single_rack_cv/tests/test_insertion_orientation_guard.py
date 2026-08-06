#!/usr/bin/env python3

import unittest

from insertion import orientation_abort_is_due


class InsertionOrientationGuardTests(unittest.TestCase):
    def test_logged_unsettled_noncontact_transient_is_allowed(self):
        self.assertFalse(
            orientation_abort_is_due(
                target_error_m=0.001862,
                settle_tolerance_m=0.0003,
                actual_port_depth_m=-0.046861,
            )
        )

    def test_settled_noncontact_pose_enforces_limit(self):
        self.assertTrue(
            orientation_abort_is_due(
                target_error_m=0.0002,
                settle_tolerance_m=0.0003,
                actual_port_depth_m=-0.046861,
            )
        )

    def test_opening_plane_enforces_limit_while_unsettled(self):
        self.assertTrue(
            orientation_abort_is_due(
                target_error_m=0.001862,
                settle_tolerance_m=0.0003,
                actual_port_depth_m=0.0,
            )
        )

    def test_inside_opening_enforces_limit_while_unsettled(self):
        self.assertTrue(
            orientation_abort_is_due(
                target_error_m=0.001862,
                settle_tolerance_m=0.0003,
                actual_port_depth_m=0.001,
            )
        )


if __name__ == "__main__":
    unittest.main()
