import unittest

import numpy as np

from grasp_control import resolve_tool_orientation


class ResolveToolOrientationTests(unittest.TestCase):
    def test_active_grasp_uses_stored_orientation(self) -> None:
        marker_orientation = np.array([0.14, 0.0, 0.99, 0.0])
        grasp_orientation = np.array([0.50, 0.0, 0.866, 0.0])

        result = resolve_tool_orientation(
            marker_orientation,
            grasp_orientation,
            grasp_active=True,
        )

        expected = grasp_orientation / np.linalg.norm(grasp_orientation)
        np.testing.assert_allclose(result, expected)

    def test_inactive_grasp_uses_marker_orientation(self) -> None:
        marker_orientation = np.array([2.0, 0.0, 0.0, 0.0])

        result = resolve_tool_orientation(
            marker_orientation,
            None,
            grasp_active=False,
        )

        np.testing.assert_allclose(result, [1.0, 0.0, 0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
