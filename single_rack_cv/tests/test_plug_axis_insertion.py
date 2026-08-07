from __future__ import annotations

import unittest

import numpy as np

from control.plug_axis_insertion import ExplicitInsertionAxisAdapter


class FakeController:
    def __init__(self):
        self.axis_world = None
        self.freeze_calls = 0

    def _freeze_from(self, sample) -> None:
        self.freeze_calls += 1
        self.axis_world = np.asarray(sample["legacy_axis"], dtype=np.float64)

    def first_target(self, sample, start_position, depth):
        self._freeze_from(sample)
        return np.asarray(start_position, dtype=np.float64) + self.axis_world * depth


class ExplicitInsertionAxisAdapterTests(unittest.TestCase):
    def test_requires_axis_before_first_freeze(self):
        controller = FakeController()
        ExplicitInsertionAxisAdapter(controller)
        with self.assertRaises(RuntimeError):
            controller.first_target(
                {"legacy_axis": (0.0, 0.0, 1.0)},
                (1.0, 2.0, 3.0),
                0.005,
            )

    def test_replaces_legacy_orientation_axis_before_first_target(self):
        controller = FakeController()
        adapter = ExplicitInsertionAxisAdapter(controller)
        adapter.set_axis_world((-1.0, 0.0, 0.0))
        target = controller.first_target(
            {"legacy_axis": (0.0, 0.0, 1.0)},
            (1.0, 2.0, 3.0),
            0.005,
        )
        np.testing.assert_allclose(
            target,
            np.array([0.995, 2.0, 3.0]),
        )
        np.testing.assert_allclose(
            controller.axis_world,
            np.array([-1.0, 0.0, 0.0]),
        )
        self.assertEqual(controller.freeze_calls, 1)

    def test_normalizes_axis_and_rejects_invalid_values(self):
        controller = FakeController()
        adapter = ExplicitInsertionAxisAdapter(controller)
        adapter.set_axis_world((-2.0, 0.0, 0.0))
        np.testing.assert_allclose(
            adapter.pending_axis_world,
            np.array([-1.0, 0.0, 0.0]),
        )
        for axis in (
            (0.0, 0.0, 0.0),
            (float("nan"), 0.0, 1.0),
            (1.0, 2.0),
        ):
            with self.subTest(axis=axis):
                with self.assertRaises(ValueError):
                    adapter.set_axis_world(axis)


if __name__ == "__main__":
    unittest.main()
