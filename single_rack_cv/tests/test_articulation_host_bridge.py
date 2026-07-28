from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np
import torch

from articulation_host_bridge import HostSafeDofPropertiesArticulation

ROOT = Path(__file__).resolve().parents[1]


class _FakeCudaTensor:
    def __init__(self, values):
        self._values = np.asarray(values, dtype=np.float32)
        self.events: list[str] = []
        self.on_cpu = False

    @property
    def shape(self):
        return self._values.shape

    def detach(self):
        self.events.append("detach")
        return self

    def cpu(self):
        self.events.append("cpu")
        self.on_cpu = True
        return self

    def numpy(self):
        self.events.append("numpy")
        if not self.on_cpu:
            raise RuntimeError("device tensor was not copied to CPU")
        return self._values


class _FakeArticulationView:
    def __init__(self, limits):
        self.limits = limits
        self.position_calls = []
        self.target_calls = []

    def get_dof_limits(self):
        return self.limits

    def set_joint_positions(self, positions, indices=None, joint_indices=None):
        self.position_calls.append((positions, indices, joint_indices))

    def set_joint_position_targets(
        self,
        positions,
        indices=None,
        joint_indices=None,
    ):
        self.target_calls.append((positions, indices, joint_indices))


class _FakeArticulation:
    def __init__(self, limits):
        self._articulation_view = _FakeArticulationView(limits)
        self._device = "cpu"
        self.marker = "delegated"

    @property
    def dof_properties(self):
        raise AssertionError("deprecated CUDA-unsafe property was accessed")

    def set_joint_positions(self, positions, joint_indices=None):
        raise AssertionError("deprecated single-articulation setter was used")


class HostSafeArticulationTests(unittest.TestCase):
    def test_cuda_dof_limits_are_copied_to_host_without_legacy_property(self):
        raw_limits = _FakeCudaTensor(
            [[[-1.0, 1.0], [-0.5, 0.5], [0.0, 0.04], [0.0, 0.04]]]
        )
        articulation = _FakeArticulation(raw_limits)
        wrapped = HostSafeDofPropertiesArticulation(articulation)

        properties = wrapped.dof_properties

        np.testing.assert_allclose(properties["lower"], [-1.0, -0.5, 0.0, 0.0])
        np.testing.assert_allclose(properties["upper"], [1.0, 0.5, 0.04, 0.04])
        self.assertEqual(raw_limits.events, ["detach", "cpu", "numpy"])
        self.assertEqual(wrapped.marker, "delegated")

    def test_numpy_finger_commands_use_batched_articulation_view_tensors(self):
        articulation = _FakeArticulation(
            np.zeros((1, 9, 2), dtype=np.float32)
        )
        wrapped = HostSafeDofPropertiesArticulation(articulation)
        positions = np.asarray([0.02, 0.02], dtype=np.float64)
        indices = np.asarray([7, 8], dtype=np.int32)

        wrapped.set_joint_positions(positions, joint_indices=indices)
        wrapped.set_joint_position_targets(positions, joint_indices=indices)

        immediate_positions, immediate_rows, immediate_indices = (
            articulation._articulation_view.position_calls[0]
        )
        target_positions, target_rows, target_indices = (
            articulation._articulation_view.target_calls[0]
        )
        self.assertEqual(immediate_positions.shape, (1, 2))
        self.assertEqual(target_positions.shape, (1, 2))
        self.assertEqual(immediate_positions.dtype, torch.float32)
        self.assertEqual(immediate_indices.dtype, torch.int64)
        self.assertEqual(target_positions.dtype, torch.float32)
        self.assertEqual(target_indices.dtype, torch.int64)
        self.assertIsNone(immediate_rows)
        self.assertIsNone(target_rows)
        np.testing.assert_allclose(immediate_positions.numpy()[0], positions)
        np.testing.assert_array_equal(immediate_indices.numpy(), indices)
        np.testing.assert_allclose(target_positions.numpy()[0], positions)
        np.testing.assert_array_equal(target_indices.numpy(), indices)

    def test_non_vector_finger_commands_are_rejected(self):
        articulation = _FakeArticulation(
            np.zeros((1, 9, 2), dtype=np.float32)
        )
        wrapped = HostSafeDofPropertiesArticulation(articulation)
        with self.assertRaisesRegex(ValueError, "one-dimensional"):
            wrapped.set_joint_positions(
                np.zeros((1, 2), dtype=np.float32),
                joint_indices=np.asarray([7, 8], dtype=np.int32),
            )

    def test_only_one_articulation_and_two_limit_columns_are_accepted(self):
        for limits in (
            np.zeros((2, 4, 2), dtype=np.float64),
            np.zeros((1, 4, 3), dtype=np.float64),
        ):
            with self.subTest(shape=limits.shape):
                wrapped = HostSafeDofPropertiesArticulation(
                    _FakeArticulation(limits)
                )
                with self.assertRaisesRegex(RuntimeError, "DOF limit layout"):
                    _ = wrapped.dof_properties

    def test_scale_aware_mount_wraps_only_the_finger_configuration_boundary(self):
        source = (ROOT / "scale_aware_cable_mount.py").read_text(encoding="utf-8")
        self.assertIn("HostSafeDofPropertiesArticulation", source)
        self.assertIn(
            "super().configure_fingers(\n"
            "            HostSafeDofPropertiesArticulation(articulation)\n"
            "        )",
            source,
        )


if __name__ == "__main__":
    unittest.main()
