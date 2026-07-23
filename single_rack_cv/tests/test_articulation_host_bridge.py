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

    def get_dof_limits(self):
        return self.limits


class _FakeArticulation:
    def __init__(self, limits):
        self._articulation_view = _FakeArticulationView(limits)
        self._device = "cpu"
        self.marker = "delegated"
        self.position_calls = []
        self.target_calls = []

    @property
    def dof_properties(self):
        raise AssertionError("deprecated CUDA-unsafe property was accessed")

    def set_joint_positions(self, positions, joint_indices=None):
        if not isinstance(positions, torch.Tensor):
            raise AssertionError(f"positions stayed on host: {type(positions)}")
        if not isinstance(joint_indices, torch.Tensor):
            raise AssertionError(
                f"joint indices stayed on host: {type(joint_indices)}"
            )
        self.position_calls.append((positions, joint_indices))

    def set_joint_position_targets(self, positions, joint_indices=None):
        if not isinstance(positions, torch.Tensor):
            raise AssertionError(f"targets stayed on host: {type(positions)}")
        if not isinstance(joint_indices, torch.Tensor):
            raise AssertionError(
                f"joint indices stayed on host: {type(joint_indices)}"
            )
        self.target_calls.append((positions, joint_indices))


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

    def test_numpy_finger_commands_are_forwarded_as_backend_tensors(self):
        articulation = _FakeArticulation(
            np.zeros((1, 9, 2), dtype=np.float32)
        )
        wrapped = HostSafeDofPropertiesArticulation(articulation)
        positions = np.asarray([0.02, 0.02], dtype=np.float64)
        indices = np.asarray([7, 8], dtype=np.int32)

        wrapped.set_joint_positions(positions, joint_indices=indices)
        wrapped.set_joint_position_targets(positions, joint_indices=indices)

        immediate_positions, immediate_indices = articulation.position_calls[0]
        target_positions, target_indices = articulation.target_calls[0]
        self.assertEqual(immediate_positions.dtype, torch.float32)
        self.assertEqual(immediate_indices.dtype, torch.int64)
        self.assertEqual(target_positions.dtype, torch.float32)
        self.assertEqual(target_indices.dtype, torch.int64)
        np.testing.assert_allclose(immediate_positions.numpy(), positions)
        np.testing.assert_array_equal(immediate_indices.numpy(), indices)
        np.testing.assert_allclose(target_positions.numpy(), positions)
        np.testing.assert_array_equal(target_indices.numpy(), indices)

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
