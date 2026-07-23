from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np

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
        self.marker = "delegated"

    @property
    def dof_properties(self):
        raise AssertionError("deprecated CUDA-unsafe property was accessed")


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
