from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np

from host_array_bridge import to_numpy_cpu

ROOT = Path(__file__).resolve().parents[1]


class _FakeCudaTensor:
    def __init__(self, values):
        self._values = np.asarray(values, dtype=np.float32)
        self.events: list[str] = []
        self.on_cpu = False

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


class HostArrayBridgeTests(unittest.TestCase):
    def test_device_tensor_is_detached_copied_to_cpu_and_converted(self):
        value = _FakeCudaTensor([1.0, 2.0, 3.0])
        result = to_numpy_cpu(value, shape=(3,), label="position")
        np.testing.assert_allclose(result, [1.0, 2.0, 3.0])
        self.assertEqual(value.events, ["detach", "cpu", "numpy"])
        self.assertEqual(result.dtype, np.float64)

    def test_shape_and_finite_values_are_enforced(self):
        with self.assertRaisesRegex(ValueError, "shape"):
            to_numpy_cpu([1.0, 2.0], shape=(3,), label="position")
        with self.assertRaisesRegex(ValueError, "finite"):
            to_numpy_cpu(
                [1.0, float("nan"), 3.0],
                shape=(3,),
                label="position",
            )

    def test_cable_runtime_wraps_and_restores_lula_base_pose_boundary(self):
        source = (ROOT / "cable_runtime.py").read_text(encoding="utf-8")
        self.assertIn("def _create_ik(self, assets_root", source)
        self.assertIn("to_numpy_cpu", source)
        self.assertIn("original_set_robot_base_pose", source)
        self.assertIn("host_safe_set_robot_base_pose", source)
        self.assertIn("finally:", source)
        self.assertIn("set_robot_base_pose = original_set_robot_base_pose", source)

    def test_real_cuda_tensor_when_available(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch is unavailable")
        if not torch.cuda.is_available():
            self.skipTest("CUDA is unavailable")

        value = torch.tensor([1.0, 0.0, 0.0, 0.0], device="cuda:0")
        result = to_numpy_cpu(value, shape=(4,), label="orientation")
        np.testing.assert_allclose(result, [1.0, 0.0, 0.0, 0.0])
        self.assertIsInstance(result, np.ndarray)


if __name__ == "__main__":
    unittest.main()
