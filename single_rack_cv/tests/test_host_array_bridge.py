from __future__ import annotations

import ast
from pathlib import Path
import unittest

import numpy as np

from host_array_bridge import (
    HostSafeJointSubset,
    HostSafePoseObject,
    install_host_safe_ik_warm_start,
    pose_to_numpy_cpu,
    to_numpy_cpu,
)

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


class _FakePoseObject:
    def __init__(self):
        self.position = _FakeCudaTensor([1.0, 2.0, 3.0])
        self.orientation = _FakeCudaTensor([1.0, 0.0, 0.0, 0.0])
        self.marker = "delegated"

    def get_world_pose(self):
        return self.position, self.orientation


class _FakeJointSubset:
    def __init__(self):
        self.positions = _FakeCudaTensor(
            [0.01, -0.57, 0.0, -1.01, 0.0, -0.02, -0.52]
        )
        self.marker = "subset delegated"

    def get_joint_positions(self):
        return self.positions


class _FakeArticulationKinematicsSolver:
    def __init__(self):
        self._joints_view = _FakeJointSubset()

    def get_joints_subset(self):
        return self._joints_view


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

    def test_pose_pair_is_copied_to_host_and_other_methods_delegate(self):
        value = _FakePoseObject()
        wrapped = HostSafePoseObject(value, label="IK target")

        position, orientation = wrapped.get_world_pose()

        np.testing.assert_allclose(position, [1.0, 2.0, 3.0])
        np.testing.assert_allclose(orientation, [1.0, 0.0, 0.0, 0.0])
        self.assertEqual(value.position.events, ["detach", "cpu", "numpy"])
        self.assertEqual(value.orientation.events, ["detach", "cpu", "numpy"])
        self.assertEqual(wrapped.marker, "delegated")

    def test_pose_pair_shape_validation_names_the_failing_component(self):
        with self.assertRaisesRegex(ValueError, "ToolCenter position"):
            pose_to_numpy_cpu(
                [1.0, 2.0],
                [1.0, 0.0, 0.0, 0.0],
                label="ToolCenter",
            )

    def test_joint_subset_copies_cuda_warm_start_to_host(self):
        subset = _FakeJointSubset()
        wrapped = HostSafeJointSubset(subset, label="Lula IK warm start")

        result = wrapped.get_joint_positions()

        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.dtype, np.float64)
        self.assertEqual(result.shape, (7,))
        np.testing.assert_allclose(
            result,
            [0.01, -0.57, 0.0, -1.01, 0.0, -0.02, -0.52],
        )
        self.assertEqual(subset.positions.events, ["detach", "cpu", "numpy"])
        self.assertEqual(wrapped.marker, "subset delegated")

    def test_ik_solver_internal_joint_subset_is_replaced_once(self):
        solver = _FakeArticulationKinematicsSolver()
        original_subset = solver.get_joints_subset()

        returned = install_host_safe_ik_warm_start(solver)

        self.assertIs(returned, solver)
        self.assertIsInstance(solver.get_joints_subset(), HostSafeJointSubset)
        self.assertIs(solver.get_joints_subset()._wrapped, original_subset)
        np.testing.assert_allclose(
            solver.get_joints_subset().get_joint_positions(),
            [0.01, -0.57, 0.0, -1.01, 0.0, -0.02, -0.52],
        )

    def test_cable_runtime_wraps_all_cuda_pose_sources_once(self):
        source = (ROOT / "cable_runtime.py").read_text(encoding="utf-8")
        self.assertIn("HostSafePoseObject", source)
        self.assertIn("install_host_safe_ik_warm_start", source)
        self.assertIn('label="Franka articulation"', source)
        self.assertIn('label="IK target"', source)
        self.assertIn('label="actual ToolCenter"', source)
        self.assertIn("def _create_ik(self, assets_root", source)
        self.assertIn("original_set_robot_base_pose", source)
        self.assertIn("host_safe_set_robot_base_pose", source)
        self.assertIn("finally:", source)

        tree = ast.parse(source)
        restoration_found = False
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                continue
            target = node.targets[0]
            value = node.value
            if (
                isinstance(target, ast.Attribute)
                and target.attr == "set_robot_base_pose"
                and isinstance(value, ast.Name)
                and value.id == "original_set_robot_base_pose"
            ):
                restoration_found = True
                break
        self.assertTrue(
            restoration_found,
            "LulaKinematicsSolver.set_robot_base_pose must be restored in finally",
        )

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
