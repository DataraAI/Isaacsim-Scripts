#!/usr/bin/env python3
"""CUDA-safe articulation adapter for legacy NumPy-only consumers."""

from __future__ import annotations

import numpy as np
import torch

from robot.host_array_bridge import to_numpy_cpu


class HostSafeDofPropertiesArticulation:
    """Bridge legacy NumPy finger setup to a CUDA-backed articulation view."""

    def __init__(self, articulation) -> None:
        self._articulation = articulation

    def __getattr__(self, name):
        return getattr(self._articulation, name)

    @property
    def _view(self):
        view = getattr(self._articulation, "_articulation_view", None)
        if view is None:
            raise RuntimeError("Articulation view is unavailable")
        return view

    def _float_tensor(self, values):
        return torch.as_tensor(
            values,
            dtype=torch.float32,
            device=self._articulation._device,
        )

    def _index_tensor(self, values):
        if values is None:
            return None
        return torch.as_tensor(
            values,
            dtype=torch.int64,
            device=self._articulation._device,
        )

    def _batched_positions(self, values):
        tensor = self._float_tensor(values)
        if tensor.ndim != 1:
            raise ValueError(
                "Finger positions must be one-dimensional, "
                f"got {tuple(tensor.shape)}"
            )
        return tensor.unsqueeze(0)

    def set_joint_positions(self, positions, joint_indices=None):
        """Write immediate positions through the GPU-capable articulation view."""

        return self._view.set_joint_positions(
            self._batched_positions(positions),
            joint_indices=self._index_tensor(joint_indices),
        )

    def set_joint_position_targets(self, positions, joint_indices=None):
        """Write PD targets through the GPU-capable articulation view."""

        return self._view.set_joint_position_targets(
            self._batched_positions(positions),
            joint_indices=self._index_tensor(joint_indices),
        )

    @property
    def dof_properties(self) -> dict[str, np.ndarray]:
        """Return only the lower/upper fields needed by cable finger setup."""

        raw_limits = self._view.get_dof_limits()
        raw_shape = getattr(raw_limits, "shape", None)
        if raw_shape is None:
            raw_shape = np.asarray(raw_limits).shape
        shape = tuple(int(value) for value in raw_shape)
        limits = to_numpy_cpu(
            raw_limits,
            shape=shape,
            label="articulation DOF limits",
        )
        if limits.ndim != 3 or limits.shape[0] != 1 or limits.shape[2] != 2:
            raise RuntimeError(
                "Unsupported articulation DOF limit layout: "
                f"expected (1, dof_count, 2), got {limits.shape}"
            )

        single_articulation_limits = limits[0]
        return {
            "lower": single_articulation_limits[:, 0].copy(),
            "upper": single_articulation_limits[:, 1].copy(),
        }
