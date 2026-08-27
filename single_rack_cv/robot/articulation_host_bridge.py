#!/usr/bin/env python3
"""Backend-adaptive articulation adapter for finger setup.

The articulation view locks its tensor backend at construction
(``_backend`` / ``_backend_utils``). Merged pickup+insertion often leaves
that view numpy-native; standalone warmup leaves it torch-native. Callers
must emit arrays that match the view, not SimulationManager's current flag.
"""

from __future__ import annotations

import numpy as np
import torch

from robot.host_array_bridge import to_numpy_cpu


class HostSafeDofPropertiesArticulation:
    """Bridge finger setup to whatever backend the articulation view locked in."""

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

    def _locked_backend(self) -> str:
        """Return the backend the view was constructed with.

        Prefer ``_view._backend`` (what ``set_joint_positions`` /
        ``_backend_utils`` actually use). Fall back to the wrapped
        articulation's ``_backend``. Do not read
        ``SimulationManager.get_backend()`` — that flag can be flipped
        after the view already locked a different utils module.
        """

        view = self._view
        backend = getattr(view, "_backend", None)
        if backend is None:
            backend = getattr(self._articulation, "_backend", None)
        if backend is None:
            raise RuntimeError(
                "Articulation view has no locked _backend; cannot choose "
                "numpy vs torch finger command arrays"
            )
        return str(backend)

    def _float_batch(self, values):
        array = np.asarray(values, dtype=np.float32)
        if array.ndim != 1:
            raise ValueError(
                "Finger positions must be one-dimensional, "
                f"got {tuple(array.shape)}"
            )
        if self._locked_backend() == "torch":
            return torch.as_tensor(
                array,
                dtype=torch.float32,
                device=self._articulation._device,
            ).unsqueeze(0)
        return array[np.newaxis, :]

    def _index_batch(self, values):
        if values is None:
            return None
        if self._locked_backend() == "torch":
            return torch.as_tensor(
                values,
                dtype=torch.int64,
                device=self._articulation._device,
            )
        return np.asarray(values, dtype=np.int64)

    def set_joint_positions(self, positions, joint_indices=None):
        """Write immediate positions using arrays matching the view backend."""

        return self._view.set_joint_positions(
            self._float_batch(positions),
            joint_indices=self._index_batch(joint_indices),
        )

    def set_joint_position_targets(self, positions, joint_indices=None):
        """Write PD targets using arrays matching the view backend."""

        return self._view.set_joint_position_targets(
            self._float_batch(positions),
            joint_indices=self._index_batch(joint_indices),
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
