#!/usr/bin/env python3
"""Install an explicit frozen insertion axis on the existing controller."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

_EPS = 1.0e-12


def _normalized_axis(value) -> np.ndarray:
    axis = np.asarray(value, dtype=np.float64).reshape(-1)
    if axis.shape != (3,):
        raise ValueError(f"axis_world must have shape (3,), got {axis.shape}")
    if not np.all(np.isfinite(axis)):
        raise ValueError("axis_world must contain only finite values")
    norm = float(np.linalg.norm(axis))
    if norm <= _EPS:
        raise ValueError("axis_world cannot have zero length")
    return axis / norm


class ExplicitInsertionAxisAdapter:
    """
    Replace the controller's legacy orientation-derived axis at freeze time.

    The existing controller still owns all command generation, settle counting,
    drift checks, timeouts, aborts, and terminal holds. This adapter changes only
    the one assumption that insertion travel must follow ToolCenter local +Z.
    """

    def __init__(self, controller: Any) -> None:
        original = getattr(controller, "_freeze_from", None)
        if original is None or not callable(original):
            raise TypeError("controller must expose callable _freeze_from(sample)")

        self.controller = controller
        self._original_freeze: Callable[[Any], None] = original
        self._pending_axis_world: np.ndarray | None = None

        def freeze_with_explicit_axis(sample: Any) -> None:
            if self._pending_axis_world is None:
                raise RuntimeError(
                    "explicit plug insertion axis was not supplied before freeze"
                )
            self._original_freeze(sample)
            controller.axis_world = self._pending_axis_world.copy()

        controller._freeze_from = freeze_with_explicit_axis

    @property
    def pending_axis_world(self) -> np.ndarray | None:
        if self._pending_axis_world is None:
            return None
        return self._pending_axis_world.copy()

    def set_axis_world(self, axis_world) -> None:
        self._pending_axis_world = _normalized_axis(axis_world)
