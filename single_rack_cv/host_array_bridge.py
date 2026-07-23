#!/usr/bin/env python3
"""Explicit host conversion for CUDA/Warp/NumPy values passed to CPU-only APIs."""

from __future__ import annotations

import numpy as np


def to_numpy_cpu(value, *, shape: tuple[int, ...], label: str) -> np.ndarray:
    """Return one finite float64 NumPy array, copying device tensors to host first."""

    current = value
    detach = getattr(current, "detach", None)
    if callable(detach):
        current = detach()

    cpu = getattr(current, "cpu", None)
    if callable(cpu):
        current = cpu()

    numpy_method = getattr(current, "numpy", None)
    if callable(numpy_method):
        current = numpy_method()

    array = np.asarray(current, dtype=np.float64)
    if array.shape != shape:
        raise ValueError(
            f"{label} must have shape {shape}, got {array.shape}"
        )
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label} must contain only finite values")
    return array.copy()
