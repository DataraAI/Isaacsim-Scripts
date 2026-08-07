#!/usr/bin/env python3
"""Small reusable stereo-ray geometry helpers."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class StereoCamera(Protocol):
    @property
    def camera_center_world_m(self) -> np.ndarray: ...

    def pixel_to_world_ray(
        self,
        pixel_uv: np.ndarray | tuple[float, float],
    ) -> tuple[np.ndarray, np.ndarray]: ...

    def project_world(self, point_world_m: np.ndarray) -> np.ndarray: ...


def unit_vector(value: np.ndarray, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(vector))
    if not np.all(np.isfinite(vector)) or norm <= 1.0e-12:
        raise ValueError(f"{name} must be finite and nonzero.")
    return vector / norm


def triangulate_pixel_pair(
    left_uv: np.ndarray,
    right_uv: np.ndarray,
    left_camera: StereoCamera,
    right_camera: StereoCamera,
) -> tuple[np.ndarray, float]:
    """Return midpoint of the closest points on two camera rays and ray gap."""
    left_origin, left_direction = left_camera.pixel_to_world_ray(left_uv)
    right_origin, right_direction = right_camera.pixel_to_world_ray(right_uv)
    left_origin = np.asarray(left_origin, dtype=np.float64).reshape(3)
    right_origin = np.asarray(right_origin, dtype=np.float64).reshape(3)
    left_direction = unit_vector(left_direction, "left stereo ray")
    right_direction = unit_vector(right_direction, "right stereo ray")

    system = np.column_stack((left_direction, -right_direction))
    values, _, rank, _ = np.linalg.lstsq(
        system,
        right_origin - left_origin,
        rcond=None,
    )
    if rank < 2:
        raise RuntimeError("Stereo rays are parallel or numerically singular.")
    left_distance, right_distance = map(float, values)
    if left_distance <= 0.0 or right_distance <= 0.0:
        raise RuntimeError("Triangulated point lies behind a camera.")

    left_point = left_origin + left_distance * left_direction
    right_point = right_origin + right_distance * right_direction
    gap = float(np.linalg.norm(left_point - right_point))
    return (left_point + right_point) / 2.0, gap
