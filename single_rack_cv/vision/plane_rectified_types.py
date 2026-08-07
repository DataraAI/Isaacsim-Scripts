#!/usr/bin/env python3
"""Shared types and safety constants for plane-rectified RGB front-lip estimation."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import numpy as np


DEFAULT_RECTIFIED_RESOLUTION_M = 0.00005
DEFAULT_RECTIFIED_PADDING_M = 0.006
MAX_CENTER_DISAGREEMENT_M = 0.0005
MAX_EDGE_REPROJECTION_PX = 1.5
MAX_OPPOSITE_EDGE_ANGLE_DEG = 5.0
MIN_EDGE_SAMPLES = 6


def _unit(value: np.ndarray, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(vector))
    if not np.all(np.isfinite(vector)) or norm <= 1.0e-12:
        raise ValueError(f"{name} must be finite and nonzero.")
    return vector / norm


@dataclass(frozen=True)
class PlaneFrame:
    origin_world_m: np.ndarray
    axis_u_world: np.ndarray
    axis_v_world: np.ndarray
    normal_world: np.ndarray

    def __post_init__(self) -> None:
        origin = np.asarray(self.origin_world_m, dtype=np.float64).reshape(3)
        u = _unit(self.axis_u_world, "plane axis u")
        v = _unit(self.axis_v_world, "plane axis v")
        normal = _unit(self.normal_world, "plane normal")
        if abs(float(u @ v)) > 1.0e-6:
            raise ValueError("Plane axes must be orthogonal.")
        if abs(float(u @ normal)) > 1.0e-6 or abs(float(v @ normal)) > 1.0e-6:
            raise ValueError("Plane axes must lie in the plane.")
        object.__setattr__(self, "origin_world_m", origin.copy())
        object.__setattr__(self, "axis_u_world", u)
        object.__setattr__(self, "axis_v_world", v)
        object.__setattr__(self, "normal_world", normal)

    def metric_to_world(self, point_uv_m: np.ndarray) -> np.ndarray:
        uv = np.asarray(point_uv_m, dtype=np.float64)
        return (
            self.origin_world_m
            + uv[..., 0, None] * self.axis_u_world
            + uv[..., 1, None] * self.axis_v_world
        )

    def world_to_metric(self, point_world_m: np.ndarray) -> np.ndarray:
        point = np.asarray(point_world_m, dtype=np.float64)
        relative = point - self.origin_world_m
        return np.stack(
            (relative @ self.axis_u_world, relative @ self.axis_v_world),
            axis=-1,
        )


@dataclass(frozen=True)
class RectifiedEye:
    rgb: np.ndarray
    mask: np.ndarray
    visible: np.ndarray
    map_u_px: np.ndarray
    map_v_px: np.ndarray
    minimum_uv_m: np.ndarray
    resolution_m: float
    plane_frame: PlaneFrame
    camera: object

    def __post_init__(self) -> None:
        rgb = np.asarray(self.rgb, dtype=np.uint8)
        mask = np.asarray(self.mask, dtype=bool)
        visible = np.asarray(self.visible, dtype=bool)
        map_u = np.asarray(self.map_u_px, dtype=np.float64)
        map_v = np.asarray(self.map_v_px, dtype=np.float64)
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("Rectified RGB must have shape HxWx3.")
        expected = rgb.shape[:2]
        if mask.shape != expected or visible.shape != expected:
            raise ValueError("Rectified masks must match RGB dimensions.")
        if map_u.shape != expected or map_v.shape != expected:
            raise ValueError("Rectified projection maps must match RGB dimensions.")
        resolution = float(self.resolution_m)
        if not math.isfinite(resolution) or resolution <= 0.0:
            raise ValueError("Rectified resolution must be finite and positive.")
        object.__setattr__(self, "rgb", np.ascontiguousarray(rgb))
        object.__setattr__(self, "mask", mask.copy())
        object.__setattr__(self, "visible", visible.copy())
        object.__setattr__(self, "map_u_px", map_u.copy())
        object.__setattr__(self, "map_v_px", map_v.copy())
        object.__setattr__(
            self,
            "minimum_uv_m",
            np.asarray(self.minimum_uv_m, dtype=np.float64).reshape(2).copy(),
        )
        object.__setattr__(self, "resolution_m", resolution)

    def pixel_to_metric(self, pixels_xy: np.ndarray) -> np.ndarray:
        pixels = np.asarray(pixels_xy, dtype=np.float64)
        return self.minimum_uv_m + pixels * self.resolution_m

    def metric_to_pixel(self, points_uv_m: np.ndarray) -> np.ndarray:
        points = np.asarray(points_uv_m, dtype=np.float64)
        return (points - self.minimum_uv_m) / self.resolution_m


@dataclass(frozen=True)
class FrontLipFit:
    left_line: tuple[float, float]
    right_line: tuple[float, float]
    top_line: tuple[float, float]
    bottom_line: tuple[float, float]
    corners_uv_m: np.ndarray
    center_uv_m: np.ndarray
    edge_samples_uv_m: Mapping[str, np.ndarray]
    support_counts: tuple[int, int, int, int]
    residual_px: float
    width_m: float
    height_m: float

    def __post_init__(self) -> None:
        corners = np.asarray(self.corners_uv_m, dtype=np.float64).reshape(4, 2)
        center = np.asarray(self.center_uv_m, dtype=np.float64).reshape(2)
        if not np.all(np.isfinite(corners)) or not np.all(np.isfinite(center)):
            raise ValueError("Front-lip geometry must be finite.")
        samples = {
            str(name): np.asarray(value, dtype=np.float64).reshape(-1, 2).copy()
            for name, value in self.edge_samples_uv_m.items()
        }
        if set(samples) != {"left", "right", "top", "bottom"}:
            raise ValueError("Front-lip samples must contain four named edges.")
        object.__setattr__(self, "corners_uv_m", corners.copy())
        object.__setattr__(self, "center_uv_m", center.copy())
        object.__setattr__(self, "edge_samples_uv_m", samples)
        object.__setattr__(
            self, "support_counts", tuple(int(v) for v in self.support_counts)
        )


@dataclass(frozen=True)
class PlaneRectifiedFrontLipDebug:
    left_rectified_rgb: np.ndarray
    right_rectified_rgb: np.ndarray
    left_overlay: np.ndarray
    right_overlay: np.ndarray
    joint_overlay: np.ndarray
    left_reprojection: np.ndarray
    right_reprojection: np.ndarray


@dataclass(frozen=True)
class PlaneRectifiedFrontLipResult:
    center_world_m: np.ndarray
    left_center_world_m: np.ndarray
    right_center_world_m: np.ndarray
    left_center_uv: np.ndarray
    right_center_uv: np.ndarray
    center_disagreement_m: float
    left_fit: FrontLipFit
    right_fit: FrontLipFit
    joint_fit: FrontLipFit
    plane_frame: PlaneFrame
    debug: PlaneRectifiedFrontLipDebug
