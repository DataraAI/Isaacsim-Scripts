#!/usr/bin/env python3
"""Cavity-anchored 2D front-bezel sampling inside a qualified YOLOE ROI."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config import FrontRimConfig

SIDE_NAMES = ("top", "right", "bottom", "left")
BEZEL_OUTWARD_OFFSET_PX = 3.0


@dataclass(frozen=True)
class RimLine2D:
    point_uv: np.ndarray
    direction_uv: np.ndarray
    normal_uv: np.ndarray
    support_uv: np.ndarray
    inlier_uv: np.ndarray

    def __post_init__(self) -> None:
        point = np.asarray(self.point_uv, dtype=np.float64).reshape(2)
        direction = np.asarray(self.direction_uv, dtype=np.float64).reshape(2)
        direction_norm = float(np.linalg.norm(direction))
        if direction_norm <= 1.0e-12:
            raise ValueError("Rim line direction must be nonzero.")
        direction = direction / direction_norm
        normal = np.array([-direction[1], direction[0]], dtype=np.float64)
        support = np.asarray(self.support_uv, dtype=np.float64).reshape(-1, 2)
        inliers = np.asarray(self.inlier_uv, dtype=np.float64).reshape(-1, 2)

        object.__setattr__(self, "point_uv", point)
        object.__setattr__(self, "direction_uv", direction)
        object.__setattr__(self, "normal_uv", normal)
        object.__setattr__(self, "support_uv", support)
        object.__setattr__(self, "inlier_uv", inliers)


@dataclass(frozen=True)
class FrontRim2D:
    roi_uv: tuple[int, int, int, int]
    corners_uv: np.ndarray
    center_uv: tuple[float, float]
    side_lines: tuple[RimLine2D, RimLine2D, RimLine2D, RimLine2D]
    side_samples_uv: np.ndarray

    def __post_init__(self) -> None:
        corners = np.asarray(self.corners_uv, dtype=np.float64)
        samples = np.asarray(self.side_samples_uv, dtype=np.float64)
        if corners.shape != (4, 2):
            raise ValueError(f"corners_uv must be (4,2), got {corners.shape}.")
        if samples.ndim != 3 or samples.shape[0] != 4 or samples.shape[2] != 2:
            raise ValueError(
                "side_samples_uv must have shape (4,samples_per_side,2)."
            )
        if len(self.side_lines) != 4:
            raise ValueError("side_lines must contain top/right/bottom/left.")
        object.__setattr__(self, "corners_uv", corners.copy())
        object.__setattr__(self, "side_samples_uv", samples.copy())


def expand_detection_roi(
    bbox_xywh: tuple[int, int, int, int],
    image_shape_hw: tuple[int, int],
    cfg: FrontRimConfig,
) -> tuple[int, int, int, int]:
    x, y, width, height = map(int, bbox_xywh)
    image_height, image_width = map(int, image_shape_hw)
    if width <= 0 or height <= 0:
        raise ValueError("Detection must have positive width and height.")
    margin = max(
        int(cfg.roi_min_margin_px),
        int(round(cfg.roi_expand_ratio * max(width, height))),
    )
    return (
        max(0, x - margin),
        max(0, y - margin),
        min(image_width, x + width + margin),
        min(image_height, y + height + margin),
    )


def _sample_segment(
    start_uv: np.ndarray,
    end_uv: np.ndarray,
    cfg: FrontRimConfig,
) -> np.ndarray:
    trim = float(cfg.sample_corner_trim_fraction)
    if not 0.0 <= trim < 0.5:
        raise ValueError("sample_corner_trim_fraction must be in [0,0.5).")
    values = np.linspace(
        trim,
        1.0 - trim,
        cfg.samples_per_side,
        dtype=np.float64,
    )
    return start_uv[None, :] + values[:, None] * (
        end_uv - start_uv
    )[None, :]


def _line_from_segment(
    start_uv: np.ndarray,
    end_uv: np.ndarray,
    support_uv: np.ndarray,
) -> RimLine2D:
    return RimLine2D(
        point_uv=start_uv,
        direction_uv=end_uv - start_uv,
        normal_uv=np.zeros(2, dtype=np.float64),
        support_uv=support_uv,
        inlier_uv=support_uv,
    )


def _offset_side_samples_outward(
    side_samples_uv: np.ndarray,
    center_uv: np.ndarray,
    offset_px: float,
) -> np.ndarray:
    samples = np.asarray(side_samples_uv, dtype=np.float64)
    center = np.asarray(center_uv, dtype=np.float64).reshape(2)
    shifted = samples.copy()
    for side_index in range(4):
        side_center = np.mean(samples[side_index], axis=0)
        outward = side_center - center
        norm = float(np.linalg.norm(outward))
        if norm <= 1.0e-12:
            raise RuntimeError(
                f"Front-bezel side {SIDE_NAMES[side_index]} has no outward direction."
            )
        shifted[side_index] += float(offset_px) * outward / norm
    return shifted


def extract_front_rim(
    rgb: np.ndarray,
    bbox_xywh: tuple[int, int, int, int],
    cfg: FrontRimConfig,
    center_uv: tuple[float, float] | np.ndarray | None = None,
) -> FrontRim2D:
    """Build a front-bezel support ring from the qualified cavity box.

    The previous implementation searched broad Sobel bands and frequently locked
    onto internal connector seams. The qualified cavity detector already gives a
    stable opening location. This function uses that location only to place
    samples outside the opening on the visible front bezel; it does not infer
    depth or read validation ground truth.
    """
    image = np.asarray(rgb)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 RGB image, got {image.shape}.")

    x, y, width, height = map(float, bbox_xywh)
    if width <= 1.0 or height <= 1.0:
        raise RuntimeError("Detection is too small for front-bezel sampling.")

    roi = expand_detection_roi(
        bbox_xywh=tuple(map(int, bbox_xywh)),
        image_shape_hw=image.shape[:2],
        cfg=cfg,
    )
    corners = np.array(
        [
            [x, y],
            [x + width, y],
            [x + width, y + height],
            [x, y + height],
        ],
        dtype=np.float64,
    )
    if center_uv is None:
        center = np.array(
            [x + 0.5 * width, y + 0.5 * height],
            dtype=np.float64,
        )
    else:
        center = np.asarray(center_uv, dtype=np.float64).reshape(2)

    inner_samples = np.stack(
        [
            _sample_segment(corners[0], corners[1], cfg),
            _sample_segment(corners[1], corners[2], cfg),
            _sample_segment(corners[3], corners[2], cfg),
            _sample_segment(corners[0], corners[3], cfg),
        ],
        axis=0,
    )
    bezel_samples = _offset_side_samples_outward(
        inner_samples,
        center,
        BEZEL_OUTWARD_OFFSET_PX,
    )

    image_height, image_width = image.shape[:2]
    if (
        np.any(bezel_samples[:, :, 0] < 0.0)
        or np.any(bezel_samples[:, :, 0] > image_width - 1.0)
        or np.any(bezel_samples[:, :, 1] < 0.0)
        or np.any(bezel_samples[:, :, 1] > image_height - 1.0)
    ):
        raise RuntimeError("Front-bezel samples leave the image.")

    lines = (
        _line_from_segment(corners[0], corners[1], inner_samples[0]),
        _line_from_segment(corners[1], corners[2], inner_samples[1]),
        _line_from_segment(corners[3], corners[2], inner_samples[2]),
        _line_from_segment(corners[0], corners[3], inner_samples[3]),
    )
    return FrontRim2D(
        roi_uv=roi,
        corners_uv=corners,
        center_uv=(float(center[0]), float(center[1])),
        side_lines=lines,
        side_samples_uv=bezel_samples,
    )
