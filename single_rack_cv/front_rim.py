#!/usr/bin/env python3
"""Dense 2D front-rim extraction inside a qualified YOLOE proposal."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from config import FrontRimConfig

SIDE_NAMES = ("top", "right", "bottom", "left")


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


def _robust_fit_line(
    points_uv: np.ndarray,
    side_name: str,
    cfg: FrontRimConfig,
) -> RimLine2D:
    support = np.asarray(points_uv, dtype=np.float64).reshape(-1, 2)
    if support.shape[0] < cfg.min_support_pixels_per_side:
        raise RuntimeError(
            f"{side_name} rim has only {support.shape[0]} support pixels."
        )

    inliers = support.copy()
    point = np.zeros(2, dtype=np.float64)
    direction = np.array([1.0, 0.0], dtype=np.float64)
    for _ in range(cfg.line_fit_iterations):
        vx, vy, x0, y0 = cv2.fitLine(
            inliers.astype(np.float32),
            cv2.DIST_L2,
            0.0,
            0.01,
            0.01,
        ).reshape(4)
        direction = np.array([float(vx), float(vy)], dtype=np.float64)
        direction /= np.linalg.norm(direction)
        point = np.array([float(x0), float(y0)], dtype=np.float64)
        normal = np.array([-direction[1], direction[0]], dtype=np.float64)
        residuals = np.abs((support - point) @ normal)
        median = float(np.median(residuals))
        mad = float(np.median(np.abs(residuals - median)))
        threshold = min(
            cfg.line_max_residual_px,
            max(0.35, median + cfg.line_mad_scale * 1.4826 * mad),
        )
        new_inliers = support[residuals <= threshold]
        if new_inliers.shape[0] < cfg.min_support_pixels_per_side:
            raise RuntimeError(
                f"{side_name} rim line fit retained only "
                f"{new_inliers.shape[0]} inliers."
            )
        if new_inliers.shape == inliers.shape and np.allclose(new_inliers, inliers):
            inliers = new_inliers
            break
        inliers = new_inliers

    return RimLine2D(
        point_uv=point,
        direction_uv=direction,
        normal_uv=np.array([-direction[1], direction[0]], dtype=np.float64),
        support_uv=support,
        inlier_uv=inliers,
    )


def _line_intersection(a: RimLine2D, b: RimLine2D) -> np.ndarray:
    system = np.column_stack((a.direction_uv, -b.direction_uv))
    values, _, rank, _ = np.linalg.lstsq(
        system,
        b.point_uv - a.point_uv,
        rcond=None,
    )
    if rank < 2:
        raise RuntimeError("Adjacent rim lines are parallel.")
    return a.point_uv + float(values[0]) * a.direction_uv


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


def _side_support_masks(
    gx: np.ndarray,
    gy: np.ndarray,
    bbox_local_xywh: tuple[int, int, int, int],
    cfg: FrontRimConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x, y, width, height = bbox_local_xywh
    image_height, image_width = gx.shape
    band_x = max(
        cfg.side_band_min_px,
        int(round(cfg.side_band_fraction * width)),
    )
    band_y = max(
        cfg.side_band_min_px,
        int(round(cfg.side_band_fraction * height)),
    )

    yy, xx = np.mgrid[0:image_height, 0:image_width]
    strong = np.hypot(gx, gy) >= cfg.gradient_min

    top = (
        strong
        & (gy <= -cfg.polarity_min)
        & (np.abs(gy) >= np.abs(gx))
        & (xx >= x - band_x)
        & (xx <= x + width + band_x)
        & (yy >= y - band_y)
        & (yy <= y + band_y)
    )
    right = (
        strong
        & (gx >= cfg.polarity_min)
        & (np.abs(gx) >= np.abs(gy))
        & (yy >= y - band_y)
        & (yy <= y + height + band_y)
        & (xx >= x + width - band_x)
        & (xx <= x + width + band_x)
    )
    bottom = (
        strong
        & (gy >= cfg.polarity_min)
        & (np.abs(gy) >= np.abs(gx))
        & (xx >= x - band_x)
        & (xx <= x + width + band_x)
        & (yy >= y + height - band_y)
        & (yy <= y + height + band_y)
    )
    left = (
        strong
        & (gx <= -cfg.polarity_min)
        & (np.abs(gx) >= np.abs(gy))
        & (yy >= y - band_y)
        & (yy <= y + height + band_y)
        & (xx >= x - band_x)
        & (xx <= x + band_x)
    )
    return top, right, bottom, left


def extract_front_rim(
    rgb: np.ndarray,
    bbox_xywh: tuple[int, int, int, int],
    cfg: FrontRimConfig,
) -> FrontRim2D:
    image = np.asarray(rgb)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 RGB image, got {image.shape}.")

    roi = expand_detection_roi(
        bbox_xywh=bbox_xywh,
        image_shape_hw=image.shape[:2],
        cfg=cfg,
    )
    u0, v0, u1, v1 = roi
    crop = np.ascontiguousarray(image[v0:v1, u0:u1])
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0.0)

    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=cfg.sobel_kernel_px)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=cfg.sobel_kernel_px)

    x, y, width, height = map(int, bbox_xywh)
    local_bbox = (x - u0, y - v0, width, height)
    masks = _side_support_masks(gx, gy, local_bbox, cfg)

    lines: list[RimLine2D] = []
    for side_name, mask in zip(SIDE_NAMES, masks, strict=True):
        rows, columns = np.nonzero(mask)
        local_points = np.column_stack((columns, rows)).astype(np.float64)
        local_line = _robust_fit_line(local_points, side_name, cfg)
        offset = np.array([u0, v0], dtype=np.float64)
        lines.append(
            RimLine2D(
                point_uv=local_line.point_uv + offset,
                direction_uv=local_line.direction_uv,
                normal_uv=local_line.normal_uv,
                support_uv=local_line.support_uv + offset,
                inlier_uv=local_line.inlier_uv + offset,
            )
        )

    top, right, bottom, left = lines
    corners = np.vstack(
        [
            _line_intersection(top, left),
            _line_intersection(top, right),
            _line_intersection(bottom, right),
            _line_intersection(bottom, left),
        ]
    )
    widths = (
        np.linalg.norm(corners[1] - corners[0]),
        np.linalg.norm(corners[2] - corners[3]),
    )
    heights = (
        np.linalg.norm(corners[3] - corners[0]),
        np.linalg.norm(corners[2] - corners[1]),
    )
    width_px = float(np.mean(widths))
    height_px = float(np.mean(heights))
    if width_px <= 0.0 or height_px <= 0.0:
        raise RuntimeError("Fitted rim has zero-sized edges.")
    aspect = width_px / height_px
    if not cfg.min_image_aspect_ratio <= aspect <= cfg.max_image_aspect_ratio:
        raise RuntimeError(
            f"Fitted rim aspect ratio {aspect:.3f} is implausible."
        )

    tolerance = cfg.max_corner_outside_roi_px
    if (
        np.any(corners[:, 0] < u0 - tolerance)
        or np.any(corners[:, 0] > u1 - 1 + tolerance)
        or np.any(corners[:, 1] < v0 - tolerance)
        or np.any(corners[:, 1] > v1 - 1 + tolerance)
    ):
        raise RuntimeError("Fitted rim corners leave the expanded YOLOE ROI.")

    samples = np.stack(
        [
            _sample_segment(corners[0], corners[1], cfg),
            _sample_segment(corners[1], corners[2], cfg),
            _sample_segment(corners[3], corners[2], cfg),
            _sample_segment(corners[0], corners[3], cfg),
        ],
        axis=0,
    )
    center = np.mean(corners, axis=0)
    return FrontRim2D(
        roi_uv=roi,
        corners_uv=corners,
        center_uv=(float(center[0]), float(center[1])),
        side_lines=(top, right, bottom, left),
        side_samples_uv=samples,
    )
