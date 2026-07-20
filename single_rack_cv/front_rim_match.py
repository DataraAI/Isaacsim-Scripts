#!/usr/bin/env python3
"""Verified epipolar patch correspondences for front-bezel samples."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from front_rim import FrontRim2D


@dataclass(frozen=True)
class EpipolarPatchConfig:
    """Local stereo matching gates around cavity-center disparity."""

    patch_radius_px: int = 3
    search_x_px: float = 4.0
    search_y_px: float = 2.0
    search_step_px: float = 0.5
    min_ncc: float = 0.72
    min_uniqueness: float = 0.035
    min_texture: float = 0.018
    roundtrip_max_error_px: float = 0.75
    uniqueness_exclusion_px: float = 1.0


DEFAULT_EPIPOLAR_CONFIG = EpipolarPatchConfig()


@dataclass(frozen=True)
class EpipolarMatchResult:
    """Matched right-eye locations and diagnostics for every left sample."""

    predicted_right_uv: np.ndarray
    right_samples_uv: np.ndarray
    valid_mask: np.ndarray
    ncc_scores: np.ndarray
    uniqueness_margins: np.ndarray
    roundtrip_errors_px: np.ndarray

    def __post_init__(self) -> None:
        predicted = np.asarray(self.predicted_right_uv, dtype=np.float64)
        matched = np.asarray(self.right_samples_uv, dtype=np.float64)
        valid = np.asarray(self.valid_mask, dtype=bool)
        scores = np.asarray(self.ncc_scores, dtype=np.float64)
        margins = np.asarray(self.uniqueness_margins, dtype=np.float64)
        roundtrip = np.asarray(self.roundtrip_errors_px, dtype=np.float64)
        if predicted.ndim != 3 or predicted.shape[0] != 4 or predicted.shape[2] != 2:
            raise ValueError("predicted_right_uv must have shape (4,N,2).")
        if matched.shape != predicted.shape:
            raise ValueError("right_samples_uv must match predicted_right_uv.")
        expected = predicted.shape[:2]
        for name, value in (
            ("valid_mask", valid),
            ("ncc_scores", scores),
            ("uniqueness_margins", margins),
            ("roundtrip_errors_px", roundtrip),
        ):
            if value.shape != expected:
                raise ValueError(f"{name} must have shape {expected}.")
        object.__setattr__(self, "predicted_right_uv", predicted.copy())
        object.__setattr__(self, "right_samples_uv", matched.copy())
        object.__setattr__(self, "valid_mask", valid.copy())
        object.__setattr__(self, "ncc_scores", scores.copy())
        object.__setattr__(self, "uniqueness_margins", margins.copy())
        object.__setattr__(self, "roundtrip_errors_px", roundtrip.copy())

    @property
    def accepted_count(self) -> int:
        return int(np.count_nonzero(self.valid_mask))

    def median_ncc(self) -> float:
        values = self.ncc_scores[self.valid_mask]
        return float(np.median(values)) if values.size else float("nan")

    def median_uniqueness(self) -> float:
        values = self.uniqueness_margins[self.valid_mask]
        return float(np.median(values)) if values.size else float("nan")

    def roundtrip_p95_px(self) -> float:
        values = self.roundtrip_errors_px[self.valid_mask]
        return float(np.percentile(values, 95.0)) if values.size else float("nan")


@dataclass(frozen=True)
class _SearchResult:
    point_uv: np.ndarray
    score: float
    uniqueness: float


def _validate_config(cfg: EpipolarPatchConfig) -> None:
    if cfg.patch_radius_px < 1:
        raise ValueError("patch_radius_px must be positive.")
    if cfg.search_x_px <= 0.0 or cfg.search_y_px < 0.0:
        raise ValueError("Epipolar search ranges are invalid.")
    if cfg.search_step_px <= 0.0:
        raise ValueError("search_step_px must be positive.")
    if not -1.0 <= cfg.min_ncc <= 1.0:
        raise ValueError("min_ncc must be in [-1,1].")
    if cfg.min_uniqueness < 0.0 or cfg.min_texture < 0.0:
        raise ValueError("Matcher thresholds must be non-negative.")
    if cfg.roundtrip_max_error_px <= 0.0:
        raise ValueError("roundtrip_max_error_px must be positive.")


def _feature_image(rgb: np.ndarray) -> np.ndarray:
    image = np.asarray(rgb)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 RGB image, got {image.shape}.")
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    gray = cv2.GaussianBlur(gray, (3, 3), 0.0)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3, scale=0.25)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3, scale=0.25)
    return np.stack((gray, gx, gy), axis=-1)


def _patch_descriptor(
    features: np.ndarray,
    center_uv: np.ndarray,
    radius_px: int,
) -> tuple[np.ndarray | None, float]:
    u, v = np.asarray(center_uv, dtype=np.float64).reshape(2)
    height, width = features.shape[:2]
    margin = radius_px + 1
    if (
        u < margin
        or v < margin
        or u > width - 1 - margin
        or v > height - 1 - margin
    ):
        return None, 0.0
    size = 2 * radius_px + 1
    patch = cv2.getRectSubPix(
        features,
        (size, size),
        (float(u), float(v)),
    ).astype(np.float64)
    texture = float(
        np.sqrt(
            np.var(patch[:, :, 0])
            + 0.25 * np.var(patch[:, :, 1])
            + 0.25 * np.var(patch[:, :, 2])
        )
    )
    channels = []
    for channel_index in range(3):
        values = patch[:, :, channel_index].reshape(-1)
        channels.append(values - float(np.mean(values)))
    descriptor = np.concatenate(channels)
    norm = float(np.linalg.norm(descriptor))
    if not np.isfinite(norm) or norm <= 1.0e-12:
        return None, texture
    return descriptor / norm, texture


def _offset_values(limit_px: float, step_px: float) -> np.ndarray:
    count = int(round(2.0 * limit_px / step_px))
    return np.linspace(-limit_px, limit_px, count + 1, dtype=np.float64)


def _search_patch(
    template: np.ndarray,
    target_features: np.ndarray,
    predicted_uv: np.ndarray,
    cfg: EpipolarPatchConfig,
) -> _SearchResult | None:
    candidates: list[tuple[float, np.ndarray]] = []
    for dy in _offset_values(cfg.search_y_px, cfg.search_step_px):
        for dx in _offset_values(cfg.search_x_px, cfg.search_step_px):
            point = np.asarray(predicted_uv, dtype=np.float64) + np.array(
                [dx, dy], dtype=np.float64
            )
            descriptor, texture = _patch_descriptor(
                target_features,
                point,
                cfg.patch_radius_px,
            )
            if descriptor is None or texture < cfg.min_texture:
                continue
            candidates.append((float(np.dot(template, descriptor)), point))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    best_score, best_point = candidates[0]
    second_score: float | None = None
    for score, point in candidates[1:]:
        if (
            float(np.linalg.norm(point - best_point))
            >= cfg.uniqueness_exclusion_px
        ):
            second_score = score
            break
    if second_score is None:
        return None
    return _SearchResult(
        point_uv=best_point,
        score=best_score,
        uniqueness=best_score - second_score,
    )


def match_front_bezel_samples(
    left_rgb: np.ndarray,
    right_rgb: np.ndarray,
    left_rim: FrontRim2D,
    right_rim: FrontRim2D,
    cfg: EpipolarPatchConfig = DEFAULT_EPIPOLAR_CONFIG,
) -> EpipolarMatchResult:
    """Match each left bezel sample to the right eye with round-trip checks."""
    _validate_config(cfg)
    left_features = _feature_image(left_rgb)
    right_features = _feature_image(right_rgb)
    left_samples = np.asarray(left_rim.side_samples_uv, dtype=np.float64)
    if left_samples.ndim != 3 or left_samples.shape[0] != 4:
        raise ValueError("Left bezel samples must have shape (4,N,2).")
    center_shift = (
        np.asarray(right_rim.center_uv, dtype=np.float64)
        - np.asarray(left_rim.center_uv, dtype=np.float64)
    )
    predicted = left_samples + center_shift[None, None, :]
    matched = np.full_like(predicted, np.nan)
    valid = np.zeros(left_samples.shape[:2], dtype=bool)
    scores = np.full(left_samples.shape[:2], np.nan, dtype=np.float64)
    uniqueness = np.full(left_samples.shape[:2], np.nan, dtype=np.float64)
    roundtrip = np.full(left_samples.shape[:2], np.nan, dtype=np.float64)

    for side_index in range(left_samples.shape[0]):
        for sample_index in range(left_samples.shape[1]):
            left_point = left_samples[side_index, sample_index]
            template, texture = _patch_descriptor(
                left_features,
                left_point,
                cfg.patch_radius_px,
            )
            if template is None or texture < cfg.min_texture:
                continue
            forward = _search_patch(
                template,
                right_features,
                predicted[side_index, sample_index],
                cfg,
            )
            if forward is None:
                continue
            matched[side_index, sample_index] = forward.point_uv
            scores[side_index, sample_index] = forward.score
            uniqueness[side_index, sample_index] = forward.uniqueness
            if (
                forward.score < cfg.min_ncc
                or forward.uniqueness < cfg.min_uniqueness
            ):
                continue
            right_template, right_texture = _patch_descriptor(
                right_features,
                forward.point_uv,
                cfg.patch_radius_px,
            )
            if right_template is None or right_texture < cfg.min_texture:
                continue
            backward = _search_patch(
                right_template,
                left_features,
                forward.point_uv - center_shift,
                cfg,
            )
            if backward is None or backward.score < cfg.min_ncc:
                continue
            error = float(np.linalg.norm(backward.point_uv - left_point))
            roundtrip[side_index, sample_index] = error
            if error > cfg.roundtrip_max_error_px:
                continue
            valid[side_index, sample_index] = True

    return EpipolarMatchResult(
        predicted_right_uv=predicted,
        right_samples_uv=matched,
        valid_mask=valid,
        ncc_scores=scores,
        uniqueness_margins=uniqueness,
        roundtrip_errors_px=roundtrip,
    )


def build_matched_right_rim(
    left_rim: FrontRim2D,
    right_rim: FrontRim2D,
    matches: EpipolarMatchResult,
    max_epipolar_error_px: float,
) -> FrontRim2D:
    """Clone the right rim with matched samples; force rejected pairs to skip."""
    left_samples = np.asarray(left_rim.side_samples_uv, dtype=np.float64)
    if matches.right_samples_uv.shape != left_samples.shape:
        raise ValueError("Match samples do not match the left rim shape.")
    samples = matches.right_samples_uv.copy()
    invalid = ~matches.valid_mask
    invalid_points = left_samples[invalid].copy()
    invalid_points[:, 1] += max(10.0, 4.0 * float(max_epipolar_error_px))
    samples[invalid] = invalid_points
    return FrontRim2D(
        roi_uv=right_rim.roi_uv,
        corners_uv=right_rim.corners_uv,
        center_uv=right_rim.center_uv,
        side_lines=right_rim.side_lines,
        side_samples_uv=samples,
    )
