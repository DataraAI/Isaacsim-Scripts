#!/usr/bin/env python3
"""Pure metrics and validation helpers for the fixed-frame prompt benchmark."""

from __future__ import annotations

from copy import deepcopy
import math
from typing import Iterable

import numpy as np


BENCHMARK_FRAME_COUNT = 60
MANIFEST_SCHEMA_VERSION = 1
MIN_PAIR_SUCCESS_RATE = 0.95
MAX_TRACK_SWITCHES = 0
MAX_CENTER_3D_JITTER_MM = 0.50
MAX_RAY_GAP_P95_MM = 0.50
MAX_SLOWDOWN_RATIO = 1.25


def _finite_values(values: Iterable[float]) -> np.ndarray:
    array = np.asarray(list(values), dtype=np.float64).reshape(-1)
    return array[np.isfinite(array)]


def _safe_percentile(values: Iterable[float], percentile: float) -> float:
    array = _finite_values(values)
    if array.size == 0:
        return math.nan
    return float(np.percentile(array, percentile))


def _safe_mean(values: Iterable[float]) -> float:
    array = _finite_values(values)
    if array.size == 0:
        return math.nan
    return float(np.mean(array))


def _safe_median(values: Iterable[float]) -> float:
    array = _finite_values(values)
    if array.size == 0:
        return math.nan
    return float(np.median(array))


def _safe_max(values: Iterable[float]) -> float:
    array = _finite_values(values)
    if array.size == 0:
        return math.nan
    return float(np.max(array))


def rms_radial_jitter(points: Iterable[Iterable[float]]) -> float:
    """Return RMS Euclidean distance from the component-wise median center."""
    array = np.asarray(list(points), dtype=np.float64)
    if array.size == 0:
        return math.nan
    if array.ndim != 2 or array.shape[1] < 1:
        raise ValueError(f"points must be a nonempty NxD array, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError("points must contain only finite values")
    center = np.median(array, axis=0)
    squared_distance = np.sum((array - center) ** 2, axis=1)
    return float(np.sqrt(np.mean(squared_distance)))


def count_track_switches(records: list[dict[str, object]], threshold_px: float) -> int:
    """Count successful frames where either selected eye jumps past the gate."""
    threshold = float(threshold_px)
    if not math.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("threshold_px must be finite and positive")

    previous: np.ndarray | None = None
    switches = 0
    for record in records:
        if not bool(record.get("pair_success", False)):
            continue
        current = np.asarray(
            [
                record.get("left_center_u", math.nan),
                record.get("left_center_v", math.nan),
                record.get("right_center_u", math.nan),
                record.get("right_center_v", math.nan),
            ],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(current)):
            continue
        if previous is not None:
            left_jump = float(np.linalg.norm(current[:2] - previous[:2]))
            right_jump = float(np.linalg.norm(current[2:] - previous[2:]))
            if left_jump > threshold or right_jump > threshold:
                switches += 1
        previous = current
    return switches


def _successful_points(records: list[dict[str, object]], keys: tuple[str, ...]) -> list[list[float]]:
    points: list[list[float]] = []
    for record in records:
        if not bool(record.get("pair_success", False)):
            continue
        point = [float(record.get(key, math.nan)) for key in keys]
        if all(math.isfinite(value) for value in point):
            points.append(point)
    return points


def summarize_records(
    strategy: str,
    records: list[dict[str, object]],
    total_frames: int,
    switch_threshold_px: float,
) -> dict[str, object]:
    """Aggregate one strategy's per-frame records and apply absolute gates."""
    if total_frames <= 0:
        raise ValueError("total_frames must be positive")
    if len(records) != total_frames:
        raise ValueError(
            f"Expected {total_frames} records, received {len(records)}"
        )

    left_success_count = sum(bool(item.get("left_success", False)) for item in records)
    right_success_count = sum(bool(item.get("right_success", False)) for item in records)
    pair_success_count = sum(bool(item.get("pair_success", False)) for item in records)

    left_points = _successful_points(records, ("left_center_u", "left_center_v"))
    right_points = _successful_points(records, ("right_center_u", "right_center_v"))
    world_points = _successful_points(
        records,
        ("center_world_x", "center_world_y", "center_world_z"),
    )

    left_jitter_px = rms_radial_jitter(left_points) if left_points else math.nan
    right_jitter_px = rms_radial_jitter(right_points) if right_points else math.nan
    center_3d_jitter_mm = (
        1000.0 * rms_radial_jitter(world_points) if world_points else math.nan
    )

    inference_ms = [float(item.get("inference_ms", math.nan)) for item in records]
    ray_gap_mm = [
        float(item.get("ray_gap_mm", math.nan))
        for item in records
        if bool(item.get("pair_success", False))
    ]
    center_error_px = [
        float(item.get("center_error_px", math.nan))
        for item in records
        if bool(item.get("pair_success", False))
    ]

    track_switch_count = count_track_switches(records, switch_threshold_px)
    pair_success_rate = pair_success_count / float(total_frames)
    ray_gap_p95_mm = _safe_percentile(ray_gap_mm, 95.0)

    base_quality_pass = bool(
        pair_success_rate >= MIN_PAIR_SUCCESS_RATE
        and track_switch_count <= MAX_TRACK_SWITCHES
        and math.isfinite(center_3d_jitter_mm)
        and center_3d_jitter_mm <= MAX_CENTER_3D_JITTER_MM
        and math.isfinite(ray_gap_p95_mm)
        and ray_gap_p95_mm <= MAX_RAY_GAP_P95_MM
    )

    return {
        "strategy": str(strategy),
        "total_frames": int(total_frames),
        "left_success_count": int(left_success_count),
        "right_success_count": int(right_success_count),
        "pair_success_count": int(pair_success_count),
        "left_success_rate": left_success_count / float(total_frames),
        "right_success_rate": right_success_count / float(total_frames),
        "pair_success_rate": pair_success_rate,
        "track_switch_count": int(track_switch_count),
        "left_center_jitter_px": left_jitter_px,
        "right_center_jitter_px": right_jitter_px,
        "center_3d_jitter_mm": center_3d_jitter_mm,
        "ray_gap_median_mm": _safe_median(ray_gap_mm),
        "ray_gap_p95_mm": ray_gap_p95_mm,
        "ray_gap_max_mm": _safe_max(ray_gap_mm),
        "center_error_median_px": _safe_median(center_error_px),
        "center_error_p95_px": _safe_percentile(center_error_px, 95.0),
        "inference_mean_ms": _safe_mean(inference_ms),
        "inference_median_ms": _safe_median(inference_ms),
        "inference_p95_ms": _safe_percentile(inference_ms, 95.0),
        "base_quality_pass": base_quality_pass,
    }


def apply_relative_speed_gate(
    summaries: list[dict[str, object]],
    max_slowdown_ratio: float = MAX_SLOWDOWN_RATIO,
) -> list[dict[str, object]]:
    """Add relative-speed and final qualification fields to summaries."""
    ratio_limit = float(max_slowdown_ratio)
    if not math.isfinite(ratio_limit) or ratio_limit < 1.0:
        raise ValueError("max_slowdown_ratio must be finite and at least 1.0")

    result = deepcopy(summaries)
    medians = [
        float(item.get("inference_median_ms", math.nan))
        for item in result
    ]
    finite_positive = [value for value in medians if math.isfinite(value) and value > 0.0]
    fastest = min(finite_positive) if finite_positive else math.nan

    for item, median in zip(result, medians):
        speed_ratio = (
            median / fastest
            if math.isfinite(fastest) and fastest > 0.0 and math.isfinite(median)
            else math.inf
        )
        speed_pass = speed_ratio <= ratio_limit
        item["speed_ratio_to_fastest"] = float(speed_ratio)
        item["speed_pass"] = bool(speed_pass)
        item["qualified"] = bool(item.get("base_quality_pass", False) and speed_pass)
    return result


def choose_winner(summaries: list[dict[str, object]]) -> str | None:
    """Choose the best qualified strategy using deterministic tie-breaks."""
    qualified = [item for item in summaries if bool(item.get("qualified", False))]
    if not qualified:
        return None

    def key(item: dict[str, object]) -> tuple[float, float, float, float, str]:
        return (
            -float(item.get("pair_success_rate", 0.0)),
            float(item.get("track_switch_count", math.inf)),
            float(item.get("center_3d_jitter_mm", math.inf)),
            float(item.get("inference_median_ms", math.inf)),
            str(item.get("strategy", "")),
        )

    return str(min(qualified, key=key)["strategy"])


def validate_manifest(manifest: dict[str, object], expected_count: int = BENCHMARK_FRAME_COUNT) -> None:
    """Validate the frozen frame-set contract before expensive inference."""
    if int(manifest.get("schema_version", -1)) != MANIFEST_SCHEMA_VERSION:
        raise ValueError("Unsupported prompt benchmark manifest schema")
    if int(manifest.get("frame_count", -1)) != int(expected_count):
        raise ValueError(
            f"Manifest frame_count must be {expected_count}, got {manifest.get('frame_count')}"
        )
    frames = manifest.get("frames")
    if not isinstance(frames, list) or len(frames) != expected_count:
        raise ValueError(f"Manifest must contain exactly {expected_count} frame entries")
    indices: list[int] = []
    for entry in frames:
        if not isinstance(entry, dict):
            raise ValueError("Every manifest frame entry must be an object")
        index = int(entry.get("frame_index", -1))
        indices.append(index)
        for key in ("left_image", "right_image"):
            value = entry.get(key)
            if not isinstance(value, str) or not value:
                raise ValueError(f"Frame {index} is missing {key}")
    if len(set(indices)) != expected_count:
        raise ValueError("Manifest frame indices must be unique")
    if sorted(indices) != list(range(1, expected_count + 1)):
        raise ValueError("Manifest frame indices must be contiguous starting at 1")
