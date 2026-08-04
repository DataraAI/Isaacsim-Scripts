#!/usr/bin/env python3
"""Independent and joint physical RGB front-lip fitting."""

from __future__ import annotations

import math
import cv2
import numpy as np

from plane_rectified_types import (
    FrontLipFit,
    MIN_EDGE_SAMPLES,
    PlaneFrame,
    RectifiedEye,
    _unit,
)


def _robust_line(
    samples_xy: np.ndarray,
    *,
    x_from_y: bool,
    residual_floor: float,
) -> tuple[tuple[float, float], np.ndarray, float]:
    samples = np.asarray(samples_xy, dtype=np.float64).reshape(-1, 2)
    if samples.shape[0] < MIN_EDGE_SAMPLES:
        raise RuntimeError("Front-lip edge has insufficient support.")
    x = samples[:, 0]
    y = samples[:, 1]
    keep = np.ones(samples.shape[0], dtype=bool)
    for _ in range(6):
        if int(np.count_nonzero(keep)) < MIN_EDGE_SAMPLES:
            raise RuntimeError("Front-lip edge fit lost support.")
        if x_from_y:
            slope, intercept = np.polyfit(y[keep], x[keep], 1)
            residual = np.abs(x - (slope * y + intercept))
        else:
            slope, intercept = np.polyfit(x[keep], y[keep], 1)
            residual = np.abs(y - (slope * x + intercept))
        median = float(np.median(residual[keep]))
        mad = float(np.median(np.abs(residual[keep] - median)))
        threshold = max(float(residual_floor), median + 2.5 * mad)
        next_keep = residual <= threshold
        if np.array_equal(next_keep, keep):
            break
        keep = next_keep
    rms = float(np.sqrt(np.mean(residual[keep] ** 2)))
    return (float(slope), float(intercept)), keep, rms


def _mask_lower_mouth_geometry(mask: np.ndarray) -> tuple[int, int, float, float]:
    component = np.where(mask, 255, 0).astype(np.uint8)
    contours, _ = cv2.findContours(
        component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if not contours:
        raise RuntimeError("Rectified aperture mask contains no contour.")
    filled = np.zeros_like(component)
    cv2.drawContours(filled, [max(contours, key=cv2.contourArea)], -1, 255, -1)
    binary = filled > 0
    supported_rows = np.flatnonzero(np.any(binary, axis=1))
    rows: list[tuple[int, int, int, int]] = []
    for row in supported_rows:
        columns = np.flatnonzero(binary[row])
        rows.append((int(row), int(columns[0]), int(columns[-1]), int(columns.size)))
    if len(rows) < 8:
        raise RuntimeError("Rectified aperture mask has too few rows.")
    maximum_width = max(row[3] for row in rows)
    candidates = [row for row in rows if row[3] >= 0.78 * maximum_width]
    runs: list[list[tuple[int, int, int, int]]] = []
    for row in candidates:
        if not runs or row[0] != runs[-1][-1][0] + 1:
            runs.append([row])
        else:
            runs[-1].append(row)
    broad = max(runs, key=lambda run: (len(run), run[-1][0]), default=[])
    if len(broad) < 6:
        raise RuntimeError("Rectified mask has no stable lower-mouth run.")
    return (
        int(broad[0][0]),
        int(broad[-1][0]),
        float(np.median([row[1] for row in broad])),
        float(np.median([row[2] for row in broad])),
    )


def _qualified_signed_index(
    values: np.ndarray,
    *,
    start: int,
    negative: bool,
    minimum_strength: float,
    relative_strength: float,
    choose_last: bool,
) -> int | None:
    samples = np.asarray(values, dtype=np.float64).reshape(-1)
    if samples.size == 0:
        return None
    if negative:
        strongest = max(0.0, -float(np.min(samples)))
        threshold = max(float(minimum_strength), float(relative_strength) * strongest)
        candidates = np.flatnonzero(samples <= -threshold)
    else:
        strongest = max(0.0, float(np.max(samples)))
        threshold = max(float(minimum_strength), float(relative_strength) * strongest)
        candidates = np.flatnonzero(samples >= threshold)
    if candidates.size == 0:
        return None
    local = int(candidates[-1] if choose_last else candidates[0])
    return int(start) + local


def _line_angle_deg(
    first: tuple[float, float],
    second: tuple[float, float],
    *,
    side: bool,
) -> float:
    if side:
        first_direction = _unit(np.array([first[0], 1.0, 0.0]), "side direction")[:2]
        second_direction = _unit(np.array([second[0], 1.0, 0.0]), "side direction")[:2]
    else:
        first_direction = _unit(np.array([1.0, first[0], 0.0]), "horizontal direction")[:2]
        second_direction = _unit(np.array([1.0, second[0], 0.0]), "horizontal direction")[:2]
    cosine = float(np.clip(abs(first_direction @ second_direction), -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def _fit_parallel_pair(
    first_samples_xy: np.ndarray,
    second_samples_xy: np.ndarray,
    *,
    x_from_y: bool,
    residual_floor: float,
) -> tuple[tuple[float, float], tuple[float, float], np.ndarray, np.ndarray]:
    """Robustly fit two distinct parallel lines with one shared slope."""

    first = np.asarray(first_samples_xy, dtype=np.float64).reshape(-1, 2)
    second = np.asarray(second_samples_xy, dtype=np.float64).reshape(-1, 2)
    if first.shape[0] < MIN_EDGE_SAMPLES or second.shape[0] < MIN_EDGE_SAMPLES:
        raise RuntimeError("Parallel front-lip edges have insufficient support.")

    def independent_and_dependent(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if x_from_y:
            return samples[:, 1], samples[:, 0]
        return samples[:, 0], samples[:, 1]

    first_independent, first_dependent = independent_and_dependent(first)
    second_independent, second_dependent = independent_and_dependent(second)
    first_keep = np.ones(first.shape[0], dtype=bool)
    second_keep = np.ones(second.shape[0], dtype=bool)

    for _ in range(6):
        first_count = int(np.count_nonzero(first_keep))
        second_count = int(np.count_nonzero(second_keep))
        first_design = np.column_stack(
            (first_independent[first_keep], np.ones(first_count), np.zeros(first_count))
        )
        second_design = np.column_stack(
            (second_independent[second_keep], np.zeros(second_count), np.ones(second_count))
        )
        design = np.vstack((first_design, second_design))
        values = np.concatenate(
            (first_dependent[first_keep], second_dependent[second_keep])
        )
        slope, first_intercept, second_intercept = np.linalg.lstsq(
            design, values, rcond=None
        )[0]
        first_residual = np.abs(
            first_dependent - (slope * first_independent + first_intercept)
        )
        second_residual = np.abs(
            second_dependent - (slope * second_independent + second_intercept)
        )

        def next_keep(residual: np.ndarray, keep: np.ndarray) -> np.ndarray:
            median = float(np.median(residual[keep]))
            mad = float(np.median(np.abs(residual[keep] - median)))
            threshold = max(float(residual_floor), median + 2.5 * mad)
            return residual <= threshold

        next_first = next_keep(first_residual, first_keep)
        next_second = next_keep(second_residual, second_keep)
        if (
            int(np.count_nonzero(next_first)) < MIN_EDGE_SAMPLES
            or int(np.count_nonzero(next_second)) < MIN_EDGE_SAMPLES
        ):
            raise RuntimeError("Parallel front-lip fit lost support.")
        if np.array_equal(next_first, first_keep) and np.array_equal(
            next_second, second_keep
        ):
            break
        first_keep = next_first
        second_keep = next_second

    return (
        (float(slope), float(first_intercept)),
        (float(slope), float(second_intercept)),
        first_keep,
        second_keep,
    )


def _intersect(side, horizontal) -> np.ndarray:
    a, b = map(float, side)
    c, d = map(float, horizontal)
    denominator = 1.0 - a * c
    if abs(denominator) <= 1.0e-10:
        raise RuntimeError("Front-lip boundary lines are parallel.")
    x = (a * d + b) / denominator
    return np.array([x, c * x + d], dtype=np.float64)


def _quad_from_lines(left, right, top, bottom) -> np.ndarray:
    corners = np.vstack(
        (
            _intersect(left, top),
            _intersect(right, top),
            _intersect(right, bottom),
            _intersect(left, bottom),
        )
    )
    if float(abs(cv2.contourArea(corners.astype(np.float32)))) <= 1.0e-12:
        raise RuntimeError("Front-lip quadrilateral is degenerate.")
    return corners


def _projective_center(corners: np.ndarray) -> np.ndarray:
    points = np.asarray(corners, dtype=np.float64).reshape(4, 2)
    matrix = np.column_stack((points[2] - points[0], -(points[3] - points[1])))
    values, _, rank, _ = np.linalg.lstsq(matrix, points[1] - points[0], rcond=None)
    if rank < 2:
        raise RuntimeError("Front-lip diagonals do not meet finitely.")
    return points[0] + float(values[0]) * (points[2] - points[0])


def point_in_convex_quad(
    point: np.ndarray,
    corners: np.ndarray,
    tolerance: float = 1.0e-9,
) -> bool:
    point = np.asarray(point, dtype=np.float64).reshape(2)
    quad = np.asarray(corners, dtype=np.float64).reshape(4, 2)
    signs = []
    for start, end in zip(quad, np.roll(quad, -1, axis=0)):
        edge = end - start
        relative = point - start
        signs.append(float(edge[0] * relative[1] - edge[1] * relative[0]))
    values = np.asarray(signs)
    return bool(np.all(values >= -tolerance) or np.all(values <= tolerance))


def _sample_projection(rectified: RectifiedEye, points_xy: np.ndarray) -> np.ndarray:
    points = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    x = np.clip(points[:, 0], 0.0, rectified.rgb.shape[1] - 1.001)
    y = np.clip(points[:, 1], 0.0, rectified.rgb.shape[0] - 1.001)
    map_x = cv2.remap(
        rectified.map_u_px.astype(np.float32),
        x.reshape(-1, 1).astype(np.float32),
        y.reshape(-1, 1).astype(np.float32),
        cv2.INTER_LINEAR,
    ).reshape(-1)
    map_y = cv2.remap(
        rectified.map_v_px.astype(np.float32),
        x.reshape(-1, 1).astype(np.float32),
        y.reshape(-1, 1).astype(np.float32),
        cv2.INTER_LINEAR,
    ).reshape(-1)
    return np.column_stack((map_x, map_y))


def _edge_reprojection_residual(
    rectified: RectifiedEye,
    name: str,
    samples_xy: np.ndarray,
    line: tuple[float, float],
) -> float:
    samples = np.asarray(samples_xy, dtype=np.float64).reshape(-1, 2)
    fitted = samples.copy()
    if name in ("left", "right"):
        fitted[:, 0] = float(line[0]) * samples[:, 1] + float(line[1])
    else:
        fitted[:, 1] = float(line[0]) * samples[:, 0] + float(line[1])
    original_samples = _sample_projection(rectified, samples)
    original_fitted = _sample_projection(rectified, fitted)
    residuals = np.linalg.norm(original_samples - original_fitted, axis=1)
    return float(np.max(residuals))


def _draw_fit(rectified: RectifiedEye, fit: FrontLipFit) -> np.ndarray:
    overlay = rectified.rgb.copy()
    corners_px = rectified.metric_to_pixel(fit.corners_uv_m)
    center_px = rectified.metric_to_pixel(fit.center_uv_m)
    cv2.polylines(
        overlay,
        [np.round(corners_px).astype(np.int32)],
        True,
        (255, 0, 255),
        2,
    )
    cv2.drawMarker(
        overlay,
        tuple(np.round(center_px).astype(int)),
        (0, 255, 255),
        cv2.MARKER_CROSS,
        15,
        2,
    )
    colors = {
        "left": (255, 0, 0),
        "right": (0, 255, 0),
        "top": (255, 255, 0),
        "bottom": (0, 255, 255),
    }
    for name, metric in fit.edge_samples_uv_m.items():
        pixels = rectified.metric_to_pixel(metric)
        for x, y in np.round(pixels).astype(int):
            if 0 <= x < overlay.shape[1] and 0 <= y < overlay.shape[0]:
                cv2.circle(overlay, (int(x), int(y)), 1, colors[name], -1)
    return overlay


def _draw_reprojection(
    rgb: np.ndarray,
    camera,
    fit: FrontLipFit,
    frame: PlaneFrame,
) -> np.ndarray:
    overlay = np.asarray(rgb, dtype=np.uint8).copy()
    corners_world = frame.metric_to_world(fit.corners_uv_m)
    center_world = frame.metric_to_world(fit.center_uv_m)
    corners = np.asarray([camera.project_world(point) for point in corners_world])
    center = np.asarray(camera.project_world(center_world))
    cv2.polylines(
        overlay,
        [np.round(corners).astype(np.int32)],
        True,
        (255, 0, 255),
        2,
    )
    cv2.drawMarker(
        overlay,
        tuple(np.round(center).astype(int)),
        (0, 255, 255),
        cv2.MARKER_CROSS,
        15,
        2,
    )
    return overlay
