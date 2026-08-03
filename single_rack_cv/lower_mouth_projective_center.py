#!/usr/bin/env python3
"""Perspective-correct center of the lower RJ45 insertion mouth."""

from __future__ import annotations

import math

import cv2
import numpy as np


_BROAD_ROW_FRACTION = 0.78
_OUTER_BAND_FRACTION = 0.30


def _largest_component_mask(mask: np.ndarray, camera) -> np.ndarray:
    source = np.asarray(mask)
    if source.ndim != 2:
        raise ValueError("Aperture mask must be a 2D array.")

    height = int(camera.image_height_px)
    width = int(camera.image_width_px)
    if min(source.shape[0], source.shape[1], height, width) <= 0:
        raise ValueError("Mask and camera dimensions must be positive.")

    binary = np.where(source > 0, 255, 0).astype(np.uint8)
    if binary.shape != (height, width):
        binary = cv2.resize(
            binary,
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        )

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )
    if not contours:
        raise RuntimeError("Aperture mask contains no external contour.")

    contour = max(contours, key=cv2.contourArea)
    if float(cv2.contourArea(contour)) < 8.0:
        raise RuntimeError("Aperture contour is too small.")

    component = np.zeros_like(binary)
    cv2.drawContours(component, [contour], -1, 255, thickness=cv2.FILLED)
    return component


def _robust_fit_x_from_y(
    points: list[tuple[float, float]],
) -> tuple[float, float]:
    samples = np.asarray(points, dtype=np.float64)
    if samples.shape[0] < 6:
        raise RuntimeError("Too few lower-mouth side samples.")

    x = samples[:, 0]
    y = samples[:, 1]
    keep = np.ones(samples.shape[0], dtype=bool)
    for _ in range(5):
        if int(np.count_nonzero(keep)) < 6:
            raise RuntimeError("Lower-mouth side fit lost support.")
        slope, intercept = np.polyfit(y[keep], x[keep], 1)
        residual = np.abs(x - (slope * y + intercept))
        median = float(np.median(residual[keep]))
        mad = float(np.median(np.abs(residual[keep] - median)))
        keep = residual <= max(1.0, median + 2.5 * mad)
    return float(slope), float(intercept)


def _robust_fit_y_from_x(
    points: list[tuple[float, float]],
) -> tuple[float, float]:
    samples = np.asarray(points, dtype=np.float64)
    if samples.shape[0] < 6:
        raise RuntimeError("Too few lower-mouth horizontal samples.")

    x = samples[:, 0]
    y = samples[:, 1]
    keep = np.ones(samples.shape[0], dtype=bool)
    for _ in range(5):
        if int(np.count_nonzero(keep)) < 6:
            raise RuntimeError("Lower-mouth horizontal fit lost support.")
        slope, intercept = np.polyfit(x[keep], y[keep], 1)
        residual = np.abs(y - (slope * x + intercept))
        median = float(np.median(residual[keep]))
        mad = float(np.median(np.abs(residual[keep] - median)))
        keep = residual <= max(1.0, median + 2.5 * mad)
    return float(slope), float(intercept)


def _intersect_side_and_horizontal(
    side: tuple[float, float],
    horizontal: tuple[float, float],
) -> np.ndarray:
    side_slope, side_intercept = map(float, side)
    horizontal_slope, horizontal_intercept = map(float, horizontal)
    denominator = 1.0 - side_slope * horizontal_slope
    if abs(denominator) <= 1.0e-9:
        raise RuntimeError("Lower-mouth boundary lines are parallel.")

    u = (
        side_slope * horizontal_intercept + side_intercept
    ) / denominator
    v = horizontal_slope * u + horizontal_intercept
    point = np.array([u, v], dtype=np.float64)
    if not np.all(np.isfinite(point)):
        raise RuntimeError("Lower-mouth corner is not finite.")
    return point


def _line_through(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    first_h = np.array([float(first[0]), float(first[1]), 1.0])
    second_h = np.array([float(second[0]), float(second[1]), 1.0])
    line = np.cross(first_h, second_h)
    if float(np.linalg.norm(line[:2])) <= 1.0e-12:
        raise RuntimeError("Lower-mouth diagonal is degenerate.")
    return line


def _projective_center(corners_uv: np.ndarray) -> np.ndarray:
    corners = np.asarray(corners_uv, dtype=np.float64).reshape(4, 2)
    first_diagonal = _line_through(corners[0], corners[2])
    second_diagonal = _line_through(corners[1], corners[3])
    center_h = np.cross(first_diagonal, second_diagonal)
    if abs(float(center_h[2])) <= 1.0e-12:
        raise RuntimeError("Lower-mouth diagonals do not meet finitely.")
    center = center_h[:2] / center_h[2]
    if not np.all(np.isfinite(center)):
        raise RuntimeError("Lower-mouth projective center is not finite.")
    return center.astype(np.float64)


def _lower_mouth_lines(
    mask: np.ndarray,
    camera,
) -> tuple[
    tuple[float, float],
    tuple[float, float],
    tuple[float, float],
    tuple[float, float],
]:
    """Fit the four projected boundaries of the wide lower insertion mouth."""

    component = _largest_component_mask(mask, camera)
    binary = component > 0
    contours, _ = cv2.findContours(
        component,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )
    boundary = np.zeros_like(component)
    cv2.drawContours(boundary, contours, -1, 255, thickness=1)
    supported_rows = np.flatnonzero(np.any(binary, axis=1))
    if supported_rows.size < 8:
        raise RuntimeError("Aperture mask has too few supported rows.")

    rows: list[tuple[int, int, int, int]] = []
    for row in supported_rows:
        columns = np.flatnonzero(binary[row])
        rows.append(
            (int(row), int(columns[0]), int(columns[-1]), int(columns.size))
        )

    maximum_width = max(width for _, _, _, width in rows)
    broad_candidates = [
        geometry
        for geometry in rows
        if geometry[3] >= _BROAD_ROW_FRACTION * maximum_width
    ]
    runs: list[list[tuple[int, int, int, int]]] = []
    for geometry in broad_candidates:
        if not runs or geometry[0] != runs[-1][-1][0] + 1:
            runs.append([geometry])
        else:
            runs[-1].append(geometry)
    broad_rows = max(
        runs,
        key=lambda run: (len(run), run[-1][0]),
        default=[],
    )
    if len(broad_rows) < 6:
        raise RuntimeError("Stepped aperture has no stable lower mouth.")

    broad_start = int(broad_rows[0][0])
    broad_end = int(broad_rows[-1][0])
    lower_height = broad_end - broad_start + 1
    if lower_height < 6:
        raise RuntimeError("Lower mouth is too short for projective fitting.")

    side_rows = [
        geometry
        for geometry in broad_rows
        if broad_start + 1 <= geometry[0] <= broad_end - 1
    ]
    if len(side_rows) < 6:
        side_rows = broad_rows

    left = _robust_fit_x_from_y(
        [(float(left_x), float(row)) for row, left_x, _, _ in side_rows]
    )
    right = _robust_fit_x_from_y(
        [(float(right_x), float(row)) for row, _, right_x, _ in side_rows]
    )

    middle_row = 0.5 * float(broad_start + broad_end)
    left_middle = left[0] * middle_row + left[1]
    right_middle = right[0] * middle_row + right[1]
    mouth_width = float(right_middle - left_middle)
    if mouth_width < 8.0:
        raise RuntimeError("Lower-mouth side walls are implausibly close.")

    first_column = max(0, int(math.floor(left_middle)))
    last_column = min(
        int(camera.image_width_px) - 1,
        int(math.ceil(right_middle)),
    )
    if last_column - first_column < 8:
        raise RuntimeError("Lower-mouth horizontal span is too small.")

    shoulder_slack = max(3, int(round(0.12 * lower_height)))
    top_samples: list[tuple[float, float]] = []
    bottom_samples: list[tuple[float, float]] = []
    for column in range(first_column, last_column + 1):
        rows_at_column = np.flatnonzero(binary[:, column])
        if rows_at_column.size == 0:
            continue

        fraction = (
            float(column) - left_middle
        ) / max(mouth_width, 1.0)
        in_outer_shoulder = (
            fraction <= _OUTER_BAND_FRACTION
            or fraction >= 1.0 - _OUTER_BAND_FRACTION
        )
        boundary_rows = np.flatnonzero(boundary[:, column])
        shoulder_rows = boundary_rows[
            (boundary_rows >= broad_start - shoulder_slack)
            & (boundary_rows <= broad_start + shoulder_slack)
        ]
        if in_outer_shoulder and shoulder_rows.size > 0:
            nearest_index = int(
                np.argmin(np.abs(shoulder_rows - broad_start))
            )
            top_samples.append(
                (
                    float(column),
                    float(shoulder_rows[nearest_index]),
                )
            )

        if 0.12 <= fraction <= 0.88:
            bottom_samples.append(
                (float(column), float(rows_at_column[-1]))
            )

    if len(top_samples) < 6:
        raise RuntimeError("Lower-mouth shoulder has insufficient support.")
    if len(bottom_samples) < 6:
        raise RuntimeError("Lower-mouth bottom has insufficient support.")

    top = _robust_fit_y_from_x(top_samples)
    bottom = _robust_fit_y_from_x(bottom_samples)

    top_middle = top[0] * (0.5 * (left_middle + right_middle)) + top[1]
    bottom_middle = (
        bottom[0] * (0.5 * (left_middle + right_middle)) + bottom[1]
    )
    fitted_height = float(bottom_middle - top_middle)
    if fitted_height < 5.0 or fitted_height > 2.0 * lower_height:
        raise RuntimeError(
            "Lower-mouth fitted height is implausible: "
            f"{fitted_height:.3f}px."
        )

    return left, right, top, bottom


def lower_mouth_corners_pixel(
    mask: np.ndarray,
    camera,
) -> np.ndarray:
    """Return TL, TR, BR, BL corners of the projected lower mouth."""

    left, right, top, bottom = _lower_mouth_lines(mask, camera)
    corners = np.vstack(
        [
            _intersect_side_and_horizontal(left, top),
            _intersect_side_and_horizontal(right, top),
            _intersect_side_and_horizontal(right, bottom),
            _intersect_side_and_horizontal(left, bottom),
        ]
    )

    area = float(abs(cv2.contourArea(corners.astype(np.float32))))
    if area < 25.0:
        raise RuntimeError("Lower-mouth quadrilateral area is too small.")
    return corners


def aperture_center_pixel(
    rgb: np.ndarray,
    mask: np.ndarray,
    camera,
) -> np.ndarray:
    """Return the projective center of the wide physical insertion mouth."""

    # RGB remains in the signature because the live estimator has a stable
    # per-eye API. The semantic contour defines the aperture silhouette; depth
    # is still supplied independently by the measured outer-bezel plane.
    del rgb

    corners = lower_mouth_corners_pixel(mask, camera)
    center = _projective_center(corners)
    if not (
        0.0 <= center[0] < float(camera.image_width_px)
        and 0.0 <= center[1] < float(camera.image_height_px)
    ):
        raise RuntimeError("Lower-mouth projective center is outside the image.")
    return center
