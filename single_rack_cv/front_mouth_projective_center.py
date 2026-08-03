#!/usr/bin/env python3
"""Projective front-mouth center using the outer RGB bezel edges."""

from __future__ import annotations

import math

import cv2
import numpy as np

from stereo_center import (
    _front_rim_lines,
    _gray_camera_image,
    _largest_component_mask,
    _robust_fit_x_from_y,
)
from stereo_center_projective import _projective_center_from_lines


_MINIMUM_EDGE_STRENGTH = 24.0
_RELATIVE_EDGE_STRENGTH = 0.40


def _outermost_signed_edge_index(
    values: np.ndarray,
    *,
    start_index: int,
    polarity: str,
    minimum_strength: float = _MINIMUM_EDGE_STRENGTH,
    relative_strength: float = _RELATIVE_EDGE_STRENGTH,
) -> int:
    """Choose the outermost credible edge with the requested contrast sign.

    A recessed cavity often creates a stronger edge with the same sign as the
    physical front lip. Selecting argmin/argmax therefore chooses the inner
    structure. This helper keeps every candidate with meaningful absolute and
    relative strength, then chooses the candidate furthest toward the exterior.
    """

    samples = np.asarray(values, dtype=np.float64).reshape(-1)
    if samples.size == 0 or not np.all(np.isfinite(samples)):
        raise RuntimeError("Front-mouth gradient samples are unavailable.")

    absolute_minimum = float(minimum_strength)
    relative = float(relative_strength)
    if not math.isfinite(absolute_minimum) or absolute_minimum <= 0.0:
        raise ValueError("minimum_strength must be finite and positive.")
    if not math.isfinite(relative) or not 0.0 < relative <= 1.0:
        raise ValueError("relative_strength must be in (0, 1].")

    if polarity == "negative":
        strongest = max(0.0, -float(np.min(samples)))
        if strongest < absolute_minimum:
            raise RuntimeError("No qualified negative front-mouth edge.")
        threshold = -max(absolute_minimum, relative * strongest)
        candidates = np.flatnonzero(samples <= threshold)
        if candidates.size == 0:
            raise RuntimeError("No qualified negative front-mouth edge.")
        local_index = int(candidates[0])
    elif polarity == "positive":
        strongest = max(0.0, float(np.max(samples)))
        if strongest < absolute_minimum:
            raise RuntimeError("No qualified positive front-mouth edge.")
        threshold = max(absolute_minimum, relative * strongest)
        candidates = np.flatnonzero(samples >= threshold)
        if candidates.size == 0:
            raise RuntimeError("No qualified positive front-mouth edge.")
        local_index = int(candidates[-1])
    else:
        raise ValueError("polarity must be 'negative' or 'positive'.")

    return int(start_index) + local_index


def _mask_side_search_geometry(mask: np.ndarray, camera):
    component = _largest_component_mask(mask, camera)
    binary = component > 0
    support_rows = np.flatnonzero(np.any(binary, axis=1))
    if support_rows.size < 8:
        raise RuntimeError("Aperture mask has too few supported rows.")

    rows: list[tuple[int, int, int, int]] = []
    for row in support_rows:
        columns = np.flatnonzero(binary[row])
        rows.append(
            (int(row), int(columns[0]), int(columns[-1]), int(columns.size))
        )

    maximum_width = max(width for _, _, _, width in rows)
    shoulder_row = min(
        row for row, _, _, width in rows
        if width >= 0.75 * maximum_width
    )
    mask_bottom = int(support_rows[-1])
    mask_height = mask_bottom - int(support_rows[0]) + 1

    lower_rows = [
        (row, left, right)
        for row, left, right, width in rows
        if (
            row >= shoulder_row + 1
            and row <= mask_bottom - 1
            and width >= 0.80 * maximum_width
        )
    ]
    if len(lower_rows) < 3:
        raise RuntimeError("Stepped aperture mask has no stable shoulder.")

    mask_left = int(round(np.median([left for _, left, _ in lower_rows])))
    mask_right = int(round(np.median([right for _, _, right in lower_rows])))
    mask_width = mask_right - mask_left
    if mask_width < 8 or mask_height < 5:
        raise RuntimeError(
            "Aperture support is too small for outer-edge fitting: "
            f"{mask_width}x{mask_height}px."
        )

    image_width = int(camera.image_width_px)
    image_height = int(camera.image_height_px)
    side_search_end = min(
        image_height - 2,
        mask_bottom + max(8, int(round(0.60 * mask_height))),
    )

    exterior_span = max(6, int(round(0.45 * mask_width)))
    interior_span = max(3, int(round(0.12 * mask_width)))
    left_start = max(1, mask_left - exterior_span)
    left_end = min(image_width - 2, mask_left + interior_span)
    right_start = max(1, mask_right - interior_span)
    right_end = min(image_width - 2, mask_right + exterior_span)

    return (
        shoulder_row,
        side_search_end,
        left_start,
        left_end,
        right_start,
        right_end,
    )


def _outer_front_side_lines(
    rgb: np.ndarray,
    mask: np.ndarray,
    camera,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Fit the physical left/right mouth edges rather than recessed edges."""

    gray = _gray_camera_image(rgb, camera)
    gradient_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    (
        shoulder_row,
        side_search_end,
        left_start,
        left_end,
        right_start,
        right_end,
    ) = _mask_side_search_geometry(mask, camera)

    left_samples: list[tuple[float, float]] = []
    right_samples: list[tuple[float, float]] = []
    for row in range(shoulder_row, side_search_end + 1):
        try:
            left_column = _outermost_signed_edge_index(
                gradient_x[row, left_start : left_end + 1],
                start_index=left_start,
                polarity="negative",
            )
            right_column = _outermost_signed_edge_index(
                gradient_x[row, right_start : right_end + 1],
                start_index=right_start,
                polarity="positive",
            )
        except RuntimeError:
            continue
        left_samples.append((float(left_column), float(row)))
        right_samples.append((float(right_column), float(row)))

    left_line = _robust_fit_x_from_y(left_samples)
    right_line = _robust_fit_x_from_y(right_samples)

    midpoint_row = 0.5 * float(shoulder_row + side_search_end)
    left_at_midpoint = left_line[0] * midpoint_row + left_line[1]
    right_at_midpoint = right_line[0] * midpoint_row + right_line[1]
    if right_at_midpoint - left_at_midpoint < 8.0:
        raise RuntimeError("Outer RGB mouth edges are implausibly close.")

    return left_line, right_line


def aperture_center_pixel(
    rgb: np.ndarray,
    mask: np.ndarray,
    camera,
) -> np.ndarray:
    """Return the projective center bounded by the physical outer mouth edges."""

    _, _, top, bottom = _front_rim_lines(rgb, mask, camera)
    left, right = _outer_front_side_lines(rgb, mask, camera)
    center = _projective_center_from_lines(left, right, top, bottom)

    if not (
        0.0 <= center[0] < float(camera.image_width_px)
        and 0.0 <= center[1] < float(camera.image_height_px)
    ):
        raise RuntimeError("Outer front-mouth center is outside the image.")
    return center
