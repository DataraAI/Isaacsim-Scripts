#!/usr/bin/env python3
"""Direct stereo reconstruction of the calibrated physical RJ45 front-rim center."""

from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np

from stereo_geometry import triangulate_pixel_pair


MAX_RAY_GAP_M = 0.0005
MAX_REPROJECTION_PX = 1.5

# Dimensionless feature calibration from the qualified horizontal-view dataset.
# The same front-rim line detector is used at runtime. These are not world-space
# offsets and do not depend on the rack pose or camera angle.
FRONT_RIM_CENTER_HORIZONTAL_FRACTION = 0.492
FRONT_RIM_CENTER_VERTICAL_FRACTION = 0.550


@dataclass(frozen=True)
class StereoApertureCenter:
    """One 3D center reconstructed from corresponding RGB front-rim centers."""

    center_world_m: np.ndarray
    left_center_uv: np.ndarray
    right_center_uv: np.ndarray
    ray_gap_m: float
    reprojection_rms_px: float
    max_reprojection_px: float

    def __post_init__(self) -> None:
        center = np.asarray(self.center_world_m, dtype=np.float64).reshape(3)
        left = np.asarray(self.left_center_uv, dtype=np.float64).reshape(2)
        right = np.asarray(self.right_center_uv, dtype=np.float64).reshape(2)
        diagnostics = (
            float(self.ray_gap_m),
            float(self.reprojection_rms_px),
            float(self.max_reprojection_px),
        )
        if not np.all(np.isfinite(center)):
            raise ValueError("center_world_m must be finite.")
        if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
            raise ValueError("Stereo front-rim pixels must be finite.")
        if not all(
            math.isfinite(value) and value >= 0.0
            for value in diagnostics
        ):
            raise ValueError(
                "Stereo front-rim diagnostics must be finite and nonnegative."
            )
        object.__setattr__(self, "center_world_m", center.copy())
        object.__setattr__(self, "left_center_uv", left.copy())
        object.__setattr__(self, "right_center_uv", right.copy())


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


def _gray_camera_image(rgb: np.ndarray, camera) -> np.ndarray:
    image = np.asarray(rgb)
    expected = (int(camera.image_height_px), int(camera.image_width_px))
    if image.ndim != 3 or image.shape[:2] != expected or image.shape[2] < 3:
        raise ValueError(
            f"RGB image must have shape {expected + (3,)}, got {image.shape}."
        )

    image = np.ascontiguousarray(
        image[:, :, :3].astype(np.uint8, copy=False)
    )
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return cv2.GaussianBlur(gray, (3, 3), 0).astype(np.float64)


def _robust_fit_x_from_y(
    points: list[tuple[float, float]],
) -> tuple[float, float]:
    samples = np.asarray(points, dtype=np.float64)
    if samples.shape[0] < 6:
        raise RuntimeError("Too few front-rim side-wall samples.")

    x = samples[:, 0]
    y = samples[:, 1]
    keep = np.ones(samples.shape[0], dtype=bool)

    for _ in range(5):
        if int(np.count_nonzero(keep)) < 6:
            raise RuntimeError("Front-rim side-wall fit lost support.")
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
        raise RuntimeError("Too few horizontal front-rim samples.")

    x = samples[:, 0]
    y = samples[:, 1]
    keep = np.ones(samples.shape[0], dtype=bool)

    for _ in range(5):
        if int(np.count_nonzero(keep)) < 6:
            raise RuntimeError("Horizontal front-rim fit lost support.")
        slope, intercept = np.polyfit(x[keep], y[keep], 1)
        residual = np.abs(y - (slope * x + intercept))
        median = float(np.median(residual[keep]))
        mad = float(np.median(np.abs(residual[keep] - median)))
        keep = residual <= max(1.0, median + 2.5 * mad)

    return float(slope), float(intercept)


def _front_rim_lines(
    rgb: np.ndarray,
    mask: np.ndarray,
    camera,
) -> tuple[
    tuple[float, float],
    tuple[float, float],
    tuple[float, float],
    tuple[float, float],
]:
    """
    Fit left, right, top, and bottom lines of the front opening.

    The dark semantic mask is used only to localize the port. The actual lines
    come from RGB bezel-to-opening gradients, so a recessed cavity silhouette
    cannot create a false stereo disparity.
    """

    component = _largest_component_mask(mask, camera)
    gray = _gray_camera_image(rgb, camera)
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
    mask_top = int(support_rows[0])
    mask_bottom = int(support_rows[-1])
    mask_height = mask_bottom - mask_top + 1

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
            "Aperture support is too small for RGB front-rim fitting: "
            f"{mask_width}x{mask_height}px."
        )

    gradient_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    side_search_end = min(
        gray.shape[0] - 2,
        mask_bottom + max(8, int(round(0.60 * mask_height))),
    )

    left_start = max(
        1,
        mask_left - max(5, int(round(0.35 * mask_width))),
    )
    left_end = min(
        gray.shape[1] - 2,
        mask_left + max(3, int(round(0.10 * mask_width))),
    )
    right_start = max(
        1,
        mask_right - max(3, int(round(0.10 * mask_width))),
    )
    right_end = min(
        gray.shape[1] - 2,
        mask_right + max(4, int(round(0.22 * mask_width))),
    )

    left_samples: list[tuple[float, float]] = []
    right_samples: list[tuple[float, float]] = []
    for row in range(shoulder_row, side_search_end + 1):
        left_slice = gradient_x[row, left_start : left_end + 1]
        right_slice = gradient_x[row, right_start : right_end + 1]
        left_column = left_start + int(np.argmin(left_slice))
        right_column = right_start + int(np.argmax(right_slice))
        left_samples.append((float(left_column), float(row)))
        right_samples.append((float(right_column), float(row)))

    left_line = _robust_fit_x_from_y(left_samples)
    right_line = _robust_fit_x_from_y(right_samples)

    gradient_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    top_rows = [
        geometry
        for geometry in rows
        if geometry[0] <= mask_top + max(5, int(round(0.25 * mask_height)))
    ]
    central_left = int(
        np.percentile([left for _, left, _, _ in top_rows], 25.0)
    )
    central_right = int(
        np.percentile([right for _, _, right, _ in top_rows], 75.0)
    )
    top_start = max(1, mask_top - 5)
    top_end = min(gray.shape[0] - 2, mask_top + 6)

    top_samples: list[tuple[float, float]] = []
    for column in range(central_left + 1, central_right):
        values = gradient_y[top_start : top_end + 1, column]
        row = top_start + int(np.argmin(values))
        top_samples.append((float(column), float(row)))
    top_line = _robust_fit_y_from_x(top_samples)

    left_at_shoulder = left_line[0] * shoulder_row + left_line[1]
    right_at_shoulder = right_line[0] * shoulder_row + right_line[1]
    front_width = right_at_shoulder - left_at_shoulder
    if front_width < 8.0:
        raise RuntimeError("RGB front-rim side walls are implausibly close.")

    interior_left = int(round(left_at_shoulder + 0.18 * front_width))
    interior_right = int(round(left_at_shoulder + 0.82 * front_width))
    interior_left = max(1, interior_left)
    interior_right = min(gray.shape[1] - 2, interior_right)
    if interior_right - interior_left < 6:
        raise RuntimeError("RGB front-rim interior band is too narrow.")

    profile = np.median(
        gray[:, interior_left : interior_right + 1],
        axis=1,
    )
    bottom_search_end = min(
        gray.shape[0] - 3,
        mask_bottom + max(16, int(round(1.10 * mask_height))),
    )
    if bottom_search_end <= mask_bottom + 4:
        raise RuntimeError("No RGB support below the aperture mask.")

    dark_end = min(
        mask_bottom,
        shoulder_row + max(6, int(round(0.35 * mask_height))),
    )
    dark_level = float(
        np.percentile(
            profile[shoulder_row + 3 : dark_end + 1],
            30.0,
        )
    )
    bright_start = min(
        bottom_search_end,
        mask_bottom + max(6, int(round(0.32 * mask_height))),
    )
    bright_level = float(
        np.percentile(
            profile[bright_start : bottom_search_end + 1],
            90.0,
        )
    )
    if bright_level - dark_level < 20.0:
        raise RuntimeError(
            "RGB front-rim lower boundary has insufficient contrast."
        )

    threshold = dark_level + 0.90 * (bright_level - dark_level)
    bottom_guess = next(
        (
            row
            for row in range(mask_bottom + 1, bottom_search_end - 1)
            if np.all(profile[row : row + 3] >= threshold)
        ),
        None,
    )
    if bottom_guess is None:
        raise RuntimeError("Could not locate the RGB front-rim lower boundary.")

    bottom_samples: list[tuple[float, float]] = []
    first_column = int(round(left_at_shoulder)) + 4
    last_column = int(round(right_at_shoulder)) - 4
    for column in range(first_column, last_column + 1):
        local_dark = float(
            np.percentile(
                gray[shoulder_row + 3 : dark_end + 1, column],
                30.0,
            )
        )
        local_bright = float(
            np.percentile(
                gray[bright_start : bottom_search_end + 1, column],
                90.0,
            )
        )
        local_threshold = local_dark + 0.90 * (
            local_bright - local_dark
        )
        row = next(
            (
                candidate
                for candidate in range(
                    mask_bottom + 1,
                    bottom_search_end - 1,
                )
                if np.all(
                    gray[candidate : candidate + 3, column]
                    >= local_threshold
                )
            ),
            None,
        )
        if row is not None and abs(row - bottom_guess) <= 5:
            bottom_samples.append((float(column), float(row)))

    bottom_line = _robust_fit_y_from_x(bottom_samples)
    return left_line, right_line, top_line, bottom_line


def aperture_center_pixel(
    rgb: np.ndarray,
    mask: np.ndarray,
    camera,
) -> np.ndarray:
    """Return the calibrated projective center of the physical front opening."""

    left, right, top, bottom = _front_rim_lines(rgb, mask, camera)

    vertical_fraction = FRONT_RIM_CENTER_VERTICAL_FRACTION
    horizontal_fraction = FRONT_RIM_CENTER_HORIZONTAL_FRACTION

    horizontal_slope = (
        (1.0 - vertical_fraction) * top[0]
        + vertical_fraction * bottom[0]
    )
    horizontal_intercept = (
        (1.0 - vertical_fraction) * top[1]
        + vertical_fraction * bottom[1]
    )
    side_slope = (
        (1.0 - horizontal_fraction) * left[0]
        + horizontal_fraction * right[0]
    )
    side_intercept = (
        (1.0 - horizontal_fraction) * left[1]
        + horizontal_fraction * right[1]
    )

    denominator = 1.0 - side_slope * horizontal_slope
    if abs(denominator) <= 1.0e-9:
        raise RuntimeError("Calibrated front-rim center lines are parallel.")

    u = (
        side_slope * horizontal_intercept + side_intercept
    ) / denominator
    v = horizontal_slope * u + horizontal_intercept
    center = np.array([u, v], dtype=np.float64)

    if not np.all(np.isfinite(center)):
        raise RuntimeError("Calibrated RGB front-rim center is not finite.")
    if not (
        0.0 <= center[0] < float(camera.image_width_px)
        and 0.0 <= center[1] < float(camera.image_height_px)
    ):
        raise RuntimeError("Calibrated RGB front-rim center is outside the image.")
    return center


def estimate_stereo_aperture_center(
    *,
    left_rgb: np.ndarray,
    right_rgb: np.ndarray,
    left_mask: np.ndarray,
    right_mask: np.ndarray,
    left_camera,
    right_camera,
    max_ray_gap_m: float = MAX_RAY_GAP_M,
    max_reprojection_px: float = MAX_REPROJECTION_PX,
) -> StereoApertureCenter:
    """Triangulate corresponding calibrated RGB front-rim centers."""

    maximum_gap = float(max_ray_gap_m)
    maximum_reprojection = float(max_reprojection_px)
    if (
        not math.isfinite(maximum_gap)
        or not 0.0 < maximum_gap <= MAX_RAY_GAP_M
    ):
        raise ValueError(
            "Stereo center ray-gap gate must be in (0, 0.5 mm]."
        )
    if (
        not math.isfinite(maximum_reprojection)
        or maximum_reprojection <= 0.0
    ):
        raise ValueError("Stereo center reprojection gate must be positive.")

    left_uv = aperture_center_pixel(left_rgb, left_mask, left_camera)
    right_uv = aperture_center_pixel(right_rgb, right_mask, right_camera)
    center_world, ray_gap = triangulate_pixel_pair(
        left_uv,
        right_uv,
        left_camera,
        right_camera,
    )
    if ray_gap > maximum_gap:
        raise RuntimeError(
            "Stereo RGB front-rim center ray gap is "
            f"{ray_gap * 1000.0:.3f} mm; "
            f"limit is {maximum_gap * 1000.0:.3f} mm."
        )

    reprojection_errors = np.asarray(
        [
            np.linalg.norm(
                left_camera.project_world(center_world) - left_uv
            ),
            np.linalg.norm(
                right_camera.project_world(center_world) - right_uv
            ),
        ],
        dtype=np.float64,
    )
    maximum_error = float(np.max(reprojection_errors))
    rms_error = float(
        np.sqrt(np.mean(reprojection_errors * reprojection_errors))
    )
    if maximum_error > maximum_reprojection:
        raise RuntimeError(
            "Stereo RGB front-rim center reprojection error is "
            f"{maximum_error:.3f} px; "
            f"limit is {maximum_reprojection:.3f} px."
        )

    return StereoApertureCenter(
        center_world_m=center_world,
        left_center_uv=left_uv,
        right_center_uv=right_uv,
        ray_gap_m=ray_gap,
        reprojection_rms_px=rms_error,
        max_reprojection_px=maximum_error,
    )
