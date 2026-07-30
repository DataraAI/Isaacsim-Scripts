#!/usr/bin/env python3
"""Direct calibrated stereo reconstruction of the physical RJ45 aperture center."""

from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np

from stereo_geometry import triangulate_pixel_pair


MAX_RAY_GAP_M = 0.0005
MAX_REPROJECTION_PX = 1.5
_EDGE_PERCENTILE = 2.0


@dataclass(frozen=True)
class StereoApertureCenter:
    """One 3D center reconstructed directly from corresponding mask centers."""

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
            raise ValueError("Stereo aperture pixels must be finite.")
        if not all(
            math.isfinite(value) and value >= 0.0
            for value in diagnostics
        ):
            raise ValueError(
                "Stereo aperture diagnostics must be finite and nonnegative."
            )
        object.__setattr__(self, "center_world_m", center.copy())
        object.__setattr__(self, "left_center_uv", left.copy())
        object.__setattr__(self, "right_center_uv", right.copy())


def _largest_contour(mask: np.ndarray) -> np.ndarray:
    binary = np.where(np.asarray(mask) > 0, 255, 0).astype(np.uint8)
    if binary.ndim != 2:
        raise ValueError("Aperture mask must be a 2D array.")
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
    return contour.reshape(-1, 2).astype(np.float64)


def _contour_camera_pixels(mask: np.ndarray, camera) -> np.ndarray:
    contour = _largest_contour(mask)
    mask_height, mask_width = map(int, np.asarray(mask).shape)
    image_height = int(camera.image_height_px)
    image_width = int(camera.image_width_px)
    if min(mask_height, mask_width, image_height, image_width) <= 0:
        raise ValueError("Mask and camera dimensions must be positive.")

    contour[:, 0] = (
        (contour[:, 0] + 0.5) * image_width / mask_width - 0.5
    )
    contour[:, 1] = (
        (contour[:, 1] + 0.5) * image_height / mask_height - 0.5
    )
    return contour


def aperture_center_pixel(mask: np.ndarray, camera) -> np.ndarray:
    """Return the projective center of the full physical aperture extents."""

    contour = _contour_camera_pixels(mask, camera)
    if contour.shape[0] < 12 or not np.all(np.isfinite(contour)):
        raise RuntimeError("Too few finite aperture contour points.")

    left, right = np.percentile(
        contour[:, 0],
        [_EDGE_PERCENTILE, 100.0 - _EDGE_PERCENTILE],
    )
    top, bottom = np.percentile(
        contour[:, 1],
        [_EDGE_PERCENTILE, 100.0 - _EDGE_PERCENTILE],
    )
    width_px = float(right - left)
    height_px = float(bottom - top)
    if width_px < 8.0 or height_px < 4.0:
        raise RuntimeError(
            "Aperture contour is too small for calibrated stereo: "
            f"{width_px:.2f}x{height_px:.2f}px."
        )

    return np.array(
        [
            0.5 * float(left + right),
            0.5 * float(top + bottom),
        ],
        dtype=np.float64,
    )


def estimate_stereo_aperture_center(
    *,
    left_mask: np.ndarray,
    right_mask: np.ndarray,
    left_camera,
    right_camera,
    max_ray_gap_m: float = MAX_RAY_GAP_M,
    max_reprojection_px: float = MAX_REPROJECTION_PX,
) -> StereoApertureCenter:
    """Triangulate corresponding physical aperture-center pixels directly."""

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

    left_uv = aperture_center_pixel(left_mask, left_camera)
    right_uv = aperture_center_pixel(right_mask, right_camera)
    center_world, ray_gap = triangulate_pixel_pair(
        left_uv,
        right_uv,
        left_camera,
        right_camera,
    )
    if ray_gap > maximum_gap:
        raise RuntimeError(
            "Stereo aperture-center ray gap is "
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
            "Stereo aperture-center reprojection error is "
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
