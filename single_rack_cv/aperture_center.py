#!/usr/bin/env python3
"""Perspective-invariant physical center of the stepped RJ45 aperture."""

from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np

from front_plane import intersect_pixel_with_plane
from stereo_geometry import unit_vector


MAX_CENTER_DISAGREEMENT_M = 0.0005
DEFAULT_APERTURE_WIDTH_M = 0.0114
DEFAULT_APERTURE_HEIGHT_M = 0.0070
_TOP_BAND_FRACTION = 0.20
_EDGE_PERCENTILE = 2.0
_BOTTOM_PERCENTILE = 5.0
_TOP_PERCENTILE = 99.0


@dataclass(frozen=True)
class PlanarApertureCenter:
    """Fused insertion center reconstructed independently by both eyes."""

    center_world_m: np.ndarray
    left_center_world_m: np.ndarray
    right_center_world_m: np.ndarray
    left_right_disagreement_m: float

    def __post_init__(self) -> None:
        center = np.asarray(self.center_world_m, dtype=np.float64).reshape(3)
        left = np.asarray(self.left_center_world_m, dtype=np.float64).reshape(3)
        right = np.asarray(self.right_center_world_m, dtype=np.float64).reshape(3)
        disagreement = float(self.left_right_disagreement_m)
        if not np.all(np.isfinite(center)):
            raise ValueError("center_world_m must be finite.")
        if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
            raise ValueError("Per-eye aperture centers must be finite.")
        if not math.isfinite(disagreement) or disagreement < 0.0:
            raise ValueError("left_right_disagreement_m must be nonnegative.")
        object.__setattr__(self, "center_world_m", center.copy())
        object.__setattr__(self, "left_center_world_m", left.copy())
        object.__setattr__(self, "right_center_world_m", right.copy())


@dataclass(frozen=True)
class _EyeApertureMeasurement:
    center_world_m: np.ndarray
    full_width_m: float
    visible_height_m: float
    notch_width_m: float


def _camera_image_up_world(camera) -> np.ndarray:
    center_uv = np.array([float(camera.cx_px), float(camera.cy_px)])
    _, center_direction = camera.pixel_to_world_ray(center_uv)
    _, upper_direction = camera.pixel_to_world_ray(center_uv + [0.0, -10.0])
    center_direction = unit_vector(center_direction, "camera center ray")
    upper_direction = unit_vector(upper_direction, "camera upper ray")
    image_up = upper_direction - float(
        np.dot(upper_direction, center_direction)
    ) * center_direction
    return unit_vector(image_up, "camera image-up direction")


def _plane_basis(left_camera, right_camera, normal_world: np.ndarray):
    normal = unit_vector(normal_world, "front-plane normal")
    baseline = (
        np.asarray(right_camera.camera_center_world_m, dtype=np.float64)
        - np.asarray(left_camera.camera_center_world_m, dtype=np.float64)
    )
    horizontal = baseline - float(np.dot(baseline, normal)) * normal
    horizontal = unit_vector(horizontal, "front-plane stereo baseline")
    vertical = unit_vector(np.cross(normal, horizontal), "front-plane vertical")

    image_up = unit_vector(
        _camera_image_up_world(left_camera)
        + _camera_image_up_world(right_camera),
        "mean camera image-up direction",
    )
    image_up_on_plane = image_up - float(np.dot(image_up, normal)) * normal
    image_up_on_plane = unit_vector(image_up_on_plane, "projected image-up direction")
    if float(np.dot(vertical, image_up_on_plane)) < 0.0:
        horizontal = -horizontal
        vertical = -vertical
    return horizontal, vertical, normal


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

    # Detector masks are normally full resolution. Scaling keeps saved reduced
    # debug masks geometrically valid without changing the runtime path.
    contour[:, 0] = (
        (contour[:, 0] + 0.5) * image_width / mask_width - 0.5
    )
    contour[:, 1] = (
        (contour[:, 1] + 0.5) * image_height / mask_height - 0.5
    )
    return contour


def _measure_eye_center(
    *,
    mask: np.ndarray,
    camera,
    plane_origin_world_m: np.ndarray,
    normal_world: np.ndarray,
    horizontal_world: np.ndarray,
    vertical_world: np.ndarray,
    aperture_width_m: float,
    aperture_height_m: float,
) -> _EyeApertureMeasurement:
    plane_origin = np.asarray(plane_origin_world_m, dtype=np.float64).reshape(3)
    contour_uv = _contour_camera_pixels(mask, camera)
    contour_xy = []
    for pixel_uv in contour_uv:
        point_world = intersect_pixel_with_plane(
            camera,
            pixel_uv,
            plane_origin,
            normal_world,
        )
        delta = point_world - plane_origin
        contour_xy.append(
            [
                float(np.dot(delta, horizontal_world)),
                float(np.dot(delta, vertical_world)),
            ]
        )
    contour_xy = np.asarray(contour_xy, dtype=np.float64)
    if contour_xy.shape[0] < 12 or not np.all(np.isfinite(contour_xy)):
        raise RuntimeError("Too few finite rectified aperture contour points.")

    left_edge, right_edge = np.percentile(
        contour_xy[:, 0],
        [_EDGE_PERCENTILE, 100.0 - _EDGE_PERCENTILE],
    )
    bottom_edge = float(np.percentile(contour_xy[:, 1], _BOTTOM_PERCENTILE))
    top_edge = float(np.percentile(contour_xy[:, 1], _TOP_PERCENTILE))
    full_width_m = float(right_edge - left_edge)
    visible_height_m = float(top_edge - bottom_edge)
    if not (
        0.75 * aperture_width_m <= full_width_m <= 1.30 * aperture_width_m
    ):
        raise RuntimeError(
            "Rectified aperture width is implausible: "
            f"{full_width_m * 1000.0:.3f} mm."
        )
    if not (
        0.75 * aperture_height_m
        <= visible_height_m
        <= 1.75 * aperture_height_m
    ):
        raise RuntimeError(
            "Rectified stepped-aperture height is implausible: "
            f"{visible_height_m * 1000.0:.3f} mm."
        )

    top_band_floor = top_edge - _TOP_BAND_FRACTION * visible_height_m
    top_band = contour_xy[contour_xy[:, 1] >= top_band_floor]
    if top_band.shape[0] < 8:
        raise RuntimeError("Top latch-notch contour has insufficient support.")
    notch_left, notch_right = np.percentile(
        top_band[:, 0],
        [_EDGE_PERCENTILE, 100.0 - _EDGE_PERCENTILE],
    )
    notch_width_m = float(notch_right - notch_left)
    if not (
        0.20 * aperture_width_m
        <= notch_width_m
        <= 0.75 * aperture_width_m
    ):
        raise RuntimeError(
            "Rectified latch-notch width is implausible: "
            f"{notch_width_m * 1000.0:.3f} mm."
        )

    # Horizontal center is the stepped latch-notch symmetry axis. Vertical
    # center is half the known physical opening height above the visible bottom
    # boundary. These are object dimensions, not a view-specific world offset.
    center_x = 0.5 * float(notch_left + notch_right)
    center_y = bottom_edge + 0.5 * aperture_height_m
    center_world = (
        plane_origin
        + center_x * horizontal_world
        + center_y * vertical_world
    )
    return _EyeApertureMeasurement(
        center_world_m=center_world,
        full_width_m=full_width_m,
        visible_height_m=visible_height_m,
        notch_width_m=notch_width_m,
    )


def estimate_planar_aperture_center(
    *,
    left_mask: np.ndarray,
    right_mask: np.ndarray,
    left_camera,
    right_camera,
    plane_origin_world_m: np.ndarray,
    plane_normal_world: np.ndarray,
    aperture_width_m: float = DEFAULT_APERTURE_WIDTH_M,
    aperture_height_m: float = DEFAULT_APERTURE_HEIGHT_M,
    max_disagreement_m: float = MAX_CENTER_DISAGREEMENT_M,
) -> PlanarApertureCenter:
    """Fuse the physical insertion center measured from both stepped masks."""
    width = float(aperture_width_m)
    height = float(aperture_height_m)
    maximum = float(max_disagreement_m)
    if not math.isfinite(width) or width <= 0.0:
        raise ValueError("aperture_width_m must be finite and positive.")
    if not math.isfinite(height) or height <= 0.0:
        raise ValueError("aperture_height_m must be finite and positive.")
    if (
        not math.isfinite(maximum)
        or maximum <= 0.0
        or maximum > MAX_CENTER_DISAGREEMENT_M
    ):
        raise ValueError("Aperture-center disagreement gate must be in (0, 0.5 mm].")

    horizontal, vertical, normal = _plane_basis(
        left_camera,
        right_camera,
        plane_normal_world,
    )
    shared = dict(
        plane_origin_world_m=plane_origin_world_m,
        normal_world=normal,
        horizontal_world=horizontal,
        vertical_world=vertical,
        aperture_width_m=width,
        aperture_height_m=height,
    )
    left = _measure_eye_center(mask=left_mask, camera=left_camera, **shared)
    right = _measure_eye_center(mask=right_mask, camera=right_camera, **shared)
    disagreement = float(
        np.linalg.norm(left.center_world_m - right.center_world_m)
    )
    if disagreement > maximum:
        raise RuntimeError(
            "Plane-rectified physical aperture centers disagree by "
            f"{disagreement * 1000.0:.3f} mm; "
            f"limit is {maximum * 1000.0:.3f} mm."
        )
    return PlanarApertureCenter(
        center_world_m=0.5 * (left.center_world_m + right.center_world_m),
        left_center_world_m=left.center_world_m,
        right_center_world_m=right.center_world_m,
        left_right_disagreement_m=disagreement,
    )
