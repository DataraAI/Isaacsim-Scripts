#!/usr/bin/env python3
"""Perspective-invariant aperture-center estimation on a measured front plane."""

from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np

from front_plane import intersect_pixel_with_plane
from stereo_geometry import unit_vector


MAX_CENTER_DISAGREEMENT_M = 0.0005


@dataclass(frozen=True)
class PlanarApertureCenter:
    """Fused physical aperture center reconstructed independently by both eyes."""

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


def _plane_basis(left_camera, right_camera, normal_world: np.ndarray):
    normal = unit_vector(normal_world, "front-plane normal")
    baseline = (
        np.asarray(right_camera.camera_center_world_m, dtype=np.float64)
        - np.asarray(left_camera.camera_center_world_m, dtype=np.float64)
    )
    horizontal = baseline - float(np.dot(baseline, normal)) * normal
    horizontal = unit_vector(horizontal, "front-plane stereo baseline")
    vertical = unit_vector(np.cross(normal, horizontal), "front-plane vertical")
    return horizontal, vertical, normal


def _polygon_centroid(points_xy: np.ndarray) -> np.ndarray:
    points = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    if points.shape[0] < 3:
        raise RuntimeError("Aperture contour has fewer than three points.")
    following = np.roll(points, -1, axis=0)
    cross = points[:, 0] * following[:, 1] - following[:, 0] * points[:, 1]
    area_twice = float(np.sum(cross))
    if abs(area_twice) <= 1.0e-12:
        raise RuntimeError("Aperture contour has negligible rectified area.")
    center = np.array(
        [
            np.sum((points[:, 0] + following[:, 0]) * cross),
            np.sum((points[:, 1] + following[:, 1]) * cross),
        ],
        dtype=np.float64,
    ) / (3.0 * area_twice)
    if not np.all(np.isfinite(center)):
        raise RuntimeError("Aperture contour center is not finite.")
    return center


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
    return contour


def _center_from_eye(
    mask: np.ndarray,
    camera,
    plane_origin_world_m: np.ndarray,
    normal_world: np.ndarray,
    horizontal_world: np.ndarray,
    vertical_world: np.ndarray,
) -> np.ndarray:
    contour = _largest_contour(mask)
    x, y, width, height = cv2.boundingRect(contour)
    if width < 3 or height < 3:
        raise RuntimeError("Aperture contour bounding box is too small.")

    # Any four non-collinear image points and their ray/plane intersections
    # define the exact pinhole image-to-plane homography. The bounding corners
    # are used only to solve that calibration transform; they do not define the
    # aperture center.
    image_quad = np.array(
        [
            [x, y],
            [x + width - 1, y],
            [x + width - 1, y + height - 1],
            [x, y + height - 1],
        ],
        dtype=np.float64,
    )
    plane_origin = np.asarray(plane_origin_world_m, dtype=np.float64).reshape(3)
    plane_quad = []
    for pixel_uv in image_quad:
        point_world = intersect_pixel_with_plane(
            camera,
            pixel_uv,
            plane_origin,
            normal_world,
        )
        delta = point_world - plane_origin
        plane_quad.append(
            [
                float(np.dot(delta, horizontal_world)),
                float(np.dot(delta, vertical_world)),
            ]
        )
    image_to_plane = cv2.getPerspectiveTransform(
        image_quad.astype(np.float32),
        np.asarray(plane_quad, dtype=np.float32),
    )
    contour_xy = cv2.perspectiveTransform(
        contour.astype(np.float32),
        image_to_plane,
    ).reshape(-1, 2)
    center_xy = _polygon_centroid(contour_xy)
    return (
        plane_origin
        + center_xy[0] * horizontal_world
        + center_xy[1] * vertical_world
    )


def estimate_planar_aperture_center(
    *,
    left_mask: np.ndarray,
    right_mask: np.ndarray,
    left_camera,
    right_camera,
    plane_origin_world_m: np.ndarray,
    plane_normal_world: np.ndarray,
    max_disagreement_m: float = MAX_CENTER_DISAGREEMENT_M,
) -> PlanarApertureCenter:
    """Rectify both stepped masks onto the measured plane and fuse their centers."""
    maximum = float(max_disagreement_m)
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
    left_center = _center_from_eye(
        left_mask,
        left_camera,
        plane_origin_world_m,
        normal,
        horizontal,
        vertical,
    )
    right_center = _center_from_eye(
        right_mask,
        right_camera,
        plane_origin_world_m,
        normal,
        horizontal,
        vertical,
    )
    disagreement = float(np.linalg.norm(left_center - right_center))
    if disagreement > maximum:
        raise RuntimeError(
            "Plane-rectified aperture centers disagree by "
            f"{disagreement * 1000.0:.3f} mm; "
            f"limit is {maximum * 1000.0:.3f} mm."
        )
    return PlanarApertureCenter(
        center_world_m=0.5 * (left_center + right_center),
        left_center_world_m=left_center,
        right_center_world_m=right_center,
        left_right_disagreement_m=disagreement,
    )
