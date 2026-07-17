#!/usr/bin/env python3
"""Pure NumPy/OpenCV stereo RGB cable perception and servo geometry."""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class PerceptionConfig:
    """Elongated dark-cable detection plus calibrated stereo geometry."""

    roi_uv: tuple[int, int, int, int] | None = None

    # Otsu's method picks the dark/light split point from the image's own
    # brightness histogram each frame, instead of a hand-picked gray level.
    # These two just guard against a degenerate frame (e.g. all one
    # brightness) producing a nonsensical threshold.
    min_otsu_threshold: int = 20
    max_otsu_threshold: int = 200
    # Disabled while tuning start poses: a non-zero margin rejected valid
    # cable blobs that merely touched the bottom image border from a
    # slightly higher / laterally offset hand pose.
    edge_margin_px: int = 0

    # The cable's visible segment can be small (far away) or span most of
    # the frame (close up), and - since it can lie at any yaw on the table -
    # its axis-aligned bounding box doesn't reliably reflect true elongation
    # (a thin rod at 45 degrees has a roughly square bbox). So width/height
    # bounds stay loose; fill_ratio (how much of the bbox the blob actually
    # fills) is the more trustworthy elongation signal here, since a thin
    # shape fills only a small fraction of its bbox at almost any angle.
    min_width_px: int = 6
    max_width_px: int = 620
    min_height_px: int = 6
    max_height_px: int = 460
    # Aspect ratio (oriented long/short side) is a loose sanity gate only -
    # a cable segment's apparent length varies hugely with distance and
    # framing, so there's no single "correct" ratio to target.
    min_aspect_ratio: float = 1.5
    max_aspect_ratio: float = 60.0
    min_area_px: int = 60
    max_area_px: int = 60000
    # Fill ratio (blob area / oriented-bbox area) is the real elongation
    # signal here: a genuinely solid, rectangle-like blob fills close to
    # 100% of its own oriented bounding box at any rotation, while noise
    # (jagged edges, partial occlusion, shadows) fills noticeably less.
    min_fill_ratio: float = 0.45
    max_fill_ratio: float = 1.05

    # Aspect ratio isn't scored (weight 0) for the reason above; fill ratio
    # close to 1.0 does all the real work of saying "this looks solid."
    target_aspect_ratio: float = 6.0
    target_fill_ratio: float = 0.85
    aspect_score_tolerance: float = 10.0
    fill_score_tolerance: float = 0.40
    aspect_score_weight: float = 0.0
    fill_score_weight: float = 1.0
    min_shape_score: float = 0.30

    # Only cable_thickness_m reflects a genuinely fixed real quantity (cable
    # diameter / connector body thickness). The visible cable *length* in
    # frame is NOT fixed - it depends on how much cable is in view and at
    # what distance - so it is used only for the overlay's target-box size
    # (a cosmetic aid), never as a "this is the right size" comparison.
    #
    # PROVISIONAL: the detected blob is actually the whole RJ45 connector
    # body + mounting clip (visible in the reference photos), not a bare
    # cable cross-section. A back-of-envelope estimate from the camera's
    # known optics (18mm focal length, 640px width) and the detector's
    # consistently-measured ~80px short axis at our ~0.15-0.18m working
    # range gives roughly 22-26mm - not the 7mm a bare cable would be.
    # cable_thickness_m and the bounds below reflect that estimate; the
    # rejection messages now report the actual measured mm value on every
    # failure, so this can be tightened against real logged numbers rather
    # than re-estimated.
    cable_length_m: float = 0.0300
    cable_thickness_m: float = 0.0220
    min_estimated_range_m: float = 0.08
    max_estimated_range_m: float = 0.35

    # Stereo matching and corner refinement.
    stereo_corner_refine_window_px: int = 5
    stereo_max_corner_refine_shift_px: float = 6.0
    stereo_max_epipolar_error_px: float = 3.0
    stereo_max_scale_ratio: float = 1.30
    stereo_min_abs_disparity_px: float = 4.0
    stereo_max_ray_gap_m: float = 0.0020
    stereo_max_reprojection_rms_px: float = 1.0
    stereo_max_reprojection_px: float = 2.0
    stereo_max_plane_residual_m: float = 0.00075
    # Width (visible segment length) is deliberately loose - a cable
    # segment can be a couple centimeters or most of the frame, unlike a
    # port's fixed opening. Height (thickness) widened to comfortably
    # cover the connector+clip assembly estimate above, pending a real
    # calibrated number from the now-detailed rejection log messages.
    stereo_min_width_m: float = 0.010
    stereo_max_width_m: float = 0.300
    stereo_min_height_m: float = 0.005
    stereo_max_height_m: float = 0.035
    stereo_max_opposite_edge_ratio: float = 1.20

    # Once acquired, prefer image continuity over an unrelated blob with a
    # slightly better single-frame shape score.
    tracking_max_center_jump_px: float = 45.0
    tracking_max_scale_ratio: float = 1.35
    tracking_center_penalty: float = 0.35
    tracking_scale_penalty: float = 0.25


@dataclass(frozen=True)
class CameraModel:
    """Pinhole calibration plus the current USD camera-to-world matrix."""

    image_height_px: int
    image_width_px: int
    focal_length_mm: float
    horizontal_aperture_mm: float
    vertical_aperture_mm: float
    world_from_camera: np.ndarray

    def __post_init__(self) -> None:
        matrix = np.asarray(self.world_from_camera, dtype=np.float64)
        if matrix.shape != (4, 4):
            raise ValueError(
                f"world_from_camera must be 4x4, got {matrix.shape}."
            )
        object.__setattr__(self, "world_from_camera", matrix.copy())

    @property
    def fx_px(self) -> float:
        return (
            self.focal_length_mm
            * self.image_width_px
            / self.horizontal_aperture_mm
        )

    @property
    def fy_px(self) -> float:
        return (
            self.focal_length_mm
            * self.image_height_px
            / self.vertical_aperture_mm
        )

    @property
    def cx_px(self) -> float:
        return (self.image_width_px - 1) / 2.0

    @property
    def cy_px(self) -> float:
        return (self.image_height_px - 1) / 2.0

    @property
    def camera_from_world(self) -> np.ndarray:
        return np.linalg.inv(self.world_from_camera)

    @property
    def camera_center_world_m(self) -> np.ndarray:
        return transform_point_to_world(
            np.zeros(3, dtype=np.float64),
            self.world_from_camera,
        )

    def camera_point_from_world(self, point_world_m: np.ndarray) -> np.ndarray:
        point = np.append(
            np.asarray(point_world_m, dtype=np.float64).reshape(3),
            1.0,
        )
        local = point @ self.camera_from_world
        if abs(local[3]) > 1.0e-12:
            local = local / local[3]
        return local[:3]

    def project_world(self, point_world_m: np.ndarray) -> np.ndarray:
        point = self.camera_point_from_world(point_world_m)
        range_m = -float(point[2])
        if range_m <= 0.0:
            raise RuntimeError("World point is behind the camera.")
        u = self.cx_px + self.fx_px * float(point[0]) / range_m
        v = self.cy_px + self.fy_px * (-float(point[1])) / range_m
        return np.array([u, v], dtype=np.float64)

    def pixel_to_world_ray(
        self,
        pixel_uv: np.ndarray | tuple[float, float],
    ) -> tuple[np.ndarray, np.ndarray]:
        u, v = np.asarray(pixel_uv, dtype=np.float64).reshape(2)
        x = (u - self.cx_px) / self.fx_px
        y = -(v - self.cy_px) / self.fy_px
        local_direction = np.array([x, y, -1.0], dtype=np.float64)
        local_direction /= np.linalg.norm(local_direction)
        world_direction = (
            np.append(local_direction, 0.0) @ self.world_from_camera
        )[:3]
        world_direction /= np.linalg.norm(world_direction)
        return self.camera_center_world_m, world_direction


@dataclass(frozen=True)
class CameraFrame:
    rgb: np.ndarray
    camera: CameraModel


@dataclass(frozen=True)
class StereoFrame:
    left: CameraFrame
    right: CameraFrame
    virtual_camera: CameraModel


@dataclass(frozen=True)
class CableDetection:
    bbox_xywh: tuple[int, int, int, int]
    center_uv: tuple[float, float]
    shape_score: float
    roi_uv: tuple[int, int, int, int]
    mask: np.ndarray
    # (long, short) side lengths in pixels from the oriented minAreaRect
    # used during detection - unlike bbox_xywh, these stay meaningful at
    # any in-image rotation instead of ballooning for a diagonal object.
    oriented_size_px: tuple[float, float]

    @property
    def scale_px(self) -> float:
        _, _, width, height = self.bbox_xywh
        return math.sqrt(float(width * height))


@dataclass(frozen=True)
class CableCorners:
    detection: CableDetection
    corners_uv: np.ndarray

    def __post_init__(self) -> None:
        corners = np.asarray(self.corners_uv, dtype=np.float64)
        if corners.shape != (4, 2):
            raise ValueError(f"corners_uv must be (4,2), got {corners.shape}.")
        object.__setattr__(self, "corners_uv", corners.copy())


@dataclass(frozen=True)
class StereoTriangulation:
    corners_world_m: np.ndarray
    center_world_m: np.ndarray
    normal_world: np.ndarray
    width_m: float
    height_m: float
    reprojection_rms_px: float
    max_reprojection_px: float
    max_ray_gap_m: float
    plane_residual_m: float
    opposite_edge_ratio: float


@dataclass(frozen=True)
class StereoCableObservation:
    left: CableCorners
    right: CableCorners
    corners_world_m: np.ndarray
    center_world_xyz_m: np.ndarray
    center_virtual_camera_usd_m: np.ndarray
    normal_world: np.ndarray
    projected_virtual_center_uv: tuple[float, float]
    desired_center_uv: tuple[float, float]
    desired_size_wh_px: tuple[float, float]
    desired_left_center_uv: tuple[float, float]
    desired_right_center_uv: tuple[float, float]
    center_error_px: np.ndarray
    estimated_range_m: float
    range_error_m: float
    correction_world_m: np.ndarray
    width_m: float
    height_m: float
    mean_disparity_px: float
    reprojection_rms_px: float
    max_reprojection_px: float
    max_ray_gap_m: float
    plane_residual_m: float


# ---------------------------------------------------------------------------
# Image normalization and strict per-eye candidate detection
# ---------------------------------------------------------------------------


def normalize_rgb(
    rgb: np.ndarray,
    resolution_hw: tuple[int, int],
) -> np.ndarray:
    """Return contiguous HxWx3 uint8 RGB data."""
    height, width = resolution_hw
    rgb = np.asarray(rgb)

    if rgb.ndim == 1:
        pixels = height * width
        channels = 4 if rgb.size == pixels * 4 else 3
        if rgb.size != pixels * channels:
            raise ValueError(
                f"Cannot reshape flat RGB array of size {rgb.size}."
            )
        rgb = rgb.reshape(height, width, channels)

    if rgb.ndim != 3 or rgb.shape[:2] != (height, width):
        raise ValueError(
            f"RGB shape {rgb.shape} does not match {(height, width)}."
        )

    if rgb.shape[2] == 4:
        rgb = rgb[:, :, :3]
    elif rgb.shape[2] != 3:
        raise ValueError(f"RGB must have 3 or 4 channels, got {rgb.shape}.")

    if rgb.dtype != np.uint8:
        rgb = rgb.astype(np.float32, copy=False)
        if np.nanmax(rgb) <= 1.0:
            rgb = rgb * 255.0
        rgb = np.clip(rgb, 0.0, 255.0).astype(np.uint8)

    return np.ascontiguousarray(rgb)


def score_cable_shape(
    aspect_ratio: float,
    fill_ratio: float,
    cfg: PerceptionConfig,
) -> float:
    aspect_error = (
        abs(aspect_ratio - cfg.target_aspect_ratio)
        / cfg.target_aspect_ratio
    )
    fill_error = abs(fill_ratio - cfg.target_fill_ratio)
    aspect_score = max(
        0.0,
        1.0 - aspect_error / cfg.aspect_score_tolerance,
    )
    fill_score = max(
        0.0,
        1.0 - fill_error / cfg.fill_score_tolerance,
    )
    weight_sum = cfg.aspect_score_weight + cfg.fill_score_weight
    if weight_sum <= 0.0:
        raise ValueError("Shape-score weights must sum to a positive value.")
    return float(
        (
            cfg.aspect_score_weight * aspect_score
            + cfg.fill_score_weight * fill_score
        )
        / weight_sum
    )


def _otsu_dark_mask(
    gray_roi: np.ndarray,
    cfg: PerceptionConfig,
) -> np.ndarray:
    """
    Threshold the ROI into a dark-foreground mask using Otsu's method.

    Otsu's method treats the image's brightness histogram as a mix of two
    groups (here: cable vs. background) and picks the split point that best
    separates them - so it adapts to the actual scene instead of relying on
    one hand-picked gray level. cv2.THRESH_OTSU returns the threshold it
    found; THRESH_BINARY_INV marks pixels *darker* than that threshold as
    foreground (255), matching "the cable is the darker thing here."
    """
    threshold, binary = cv2.threshold(
        gray_roi,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )
    if not (cfg.min_otsu_threshold <= threshold <= cfg.max_otsu_threshold):
        # Otsu found an implausible split (e.g. a near-uniform frame) -
        # rather than trust a degenerate threshold, fall back to a plain
        # midpoint clamp so this frame yields "no candidates" instead of a
        # mask covering the whole image or none of it.
        clamped = float(
            np.clip(threshold, cfg.min_otsu_threshold, cfg.max_otsu_threshold)
        )
        _, binary = cv2.threshold(
            gray_roi,
            clamped,
            255,
            cv2.THRESH_BINARY_INV,
        )
    return binary


def detect_cable_candidates(
    rgb: np.ndarray,
    cfg: PerceptionConfig,
) -> list[CableDetection]:
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 RGB, got {rgb.shape}.")
    image_height, image_width = rgb.shape[:2]
    if cfg.roi_uv is None:
        u0, v0, u1, v1 = 0, 0, image_width, image_height
    else:
        u0, v0, u1, v1 = cfg.roi_uv
    if not (
        0 <= u0 < u1 <= image_width
        and 0 <= v0 < v1 <= image_height
    ):
        raise ValueError(
            f"Search area {(u0, v0, u1, v1)} is outside "
            f"image {image_width}x{image_height}."
        )
    roi = rgb[v0:v1, u0:u1]
    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    binary = _otsu_dark_mask(gray, cfg)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8,
    )
    full_mask = np.zeros((image_height, image_width), dtype=np.uint8)
    full_mask[v0:v1, u0:u1] = binary
    candidates: list[CableDetection] = []
    for index in range(1, count):
        x, y, width, height, area = map(int, stats[index])
        if width <= 0 or height <= 0:
            continue
        global_x = u0 + x
        global_y = v0 + y

        # Axis-aligned width/height/fill_ratio don't reflect true elongation
        # once the object isn't roughly axis-aligned in the image (a thin
        # diagonal rod's axis-aligned bbox is mostly empty corners). Fit an
        # oriented rectangle to this blob's own pixels instead, so aspect
        # ratio and fill ratio stay meaningful at any in-image rotation.
        component_crop = np.where(
            labels[y:y + height, x:x + width] == index,
            np.uint8(255),
            np.uint8(0),
        )
        contours, _ = cv2.findContours(
            component_crop,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        if not contours:
            continue
        largest_contour = max(contours, key=cv2.contourArea)
        (_, _), (rect_w, rect_h), _ = cv2.minAreaRect(largest_contour)
        if rect_w <= 0.0 or rect_h <= 0.0:
            continue
        oriented_long = max(rect_w, rect_h)
        oriented_short = min(rect_w, rect_h)
        aspect_ratio = oriented_long / oriented_short
        fill_ratio = area / (rect_w * rect_h)

        touches_edge = (
            global_x < cfg.edge_margin_px
            or global_y < cfg.edge_margin_px
            or global_x + width > image_width - cfg.edge_margin_px
            or global_y + height > image_height - cfg.edge_margin_px
        )
        accepted = (
            not touches_edge
            and cfg.min_width_px <= width <= cfg.max_width_px
            and cfg.min_height_px <= height <= cfg.max_height_px
            and cfg.min_aspect_ratio <= aspect_ratio <= cfg.max_aspect_ratio
            and cfg.min_area_px <= area <= cfg.max_area_px
            and cfg.min_fill_ratio <= fill_ratio <= cfg.max_fill_ratio
        )
        if not accepted:
            continue
        shape_score = score_cable_shape(aspect_ratio, fill_ratio, cfg)
        if shape_score < cfg.min_shape_score:
            continue
        local_center_u, local_center_v = centroids[index]
        candidates.append(
            CableDetection(
                bbox_xywh=(global_x, global_y, width, height),
                center_uv=(
                    float(u0 + local_center_u),
                    float(v0 + local_center_v),
                ),
                shape_score=shape_score,
                roi_uv=(u0, v0, u1, v1),
                mask=full_mask,
                oriented_size_px=(oriented_long, oriented_short),
            )
        )
    candidates.sort(key=lambda item: item.shape_score, reverse=True)
    return candidates


def select_cable_candidate(
    candidates: list[CableDetection],
    previous: CableDetection | None,
    cfg: PerceptionConfig,
) -> CableDetection:
    if not candidates:
        raise RuntimeError("No RGB cable candidate passed the shape filters.")
    if previous is None:
        return candidates[0]
    previous_center = np.asarray(previous.center_uv, dtype=np.float64)
    previous_scale = previous.scale_px
    max_log_scale = math.log(cfg.tracking_max_scale_ratio)
    tracked: list[tuple[float, CableDetection]] = []
    for candidate in candidates:
        center_distance = float(
            np.linalg.norm(
                np.asarray(candidate.center_uv, dtype=np.float64)
                - previous_center
            )
        )
        scale_ratio = max(
            candidate.scale_px / previous_scale,
            previous_scale / candidate.scale_px,
        )
        if center_distance > cfg.tracking_max_center_jump_px:
            continue
        if scale_ratio > cfg.tracking_max_scale_ratio:
            continue
        center_penalty = (
            cfg.tracking_center_penalty
            * center_distance
            / cfg.tracking_max_center_jump_px
        )
        scale_penalty = (
            cfg.tracking_scale_penalty
            * abs(math.log(scale_ratio))
            / max_log_scale
        )
        tracked.append(
            (candidate.shape_score - center_penalty - scale_penalty, candidate)
        )
    if not tracked:
        raise RuntimeError(
            "RGB cable track was lost: no candidate passed the "
            "center/scale continuity gates."
        )
    return max(tracked, key=lambda item: item[0])[1]


# ---------------------------------------------------------------------------
# Camera and servo geometry
# ---------------------------------------------------------------------------


def compute_desired_cable_camera_usd(
    camera_position_hand_m: np.ndarray,
    hand_from_camera: np.ndarray,
    tool_center_position_hand_m: np.ndarray,
    hand_from_tool: np.ndarray,
    grasp_standoff_m: float,
) -> np.ndarray:
    camera_position = np.asarray(
        camera_position_hand_m,
        dtype=np.float64,
    ).reshape(3)
    tool_position = np.asarray(
        tool_center_position_hand_m,
        dtype=np.float64,
    ).reshape(3)
    hand_from_camera = np.asarray(
        hand_from_camera,
        dtype=np.float64,
    ).reshape(3, 3)
    hand_from_tool = np.asarray(
        hand_from_tool,
        dtype=np.float64,
    ).reshape(3, 3)
    if not math.isfinite(grasp_standoff_m) or grasp_standoff_m <= 0.0:
        raise ValueError("grasp_standoff_m must be finite and positive.")
    cable_in_tool = np.array([0.0, 0.0, grasp_standoff_m])
    cable_in_hand = tool_position + hand_from_tool @ cable_in_tool
    cable_in_camera = hand_from_camera.T @ (cable_in_hand - camera_position)
    if cable_in_camera[2] >= 0.0:
        raise RuntimeError(
            "Configured pre-insert point is not in front of the camera: "
            f"{np.round(cable_in_camera, 6).tolist()}"
        )
    return cable_in_camera


def project_cable_feature(
    point_camera_usd_m: np.ndarray,
    camera: CameraModel,
    cfg: PerceptionConfig,
) -> tuple[tuple[float, float], tuple[float, float]]:
    point = np.asarray(point_camera_usd_m, dtype=np.float64).reshape(3)
    range_m = -float(point[2])
    if range_m <= 0.0:
        raise ValueError("Desired cable point must be in front of the camera.")
    u = camera.cx_px + camera.fx_px * float(point[0]) / range_m
    v = camera.cy_px + camera.fy_px * (-float(point[1])) / range_m
    width_px = camera.fx_px * cfg.cable_length_m / range_m
    height_px = camera.fy_px * cfg.cable_thickness_m / range_m
    return (float(u), float(v)), (float(width_px), float(height_px))


def transform_point_to_world(
    point_usd_local: np.ndarray,
    world_from_camera: np.ndarray,
) -> np.ndarray:
    matrix = np.asarray(world_from_camera, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"Expected a 4x4 camera matrix, got {matrix.shape}.")
    homogeneous = np.append(
        np.asarray(point_usd_local, dtype=np.float64),
        1.0,
    )
    world = homogeneous @ matrix
    if abs(world[3]) > 1.0e-12:
        world = world / world[3]
    return world[:3]


def camera_point_error_to_world(
    current_point_usd: np.ndarray,
    desired_point_usd: np.ndarray,
    world_from_camera: np.ndarray,
) -> np.ndarray:
    current = np.asarray(current_point_usd, dtype=np.float64).reshape(3)
    desired = np.asarray(desired_point_usd, dtype=np.float64).reshape(3)
    matrix = np.asarray(world_from_camera, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"Expected a 4x4 camera matrix, got {matrix.shape}.")
    local_motion = current - desired
    return np.asarray((np.append(local_motion, 0.0) @ matrix)[:3])


def compute_bounded_step(
    correction_world_m: np.ndarray,
    gain: float,
    max_step_m: float,
) -> np.ndarray:
    correction = np.asarray(correction_world_m, dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(correction)):
        raise ValueError("Visual correction must be finite.")
    if not math.isfinite(gain) or gain <= 0.0:
        raise ValueError("Control gain must be finite and positive.")
    if not math.isfinite(max_step_m) or max_step_m <= 0.0:
        raise ValueError("Maximum target step must be finite and positive.")
    step = gain * correction
    norm = float(np.linalg.norm(step))
    if norm > max_step_m:
        step *= max_step_m / norm
    return step


def build_virtual_camera_model(
    left: CameraModel,
    right: CameraModel,
) -> CameraModel:
    """Return a mathematical midpoint eye; no rendered sensor is created."""
    scalar_pairs = (
        (left.image_height_px, right.image_height_px),
        (left.image_width_px, right.image_width_px),
        (left.focal_length_mm, right.focal_length_mm),
        (left.horizontal_aperture_mm, right.horizontal_aperture_mm),
        (left.vertical_aperture_mm, right.vertical_aperture_mm),
    )
    for a, b in scalar_pairs:
        if not np.isclose(a, b, atol=1.0e-9):
            raise ValueError("Stereo cameras must have identical intrinsics.")
    left_rotation = left.world_from_camera[:3, :3]
    right_rotation = right.world_from_camera[:3, :3]
    if not np.allclose(left_rotation, right_rotation, atol=1.0e-8):
        raise ValueError("Stereo cameras must have parallel optical axes.")
    matrix = left.world_from_camera.copy()
    matrix[3, :3] = (
        left.camera_center_world_m + right.camera_center_world_m
    ) / 2.0
    return CameraModel(
        image_height_px=left.image_height_px,
        image_width_px=left.image_width_px,
        focal_length_mm=left.focal_length_mm,
        horizontal_aperture_mm=left.horizontal_aperture_mm,
        vertical_aperture_mm=left.vertical_aperture_mm,
        world_from_camera=matrix,
    )


def triangulate_pixel_pair(
    left_uv: np.ndarray | tuple[float, float],
    right_uv: np.ndarray | tuple[float, float],
    left_camera: CameraModel,
    right_camera: CameraModel,
) -> tuple[np.ndarray, float]:
    left_origin, left_direction = left_camera.pixel_to_world_ray(left_uv)
    right_origin, right_direction = right_camera.pixel_to_world_ray(right_uv)
    system = np.column_stack((left_direction, -right_direction))
    values, _, rank, _ = np.linalg.lstsq(
        system,
        right_origin - left_origin,
        rcond=None,
    )
    if rank < 2:
        raise RuntimeError("Stereo rays are parallel or numerically singular.")
    left_distance, right_distance = map(float, values)
    if left_distance <= 0.0 or right_distance <= 0.0:
        raise RuntimeError("Triangulated point lies behind a camera.")
    left_point = left_origin + left_distance * left_direction
    right_point = right_origin + right_distance * right_direction
    gap = float(np.linalg.norm(left_point - right_point))
    return (left_point + right_point) / 2.0, gap


def triangulate_detection_centers(
    left_detection: CableDetection,
    right_detection: CableDetection,
    left_camera: CameraModel,
    right_camera: CameraModel,
    virtual_camera: CameraModel,
) -> StereoTriangulation:
    """Triangulate the matched dark-cable center and build a diagnostic plane."""
    center_world, ray_gap = triangulate_pixel_pair(
        left_detection.center_uv,
        right_detection.center_uv,
        left_camera,
        right_camera,
    )
    reprojection_errors = np.asarray(
        [
            np.linalg.norm(
                left_camera.project_world(center_world)
                - np.asarray(left_detection.center_uv, dtype=np.float64)
            ),
            np.linalg.norm(
                right_camera.project_world(center_world)
                - np.asarray(right_detection.center_uv, dtype=np.float64)
            ),
        ],
        dtype=np.float64,
    )
    center_virtual = virtual_camera.camera_point_from_world(center_world)
    range_m = -float(center_virtual[2])
    if range_m <= 0.0:
        raise RuntimeError("Triangulated center lies behind the virtual camera.")

    # bbox_xywh is axis-aligned and inflates at diagonal angles (the same
    # issue already fixed for accept/reject scoring in detect_cable_
    # candidates); oriented_size_px stays meaningful at any rotation, so use
    # that here instead - this is what was actually causing "cable height"
    # to read the connector's long axis instead of its thickness.
    left_long_px, left_short_px = left_detection.oriented_size_px
    right_long_px, right_short_px = right_detection.oriented_size_px
    width_m = range_m * 0.5 * (
        left_long_px / left_camera.fx_px
        + right_long_px / right_camera.fx_px
    )
    height_m = range_m * 0.5 * (
        left_short_px / left_camera.fy_px
        + right_short_px / right_camera.fy_px
    )

    half_width = width_m / 2.0
    half_height = height_m / 2.0
    local_corners = np.asarray(
        [
            center_virtual + [-half_width, +half_height, 0.0],
            center_virtual + [+half_width, +half_height, 0.0],
            center_virtual + [+half_width, -half_height, 0.0],
            center_virtual + [-half_width, -half_height, 0.0],
        ],
        dtype=np.float64,
    )
    corners_world = np.vstack(
        [
            transform_point_to_world(point, virtual_camera.world_from_camera)
            for point in local_corners
        ]
    )
    normal_world = (
        np.asarray([0.0, 0.0, 1.0, 0.0])
        @ virtual_camera.world_from_camera
    )[:3]
    normal_world /= np.linalg.norm(normal_world)

    rms = float(np.sqrt(np.mean(reprojection_errors * reprojection_errors)))
    maximum = float(np.max(reprojection_errors))
    return StereoTriangulation(
        corners_world_m=corners_world,
        center_world_m=center_world,
        normal_world=normal_world,
        width_m=float(width_m),
        height_m=float(height_m),
        reprojection_rms_px=rms,
        max_reprojection_px=maximum,
        max_ray_gap_m=float(ray_gap),
        plane_residual_m=0.0,
        opposite_edge_ratio=1.0,
    )


def _edge_ratio(a: float, b: float) -> float:
    minimum = min(a, b)
    if minimum <= 1.0e-12:
        return math.inf
    return max(a, b) / minimum


def triangulate_cable_corners(
    left_corners_uv: np.ndarray,
    right_corners_uv: np.ndarray,
    left_camera: CameraModel,
    right_camera: CameraModel,
) -> StereoTriangulation:
    left_uv = np.asarray(left_corners_uv, dtype=np.float64)
    right_uv = np.asarray(right_corners_uv, dtype=np.float64)
    if left_uv.shape != (4, 2) or right_uv.shape != (4, 2):
        raise ValueError("Stereo corner arrays must both have shape (4,2).")
    points: list[np.ndarray] = []
    ray_gaps: list[float] = []
    reprojection_errors: list[float] = []
    for left_pixel, right_pixel in zip(left_uv, right_uv, strict=True):
        point, gap = triangulate_pixel_pair(
            left_pixel,
            right_pixel,
            left_camera,
            right_camera,
        )
        points.append(point)
        ray_gaps.append(gap)
        reprojection_errors.extend(
            [
                float(np.linalg.norm(left_camera.project_world(point) - left_pixel)),
                float(np.linalg.norm(right_camera.project_world(point) - right_pixel)),
            ]
        )
    corners = np.vstack(points)
    center = np.mean(corners, axis=0)
    _, _, vh = np.linalg.svd(corners - center)
    normal = vh[-1]
    normal /= np.linalg.norm(normal)
    camera_midpoint = (
        left_camera.camera_center_world_m
        + right_camera.camera_center_world_m
    ) / 2.0
    if float(np.dot(normal, camera_midpoint - center)) < 0.0:
        normal *= -1.0
    residuals = np.abs((corners - center) @ normal)
    top = float(np.linalg.norm(corners[1] - corners[0]))
    bottom = float(np.linalg.norm(corners[2] - corners[3]))
    left_edge = float(np.linalg.norm(corners[3] - corners[0]))
    right_edge = float(np.linalg.norm(corners[2] - corners[1]))
    # Corner indices come from axis-aligned bbox seeding (see
    # refine_cable_corners), so "corners 0-1" isn't reliably the long edge
    # once the cable sits at a diagonal angle - the same issue we already
    # hit once in the blob detector. Measure both edge-pairs and assign
    # width/height by which is actually longer, not by position, so they
    # can't silently swap identity as the cable's angle changes.
    edge_pair_a = (top + bottom) / 2.0
    edge_pair_b = (left_edge + right_edge) / 2.0
    width = max(edge_pair_a, edge_pair_b)
    height = min(edge_pair_a, edge_pair_b)
    opposite_ratio = max(
        _edge_ratio(top, bottom),
        _edge_ratio(left_edge, right_edge),
    )
    reprojection = np.asarray(reprojection_errors, dtype=np.float64)
    rms = float(np.sqrt(np.mean(reprojection * reprojection)))
    maximum = float(np.max(reprojection))
    max_gap = float(np.max(ray_gaps))
    plane_residual = float(np.max(residuals))

    # Broad structural gates belong here so obviously wrong corner ordering
    # never reaches application-specific dimension checks.
    if not np.all(np.isfinite(corners)):
        raise RuntimeError("Triangulation produced non-finite corners.")
    if width <= 0.0 or height <= 0.0:
        raise RuntimeError("Triangulated rectangle has zero-sized edges.")
    if max_gap > 0.010 or maximum > 10.0:
        raise RuntimeError("Corner correspondence has excessive ray/reprojection error.")
    if opposite_ratio > 3.0:
        raise RuntimeError("Corner correspondence does not form a rectangle.")

    return StereoTriangulation(
        corners_world_m=corners,
        center_world_m=center,
        normal_world=normal,
        width_m=width,
        height_m=height,
        reprojection_rms_px=rms,
        max_reprojection_px=maximum,
        max_ray_gap_m=max_gap,
        plane_residual_m=plane_residual,
        opposite_edge_ratio=opposite_ratio,
    )


# ---------------------------------------------------------------------------
# Four-corner extraction and stereo processing
# ---------------------------------------------------------------------------


def refine_cable_corners(
    rgb: np.ndarray,
    detection: CableDetection,
    cfg: PerceptionConfig,
) -> CableCorners:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    x, y, width, height = detection.bbox_xywh
    seeds = np.array(
        [
            [x, y],
            [x + width - 1, y],
            [x + width - 1, y + height - 1],
            [x, y + height - 1],
        ],
        dtype=np.float32,
    )
    refined = seeds.reshape(-1, 1, 2).copy()
    window = int(cfg.stereo_corner_refine_window_px)
    cv2.cornerSubPix(
        gray,
        refined,
        (window, window),
        (-1, -1),
        (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            40,
            0.01,
        ),
    )
    corners = refined.reshape(4, 2).astype(np.float64)
    shifts = np.linalg.norm(corners - seeds.astype(np.float64), axis=1)
    if float(np.max(shifts)) > cfg.stereo_max_corner_refine_shift_px:
        raise RuntimeError("corner refinement moved outside the cable boundary")
    image_height, image_width = gray.shape
    if (
        np.any(corners[:, 0] < 0.0)
        or np.any(corners[:, 0] >= image_width)
        or np.any(corners[:, 1] < 0.0)
        or np.any(corners[:, 1] >= image_height)
    ):
        raise RuntimeError("refined corner left the image")
    contour = np.round(corners).astype(np.int32).reshape(-1, 1, 2)
    if not cv2.isContourConvex(contour):
        raise RuntimeError("refined cable corners are not convex")
    area = abs(float(cv2.contourArea(contour)))
    if area < cfg.min_area_px * 0.5:
        raise RuntimeError("refined cable quadrilateral is too small")
    return CableCorners(detection=detection, corners_uv=corners)


def _candidate_continuity_ok(
    candidate: CableDetection,
    previous: CableDetection | None,
    cfg: PerceptionConfig,
) -> bool:
    if previous is None:
        return True
    center_distance = float(
        np.linalg.norm(
            np.asarray(candidate.center_uv) - np.asarray(previous.center_uv)
        )
    )
    scale_ratio = max(
        candidate.scale_px / previous.scale_px,
        previous.scale_px / candidate.scale_px,
    )
    return (
        center_distance <= cfg.tracking_max_center_jump_px
        and scale_ratio <= cfg.tracking_max_scale_ratio
    )


def _validate_stereo_result(
    result: StereoTriangulation,
    virtual_camera: CameraModel,
    cfg: PerceptionConfig,
) -> None:
    center_virtual = virtual_camera.camera_point_from_world(
        result.center_world_m
    )
    range_m = -float(center_virtual[2])
    checks = (
        (
            cfg.min_estimated_range_m <= range_m <= cfg.max_estimated_range_m,
            "stereo range is outside the configured working distance: "
            f"{range_m * 1000.0:.2f}mm (expected "
            f"{cfg.min_estimated_range_m * 1000.0:.1f}-"
            f"{cfg.max_estimated_range_m * 1000.0:.1f}mm)",
        ),
        (
            result.reprojection_rms_px
            <= cfg.stereo_max_reprojection_rms_px,
            "stereo reprojection RMS is too high: "
            f"{result.reprojection_rms_px:.3f}px (max "
            f"{cfg.stereo_max_reprojection_rms_px:.3f}px)",
        ),
        (
            result.max_reprojection_px <= cfg.stereo_max_reprojection_px,
            "stereo corner reprojection error is too high: "
            f"{result.max_reprojection_px:.3f}px (max "
            f"{cfg.stereo_max_reprojection_px:.3f}px)",
        ),
        (
            result.max_ray_gap_m <= cfg.stereo_max_ray_gap_m,
            "stereo rays do not intersect closely enough: "
            f"{result.max_ray_gap_m * 1000.0:.3f}mm gap (max "
            f"{cfg.stereo_max_ray_gap_m * 1000.0:.3f}mm)",
        ),
        (
            result.plane_residual_m <= cfg.stereo_max_plane_residual_m,
            "triangulated corners are not coplanar: "
            f"{result.plane_residual_m * 1000.0:.3f}mm residual (max "
            f"{cfg.stereo_max_plane_residual_m * 1000.0:.3f}mm)",
        ),
        (
            cfg.stereo_min_width_m
            <= result.width_m
            <= cfg.stereo_max_width_m,
            "triangulated cable width is implausible: "
            f"{result.width_m * 1000.0:.2f}mm (expected "
            f"{cfg.stereo_min_width_m * 1000.0:.1f}-"
            f"{cfg.stereo_max_width_m * 1000.0:.1f}mm)",
        ),
        (
            cfg.stereo_min_height_m
            <= result.height_m
            <= cfg.stereo_max_height_m,
            "triangulated cable height is implausible: "
            f"{result.height_m * 1000.0:.2f}mm (expected "
            f"{cfg.stereo_min_height_m * 1000.0:.1f}-"
            f"{cfg.stereo_max_height_m * 1000.0:.1f}mm)",
        ),
        (
            result.opposite_edge_ratio
            <= cfg.stereo_max_opposite_edge_ratio,
            "triangulated opposite edges disagree: ratio "
            f"{result.opposite_edge_ratio:.3f} (max "
            f"{cfg.stereo_max_opposite_edge_ratio:.3f})",
        ),
    )
    for accepted, reason in checks:
        if not accepted:
            raise RuntimeError(reason)


def process_stereo_cable(
    frame: StereoFrame,
    cfg: PerceptionConfig,
    desired_cable_virtual_camera_usd: np.ndarray,
    previous_left: CableDetection | None,
    previous_right: CableDetection | None,
) -> StereoCableObservation:
    """Require both eyes, triangulate the matched cable center, and compute one correction."""
    left_candidates = detect_cable_candidates(frame.left.rgb, cfg)
    if not left_candidates:
        raise RuntimeError("left eye did not detect a valid RGB cable")
    right_candidates = detect_cable_candidates(frame.right.rgb, cfg)
    if not right_candidates:
        raise RuntimeError("right eye did not detect a valid RGB cable")

    left_corners: list[CableCorners] = []
    right_corners: list[CableCorners] = []
    left_corner_errors: list[str] = []
    right_corner_errors: list[str] = []
    for candidate in left_candidates:
        if not _candidate_continuity_ok(candidate, previous_left, cfg):
            continue
        try:
            left_corners.append(refine_cable_corners(frame.left.rgb, candidate, cfg))
        except (RuntimeError, cv2.error) as exc:
            left_corner_errors.append(str(exc))
            continue
    for candidate in right_candidates:
        if not _candidate_continuity_ok(candidate, previous_right, cfg):
            continue
        try:
            right_corners.append(refine_cable_corners(frame.right.rgb, candidate, cfg))
        except (RuntimeError, cv2.error) as exc:
            right_corner_errors.append(str(exc))
            continue
    if not left_corners:
        detail = left_corner_errors[-1] if left_corner_errors else "no candidate"
        raise RuntimeError(f"left eye corner refinement failed: {detail}")
    if not right_corners:
        detail = right_corner_errors[-1] if right_corner_errors else "no candidate"
        raise RuntimeError(f"right eye corner refinement failed: {detail}")

    pair_results: list[
        tuple[float, CableCorners, CableCorners, StereoTriangulation]
    ] = []
    pair_rejections: list[str] = []
    for left in left_corners:
        for right in right_corners:
            vertical_error = abs(
                left.detection.center_uv[1] - right.detection.center_uv[1]
            )
            if vertical_error > cfg.stereo_max_epipolar_error_px:
                continue
            scale_ratio = max(
                left.detection.scale_px / right.detection.scale_px,
                right.detection.scale_px / left.detection.scale_px,
            )
            if scale_ratio > cfg.stereo_max_scale_ratio:
                continue
            disparity = abs(
                left.detection.center_uv[0] - right.detection.center_uv[0]
            )
            if disparity < cfg.stereo_min_abs_disparity_px:
                continue
            try:
                result = triangulate_detection_centers(
                    left.detection,
                    right.detection,
                    frame.left.camera,
                    frame.right.camera,
                    frame.virtual_camera,
                )
                _validate_stereo_result(result, frame.virtual_camera, cfg)
            except RuntimeError as exc:
                pair_rejections.append(str(exc))
                continue
            # Only thickness has a real "correct" value to compare against;
            # the visible length varies with framing, so it isn't part of
            # this penalty (unlike the port version this was adapted from).
            dimension_error = (
                abs(result.height_m - cfg.cable_thickness_m)
                / cfg.cable_thickness_m
            )
            score = (
                left.detection.shape_score
                + right.detection.shape_score
                - 0.05 * vertical_error
                - 0.75 * dimension_error
                - result.reprojection_rms_px
            )
            pair_results.append((score, left, right, result))

    if not pair_results:
        detail = pair_rejections[-1] if pair_rejections else (
            "epipolar, scale, or disparity gate rejected every pair"
        )
        raise RuntimeError(
            "no left/right cable pair passed stereo correspondence and "
            f"geometry checks: {detail}"
        )
    _, left, right, result = max(pair_results, key=lambda item: item[0])

    virtual_camera = frame.virtual_camera
    center_virtual = virtual_camera.camera_point_from_world(
        result.center_world_m
    )
    desired = np.asarray(
        desired_cable_virtual_camera_usd,
        dtype=np.float64,
    ).reshape(3)
    desired_center, desired_size = project_cable_feature(
        desired,
        virtual_camera,
        cfg,
    )
    projected_center = virtual_camera.project_world(result.center_world_m)
    center_error = projected_center - np.asarray(desired_center)
    estimated_range_m = -float(center_virtual[2])
    range_error_m = estimated_range_m - (-float(desired[2]))
    correction_world = camera_point_error_to_world(
        center_virtual,
        desired,
        virtual_camera.world_from_camera,
    )
    desired_world = transform_point_to_world(
        desired,
        virtual_camera.world_from_camera,
    )
    desired_left = frame.left.camera.project_world(desired_world)
    desired_right = frame.right.camera.project_world(desired_world)
    center_disparity = abs(
        left.detection.center_uv[0] - right.detection.center_uv[0]
    )

    return StereoCableObservation(
        left=left,
        right=right,
        corners_world_m=result.corners_world_m,
        center_world_xyz_m=result.center_world_m,
        center_virtual_camera_usd_m=center_virtual,
        normal_world=result.normal_world,
        projected_virtual_center_uv=(
            float(projected_center[0]),
            float(projected_center[1]),
        ),
        desired_center_uv=desired_center,
        desired_size_wh_px=desired_size,
        desired_left_center_uv=(float(desired_left[0]), float(desired_left[1])),
        desired_right_center_uv=(float(desired_right[0]), float(desired_right[1])),
        center_error_px=center_error,
        estimated_range_m=estimated_range_m,
        range_error_m=float(range_error_m),
        correction_world_m=correction_world,
        width_m=result.width_m,
        height_m=result.height_m,
        mean_disparity_px=float(center_disparity),
        reprojection_rms_px=result.reprojection_rms_px,
        max_reprojection_px=result.max_reprojection_px,
        max_ray_gap_m=result.max_ray_gap_m,
        plane_residual_m=result.plane_residual_m,
    )