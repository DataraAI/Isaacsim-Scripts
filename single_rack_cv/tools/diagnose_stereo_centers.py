#!/usr/bin/env python3
"""Offline comparison of stereo port-center estimators on frozen RGB pairs.

This script deliberately leaves the runtime controller untouched. It reuses the
winning Prompt-B YOLOE configuration, selects the same stereo pair with the
existing production logic, and then compares four ways of locating that same
physical opening:

1. bbox_midpoint_current: the current production center (integer box midpoint)
2. mask_centroid: centroid of the refined dark-cavity component pixels
3. refined_corner_centroid: mean of four subpixel-refined image corners
4. full_corner_mean_3d: triangulate all four corresponding corners and average

The output answers one question: is the current 0.739 mm 3D jitter caused by the
center estimator, or is image sampling/stereo disparity itself the bottleneck?
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, replace
import json
import math
from pathlib import Path
import sys
import time
from typing import Iterable

import cv2
import numpy as np
from PIL import Image, ImageDraw

from config import CONFIG
from perception import (
    CameraFrame,
    CameraModel,
    PortDetection,
    StereoFrame,
    YOLOEPortDetector,
    process_stereo_port,
    triangulate_pixel_pair,
    triangulate_port_corners,
)


METHOD_BBOX = "bbox_midpoint_current"
METHOD_MASK = "mask_centroid"
METHOD_CORNER_CENTER = "refined_corner_centroid"
METHOD_FULL_CORNERS = "full_corner_mean_3d"
METHOD_ORDER = (
    METHOD_BBOX,
    METHOD_MASK,
    METHOD_CORNER_CENTER,
    METHOD_FULL_CORNERS,
)
MAX_ACCEPTABLE_JITTER_MM = 0.50
MAX_ACCEPTABLE_RAY_GAP_P95_MM = 0.50
MIN_SUCCESS_RATE = 0.95
MIN_DECISION_FRAME_COUNT = 60

DETAIL_FIELDS = (
    "method",
    "frame_index",
    "success",
    "left_u",
    "left_v",
    "right_u",
    "right_v",
    "disparity_px",
    "center_world_x",
    "center_world_y",
    "center_world_z",
    "ray_gap_mm",
    "reprojection_rms_px",
    "left_corner_shift_mean_px",
    "left_corner_shift_max_px",
    "right_corner_shift_mean_px",
    "right_corner_shift_max_px",
    "left_corners_json",
    "right_corners_json",
    "error",
)


@dataclass(frozen=True)
class CachedDetector:
    """Expose one already-computed candidate batch to production pair selection."""

    left: list[PortDetection]
    right: list[PortDetection]
    diagnostics: dict[str, str]

    def detect_stereo(
        self,
        left_rgb: np.ndarray,
        right_rgb: np.ndarray,
    ) -> tuple[list[PortDetection], list[PortDetection]]:
        del left_rgb, right_rgb
        return self.left, self.right

    def diagnostic(self, eye_name: str) -> str:
        return self.diagnostics.get(eye_name, "no cached diagnostic")


def _camera_from_dict(data: dict[str, object]) -> CameraModel:
    return CameraModel(
        image_height_px=int(data["image_height_px"]),
        image_width_px=int(data["image_width_px"]),
        focal_length_mm=float(data["focal_length_mm"]),
        horizontal_aperture_mm=float(data["horizontal_aperture_mm"]),
        vertical_aperture_mm=float(data["vertical_aperture_mm"]),
        world_from_camera=np.asarray(
            data["world_from_camera"],
            dtype=np.float64,
        ),
    )


def load_stereo_frame(
    benchmark_root: Path,
    entry: dict[str, object],
) -> StereoFrame:
    with Image.open(benchmark_root / str(entry["left_image"])) as image:
        left_rgb = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    with Image.open(benchmark_root / str(entry["right_image"])) as image:
        right_rgb = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    return StereoFrame(
        left=CameraFrame(
            rgb=left_rgb,
            camera=_camera_from_dict(entry["left_camera"]),
        ),
        right=CameraFrame(
            rgb=right_rgb,
            camera=_camera_from_dict(entry["right_camera"]),
        ),
        virtual_camera=_camera_from_dict(entry["virtual_camera"]),
    )


def mask_centroid_uv(detection: PortDetection) -> np.ndarray:
    """Return the subpixel centroid of nonzero refined-cavity mask pixels."""
    mask = np.asarray(detection.mask)
    if mask.ndim != 2:
        raise ValueError(f"Port mask must be 2D, got {mask.shape}.")
    rows, columns = np.nonzero(mask > 0)
    if columns.size == 0:
        raise RuntimeError("Refined port mask contains no foreground pixels.")
    return np.asarray(
        [float(np.mean(columns)), float(np.mean(rows))],
        dtype=np.float64,
    )


def box_corners_uv(detection: PortDetection) -> np.ndarray:
    """Return TL, TR, BR, BL corners of the current integer cavity box."""
    x, y, width, height = map(int, detection.bbox_xywh)
    if width < 2 or height < 2:
        raise RuntimeError(
            f"Port box is too small for corner refinement: {detection.bbox_xywh}."
        )
    return np.asarray(
        [
            [x, y],
            [x + width - 1, y],
            [x + width - 1, y + height - 1],
            [x, y + height - 1],
        ],
        dtype=np.float64,
    )


def _validate_corner_order(corners: np.ndarray) -> None:
    corners = np.asarray(corners, dtype=np.float64)
    if corners.shape != (4, 2) or not np.all(np.isfinite(corners)):
        raise RuntimeError("Subpixel corner refinement returned invalid points.")
    top_left, top_right, bottom_right, bottom_left = corners
    if not (
        top_left[0] < top_right[0]
        and bottom_left[0] < bottom_right[0]
        and top_left[1] < bottom_left[1]
        and top_right[1] < bottom_right[1]
    ):
        raise RuntimeError("Subpixel corners crossed or changed ordering.")
    area = abs(float(cv2.contourArea(corners.astype(np.float32))))
    if area < 16.0:
        raise RuntimeError("Subpixel corner polygon has implausibly small area.")


def refine_box_corners_subpixel(
    rgb: np.ndarray,
    detection: PortDetection,
    *,
    window_radius_px: int = 5,
    max_corner_shift_px: float = 6.0,
) -> np.ndarray:
    """Refine the four current box corners against local grayscale gradients.

    The routine intentionally starts from the already-selected dark-cavity box.
    It does not search the full image and therefore cannot switch to another
    rack feature. A shift gate rejects cornerSubPix convergence onto an internal
    contact pin or unrelated bezel edge.
    """
    image = np.asarray(rgb)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 RGB image, got {image.shape}.")
    radius = int(window_radius_px)
    if radius < 1:
        raise ValueError("window_radius_px must be positive.")
    max_shift = float(max_corner_shift_px)
    if not math.isfinite(max_shift) or max_shift <= 0.0:
        raise ValueError("max_corner_shift_px must be finite and positive.")

    initial = box_corners_uv(detection).astype(np.float32).reshape(-1, 1, 2)
    height, width = image.shape[:2]
    margin = radius + 1
    if (
        np.any(initial[:, 0, 0] < margin)
        or np.any(initial[:, 0, 0] >= width - margin)
        or np.any(initial[:, 0, 1] < margin)
        or np.any(initial[:, 0, 1] >= height - margin)
    ):
        raise RuntimeError("Port box is too close to the image boundary.")

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0.0).astype(np.float32)
    refined = initial.copy()
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER,
        80,
        1.0e-4,
    )
    cv2.cornerSubPix(
        gray,
        refined,
        (radius, radius),
        (-1, -1),
        criteria,
    )
    refined_2d = refined.reshape(4, 2).astype(np.float64)
    initial_2d = initial.reshape(4, 2).astype(np.float64)
    shifts = np.linalg.norm(refined_2d - initial_2d, axis=1)
    if not np.all(np.isfinite(shifts)) or float(np.max(shifts)) > max_shift:
        raise RuntimeError(
            "Subpixel corner refinement exceeded the shift gate: "
            f"shifts={np.round(shifts, 3).tolist()} px, "
            f"limit={max_shift:.3f} px."
        )
    _validate_corner_order(refined_2d)
    return refined_2d


def _reprojection_rms(
    point_world_m: np.ndarray,
    left_uv: np.ndarray,
    right_uv: np.ndarray,
    left_camera: CameraModel,
    right_camera: CameraModel,
) -> float:
    errors = np.asarray(
        [
            np.linalg.norm(left_camera.project_world(point_world_m) - left_uv),
            np.linalg.norm(right_camera.project_world(point_world_m) - right_uv),
        ],
        dtype=np.float64,
    )
    return float(np.sqrt(np.mean(errors * errors)))


def _blank_record(method: str, frame_index: int) -> dict[str, object]:
    record: dict[str, object] = {field: math.nan for field in DETAIL_FIELDS}
    record.update(
        {
            "method": method,
            "frame_index": int(frame_index),
            "success": False,
            "left_corners_json": "",
            "right_corners_json": "",
            "error": "",
        }
    )
    return record


def _record_center_estimate(
    method: str,
    frame_index: int,
    left_uv: np.ndarray,
    right_uv: np.ndarray,
    frame: StereoFrame,
    *,
    left_corners: np.ndarray | None = None,
    right_corners: np.ndarray | None = None,
    left_corner_shifts: np.ndarray | None = None,
    right_corner_shifts: np.ndarray | None = None,
) -> dict[str, object]:
    record = _blank_record(method, frame_index)
    try:
        left_point = np.asarray(left_uv, dtype=np.float64).reshape(2)
        right_point = np.asarray(right_uv, dtype=np.float64).reshape(2)
        center_world, ray_gap = triangulate_pixel_pair(
            left_point,
            right_point,
            frame.left.camera,
            frame.right.camera,
        )
        reprojection = _reprojection_rms(
            center_world,
            left_point,
            right_point,
            frame.left.camera,
            frame.right.camera,
        )
        record.update(
            {
                "success": True,
                "left_u": float(left_point[0]),
                "left_v": float(left_point[1]),
                "right_u": float(right_point[0]),
                "right_v": float(right_point[1]),
                "disparity_px": float(left_point[0] - right_point[0]),
                "center_world_x": float(center_world[0]),
                "center_world_y": float(center_world[1]),
                "center_world_z": float(center_world[2]),
                "ray_gap_mm": float(ray_gap * 1000.0),
                "reprojection_rms_px": reprojection,
            }
        )
        if left_corners is not None:
            record["left_corners_json"] = json.dumps(
                np.asarray(left_corners, dtype=float).tolist()
            )
        if right_corners is not None:
            record["right_corners_json"] = json.dumps(
                np.asarray(right_corners, dtype=float).tolist()
            )
        if left_corner_shifts is not None:
            shifts = np.asarray(left_corner_shifts, dtype=np.float64)
            record["left_corner_shift_mean_px"] = float(np.mean(shifts))
            record["left_corner_shift_max_px"] = float(np.max(shifts))
        if right_corner_shifts is not None:
            shifts = np.asarray(right_corner_shifts, dtype=np.float64)
            record["right_corner_shift_mean_px"] = float(np.mean(shifts))
            record["right_corner_shift_max_px"] = float(np.max(shifts))
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
    return record


def _record_full_corner_estimate(
    frame_index: int,
    left_corners: np.ndarray,
    right_corners: np.ndarray,
    frame: StereoFrame,
    left_corner_shifts: np.ndarray,
    right_corner_shifts: np.ndarray,
) -> dict[str, object]:
    record = _blank_record(METHOD_FULL_CORNERS, frame_index)
    try:
        result = triangulate_port_corners(
            left_corners,
            right_corners,
            frame.left.camera,
            frame.right.camera,
        )
        left_center = np.mean(left_corners, axis=0)
        right_center = np.mean(right_corners, axis=0)
        record.update(
            {
                "success": True,
                "left_u": float(left_center[0]),
                "left_v": float(left_center[1]),
                "right_u": float(right_center[0]),
                "right_v": float(right_center[1]),
                "disparity_px": float(left_center[0] - right_center[0]),
                "center_world_x": float(result.center_world_m[0]),
                "center_world_y": float(result.center_world_m[1]),
                "center_world_z": float(result.center_world_m[2]),
                "ray_gap_mm": float(result.max_ray_gap_m * 1000.0),
                "reprojection_rms_px": float(result.reprojection_rms_px),
                "left_corner_shift_mean_px": float(
                    np.mean(left_corner_shifts)
                ),
                "left_corner_shift_max_px": float(
                    np.max(left_corner_shifts)
                ),
                "right_corner_shift_mean_px": float(
                    np.mean(right_corner_shifts)
                ),
                "right_corner_shift_max_px": float(
                    np.max(right_corner_shifts)
                ),
                "left_corners_json": json.dumps(
                    np.asarray(left_corners, dtype=float).tolist()
                ),
                "right_corners_json": json.dumps(
                    np.asarray(right_corners, dtype=float).tolist()
                ),
            }
        )
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
    return record


def estimate_frame_methods(
    frame_index: int,
    frame: StereoFrame,
    left_detection: PortDetection,
    right_detection: PortDetection,
    *,
    corner_window_radius_px: int,
    max_corner_shift_px: float,
) -> list[dict[str, object]]:
    """Compute all four estimators from one fixed selected stereo pair."""
    records: list[dict[str, object]] = []

    records.append(
        _record_center_estimate(
            METHOD_BBOX,
            frame_index,
            np.asarray(left_detection.center_uv, dtype=np.float64),
            np.asarray(right_detection.center_uv, dtype=np.float64),
            frame,
        )
    )

    try:
        left_mask_center = mask_centroid_uv(left_detection)
        right_mask_center = mask_centroid_uv(right_detection)
        records.append(
            _record_center_estimate(
                METHOD_MASK,
                frame_index,
                left_mask_center,
                right_mask_center,
                frame,
            )
        )
    except Exception as exc:
        failed = _blank_record(METHOD_MASK, frame_index)
        failed["error"] = f"{type(exc).__name__}: {exc}"
        records.append(failed)

    try:
        left_initial = box_corners_uv(left_detection)
        right_initial = box_corners_uv(right_detection)
        left_corners = refine_box_corners_subpixel(
            frame.left.rgb,
            left_detection,
            window_radius_px=corner_window_radius_px,
            max_corner_shift_px=max_corner_shift_px,
        )
        right_corners = refine_box_corners_subpixel(
            frame.right.rgb,
            right_detection,
            window_radius_px=corner_window_radius_px,
            max_corner_shift_px=max_corner_shift_px,
        )
        left_shifts = np.linalg.norm(left_corners - left_initial, axis=1)
        right_shifts = np.linalg.norm(right_corners - right_initial, axis=1)
        left_corner_center = np.mean(left_corners, axis=0)
        right_corner_center = np.mean(right_corners, axis=0)
        records.append(
            _record_center_estimate(
                METHOD_CORNER_CENTER,
                frame_index,
                left_corner_center,
                right_corner_center,
                frame,
                left_corners=left_corners,
                right_corners=right_corners,
                left_corner_shifts=left_shifts,
                right_corner_shifts=right_shifts,
            )
        )
        records.append(
            _record_full_corner_estimate(
                frame_index,
                left_corners,
                right_corners,
                frame,
                left_shifts,
                right_shifts,
            )
        )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        for method in (METHOD_CORNER_CENTER, METHOD_FULL_CORNERS):
            failed = _blank_record(method, frame_index)
            failed["error"] = error
            records.append(failed)

    return records


def _finite(values: Iterable[float]) -> np.ndarray:
    array = np.asarray(list(values), dtype=np.float64).reshape(-1)
    return array[np.isfinite(array)]


def _radial_rms(points: np.ndarray) -> float:
    if points.size == 0:
        return math.nan
    center = np.median(points, axis=0)
    return float(
        np.sqrt(np.mean(np.sum((points - center) ** 2, axis=1)))
    )


def _scalar_rms(values: np.ndarray) -> float:
    if values.size == 0:
        return math.nan
    center = float(np.median(values))
    return float(np.sqrt(np.mean((values - center) ** 2)))


def _axis_rms(points: np.ndarray) -> np.ndarray:
    if points.size == 0:
        return np.full(3, math.nan, dtype=np.float64)
    center = np.median(points, axis=0)
    return np.sqrt(np.mean((points - center) ** 2, axis=0))


def _safe_percentile(values: Iterable[float], percentile: float) -> float:
    finite = _finite(values)
    return float(np.percentile(finite, percentile)) if finite.size else math.nan


def _safe_mean(values: Iterable[float]) -> float:
    finite = _finite(values)
    return float(np.mean(finite)) if finite.size else math.nan


def summarize_method_records(
    method: str,
    records: list[dict[str, object]],
    *,
    expected_frames: int,
) -> dict[str, object]:
    successful = [record for record in records if bool(record.get("success"))]
    left = np.asarray(
        [[float(r["left_u"]), float(r["left_v"])] for r in successful],
        dtype=np.float64,
    ).reshape(-1, 2)
    right = np.asarray(
        [[float(r["right_u"]), float(r["right_v"])] for r in successful],
        dtype=np.float64,
    ).reshape(-1, 2)
    world = np.asarray(
        [
            [
                float(r["center_world_x"]),
                float(r["center_world_y"]),
                float(r["center_world_z"]),
            ]
            for r in successful
        ],
        dtype=np.float64,
    ).reshape(-1, 3)
    disparity = np.asarray(
        [float(r["disparity_px"]) for r in successful],
        dtype=np.float64,
    )
    axis = 1000.0 * _axis_rms(world)
    world_jitter = 1000.0 * _radial_rms(world)
    success_count = len(successful)
    success_rate = success_count / float(expected_frames)
    ray_gap_p95 = _safe_percentile(
        [float(r["ray_gap_mm"]) for r in successful],
        95.0,
    )
    decision_frame_count_met = expected_frames >= MIN_DECISION_FRAME_COUNT
    qualified = bool(
        decision_frame_count_met
        and success_rate >= MIN_SUCCESS_RATE
        and math.isfinite(world_jitter)
        and world_jitter <= MAX_ACCEPTABLE_JITTER_MM
        and math.isfinite(ray_gap_p95)
        and ray_gap_p95 <= MAX_ACCEPTABLE_RAY_GAP_P95_MM
    )
    return {
        "method": method,
        "expected_frames": int(expected_frames),
        "success_count": success_count,
        "failure_count": int(expected_frames - success_count),
        "success_rate": success_rate,
        "all_frames_succeeded": success_count == expected_frames,
        "decision_frame_count_met": decision_frame_count_met,
        "left_center_jitter_px": _radial_rms(left),
        "right_center_jitter_px": _radial_rms(right),
        "disparity_jitter_px": _scalar_rms(disparity),
        "center_3d_jitter_mm": world_jitter,
        "axis_x_jitter_mm": float(axis[0]),
        "axis_y_jitter_mm": float(axis[1]),
        "axis_z_jitter_mm": float(axis[2]),
        "ray_gap_p95_mm": ray_gap_p95,
        "reprojection_rms_mean_px": _safe_mean(
            [float(r["reprojection_rms_px"]) for r in successful]
        ),
        "left_corner_shift_mean_px": _safe_mean(
            [float(r["left_corner_shift_mean_px"]) for r in successful]
        ),
        "right_corner_shift_mean_px": _safe_mean(
            [float(r["right_corner_shift_mean_px"]) for r in successful]
        ),
        "qualified_at_0_5mm": qualified,
    }


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def _write_csv(path: Path, rows: list[dict[str, object]], fields) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)


def _world_deviation_mm(
    record: dict[str, object],
    median_world: np.ndarray,
) -> float:
    point = np.asarray(
        [
            float(record["center_world_x"]),
            float(record["center_world_y"]),
            float(record["center_world_z"]),
        ],
        dtype=np.float64,
    )
    return float(np.linalg.norm(point - median_world) * 1000.0)


def _draw_cross(
    draw: ImageDraw.ImageDraw,
    center: tuple[float, float],
    offset_x: int,
    color: tuple[int, int, int],
    radius: int = 6,
    width: int = 2,
) -> None:
    x = float(center[0]) + offset_x
    y = float(center[1])
    draw.line((x - radius, y, x + radius, y), fill=color, width=width)
    draw.line((x, y - radius, x, y + radius), fill=color, width=width)


def _draw_corner_polygon(
    draw: ImageDraw.ImageDraw,
    corners_json: str,
    offset_x: int,
    color: tuple[int, int, int],
) -> None:
    if not corners_json:
        return
    corners = np.asarray(json.loads(corners_json), dtype=np.float64)
    if corners.shape != (4, 2):
        return
    points = [
        (float(point[0]) + offset_x, float(point[1]))
        for point in corners
    ]
    draw.line(points + [points[0]], fill=color, width=2)


def save_worst_frame_annotations(
    benchmark_root: Path,
    manifest_by_index: dict[int, dict[str, object]],
    records_by_method: dict[str, list[dict[str, object]]],
    output_dir: Path,
    *,
    count: int,
) -> None:
    if count <= 0:
        return
    method_colors = {
        METHOD_BBOX: (255, 220, 0),
        METHOD_MASK: (0, 255, 255),
        METHOD_CORNER_CENTER: (0, 255, 0),
        METHOD_FULL_CORNERS: (255, 0, 255),
    }
    for method, records in records_by_method.items():
        successful = [r for r in records if bool(r.get("success"))]
        if not successful:
            continue
        points = np.asarray(
            [
                [
                    float(r["center_world_x"]),
                    float(r["center_world_y"]),
                    float(r["center_world_z"]),
                ]
                for r in successful
            ],
            dtype=np.float64,
        )
        median_world = np.median(points, axis=0)
        ranked = sorted(
            successful,
            key=lambda r: _world_deviation_mm(r, median_world),
            reverse=True,
        )[:count]
        method_dir = output_dir / "worst_frames" / method
        method_dir.mkdir(parents=True, exist_ok=True)
        color = method_colors[method]

        for record in ranked:
            frame_index = int(record["frame_index"])
            entry = manifest_by_index[frame_index]
            with Image.open(
                benchmark_root / str(entry["left_image"])
            ) as left_image:
                left = left_image.convert("RGB")
            with Image.open(
                benchmark_root / str(entry["right_image"])
            ) as right_image:
                right = right_image.convert("RGB")
            combined = Image.new(
                "RGB",
                (left.width + right.width, max(left.height, right.height)),
            )
            combined.paste(left, (0, 0))
            combined.paste(right, (left.width, 0))
            draw = ImageDraw.Draw(combined)
            _draw_cross(
                draw,
                (float(record["left_u"]), float(record["left_v"])),
                0,
                color,
            )
            _draw_cross(
                draw,
                (float(record["right_u"]), float(record["right_v"])),
                left.width,
                color,
            )
            _draw_corner_polygon(
                draw,
                str(record.get("left_corners_json", "")),
                0,
                color,
            )
            _draw_corner_polygon(
                draw,
                str(record.get("right_corners_json", "")),
                left.width,
                color,
            )
            deviation = _world_deviation_mm(record, median_world)
            label = (
                f"{method} frame={frame_index:04d}\n"
                f"3D deviation={deviation:.3f} mm  "
                f"disparity={float(record['disparity_px']):.4f} px"
            )
            bounds = draw.multiline_textbbox((8, 8), label)
            draw.rectangle(
                (bounds[0] - 4, bounds[1] - 3, bounds[2] + 4, bounds[3] + 3),
                fill=(0, 0, 0),
            )
            draw.multiline_text((8, 8), label, fill=color, spacing=3)
            combined.save(method_dir / f"frame_{frame_index:04d}.png")


def _format_number(value: object, digits: int = 6) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return f"{number:.{digits}f}" if math.isfinite(number) else "n/a"


def build_report(
    summaries: list[dict[str, object]],
    detector_failures: list[dict[str, object]],
    *,
    expected_frames: int,
    elapsed_s: float,
) -> str:
    lines = [
        "STEREO CENTER-ESTIMATOR DIAGNOSTIC",
        "=" * 72,
        "",
        "Measured code path:",
        "  The production controller currently triangulates PortDetection.center_uv.",
        "  In the uploaded code that value is the integer refined-box midpoint,",
        "  not a mask centroid and not a subpixel-refined corner center.",
        "",
        f"Frozen frames evaluated: {expected_frames}",
        f"Detector/pair-selection failures: {len(detector_failures)}",
        f"Elapsed time: {elapsed_s:.1f} s",
        "",
        "METHOD RESULTS",
        "-" * 72,
    ]
    for summary in summaries:
        lines.extend(
            [
                f"{summary['method']}",
                f"  success:          {summary['success_count']}/{expected_frames}",
                "  left pixel RMS:   "
                f"{_format_number(summary['left_center_jitter_px'])} px",
                "  right pixel RMS:  "
                f"{_format_number(summary['right_center_jitter_px'])} px",
                "  disparity RMS:    "
                f"{_format_number(summary['disparity_jitter_px'])} px",
                "  3D radial RMS:    "
                f"{_format_number(summary['center_3d_jitter_mm'])} mm",
                "  per-axis RMS:     "
                f"X={_format_number(summary['axis_x_jitter_mm'])} mm, "
                f"Y={_format_number(summary['axis_y_jitter_mm'])} mm, "
                f"Z={_format_number(summary['axis_z_jitter_mm'])} mm",
                "  ray-gap p95:      "
                f"{_format_number(summary['ray_gap_p95_mm'])} mm",
                "  <=0.5 mm gate:    "
                f"{'PASS' if summary['qualified_at_0_5mm'] else 'FAIL'}",
                "",
            ]
        )

    qualified = [s for s in summaries if s["qualified_at_0_5mm"]]
    lines.extend(["DECISION", "-" * 72])
    if expected_frames < MIN_DECISION_FRAME_COUNT:
        lines.append(
            f"Smoke test only: {expected_frames} frames were evaluated. Run all "
            f"{MIN_DECISION_FRAME_COUNT} frozen frames before changing production code."
        )
    elif qualified:
        winner = min(qualified, key=lambda item: item["center_3d_jitter_mm"])
        lines.append(
            f"Use {winner['method']} as the production-center candidate. "
            "Implement only that one change, then rerun the frozen benchmark."
        )
    else:
        best = min(
            (s for s in summaries if math.isfinite(float(s["center_3d_jitter_mm"]))),
            key=lambda item: item["center_3d_jitter_mm"],
            default=None,
        )
        if best is not None:
            lines.append(
                f"No estimator met 0.5 mm. Best was {best['method']} at "
                f"{float(best['center_3d_jitter_mm']):.3f} mm. "
                "The next bottleneck is image sampling/disparity; test higher "
                "camera resolution or local stereo correlation instead of adding "
                "another temporal filter."
            )
        else:
            lines.append(
                "No estimator produced enough valid data. Inspect detector and "
                "corner-refinement failures before changing production code."
            )
    return "\n".join(lines) + "\n"


def _parse_device(value: str | None):
    if value is None:
        return CONFIG.yoloe.device
    stripped = value.strip()
    if stripped.isdigit():
        return int(stripped)
    return stripped


def _build_prompt_b_config(args: argparse.Namespace):
    model_path = Path(CONFIG.yoloe.model_name)
    if not model_path.is_absolute():
        local_model = Path(__file__).resolve().parent / model_path
        if local_model.is_file():
            model_path = local_model
    device = _parse_device(args.device)
    quantize = CONFIG.yoloe.quantize
    if str(device).lower() == "cpu":
        quantize = None
    return replace(
        CONFIG.yoloe,
        model_name=str(model_path),
        reference_boxes_xyxy=(CONFIG.yoloe.reference_boxes_xyxy[-1],),
        reference_class_ids=(0,),
        device=device,
        quantize=quantize,
        imgsz=(args.imgsz if args.imgsz is not None else CONFIG.yoloe.imgsz),
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    default_benchmark = (
        CONFIG.camera.output_dir / "prompt_ab_benchmark_v1"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark-dir",
        type=Path,
        default=default_benchmark,
        help="Directory containing manifest.json and frozen frames.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output directory. Default: <benchmark-dir>/center_estimator_diagnostic"
        ),
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Ultralytics device override, e.g. 0 or cpu.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=None,
        help="YOLOE inference-size override. Omit for exact Prompt-B settings.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N frames for a smoke test.",
    )
    parser.add_argument(
        "--corner-window",
        type=int,
        default=5,
        help="cornerSubPix half-window in pixels.",
    )
    parser.add_argument(
        "--max-corner-shift",
        type=float,
        default=6.0,
        help="Reject a refined corner that moves farther than this many pixels.",
    )
    parser.add_argument(
        "--worst-count",
        type=int,
        default=5,
        help="Worst-frame annotations saved per method.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    benchmark_root = args.benchmark_dir.expanduser().resolve()
    manifest_path = benchmark_root / "manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"Missing benchmark manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    frames = list(manifest.get("frames", []))
    if not frames:
        raise SystemExit("Benchmark manifest contains no frames.")
    if args.limit is not None:
        if args.limit <= 0:
            raise SystemExit("--limit must be positive.")
        frames = frames[: args.limit]
    expected_frames = len(frames)
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else benchmark_root / "center_estimator_diagnostic"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    prompt_cfg = _build_prompt_b_config(args)
    print("STEREO CENTER-ESTIMATOR DIAGNOSTIC", flush=True)
    print(f"Benchmark: {benchmark_root}", flush=True)
    print(f"Frames:    {expected_frames}", flush=True)
    print(f"Model:     {prompt_cfg.model_name}", flush=True)
    print(f"Device:    {prompt_cfg.device}", flush=True)
    print(f"Image size:{prompt_cfg.imgsz}", flush=True)
    print(
        "Prompt B:  one runtime-scale visual example "
        f"{prompt_cfg.reference_boxes_xyxy[0]}",
        flush=True,
    )

    detector = YOLOEPortDetector(prompt_cfg)
    started = time.perf_counter()
    detector.initialize()
    warmup = load_stereo_frame(benchmark_root, frames[0])
    detector.detect_stereo(warmup.left.rgb, warmup.right.rgb)

    desired = np.asarray(
        manifest["desired_port_virtual_camera_usd"],
        dtype=np.float64,
    )
    records_by_method = {method: [] for method in METHOD_ORDER}
    detector_failures: list[dict[str, object]] = []
    manifest_by_index: dict[int, dict[str, object]] = {}

    for sequence_index, entry in enumerate(frames, start=1):
        frame_index = int(entry["frame_index"])
        manifest_by_index[frame_index] = entry
        frame = load_stereo_frame(benchmark_root, entry)
        try:
            left_candidates, right_candidates = detector.detect_stereo(
                frame.left.rgb,
                frame.right.rgb,
            )
            cached = CachedDetector(
                left_candidates,
                right_candidates,
                {
                    "left": detector.diagnostic("left"),
                    "right": detector.diagnostic("right"),
                },
            )
            observation = process_stereo_port(
                frame=frame,
                cfg=CONFIG.perception,
                desired_port_virtual_camera_usd=desired,
                previous_left=None,
                previous_right=None,
                detector=cached,
            )
            frame_records = estimate_frame_methods(
                frame_index,
                frame,
                observation.left.detection,
                observation.right.detection,
                corner_window_radius_px=args.corner_window,
                max_corner_shift_px=args.max_corner_shift,
            )
            for record in frame_records:
                records_by_method[str(record["method"])].append(record)
            status = ", ".join(
                f"{record['method']}={'OK' if record['success'] else 'FAIL'}"
                for record in frame_records
            )
            print(
                f"[{sequence_index:02d}/{expected_frames:02d}] "
                f"frame={frame_index:04d} {status}",
                flush=True,
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            detector_failures.append(
                {"frame_index": frame_index, "error": error}
            )
            for method in METHOD_ORDER:
                failed = _blank_record(method, frame_index)
                failed["error"] = "detector/pair selection: " + error
                records_by_method[method].append(failed)
            print(
                f"[{sequence_index:02d}/{expected_frames:02d}] "
                f"frame={frame_index:04d} DETECTOR FAIL: {error}",
                flush=True,
            )

    elapsed_s = time.perf_counter() - started
    all_records = [
        record
        for method in METHOD_ORDER
        for record in records_by_method[method]
    ]
    summaries = [
        summarize_method_records(
            method,
            records_by_method[method],
            expected_frames=expected_frames,
        )
        for method in METHOD_ORDER
    ]

    _write_csv(output_dir / "details.csv", all_records, DETAIL_FIELDS)
    summary_fields = list(summaries[0].keys())
    _write_csv(output_dir / "summary.csv", summaries, summary_fields)
    (output_dir / "summary.json").write_text(
        json.dumps(
            _json_safe(
                {
                    "criteria": {
                        "minimum_success_rate": MIN_SUCCESS_RATE,
                        "minimum_decision_frame_count": MIN_DECISION_FRAME_COUNT,
                        "maximum_center_3d_jitter_mm": MAX_ACCEPTABLE_JITTER_MM,
                        "maximum_ray_gap_p95_mm": MAX_ACCEPTABLE_RAY_GAP_P95_MM,
                    },
                    "prompt": {
                        "strategy": "B_single_runtime_scale",
                        "model": str(prompt_cfg.model_name),
                        "device": str(prompt_cfg.device),
                        "imgsz": int(prompt_cfg.imgsz),
                        "reference_box": list(prompt_cfg.reference_boxes_xyxy[0]),
                    },
                    "detector_failures": detector_failures,
                    "methods": summaries,
                }
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    save_worst_frame_annotations(
        benchmark_root,
        manifest_by_index,
        records_by_method,
        output_dir,
        count=args.worst_count,
    )
    report = build_report(
        summaries,
        detector_failures,
        expected_frames=expected_frames,
        elapsed_s=elapsed_s,
    )
    (output_dir / "diagnostic_report.txt").write_text(
        report,
        encoding="utf-8",
    )
    if detector_failures:
        _write_csv(
            output_dir / "detector_failures.csv",
            detector_failures,
            ("frame_index", "error"),
        )
    print("", flush=True)
    print(report, end="", flush=True)
    print(f"Outputs: {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
