#!/usr/bin/env python3
"""Strict 1280x960 qualification for the canonical front-plane estimator."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import sys
import time
from typing import Iterable

import cv2
import numpy as np
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import CONFIG
from vision.front_plane import (
    DEFAULT_FRONT_PLANE_CONFIG,
    FrontPlaneResult,
    LocalDisparityResult,
    compute_local_disparity,
    estimate_front_plane,
)
from vision.perception import (
    CameraFrame,
    CameraModel,
    PortDetection,
    StereoFrame,
    YOLOEPortDetector,
    process_stereo_port,
)

EXPECTED_RESOLUTION = [960, 1280]
EXPECTED_FRAME_COUNT = 60
DATASET_DIR = CONFIG.camera.output_dir / "prompt_ab_benchmark_v1"
OUTPUT_DIR = CONFIG.camera.output_dir / "front_plane_benchmark"
GROUND_TRUTH_PATH = PROJECT_ROOT / "benchmarks" / "front_plane_ground_truth.json"


@dataclass(frozen=True)
class QualificationGates:
    minimum_pair_success_rate: float = 0.95
    maximum_track_switch_count: int = 0
    maximum_radial_jitter_mm: float = 0.50
    maximum_ray_gap_p95_mm: float = 0.50
    maximum_plane_residual_p95_mm: float = 0.50
    maximum_plane_error_median_mm: float = 0.50
    maximum_plane_error_p95_mm: float = 1.00


GATES = QualificationGates()


@dataclass
class FrameRecord:
    frame_index: int
    pair_success: bool = False
    left_candidate_count: int = 0
    right_candidate_count: int = 0
    inference_ms: float = math.nan
    left_detection_u: float = math.nan
    left_detection_v: float = math.nan
    right_detection_u: float = math.nan
    right_detection_v: float = math.nan
    center_world_x: float = math.nan
    center_world_y: float = math.nan
    center_world_z: float = math.nan
    width_mm: float = math.nan
    height_mm: float = math.nan
    ray_gap_mm: float = math.nan
    reprojection_rms_px: float = math.nan
    max_reprojection_px: float = math.nan
    plane_residual_mm: float = math.nan
    plane_error_mm: float = math.nan
    left_projected_gt_error_px: float = math.nan
    right_projected_gt_error_px: float = math.nan
    valid_disparity_count: int = 0
    consistent_disparity_count: int = 0
    ring_candidate_count: int = 0
    triangulated_count: int = 0
    cluster_count: int = 0
    top_support: int = 0
    right_support: int = 0
    bottom_support: int = 0
    left_support: int = 0
    median_disparity_px: float = math.nan
    rejection_reason: str = ""


class CachedDetector:
    def __init__(
        self,
        left: list[PortDetection],
        right: list[PortDetection],
        diagnostics: dict[str, str],
    ) -> None:
        self.left = left
        self.right = right
        self.diagnostics = diagnostics

    def detect_stereo(self, left_rgb, right_rgb):
        del left_rgb, right_rgb
        return self.left, self.right

    def diagnostic(self, eye_name: str) -> str:
        return self.diagnostics.get(eye_name, "no cached diagnostic")


def _finite(values: Iterable[float]) -> np.ndarray:
    array = np.asarray(list(values), dtype=np.float64).reshape(-1)
    return array[np.isfinite(array)]


def _percentile(values: Iterable[float], q: float) -> float:
    array = _finite(values)
    return float(np.percentile(array, q)) if array.size else math.nan


def _median(values: Iterable[float]) -> float:
    array = _finite(values)
    return float(np.median(array)) if array.size else math.nan


def _rms_radial_jitter_mm(points: list[np.ndarray]) -> float:
    if not points:
        return math.nan
    array = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    median = np.median(array, axis=0)
    return 1000.0 * float(
        np.sqrt(np.mean(np.sum((array - median) ** 2, axis=1)))
    )


def _count_track_switches(records: list[FrameRecord], threshold_px: float) -> int:
    previous: np.ndarray | None = None
    switches = 0
    for record in records:
        if not record.pair_success:
            continue
        current = np.asarray(
            [
                record.left_detection_u,
                record.left_detection_v,
                record.right_detection_u,
                record.right_detection_v,
            ],
            dtype=np.float64,
        )
        if previous is not None and (
            np.linalg.norm(current[:2] - previous[:2]) > threshold_px
            or np.linalg.norm(current[2:] - previous[2:]) > threshold_px
        ):
            switches += 1
        previous = current
    return switches


def point_to_plane_error_m(point, plane_center, plane_normal) -> float:
    point = np.asarray(point, dtype=np.float64).reshape(3)
    center = np.asarray(plane_center, dtype=np.float64).reshape(3)
    normal = np.asarray(plane_normal, dtype=np.float64).reshape(3)
    normal /= np.linalg.norm(normal)
    return abs(float(np.dot(point - center, normal)))


def _camera_from_dict(data: dict[str, object]) -> CameraModel:
    return CameraModel(
        image_height_px=int(data["image_height_px"]),
        image_width_px=int(data["image_width_px"]),
        focal_length_mm=float(data["focal_length_mm"]),
        horizontal_aperture_mm=float(data["horizontal_aperture_mm"]),
        vertical_aperture_mm=float(data["vertical_aperture_mm"]),
        world_from_camera=np.asarray(data["world_from_camera"], dtype=np.float64),
    )


def _load_frame(root: Path, entry: dict[str, object]) -> StereoFrame:
    with Image.open(root / str(entry["left_image"])) as image:
        left_rgb = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    with Image.open(root / str(entry["right_image"])) as image:
        right_rgb = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    return StereoFrame(
        left=CameraFrame(left_rgb, _camera_from_dict(entry["left_camera"])),
        right=CameraFrame(right_rgb, _camera_from_dict(entry["right_camera"])),
        virtual_camera=_camera_from_dict(entry["virtual_camera"]),
    )


def _load_inputs() -> tuple[dict[str, object], dict[str, object]]:
    manifest_path = DATASET_DIR / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Frozen benchmark manifest not found: {manifest_path}")
    if not GROUND_TRUTH_PATH.is_file():
        raise FileNotFoundError(f"Ground truth not found: {GROUND_TRUTH_PATH}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    truth = json.loads(GROUND_TRUTH_PATH.read_text(encoding="utf-8"))
    if manifest.get("resolution_height_width") != EXPECTED_RESOLUTION:
        raise RuntimeError("Frozen dataset is not 1280x960.")
    if truth.get("camera_resolution_height_width") != EXPECTED_RESOLUTION:
        raise RuntimeError("Ground truth is not 1280x960.")
    frames = manifest.get("frames")
    if int(manifest.get("frame_count", -1)) != EXPECTED_FRAME_COUNT:
        raise ValueError("Frozen benchmark frame count is not 60.")
    if not isinstance(frames, list) or len(frames) != EXPECTED_FRAME_COUNT:
        raise ValueError("Frozen benchmark frame list is incomplete.")
    if not str(truth.get("control_usage", "")).lower().startswith("forbidden"):
        raise ValueError("Ground truth must be marked benchmark-only.")
    return manifest, truth


def _sync_cuda() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def _save_disparity(path: Path, disparity: LocalDisparityResult) -> None:
    values = np.asarray(disparity.disparity_crop_px, dtype=np.float32)
    center = float(disparity.center_disparity_px)
    half = float(DEFAULT_FRONT_PLANE_CONFIG.disparity_half_range_px)
    normalized = np.clip(
        (values - (center - half)) / (2.0 * half),
        0.0,
        1.0,
    )
    gray = np.round(255.0 * normalized).astype(np.uint8)
    color = cv2.cvtColor(
        cv2.applyColorMap(gray, cv2.COLORMAP_TURBO),
        cv2.COLOR_BGR2RGB,
    )
    color[~disparity.valid_mask] = 0
    consistent = disparity.consistent_mask
    color[..., 1][consistent] = 255
    color[..., 0][consistent] //= 2
    color[..., 2][consistent] //= 2
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(color, mode="RGB").save(path)


def _draw_cross(draw: ImageDraw.ImageDraw, uv, color) -> None:
    u, v = map(float, uv)
    draw.line((u - 4, v, u + 4, v), fill=color, width=1)
    draw.line((u, v - 4, u, v + 4), fill=color, width=1)


def _save_failure_annotation(
    path: Path,
    frame: StereoFrame,
    record: FrameRecord,
    left_gt_uv: np.ndarray,
    right_gt_uv: np.ndarray,
) -> None:
    combined = np.concatenate((frame.left.rgb, frame.right.rgb), axis=1)
    image = Image.fromarray(combined, mode="RGB")
    draw = ImageDraw.Draw(image)
    eye_width = frame.left.rgb.shape[1]
    _draw_cross(draw, left_gt_uv, (0, 255, 0))
    _draw_cross(
        draw,
        (right_gt_uv[0] + eye_width, right_gt_uv[1]),
        (0, 255, 0),
    )
    if math.isfinite(record.left_detection_u):
        _draw_cross(
            draw,
            (record.left_detection_u, record.left_detection_v),
            (0, 255, 255),
        )
    if math.isfinite(record.right_detection_u):
        _draw_cross(
            draw,
            (
                record.right_detection_u + eye_width,
                record.right_detection_v,
            ),
            (0, 255, 255),
        )
    label = (
        f"frame={record.frame_index:03d} FAIL\n"
        f"valid={record.valid_disparity_count} "
        f"consistent={record.consistent_disparity_count}\n"
        f"{record.rejection_reason[:160]}"
    )
    bounds = draw.multiline_textbbox((8, 8), label)
    draw.rectangle(
        (bounds[0] - 4, bounds[1] - 3, bounds[2] + 4, bounds[3] + 3),
        fill=(0, 0, 0),
    )
    draw.multiline_text((8, 8), label, fill=(255, 255, 255))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def _qualified(summary: dict[str, object]) -> bool:
    return bool(
        summary["pair_success_rate"] >= GATES.minimum_pair_success_rate
        and summary["track_switch_count"] <= GATES.maximum_track_switch_count
        and summary["radial_jitter_mm"] <= GATES.maximum_radial_jitter_mm
        and summary["ray_gap_p95_mm"] <= GATES.maximum_ray_gap_p95_mm
        and summary["plane_residual_p95_mm"]
        <= GATES.maximum_plane_residual_p95_mm
        and summary["plane_error_median_mm"]
        <= GATES.maximum_plane_error_median_mm
        and summary["plane_error_p95_mm"]
        <= GATES.maximum_plane_error_p95_mm
    )


def _write_outputs(records: list[FrameRecord], summary: dict[str, object]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fields = list(asdict(records[0]).keys()) if records else []
    with (OUTPUT_DIR / "details.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(asdict(record) for record in records)
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    keys = (
        "pair_success_rate",
        "track_switch_count",
        "radial_jitter_mm",
        "ray_gap_p95_mm",
        "plane_residual_p95_mm",
        "plane_error_median_mm",
        "plane_error_p95_mm",
        "valid_count_median",
        "consistent_count_median",
        "ring_count_median",
        "triangulated_count_median",
        "cluster_count_median",
        "median_disparity_px",
        "QUALIFIED",
    )
    lines = ["FRONT-PLANE 1280x960 BENCHMARK SUMMARY"]
    lines.extend(f"{key}={summary[key]}" for key in keys)
    lines.extend(
        (
            "",
            "Strict qualification:",
            "  pair_success_rate>=0.95",
            "  track_switch_count=0",
            "  radial_jitter_mm<=0.5",
            "  ray_gap_p95_mm<=0.5",
            "  plane_residual_p95_mm<=0.5",
            "  plane_error_median_mm<=0.5",
            "  plane_error_p95_mm<=1.0",
            "",
            "Rejection counts:",
        )
    )
    for reason, count in summary["rejection_counts"].items():
        lines.append(f"  {count:3d}  {reason}")
    text = "\n".join(lines) + "\n"
    (OUTPUT_DIR / "report.txt").write_text(text, encoding="utf-8")
    print(text, flush=True)


def main() -> int:
    manifest, truth = _load_inputs()
    frames = manifest["frames"]
    desired = np.asarray(
        manifest["desired_port_virtual_camera_usd"],
        dtype=np.float64,
    )
    truth_center = np.asarray(truth["center_world_m"], dtype=np.float64)
    truth_normal = np.asarray(truth["normal_world"], dtype=np.float64)

    detector = YOLOEPortDetector(CONFIG.yoloe)
    detector.initialize()
    warmup = _load_frame(DATASET_DIR, frames[0])
    detector.detect_stereo(warmup.left.rgb, warmup.right.rgb)
    _sync_cuda()

    records: list[FrameRecord] = []
    successful_centers: list[np.ndarray] = []
    rejection_counts: dict[str, int] = {}

    for entry in frames:
        frame_index = int(entry["frame_index"])
        frame = _load_frame(DATASET_DIR, entry)
        record = FrameRecord(frame_index=frame_index)
        disparity: LocalDisparityResult | None = None
        left_gt_uv = frame.left.camera.project_world(truth_center)
        right_gt_uv = frame.right.camera.project_world(truth_center)
        try:
            _sync_cuda()
            started = time.perf_counter()
            try:
                left_candidates, right_candidates = detector.detect_stereo(
                    frame.left.rgb,
                    frame.right.rgb,
                )
            finally:
                _sync_cuda()
                record.inference_ms = 1000.0 * (
                    time.perf_counter() - started
                )
            record.left_candidate_count = len(left_candidates)
            record.right_candidate_count = len(right_candidates)
            cached = CachedDetector(
                left_candidates,
                right_candidates,
                {
                    "left": detector.diagnostic("left"),
                    "right": detector.diagnostic("right"),
                },
            )
            selected = process_stereo_port(
                frame=frame,
                cfg=CONFIG.perception,
                desired_port_virtual_camera_usd=desired,
                previous_left=None,
                previous_right=None,
                detector=cached,
            )
            left_detection = selected.left.detection
            right_detection = selected.right.detection
            (
                record.left_detection_u,
                record.left_detection_v,
            ) = left_detection.center_uv
            (
                record.right_detection_u,
                record.right_detection_v,
            ) = right_detection.center_uv

            disparity = compute_local_disparity(
                frame.left.rgb,
                frame.right.rgb,
                left_detection.bbox_xywh,
                left_detection.center_uv,
                right_detection.bbox_xywh,
                right_detection.center_uv,
            )
            record.valid_disparity_count = disparity.valid_count
            record.consistent_disparity_count = disparity.consistent_count
            result: FrontPlaneResult = estimate_front_plane(
                frame.left.rgb,
                frame.right.rgb,
                left_detection.bbox_xywh,
                left_detection.center_uv,
                right_detection.bbox_xywh,
                right_detection.center_uv,
                frame.left.camera,
                frame.right.camera,
                disparity=disparity,
            )

            record.pair_success = True
            (
                record.center_world_x,
                record.center_world_y,
                record.center_world_z,
            ) = result.center_world_m
            record.width_mm = 1000.0 * result.width_m
            record.height_mm = 1000.0 * result.height_m
            record.ray_gap_mm = 1000.0 * result.max_ray_gap_m
            record.reprojection_rms_px = result.reprojection_rms_px
            record.max_reprojection_px = result.max_reprojection_px
            record.plane_residual_mm = 1000.0 * result.plane_residual_m
            record.plane_error_mm = 1000.0 * point_to_plane_error_m(
                result.center_world_m,
                truth_center,
                truth_normal,
            )
            record.left_projected_gt_error_px = float(
                np.linalg.norm(
                    np.asarray(left_detection.center_uv) - left_gt_uv
                )
            )
            record.right_projected_gt_error_px = float(
                np.linalg.norm(
                    np.asarray(right_detection.center_uv) - right_gt_uv
                )
            )
            record.ring_candidate_count = result.ring_candidate_count
            record.triangulated_count = result.triangulated_count
            record.cluster_count = result.cluster_count
            (
                record.top_support,
                record.right_support,
                record.bottom_support,
                record.left_support,
            ) = result.side_support_counts
            record.median_disparity_px = result.median_disparity_px
            successful_centers.append(
                np.asarray(result.center_world_m, dtype=np.float64)
            )
        except Exception as exc:
            record.rejection_reason = str(exc)
            rejection_counts[record.rejection_reason] = (
                rejection_counts.get(record.rejection_reason, 0) + 1
            )
        records.append(record)

        if disparity is not None:
            _save_disparity(
                OUTPUT_DIR / "disparity" / f"frame_{frame_index:03d}.png",
                disparity,
            )
        if not record.pair_success:
            _save_failure_annotation(
                OUTPUT_DIR / "annotated" / f"failed_{frame_index:03d}.png",
                frame,
                record,
                left_gt_uv,
                right_gt_uv,
            )

    success_count = sum(record.pair_success for record in records)
    summary: dict[str, object] = {
        "schema_version": 1,
        "mode": "front_plane_highres_v1",
        "camera_resolution_height_width": EXPECTED_RESOLUTION,
        "dataset": str(DATASET_DIR),
        "ground_truth": str(GROUND_TRUTH_PATH),
        "total_pairs": len(records),
        "successful_pairs": success_count,
        "pair_success_rate": success_count / float(len(records)),
        "track_switch_count": _count_track_switches(
            records,
            CONFIG.perception.tracking_max_center_jump_px,
        ),
        "radial_jitter_mm": _rms_radial_jitter_mm(successful_centers),
        "ray_gap_p95_mm": _percentile(
            (
                record.ray_gap_mm
                for record in records
                if record.pair_success
            ),
            95.0,
        ),
        "plane_residual_p95_mm": _percentile(
            (
                record.plane_residual_mm
                for record in records
                if record.pair_success
            ),
            95.0,
        ),
        "plane_error_median_mm": _median(
            record.plane_error_mm
            for record in records
            if record.pair_success
        ),
        "plane_error_p95_mm": _percentile(
            (
                record.plane_error_mm
                for record in records
                if record.pair_success
            ),
            95.0,
        ),
        "valid_count_median": _median(
            record.valid_disparity_count for record in records
        ),
        "consistent_count_median": _median(
            record.consistent_disparity_count for record in records
        ),
        "ring_count_median": _median(
            record.ring_candidate_count for record in records
        ),
        "triangulated_count_median": _median(
            record.triangulated_count for record in records
        ),
        "cluster_count_median": _median(
            record.cluster_count for record in records
        ),
        "median_disparity_px": _median(
            record.median_disparity_px for record in records
        ),
        "front_plane_config": asdict(DEFAULT_FRONT_PLANE_CONFIG),
        "gates": asdict(GATES),
        "rejection_counts": dict(
            sorted(
                rejection_counts.items(),
                key=lambda item: (-item[1], item[0]),
            )
        ),
    }
    summary["QUALIFIED"] = _qualified(summary)
    _write_outputs(records, summary)
    return 0 if summary["QUALIFIED"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
