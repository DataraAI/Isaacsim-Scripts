#!/usr/bin/env python3
"""Evaluate dense front-rim stereo geometry on the frozen 60-pair dataset."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import sys
import time
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import CONFIG
from front_rim import FrontRim2D, extract_front_rim
from front_rim_stereo import FrontRim3D, triangulate_front_rims
from perception import (
    CameraFrame,
    CameraModel,
    PortDetection,
    StereoFrame,
    YOLOEPortDetector,
    process_stereo_port,
)

DATASET_DIR = CONFIG.camera.output_dir / "prompt_ab_benchmark_v1"
OUTPUT_DIR = CONFIG.camera.output_dir / "front_rim_benchmark_v1"
GROUND_TRUTH_PATH = PROJECT_ROOT / "benchmarks" / "front_rim_ground_truth.json"
EXPECTED_FRAME_COUNT = 60


@dataclass(frozen=True)
class QualificationGates:
    minimum_pair_success_rate: float = 0.95
    maximum_track_switch_count: int = 0
    maximum_radial_jitter_mm: float = 0.50
    maximum_ray_gap_p95_mm: float = 0.50
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
    left_rim_u: float = math.nan
    left_rim_v: float = math.nan
    right_rim_u: float = math.nan
    right_rim_v: float = math.nan
    center_world_x: float = math.nan
    center_world_y: float = math.nan
    center_world_z: float = math.nan
    width_mm: float = math.nan
    height_mm: float = math.nan
    accepted_sample_count: int = 0
    ray_gap_mm: float = math.nan
    reprojection_rms_px: float = math.nan
    max_reprojection_px: float = math.nan
    plane_residual_mm: float = math.nan
    plane_error_mm: float = math.nan
    left_projected_gt_error_px: float = math.nan
    right_projected_gt_error_px: float = math.nan
    rejection_reason: str = ""


class CachedDetector:
    """Feed cached YOLOE detections into production pair selection."""

    def __init__(
        self,
        left: list[PortDetection],
        right: list[PortDetection],
        diagnostics: dict[str, str],
    ) -> None:
        self.left = left
        self.right = right
        self.diagnostics = diagnostics

    def detect_stereo(
        self,
        left_rgb: np.ndarray,
        right_rgb: np.ndarray,
    ) -> tuple[list[PortDetection], list[PortDetection]]:
        del left_rgb, right_rgb
        return self.left, self.right

    def diagnostic(self, eye_name: str) -> str:
        return self.diagnostics.get(eye_name, "no cached diagnostic")


def point_to_plane_error_m(
    point_world_m: np.ndarray,
    plane_center_world_m: np.ndarray,
    plane_normal_world: np.ndarray,
) -> float:
    point = np.asarray(point_world_m, dtype=np.float64).reshape(3)
    center = np.asarray(plane_center_world_m, dtype=np.float64).reshape(3)
    normal = np.asarray(plane_normal_world, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(normal))
    if not np.all(np.isfinite(point)) or not np.all(np.isfinite(center)):
        raise ValueError("Point and plane center must be finite.")
    if not np.isfinite(norm) or norm <= 1.0e-12:
        raise ValueError("Plane normal must be finite and nonzero.")
    return abs(float(np.dot(point - center, normal / norm)))


def qualification_passes(
    *,
    pair_success_rate: float,
    track_switch_count: int,
    radial_jitter_mm: float,
    ray_gap_p95_mm: float,
    plane_error_median_mm: float,
    plane_error_p95_mm: float,
    gates: QualificationGates = GATES,
) -> bool:
    values = (
        radial_jitter_mm,
        ray_gap_p95_mm,
        plane_error_median_mm,
        plane_error_p95_mm,
    )
    return bool(
        pair_success_rate >= gates.minimum_pair_success_rate
        and track_switch_count <= gates.maximum_track_switch_count
        and all(math.isfinite(float(value)) for value in values)
        and radial_jitter_mm <= gates.maximum_radial_jitter_mm
        and ray_gap_p95_mm <= gates.maximum_ray_gap_p95_mm
        and plane_error_median_mm <= gates.maximum_plane_error_median_mm
        and plane_error_p95_mm <= gates.maximum_plane_error_p95_mm
    )


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
                record.left_rim_u,
                record.left_rim_v,
                record.right_rim_u,
                record.right_rim_v,
            ],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(current)):
            continue
        if previous is not None:
            if (
                np.linalg.norm(current[:2] - previous[:2]) > threshold_px
                or np.linalg.norm(current[2:] - previous[2:]) > threshold_px
            ):
                switches += 1
        previous = current
    return switches


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
        raise FileNotFoundError(
            "Automatic ground truth is missing. Run "
            "tools/run_front_rim_ground_truth.sh first: "
            f"{GROUND_TRUTH_PATH}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    truth = json.loads(GROUND_TRUTH_PATH.read_text(encoding="utf-8"))
    frames = manifest.get("frames")
    if int(manifest.get("frame_count", -1)) != EXPECTED_FRAME_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_FRAME_COUNT} frozen pairs, got "
            f"{manifest.get('frame_count')}."
        )
    if not isinstance(frames, list) or len(frames) != EXPECTED_FRAME_COUNT:
        raise ValueError("Frozen manifest frame list is incomplete.")
    if not str(truth.get("control_usage", "")).lower().startswith("forbidden"):
        raise ValueError("Ground-truth JSON must be marked benchmark-only.")
    return manifest, truth


def _sync_cuda() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def _draw_cross(draw: ImageDraw.ImageDraw, uv: tuple[float, float], color) -> None:
    u, v = map(float, uv)
    draw.line((u - 4, v, u + 4, v), fill=color, width=1)
    draw.line((u, v - 4, u, v + 4), fill=color, width=1)


def _save_annotation(
    path: Path,
    frame: StereoFrame,
    left_rim: FrontRim2D | None,
    right_rim: FrontRim2D | None,
    left_gt_uv: np.ndarray,
    right_gt_uv: np.ndarray,
    label: str,
) -> None:
    combined = np.concatenate((frame.left.rgb, frame.right.rgb), axis=1)
    image = Image.fromarray(combined, mode="RGB")
    draw = ImageDraw.Draw(image)
    eye_width = frame.left.rgb.shape[1]
    for rim, offset in ((left_rim, 0), (right_rim, eye_width)):
        if rim is None:
            continue
        corners = np.asarray(rim.corners_uv, dtype=np.float64).copy()
        corners[:, 0] += offset
        draw.line(
            [tuple(point) for point in np.vstack((corners, corners[0]))],
            fill=(255, 0, 255),
            width=2,
        )
        for point in rim.side_samples_uv.reshape(-1, 2):
            _draw_cross(draw, (point[0] + offset, point[1]), (255, 255, 0))
        _draw_cross(
            draw,
            (rim.center_uv[0] + offset, rim.center_uv[1]),
            (0, 255, 255),
        )
    _draw_cross(draw, tuple(left_gt_uv), (0, 255, 0))
    _draw_cross(
        draw,
        (float(right_gt_uv[0] + eye_width), float(right_gt_uv[1])),
        (0, 255, 0),
    )
    bounds = draw.multiline_textbbox((8, 8), label)
    draw.rectangle(
        (bounds[0] - 4, bounds[1] - 3, bounds[2] + 4, bounds[3] + 3),
        fill=(0, 0, 0),
    )
    draw.multiline_text((8, 8), label, fill=(255, 255, 255))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def _write_outputs(records: list[FrameRecord], summary: dict[str, object]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fields = list(asdict(records[0]).keys()) if records else list(asdict(FrameRecord(0)))
    with (OUTPUT_DIR / "details.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(asdict(record) for record in records)
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    report = ["FRONT-RIM BENCHMARK SUMMARY"]
    for key in (
        "pair_success_rate",
        "track_switch_count",
        "radial_jitter_mm",
        "ray_gap_p95_mm",
        "plane_error_median_mm",
        "plane_error_p95_mm",
        "QUALIFIED",
    ):
        report.append(f"{key}={summary[key]}")
    report.append("")
    report.append("Rejection counts:")
    for reason, count in summary["rejection_counts"].items():
        report.append(f"  {count:3d}  {reason}")
    text = "\n".join(report) + "\n"
    (OUTPUT_DIR / "report.txt").write_text(text, encoding="utf-8")
    print(text, flush=True)


def main() -> int:
    manifest, truth = _load_inputs()
    frames = manifest["frames"]
    desired = np.asarray(
        manifest["desired_port_virtual_camera_usd"], dtype=np.float64
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
        left_rim: FrontRim2D | None = None
        right_rim: FrontRim2D | None = None
        left_gt_uv = frame.left.camera.project_world(truth_center)
        right_gt_uv = frame.right.camera.project_world(truth_center)
        try:
            _sync_cuda()
            started = time.perf_counter()
            try:
                left_candidates, right_candidates = detector.detect_stereo(
                    frame.left.rgb, frame.right.rgb
                )
            finally:
                _sync_cuda()
                record.inference_ms = 1000.0 * (time.perf_counter() - started)
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

            left_rim = extract_front_rim(
                frame.left.rgb,
                left_detection.bbox_xywh,
                CONFIG.front_rim,
            )
            right_rim = extract_front_rim(
                frame.right.rgb,
                right_detection.bbox_xywh,
                CONFIG.front_rim,
            )
            result: FrontRim3D = triangulate_front_rims(
                left_rim=left_rim,
                right_rim=right_rim,
                left_camera=frame.left.camera,
                right_camera=frame.right.camera,
                cfg=CONFIG.front_rim,
            )

            record.pair_success = True
            record.left_rim_u, record.left_rim_v = left_rim.center_uv
            record.right_rim_u, record.right_rim_v = right_rim.center_uv
            (
                record.center_world_x,
                record.center_world_y,
                record.center_world_z,
            ) = result.center_world_m
            record.width_mm = 1000.0 * result.width_m
            record.height_mm = 1000.0 * result.height_m
            record.accepted_sample_count = result.accepted_sample_count
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
                np.linalg.norm(np.asarray(left_rim.center_uv) - left_gt_uv)
            )
            record.right_projected_gt_error_px = float(
                np.linalg.norm(np.asarray(right_rim.center_uv) - right_gt_uv)
            )
            successful_centers.append(
                np.asarray(result.center_world_m, dtype=np.float64)
            )
        except Exception as exc:
            record.rejection_reason = str(exc)
            rejection_counts[record.rejection_reason] = (
                rejection_counts.get(record.rejection_reason, 0) + 1
            )
        records.append(record)

        if not record.pair_success:
            _save_annotation(
                OUTPUT_DIR / "annotated" / f"failed_{frame_index:03d}.png",
                frame,
                left_rim,
                right_rim,
                left_gt_uv,
                right_gt_uv,
                f"frame={frame_index:03d} FAIL\n"
                f"{record.rejection_reason[:180]}",
            )

    success_count = sum(record.pair_success for record in records)
    success_rate = success_count / float(len(records))
    track_switches = _count_track_switches(
        records,
        CONFIG.perception.tracking_max_center_jump_px,
    )
    radial_jitter_mm = _rms_radial_jitter_mm(successful_centers)
    ray_gap_p95_mm = _percentile(
        (record.ray_gap_mm for record in records if record.pair_success),
        95.0,
    )
    plane_error_median_mm = _median(
        record.plane_error_mm for record in records if record.pair_success
    )
    plane_error_p95_mm = _percentile(
        (record.plane_error_mm for record in records if record.pair_success),
        95.0,
    )
    qualified = qualification_passes(
        pair_success_rate=success_rate,
        track_switch_count=track_switches,
        radial_jitter_mm=radial_jitter_mm,
        ray_gap_p95_mm=ray_gap_p95_mm,
        plane_error_median_mm=plane_error_median_mm,
        plane_error_p95_mm=plane_error_p95_mm,
    )
    summary = {
        "schema_version": 1,
        "dataset": str(DATASET_DIR),
        "ground_truth": str(GROUND_TRUTH_PATH),
        "total_pairs": len(records),
        "successful_pairs": success_count,
        "pair_success_rate": success_rate,
        "track_switch_count": track_switches,
        "radial_jitter_mm": radial_jitter_mm,
        "ray_gap_p95_mm": ray_gap_p95_mm,
        "plane_error_median_mm": plane_error_median_mm,
        "plane_error_p95_mm": plane_error_p95_mm,
        "QUALIFIED": qualified,
        "gates": asdict(GATES),
        "rejection_counts": dict(
            sorted(
                rejection_counts.items(),
                key=lambda item: (-item[1], item[0]),
            )
        ),
    }
    _write_outputs(records, summary)
    return 0 if qualified else 2


if __name__ == "__main__":
    raise SystemExit(main())
