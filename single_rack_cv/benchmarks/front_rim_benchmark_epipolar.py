#!/usr/bin/env python3
"""Evaluate epipolar-matched front-bezel geometry on the frozen dataset."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import json
import math
import time

import numpy as np

from benchmarks.front_rim_benchmark import (
    CachedDetector,
    DATASET_DIR,
    EXPECTED_FRAME_COUNT,
    FrameRecord,
    GATES,
    GROUND_TRUTH_PATH,
    OUTPUT_DIR,
    _count_track_switches,
    _load_frame,
    _load_inputs,
    _median,
    _percentile,
    _rms_radial_jitter_mm,
    _save_annotation,
    _sync_cuda,
    point_to_plane_error_m,
    qualification_passes,
)
from config import CONFIG
from front_rim import FrontRim2D, extract_front_rim
from front_rim_match import (
    DEFAULT_EPIPOLAR_CONFIG,
    EpipolarMatchResult,
    build_matched_right_rim,
    match_front_bezel_samples,
)
from front_rim_stereo import FrontRim3D, triangulate_front_rims
from perception import YOLOEPortDetector, process_stereo_port


@dataclass
class EpipolarFrameRecord(FrameRecord):
    matched_sample_count: int = 0
    match_ncc_median: float = math.nan
    match_uniqueness_median: float = math.nan
    match_roundtrip_p95_px: float = math.nan


def _write_outputs(
    records: list[EpipolarFrameRecord],
    summary: dict[str, object],
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fields = list(asdict(records[0]).keys()) if records else []
    with (OUTPUT_DIR / "details.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(asdict(record) for record in records)
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    report = ["FRONT-RIM EPIPOLAR BENCHMARK SUMMARY"]
    for key in (
        "pair_success_rate",
        "track_switch_count",
        "radial_jitter_mm",
        "ray_gap_p95_mm",
        "plane_error_median_mm",
        "plane_error_p95_mm",
        "match_count_median",
        "match_ncc_median",
        "match_uniqueness_median",
        "match_roundtrip_p95_px",
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
    if len(frames) != EXPECTED_FRAME_COUNT:
        raise ValueError("Frozen benchmark frame list is incomplete.")
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

    records: list[EpipolarFrameRecord] = []
    successful_centers: list[np.ndarray] = []
    rejection_counts: dict[str, int] = {}

    for entry in frames:
        frame_index = int(entry["frame_index"])
        frame = _load_frame(DATASET_DIR, entry)
        record = EpipolarFrameRecord(frame_index=frame_index)
        left_rim: FrontRim2D | None = None
        right_rim: FrontRim2D | None = None
        annotation_right_rim: FrontRim2D | None = None
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
            record.left_detection_u, record.left_detection_v = (
                left_detection.center_uv
            )
            record.right_detection_u, record.right_detection_v = (
                right_detection.center_uv
            )

            left_rim = extract_front_rim(
                frame.left.rgb,
                left_detection.bbox_xywh,
                CONFIG.front_rim,
                center_uv=left_detection.center_uv,
            )
            right_rim = extract_front_rim(
                frame.right.rgb,
                right_detection.bbox_xywh,
                CONFIG.front_rim,
                center_uv=right_detection.center_uv,
            )
            matches: EpipolarMatchResult = match_front_bezel_samples(
                frame.left.rgb,
                frame.right.rgb,
                left_rim,
                right_rim,
            )
            record.matched_sample_count = matches.accepted_count
            record.match_ncc_median = matches.median_ncc()
            record.match_uniqueness_median = matches.median_uniqueness()
            record.match_roundtrip_p95_px = matches.roundtrip_p95_px()
            annotation_right_rim = build_matched_right_rim(
                left_rim,
                right_rim,
                matches,
                CONFIG.front_rim.sample_max_epipolar_error_px,
            )
            result: FrontRim3D = triangulate_front_rims(
                left_rim=left_rim,
                right_rim=annotation_right_rim,
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
                annotation_right_rim or right_rim,
                left_gt_uv,
                right_gt_uv,
                f"frame={frame_index:03d} FAIL\n"
                f"matches={record.matched_sample_count}/28\n"
                f"{record.rejection_reason[:160]}",
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
        "schema_version": 2,
        "mode": "epipolar_patch_v4",
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
        "match_count_median": _median(
            record.matched_sample_count for record in records
        ),
        "match_ncc_median": _median(
            record.match_ncc_median for record in records
        ),
        "match_uniqueness_median": _median(
            record.match_uniqueness_median for record in records
        ),
        "match_roundtrip_p95_px": _percentile(
            (record.match_roundtrip_p95_px for record in records),
            95.0,
        ),
        "matcher": asdict(DEFAULT_EPIPOLAR_CONFIG),
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
