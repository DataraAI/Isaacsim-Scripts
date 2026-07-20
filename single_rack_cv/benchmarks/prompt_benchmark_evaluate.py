#!/usr/bin/env python3
"""Evaluate two YOLOE visual prompts on one frozen stereo frame set."""

from __future__ import annotations

import csv
from dataclasses import dataclass, replace
import gc
import json
import math
from pathlib import Path
import sys
import time
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from PIL import Image, ImageDraw

from config import CONFIG, YOLOEConfig
from perception import (
    CameraFrame,
    CameraModel,
    PortDetection,
    StereoFrame,
    YOLOEPortDetector,
    process_stereo_port,
)
from prompt_benchmark_core import (
    BENCHMARK_FRAME_COUNT,
    MAX_CENTER_3D_JITTER_MM,
    MAX_RAY_GAP_P95_MM,
    MAX_SLOWDOWN_RATIO,
    MIN_PAIR_SUCCESS_RATE,
    apply_relative_speed_gate,
    choose_winner,
    summarize_records,
    validate_manifest,
)


BENCHMARK_DIR_NAME = "prompt_ab_benchmark_v1"
DETAIL_FIELDS = [
    "strategy",
    "frame_index",
    "left_success",
    "right_success",
    "pair_success",
    "left_candidate_count",
    "right_candidate_count",
    "inference_ms",
    "left_center_u",
    "left_center_v",
    "right_center_u",
    "right_center_v",
    "center_world_x",
    "center_world_y",
    "center_world_z",
    "estimated_range_mm",
    "ray_gap_mm",
    "reprojection_rms_px",
    "center_error_px",
    "left_diagnostic",
    "right_diagnostic",
    "error",
]


@dataclass(frozen=True)
class StrategySpec:
    name: str
    description: str
    cfg: YOLOEConfig


class CachedDetector:
    """Expose already-computed detections to the existing stereo pipeline."""

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


def _sync_cuda() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def _release_gpu(detector: YOLOEPortDetector) -> None:
    try:
        detector._model = None
    except Exception:
        pass
    del detector
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass


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


def _load_frame(
    output_root: Path,
    entry: dict[str, object],
) -> StereoFrame:
    with Image.open(output_root / str(entry["left_image"])) as image:
        left_rgb = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    with Image.open(output_root / str(entry["right_image"])) as image:
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


def _strategy_specs() -> list[StrategySpec]:
    if not CONFIG.yoloe.reference_boxes_xyxy:
        raise ValueError("The current YOLOE atlas contains no reference boxes.")

    # The final atlas example is closest to the current 25–30 px runtime port.
    single_runtime_box = CONFIG.yoloe.reference_boxes_xyxy[-1]
    single_cfg = replace(
        CONFIG.yoloe,
        reference_boxes_xyxy=(single_runtime_box,),
        reference_class_ids=(0,),
    )

    return [
        StrategySpec(
            name="A_five_scale_atlas",
            description=(
                f"current atlas with {len(CONFIG.yoloe.reference_boxes_xyxy)} "
                "same-class scale examples"
            ),
            cfg=CONFIG.yoloe,
        ),
        StrategySpec(
            name="B_single_runtime_scale",
            description=(
                "one tight atlas example closest to the rendered runtime scale; "
                f"box={tuple(map(float, single_runtime_box))}"
            ),
            cfg=single_cfg,
        ),
    ]


def _draw_detection(
    draw: ImageDraw.ImageDraw,
    detection: PortDetection,
    offset_x: int,
    color: tuple[int, int, int],
    width: int,
) -> None:
    x, y, box_width, box_height = detection.bbox_xywh
    draw.rectangle(
        [
            offset_x + x,
            y,
            offset_x + x + box_width - 1,
            y + box_height - 1,
        ],
        outline=color,
        width=width,
    )
    u, v = detection.center_uv
    u += offset_x
    draw.line([u - 5, v, u + 5, v], fill=color, width=width)
    draw.line([u, v - 5, u, v + 5], fill=color, width=width)


def _save_annotation(
    path: Path,
    strategy: StrategySpec,
    frame_index: int,
    frame: StereoFrame,
    left_candidates: list[PortDetection],
    right_candidates: list[PortDetection],
    observation,
    inference_ms: float,
    error: str,
) -> None:
    combined = np.concatenate((frame.left.rgb, frame.right.rgb), axis=1)
    image = Image.fromarray(combined, mode="RGB")
    draw = ImageDraw.Draw(image)
    eye_width = frame.left.rgb.shape[1]

    for candidate in left_candidates:
        _draw_detection(draw, candidate, 0, (255, 165, 0), 1)
    for candidate in right_candidates:
        _draw_detection(draw, candidate, eye_width, (255, 165, 0), 1)

    if observation is not None:
        _draw_detection(
            draw,
            observation.left.detection,
            0,
            (0, 255, 0),
            3,
        )
        _draw_detection(
            draw,
            observation.right.detection,
            eye_width,
            (0, 255, 0),
            3,
        )
        label = (
            f"{strategy.name} frame={frame_index:03d} PASS "
            f"inference={inference_ms:.1f}ms\n"
            f"range={observation.estimated_range_m * 1000.0:.2f}mm "
            f"ray_gap={observation.max_ray_gap_m * 1000.0:.3f}mm "
            f"center_error={float(np.linalg.norm(observation.center_error_px)):.2f}px"
        )
        color = (0, 255, 0)
    else:
        label = (
            f"{strategy.name} frame={frame_index:03d} FAIL "
            f"inference={inference_ms:.1f}ms\n{error[:180]}"
        )
        color = (255, 64, 64)

    bounds = draw.multiline_textbbox((8, 8), label)
    draw.rectangle(
        [bounds[0] - 4, bounds[1] - 3, bounds[2] + 4, bounds[3] + 3],
        fill=(0, 0, 0),
    )
    draw.multiline_text((8, 8), label, fill=color, spacing=3)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def _blank_record(strategy: str, frame_index: int) -> dict[str, object]:
    record: dict[str, object] = {field: "" for field in DETAIL_FIELDS}
    record.update(
        {
            "strategy": strategy,
            "frame_index": frame_index,
            "left_success": False,
            "right_success": False,
            "pair_success": False,
            "left_candidate_count": 0,
            "right_candidate_count": 0,
            "inference_ms": math.nan,
        }
    )
    return record


def _evaluate_strategy(
    strategy: StrategySpec,
    output_root: Path,
    manifest: dict[str, object],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    frames = manifest["frames"]
    desired = np.asarray(
        manifest["desired_port_virtual_camera_usd"],
        dtype=np.float64,
    )

    strategy_dir = output_root / "results" / strategy.name
    annotation_dir = strategy_dir / "annotated"
    strategy_dir.mkdir(parents=True, exist_ok=True)
    annotation_dir.mkdir(parents=True, exist_ok=True)

    detector = YOLOEPortDetector(strategy.cfg)
    initialization_started = time.perf_counter()
    detector.initialize()
    _sync_cuda()
    initialization_ms = 1000.0 * (
        time.perf_counter() - initialization_started
    )

    # Equal untimed warm-up for both strategies.
    warmup_frame = _load_frame(output_root, frames[0])
    detector.detect_stereo(
        warmup_frame.left.rgb,
        warmup_frame.right.rgb,
    )
    _sync_cuda()

    records: list[dict[str, object]] = []

    try:
        for entry in frames:
            frame_index = int(entry["frame_index"])
            frame = _load_frame(output_root, entry)
            record = _blank_record(strategy.name, frame_index)
            left_candidates: list[PortDetection] = []
            right_candidates: list[PortDetection] = []
            observation = None
            error = ""

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
                    record["inference_ms"] = 1000.0 * (
                        time.perf_counter() - started
                    )

                inference_ms = float(record["inference_ms"])
                left_diagnostic = detector.diagnostic("left")
                right_diagnostic = detector.diagnostic("right")
                record.update(
                    {
                        "left_success": bool(left_candidates),
                        "right_success": bool(right_candidates),
                        "left_candidate_count": len(left_candidates),
                        "right_candidate_count": len(right_candidates),
                        "inference_ms": inference_ms,
                        "left_diagnostic": left_diagnostic,
                        "right_diagnostic": right_diagnostic,
                    }
                )

                cached = CachedDetector(
                    left_candidates,
                    right_candidates,
                    {
                        "left": left_diagnostic,
                        "right": right_diagnostic,
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

                left_u, left_v = observation.left.detection.center_uv
                right_u, right_v = observation.right.detection.center_uv
                center_world = np.asarray(
                    observation.center_world_xyz_m,
                    dtype=np.float64,
                )
                record.update(
                    {
                        "pair_success": True,
                        "left_center_u": float(left_u),
                        "left_center_v": float(left_v),
                        "right_center_u": float(right_u),
                        "right_center_v": float(right_v),
                        "center_world_x": float(center_world[0]),
                        "center_world_y": float(center_world[1]),
                        "center_world_z": float(center_world[2]),
                        "estimated_range_mm": float(
                            observation.estimated_range_m * 1000.0
                        ),
                        "ray_gap_mm": float(
                            observation.max_ray_gap_m * 1000.0
                        ),
                        "reprojection_rms_px": float(
                            observation.reprojection_rms_px
                        ),
                        "center_error_px": float(
                            np.linalg.norm(observation.center_error_px)
                        ),
                    }
                )

            except Exception as exc:
                if not math.isfinite(float(record.get("inference_ms", math.nan))):
                    record["inference_ms"] = math.nan
                error = f"{type(exc).__name__}: {exc}"
                record["error"] = error

            records.append(record)
            _save_annotation(
                annotation_dir / f"frame_{frame_index:04d}.png",
                strategy,
                frame_index,
                frame,
                left_candidates,
                right_candidates,
                observation,
                float(record["inference_ms"]),
                error,
            )

            status = "PASS" if record["pair_success"] else "FAIL"
            print(
                f"[{strategy.name}] {frame_index:02d}/{BENCHMARK_FRAME_COUNT} "
                f"{status} inference={float(record['inference_ms']):.1f}ms",
                flush=True,
            )

        detail_path = strategy_dir / "details.csv"
        with detail_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=DETAIL_FIELDS)
            writer.writeheader()
            writer.writerows(records)

        summary = summarize_records(
            strategy=strategy.name,
            records=records,
            total_frames=BENCHMARK_FRAME_COUNT,
            switch_threshold_px=(
                CONFIG.perception.tracking_max_center_jump_px
            ),
        )
        summary["description"] = strategy.description
        summary["prompt_initialization_ms"] = initialization_ms
        return records, summary

    finally:
        _release_gpu(detector)


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_outputs(
    output_root: Path,
    summaries: list[dict[str, object]],
    winner: str | None,
) -> None:
    summary_path = output_root / "summary.csv"
    fields = []
    for item in summaries:
        for key in item:
            if key not in fields:
                fields.append(key)
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summaries)

    payload = {
        "criteria": {
            "minimum_pair_success_rate": MIN_PAIR_SUCCESS_RATE,
            "maximum_track_switches": 0,
            "maximum_center_3d_jitter_mm": MAX_CENTER_3D_JITTER_MM,
            "maximum_ray_gap_p95_mm": MAX_RAY_GAP_P95_MM,
            "maximum_slowdown_ratio": MAX_SLOWDOWN_RATIO,
        },
        "strategies": summaries,
        "winner": winner,
    }
    (output_root / "summary.json").write_text(
        json.dumps(_json_safe(payload), indent=2),
        encoding="utf-8",
    )

    if winner is None:
        winner_text = (
            "NO WINNER\n"
            "Neither prompt met every qualification gate. Do not move the "
            "robot closer; inspect details.csv and annotated frames first.\n"
        )
    else:
        winner_text = (
            f"WINNER: {winner}\n"
            "Freeze this prompt configuration before implementing staged "
            "50→30→20→10 mm motion.\n"
        )
    (output_root / "winner.txt").write_text(
        winner_text,
        encoding="utf-8",
    )


def main() -> int:
    try:
        output_root = CONFIG.camera.output_dir / BENCHMARK_DIR_NAME
        manifest_path = output_root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Frozen frame manifest not found: {manifest_path}. "
                "Run capture first."
            )

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        validate_manifest(manifest, BENCHMARK_FRAME_COUNT)

        summaries: list[dict[str, object]] = []
        for strategy in _strategy_specs():
            print(
                "\n============================================================\n"
                f"EVALUATING {strategy.name}\n"
                f"{strategy.description}\n"
                "============================================================",
                flush=True,
            )
            _, summary = _evaluate_strategy(
                strategy,
                output_root,
                manifest,
            )
            summaries.append(summary)

        summaries = apply_relative_speed_gate(
            summaries,
            MAX_SLOWDOWN_RATIO,
        )
        winner = choose_winner(summaries)
        _write_outputs(output_root, summaries, winner)

        print("\nPROMPT A/B BENCHMARK SUMMARY", flush=True)
        for item in summaries:
            print(
                f"  {item['strategy']}: "
                f"pair={100.0 * float(item['pair_success_rate']):.1f}% "
                f"switches={item['track_switch_count']} "
                f"3d_jitter={float(item['center_3d_jitter_mm']):.3f}mm "
                f"ray_p95={float(item['ray_gap_p95_mm']):.3f}mm "
                f"median={float(item['inference_median_ms']):.1f}ms "
                f"qualified={item['qualified']}",
                flush=True,
            )

        print(
            f"\nRESULT: {winner if winner is not None else 'NO WINNER'}\n"
            f"Outputs: {output_root}",
            flush=True,
        )
        return 0

    except Exception:
        print(
            "\n[FAIL] PROMPT A/B EVALUATION\n"
            + traceback.format_exc(),
            flush=True,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
