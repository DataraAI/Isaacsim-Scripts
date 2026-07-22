#!/usr/bin/env python3
"""Run refined SGBM qualification only on fresh 1280x960 inputs."""

from __future__ import annotations

import json

from benchmarks import front_rim_sgbm_refined_benchmark as refined
from benchmarks.front_rim_benchmark import DATASET_DIR, GROUND_TRUTH_PATH
from highres_config import CAMERA_RESOLUTION_HEIGHT_WIDTH


EXPECTED_RESOLUTION = list(CAMERA_RESOLUTION_HEIGHT_WIDTH)


def _validate_resolution_inputs() -> None:
    manifest_path = DATASET_DIR / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            "High-resolution frozen dataset is missing. Run "
            "benchmarks/prompt_benchmark_capture_highres.py first."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    actual_manifest = manifest.get("resolution_height_width")
    if actual_manifest != EXPECTED_RESOLUTION:
        raise RuntimeError(
            "Frozen benchmark resolution is stale: "
            f"expected {EXPECTED_RESOLUTION}, got {actual_manifest}. "
            "Recapture the 60 stereo pairs at 1280x960."
        )

    if not GROUND_TRUTH_PATH.is_file():
        raise FileNotFoundError(
            "High-resolution automatic ground truth is missing."
        )
    truth = json.loads(GROUND_TRUTH_PATH.read_text(encoding="utf-8"))
    actual_truth = truth.get("camera_resolution_height_width")
    if actual_truth != EXPECTED_RESOLUTION:
        raise RuntimeError(
            "Automatic ground truth resolution is stale: "
            f"expected {EXPECTED_RESOLUTION}, got {actual_truth}. "
            "Regenerate RTX ground truth at 1280x960."
        )


def main() -> int:
    _validate_resolution_inputs()
    status = int(refined.main())

    summary_path = refined.base.OUTPUT_DIR / "summary.json"
    report_path = refined.base.OUTPUT_DIR / "report.txt"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary.update(
            {
                "schema_version": 6,
                "mode": "local_sgbm_refined_highres_v8",
                "camera_resolution_height_width": EXPECTED_RESOLUTION,
            }
        )
        summary_path.write_text(
            json.dumps(summary, indent=2) + "\n",
            encoding="utf-8",
        )
    if report_path.is_file():
        original = report_path.read_text(encoding="utf-8")
        report_path.write_text(
            "CAMERA RESOLUTION: 1280x960\n" + original,
            encoding="utf-8",
        )
    return status


if __name__ == "__main__":
    raise SystemExit(main())
