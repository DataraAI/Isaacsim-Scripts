#!/usr/bin/env python3
"""Run the prompt benchmark with Isaac-bootstrapped CUDA evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


BENCHMARK_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = BENCHMARK_ROOT.parent
CAPTURE_SCRIPT = BENCHMARK_ROOT / "prompt_benchmark_capture.py"
EVALUATE_SCRIPT = (
    BENCHMARK_ROOT / "prompt_benchmark_evaluate_isaac_bootstrap.py"
)


def _run(script: Path) -> int:
    print(
        "\n============================================================\n"
        f"RUNNING {script.name}\n"
        "============================================================",
        flush=True,
    )
    completed = subprocess.run(
        [sys.executable, str(script)],
        cwd=PROJECT_ROOT,
        check=False,
    )
    return int(completed.returncode)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the frozen-frame YOLOE prompt A/B benchmark with "
            "Isaac Sim started before PyTorch/Ultralytics evaluation."
        )
    )
    parser.add_argument(
        "--reuse-frames",
        action="store_true",
        help=(
            "Skip capture and evaluate the existing 60 frozen stereo pairs."
        ),
    )
    args = parser.parse_args()

    required = [EVALUATE_SCRIPT]
    if not args.reuse_frames:
        required.append(CAPTURE_SCRIPT)

    for script in required:
        if not script.is_file():
            print(f"Missing benchmark component: {script}", flush=True)
            return 1

    print(
        "YOLOE PROMPT A/B BENCHMARK\n"
        f"project root: {PROJECT_ROOT}\n"
        "CUDA bootstrap: Isaac Sim before PyTorch/Ultralytics\n"
        "robot corrections during benchmark: disabled",
        flush=True,
    )

    if args.reuse_frames:
        print(
            "\n[REUSE] Existing frozen 60-frame stereo set will be used.",
            flush=True,
        )
    else:
        capture_status = _run(CAPTURE_SCRIPT)
        if capture_status != 0:
            print(
                f"\n[FAIL] Capture process returned {capture_status}.",
                flush=True,
            )
            return capture_status

    evaluation_status = _run(EVALUATE_SCRIPT)
    if evaluation_status != 0:
        print(
            f"\n[FAIL] Evaluation process returned {evaluation_status}.",
            flush=True,
        )
        return evaluation_status

    winner_path = (
        PROJECT_ROOT
        / "camera_output"
        / "prompt_ab_benchmark_v1"
        / "winner.txt"
    )
    winner_text = (
        winner_path.read_text(encoding="utf-8").strip()
        if winner_path.is_file()
        else "Evaluation completed, but winner.txt was not found."
    )

    print(
        "\n============================================================\n"
        "PROMPT A/B BENCHMARK COMPLETE\n"
        f"{winner_text}\n"
        "============================================================",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
