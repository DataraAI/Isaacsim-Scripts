#!/usr/bin/env python3
"""Capture the canonical frozen 60-pair 1280x960 stereo dataset."""

from __future__ import annotations

from pathlib import Path
import runpy
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import CONFIG

EXPECTED_RESOLUTION = (960, 1280)
if tuple(CONFIG.camera.resolution) != EXPECTED_RESOLUTION:
    raise RuntimeError(
        f"Canonical camera resolution must be {EXPECTED_RESOLUTION}, "
        f"got {CONFIG.camera.resolution}."
    )


if __name__ == "__main__":
    runpy.run_path(
        str(Path(__file__).resolve().with_name("prompt_benchmark_capture.py")),
        run_name="__main__",
    )
