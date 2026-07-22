#!/usr/bin/env python3
"""Capture the frozen 60-pair dataset with 1280x960 stereo cameras."""

from __future__ import annotations

from pathlib import Path
import runpy
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import highres_config  # noqa: E402,F401


if __name__ == "__main__":
    runpy.run_path(
        str(Path(__file__).resolve().with_name("prompt_benchmark_capture.py")),
        run_name="__main__",
    )
