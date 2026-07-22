#!/usr/bin/env python3
"""Run the single-rack demo with 1280x960 stereo camera sensors."""

from __future__ import annotations

from pathlib import Path
import runpy

# Importing this first installs the high-resolution CONFIG object onto the base
# config module before main.py or any downstream project module imports it.
import highres_config  # noqa: F401


if __name__ == "__main__":
    runpy.run_path(
        str(Path(__file__).resolve().with_name("main.py")),
        run_name="__main__",
    )
