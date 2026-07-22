#!/usr/bin/env python3
"""High-resolution runtime configuration for the single-rack stereo pipeline."""

from __future__ import annotations

from dataclasses import replace

import config as _base_config


CAMERA_RESOLUTION_HEIGHT_WIDTH: tuple[int, int] = (960, 1280)
RENDER_RESOLUTION_WIDTH_HEIGHT: tuple[int, int] = (1280, 960)

# Keep the canonical config dataclasses unchanged while making the selected
# runtime resolution explicit and auditable. Installing the replacement back
# onto the base module ensures later `from config import CONFIG` imports inside
# project modules receive the same high-resolution object.
CONFIG = replace(
    _base_config.CONFIG,
    camera=replace(
        _base_config.CONFIG.camera,
        resolution=CAMERA_RESOLUTION_HEIGHT_WIDTH,
    ),
)
_base_config.CONFIG = CONFIG
