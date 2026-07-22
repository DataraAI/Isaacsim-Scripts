#!/usr/bin/env python3
"""Start Isaac before CUDA/OpenCV imports, then run high-resolution SGBM."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config.py"
HIGHRES_CONFIG_PATH = PROJECT_ROOT / "highres_config.py"
BENCHMARK_PATH = (
    PROJECT_ROOT / "benchmarks" / "front_rim_sgbm_highres_benchmark.py"
)


def _load_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _install_safe_disparity_writer(benchmark) -> None:
    """Replace the base benchmark writer with channel-safe NumPy indexing."""
    import cv2
    import numpy as np
    from PIL import Image

    refined = getattr(benchmark, "refined", benchmark)
    target = getattr(refined, "base", refined)

    def save(path, disparity) -> None:
        values = np.asarray(disparity.disparity_crop_px, dtype=np.float32)
        center = float(disparity.center_disparity_px)
        half = float(target.DEFAULT_SGBM_CONFIG.disparity_half_range_px)
        normalized = np.clip(
            (values - (center - half)) / (2.0 * half),
            0.0,
            1.0,
        )
        gray = np.round(255.0 * normalized).astype(np.uint8)
        color_bgr = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
        color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
        color_rgb[~disparity.valid_mask] = 0
        consistent = disparity.consistent_mask
        color_rgb[..., 1][consistent] = 255
        color_rgb[..., 0][consistent] //= 2
        color_rgb[..., 2][consistent] //= 2
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(color_rgb, mode="RGB").save(path)

    target._save_disparity_debug = save


def main() -> int:
    from isaacsim import SimulationApp

    # DLSS internally scales the renderer input. Starting at 1280x960 keeps the
    # scaled input above the 300-pixel minimum and matches the stereo sensors.
    app = SimulationApp(
        {"headless": False, "width": 1280, "height": 960}
    )
    try:
        sys.path.insert(0, str(PROJECT_ROOT))
        sys.modules.pop("config", None)
        _load_path("config", CONFIG_PATH)
        _load_path("highres_config", HIGHRES_CONFIG_PATH)
        benchmark = _load_path("front_rim_benchmark_impl", BENCHMARK_PATH)
        _install_safe_disparity_writer(benchmark)
        return int(benchmark.main())
    except Exception:
        print(
            "[FRONT-RIM BENCHMARK FAILED]\n" + traceback.format_exc(),
            flush=True,
        )
        return 1
    finally:
        app.close()


if __name__ == "__main__":
    raise SystemExit(main())
