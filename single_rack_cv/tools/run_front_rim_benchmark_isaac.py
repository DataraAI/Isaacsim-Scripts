#!/usr/bin/env python3
"""Start Isaac before CUDA/OpenCV imports, then run epipolar benchmark."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config.py"
BENCHMARK_PATH = (
    PROJECT_ROOT / "benchmarks" / "front_rim_benchmark_epipolar.py"
)


def _load_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": False, "width": 640, "height": 480})
    try:
        sys.path.insert(0, str(PROJECT_ROOT))
        sys.modules.pop("config", None)
        _load_path("config", CONFIG_PATH)
        benchmark = _load_path("front_rim_benchmark_impl", BENCHMARK_PATH)
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
