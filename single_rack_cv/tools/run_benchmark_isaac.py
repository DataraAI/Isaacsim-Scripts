#!/usr/bin/env python3
"""Start Isaac before importing CUDA/OpenCV benchmark modules."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config.py"


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


def _load_project_config():
    """Load this repository's config before Isaac/OpenCV can shadow it."""
    expected = CONFIG_PATH.resolve()
    existing = sys.modules.get("config")
    if existing is not None:
        existing_file = getattr(existing, "__file__", None)
        if existing_file and Path(existing_file).resolve() == expected:
            return existing
        sys.modules.pop("config", None)
    return _load_module("config", CONFIG_PATH)


def main() -> int:
    # Protect the generic module name before SimulationApp modifies sys.path.
    _load_project_config()

    from isaacsim import SimulationApp

    app = SimulationApp(
        {
            "headless": False,
            "width": 1280,
            "height": 960,
        }
    )
    try:
        project_root = str(PROJECT_ROOT)
        sys.path[:] = [entry for entry in sys.path if entry != project_root]
        sys.path.insert(0, project_root)
        from benchmarks.front_plane_benchmark import main as benchmark_main

        return int(benchmark_main())
    except Exception:
        print(
            "[FRONT-PLANE BENCHMARK FAILED]\n" + traceback.format_exc(),
            flush=True,
        )
        return 1
    finally:
        app.close()


if __name__ == "__main__":
    raise SystemExit(main())
