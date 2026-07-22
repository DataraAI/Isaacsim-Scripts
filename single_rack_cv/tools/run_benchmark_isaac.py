#!/usr/bin/env python3
"""Start Isaac before importing CUDA/OpenCV benchmark modules."""

from __future__ import annotations

from pathlib import Path
import sys
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    from isaacsim import SimulationApp

    app = SimulationApp(
        {
            "headless": False,
            "width": 1280,
            "height": 960,
        }
    )
    try:
        sys.path.insert(0, str(PROJECT_ROOT))
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
