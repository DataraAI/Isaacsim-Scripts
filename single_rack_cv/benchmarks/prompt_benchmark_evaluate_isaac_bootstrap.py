#!/usr/bin/env python3
"""Start Isaac Sim first so YOLOE evaluation uses Isaac's CUDA 12.8 PyTorch."""

from __future__ import annotations

from pathlib import Path
import sys
import traceback


BENCHMARK_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = BENCHMARK_ROOT.parent
for path in (PROJECT_ROOT, BENCHMARK_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

# Pin the project's config module before Isaac/OpenCV adds pip-prebundle paths.
from config import CONFIG
from isaacsim import SimulationApp


def main() -> int:
    simulation_app = None

    try:
        simulation_app = SimulationApp(
            {
                "headless": True,
                "width": 640,
                "height": 480,
            }
        )

        import torch

        print(
            "\n=== CUDA ENVIRONMENT AFTER ISAAC STARTUP ===",
            flush=True,
        )
        print(f"torch version: {torch.__version__}", flush=True)
        print(f"torch CUDA build: {torch.version.cuda}", flush=True)
        print(
            f"CUDA available: {torch.cuda.is_available()}",
            flush=True,
        )
        print(
            f"device count: {torch.cuda.device_count()}",
            flush=True,
        )

        if not torch.cuda.is_available():
            raise RuntimeError(
                "Isaac Sim started, but PyTorch CUDA is still unavailable. "
                "Do not run the benchmark on CPU because the timing comparison "
                "would not represent the working YOLOE runtime."
            )

        print(
            f"GPU: {torch.cuda.get_device_name(0)}",
            flush=True,
        )

        # Import only after SimulationApp has activated Isaac's compatible
        # torch 2.11.0+cu128 environment.
        import prompt_benchmark_evaluate

        return int(prompt_benchmark_evaluate.main())

    except Exception:
        print(
            "\n[FAIL] ISAAC-BOOTSTRAPPED PROMPT EVALUATION\n"
            + traceback.format_exc(),
            flush=True,
        )
        return 1

    finally:
        if simulation_app is not None:
            simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
