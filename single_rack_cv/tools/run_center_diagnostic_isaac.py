#!/usr/bin/env python3
"""Isaac Sim bootstrap for the offline stereo-center diagnostic.

This version explicitly loads the project's config.py and perception.py by
absolute file path. That prevents Isaac Sim/OpenCV's bundled ``cv2/config.py``
from shadowing the project's top-level ``config.py``.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import traceback
from types import ModuleType


TOOLS_DIR = Path(__file__).resolve().parent
ROOT = TOOLS_DIR.parent


def _prioritize_project_paths(root: Path, tools_dir: Path) -> None:
    """Put project root first and tools second, removing stale duplicates."""
    ordered = [str(root.resolve()), str(tools_dir.resolve())]
    sys.path[:] = [
        entry
        for entry in sys.path
        if str(Path(entry or ".").resolve()) not in set(ordered)
    ]
    sys.path[:0] = ordered


def _load_module_from_file(name: str, path: Path) -> ModuleType:
    """Load one module from an exact path and register it under ``name``."""
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Required module file does not exist: {resolved}")

    spec = importlib.util.spec_from_file_location(name, resolved)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create an import spec for {resolved}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module

    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise

    return module


def _load_project_modules() -> ModuleType:
    """Load project modules in dependency order using exact file paths."""
    _prioritize_project_paths(ROOT, TOOLS_DIR)

    config_module = _load_module_from_file("config", ROOT / "config.py")
    config_path = Path(config_module.__file__).resolve()
    expected_config = (ROOT / "config.py").resolve()
    if config_path != expected_config:
        raise RuntimeError(
            "Wrong config module loaded: "
            f"{config_path}; expected {expected_config}"
        )

    _load_module_from_file("perception", ROOT / "perception.py")
    return _load_module_from_file(
        "diagnose_stereo_centers",
        TOOLS_DIR / "diagnose_stereo_centers.py",
    )


def main() -> int:
    simulation_app = None

    try:
        from isaacsim import SimulationApp

        simulation_app = SimulationApp(
            {
                "headless": True,
                "width": 640,
                "height": 480,
            }
        )

        import torch

        print("\n=== CENTER DIAGNOSTIC CUDA ENVIRONMENT ===", flush=True)
        print(f"torch version: {torch.__version__}", flush=True)
        print(f"torch CUDA build: {torch.version.cuda}", flush=True)
        print(f"CUDA available: {torch.cuda.is_available()}", flush=True)

        if not torch.cuda.is_available():
            raise RuntimeError(
                "Isaac Sim started, but PyTorch CUDA is unavailable."
            )

        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

        diagnostic = _load_project_modules()

        loaded_config = Path(sys.modules["config"].__file__).resolve()
        print(f"Project config: {loaded_config}", flush=True)
        print(
            f"Diagnostic script: "
            f"{Path(diagnostic.__file__).resolve()}",
            flush=True,
        )

        return int(
            diagnostic.main(
                [
                    "--benchmark-dir",
                    str(ROOT / "camera_output" / "prompt_ab_benchmark_v1"),
                    "--output-dir",
                    str(
                        ROOT
                        / "camera_output"
                        / "prompt_ab_benchmark_v1"
                        / "center_estimator_diagnostic"
                    ),
                    "--device",
                    "0",
                ]
            )
        )

    except Exception:
        print(
            "\n[FAIL] ISAAC-BOOTSTRAPPED CENTER DIAGNOSTIC\n"
            + traceback.format_exc(),
            flush=True,
        )
        return 1

    finally:
        if simulation_app is not None:
            simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
