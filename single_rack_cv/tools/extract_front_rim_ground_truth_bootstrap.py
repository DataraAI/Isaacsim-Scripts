#!/usr/bin/env python3
"""Start Isaac Sim before loading OpenCV-dependent front-rim code."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config.py"
IMPLEMENTATION_PATH = PROJECT_ROOT / "tools" / "extract_front_rim_ground_truth.py"


def _put_project_root_first() -> None:
    """Prevent Kit/OpenCV paths from shadowing project modules."""
    project_root = str(PROJECT_ROOT)
    sys.path[:] = [entry for entry in sys.path if entry != project_root]
    sys.path.insert(0, project_root)


def _load_module_from_path(module_name: str, module_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module: {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


def _load_project_config():
    """Load this project's config.py as the top-level `config` module."""
    existing = sys.modules.get("config")
    if existing is not None:
        existing_path = Path(getattr(existing, "__file__", "")).resolve()
        if existing_path == CONFIG_PATH.resolve():
            return existing
        sys.modules.pop("config", None)

    return _load_module_from_path("config", CONFIG_PATH)


def _load_implementation():
    return _load_module_from_path(
        "front_rim_ground_truth_impl",
        IMPLEMENTATION_PATH,
    )


def main() -> int:
    # Kit must own native-library initialization before front_rim imports cv2.
    from isaacsim import SimulationApp

    app = SimulationApp(
        {
            "headless": False,
            "width": 1600,
            "height": 900,
        }
    )

    import isaacsim

    original_simulation_app = isaacsim.SimulationApp
    try:
        # Kit prepends its own package paths, including cv2/config.py. Force the
        # project root back to sys.path[0] and register the exact project config
        # module before importing any OpenCV-dependent project code.
        _put_project_root_first()
        _load_project_config()
        implementation = _load_implementation()

        # The implementation's main() imports SimulationApp internally. Return
        # the already-running app instead of attempting a second Kit startup.
        isaacsim.SimulationApp = lambda *_args, **_kwargs: app
        return int(implementation.main())
    finally:
        isaacsim.SimulationApp = original_simulation_app


if __name__ == "__main__":
    raise SystemExit(main())
