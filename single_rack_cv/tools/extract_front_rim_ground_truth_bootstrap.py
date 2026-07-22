#!/usr/bin/env python3
"""Start Isaac before loading high-resolution OpenCV front-rim code."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config.py"
HIGHRES_CONFIG_PATH = PROJECT_ROOT / "highres_config.py"
IMPLEMENTATION_PATH = PROJECT_ROOT / "tools" / "extract_front_rim_ground_truth.py"
OUTPUT_PATH = PROJECT_ROOT / "benchmarks" / "front_rim_ground_truth.json"


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
    """Load base config and install the 1280x960 runtime replacement."""
    existing = sys.modules.get("config")
    if existing is not None:
        existing_path = Path(getattr(existing, "__file__", "")).resolve()
        if existing_path != CONFIG_PATH.resolve():
            sys.modules.pop("config", None)
    if "config" not in sys.modules:
        _load_module_from_path("config", CONFIG_PATH)
    return _load_module_from_path("highres_config", HIGHRES_CONFIG_PATH)


def _load_implementation():
    return _load_module_from_path(
        "front_rim_ground_truth_impl",
        IMPLEMENTATION_PATH,
    )


def _stamp_resolution_metadata() -> None:
    if not OUTPUT_PATH.is_file():
        raise RuntimeError(
            "Ground-truth extractor returned without writing its JSON output."
        )
    payload = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
    payload["schema_version"] = max(int(payload.get("schema_version", 0)), 4)
    payload["camera_resolution_height_width"] = [960, 1280]
    OUTPUT_PATH.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
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
    implementation = None
    original_write_result = None
    try:
        _put_project_root_first()
        _load_project_config()
        implementation = _load_implementation()

        # Write the resolution metadata while the extractor is writing the JSON,
        # before implementation.main() closes the shared SimulationApp.
        original_write_result = implementation._write_result

        def write_result_with_resolution(*args, **kwargs) -> None:
            original_write_result(*args, **kwargs)
            _stamp_resolution_metadata()

        implementation._write_result = write_result_with_resolution

        # The implementation's main() imports SimulationApp internally. Return
        # the already-running app instead of attempting a second Kit startup.
        isaacsim.SimulationApp = lambda *_args, **_kwargs: app
        return int(implementation.main())
    finally:
        if implementation is not None and original_write_result is not None:
            implementation._write_result = original_write_result
        isaacsim.SimulationApp = original_simulation_app


if __name__ == "__main__":
    raise SystemExit(main())
