#!/usr/bin/env python3
"""Generate benchmark-only RTX ground truth for the front opening."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config.py"
IMPLEMENTATION_PATH = PROJECT_ROOT / "tools" / "extract_front_rim_ground_truth.py"
OUTPUT_PATH = PROJECT_ROOT / "benchmarks" / "front_plane_ground_truth.json"
DEBUG_PATH = PROJECT_ROOT / "camera_output" / "front_plane_ground_truth_debug"
EXPECTED_RESOLUTION = [960, 1280]


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
    """Load this repository's config even after Isaac adds cv2 paths."""
    expected = CONFIG_PATH.resolve()
    existing = sys.modules.get("config")
    if existing is not None:
        existing_file = getattr(existing, "__file__", None)
        if existing_file and Path(existing_file).resolve() == expected:
            return existing
        sys.modules.pop("config", None)
    return _load_module("config", CONFIG_PATH)


class _GroundTruthConfigProxy:
    """Expose canonical config plus the two legacy ray-ring sample values."""

    def __init__(self, base) -> None:
        self._base = base
        self.front_rim = SimpleNamespace(
            samples_per_side=7,
            sample_corner_trim_fraction=0.15,
        )

    def __getattr__(self, name):
        return getattr(self._base, name)


def _stamp_resolution_metadata() -> None:
    if not OUTPUT_PATH.is_file():
        raise RuntimeError(
            "Ground-truth extractor returned without writing its JSON output."
        )
    payload = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
    payload["schema_version"] = max(int(payload.get("schema_version", 0)), 4)
    payload["camera_resolution_height_width"] = EXPECTED_RESOLUTION
    OUTPUT_PATH.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        "[GROUND TRUTH METADATA STAMPED]\n"
        f"  schema_version: {payload['schema_version']}\n"
        "  camera resolution: 1280x960",
        flush=True,
    )


def main() -> int:
    # Protect the generic module name before Isaac/OpenCV modifies sys.path.
    project_config = _load_project_config()
    CONFIG = project_config.CONFIG

    from isaacsim import SimulationApp

    app = SimulationApp(
        {
            "headless": False,
            "width": 1280,
            "height": 960,
        }
    )

    import isaacsim

    original_simulation_app = isaacsim.SimulationApp
    implementation = None
    original_write_result = None
    try:
        project_root = str(PROJECT_ROOT)
        sys.path[:] = [entry for entry in sys.path if entry != project_root]
        sys.path.insert(0, project_root)
        implementation = _load_module(
            "front_plane_ground_truth_impl",
            IMPLEMENTATION_PATH,
        )
        implementation.CONFIG = _GroundTruthConfigProxy(CONFIG)
        implementation.OUTPUT_PATH = OUTPUT_PATH
        implementation.DEBUG_DIR = DEBUG_PATH
        original_write_result = implementation._write_result

        def write_result_with_resolution(*args, **kwargs) -> None:
            original_write_result(*args, **kwargs)
            _stamp_resolution_metadata()

        implementation._write_result = write_result_with_resolution
        isaacsim.SimulationApp = lambda *_args, **_kwargs: app
        return int(implementation.main())
    finally:
        if implementation is not None and original_write_result is not None:
            implementation._write_result = original_write_result
        isaacsim.SimulationApp = original_simulation_app


if __name__ == "__main__":
    raise SystemExit(main())
