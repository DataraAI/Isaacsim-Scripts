#!/usr/bin/env python3
"""Start Isaac Sim before loading OpenCV-dependent front-rim code."""

from __future__ import annotations

import importlib.util
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMPLEMENTATION_PATH = PROJECT_ROOT / "tools" / "extract_front_rim_ground_truth.py"


def _load_implementation():
    spec = importlib.util.spec_from_file_location(
        "front_rim_ground_truth_impl",
        IMPLEMENTATION_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Could not load extractor implementation: {IMPLEMENTATION_PATH}"
        )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
        implementation = _load_implementation()

        # The implementation's main() imports SimulationApp internally. Return
        # the already-running app instead of attempting a second Kit startup.
        isaacsim.SimulationApp = lambda *_args, **_kwargs: app
        return int(implementation.main())
    finally:
        isaacsim.SimulationApp = original_simulation_app


if __name__ == "__main__":
    raise SystemExit(main())
