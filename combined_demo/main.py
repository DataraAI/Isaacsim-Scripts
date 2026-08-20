"""Combined entrypoint: run esha's pickup phase, then hand off directly
into single_rack_cv's insertion phase, in one continuous Isaac Sim
process with one shared robot and cable.

Uses sys.path manipulation + sys.modules eviction between phases,
since both projects independently define same-named local modules
(config.py, sim.py, run_logger.py, perception.py). See
test_module_isolation.py for the proof of concept this is based on.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ESHA_PATH = str(REPO_ROOT / "esha" / "ethernet_cable_pick_up")
SINGLE_RACK_PATH = str(REPO_ROOT / "single_rack_cv")

LOCAL_MODULE_NAMES = [
    "config",
    "sim",
    "run_logger",
    "perception",
    "main",
    "head_detector",
    "logging_tee",
]


def _evict_local_modules() -> None:
    for name in LOCAL_MODULE_NAMES:
        sys.modules.pop(name, None)


def _force_load_module(name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def run_pickup_phase(simulation_app):
    sys.path.insert(0, ESHA_PATH)
    try:
        esha_root = Path(ESHA_PATH)
        # Force-load esha's local modules by exact path so their bare module
        # names resolve to these copies (pre-seeded in sys.modules) instead of
        # Isaac Sim's bundled extensions hijacking names like "config".
        # config.py must come first: the other modules do "from config import
        # CONFIG" internally and will find this forced copy already present.
        print("[DEBUG] about to force-load config", flush=True)
        _force_load_module("config", esha_root / "config.py")
        print("[DEBUG] force-load config returned", flush=True)
        print("[DEBUG] about to force-load sim", flush=True)
        _force_load_module("sim", esha_root / "sim.py")
        print("[DEBUG] force-load sim returned", flush=True)
        print("[DEBUG] about to force-load run_logger", flush=True)
        _force_load_module("run_logger", esha_root / "run_logger.py")
        print("[DEBUG] force-load run_logger returned", flush=True)
        print("[DEBUG] about to force-load perception", flush=True)
        _force_load_module("perception", esha_root / "perception.py")
        print("[DEBUG] force-load perception returned", flush=True)
        print("[DEBUG] about to force-load head_detector", flush=True)
        _force_load_module("head_detector", esha_root / "head_detector.py")
        print("[DEBUG] force-load head_detector returned", flush=True)
        print("[DEBUG] about to force-load logging_tee", flush=True)
        _force_load_module("logging_tee", esha_root / "logging_tee.py")
        print("[DEBUG] force-load logging_tee returned", flush=True)
        print("[DEBUG] about to force-load debug", flush=True)
        _force_load_module("debug", esha_root / "debug.py")
        print("[DEBUG] force-load debug returned", flush=True)
        print("[DEBUG] about to force-load main", flush=True)
        main_module = _force_load_module("main", esha_root / "main.py")
        print("[DEBUG] force-load main returned", flush=True)
        print("[DEBUG] about to call esha run_pickup_phase()", flush=True)
        result = main_module.run_pickup_phase(
            simulation_app, close_app_when_done=False
        )
        print("[DEBUG] esha run_pickup_phase() returned", flush=True)
        return result
    finally:
        print("[DEBUG] about to call sys.path.remove(ESHA_PATH)", flush=True)
        sys.path.remove(ESHA_PATH)
        print("[DEBUG] sys.path.remove(ESHA_PATH) returned", flush=True)
        print("[DEBUG] about to call _evict_local_modules()", flush=True)
        _evict_local_modules()
        print("[DEBUG] _evict_local_modules() returned", flush=True)


def run_insertion_phase(simulation_app):
    os.environ["ALREADY_GRASPED_BY_PICKUP_PIPELINE"] = "1"
    sys.path.insert(0, SINGLE_RACK_PATH)
    try:
        single_rack_root = Path(SINGLE_RACK_PATH)
        # Same force-load pattern as run_pickup_phase, using single_rack_cv's
        # own flat local modules. config.py first for the same reason; the
        # vision.*/runtime.* package imports resolve via the inserted sys.path.
        _force_load_module("config", single_rack_root / "config.py")
        _force_load_module("sim", single_rack_root / "sim.py")
        _force_load_module("run_logger", single_rack_root / "run_logger.py")
        _force_load_module("debug", single_rack_root / "debug.py")
        main_module = _force_load_module(
            "main", single_rack_root / "main.py"
        )
        return main_module.run_insertion_phase(simulation_app)
    finally:
        sys.path.remove(SINGLE_RACK_PATH)
        _evict_local_modules()


def main() -> None:
    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": False})

    pickup_logger = run_pickup_phase(simulation_app)
    print(f"[MERGED] Pickup phase complete: {pickup_logger.run_id}")

    insertion_logger = run_insertion_phase(simulation_app)
    print(f"[MERGED] Insertion phase complete: {insertion_logger.run_id}")

    simulation_app.close()


if __name__ == "__main__":
    main()
