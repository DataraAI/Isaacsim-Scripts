"""Combined entrypoint: run esha's pickup phase, then hand off directly
into single_rack_cv's insertion phase, in one continuous Isaac Sim
process with one shared robot and cable.

Uses sys.path manipulation + sys.modules eviction between phases,
since both projects independently define same-named local modules
(config.py, sim.py, run_logger.py, perception.py). See
test_module_isolation.py for the proof of concept this is based on.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
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


def run_pickup_phase(simulation_app):
    sys.path.insert(0, ESHA_PATH)
    try:
        from main import run_pickup_phase as _run
        return _run(simulation_app)
    finally:
        sys.path.remove(ESHA_PATH)
        _evict_local_modules()


def run_insertion_phase(simulation_app):
    os.environ["ALREADY_GRASPED_BY_PICKUP_PIPELINE"] = "1"
    sys.path.insert(0, SINGLE_RACK_PATH)
    try:
        from main import run_insertion_phase as _run
        return _run(simulation_app)
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
