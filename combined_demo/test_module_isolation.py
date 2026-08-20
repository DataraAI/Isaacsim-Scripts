"""Standalone proof-of-concept: can we import esha's config.py, then
single_rack_cv's config.py, in the same process, and correctly get two
different CONFIG objects — without sys.modules caching returning the
wrong project's module the second time?
"""
import sys
from pathlib import Path

REPO_ROOT = Path("/home/aayush/Isaacsim-Scripts")
ESHA_PATH = str(REPO_ROOT / "esha" / "ethernet_cable_pick_up")
SINGLE_RACK_PATH = str(REPO_ROOT / "single_rack_cv")

# Module names both projects define locally — anything imported under
# these names needs to be evicted from sys.modules between phases.
LOCAL_MODULE_NAMES = ["config", "sim", "run_logger", "perception"]


def _evict_local_modules() -> None:
    for name in LOCAL_MODULE_NAMES:
        sys.modules.pop(name, None)


def load_esha_config():
    sys.path.insert(0, ESHA_PATH)
    try:
        from config import CONFIG
        return CONFIG
    finally:
        sys.path.remove(ESHA_PATH)


def load_single_rack_cv_config():
    sys.path.insert(0, SINGLE_RACK_PATH)
    try:
        from config import CONFIG
        return CONFIG
    finally:
        sys.path.remove(SINGLE_RACK_PATH)


def main() -> None:
    esha_config = load_esha_config()
    print("esha CONFIG.scene.franka_position:", esha_config.scene.franka_position)
    print("esha CONFIG.scene.cable_usd_path:", esha_config.scene.cable_usd_path)

    _evict_local_modules()

    rack_config = load_single_rack_cv_config()
    print("single_rack_cv CONFIG.scene.franka_position:", rack_config.scene.franka_position)
    print("single_rack_cv CONFIG.scene.rack_usd_path:", rack_config.scene.rack_usd_path)

    assert esha_config.scene.franka_position != rack_config.scene.franka_position, (
        "FAILED: both CONFIG objects report the same franka_position — "
        "module caching likely returned the wrong project's config."
    )
    print("\nPASSED: got two distinct, correct CONFIG objects.")


if __name__ == "__main__":
    main()
