"""Per-station home arm joint cache (work-table X, gripper down)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

CACHE_PATH = Path(__file__).resolve().parent / "home_cache.json"


def _load_root(path: Path) -> dict:
    if not path.is_file():
        return {"stations": {}}
    try:
        raw = json.loads(path.read_text())
    except Exception:
        return {"stations": {}}
    if not isinstance(raw, dict):
        return {"stations": {}}
    stations = raw.get("stations")
    if not isinstance(stations, dict):
        stations = {}
    return {"stations": stations}


def load_station(
    station_id: str, *, path: Path | None = None
) -> dict[str, Any] | None:
    root = _load_root(path or CACHE_PATH)
    raw = root["stations"].get(str(station_id))
    if not isinstance(raw, dict):
        return None
    arm = np.asarray(raw.get("arm_rad", []), dtype=np.float64).reshape(-1)
    if arm.size != 6 or not np.all(np.isfinite(arm)):
        return None
    tip = raw.get("tip_m")
    tip_m = None
    if tip is not None:
        arr = np.asarray(tip, dtype=np.float64).reshape(-1)
        if arr.size >= 3 and np.all(np.isfinite(arr[:3])):
            tip_m = arr[:3].copy()
    return {"arm_rad": arm.copy(), "tip_m": tip_m}


def save_station(
    station_id: str,
    *,
    arm_rad,
    tip_m=None,
    path: Path | None = None,
) -> Path:
    out = path or CACHE_PATH
    root = _load_root(out)
    arm = np.asarray(arm_rad, dtype=np.float64).reshape(6)
    payload: dict[str, Any] = {
        "arm_rad": [float(x) for x in arm],
    }
    if tip_m is not None:
        tip = np.asarray(tip_m, dtype=np.float64).reshape(3)
        payload["tip_m"] = [float(x) for x in tip]
    root["stations"][str(station_id)] = payload
    out.write_text(json.dumps(root, indent=2) + "\n")
    print(
        f"[HOME CACHE] wrote {out.name} station={station_id} "
        f"arm_rad={np.round(arm, 3)}"
    )
    return out
