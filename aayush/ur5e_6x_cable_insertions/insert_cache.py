"""Offset→insert tip pose cache (align + translate cycle)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

CACHE_PATH = Path(__file__).resolve().parent / "insert_cache.json"


def _as_vec3(value) -> np.ndarray | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size < 3 or not np.all(np.isfinite(arr[:3])):
        return None
    return arr[:3].copy()


def _as_quat(value) -> np.ndarray | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size < 4 or not np.all(np.isfinite(arr[:4])):
        return None
    q = arr[:4].copy()
    n = float(np.linalg.norm(q))
    if n < 1e-12:
        return None
    return q / n


def _waypoint_to_json(wp: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "t": float(wp.get("t", 0.0)),
        "label": str(wp.get("label", "")),
        "tip": [float(x) for x in np.asarray(wp["tip"]).reshape(3)],
        "orientation_wxyz": [
            float(x) for x in np.asarray(wp["orientation_wxyz"]).reshape(4)
        ],
    }
    axis = _as_vec3(wp.get("insertion_axis"))
    if axis is not None:
        out["insertion_axis"] = [float(x) for x in axis]
    kind = wp.get("kind")
    if kind:
        out["kind"] = str(kind)
    return out


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
    waypoints = []
    for item in raw.get("waypoints") or []:
        if not isinstance(item, dict):
            continue
        tip = _as_vec3(item.get("tip") or item.get("position"))
        ori = _as_quat(item.get("orientation_wxyz"))
        if tip is None or ori is None:
            continue
        waypoints.append(
            {
                "t": float(item.get("t", 0.0)),
                "label": str(item.get("label", "")),
                "kind": str(item.get("kind") or item.get("label") or ""),
                "tip": tip,
                "orientation_wxyz": ori,
                "insertion_axis": _as_vec3(item.get("insertion_axis")),
            }
        )
    waypoints.sort(key=lambda w: float(w["t"]))
    return {
        "station_id": str(station_id),
        "verified": bool(raw.get("verified", False)),
        "waypoints": waypoints,
    }


def save_station(
    station_id: str,
    *,
    waypoints: list[dict[str, Any]],
    verified: bool = False,
    path: Path | None = None,
) -> Path:
    out = path or CACHE_PATH
    root = _load_root(out)
    root["stations"][str(station_id)] = {
        "verified": bool(verified),
        "waypoints": [_waypoint_to_json(w) for w in waypoints],
    }
    out.write_text(json.dumps(root, indent=2) + "\n")
    print(
        f"[INSERT CACHE] wrote {out.name} station={station_id} "
        f"waypoints={len(waypoints)} verified={verified}"
    )
    return out
